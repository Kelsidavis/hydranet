"""
Resident draft model for speculative decoding.

Uses Mistral-7B-Instruct-v0.2 as draft model.
- Fully resident on GPU (no offloading)
- Fast single-token generation
- Tokenizer-aligned with Mixtral (v0.2 uses tokenizer v1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import math

from ..config import DraftConfig


class RMSNorm(nn.Module):
    """RMS LayerNorm."""

    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class MistralAttention(nn.Module):
    """Attention for Mistral-7B (GQA 4:1)."""

    def __init__(self, config: DraftConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=False)

        # RoPE (precomputed)
        self._init_rope(config.max_position_embeddings)

    def _init_rope(self, max_pos: int):
        """Initialize RoPE cache."""
        dim = self.head_dim
        inv_freq = 1.0 / (
            self.config.rope_theta ** (torch.arange(0, dim, 2).float() / dim)
        )
        t = torch.arange(max_pos)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)

        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        # KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        present_kv = (k, v) if use_cache else None

        # GQA expansion
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.full((seq_len, k.shape[2]), float("-inf"), device=q.device),
                diagonal=k.shape[2] - seq_len + 1
            )
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output), present_kv


class MistralFFN(nn.Module):
    """Feed-forward network for Mistral (SwiGLU)."""

    def __init__(self, config: DraftConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MistralBlock(nn.Module):
    """Transformer block for Mistral."""

    def __init__(self, config: DraftConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.self_attn = MistralAttention(config, layer_idx)
        self.mlp = MistralFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_kv = self.self_attn(
            hidden_states, position_ids, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_kv


class ResidentDraftModel(nn.Module):
    """
    Mistral-7B-Instruct-v0.2 as resident draft model.

    Fully on GPU, no offloading.
    Used for speculative decoding with Mixtral.
    """

    def __init__(
        self,
        config: DraftConfig,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([
            MistralBlock(config, i) for i in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple]]]:
        """Forward pass."""
        batch_size, seq_len = input_ids.shape

        hidden_states = self.embed_tokens(input_ids)

        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=self.device
            ).unsqueeze(0).expand(batch_size, -1)

        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, present_kv = layer(
                hidden_states, position_ids, past_kv, use_cache
            )
            if use_cache:
                present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, present_key_values

    def draft_tokens(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
        past_key_values: Optional[List[Tuple]] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[Tuple]]:
        """
        Draft multiple tokens.

        Args:
            input_ids: Input token(s)
            num_tokens: Number of tokens to draft
            past_key_values: Cached KV
            temperature: Sampling temperature

        Returns:
            draft_tokens: (batch, num_tokens) drafted token IDs
            draft_probs: List of probability distributions
            final_kv: Updated KV cache
        """
        self.eval()
        drafted = []
        probs_list = []
        kv = past_key_values

        with torch.no_grad():
            curr_input = input_ids

            for _ in range(num_tokens):
                logits, kv = self.forward(curr_input, past_key_values=kv, use_cache=True)
                logits = logits[:, -1, :] / temperature

                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs)

                next_token = torch.multinomial(probs, num_samples=1)
                drafted.append(next_token)
                curr_input = next_token

        draft_tokens = torch.cat(drafted, dim=1)
        return draft_tokens, probs_list, kv

    def to_device(self):
        """Move model to device."""
        return self.to(self.device, self.dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ) -> "ResidentDraftModel":
        """
        Load from HuggingFace checkpoint.

        Args:
            model_path: Path to Mistral-7B-Instruct-v0.2
            device: Target device
            dtype: Data type

        Returns:
            Loaded model
        """
        from safetensors import safe_open
        from pathlib import Path
        import json

        path = Path(model_path)
        config = DraftConfig()  # Use defaults (matches Mistral-7B)

        model = cls(config, device, dtype)

        # Load weights
        shard_files = sorted(path.glob("*.safetensors"))

        for shard in shard_files:
            with safe_open(shard, framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key).to(dtype)

                    # Map HuggingFace keys to our keys
                    if "embed_tokens" in key:
                        model.embed_tokens.weight.data.copy_(tensor)
                    elif "lm_head" in key:
                        model.lm_head.weight.data.copy_(tensor)
                    elif "model.norm" in key:
                        model.norm.weight.data.copy_(tensor)
                    elif "layers" in key:
                        # Parse layer index
                        parts = key.split(".")
                        layer_idx = int(parts[2])

                        if "input_layernorm" in key:
                            model.layers[layer_idx].input_layernorm.weight.data.copy_(tensor)
                        elif "post_attention_layernorm" in key:
                            model.layers[layer_idx].post_attention_layernorm.weight.data.copy_(tensor)
                        elif "self_attn.q_proj" in key:
                            model.layers[layer_idx].self_attn.q_proj.weight.data.copy_(tensor)
                        elif "self_attn.k_proj" in key:
                            model.layers[layer_idx].self_attn.k_proj.weight.data.copy_(tensor)
                        elif "self_attn.v_proj" in key:
                            model.layers[layer_idx].self_attn.v_proj.weight.data.copy_(tensor)
                        elif "self_attn.o_proj" in key:
                            model.layers[layer_idx].self_attn.o_proj.weight.data.copy_(tensor)
                        elif "mlp.gate_proj" in key:
                            model.layers[layer_idx].mlp.gate_proj.weight.data.copy_(tensor)
                        elif "mlp.up_proj" in key:
                            model.layers[layer_idx].mlp.up_proj.weight.data.copy_(tensor)
                        elif "mlp.down_proj" in key:
                            model.layers[layer_idx].mlp.down_proj.weight.data.copy_(tensor)

        return model.to_device()
