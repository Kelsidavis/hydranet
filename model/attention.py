"""Grouped Query Attention with RoPE for HydraNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .config import HydraNetConfig


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with extended context support."""

    def __init__(self, dim: int, max_position: int = 16384, theta: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.theta = theta

        # Precompute frequency bands
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for cos/sin values
        self._cos_cached = None
        self._sin_cached = None
        self._cached_seq_len = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = max(seq_len, self.max_position)
            t = torch.arange(self._cached_seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors."""
        seq_len = q.shape[2]
        self._update_cache(seq_len, q.device, q.dtype)

        if position_ids is None:
            cos = self._cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self._sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        else:
            cos = self._cos_cached[position_ids].unsqueeze(1)
            sin = self._sin_cached[position_ids].unsqueeze(1)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) for efficient inference.

    Multiple query heads share fewer key-value heads, reducing
    KV cache size while maintaining model quality.
    """

    def __init__(self, config: HydraNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # Projection layers
        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=False)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position=config.max_context,
            theta=config.rope_theta,
        )

        # Attention scaling
        self.scale = self.head_dim ** -0.5

        # Dropout (usually 0 for inference)
        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for grouped query attention.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: Optional causal mask
            position_ids: Position indices for RoPE
            past_key_value: Cached (K, V) from previous steps
            use_cache: Whether to return updated KV cache

        Returns:
            output: (batch, seq_len, hidden_dim)
            present_key_value: Updated KV cache if use_cache=True
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # Expand KV heads to match query heads (GQA)
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, hidden)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match query heads for GQA."""
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


def create_causal_mask(
    seq_len: int,
    past_len: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> torch.Tensor:
    """Create causal attention mask."""
    # Causal mask: positions can only attend to previous positions
    total_len = past_len + seq_len
    mask = torch.full((seq_len, total_len), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=past_len + 1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, total)
