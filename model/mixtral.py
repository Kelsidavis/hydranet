"""
Offloaded Mixtral-8x7B model.

Main inference model with:
- Expert offloading (RAM -> VRAM on demand)
- Top-K switching (Top-2 prefill / Top-1 decode)
- KV cache paging
- Microbatch pipelining for prefetch overlap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import math

from ..config import MixtralConfig, ExpertCacheConfig, KVCacheConfig, TopKMode
from ..cache import ExpertCacheManager, KVPageManager
from ..cache.kv_cache import SimpleKVCache
from .router import TopKRouter, RouterOutput


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 to avoid overflow when squaring large values
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 1000000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cache
        self._set_cos_sin_cache(max_position_embeddings, device)

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for given positions."""
        # position_ids: (batch, seq)
        # cos_cached: (max_pos, head_dim)
        cos = self.cos_cached[position_ids]  # (batch, seq, head_dim)
        sin = self.sin_cached[position_ids]  # (batch, seq, head_dim)
        # Unsqueeze to (batch, 1, seq, head_dim) for broadcasting with (batch, heads, seq, head_dim)
        return cos.unsqueeze(1), sin.unsqueeze(1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to Q and K."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MixtralAttention(nn.Module):
    """
    Grouped Query Attention for Mixtral.

    Uses 32 query heads and 8 KV heads (GQA 4:1 ratio).
    """

    def __init__(
        self,
        config: MixtralConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # Projections (loaded from pretrained weights)
        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=False)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        kv_cache: Optional["SimpleKVCache"] = None,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV caching.

        Args:
            hidden_states: (batch, seq, hidden_dim)
            position_ids: (batch, seq)
            attention_mask: (batch, 1, seq, total_seq)
            past_key_value: Legacy cached (K, V) tensors (deprecated, use kv_cache)
            use_cache: Whether to return updated KV cache (legacy mode)
            kv_cache: SimpleKVCache instance for efficient preallocated caching
            layer_idx: Layer index (for kv_cache)

        Returns:
            output: (batch, seq, hidden_dim)
            present_key_value: Updated KV cache (if use_cache, legacy mode)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (position_ids already account for cache position)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache handling
        if kv_cache is not None:
            # Derive start_pos from position_ids (works for any seq_len)
            start_pos = int(position_ids[0, 0].item())
            kv_cache.update(layer_idx, key_states, value_states, start_pos)

            # Get full K, V up to current position + new tokens
            total_len = start_pos + seq_len
            key_states, value_states = kv_cache.get(layer_idx, total_len)
            present_key_value = None  # Don't return tuple in new mode

        elif past_key_value is not None:
            # Legacy path: concatenate
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
            present_key_value = (key_states, value_states) if use_cache else None
        else:
            present_key_value = (key_states, value_states) if use_cache else None

        # Expand KV for GQA
        key_states_expanded = self._repeat_kv(key_states)
        value_states_expanded = self._repeat_kv(value_states)

        # Attention
        attn_weights = torch.matmul(query_states, key_states_expanded.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_expanded)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query heads for GQA."""
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        if self.num_key_value_groups == 1:
            return hidden_states

        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, self.num_key_value_groups, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * self.num_key_value_groups, seq_len, head_dim)


class MixtralMoELayer(nn.Module):
    """
    MoE layer with expert offloading.

    Experts are loaded from RAM to VRAM on demand via cache manager.
    """

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: ExpertCacheConfig,
        layer_idx: int,
        cache_manager: ExpertCacheManager,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.cache_manager = cache_manager

        # Router (on GPU)
        self.router = TopKRouter(config, cache_config, layer_idx)

        # Set cache checker for affinity bonus
        self.router.set_cache_checker(
            lambda idx: cache_manager.is_cached(layer_idx, idx)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k: int = 2,
    ) -> torch.Tensor:
        """
        Forward through MoE layer.

        Args:
            hidden_states: (batch, seq, hidden_dim)
            top_k: Number of experts per token (2 for prefill, 1 for decode)

        Returns:
            output: (batch, seq, hidden_dim)
        """
        import os
        _debug_mem = os.environ.get("DEBUG_MEM", "0") == "1"

        batch_size, seq_len, hidden_dim = hidden_states.shape

        if _debug_mem and self.layer_idx == 0:
            print(f"    [MoE L0] Start: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # Route tokens
        router_output = self.router(hidden_states, top_k_override=top_k)
        expert_indices = router_output.expert_indices  # (batch, seq, top_k)
        expert_weights = router_output.expert_weights  # (batch, seq, top_k)

        if _debug_mem and self.layer_idx == 0:
            print(f"    [MoE L0] After routing: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # Flatten for processing
        flat_hidden = hidden_states.view(-1, hidden_dim)  # (batch*seq, hidden_dim)
        flat_indices = expert_indices.view(-1, top_k)  # (batch*seq, top_k)
        flat_weights = expert_weights.view(-1, top_k)  # (batch*seq, top_k)

        # Get unique experts needed for this layer
        unique_experts = flat_indices.unique().tolist()

        # Process each expert - output accumulates weighted results
        output = torch.zeros_like(flat_hidden)

        # Use fused INT4 kernel when available (no fp16 materialization)
        use_fused_int4 = self.cache_manager.use_packed_mode

        # 2-phase overlap pattern: only when there's work to overlap
        # (multiple experts = top-2 or batched tokens)
        do_overlap = use_fused_int4 and len(unique_experts) > 1

        if do_overlap:
            # Partition into hits (already cached) vs misses
            hits, misses = [], []
            for e in unique_experts:
                if self.cache_manager.is_cached(self.layer_idx, e):
                    hits.append(e)
                else:
                    misses.append(e)

            # Schedule async loads for misses (may evict, but hits identified first)
            for e in misses:
                self.cache_manager.prefetch_int4_slot(self.layer_idx, e)

            # Process hits first (while misses load in background)
            expert_order = hits + misses
        else:
            # Single expert or non-fused: just process in order
            expert_order = unique_experts

        for expert_idx in expert_order:
            # Find tokens routed to this expert
            expert_mask = (flat_indices == expert_idx)  # (batch*seq, top_k)
            if not expert_mask.any():
                continue

            token_indices, k_positions = torch.where(expert_mask)
            if len(token_indices) == 0:
                continue

            # Gather input tokens
            expert_input = flat_hidden[token_indices]  # (num_tokens, hidden_dim)

            if use_fused_int4:
                # Fused path: use INT4 directly without fp16 materialization
                from ..kernels.int4_gemm import expert_mlp_int4_fused

                # Get INT4 slot (wait_ready called inside if still loading)
                slot = self.cache_manager.get_int4_slot(self.layer_idx, expert_idx)

                # Run fused MLP with INT4 weights - no fp16 allocation!
                expert_out = expert_mlp_int4_fused(
                    expert_input,
                    slot.gate_packed, slot.gate_scales,
                    slot.up_packed, slot.up_scales,
                    slot.down_packed, slot.down_scales,
                    group_size=slot.group_size,
                )
            else:
                # Fallback: dequantize to fp16 then compute
                weights = self.cache_manager.get_expert_weights(self.layer_idx, expert_idx)

                # SwiGLU FFN: down(silu(gate(x)) * up(x))
                gate_out = F.silu(F.linear(expert_input, weights["gate_proj"]))
                up_out = F.linear(expert_input, weights["up_proj"])
                expert_out = F.linear(gate_out * up_out, weights["down_proj"])

            # Get routing weights for these tokens
            routing_weights = flat_weights[token_indices, k_positions].unsqueeze(-1)

            # Accumulate weighted output
            output.index_add_(0, token_indices, expert_out * routing_weights)

        return output.view(batch_size, seq_len, hidden_dim)


class MixtralBlock(nn.Module):
    """Single transformer block with attention + MoE."""

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: ExpertCacheConfig,
        layer_idx: int,
        cache_manager: ExpertCacheManager,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

        # Attention
        self.self_attn = MixtralAttention(config, layer_idx)

        # MoE
        self.moe = MixtralMoELayer(config, cache_config, layer_idx, cache_manager)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        top_k: int = 2,
        kv_cache: Optional["SimpleKVCache"] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """Forward through block."""
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_kv = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
        )
        hidden_states = residual + hidden_states

        # MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.moe(hidden_states, top_k=top_k)
        hidden_states = residual + hidden_states

        return hidden_states, present_kv


class OffloadedMixtral(nn.Module):
    """
    Mixtral-8x7B with expert offloading.

    Architecture:
    - Embeddings: GPU resident
    - Attention: GPU resident
    - Routers: GPU resident
    - Experts: RAM with on-demand GPU loading
    - KV cache: Paged (VRAM window + RAM pages)
    """

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: ExpertCacheConfig,
        kv_config: KVCacheConfig,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.config = config
        self.cache_config = cache_config
        self.kv_config = kv_config
        self.device = device
        self.dtype = dtype

        # Expert cache manager
        self.expert_cache = ExpertCacheManager(
            model_config=config,
            cache_config=cache_config,
            device=device,
            dtype=dtype,
        )

        # KV page manager
        self.kv_cache = KVPageManager(
            model_config=config,
            kv_config=kv_config,
            device=device,
            dtype=dtype,
        )

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim).to(device=device, dtype=dtype)

        # Transformer blocks (moved to device)
        self.layers = nn.ModuleList([
            MixtralBlock(config, cache_config, i, self.expert_cache).to(device=device, dtype=dtype)
            for i in range(config.num_layers)
        ])

        # Output
        self.norm = RMSNorm(config.hidden_dim, config.rms_norm_eps).to(device=device, dtype=dtype)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False).to(device=device, dtype=dtype)

        # Top-K mode
        self.topk_mode = TopKMode.DECODE

    def set_topk_mode(self, mode: TopKMode):
        """Set top-k mode (PREFILL=2, DECODE=1)."""
        self.topk_mode = mode

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        force_top_k: Optional[int] = None,
        kv_cache: Optional["SimpleKVCache"] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple]]]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq)
            position_ids: (batch, seq)
            attention_mask: (batch, 1, seq, total_seq)
            past_key_values: Legacy list of (K, V) per layer (deprecated)
            use_cache: Whether to return KV cache (legacy mode)
            kv_cache: SimpleKVCache for efficient preallocated caching

        Returns:
            logits: (batch, seq, vocab_size)
            present_key_values: Updated KV cache (legacy mode, None if using kv_cache)
        """
        batch_size, seq_len = input_ids.shape

        # Determine top-k based on prefill vs decode
        if kv_cache is not None:
            is_prefill = kv_cache.cur_len == 0
        else:
            is_prefill = past_key_values is None or seq_len > 1

        if force_top_k is not None:
            top_k = force_top_k  # Override
        elif is_prefill:
            top_k = 2  # Top-2 for prefill
        else:
            top_k = 1 if self.topk_mode == TopKMode.DECODE else 2

        # Embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Position IDs (account for KV cache position)
        if position_ids is None:
            if kv_cache is not None:
                past_len = kv_cache.cur_len
            elif past_key_values is not None:
                past_len = past_key_values[0][0].shape[2]
            else:
                past_len = 0
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                device=self.device,
            ).unsqueeze(0).expand(batch_size, -1)

        # Attention mask
        if attention_mask is None:
            if kv_cache is not None:
                total_len = kv_cache.cur_len + seq_len
            elif past_key_values is not None:
                total_len = past_key_values[0][0].shape[2] + seq_len
            else:
                total_len = seq_len
            attention_mask = self._make_causal_mask(seq_len, total_len)

        # Update slot reallocation check
        self.expert_cache.maybe_reallocate_slots(seq_len)

        # Forward through layers
        present_key_values = [] if use_cache and kv_cache is None else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None

            hidden_states, present_kv = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache and kv_cache is None,  # Only legacy mode
                top_k=top_k,
                kv_cache=kv_cache,
            )

            if present_key_values is not None:
                present_key_values.append(present_kv)

            # Clear fp16 cache after each layer (lightweight - no gc/sync)
            # In fused INT4 mode, this is a no-op since gpu_weights is empty
            if self.expert_cache.use_packed_mode:
                self.expert_cache.clear_layer_fp16(i)

        # Advance KV cache position after all layers processed
        if kv_cache is not None:
            if kv_cache.cur_len == 0:
                # Prefill: set length to input sequence
                kv_cache.set_len(seq_len)
            else:
                # Decode: advance by number of new tokens
                kv_cache.advance(seq_len)

        # Output
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, present_key_values

    def _make_causal_mask(
        self,
        query_len: int,
        key_len: int,
    ) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.full(
            (query_len, key_len),
            float("-inf"),
            device=self.device,
        )
        mask = torch.triu(mask, diagonal=key_len - query_len + 1)
        return mask.unsqueeze(0).unsqueeze(0)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Uses Top-2 for prefill, Top-1 for decode.
        """
        self.eval()
        past_key_values = None
        generated = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get input for this step
                if past_key_values is not None:
                    curr_input = generated[:, -1:]
                else:
                    curr_input = generated

                # Forward
                logits, past_key_values = self.forward(
                    curr_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                # Sample
                logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits = logits.masked_fill(indices_to_remove, float("-inf"))

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def warm_scan(
        self,
        input_ids: torch.Tensor,
        top_k_per_layer: int = 4,
        max_tokens: Optional[int] = 512,
    ) -> List[Tuple[int, int]]:
        """
        Router-only warm scan for cache warmup.

        Runs embeddings + attention + router (skips expert MLP) to identify
        which experts will likely be needed. Use before full prefill to
        pre-populate the cache and avoid cold-start misses.

        Args:
            input_ids: (batch, seq) input token IDs
            top_k_per_layer: Number of top experts to collect per layer
            max_tokens: Maximum tokens to scan (truncates long prompts)

        Returns:
            List of (layer_idx, expert_idx) tuples to prefetch
        """
        self.eval()
        batch_size, seq_len = input_ids.shape

        # Truncate if needed
        if max_tokens and seq_len > max_tokens:
            input_ids = input_ids[:, :max_tokens]
            seq_len = max_tokens

        # Track expert votes per layer
        expert_votes: List[Dict[int, int]] = [
            {} for _ in range(self.config.num_layers)
        ]

        with torch.no_grad():
            # Embeddings
            hidden_states = self.embed_tokens(input_ids)

            # Position IDs
            position_ids = torch.arange(
                seq_len, device=self.device
            ).unsqueeze(0).expand(batch_size, -1)

            # Attention mask
            attention_mask = self._make_causal_mask(seq_len, seq_len)

            # Run through layers - attention + router only
            for layer_idx, layer in enumerate(self.layers):
                # Self-attention (needed for router input)
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
                hidden_states, _ = layer.self_attn(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_value=None,
                    use_cache=False,
                )
                hidden_states = residual + hidden_states

                # Router only (skip MLP)
                normed = layer.post_attention_layernorm(hidden_states)
                router_output = layer.moe.router(normed, top_k_override=2)

                # Collect expert indices
                expert_indices = router_output.expert_indices  # (batch, seq, top_k)
                for expert_idx in expert_indices.view(-1).tolist():
                    expert_votes[layer_idx][expert_idx] = \
                        expert_votes[layer_idx].get(expert_idx, 0) + 1

                # Use attention output as input for next layer
                # (skip MLP contribution - approximation is fine for warmup)

        # Collect top-K experts per layer by vote count
        experts_to_prefetch = []
        for layer_idx, votes in enumerate(expert_votes):
            sorted_experts = sorted(votes.items(), key=lambda x: -x[1])
            for expert_idx, _ in sorted_experts[:top_k_per_layer]:
                experts_to_prefetch.append((layer_idx, expert_idx))

        return experts_to_prefetch

    def prefetch_experts(self, experts: List[Tuple[int, int]]):
        """
        Prefetch experts into cache.

        Args:
            experts: List of (layer_idx, expert_idx) tuples
        """
        for layer_idx, expert_idx in experts:
            if not self.expert_cache.is_cached(layer_idx, expert_idx):
                # This triggers load to cache
                _ = self.expert_cache.get_expert_weights(layer_idx, expert_idx)

    def warm_cache(
        self,
        input_ids: torch.Tensor,
        top_k_per_layer: int = 4,
        max_tokens: Optional[int] = 512,
        verbose: bool = True,
    ):
        """
        Combined warm scan + prefetch for cache warmup.

        Args:
            input_ids: Input tokens to scan
            top_k_per_layer: Experts to warm per layer
            max_tokens: Max tokens to scan
            verbose: Print progress
        """
        if verbose:
            print(f"Warming cache with router scan (top-{top_k_per_layer}/layer)...")

        experts = self.warm_scan(input_ids, top_k_per_layer, max_tokens)

        if verbose:
            unique_experts = len(set(experts))
            print(f"  Identified {unique_experts} unique experts to prefetch")

        self.prefetch_experts(experts)

        if verbose:
            stats = self.expert_cache.get_stats()
            print(f"  Cache warmed: {stats['total_hits'] + stats['total_misses']} loads")

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        """Load non-expert weights to GPU."""
        # Embeddings
        self.embed_tokens.weight.data.copy_(weights["embed_tokens"].to(self.device))
        self.lm_head.weight.data.copy_(weights["lm_head"].to(self.device))
        self.norm.weight.data.copy_(weights["final_norm"].to(self.device))

        # Layers
        for i, layer in enumerate(self.layers):
            # Attention
            layer.self_attn.q_proj.weight.data.copy_(
                weights[f"layers.{i}.attn.q_proj"].to(self.device)
            )
            layer.self_attn.k_proj.weight.data.copy_(
                weights[f"layers.{i}.attn.k_proj"].to(self.device)
            )
            layer.self_attn.v_proj.weight.data.copy_(
                weights[f"layers.{i}.attn.v_proj"].to(self.device)
            )
            layer.self_attn.o_proj.weight.data.copy_(
                weights[f"layers.{i}.attn.o_proj"].to(self.device)
            )

            # Norms
            layer.input_layernorm.weight.data.copy_(
                weights[f"layers.{i}.input_layernorm"].to(self.device)
            )
            layer.post_attention_layernorm.weight.data.copy_(
                weights[f"layers.{i}.post_attention_layernorm"].to(self.device)
            )

            # Router
            layer.moe.router.gate.weight.data.copy_(
                weights[f"layers.{i}.router"].to(self.device)
            )

    def get_cache_stats(self) -> Dict:
        """Get expert cache statistics."""
        return self.expert_cache.get_stats()

    def reset_cache_stats(self):
        """Reset cache statistics."""
        self.expert_cache.reset_stats()

    def validate_weights(self, verbose: bool = True) -> bool:
        """
        Validate model weights are loaded correctly.

        Checks:
        1. Router gate weights (no accidental transpose)
        2. Weight shapes match config

        Args:
            verbose: Print validation progress

        Returns:
            True if all validations pass

        Raises:
            ValueError if validation fails
        """
        if verbose:
            print("Validating model weights...")

        # Validate all router gates
        for layer_idx, layer in enumerate(self.layers):
            try:
                layer.moe.router.validate_gate_weights()
                if verbose:
                    print(f"  Layer {layer_idx} router: OK")
            except ValueError as e:
                raise ValueError(f"Layer {layer_idx} router validation failed: {e}")

        # Validate embedding shapes
        assert self.embed_tokens.weight.shape == (self.config.vocab_size, self.config.hidden_dim), \
            f"Embedding shape mismatch: {self.embed_tokens.weight.shape}"

        assert self.lm_head.weight.shape == (self.config.vocab_size, self.config.hidden_dim), \
            f"LM head shape mismatch: {self.lm_head.weight.shape}"

        if verbose:
            print("  Embeddings: OK")
            print("  All validations passed!")

        return True
