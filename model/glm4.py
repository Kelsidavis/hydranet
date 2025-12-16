"""
Offloaded GLM-4.5-Air MoE model.

Key differences from Mixtral:
- 128 routed experts + 1 shared expert per MoE layer
- Top-8 sigmoid routing (not softmax)
- First layer is dense (no MoE)
- Partial RoPE (50% of head dim)
- Attention with bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Union
import math

from ..config import GLM4AirConfig, ExpertCacheConfig
from ..cache import ExpertCacheManager
from ..cache.kv_cache import SimpleKVCache


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class PartialRotaryEmbedding(nn.Module):
    """
    Partial Rotary Position Embedding.

    GLM4 applies RoPE to only part of the head dimension (partial_rotary_factor).
    """

    def __init__(
        self,
        dim: int,
        partial_rotary_factor: float = 0.5,
        max_position_embeddings: int = 131072,
        base: float = 1000000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(dim * partial_rotary_factor)
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.rotary_dim, 2, device=device).float() / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
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
        cos = self.cos_cached[position_ids].to(x.dtype)
        sin = self.sin_cached[position_ids].to(x.dtype)
        return cos.unsqueeze(1), sin.unsqueeze(1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply partial rotary position embedding to Q and K."""
    # Split into rotary and pass-through parts
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary to the rotary part
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back
    q_embed = torch.cat([q_rot, q_pass], dim=-1)
    k_embed = torch.cat([k_rot, k_pass], dim=-1)
    return q_embed, k_embed


class GLM4Attention(nn.Module):
    """
    Grouped Query Attention for GLM4.

    Uses 96 query heads and 8 KV heads (GQA 12:1 ratio).
    Has attention bias unlike Mixtral.
    """

    def __init__(
        self,
        config: GLM4AirConfig,
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
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)

        # Projections - Q/K/V have bias, O does not (matches HuggingFace GLM4)
        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=False)

        # Partial RoPE
        self.rotary_emb = PartialRotaryEmbedding(
            self.head_dim,
            partial_rotary_factor=config.partial_rotary_factor,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional["SimpleKVCache"] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply partial RoPE
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_partial_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rotary_dim
        )

        # KV cache
        if kv_cache is not None:
            start_pos = int(position_ids[0, 0].item())
            kv_cache.update(self.layer_idx, key_states, value_states, start_pos)
            total_len = start_pos + seq_len
            key_states, value_states = kv_cache.get(self.layer_idx, total_len)

        # Expand KV for GQA
        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)

        # Use Flash Attention (SDPA) when available - faster and more memory efficient
        use_sdpa = hasattr(F, 'scaled_dot_product_attention')

        if use_sdpa:
            # SDPA handles causal masking internally
            # is_causal=True for prefill (seq_len > 1), False for decode (seq_len = 1)
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=None,
                is_causal=(seq_len > 1),
            )
        else:
            # Fallback to manual attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        if self.num_key_value_groups == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, self.num_key_value_groups, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * self.num_key_value_groups, seq_len, head_dim)


class GLM4DenseMLP(nn.Module):
    """Dense MLP for non-MoE layers and shared expert."""

    def __init__(self, config: GLM4AirConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.intermediate_dim = intermediate_size or config.intermediate_dim

        # Separate gate/up projections to match HuggingFace format
        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GLM4SigmoidRouter(nn.Module):
    """
    Sigmoid-based top-k router for GLM4.

    Unlike Mixtral's softmax router, GLM4 uses sigmoid activation
    and selects top-k experts per token.
    """

    def __init__(self, config: GLM4AirConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.top_k = config.experts_per_token
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        # Router weight (no bias)
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts using sigmoid gating.

        Args:
            hidden_states: (batch, seq, hidden_dim)

        Returns:
            expert_indices: (batch*seq, top_k)
            expert_weights: (batch*seq, top_k)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)  # (batch*seq, hidden_dim)

        # Compute router logits
        router_logits = F.linear(hidden_states_flat.float(), self.weight.float())  # (batch*seq, num_experts)

        # Sigmoid activation (not softmax!)
        router_scores = torch.sigmoid(router_logits)  # (batch*seq, num_experts)

        # Group-based selection (if configured)
        if self.n_group > 1 and self.topk_group > 0:
            # Reshape into groups
            group_size = self.num_experts // self.n_group
            scores_grouped = router_scores.view(-1, self.n_group, group_size)

            # Select top groups
            group_scores = scores_grouped.max(dim=-1).values  # (batch*seq, n_group)
            _, top_groups = group_scores.topk(self.topk_group, dim=-1)  # (batch*seq, topk_group)

            # Mask non-selected groups
            mask = torch.zeros_like(scores_grouped)
            mask.scatter_(1, top_groups.unsqueeze(-1).expand(-1, -1, group_size), 1.0)
            router_scores = (scores_grouped * mask).view(-1, self.num_experts)

        # Select top-k experts
        topk_weights, topk_indices = router_scores.topk(self.top_k, dim=-1)  # (batch*seq, top_k)

        # Normalize weights if configured
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Apply scaling factor
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_indices, topk_weights.to(hidden_states.dtype)


class GLM4MoELayer(nn.Module):
    """
    MoE layer with expert offloading for GLM4.

    128 routed experts + 1 shared expert per layer.
    Top-8 sigmoid routing.

    Optimization: Cross-layer prefetching
    - Stores last selected experts for next layer to prefetch speculatively
    - Expert selection is often correlated between adjacent layers
    """

    def __init__(
        self,
        config: GLM4AirConfig,
        layer_idx: int,
        cache_manager: ExpertCacheManager,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.cache_manager = cache_manager
        # Store last selected experts for cross-layer prediction
        self._last_selected_experts: list = []

        # Router (on GPU)
        self.router = GLM4SigmoidRouter(config, layer_idx)

        # Shared expert (always resident on GPU)
        self.shared_expert = GLM4DenseMLP(
            config,
            intermediate_size=config.moe_intermediate_dim * config.num_shared_experts
        )

    def prefetch_predicted(self, predicted_experts: list):
        """
        Speculatively prefetch experts based on prediction (e.g., from previous layer).

        This is called BEFORE the forward pass to start loading experts in background
        while attention is still computing.
        """
        if not self.cache_manager.use_packed_mode:
            return

        # Only prefetch experts not already cached
        for expert_idx in predicted_experts:
            if not self.cache_manager.is_cached(self.layer_idx, expert_idx):
                self.cache_manager.prefetch_int4_slot(self.layer_idx, expert_idx)

    def get_last_selected_experts(self) -> list:
        """Return experts selected in the last forward pass (for cross-layer prediction)."""
        return self._last_selected_experts

    def forward(self, hidden_states: torch.Tensor, skip_routed: bool = False) -> torch.Tensor:
        """
        Forward through MoE layer.

        Output = shared_expert(x) + sum(weight_i * routed_expert_i(x))

        Args:
            hidden_states: Input tensor [batch, seq, hidden_dim]
            skip_routed: If True, only run shared expert (faster decode mode)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Shared expert output (always computed)
        shared_output = self.shared_expert(hidden_states)

        # Fast path: skip routed experts (for layer skipping during decode)
        if skip_routed:
            return shared_output

        # Route tokens to experts
        expert_indices, expert_weights = self.router(hidden_states)  # (batch*seq, top_k)

        # Flatten hidden states
        flat_hidden = hidden_states.view(-1, hidden_dim)  # (batch*seq, hidden_dim)

        # Get unique experts
        unique_experts = expert_indices.unique().tolist()

        # Output accumulator
        routed_output = torch.zeros_like(flat_hidden)

        # Use fused INT4 when available
        use_fused_int4 = self.cache_manager.use_packed_mode

        # 2-phase overlap: with top-8, always have overlap opportunity
        do_overlap = use_fused_int4 and len(unique_experts) > 1

        if do_overlap:
            # Partition into hits vs misses
            hits, misses = [], []
            for e in unique_experts:
                if self.cache_manager.is_cached(self.layer_idx, e):
                    hits.append(e)
                else:
                    misses.append(e)

            # Prefetch all misses upfront (async H2D starts immediately)
            for e in misses:
                self.cache_manager.prefetch_int4_slot(self.layer_idx, e)

            # Process hits first (while misses are loading in background)
            expert_order = hits + misses
        else:
            expert_order = unique_experts

        # Process each expert
        for expert_idx in expert_order:
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx)  # (batch*seq, top_k)
            if not expert_mask.any():
                continue

            token_indices, k_positions = torch.where(expert_mask)
            if len(token_indices) == 0:
                continue

            # Gather input tokens
            expert_input = flat_hidden[token_indices]

            if use_fused_int4:
                slot = self.cache_manager.get_int4_slot(self.layer_idx, expert_idx)

                # Check if we should use FP16 compute (dequant + cuBLAS) instead of INT4 fused
                use_fp16_compute = getattr(self.cache_manager, 'use_fp16_compute', False)

                if use_fp16_compute:
                    # Dequantize to FP16 and use cuBLAS (4x faster for small batches)
                    weights = slot.dequantize()
                    gate_out = F.silu(F.linear(expert_input, weights["gate_proj"]))
                    up_out = F.linear(expert_input, weights["up_proj"])
                    expert_out = F.linear(gate_out * up_out, weights["down_proj"])
                else:
                    # Use INT4 fused Triton kernel (slower but no FP16 memory overhead)
                    from ..kernels.int4_gemm import expert_mlp_int4_fused
                    expert_out = expert_mlp_int4_fused(
                        expert_input,
                        slot.gate_packed, slot.gate_scales,
                        slot.up_packed, slot.up_scales,
                        slot.down_packed, slot.down_scales,
                        group_size=slot.group_size,
                    )
            else:
                weights = self.cache_manager.get_expert_weights(self.layer_idx, expert_idx)
                gate_out = F.silu(F.linear(expert_input, weights["gate_proj"]))
                up_out = F.linear(expert_input, weights["up_proj"])
                expert_out = F.linear(gate_out * up_out, weights["down_proj"])

            # Get routing weights
            routing_weights = expert_weights[token_indices, k_positions].unsqueeze(-1)

            # Accumulate
            routed_output.index_add_(0, token_indices, expert_out * routing_weights)

        # Store selected experts for cross-layer prediction
        self._last_selected_experts = unique_experts

        # Combine shared + routed
        routed_output = routed_output.view(batch_size, seq_len, hidden_dim)
        return shared_output + routed_output


class GLM4Block(nn.Module):
    """Single transformer block - can be dense or MoE."""

    def __init__(
        self,
        config: GLM4AirConfig,
        layer_idx: int,
        cache_manager: Optional[ExpertCacheManager] = None,
        is_moe: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_moe = is_moe

        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

        # Attention
        self.self_attn = GLM4Attention(config, layer_idx)

        # FFN: MoE or Dense
        if is_moe and cache_manager is not None:
            self.mlp = GLM4MoELayer(config, layer_idx, cache_manager)
        else:
            self.mlp = GLM4DenseMLP(config)

    def prefetch_experts(self, predicted_experts: list):
        """Speculatively prefetch experts based on prediction from previous layer."""
        if self.is_moe and hasattr(self.mlp, 'prefetch_predicted'):
            self.mlp.prefetch_predicted(predicted_experts)

    def get_last_selected_experts(self) -> list:
        """Get experts selected in last forward (for cross-layer prediction)."""
        if self.is_moe and hasattr(self.mlp, 'get_last_selected_experts'):
            return self.mlp.get_last_selected_experts()
        return []

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional["SimpleKVCache"] = None,
        predicted_experts: Optional[list] = None,
        skip_routed: bool = False,
    ) -> torch.Tensor:
        # Start prefetching predicted experts BEFORE attention (overlap)
        if predicted_experts and self.is_moe and not skip_routed:
            self.prefetch_experts(predicted_experts)

        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # FFN with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe and hasattr(self.mlp, 'forward'):
            # MoE layer - pass skip_routed flag
            hidden_states = self.mlp(hidden_states, skip_routed=skip_routed)
        else:
            # Dense layer
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class OffloadedGLM4(nn.Module):
    """
    GLM-4.5-Air with expert offloading.

    - First layer is dense
    - Remaining 45 layers are MoE with 128 routed + 1 shared expert
    - Expert weights offloaded to RAM, loaded on demand

    Layer Skipping (for 16GB GPUs):
    - Set decode_skip_ratio to skip some MoE layers during decode
    - Skipped layers still run attention and shared expert, just not routed experts
    - Trade quality for speed: 50% skip gives ~2x decode speedup
    """

    def __init__(
        self,
        config: GLM4AirConfig,
        cache_config: ExpertCacheConfig,
        kv_config: Optional[object] = None,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
        decode_skip_ratio: float = 0.0,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype
        self.decode_skip_ratio = decode_skip_ratio

        # Compute which layers to skip during decode
        # Skip evenly distributed layers for minimal quality impact
        num_moe = config.num_layers - config.first_k_dense_replace
        num_skip = int(num_moe * decode_skip_ratio)
        if num_skip > 0:
            # Skip every N-th layer
            skip_interval = num_moe // num_skip
            self._skip_layers = set(
                config.first_k_dense_replace + i * skip_interval
                for i in range(num_skip)
            )
        else:
            self._skip_layers = set()

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Expert cache manager (for MoE layers only)
        # MoE layers are indexed 0 to num_moe_layers-1 in the cache
        # Create an adapter config that looks like MixtralConfig for the cache manager
        from dataclasses import dataclass

        @dataclass
        class _CacheAdapterConfig:
            """Adapter to make GLM4 config compatible with ExpertCacheManager."""
            num_layers: int
            num_experts: int
            expert_size_bytes: int

        cache_adapter = _CacheAdapterConfig(
            num_layers=config.num_moe_layers,
            num_experts=config.num_experts,
            expert_size_bytes=config.expert_size_bytes,
        )

        self.expert_cache = ExpertCacheManager(
            model_config=cache_adapter,
            cache_config=cache_config,
            device=device,
            dtype=dtype,
        )

        # Transformer blocks
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            is_moe = i >= config.first_k_dense_replace
            # MoE layer index for cache manager (0-indexed from first MoE layer)
            moe_layer_idx = i - config.first_k_dense_replace if is_moe else 0

            self.layers.append(
                GLM4Block(
                    config=config,
                    layer_idx=moe_layer_idx if is_moe else i,
                    cache_manager=self.expert_cache if is_moe else None,
                    is_moe=is_moe,
                )
            )

        # Output
        self.norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional["SimpleKVCache"] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs (auto-generated if not provided)
            kv_cache: Optional KV cache

        Returns:
            logits: (batch, seq, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Generate position IDs if not provided
        if position_ids is None:
            if kv_cache is not None:
                # Get position from KV cache
                start_pos = kv_cache.cur_len
            else:
                start_pos = 0
            position_ids = torch.arange(
                start_pos, start_pos + seq_len, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Build causal mask
        if attention_mask is None:
            if kv_cache is not None:
                total_len = position_ids[0, 0].item() + seq_len
            else:
                total_len = seq_len

            attention_mask = torch.full(
                (batch_size, 1, seq_len, total_len),
                float("-inf"),
                device=input_ids.device,
                dtype=self.dtype,
            )
            attention_mask = torch.triu(attention_mask, diagonal=total_len - seq_len + 1)

        # Embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Detect decode mode: single token with KV cache
        is_decode = seq_len == 1 and kv_cache is not None and kv_cache.cur_len > 0

        # Transformer layers with cross-layer expert prefetching
        predicted_experts = None  # No prediction for first layer
        for layer_idx, layer in enumerate(self.layers):
            # Skip routed experts during decode for selected layers
            skip_routed = is_decode and layer_idx in self._skip_layers

            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                predicted_experts=predicted_experts,
                skip_routed=skip_routed,
            )
            # Get selected experts for cross-layer prediction
            # Next layer will prefetch these while running attention
            if not skip_routed:
                predicted_experts = layer.get_last_selected_experts()

        # Output
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, None

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        """Load non-expert weights."""
        for name, param in self.named_parameters():
            if name in weights:
                param.data.copy_(weights[name].to(self.dtype))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        kv_cache: Optional["SimpleKVCache"] = None,
    ) -> torch.Tensor:
        """Simple greedy/sampling generation."""
        for _ in range(max_new_tokens):
            logits, _ = self.forward(input_ids, kv_cache=kv_cache)
            next_logits = logits[:, -1, :] / temperature

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # For KV cache, only pass new token next iteration
            if kv_cache is not None:
                input_ids = next_token

        return input_ids
