"""
HydraNet with Expert Offloading (v2)

Key improvements:
- Per-layer expert cache (no cross-layer thrashing)
- Top-2 prefill / Top-1 decode switching
- Fixed VRAM slots per layer
- Better prefetch strategy
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import time

# ========== RESOURCE LIMITS ==========
RESERVED_THREADS = 2
MAX_CPU_THREADS = max(1, os.cpu_count() - RESERVED_THREADS) if os.cpu_count() else 4

if torch.get_num_threads() > MAX_CPU_THREADS:
    torch.set_num_threads(MAX_CPU_THREADS)
    torch.set_num_interop_threads(max(1, MAX_CPU_THREADS // 2))
# =====================================

from ..model.config import HydraNetConfig
from ..model.attention import GroupedQueryAttention, RMSNorm, RotaryEmbedding, create_causal_mask
from ..model.expert import ExpertFFN
from ..model.router import TopKRouter
from .cache_manager import ExpertCacheManager, PerLayerCacheConfig


@dataclass
class OffloadConfig:
    """Configuration for expert offloading."""
    # Per-layer cache settings
    hot_experts_per_layer: int = 4   # Always in VRAM (from config)
    warm_experts_per_layer: int = 2  # Can be evicted via LRU

    # Total GPU memory budget for experts
    expert_gpu_budget_gb: float = 6.0

    # Use pinned memory for faster CPU->GPU transfer
    use_pinned_memory: bool = True

    # Prefetch settings
    enable_prefetch: bool = True
    prefetch_next_layer: bool = True

    # CUDA streams
    use_async_loading: bool = True
    num_load_streams: int = 2


class OffloadedMoELayer(nn.Module):
    """
    MoE layer with per-layer caching and Top-K switching.

    - Prefill: Uses experts_per_token (Top-2)
    - Decode: Uses decode_experts_per_token (Top-1)
    """

    def __init__(
        self,
        config: HydraNetConfig,
        layer_idx: int,
        cache_manager: ExpertCacheManager,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.cache_manager = cache_manager

        # Router stays on GPU
        self.router = TopKRouter(config, layer_idx)

        # Shared expert (always on GPU)
        if config.use_shared_expert:
            self.shared_expert = ExpertFFN(
                config.hidden_dim,
                config.expert_intermediate_dim,
            )
            self.shared_scale = nn.Parameter(torch.ones(1))
        else:
            self.shared_expert = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        """
        Forward with on-demand expert loading.

        Args:
            hidden_states: (batch, seq, hidden_dim)
            is_prefill: If True, use experts_per_token; else decode_experts_per_token
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Select top-k based on prefill vs decode
        top_k = self.config.experts_per_token if is_prefill else self.config.decode_experts_per_token

        # Route tokens to experts
        router_output = self.router(hidden_states, return_aux_loss=False, top_k_override=top_k)
        expert_indices = router_output.expert_indices  # (batch, seq, top_k)
        expert_weights = router_output.expert_weights

        # Flatten for processing
        flat_hidden = hidden_states.view(-1, hidden_dim)
        flat_indices = expert_indices.view(-1, top_k)
        flat_weights = expert_weights.view(-1, top_k)

        # Get unique experts needed
        unique_experts = flat_indices.unique().tolist()

        # Prefetch next layer's likely experts
        if self.cache_manager.config.enable_prefetch:
            self.cache_manager.prefetch_for_next_layer(
                self.layer_idx,
                unique_experts[:4],  # Prefetch top 4 most common
            )

        # Process each expert
        output = torch.zeros_like(flat_hidden)

        for expert_idx in unique_experts:
            # Find tokens for this expert
            expert_mask = (flat_indices == expert_idx)
            if not expert_mask.any():
                continue

            token_indices, k_positions = torch.where(expert_mask)
            if len(token_indices) == 0:
                continue

            # Get expert weights from cache manager
            weights = self.cache_manager.get_expert_weights(self.layer_idx, expert_idx)

            # Manually compute SwiGLU: down(silu(gate(x)) * up(x))
            expert_input = flat_hidden[token_indices]

            gate_out = F.silu(F.linear(expert_input, weights["gate_proj"]))
            up_out = F.linear(expert_input, weights["up_proj"])
            expert_out = F.linear(gate_out * up_out, weights["down_proj"])

            # Get routing weights
            routing_weights = flat_weights[token_indices, k_positions].unsqueeze(-1)

            # Accumulate
            output.index_add_(0, token_indices, expert_out * routing_weights)

        # Add shared expert
        if self.shared_expert is not None:
            shared_out = self.shared_expert(flat_hidden) * self.shared_scale
            output = output + shared_out

        return output.view(batch_size, seq_len, hidden_dim)


class OffloadedHydraNetBlock(nn.Module):
    """Transformer block with offloaded MoE."""

    def __init__(
        self,
        config: HydraNetConfig,
        layer_idx: int,
        cache_manager: ExpertCacheManager,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.moe = OffloadedMoELayer(config, layer_idx, cache_manager)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_prefill: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """Forward pass."""
        residual = hidden_states

        # Attention
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        # MoE (with prefill vs decode switching)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        moe_output = self.moe(hidden_states, is_prefill=is_prefill)
        hidden_states = residual + moe_output

        return hidden_states, present_kv


class OffloadedHydraNet(nn.Module):
    """
    HydraNet with expert offloading for inference on limited VRAM.

    Key features:
    - Per-layer expert cache (prevents cross-layer thrashing)
    - Top-2 prefill / Top-1 decode (halves decode bandwidth)
    - Fixed VRAM slots per layer (no fragmentation)
    - Attention layers always on GPU
    """

    def __init__(
        self,
        config: HydraNetConfig,
        offload_config: Optional[OffloadConfig] = None,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.config = config
        self.offload_config = offload_config or OffloadConfig()
        self.device = device
        self.dtype = dtype

        # Calculate expert size
        expert_params = 3 * config.hidden_dim * config.expert_intermediate_dim
        self.expert_size_bytes = expert_params * (2 if dtype == torch.float16 else 4)

        # Create per-layer cache manager
        cache_config = PerLayerCacheConfig(
            hot_slots_per_layer=self.offload_config.hot_experts_per_layer,
            warm_slots_per_layer=self.offload_config.warm_experts_per_layer,
            total_gpu_budget_gb=self.offload_config.expert_gpu_budget_gb,
            use_cuda_streams=self.offload_config.use_async_loading,
            num_load_streams=self.offload_config.num_load_streams,
            enable_prefetch=self.offload_config.enable_prefetch,
            prefetch_next_layer=self.offload_config.prefetch_next_layer,
        )

        self.cache_manager = ExpertCacheManager(
            cache_config,
            num_layers=config.num_layers,
            num_experts=config.num_experts,
            expert_size_bytes=self.expert_size_bytes,
            device=device,
            dtype=dtype,
        )

        # Embeddings (on GPU)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            OffloadedHydraNetBlock(config, i, self.cache_manager)
            for i in range(config.num_layers)
        ])

        # Output (on GPU)
        self.norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def _init_expert_weights(self):
        """Initialize and register expert weights with cache manager."""
        print(f"Initializing {self.config.num_layers * self.config.num_experts} experts in RAM...")

        for layer_idx in range(self.config.num_layers):
            for expert_idx in range(self.config.num_experts):
                weights = {
                    "gate_proj": torch.randn(
                        self.config.expert_intermediate_dim,
                        self.config.hidden_dim,
                        dtype=self.dtype,
                    ) * 0.02,
                    "up_proj": torch.randn(
                        self.config.expert_intermediate_dim,
                        self.config.hidden_dim,
                        dtype=self.dtype,
                    ) * 0.02,
                    "down_proj": torch.randn(
                        self.config.hidden_dim,
                        self.config.expert_intermediate_dim,
                        dtype=self.dtype,
                    ) * 0.02,
                }
                self.cache_manager.register_expert(layer_idx, expert_idx, weights)

        # Set initial hot experts (first N experts per layer)
        self.cache_manager.set_hot_experts_uniform(self.offload_config.hot_experts_per_layer)

    def to_device(self):
        """Move non-expert components to GPU."""
        self.embed_tokens = self.embed_tokens.to(self.device, self.dtype)
        self.norm = self.norm.to(self.device, self.dtype)
        self.lm_head = self.lm_head.to(self.device, self.dtype)

        for layer in self.layers:
            layer.input_layernorm = layer.input_layernorm.to(self.device, self.dtype)
            layer.post_attention_layernorm = layer.post_attention_layernorm.to(self.device, self.dtype)
            layer.self_attn = layer.self_attn.to(self.device, self.dtype)
            layer.moe.router = layer.moe.router.to(self.device, self.dtype)
            if layer.moe.shared_expert is not None:
                layer.moe.shared_expert = layer.moe.shared_expert.to(self.device, self.dtype)
                layer.moe.shared_scale = layer.moe.shared_scale.to(self.device, self.dtype)

        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Forward pass with expert offloading.

        Automatically uses Top-2 for prefill, Top-1 for decode.
        """
        batch_size, seq_len = input_ids.shape

        # Determine if this is prefill or decode
        is_prefill = past_key_values is None or seq_len > 1

        hidden_states = self.embed_tokens(input_ids)

        # Position IDs
        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                device=self.device,
            ).unsqueeze(0).expand(batch_size, -1)

        # Causal mask
        if attention_mask is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values else 0
            attention_mask = create_causal_mask(
                seq_len, past_len, hidden_states.dtype, self.device
            )

        new_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None

            hidden_states, present_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
                is_prefill=is_prefill,
            )

            if use_cache:
                new_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, new_key_values

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate tokens with KV caching.

        Uses Top-2 for prefill, Top-1 for decode (automatic).
        """
        self.eval()
        past_key_values = None
        generated = input_ids

        with torch.no_grad():
            for step in range(max_new_tokens):
                if past_key_values is not None:
                    curr_input = generated[:, -1:]
                else:
                    curr_input = generated

                logits, past_key_values = self.forward(
                    curr_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def get_cache_stats(self) -> dict:
        """Get expert cache statistics."""
        return self.cache_manager.get_stats()

    def update_hot_experts(self):
        """Update hot experts based on usage statistics."""
        self.cache_manager.update_hot_experts_from_usage()


def create_offloaded_hydranet(
    size: str = "medium",
    offload_config: Optional[OffloadConfig] = None,
    device: str = "cuda",
) -> OffloadedHydraNet:
    """Create an offloaded HydraNet model."""
    from ..model.config import HydraNetConfigs

    configs = {
        "tiny": HydraNetConfigs.tiny,
        "small": HydraNetConfigs.small,
        "medium": HydraNetConfigs.medium,
        "large": HydraNetConfigs.large,
    }

    config = configs[size]()

    # Use config's hot_experts_per_layer if offload_config not specified
    if offload_config is None:
        offload_config = OffloadConfig(
            hot_experts_per_layer=config.hot_experts_per_layer,
        )

    model = OffloadedHydraNet(
        config,
        offload_config=offload_config,
        device=torch.device(device),
    )

    # Initialize experts in RAM
    model._init_expert_weights()

    # Move non-experts to GPU
    model.to_device()

    return model
