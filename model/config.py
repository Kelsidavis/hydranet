"""HydraNet Model Configuration"""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class HydraNetConfig:
    """Configuration for HydraNet MoE model."""

    # Core transformer dimensions
    hidden_dim: int = 4096
    num_layers: int = 28
    num_attention_heads: int = 32
    num_kv_heads: int = 8  # GQA: 4 query heads per KV head
    head_dim: int = 128

    # Vocabulary and context
    vocab_size: int = 32000
    max_context: int = 16384
    rope_theta: float = 500000.0  # Extended RoPE
    rope_scaling: Optional[dict] = None

    # MoE configuration
    num_experts: int = 16  # Per layer (reduced from 64 for cache efficiency)
    experts_per_token: int = 2  # Top-k routing for prefill
    decode_experts_per_token: int = 1  # Top-k for decode (Top-1 = faster, less cache thrash)
    expert_intermediate_dim: int = 6144
    use_shared_expert: bool = True
    expert_capacity_factor: float = 1.25  # For load balancing

    # Per-layer cache configuration
    hot_experts_per_layer: int = 4  # Experts to keep hot in VRAM per layer

    # Router configuration
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001

    # Numerical precision
    dtype: str = "bfloat16"
    attention_dtype: str = "float32"  # For stability

    # Regularization
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    expert_dropout: float = 0.0

    # Initialization
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6

    # Activation
    hidden_act: str = "silu"  # SwiGLU

    # Inference optimization
    use_cache: bool = True
    tie_word_embeddings: bool = False

    @property
    def total_params_billions(self) -> float:
        """Estimate total parameters in billions."""
        # Embeddings
        embed_params = self.vocab_size * self.hidden_dim * 2  # in + out

        # Attention per layer
        q_params = self.hidden_dim * self.num_attention_heads * self.head_dim
        kv_params = self.hidden_dim * self.num_kv_heads * self.head_dim * 2
        o_params = self.num_attention_heads * self.head_dim * self.hidden_dim
        attn_per_layer = q_params + kv_params + o_params

        # Expert params (SwiGLU: gate + up + down)
        expert_params = 3 * self.hidden_dim * self.expert_intermediate_dim
        total_experts = self.num_layers * self.num_experts
        if self.use_shared_expert:
            total_experts += self.num_layers

        # Norms
        norm_params = self.num_layers * self.hidden_dim * 4  # 4 norms per layer

        total = (
            embed_params +
            self.num_layers * attn_per_layer +
            total_experts * expert_params +
            norm_params
        )

        return total / 1e9

    @property
    def active_params_billions(self) -> float:
        """Estimate active parameters per forward pass."""
        embed_params = self.vocab_size * self.hidden_dim

        q_params = self.hidden_dim * self.num_attention_heads * self.head_dim
        kv_params = self.hidden_dim * self.num_kv_heads * self.head_dim * 2
        o_params = self.num_attention_heads * self.head_dim * self.hidden_dim
        attn_per_layer = q_params + kv_params + o_params

        expert_params = 3 * self.hidden_dim * self.expert_intermediate_dim
        active_experts_per_layer = self.experts_per_token
        if self.use_shared_expert:
            active_experts_per_layer += 1

        norm_params = self.num_layers * self.hidden_dim * 4

        total = (
            embed_params +
            self.num_layers * attn_per_layer +
            self.num_layers * active_experts_per_layer * expert_params +
            norm_params
        )

        return total / 1e9

    @property
    def expert_size_mb(self) -> float:
        """Size of single expert in MB at 4-bit quantization."""
        params = 3 * self.hidden_dim * self.expert_intermediate_dim
        return params * 0.5 / 1e6  # 4-bit = 0.5 bytes per param

    @property
    def kv_cache_size_mb(self) -> float:
        """KV cache size in MB for max context at fp16."""
        # Per layer: 2 (K,V) * num_kv_heads * head_dim * context * 2 bytes
        per_layer = 2 * self.num_kv_heads * self.head_dim * self.max_context * 2
        return self.num_layers * per_layer / 1e6

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "HydraNetConfig":
        return cls(**d)

    @classmethod
    def load(cls, path: str) -> "HydraNetConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# Preset configurations
class HydraNetConfigs:
    """Preset configurations for different scales."""

    @staticmethod
    def tiny() -> HydraNetConfig:
        """Tiny model for testing (~500M total, ~200M active)."""
        return HydraNetConfig(
            hidden_dim=1024,
            num_layers=12,
            num_attention_heads=16,
            num_kv_heads=4,
            head_dim=64,
            num_experts=8,
            experts_per_token=2,
            expert_intermediate_dim=2048,
            max_context=4096,
        )

    @staticmethod
    def small() -> HydraNetConfig:
        """Small model for initial training (~8B total, ~2B active)."""
        return HydraNetConfig(
            hidden_dim=2048,
            num_layers=20,
            num_attention_heads=16,
            num_kv_heads=4,
            head_dim=128,
            num_experts=16,
            experts_per_token=2,
            expert_intermediate_dim=4096,
            max_context=8192,
        )

    @staticmethod
    def medium() -> HydraNetConfig:
        """Medium model (~25B total, ~4B active)."""
        return HydraNetConfig(
            hidden_dim=3072,
            num_layers=24,
            num_attention_heads=24,
            num_kv_heads=6,
            head_dim=128,
            num_experts=16,
            experts_per_token=2,
            decode_experts_per_token=1,
            expert_intermediate_dim=5120,
            max_context=16384,
            hot_experts_per_layer=4,
        )

    @staticmethod
    def large() -> HydraNetConfig:
        """Full HydraNet (~37B total, ~6B active) - optimized for RAM↔VRAM swapping."""
        return HydraNetConfig(
            hidden_dim=4096,
            num_layers=28,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            num_experts=16,  # Reduced from 64 for cache efficiency
            experts_per_token=2,  # Top-2 for prefill
            decode_experts_per_token=1,  # Top-1 for decode (2x speedup)
            expert_intermediate_dim=8192,  # Larger experts to compensate
            max_context=16384,
            hot_experts_per_layer=6,  # 6 hot per layer = 168 total in VRAM
        )


if __name__ == "__main__":
    # Print stats for all configs
    for name, config_fn in [
        ("tiny", HydraNetConfigs.tiny),
        ("small", HydraNetConfigs.small),
        ("medium", HydraNetConfigs.medium),
        ("large", HydraNetConfigs.large),
    ]:
        cfg = config_fn()
        print(f"\n{name.upper()} Config:")
        print(f"  Total params:  {cfg.total_params_billions:.1f}B")
        print(f"  Active params: {cfg.active_params_billions:.1f}B")
        print(f"  Expert size:   {cfg.expert_size_mb:.1f}MB (4-bit)")
        print(f"  KV cache:      {cfg.kv_cache_size_mb:.0f}MB (fp16, max context)")
