#!/usr/bin/env python3
"""Smoke test for GLM4 implementation - verifies imports and basic config."""

import sys
from pathlib import Path

# Add parent directory so 'hydranet' package is found
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config():
    """Test GLM4 config values."""
    from hydranet.config import GLM4AirConfig

    config = GLM4AirConfig()

    print("GLM-4.5-Air Configuration")
    print("=" * 50)
    print(f"Hidden dim: {config.hidden_dim}")
    print(f"Num layers: {config.num_layers} ({config.first_k_dense_replace} dense + {config.num_moe_layers} MoE)")
    print(f"Attention heads: {config.num_attention_heads} Q, {config.num_kv_heads} KV")
    print(f"Head dim: {config.head_dim}")
    print(f"Vocab size: {config.vocab_size:,}")
    print()
    print("MoE Config:")
    print(f"  Routed experts: {config.num_experts}")
    print(f"  Shared experts: {config.num_shared_experts}")
    print(f"  Experts per token: {config.experts_per_token}")
    print(f"  MoE intermediate: {config.moe_intermediate_dim}")
    print(f"  Dense intermediate: {config.intermediate_dim}")
    print()
    print("Memory Estimates:")
    print(f"  Total params: {config.total_params_b:.1f}B")
    print(f"  Active params: {config.active_params_b:.1f}B")
    print(f"  Expert size (INT4): {config.expert_size_mb:.2f} MB")
    print(f"  Total routed experts: {config.total_routed_experts:,}")

    # Verify key values match HF config
    assert config.hidden_dim == 4096
    assert config.num_layers == 46
    assert config.num_experts == 128
    assert config.experts_per_token == 8
    assert config.moe_intermediate_dim == 1408
    print("\n✓ Config values verified!")


def test_imports():
    """Test that all GLM4 modules import correctly."""
    print("\nTesting imports...")

    from hydranet.model.glm4 import (
        OffloadedGLM4,
        GLM4Attention,
        GLM4SigmoidRouter,
        GLM4MoELayer,
        GLM4Block,
        PartialRotaryEmbedding,
    )
    print("  ✓ glm4.py imports OK")

    from hydranet.model.glm4_loader import GLM4WeightLoader
    print("  ✓ glm4_loader.py imports OK")

    from hydranet import GLM4AirConfig
    print("  ✓ Top-level exports OK")


def test_model_init():
    """Test model initialization (without weights)."""
    import torch
    from hydranet.config import GLM4AirConfig, ExpertCacheConfig
    from hydranet.model.glm4 import OffloadedGLM4

    print("\nTesting model initialization...")

    config = GLM4AirConfig()
    cache_config = ExpertCacheConfig(
        pinned_slots=0,
        hot_slots=4,
        probation_slots=4,
    )

    # Initialize on CPU to avoid VRAM usage
    model = OffloadedGLM4(
        config=config,
        cache_config=cache_config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    print(f"  Model created with {len(model.layers)} layers")
    print(f"  First layer MoE: {model.layers[0].is_moe}")
    print(f"  Second layer MoE: {model.layers[1].is_moe}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Initialized parameters: {total_params / 1e9:.2f}B")

    # Verify layer structure
    assert not model.layers[0].is_moe, "Layer 0 should be dense"
    assert model.layers[1].is_moe, "Layer 1+ should be MoE"
    assert len(model.layers) == config.num_layers

    print("  ✓ Model structure verified!")


def test_router():
    """Test sigmoid router."""
    import torch
    from hydranet.config import GLM4AirConfig
    from hydranet.model.glm4 import GLM4SigmoidRouter

    print("\nTesting sigmoid router...")

    config = GLM4AirConfig()
    router = GLM4SigmoidRouter(config, layer_idx=0)

    # Test input
    batch, seq, hidden = 2, 4, config.hidden_dim
    x = torch.randn(batch, seq, hidden)

    indices, weights = router(x)

    print(f"  Input: {x.shape}")
    print(f"  Expert indices: {indices.shape}")  # Should be (batch*seq, top_k)
    print(f"  Expert weights: {weights.shape}")

    assert indices.shape == (batch * seq, config.experts_per_token)
    assert weights.shape == (batch * seq, config.experts_per_token)
    assert indices.max() < config.num_experts
    assert indices.min() >= 0

    # Weights should sum to ~1 (normalized)
    if config.norm_topk_prob:
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=0.01)

    print("  ✓ Router output verified!")


def test_partial_rope():
    """Test partial rotary embeddings."""
    import torch
    from hydranet.config import GLM4AirConfig
    from hydranet.model.glm4 import PartialRotaryEmbedding, apply_partial_rotary_pos_emb

    print("\nTesting partial RoPE...")

    config = GLM4AirConfig()
    rope = PartialRotaryEmbedding(
        dim=config.head_dim,
        partial_rotary_factor=config.partial_rotary_factor,
    )

    rotary_dim = int(config.head_dim * config.partial_rotary_factor)
    print(f"  Head dim: {config.head_dim}, Rotary dim: {rotary_dim}")

    # Test
    batch, heads, seq = 2, 4, 8
    q = torch.randn(batch, heads, seq, config.head_dim)
    k = torch.randn(batch, heads, seq, config.head_dim)
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)

    cos, sin = rope(q, position_ids)
    q_rot, k_rot = apply_partial_rotary_pos_emb(q, k, cos, sin, rotary_dim)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

    # The non-rotary part should be unchanged
    assert torch.allclose(q[..., rotary_dim:], q_rot[..., rotary_dim:])
    assert torch.allclose(k[..., rotary_dim:], k_rot[..., rotary_dim:])

    print("  ✓ Partial RoPE verified!")


def main():
    print("=" * 60)
    print("GLM-4.5-Air Implementation Smoke Test")
    print("=" * 60)

    test_config()
    test_imports()
    test_model_init()
    test_router()
    test_partial_rope()

    print("\n" + "=" * 60)
    print("All smoke tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
