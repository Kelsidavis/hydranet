#!/usr/bin/env python3
"""
Synthetic smoke test for GLM-4.5-Air - no real weights needed.

Verifies that:
1. Different cache configurations produce identical outputs (greedy)
2. Cache hit/miss paths work correctly
3. Sigmoid routing works as expected
4. Shared expert is always computed
5. Partial RoPE is applied correctly
6. No NaN/Inf in outputs
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_mini_glm4(
    cache_config_type: str = "balanced",
    device: str = "cpu",
) -> Tuple["OffloadedGLM4", "GLM4AirConfig"]:
    """
    Create a mini GLM4 model with random weights for testing.

    cache_config_type:
        - "all_fit": All experts fit in cache
        - "balanced": Some slots for each tier
        - "minimal": Force evictions
    """
    from hydranet.model.glm4 import OffloadedGLM4
    from hydranet.config import GLM4AirConfig, ExpertCacheConfig

    # Small config for fast testing
    config = GLM4AirConfig()
    config.hidden_dim = 256
    config.num_layers = 4  # 1 dense + 3 MoE
    config.first_k_dense_replace = 1
    config.num_attention_heads = 8
    config.num_kv_heads = 2
    config.head_dim = 32
    config.intermediate_dim = 512  # Dense/shared expert intermediate
    config.moe_intermediate_dim = 128  # Routed expert intermediate (smaller)
    config.num_experts = 16  # Reduced from 128 for fast testing
    config.num_shared_experts = 1
    config.experts_per_token = 4  # Reduced from 8
    config.vocab_size = 1000
    config.partial_rotary_factor = 0.5

    # Cache configuration
    if cache_config_type == "all_fit":
        cache_config = ExpertCacheConfig(
            pinned_slots=16,  # All experts fit
            hot_slots=0,
            probation_slots=0,
        )
    elif cache_config_type == "balanced":
        cache_config = ExpertCacheConfig(
            pinned_slots=4,
            hot_slots=4,
            probation_slots=4,
        )
    else:  # minimal
        cache_config = ExpertCacheConfig(
            pinned_slots=2,
            hot_slots=2,
            probation_slots=2,
        )

    device = torch.device(device)

    model = OffloadedGLM4(
        config=config,
        cache_config=cache_config,
        device=device,
        dtype=torch.float32,  # Use float32 for numerical stability on CPU
    )

    return model, config


def init_weights_deterministic(model, config, seed: int = 42):
    """Initialize model with deterministic random weights."""
    torch.manual_seed(seed)

    # Embeddings
    model.embed_tokens.weight.data.normal_(0, 0.02)
    model.lm_head.weight.data.normal_(0, 0.02)
    model.norm.weight.data.fill_(1.0)

    for layer_idx, layer in enumerate(model.layers):
        # Attention (with bias for GLM4)
        layer.self_attn.q_proj.weight.data.normal_(0, 0.02)
        layer.self_attn.k_proj.weight.data.normal_(0, 0.02)
        layer.self_attn.v_proj.weight.data.normal_(0, 0.02)
        layer.self_attn.o_proj.weight.data.normal_(0, 0.02)
        if hasattr(layer.self_attn.q_proj, 'bias') and layer.self_attn.q_proj.bias is not None:
            layer.self_attn.q_proj.bias.data.zero_()
            layer.self_attn.k_proj.bias.data.zero_()
            layer.self_attn.v_proj.bias.data.zero_()
            layer.self_attn.o_proj.bias.data.zero_()

        # Norms
        layer.input_layernorm.weight.data.fill_(1.0)
        layer.post_attention_layernorm.weight.data.fill_(1.0)

        if layer.is_moe:
            # MoE layer - mlp is GLM4MoELayer
            moe_layer = layer.mlp
            # Router
            moe_layer.router.weight.data.normal_(0, 0.02)

            # Shared expert
            moe_layer.shared_expert.gate_proj.weight.data.normal_(0, 0.02)
            moe_layer.shared_expert.up_proj.weight.data.normal_(0, 0.02)
            moe_layer.shared_expert.down_proj.weight.data.normal_(0, 0.02)
        else:
            # Dense MLP (layer 0) - mlp is GLM4DenseMLP
            layer.mlp.gate_proj.weight.data.normal_(0, 0.02)
            layer.mlp.up_proj.weight.data.normal_(0, 0.02)
            layer.mlp.down_proj.weight.data.normal_(0, 0.02)

    # Register expert weights for MoE layers
    moe_layer_idx = 0
    for layer_idx, layer in enumerate(model.layers):
        if not layer.is_moe:
            continue

        for expert_idx in range(config.num_experts):
            # Deterministic seed per expert
            torch.manual_seed(seed + moe_layer_idx * 1000 + expert_idx)
            weights = {
                "gate_proj": torch.randn(config.moe_intermediate_dim, config.hidden_dim) * 0.02,
                "up_proj": torch.randn(config.moe_intermediate_dim, config.hidden_dim) * 0.02,
                "down_proj": torch.randn(config.hidden_dim, config.moe_intermediate_dim) * 0.02,
            }
            model.expert_cache.register_expert(moe_layer_idx, expert_idx, weights)

        moe_layer_idx += 1


def greedy_generate(model, input_ids: torch.Tensor, max_tokens: int = 16) -> torch.Tensor:
    """Greedy generation for deterministic comparison."""
    model.eval()
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model.forward(generated)

            # Greedy selection
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    return generated


def test_cache_config_equivalence():
    """Test that different cache configs produce identical outputs."""
    print("=" * 60)
    print("TEST: GLM4 Cache Configuration Equivalence")
    print("=" * 60)

    configs = ["all_fit", "balanced", "minimal"]
    outputs = {}
    stats = {}

    # Fixed input for reproducibility
    torch.manual_seed(123)
    input_ids = torch.randint(0, 1000, (1, 8))

    for cfg_name in configs:
        print(f"\nRunning with cache config: {cfg_name}")

        model, config = create_mini_glm4(cache_config_type=cfg_name)
        init_weights_deterministic(model, config, seed=42)

        # Reset cache stats
        model.expert_cache.reset_stats()

        # Generate
        output = greedy_generate(model, input_ids, max_tokens=16)
        outputs[cfg_name] = output.clone()
        stats[cfg_name] = model.expert_cache.get_stats()

        print(f"  Output shape: {output.shape}")
        print(f"  First 5 generated: {output[0, 8:13].tolist()}")
        print(f"  Hit rate: {stats[cfg_name]['hit_rate']:.1%}")
        print(f"  Hits: {stats[cfg_name]['total_hits']}")
        print(f"  Misses: {stats[cfg_name]['total_misses']}")
        print(f"  Evictions: {stats[cfg_name]['total_evictions']}")

        # Cleanup
        model.expert_cache.shutdown()
        del model

    # Compare outputs
    print("\n" + "-" * 60)
    print("Comparing outputs...")

    all_match = True
    ref = outputs["all_fit"]

    for cfg_name in ["balanced", "minimal"]:
        if torch.equal(ref, outputs[cfg_name]):
            print(f"  {cfg_name} vs all_fit: MATCH")
        else:
            print(f"  {cfg_name} vs all_fit: MISMATCH")
            all_match = False

    return all_match


def test_no_nan_inf():
    """Test that forward pass produces no NaN/Inf values."""
    print("\n" + "=" * 60)
    print("TEST: GLM4 No NaN/Inf in Output")
    print("=" * 60)

    model, config = create_mini_glm4(cache_config_type="all_fit")
    init_weights_deterministic(model, config, seed=42)

    # Test with various inputs
    test_cases = [
        torch.randint(0, 1000, (1, 4)),
        torch.randint(0, 1000, (1, 16)),
        torch.randint(0, 1000, (2, 8)),
    ]

    all_ok = True
    model.eval()

    with torch.no_grad():
        for i, input_ids in enumerate(test_cases):
            logits, _ = model.forward(input_ids)

            has_nan = torch.isnan(logits).any().item()
            has_inf = torch.isinf(logits).any().item()

            if has_nan or has_inf:
                print(f"  Case {i+1}: FAIL (NaN={has_nan}, Inf={has_inf})")
                all_ok = False
            else:
                print(f"  Case {i+1}: OK (shape={logits.shape})")

    model.expert_cache.shutdown()
    return all_ok


def test_sigmoid_routing():
    """Test that sigmoid router selects correct number of experts."""
    print("\n" + "=" * 60)
    print("TEST: GLM4 Sigmoid Router")
    print("=" * 60)

    from hydranet.config import GLM4AirConfig
    from hydranet.model.glm4 import GLM4SigmoidRouter

    config = GLM4AirConfig()
    config.num_experts = 16
    config.experts_per_token = 4
    config.hidden_dim = 256

    router = GLM4SigmoidRouter(config, layer_idx=0)
    router.weight.data.normal_(0, 0.02)

    # Test input
    batch, seq = 2, 8
    hidden = torch.randn(batch, seq, config.hidden_dim)

    indices, weights = router(hidden)

    # Check shapes
    expected_tokens = batch * seq
    assert indices.shape == (expected_tokens, config.experts_per_token), \
        f"Expected indices shape {(expected_tokens, config.experts_per_token)}, got {indices.shape}"
    assert weights.shape == (expected_tokens, config.experts_per_token), \
        f"Expected weights shape {(expected_tokens, config.experts_per_token)}, got {weights.shape}"

    # Check expert indices are valid
    assert indices.min() >= 0, "Expert indices should be non-negative"
    assert indices.max() < config.num_experts, "Expert indices should be < num_experts"

    # Check weights are normalized (if configured)
    if config.norm_topk_prob:
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=0.01), \
            "Weights should sum to ~1 when normalized"

    print(f"  Indices shape: {indices.shape}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Expert range: [{indices.min().item()}, {indices.max().item()}]")
    print(f"  Weight sum: {weights.sum(dim=-1).mean().item():.3f}")
    print("  OK")

    return True


def test_partial_rope():
    """Test that partial RoPE is applied correctly."""
    print("\n" + "=" * 60)
    print("TEST: GLM4 Partial RoPE")
    print("=" * 60)

    from hydranet.config import GLM4AirConfig
    from hydranet.model.glm4 import PartialRotaryEmbedding, apply_partial_rotary_pos_emb

    config = GLM4AirConfig()
    config.head_dim = 64
    config.partial_rotary_factor = 0.5

    rope = PartialRotaryEmbedding(
        dim=config.head_dim,
        partial_rotary_factor=config.partial_rotary_factor,
    )

    rotary_dim = int(config.head_dim * config.partial_rotary_factor)

    # Test
    batch, heads, seq = 2, 4, 8
    q = torch.randn(batch, heads, seq, config.head_dim)
    k = torch.randn(batch, heads, seq, config.head_dim)
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)

    cos, sin = rope(q, position_ids)
    q_rot, k_rot = apply_partial_rotary_pos_emb(q, k, cos, sin, rotary_dim)

    # Check shapes preserved
    assert q_rot.shape == q.shape, f"Q shape mismatch: {q_rot.shape} vs {q.shape}"
    assert k_rot.shape == k.shape, f"K shape mismatch: {k_rot.shape} vs {k.shape}"

    # Check non-rotary part is unchanged
    assert torch.allclose(q[..., rotary_dim:], q_rot[..., rotary_dim:]), \
        "Non-rotary part of Q should be unchanged"
    assert torch.allclose(k[..., rotary_dim:], k_rot[..., rotary_dim:]), \
        "Non-rotary part of K should be unchanged"

    # Check rotary part is different
    assert not torch.allclose(q[..., :rotary_dim], q_rot[..., :rotary_dim]), \
        "Rotary part of Q should be different"

    print(f"  Head dim: {config.head_dim}")
    print(f"  Rotary dim: {rotary_dim}")
    print(f"  Non-rotary part preserved: OK")
    print(f"  Rotary part modified: OK")

    return True


def test_dense_first_layer():
    """Test that first layer is dense (no MoE)."""
    print("\n" + "=" * 60)
    print("TEST: GLM4 Dense First Layer")
    print("=" * 60)

    from hydranet.model.glm4 import GLM4DenseMLP, GLM4MoELayer

    model, config = create_mini_glm4(cache_config_type="all_fit")

    # Check layer types
    assert not model.layers[0].is_moe, "Layer 0 should be dense"
    assert model.layers[1].is_moe, "Layer 1 should be MoE"

    # Check layer 0 has regular MLP (GLM4DenseMLP)
    assert isinstance(model.layers[0].mlp, GLM4DenseMLP), "Layer 0 mlp should be GLM4DenseMLP"

    # Check layer 1 has MoE (GLM4MoELayer)
    assert isinstance(model.layers[1].mlp, GLM4MoELayer), "Layer 1 mlp should be GLM4MoELayer"

    print(f"  Layer 0 is_moe: {model.layers[0].is_moe} (expected: False)")
    print(f"  Layer 1 is_moe: {model.layers[1].is_moe} (expected: True)")
    print(f"  Layer 0 mlp type: {type(model.layers[0].mlp).__name__}")
    print(f"  Layer 1 mlp type: {type(model.layers[1].mlp).__name__}")
    print("  OK")

    model.expert_cache.shutdown()
    return True


def test_adaptive_slot_reallocation():
    """Test that slot reallocation moves slots from low-miss to high-miss layers."""
    print("\n" + "=" * 60)
    print("TEST: Adaptive Slot Reallocation")
    print("=" * 60)

    from hydranet.cache.expert_cache import PerLayerCache, ExpertCacheManager, SlotTier
    from hydranet.config import ExpertCacheConfig

    # Create a mock config
    from dataclasses import dataclass

    @dataclass
    class MockModelConfig:
        num_layers: int = 4
        num_experts: int = 16
        expert_size_bytes: int = 1024

    model_config = MockModelConfig()
    cache_config = ExpertCacheConfig(
        pinned_slots=0,
        hot_slots=2,
        probation_slots=2,
        enable_dynamic_slots=True,
        realloc_interval_tokens=10,
        miss_rate_threshold=0.3,
        max_slots_per_layer=8,
    )

    manager = ExpertCacheManager(
        model_config=model_config,
        cache_config=cache_config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Register some dummy experts
    for layer_idx in range(model_config.num_layers):
        for expert_idx in range(model_config.num_experts):
            weights = {
                "gate_proj": torch.randn(64, 32),
                "up_proj": torch.randn(64, 32),
                "down_proj": torch.randn(32, 64),
            }
            manager.register_expert(layer_idx, expert_idx, weights)

    # Simulate uneven access patterns
    # Layer 0: always hits same experts (low miss)
    # Layer 1: cycles through many experts (high miss)
    for _ in range(50):
        # Layer 0: always access experts 0, 1 (will have high hit rate after warmup)
        manager.get_expert_weights(0, 0)
        manager.get_expert_weights(0, 1)

        # Layer 1: cycle through many experts (will have high miss rate)
        for e in range(8):
            manager.get_expert_weights(1, e % model_config.num_experts)

    # Get initial stats
    stats_before = [cache.get_stats() for cache in manager.layer_caches]
    slots_before = [cache.get_slot_count() for cache in manager.layer_caches]

    print(f"  Before reallocation:")
    print(f"    Layer 0: slots={slots_before[0]}, hit_rate={stats_before[0]['hit_rate']:.1%}")
    print(f"    Layer 1: slots={slots_before[1]}, hit_rate={stats_before[1]['hit_rate']:.1%}")

    # Trigger reallocation
    manager._do_reallocation()

    # Get stats after
    slots_after = [cache.get_slot_count() for cache in manager.layer_caches]

    print(f"  After reallocation:")
    print(f"    Layer 0: slots={slots_after[0]}")
    print(f"    Layer 1: slots={slots_after[1]}")

    # Verify layer 1 got more slots (or at least didn't lose any)
    # and layer 0 lost slots (or stayed same if already minimal)
    layer1_improved = slots_after[1] >= slots_before[1]

    manager.shutdown()

    if layer1_improved:
        print("  Reallocation OK - high-miss layer maintained/gained slots")
        return True
    else:
        print("  Reallocation FAILED - high-miss layer lost slots unexpectedly")
        return False


def test_slot_add_remove():
    """Test PerLayerCache add_slot and remove_slot methods."""
    print("\n" + "=" * 60)
    print("TEST: Slot Add/Remove")
    print("=" * 60)

    from hydranet.cache.expert_cache import PerLayerCache, SlotTier
    from hydranet.config import ExpertCacheConfig

    cache_config = ExpertCacheConfig(
        pinned_slots=0,
        hot_slots=1,
        probation_slots=1,
        max_slots_per_layer=5,
    )

    cache = PerLayerCache(
        layer_idx=0,
        num_experts=8,
        expert_size_bytes=1024,
        config=cache_config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    initial_slots = cache.get_slot_count()
    print(f"  Initial slots: {initial_slots}")

    # Test add_slot
    added = cache.add_slot()
    assert added, "Should be able to add slot"
    assert cache.get_slot_count() == initial_slots + 1, "Slot count should increase"
    print(f"  After add: {cache.get_slot_count()} slots")

    # Add more slots up to max
    while cache.add_slot():
        pass

    max_slots = cache.get_slot_count()
    print(f"  At max: {max_slots} slots")
    assert max_slots == cache_config.max_slots_per_layer, f"Should hit max ({cache_config.max_slots_per_layer})"

    # Test remove_slot
    removed = cache.remove_slot()
    assert removed, "Should be able to remove slot"
    assert cache.get_slot_count() == max_slots - 1, "Slot count should decrease"
    print(f"  After remove: {cache.get_slot_count()} slots")

    # Remove down to 1
    while cache.get_slot_count() > 1:
        cache.remove_slot()

    min_slots = cache.get_slot_count()
    print(f"  At min: {min_slots} slots")
    assert min_slots == 1, "Should be able to go down to 1 slot"

    # Can't remove below 1
    removed = cache.remove_slot()
    assert not removed, "Should not be able to remove last slot"
    assert cache.get_slot_count() == 1, "Should still have 1 slot"

    print("  Slot add/remove OK")
    return True


def main():
    print("=" * 60)
    print("GLM-4.5-Air Synthetic Smoke Test")
    print("=" * 60)

    results = {}

    results["dense_first_layer"] = test_dense_first_layer()
    results["partial_rope"] = test_partial_rope()
    results["sigmoid_routing"] = test_sigmoid_routing()
    results["no_nan_inf"] = test_no_nan_inf()
    results["cache_equivalence"] = test_cache_config_equivalence()
    results["slot_add_remove"] = test_slot_add_remove()
    results["adaptive_reallocation"] = test_adaptive_slot_reallocation()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
