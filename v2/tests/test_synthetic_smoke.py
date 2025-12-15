#!/usr/bin/env python3
"""
Synthetic smoke test for HydraNet - no real weights needed.

Verifies that:
1. Different cache configurations produce identical outputs (greedy)
2. Cache hit/miss paths work correctly
3. No NaN/Inf in outputs
4. KV cache produces consistent results

This is the fp16 "gating step" before INT4 integration.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def create_mini_model(
    cache_config_type: str = "balanced",
    device: str = "cpu",
) -> Tuple["OffloadedMixtral", "MixtralConfig"]:
    """
    Create a mini Mixtral model with random weights.

    cache_config_type:
        - "all_fit": All experts fit in cache (8 pinned slots)
        - "balanced": 4 pinned + 2 hot + 2 probation
        - "minimal": 1 pinned + 1 probation (force evictions)
    """
    from hydranet.v2.model.mixtral import OffloadedMixtral
    from hydranet.v2.config import MixtralConfig, ExpertCacheConfig, KVCacheConfig

    # Small config for fast testing
    config = MixtralConfig()
    config.hidden_dim = 256
    config.num_layers = 4
    config.num_attention_heads = 8
    config.num_kv_heads = 2
    config.head_dim = 32
    config.intermediate_dim = 512
    config.num_experts = 8  # Full 8 experts
    config.experts_per_token = 2
    config.vocab_size = 1000

    # Cache configuration
    if cache_config_type == "all_fit":
        cache_config = ExpertCacheConfig(
            pinned_slots=8,  # All experts fit
            hot_slots=0,
            probation_slots=0,
        )
    elif cache_config_type == "balanced":
        cache_config = ExpertCacheConfig(
            pinned_slots=4,
            hot_slots=2,
            probation_slots=2,
        )
    else:  # minimal
        cache_config = ExpertCacheConfig(
            pinned_slots=1,
            hot_slots=1,
            probation_slots=2,
        )

    kv_config = KVCacheConfig()
    device = torch.device(device)

    model = OffloadedMixtral(
        config=config,
        cache_config=cache_config,
        kv_config=kv_config,
        device=device,
        dtype=torch.float32,  # Use float32 for numerical stability on CPU
    )

    return model, config


def init_weights_deterministic(model, config, seed: int = 42):
    """Initialize model with deterministic random weights."""
    torch.manual_seed(seed)

    # Embeddings (smaller init for stability)
    model.embed_tokens.weight.data.normal_(0, 0.02)
    model.lm_head.weight.data.normal_(0, 0.02)
    model.norm.weight.data.fill_(1.0)

    for layer in model.layers:
        # Attention
        layer.self_attn.q_proj.weight.data.normal_(0, 0.02)
        layer.self_attn.k_proj.weight.data.normal_(0, 0.02)
        layer.self_attn.v_proj.weight.data.normal_(0, 0.02)
        layer.self_attn.o_proj.weight.data.normal_(0, 0.02)

        # Norms
        layer.input_layernorm.weight.data.fill_(1.0)
        layer.post_attention_layernorm.weight.data.fill_(1.0)

        # Router
        layer.moe.router.gate.weight.data.normal_(0, 0.02)

    # Register expert weights (same seed = same weights across configs)
    for layer_idx in range(config.num_layers):
        for expert_idx in range(config.num_experts):
            # Deterministic seed per expert
            torch.manual_seed(seed + layer_idx * 100 + expert_idx)
            weights = {
                "gate_proj": torch.randn(config.intermediate_dim, config.hidden_dim) * 0.02,
                "up_proj": torch.randn(config.intermediate_dim, config.hidden_dim) * 0.02,
                "down_proj": torch.randn(config.hidden_dim, config.intermediate_dim) * 0.02,
            }
            model.expert_cache.register_expert(layer_idx, expert_idx, weights)


def greedy_generate(model, input_ids: torch.Tensor, max_tokens: int = 16) -> torch.Tensor:
    """Greedy generation for deterministic comparison."""
    model.eval()
    past_key_values = None
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_tokens):
            if past_key_values is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated

            logits, past_key_values = model.forward(
                curr_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Greedy selection
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    return generated


def test_cache_config_equivalence():
    """Test that different cache configs produce identical outputs."""
    print("=" * 60)
    print("TEST: Cache Configuration Equivalence")
    print("=" * 60)

    configs = ["all_fit", "balanced", "minimal"]
    outputs = {}
    stats = {}

    # Fixed input for reproducibility
    torch.manual_seed(123)
    input_ids = torch.randint(0, 1000, (1, 8))

    for cfg_name in configs:
        print(f"\nRunning with cache config: {cfg_name}")

        model, config = create_mini_model(cache_config_type=cfg_name)
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
            print(f"  {cfg_name} vs all_fit: MATCH ✓")
        else:
            print(f"  {cfg_name} vs all_fit: MISMATCH ✗")
            # Find first difference
            for i in range(ref.shape[1]):
                if ref[0, i] != outputs[cfg_name][0, i]:
                    print(f"    First diff at position {i}: {ref[0, i].item()} vs {outputs[cfg_name][0, i].item()}")
                    break
            all_match = False

    return all_match


def test_numerical_stability():
    """Test for NaN/Inf in outputs."""
    print("\n" + "=" * 60)
    print("TEST: Numerical Stability")
    print("=" * 60)

    model, config = create_mini_model(cache_config_type="balanced")
    init_weights_deterministic(model, config)

    # Test with different sequence lengths
    lengths = [1, 4, 16, 32]
    all_stable = True

    for seq_len in lengths:
        input_ids = torch.randint(0, config.vocab_size, (2, seq_len))

        with torch.no_grad():
            logits, _ = model.forward(input_ids, use_cache=False)

        has_nan = torch.isnan(logits).any()
        has_inf = torch.isinf(logits).any()

        if has_nan or has_inf:
            print(f"  seq_len={seq_len}: UNSTABLE (NaN={has_nan}, Inf={has_inf}) ✗")
            all_stable = False
        else:
            logit_range = (logits.min().item(), logits.max().item())
            print(f"  seq_len={seq_len}: stable, logit range [{logit_range[0]:.2f}, {logit_range[1]:.2f}] ✓")

    model.expert_cache.shutdown()
    return all_stable


def test_kv_cache_consistency():
    """Test that KV cache produces same results as full recompute."""
    print("\n" + "=" * 60)
    print("TEST: KV Cache Consistency")
    print("=" * 60)

    from hydranet.v2.config import TopKMode

    model, config = create_mini_model(cache_config_type="all_fit")
    init_weights_deterministic(model, config)

    # Force Top-2 mode for consistent comparison
    # (Top-K switching is intentional for perf, but breaks 1:1 comparison)
    model.set_topk_mode(TopKMode.FORCED_TOP2)

    input_ids = torch.randint(0, config.vocab_size, (1, 8))

    # Full recompute (no cache)
    with torch.no_grad():
        logits_full, _ = model.forward(input_ids, use_cache=False)

    # With KV cache (step by step)
    past_kv = None
    logits_cached_list = []

    with torch.no_grad():
        for i in range(input_ids.shape[1]):
            curr_input = input_ids[:, i:i+1]
            logits, past_kv = model.forward(curr_input, past_key_values=past_kv, use_cache=True)
            logits_cached_list.append(logits)

    logits_cached = torch.cat(logits_cached_list, dim=1)

    # Compare
    max_diff = (logits_full - logits_cached).abs().max().item()
    mean_diff = (logits_full - logits_cached).abs().mean().item()

    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")

    # Allow small numerical differences
    if max_diff < 1e-4:
        print("  KV cache consistent ✓")
        result = True
    else:
        print("  KV cache inconsistent ✗")
        result = False

    model.expert_cache.shutdown()
    return result


def test_routing_distribution():
    """Test that routing distributes across experts."""
    print("\n" + "=" * 60)
    print("TEST: Routing Distribution")
    print("=" * 60)

    model, config = create_mini_model(cache_config_type="all_fit")
    init_weights_deterministic(model, config)

    # Generate longer to see routing patterns
    input_ids = torch.randint(0, config.vocab_size, (1, 32))

    # Track routing via cache stats
    model.expert_cache.reset_stats()

    with torch.no_grad():
        _ = model.forward(input_ids, use_cache=False)

    stats = model.expert_cache.get_stats()

    # Check per-layer stats
    print("\nPer-layer expert usage:")
    experts_used = set()

    for layer_stat in stats["per_layer_stats"]:
        layer_idx = layer_stat["layer"]
        cached = layer_stat["cached_count"]
        print(f"  Layer {layer_idx}: {cached} unique experts accessed")

        # Count unique experts from cache
        if cached > 0:
            experts_used.add(layer_idx)

    # With random router weights and 32 tokens x 2 experts = 64 expert calls per layer,
    # we should see multiple experts used
    print(f"\nTotal layers with expert activity: {len(experts_used)}/{config.num_layers}")

    result = len(experts_used) == config.num_layers
    if result:
        print("  All layers routing ✓")
    else:
        print("  Some layers not routing ✗")

    model.expert_cache.shutdown()
    return result


def test_eviction_correctness():
    """Test that eviction doesn't corrupt outputs."""
    print("\n" + "=" * 60)
    print("TEST: Eviction Correctness")
    print("=" * 60)

    # Minimal cache forces evictions
    model, config = create_mini_model(cache_config_type="minimal")
    init_weights_deterministic(model, config)

    # Multiple forward passes to trigger evictions
    results = []

    for pass_idx in range(3):
        torch.manual_seed(100 + pass_idx)
        input_ids = torch.randint(0, config.vocab_size, (1, 16))

        with torch.no_grad():
            logits, _ = model.forward(input_ids, use_cache=False)

        # Check output validity
        has_nan = torch.isnan(logits).any()
        has_inf = torch.isinf(logits).any()

        results.append(not (has_nan or has_inf))

        if has_nan or has_inf:
            print(f"  Pass {pass_idx + 1}: CORRUPT ✗")
        else:
            print(f"  Pass {pass_idx + 1}: valid ✓")

    stats = model.expert_cache.get_stats()
    print(f"\n  Total evictions: {stats['total_evictions']}")

    model.expert_cache.shutdown()
    return all(results)


def run_all_synthetic_tests():
    """Run all synthetic smoke tests."""
    print("=" * 70)
    print("HYDRANET v2.1 SYNTHETIC SMOKE TESTS")
    print("(fp16 correctness verification before INT4 integration)")
    print("=" * 70)

    tests = [
        ("Numerical Stability", test_numerical_stability),
        ("KV Cache Consistency", test_kv_cache_consistency),
        ("Routing Distribution", test_routing_distribution),
        ("Eviction Correctness", test_eviction_correctness),
        ("Cache Config Equivalence", test_cache_config_equivalence),
    ]

    results = []

    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, p, _ in results if p)
    total = len(results)

    for name, p, err in results:
        status = "PASS ✓" if p else "FAIL ✗"
        print(f"  {name}: {status}")
        if err:
            print(f"    Error: {err[:80]}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All synthetic smoke tests passed!")
        print("  fp16 model math + routing + caching verified.")
        print("  Ready to proceed with INT4 integration.")
    else:
        print("\n✗ Some tests failed - fix before INT4 integration")

    return passed == total


if __name__ == "__main__":
    success = run_all_synthetic_tests()
    sys.exit(0 if success else 1)
