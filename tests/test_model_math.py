#!/usr/bin/env python3
"""
Unit tests for HydraNet model math with mock weights.

Tests model components in isolation before full integration:
1. RoPE implementation
2. GQA attention
3. Router top-k selection
4. Expert MLP (SwiGLU)
5. Full forward pass with mock weights

Run: python -m hydranet.v2.tests.test_model_math
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_rope():
    """Test Rotary Position Embedding."""
    print("Testing RoPE...")

    from hydranet.model.mixtral import RotaryEmbedding, apply_rotary_pos_emb

    head_dim = 128
    max_pos = 1024

    rope = RotaryEmbedding(head_dim, max_pos)

    # Test shapes
    batch, seq, heads = 2, 16, 8
    q = torch.randn(batch, heads, seq, head_dim)
    k = torch.randn(batch, heads, seq, head_dim)

    pos_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    cos, sin = rope(q, pos_ids)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    assert q_rot.shape == q.shape, f"Q shape mismatch: {q_rot.shape} vs {q.shape}"
    assert k_rot.shape == k.shape, f"K shape mismatch: {k_rot.shape} vs {k.shape}"

    # RoPE should not change magnitude significantly
    q_norm = q.norm(dim=-1).mean()
    q_rot_norm = q_rot.norm(dim=-1).mean()
    assert abs(q_norm - q_rot_norm) / q_norm < 0.1, "RoPE changed magnitude too much"

    print("  ✓ RoPE shapes correct")
    print("  ✓ RoPE preserves magnitude")
    return True


def test_rms_norm():
    """Test RMS LayerNorm."""
    print("Testing RMSNorm...")

    from hydranet.model.mixtral import RMSNorm

    hidden_dim = 256
    norm = RMSNorm(hidden_dim)

    x = torch.randn(2, 16, hidden_dim)
    y = norm(x)

    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

    # Output should have roughly unit variance per position
    var = y.var(dim=-1).mean()
    assert 0.5 < var < 2.0, f"Variance out of range: {var}"

    print("  ✓ RMSNorm shapes correct")
    print("  ✓ RMSNorm normalizes properly")
    return True


def test_gqa_attention():
    """Test Grouped Query Attention."""
    print("Testing GQA Attention...")

    from hydranet.model.mixtral import MixtralAttention
    from hydranet.config import MixtralConfig

    # Small config for testing
    config = MixtralConfig()
    config.hidden_dim = 256
    config.num_attention_heads = 8
    config.num_kv_heads = 2  # GQA 4:1
    config.head_dim = 32

    attn = MixtralAttention(config, layer_idx=0)

    batch, seq = 2, 16
    x = torch.randn(batch, seq, config.hidden_dim)
    pos_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)

    # Without cache
    out, kv = attn(x, pos_ids, use_cache=False)
    assert out.shape == x.shape, f"Output shape: {out.shape} vs {x.shape}"
    assert kv is None, "KV should be None when use_cache=False"

    # With cache
    out, kv = attn(x, pos_ids, use_cache=True)
    assert kv is not None, "KV should not be None when use_cache=True"
    k, v = kv
    assert k.shape == (batch, config.num_kv_heads, seq, config.head_dim)
    assert v.shape == (batch, config.num_kv_heads, seq, config.head_dim)

    # Decode step with cache
    new_x = torch.randn(batch, 1, config.hidden_dim)
    new_pos = torch.tensor([[seq]]).expand(batch, -1)
    out2, kv2 = attn(new_x, new_pos, past_key_value=kv, use_cache=True)
    assert out2.shape == (batch, 1, config.hidden_dim)
    k2, v2 = kv2
    assert k2.shape[2] == seq + 1, f"KV cache should grow: {k2.shape[2]} vs {seq + 1}"

    print("  ✓ GQA shapes correct")
    print("  ✓ KV cache works")
    return True


def test_router():
    """Test Top-K Router."""
    print("Testing Router...")

    from hydranet.model.router import TopKRouter
    from hydranet.config import MixtralConfig, ExpertCacheConfig

    config = MixtralConfig()
    config.hidden_dim = 256
    config.num_experts = 8
    config.experts_per_token = 2

    cache_config = ExpertCacheConfig()

    router = TopKRouter(config, cache_config, layer_idx=0)

    batch, seq = 2, 16
    x = torch.randn(batch, seq, config.hidden_dim)

    # Top-2 routing
    output = router(x, top_k_override=2)
    assert output.expert_indices.shape == (batch, seq, 2)
    assert output.expert_weights.shape == (batch, seq, 2)

    # Weights should sum to 1
    weight_sum = output.expert_weights.sum(dim=-1)
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5), \
        f"Weights don't sum to 1: {weight_sum}"

    # Indices should be in valid range
    assert output.expert_indices.min() >= 0
    assert output.expert_indices.max() < config.num_experts

    # Top-1 routing
    output1 = router(x, top_k_override=1)
    assert output1.expert_indices.shape == (batch, seq, 1)

    print("  ✓ Router shapes correct")
    print("  ✓ Weights sum to 1")
    print("  ✓ Top-K switching works")
    return True


def test_expert_mlp():
    """Test Expert MLP (SwiGLU)."""
    print("Testing Expert MLP...")

    hidden_dim = 256
    intermediate_dim = 512

    # Create manual SwiGLU
    gate_proj = torch.randn(intermediate_dim, hidden_dim) * 0.02
    up_proj = torch.randn(intermediate_dim, hidden_dim) * 0.02
    down_proj = torch.randn(hidden_dim, intermediate_dim) * 0.02

    x = torch.randn(8, hidden_dim)  # 8 tokens

    # Manual forward
    gate_out = F.silu(F.linear(x, gate_proj))
    up_out = F.linear(x, up_proj)
    out = F.linear(gate_out * up_out, down_proj)

    assert out.shape == x.shape, f"Output shape: {out.shape} vs {x.shape}"

    # Check not all zeros or NaN
    assert not torch.isnan(out).any(), "NaN in output"
    assert out.abs().mean() > 1e-6, "Output is all zeros"

    print("  ✓ SwiGLU shapes correct")
    print("  ✓ No NaN/zeros")
    return True


def test_expert_cache():
    """Test Expert Cache Manager."""
    print("Testing Expert Cache...")

    from hydranet.cache.expert_cache import PerLayerCache, ExpertCacheManager
    from hydranet.config import MixtralConfig, ExpertCacheConfig

    config = MixtralConfig()
    config.hidden_dim = 256
    config.intermediate_dim = 512
    config.num_experts = 8
    config.num_layers = 4

    cache_config = ExpertCacheConfig(
        pinned_slots=2,
        hot_slots=1,
        probation_slots=1,
    )

    # Test single layer cache
    expert_size = 3 * config.hidden_dim * config.intermediate_dim * 2  # fp16
    layer_cache = PerLayerCache(
        layer_idx=0,
        num_experts=config.num_experts,
        expert_size_bytes=expert_size,
        config=cache_config,
        device=torch.device("cpu"),  # Use CPU for testing
        dtype=torch.float16,
    )

    # Register some experts
    for i in range(config.num_experts):
        weights = {
            "gate_proj": torch.randn(config.intermediate_dim, config.hidden_dim, dtype=torch.float16),
            "up_proj": torch.randn(config.intermediate_dim, config.hidden_dim, dtype=torch.float16),
            "down_proj": torch.randn(config.hidden_dim, config.intermediate_dim, dtype=torch.float16),
        }
        layer_cache.register_expert(i, weights)

    # Test get_weights (should be cache miss then hit)
    w1 = layer_cache.get_weights(0)
    stats1 = layer_cache.get_stats()
    assert stats1["misses"] == 1, f"Expected 1 miss, got {stats1['misses']}"

    w2 = layer_cache.get_weights(0)  # Should be hit now
    stats2 = layer_cache.get_stats()
    assert stats2["hits"] == 1, f"Expected 1 hit, got {stats2['hits']}"

    # Weights should be same
    assert torch.equal(w1["gate_proj"], w2["gate_proj"]), "Cached weights differ!"

    print("  ✓ Expert registration works")
    print("  ✓ Cache hits/misses tracked")
    print("  ✓ Cached weights consistent")
    return True


def test_full_forward_mock():
    """Test full model forward with mock weights."""
    print("Testing Full Forward (mock weights)...")

    from hydranet.model.mixtral import OffloadedMixtral
    from hydranet.config import MixtralConfig, ExpertCacheConfig, KVCacheConfig

    # Tiny config for fast testing
    config = MixtralConfig()
    config.hidden_dim = 128
    config.num_layers = 2
    config.num_attention_heads = 4
    config.num_kv_heads = 2
    config.head_dim = 32
    config.intermediate_dim = 256
    config.num_experts = 4
    config.experts_per_token = 2
    config.vocab_size = 1000

    cache_config = ExpertCacheConfig(
        pinned_slots=4,  # All experts fit
        hot_slots=0,
        probation_slots=0,
    )
    kv_config = KVCacheConfig()

    device = torch.device("cpu")  # Use CPU for testing without GPU

    model = OffloadedMixtral(
        config=config,
        cache_config=cache_config,
        kv_config=kv_config,
        device=device,
        dtype=torch.float32,  # Use float32 on CPU
    )

    # Initialize with random weights
    def init_random_weights():
        # Embeddings
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

    init_random_weights()

    # Register mock experts
    for layer_idx in range(config.num_layers):
        for expert_idx in range(config.num_experts):
            weights = {
                "gate_proj": torch.randn(config.intermediate_dim, config.hidden_dim) * 0.02,
                "up_proj": torch.randn(config.intermediate_dim, config.hidden_dim) * 0.02,
                "down_proj": torch.randn(config.hidden_dim, config.intermediate_dim) * 0.02,
            }
            model.expert_cache.register_expert(layer_idx, expert_idx, weights)

    # Test forward
    batch, seq = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))

    with torch.no_grad():
        logits, kv = model(input_ids, use_cache=False)

    assert logits.shape == (batch, seq, config.vocab_size), \
        f"Logits shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in logits"

    # Test with cache
    with torch.no_grad():
        logits1, kv1 = model(input_ids, use_cache=True)

        # Decode step
        new_ids = torch.randint(0, config.vocab_size, (batch, 1))
        logits2, kv2 = model(new_ids, past_key_values=kv1, use_cache=True)

    assert logits2.shape == (batch, 1, config.vocab_size)

    print("  ✓ Forward pass works")
    print("  ✓ KV cache integration works")
    print("  ✓ No NaN in outputs")
    return True


def test_generate_mock():
    """Test generation with mock weights."""
    print("Testing Generation (mock weights)...")

    from hydranet.model.mixtral import OffloadedMixtral
    from hydranet.config import MixtralConfig, ExpertCacheConfig, KVCacheConfig

    # Tiny config
    config = MixtralConfig()
    config.hidden_dim = 128
    config.num_layers = 2
    config.num_attention_heads = 4
    config.num_kv_heads = 2
    config.head_dim = 32
    config.intermediate_dim = 256
    config.num_experts = 4
    config.experts_per_token = 2
    config.vocab_size = 1000

    cache_config = ExpertCacheConfig(pinned_slots=4, hot_slots=0, probation_slots=0)
    kv_config = KVCacheConfig()

    device = torch.device("cpu")

    model = OffloadedMixtral(
        config=config,
        cache_config=cache_config,
        kv_config=kv_config,
        device=device,
        dtype=torch.float32,
    )

    # Init random weights
    model.embed_tokens.weight.data.normal_(0, 0.02)
    model.lm_head.weight.data.normal_(0, 0.02)
    model.norm.weight.data.fill_(1.0)

    for layer in model.layers:
        layer.self_attn.q_proj.weight.data.normal_(0, 0.02)
        layer.self_attn.k_proj.weight.data.normal_(0, 0.02)
        layer.self_attn.v_proj.weight.data.normal_(0, 0.02)
        layer.self_attn.o_proj.weight.data.normal_(0, 0.02)
        layer.input_layernorm.weight.data.fill_(1.0)
        layer.post_attention_layernorm.weight.data.fill_(1.0)
        layer.moe.router.gate.weight.data.normal_(0, 0.02)

    # Register mock experts
    for layer_idx in range(config.num_layers):
        for expert_idx in range(config.num_experts):
            weights = {
                "gate_proj": torch.randn(config.intermediate_dim, config.hidden_dim) * 0.02,
                "up_proj": torch.randn(config.intermediate_dim, config.hidden_dim) * 0.02,
                "down_proj": torch.randn(config.hidden_dim, config.intermediate_dim) * 0.02,
            }
            model.expert_cache.register_expert(layer_idx, expert_idx, weights)

    # Test generate
    input_ids = torch.randint(0, config.vocab_size, (1, 4))

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=8, temperature=1.0, top_k=10)

    assert output.shape[1] == 4 + 8, f"Generated length: {output.shape[1]} vs expected 12"

    # Check cache stats
    stats = model.get_cache_stats()
    print(f"  Cache hit rate: {stats['hit_rate']:.1%}")
    print(f"  Total hits: {stats['total_hits']}")
    print(f"  Total misses: {stats['total_misses']}")

    print("  ✓ Generation works")
    print("  ✓ Correct output length")
    return True


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("HYDRANET v2.1 UNIT TESTS")
    print("=" * 60)

    tests = [
        ("RoPE", test_rope),
        ("RMSNorm", test_rms_norm),
        ("GQA Attention", test_gqa_attention),
        ("Router", test_router),
        ("Expert MLP", test_expert_mlp),
        ("Expert Cache", test_expert_cache),
        ("Full Forward", test_full_forward_mock),
        ("Generation", test_generate_mock),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
