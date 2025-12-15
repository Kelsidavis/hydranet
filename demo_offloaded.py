#!/usr/bin/env python3
"""
Demo: HydraNet v2 with Per-Layer Expert Cache

Key improvements over v1:
- 16 experts/layer (vs 64) for realistic cache hit rates
- Per-layer cache instead of global LRU
- Top-2 prefill / Top-1 decode switching
- Fixed VRAM slots per layer
"""

import os
import torch
import time
import sys
from pathlib import Path

# ========== RESOURCE LIMITS ==========
RESERVED_THREADS = 2
RESERVED_RAM_GB = 6
MAX_CPU_THREADS = max(1, os.cpu_count() - RESERVED_THREADS) if os.cpu_count() else 4

torch.set_num_threads(MAX_CPU_THREADS)
torch.set_num_interop_threads(max(1, MAX_CPU_THREADS // 2))

os.environ["OMP_NUM_THREADS"] = str(MAX_CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MAX_CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_CPU_THREADS)

TOTAL_RAM_GB = 128
AVAILABLE_RAM_GB = TOTAL_RAM_GB - RESERVED_RAM_GB
try:
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('MemTotal:'):
                total_kb = int(line.split()[1])
                TOTAL_RAM_GB = total_kb / (1024 * 1024)
                AVAILABLE_RAM_GB = TOTAL_RAM_GB - RESERVED_RAM_GB
                break
except Exception:
    pass
# =====================================

sys.path.insert(0, str(Path(__file__).parent.parent))

from hydranet.model.config import HydraNetConfig
from hydranet.inference.offloaded_model import (
    OffloadedHydraNet,
    OffloadConfig,
    create_offloaded_hydranet,
)


def demo_v2_inference():
    """Demonstrate HydraNet v2 with per-layer caching."""

    print("=" * 70)
    print("HydraNet v2 - Per-Layer Expert Cache Demo")
    print("=" * 70)

    print(f"\nResource limits:")
    print(f"  CPU threads: {MAX_CPU_THREADS} (reserved {RESERVED_THREADS} for system)")
    print(f"  Available RAM: {AVAILABLE_RAM_GB:.1f} GB (reserved {RESERVED_RAM_GB} GB)")

    device = torch.device("cuda")
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create v2 config: 16 experts/layer instead of 64
    print("\n" + "-" * 50)
    print("Creating HydraNet v2 configuration...")
    print("-" * 50)

    config = HydraNetConfig(
        hidden_dim=2048,
        num_layers=24,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=128,
        num_experts=16,              # Key change: 16 instead of 64
        experts_per_token=2,         # Top-2 for prefill
        decode_experts_per_token=1,  # Top-1 for decode
        expert_intermediate_dim=5120,
        max_context=8192,
        use_shared_expert=True,
        hot_experts_per_layer=4,     # 4 hot + 2 warm per layer
    )

    print(f"Total parameters: {config.total_params_billions:.1f}B")
    print(f"Active parameters: {config.active_params_billions:.1f}B")
    print(f"Expert size: {config.expert_size_mb:.1f} MB each")
    print(f"Total experts: {config.num_layers * config.num_experts}")
    print(f"Experts per layer: {config.num_experts}")
    print(f"Hot experts per layer: {config.hot_experts_per_layer}")

    # Memory estimate
    expert_params = 3 * config.hidden_dim * config.expert_intermediate_dim
    total_experts = config.num_layers * config.num_experts
    expert_ram_gb = total_experts * expert_params * 2 / 1e9  # fp16

    print(f"\nMemory estimate:")
    print(f"  Expert weights (RAM): {expert_ram_gb:.1f} GB")
    print(f"  Available: {AVAILABLE_RAM_GB:.1f} GB")

    if expert_ram_gb > AVAILABLE_RAM_GB * 0.8:
        print(f"  WARNING: Model may use too much RAM")
    else:
        print(f"  OK: Model fits comfortably")

    # Offload config with per-layer cache
    offload_config = OffloadConfig(
        hot_experts_per_layer=4,
        warm_experts_per_layer=2,
        expert_gpu_budget_gb=4.0,
        use_pinned_memory=True,
        enable_prefetch=True,
        prefetch_next_layer=True,
        use_async_loading=True,
        num_load_streams=2,
    )

    print(f"\nPer-layer cache config:")
    print(f"  Hot slots/layer: {offload_config.hot_experts_per_layer}")
    print(f"  Warm slots/layer: {offload_config.warm_experts_per_layer}")
    print(f"  Total VRAM slots: {config.num_layers * (offload_config.hot_experts_per_layer + offload_config.warm_experts_per_layer)}")
    print(f"  GPU expert budget: {offload_config.expert_gpu_budget_gb} GB")

    # Create model
    print("\n" + "-" * 50)
    print("Creating model with per-layer cache...")
    print("-" * 50)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = OffloadedHydraNet(
        config,
        offload_config=offload_config,
        device=device,
        dtype=torch.float16,
    )

    # Initialize expert weights
    print("Initializing expert weights in RAM...")
    model._init_expert_weights()

    # Move non-experts to GPU
    print("Moving attention layers to GPU...")
    model.to_device()

    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory (attention + embeddings): {gpu_mem:.2f} GB")

    # Run inference
    print("\n" + "-" * 50)
    print("Running inference benchmark...")
    print("-" * 50)

    prompt_len = 64
    input_ids = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)
    print(f"Prompt length: {prompt_len} tokens")

    # Warmup
    print("\nWarmup (2 iterations)...")
    with torch.no_grad():
        for _ in range(2):
            _ = model(input_ids)
    torch.cuda.synchronize()

    # Reset stats after warmup
    model.cache_manager.reset_stats()

    # Benchmark prefill (Top-2)
    print("\nPrefill benchmark (Top-2 routing)...")
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        logits, _ = model(input_ids)
    torch.cuda.synchronize()
    prefill_time = (time.time() - start) * 1000
    print(f"Prefill time ({prompt_len} tokens): {prefill_time:.1f} ms")
    print(f"Prefill throughput: {prompt_len / (prefill_time/1000):.0f} tokens/sec")

    # Show prefill cache stats
    prefill_stats = model.get_cache_stats()
    print(f"Prefill cache hit rate: {prefill_stats['hit_rate']:.1%}")

    # Reset for decode benchmark
    model.cache_manager.reset_stats()

    # Benchmark decode (Top-1)
    print("\nDecode benchmark (Top-1 routing, 64 tokens)...")
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=64, temperature=0.8)
    torch.cuda.synchronize()
    gen_time = time.time() - start

    tokens_generated = generated.shape[1] - prompt_len
    tok_per_sec = tokens_generated / gen_time

    print(f"Generated {tokens_generated} tokens in {gen_time:.2f}s")
    print(f"Decode speed: {tok_per_sec:.1f} tokens/second")

    # Cache statistics
    print("\n" + "-" * 50)
    print("Per-layer cache statistics:")
    print("-" * 50)
    stats = model.get_cache_stats()
    print(f"Total cache hits: {stats['total_hits']}")
    print(f"Total cache misses: {stats['total_misses']}")
    print(f"Overall hit rate: {stats['hit_rate']:.1%}")
    print(f"Total evictions: {stats['total_evictions']}")
    print(f"Avg load time: {stats['avg_load_time_ms']:.2f} ms per miss")

    # Per-layer breakdown (first few layers)
    print("\nPer-layer breakdown (first 5 layers):")
    for layer_stat in stats['per_layer_stats'][:5]:
        print(f"  Layer {layer_stat['layer']}: "
              f"hits={layer_stat['hits']}, "
              f"misses={layer_stat['misses']}, "
              f"hit_rate={layer_stat['hit_rate']:.0%}, "
              f"cached={layer_stat['cached_count']}")

    # Memory usage
    print("\n" + "-" * 50)
    print("Memory usage:")
    print("-" * 50)
    print(f"GPU peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"Experts in RAM: {expert_ram_gb:.1f} GB")

    # Update hot experts based on usage
    print("\nUpdating hot experts based on usage patterns...")
    model.update_hot_experts()

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


def compare_v1_v2():
    """Compare v1 (global cache) vs v2 (per-layer cache) conceptually."""

    print("\n" + "=" * 70)
    print("v1 vs v2 Architecture Comparison")
    print("=" * 70)

    print("""
v1 (64 experts/layer, global LRU):
  - 1,792 total experts
  - 80 global hot slots = ~3/layer average
  - Top-2 always = 2 experts needed/layer
  - Result: Constant cache thrashing, PCIe-bound

v2 (16 experts/layer, per-layer cache):
  - 448 total experts
  - 6 hot slots/layer = 37.5% of each layer
  - Top-1 decode = 1 expert needed/layer
  - Result: High hit rate, compute-bound

Expected performance:
  v1: ~5-10 tok/s (PCIe limited)
  v2: ~30-50 tok/s (with good cache hits)
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Show v1 vs v2 comparison")
    args = parser.parse_args()

    if args.compare:
        compare_v1_v2()
    else:
        demo_v2_inference()
