#!/usr/bin/env python3
"""Quick benchmark test with packed INT4."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    import torch
    from hydranet.v2.bench import HydraNetBenchmark

    print("=" * 60)
    print("QUICK BENCHMARK TEST (Packed INT4)")
    print("=" * 60)

    bench = HydraNetBenchmark(
        model_path="/home/k/models/mixtral-8x7b-instruct",
        device="cuda",
        dtype="float16",
        use_mock=False,
        use_int4=True,
        int4_mode="packed",
        packed_dir="/home/k/models/mixtral-8x7b-instruct/packed_int4",
        slot_budget=1,
    )

    try:
        bench.setup()

        # Run small decode benchmark
        result = bench.run_decode_benchmark(
            name="decode_32_b1",
            num_tokens=32,
            batch_size=1,
            warmup_tokens=8,
        )

        print(f"\nResult: {result.name}")
        print(f"  Passed: {result.passed}")
        if result.passed:
            print(f"  tok/s: {result.tok_per_s:.1f}")
            print(f"  p50/p95/p99: {result.p50_latency_ms:.1f}/{result.p95_latency_ms:.1f}/{result.p99_latency_ms:.1f} ms")
            print(f"  hit rate: {result.cache_hit_rate:.1%}")
        else:
            print(f"  Error: {result.error}")

    finally:
        bench.cleanup()


if __name__ == "__main__":
    main()
