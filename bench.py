#!/usr/bin/env python3
"""
HydraNet v2 Benchmark Suite.

Runs standardized benchmarks and outputs metrics in JSON/CSV format.

Usage:
    python -m hydranet.v2.bench --model-path /path/to/mixtral --output bench_results.json
    python -m hydranet.v2.bench --mock --output mock_results.json  # Mock weights for CI

Benchmarks:
    1. decode_512_b1: Decode 512 tokens, batch=1
    2. decode_512_b4: Decode 512 tokens, batch=4
    3. prefill_4k_decode_256: Prefill 4K, then decode 256, batch=4
    4. long_context_32k: 32K context with paging, decode 256
"""

import torch
import time
import json
import csv
import argparse
import gc
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    tokens_generated: int
    total_time_s: float
    tok_per_s: float

    # Latency percentiles (ms)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Cache stats
    cache_hit_rate: float = 0.0
    total_misses: int = 0
    total_evictions: int = 0
    swaps_per_token: float = 0.0

    # Memory
    peak_vram_gb: float = 0.0
    peak_ram_gb: float = 0.0

    # Transfer stats
    h2d_gb_per_s: float = 0.0
    overlap_ratio: float = 0.0

    # Validation
    passed: bool = True
    error: Optional[str] = None


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    model_path: str
    device: str
    dtype: str
    timestamp: str
    results: List[BenchmarkResult] = field(default_factory=list)

    # Thresholds for pass/fail
    min_tok_per_s: float = 10.0  # Minimum acceptable throughput
    max_p99_latency_ms: float = 500.0  # Maximum acceptable p99

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def summary(self) -> Dict:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        return {
            "passed": passed,
            "total": total,
            "all_passed": passed == total,
            "avg_tok_per_s": sum(r.tok_per_s for r in self.results) / max(1, total),
        }

    def to_dict(self) -> Dict:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "dtype": self.dtype,
            "timestamp": self.timestamp,
            "summary": self.summary(),
            "results": [asdict(r) for r in self.results],
        }


class HydraNetBenchmark:
    """Benchmark runner for HydraNet."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "float16",
        use_mock: bool = False,
        use_int4: bool = False,
        int4_mode: str = "packed",
        packed_dir: Optional[str] = None,
        slot_budget: int = 3,  # Slots per layer for testing eviction
    ):
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.use_mock = use_mock
        self.use_int4 = use_int4
        self.int4_mode = int4_mode
        self.packed_dir = packed_dir
        self.slot_budget = slot_budget

        self.model = None
        self.tokenizer = None
        self.packed_store = None  # For packed INT4 mode

    def setup(self):
        """Load model and tokenizer."""
        from hydranet.config import MixtralConfig, ExpertCacheConfig, KVCacheConfig

        if self.use_mock:
            self._setup_mock()
        else:
            self._setup_real()

    def _setup_mock(self):
        """Setup with mock weights for testing."""
        from hydranet.model.mixtral import OffloadedMixtral
        from hydranet.config import MixtralConfig, ExpertCacheConfig, KVCacheConfig

        print("Setting up mock model...")

        config = MixtralConfig()
        config.hidden_dim = 256
        config.num_layers = 4
        config.num_attention_heads = 8
        config.num_kv_heads = 2
        config.head_dim = 32
        config.intermediate_dim = 512
        config.num_experts = 8
        config.experts_per_token = 2
        config.vocab_size = 1000

        cache_config = ExpertCacheConfig(
            pinned_slots=self.slot_budget,
            hot_slots=1,
            probation_slots=1,
        )
        kv_config = KVCacheConfig()

        # Use CPU for mock to avoid VRAM requirements
        device = torch.device("cpu")

        self.model = OffloadedMixtral(
            config=config,
            cache_config=cache_config,
            kv_config=kv_config,
            device=device,
            dtype=torch.float32,
        )

        # Initialize with random weights
        torch.manual_seed(42)
        self.model.embed_tokens.weight.data.normal_(0, 0.02)
        self.model.lm_head.weight.data.normal_(0, 0.02)
        self.model.norm.weight.data.fill_(1.0)

        for layer in self.model.layers:
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
                torch.manual_seed(42 + layer_idx * 100 + expert_idx)
                weights = {
                    "gate_proj": torch.randn(config.intermediate_dim, config.hidden_dim) * 0.02,
                    "up_proj": torch.randn(config.intermediate_dim, config.hidden_dim) * 0.02,
                    "down_proj": torch.randn(config.hidden_dim, config.intermediate_dim) * 0.02,
                }
                self.model.expert_cache.register_expert(layer_idx, expert_idx, weights)

        self.config = config

        # Mock tokenizer (just random IDs)
        class MockTokenizer:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size

            def encode(self, text, return_tensors=None):
                # Return random tokens based on text length
                length = len(text.split())
                ids = torch.randint(0, self.vocab_size, (length,))
                if return_tensors == "pt":
                    return ids.unsqueeze(0)
                return ids.tolist()

            def decode(self, ids, skip_special_tokens=True):
                return f"<mock output {len(ids)} tokens>"

        self.tokenizer = MockTokenizer(config.vocab_size)
        print("Mock model ready")

    def _setup_real(self):
        """Setup with real Mixtral weights."""
        from hydranet.model.mixtral import OffloadedMixtral
        from hydranet.model.loader import MixtralWeightLoader
        from hydranet.config import MixtralConfig, ExpertCacheConfig, KVCacheConfig
        from transformers import AutoTokenizer

        print(f"Loading model from {self.model_path}...")

        config = MixtralConfig()

        # Configure cache based on mode
        if self.use_int4 and self.int4_mode == "packed":
            # Packed INT4 mode: 2 slots for top-2 routing, no prefetch
            # Memory = base (3.75GB) + INT4 slots (32 layers × 2 slots × ~90MB) + transient fp16
            # Total: ~10 GB, fits in 16GB GPU
            cache_config = ExpertCacheConfig(
                pinned_slots=0,
                hot_slots=1,
                probation_slots=1,  # 2 evictable slots per layer total
                enable_prefetch=False,  # Disable async prefetch for now
            )
        else:
            # Standard mode: more slots for caching
            cache_config = ExpertCacheConfig(
                pinned_slots=self.slot_budget,
                hot_slots=1,
                probation_slots=1,
            )

        kv_config = KVCacheConfig()

        self.model = OffloadedMixtral(
            config=config,
            cache_config=cache_config,
            kv_config=kv_config,
            device=self.device,
            dtype=self.dtype,
        )

        # Load weights
        loader = MixtralWeightLoader(
            self.model_path, config, self.device, self.dtype
        )

        # Load non-expert weights to GPU
        weights = loader.load_non_expert_weights()
        self.model.load_weights(weights)
        del weights
        gc.collect()

        # Register experts (RAM)
        if self.use_int4:
            if self.int4_mode == "packed":
                self._load_int4_experts_packed()
            else:
                self._load_int4_experts_dequant()
        else:
            loader.register_all_experts(self.model.expert_cache)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.config = config

        # Run sanity checks: gate weights, routing correctness
        print("Running weight validation...")
        self.model.validate_weights(verbose=True)

        print("Model loaded")

    def _load_int4_experts_dequant(self):
        """Load INT4 experts with startup dequantization to fp16 (compat/debug mode)."""
        import numpy as np
        from hydranet.cache.packed_expert_store import PackedExpertStore
        from hydranet.preprocess.pack_weights import ExpertWeightPacker, QuantConfig

        packed_dir = Path(self.packed_dir or self.model_path).expanduser()
        idx_path = packed_dir / "experts.idx"
        bin_path = packed_dir / "experts.bin"

        if not idx_path.exists() or not bin_path.exists():
            raise FileNotFoundError(
                f"Packed experts not found. Expected:\n  {idx_path}\n  {bin_path}\n"
                f"Provide --packed-dir pointing to folder with experts.idx/experts.bin."
            )

        print(f"Loading INT4 packed experts from {packed_dir}...")

        # Load the packed store (mmap'd for efficiency)
        store = PackedExpertStore(
            index_path=idx_path,
            bin_path=bin_path,
            device=torch.device("cpu"),  # Load to CPU for dequantization
            use_mmap=True,
        )

        # Get config from store metadata
        group_size = store.quant_config["group_size"]
        hidden_dim = store.model_config["hidden_dim"]
        intermediate_dim = store.model_config["intermediate_dim"]
        num_layers = store.model_config["num_layers"]
        num_experts = store.model_config["num_experts"]

        # Create dequantizer
        packer = ExpertWeightPacker(QuantConfig(group_size=group_size))

        # Load and register each expert
        for layer_idx in range(num_layers):
            for expert_idx in range(num_experts):
                # Get raw blob and layout
                blob_bytes, layout = store.get_expert_blob(layer_idx, expert_idx)
                blob = np.frombuffer(blob_bytes, dtype=np.uint8)

                # Parse blob into packed tensors and scales
                gate_packed = torch.from_numpy(
                    blob[layout.gate_packed_offset:layout.gate_packed_offset + layout.gate_packed_size].copy()
                ).view(intermediate_dim, hidden_dim // 2)

                gate_scales_bytes = blob[layout.gate_scales_offset:layout.gate_scales_offset + layout.gate_scales_size]
                gate_scales = torch.from_numpy(
                    np.frombuffer(gate_scales_bytes.tobytes(), dtype=np.float16).copy()
                ).view(intermediate_dim, hidden_dim // group_size)

                up_packed = torch.from_numpy(
                    blob[layout.up_packed_offset:layout.up_packed_offset + layout.up_packed_size].copy()
                ).view(intermediate_dim, hidden_dim // 2)

                up_scales_bytes = blob[layout.up_scales_offset:layout.up_scales_offset + layout.up_scales_size]
                up_scales = torch.from_numpy(
                    np.frombuffer(up_scales_bytes.tobytes(), dtype=np.float16).copy()
                ).view(intermediate_dim, hidden_dim // group_size)

                down_packed = torch.from_numpy(
                    blob[layout.down_packed_offset:layout.down_packed_offset + layout.down_packed_size].copy()
                ).view(hidden_dim, intermediate_dim // 2)

                down_scales_bytes = blob[layout.down_scales_offset:layout.down_scales_offset + layout.down_scales_size]
                down_scales = torch.from_numpy(
                    np.frombuffer(down_scales_bytes.tobytes(), dtype=np.float16).copy()
                ).view(hidden_dim, intermediate_dim // group_size)

                # Dequantize INT4 to fp16
                gate_proj = packer.dequantize_tensor(gate_packed, gate_scales, intermediate_dim, hidden_dim)
                up_proj = packer.dequantize_tensor(up_packed, up_scales, intermediate_dim, hidden_dim)
                down_proj = packer.dequantize_tensor(down_packed, down_scales, hidden_dim, intermediate_dim)

                # Register with cache manager
                weights = {
                    "gate_proj": gate_proj,
                    "up_proj": up_proj,
                    "down_proj": down_proj,
                }
                self.model.expert_cache.register_expert(layer_idx, expert_idx, weights)

            if (layer_idx + 1) % 4 == 0:
                print(f"  Loaded layers {layer_idx + 1}/{num_layers}")

        store.close()
        print(f"Loaded {num_layers * num_experts} INT4 experts (dequantized to fp16)")
        # In dequant mode, H2D transfers fp16 weights
        fp16_expert_bytes = hidden_dim * intermediate_dim * 3 * 2
        self._print_bandwidth_estimate(mode="dequant", expert_bytes=fp16_expert_bytes, num_layers=num_layers)

    def _load_int4_experts_packed(self):
        """Load INT4 experts in packed mode - keeps INT4 in RAM, dequants on GPU (preferred)."""
        from hydranet.cache.packed_expert_store import PackedExpertStore

        packed_dir = Path(self.packed_dir or self.model_path).expanduser()
        idx_path = packed_dir / "experts.idx"
        bin_path = packed_dir / "experts.bin"

        if not idx_path.exists() or not bin_path.exists():
            raise FileNotFoundError(
                f"Packed experts not found. Expected:\n  {idx_path}\n  {bin_path}\n"
                f"Provide --packed-dir pointing to folder with experts.idx/experts.bin."
            )

        print(f"Loading INT4 packed experts from {packed_dir} (packed mode)...")

        # Load the packed store (mmap'd)
        self.packed_store = PackedExpertStore(
            index_path=idx_path,
            bin_path=bin_path,
            device=self.device,
            use_mmap=True,
        )

        # Attach store to cache manager for on-demand loading
        self.model.expert_cache.set_packed_store(self.packed_store)

        # Get config from store metadata
        num_layers = self.packed_store.model_config["num_layers"]
        num_experts = self.packed_store.model_config["num_experts"]
        hidden_dim = self.packed_store.model_config["hidden_dim"]
        intermediate_dim = self.packed_store.model_config["intermediate_dim"]
        group_size = self.packed_store.quant_config["group_size"]

        # Calculate INT4 expert size (packed + scales)
        # gate: [inter, hidden/2] + scales [inter, hidden/group]
        # up: same as gate
        # down: [hidden, inter/2] + scales [hidden, inter/group]
        int4_expert_bytes = self.packed_store.layout.total_size

        fp16_expert_bytes = hidden_dim * intermediate_dim * 3 * 2  # 3 projections, fp16

        print(f"  Attached PackedExpertStore: {num_layers}×{num_experts} experts")
        print(f"  INT4 expert size: {int4_expert_bytes / (1024*1024):.2f} MB")
        print(f"  fp16 expert size: {fp16_expert_bytes / (1024*1024):.2f} MB")
        print(f"  Compression ratio: {fp16_expert_bytes / int4_expert_bytes:.2f}x")

        self._print_bandwidth_estimate(mode="packed", expert_bytes=int4_expert_bytes, num_layers=num_layers)

    def _print_bandwidth_estimate(self, mode: str, expert_bytes: int, num_layers: int = 32):
        """Print bandwidth reality check for INT4 loading."""
        # Assume PCIe 4.0 x16: ~25 GB/s practical
        pcie_gb_s = 25.0
        pcie_bytes_s = pcie_gb_s * 1e9

        expert_mb = expert_bytes / (1024 * 1024)

        print(f"\n  === Bandwidth Reality Check ({mode} mode) ===")
        print(f"  Bytes per expert blob: {expert_bytes:,} ({expert_mb:.2f} MB)")
        print(f"  Layers: {num_layers}")
        print(f"  PCIe 4.0 practical: {pcie_gb_s:.0f} GB/s")

        # Calculate PCIe ceiling for different hit rates
        # For Top-1 decode: up to num_layers expert loads per token
        print(f"\n  PCIe-limited tok/s ceiling (Top-1 decode, {num_layers} layers):")

        for hit_rate in [0.95, 0.90, 0.80, 0.70]:
            p_miss = 1.0 - hit_rate
            bytes_per_token = expert_bytes * num_layers * p_miss
            mb_per_token = bytes_per_token / (1024 * 1024)
            tok_s_ceiling = pcie_bytes_s / bytes_per_token if bytes_per_token > 0 else float('inf')
            print(f"    {hit_rate:.0%} hit rate → {mb_per_token:.1f} MB/tok → {tok_s_ceiling:.0f} tok/s max")

        print()

    def run_decode_benchmark(
        self,
        name: str,
        num_tokens: int,
        batch_size: int,
        warmup_tokens: int = 32,
    ) -> BenchmarkResult:
        """Run decode-only benchmark."""
        print(f"\nRunning {name}...")

        # Create prompt
        if self.use_mock:
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, 8))
        else:
            prompt = "Once upon a time"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].expand(batch_size, -1)
            input_ids = input_ids.to(self.model.device)

        # Reset stats
        self.model.expert_cache.reset_stats()
        latencies = []

        try:
            self.model.eval()
            past_key_values = None
            generated = input_ids.clone()

            # Warmup
            with torch.no_grad():
                for _ in range(warmup_tokens):
                    curr_input = generated[:, -1:] if past_key_values else generated
                    _, past_key_values = self.model.forward(
                        curr_input, past_key_values=past_key_values, use_cache=True
                    )
                    generated = torch.cat([generated, torch.randint(
                        0, self.config.vocab_size, (batch_size, 1),
                        device=generated.device
                    )], dim=1)

            # Benchmark
            start_total = time.time()

            with torch.no_grad():
                for _ in range(num_tokens):
                    step_start = time.time()

                    curr_input = generated[:, -1:]
                    logits, past_key_values = self.model.forward(
                        curr_input, past_key_values=past_key_values, use_cache=True
                    )

                    # Sample next token (greedy for determinism)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)

                    step_time = time.time() - step_start
                    latencies.append(step_time * 1000)  # ms

            total_time = time.time() - start_total

            # Compute stats
            cache_stats = self.model.expert_cache.get_stats()
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)

            result = BenchmarkResult(
                name=name,
                tokens_generated=num_tokens * batch_size,
                total_time_s=total_time,
                tok_per_s=(num_tokens * batch_size) / total_time,
                p50_latency_ms=latencies_sorted[n // 2],
                p95_latency_ms=latencies_sorted[int(n * 0.95)],
                p99_latency_ms=latencies_sorted[int(n * 0.99)],
                cache_hit_rate=cache_stats["hit_rate"],
                total_misses=cache_stats["total_misses"],
                total_evictions=cache_stats["total_evictions"],
                swaps_per_token=cache_stats["total_evictions"] / max(1, num_tokens),
                passed=True,
            )

        except Exception as e:
            import traceback
            result = BenchmarkResult(
                name=name,
                tokens_generated=0,
                total_time_s=0,
                tok_per_s=0,
                passed=False,
                error=f"{e}\n{traceback.format_exc()}",
            )

        return result

    def run_prefill_decode_benchmark(
        self,
        name: str,
        prefill_tokens: int,
        decode_tokens: int,
        batch_size: int,
    ) -> BenchmarkResult:
        """Run prefill then decode benchmark."""
        print(f"\nRunning {name}...")

        try:
            # Create long prompt for prefill
            if self.use_mock:
                input_ids = torch.randint(
                    0, self.config.vocab_size, (batch_size, prefill_tokens)
                )
            else:
                # Repeat prompt to hit target length
                base_prompt = "This is a test prompt for benchmarking. " * 100
                tokens = self.tokenizer.encode(base_prompt)[:prefill_tokens]
                input_ids = torch.tensor([tokens] * batch_size, device=self.model.device)

            self.model.expert_cache.reset_stats()
            latencies = []

            self.model.eval()

            with torch.no_grad():
                # Prefill phase
                prefill_start = time.time()
                logits, past_key_values = self.model.forward(
                    input_ids, use_cache=True
                )
                prefill_time = time.time() - prefill_start

                generated = input_ids

                # Decode phase
                decode_start = time.time()
                for _ in range(decode_tokens):
                    step_start = time.time()

                    curr_input = generated[:, -1:]
                    logits, past_key_values = self.model.forward(
                        curr_input, past_key_values=past_key_values, use_cache=True
                    )

                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)

                    latencies.append((time.time() - step_start) * 1000)

                decode_time = time.time() - decode_start

            total_time = prefill_time + decode_time

            cache_stats = self.model.expert_cache.get_stats()
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)

            result = BenchmarkResult(
                name=name,
                tokens_generated=decode_tokens * batch_size,
                total_time_s=total_time,
                tok_per_s=(decode_tokens * batch_size) / decode_time,
                p50_latency_ms=latencies_sorted[n // 2] if n > 0 else 0,
                p95_latency_ms=latencies_sorted[int(n * 0.95)] if n > 0 else 0,
                p99_latency_ms=latencies_sorted[int(n * 0.99)] if n > 0 else 0,
                cache_hit_rate=cache_stats["hit_rate"],
                total_misses=cache_stats["total_misses"],
                total_evictions=cache_stats["total_evictions"],
                swaps_per_token=cache_stats["total_evictions"] / max(1, decode_tokens),
                passed=True,
            )

        except Exception as e:
            import traceback
            result = BenchmarkResult(
                name=name,
                tokens_generated=0,
                total_time_s=0,
                tok_per_s=0,
                passed=False,
                error=f"{e}\n{traceback.format_exc()}",
            )

        return result

    def run_all(self) -> BenchmarkSuite:
        """Run full benchmark suite."""
        from datetime import datetime

        suite = BenchmarkSuite(
            model_path=self.model_path or "mock",
            device=str(self.device),
            dtype="float16" if self.dtype == torch.float16 else "bfloat16",
            timestamp=datetime.now().isoformat(),
        )

        # Scale down for mock model
        if self.use_mock:
            decode_tokens = 64
            prefill_tokens = 128
            long_context = 256
        else:
            decode_tokens = 512
            prefill_tokens = 4096
            long_context = 32768

        # Benchmark 1: Decode only, batch=1
        result = self.run_decode_benchmark(
            name="decode_512_b1",
            num_tokens=decode_tokens,
            batch_size=1,
        )
        suite.add_result(result)
        self._print_result(result)

        # Benchmark 2: Decode only, batch=4
        result = self.run_decode_benchmark(
            name="decode_512_b4",
            num_tokens=decode_tokens,
            batch_size=4,
        )
        suite.add_result(result)
        self._print_result(result)

        # Benchmark 3: Prefill + decode
        result = self.run_prefill_decode_benchmark(
            name="prefill_4k_decode_256",
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens // 2,
            batch_size=4,
        )
        suite.add_result(result)
        self._print_result(result)

        # Benchmark 4: Long context (scaled for mock)
        if not self.use_mock:
            result = self.run_prefill_decode_benchmark(
                name="long_context_32k",
                prefill_tokens=long_context,
                decode_tokens=256,
                batch_size=1,  # Single sequence for 32K
            )
            suite.add_result(result)
            self._print_result(result)

        return suite

    def _print_result(self, result: BenchmarkResult):
        """Print result summary."""
        status = "PASS" if result.passed else "FAIL"
        print(f"  {result.name}: {status}")
        if result.passed:
            print(f"    tok/s: {result.tok_per_s:.1f}")
            print(f"    p50/p95/p99: {result.p50_latency_ms:.1f}/{result.p95_latency_ms:.1f}/{result.p99_latency_ms:.1f} ms")
            print(f"    hit rate: {result.cache_hit_rate:.1%}")
            print(f"    swaps/tok: {result.swaps_per_token:.2f}")
        else:
            print(f"    ERROR: {result.error[:100] if result.error else 'Unknown'}")

    def cleanup(self):
        """Cleanup resources."""
        if self.packed_store is not None:
            self.packed_store.close()
            self.packed_store = None
        if self.model is not None:
            self.model.expert_cache.shutdown()
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="HydraNet v2 Benchmark Suite")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to Mixtral checkpoint",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock weights for testing",
    )
    parser.add_argument(
        "--int4",
        action="store_true",
        help="Use INT4 quantized experts",
    )
    parser.add_argument(
        "--int4-mode",
        type=str,
        default="packed",
        choices=["packed", "dequant"],
        help="INT4 loading mode: 'packed' keeps INT4 in RAM and dequants on GPU (preferred), "
             "'dequant' dequants at startup to fp16 (compat/debug)",
    )
    parser.add_argument(
        "--packed-dir",
        type=str,
        help="Path to packed INT4 experts (experts.idx + experts.bin). Defaults to model-path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bench_results.json",
        help="Output file (JSON)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Also output CSV file",
    )
    parser.add_argument(
        "--slot-budget",
        type=int,
        default=3,
        help="Slots per layer (lower forces more eviction)",
    )
    args = parser.parse_args()

    if not args.mock and not args.model_path:
        parser.error("Either --model-path or --mock is required")

    print("=" * 70)
    print("HYDRANET v2 BENCHMARK SUITE")
    print("=" * 70)

    bench = HydraNetBenchmark(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        use_mock=args.mock,
        use_int4=args.int4,
        int4_mode=args.int4_mode,
        packed_dir=args.packed_dir,
        slot_budget=args.slot_budget,
    )

    try:
        bench.setup()
        suite = bench.run_all()

        # Save results
        with open(args.output, "w") as f:
            json.dump(suite.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")

        # Optional CSV output
        if args.csv:
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(suite.results[0]).keys()))
                writer.writeheader()
                for r in suite.results:
                    writer.writerow(asdict(r))
            print(f"CSV saved to {args.csv}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        summary = suite.summary()
        print(f"  Passed: {summary['passed']}/{summary['total']}")
        print(f"  Avg tok/s: {summary['avg_tok_per_s']:.1f}")

        if summary["all_passed"]:
            print("\n✓ ALL BENCHMARKS PASSED")
        else:
            print("\n✗ SOME BENCHMARKS FAILED")
            sys.exit(1)

    finally:
        bench.cleanup()


if __name__ == "__main__":
    main()
