#!/usr/bin/env python3
"""
Smoke test for HydraNet Mixtral inference.

Compares HydraNet output against HuggingFace reference to verify correctness.

Tests:
1. HF reference (Transformers) - baseline
2. HydraNet no offload (all experts GPU) - verify model architecture
3. HydraNet with offload, large slots (force hits) - verify caching
4. HydraNet with offload, small slots (force misses) - verify loading

Usage:
    python -m hydranet.v2.tests.smoke_mixtral --model-path /path/to/mixtral

Requirements:
    pip install transformers accelerate
"""

import torch
import argparse
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    tokens: List[int]
    time_ms: float
    error: Optional[str] = None
    stats: Optional[Dict] = None


@dataclass
class ComparisonResult:
    """Result of comparing two outputs."""
    match: bool
    first_mismatch_idx: Optional[int] = None
    reference_token: Optional[int] = None
    test_token: Optional[int] = None


def compare_outputs(
    reference: List[int],
    test: List[int],
    max_check: int = 32,
) -> ComparisonResult:
    """Compare two token sequences."""
    for i in range(min(len(reference), len(test), max_check)):
        if reference[i] != test[i]:
            return ComparisonResult(
                match=False,
                first_mismatch_idx=i,
                reference_token=reference[i],
                test_token=test[i],
            )
    return ComparisonResult(match=True)


class SmokeTest:
    """Smoke test runner for Mixtral."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
    ):
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = torch.float16 if dtype == "float16" else torch.bfloat16

        self.results: List[TestResult] = []

        # Test prompts
        self.prompts = [
            "The capital of France is",
            "def fibonacci(n):",
            "Once upon a time",
        ]

    def run_hf_reference(
        self,
        prompt: str,
        max_tokens: int = 32,
    ) -> TestResult:
        """Run HuggingFace Transformers reference."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print("Loading HF Transformers model...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True,
            )

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate (greedy)
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Greedy
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000

            tokens = outputs[0].tolist()

            # Cleanup
            del model
            torch.cuda.empty_cache()

            return TestResult(
                name="hf_reference",
                passed=True,
                tokens=tokens,
                time_ms=elapsed,
            )

        except Exception as e:
            return TestResult(
                name="hf_reference",
                passed=False,
                tokens=[],
                time_ms=0,
                error=str(e),
            )

    def run_hydranet_no_offload(
        self,
        prompt: str,
        max_tokens: int = 32,
    ) -> TestResult:
        """Run HydraNet with all experts on GPU (no offloading)."""
        try:
            from ..config import (
                MixtralConfig, ExpertCacheConfig, KVCacheConfig, RuntimeConfig
            )
            from ..model.mixtral import OffloadedMixtral
            from ..model.loader import MixtralWeightLoader
            from transformers import AutoTokenizer

            print("Loading HydraNet (no offload)...")

            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            config = MixtralConfig()

            # Large cache config (all experts fit)
            cache_config = ExpertCacheConfig(
                pinned_slots=8,  # All experts
                hot_slots=0,
                probation_slots=0,
            )
            kv_config = KVCacheConfig()

            model = OffloadedMixtral(
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
            weights = loader.load_non_expert_weights()
            model.load_weights(weights)
            del weights

            # Register all experts (they'll all be cached)
            loader.register_all_experts(model.expert_cache)

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)

            # Generate (greedy)
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.0,  # Greedy via 0 temp
                    top_k=1,
                )
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000

            tokens = outputs[0].tolist()
            stats = model.get_cache_stats()

            # Cleanup
            del model
            torch.cuda.empty_cache()

            return TestResult(
                name="hydranet_no_offload",
                passed=True,
                tokens=tokens,
                time_ms=elapsed,
                stats=stats,
            )

        except Exception as e:
            import traceback
            return TestResult(
                name="hydranet_no_offload",
                passed=False,
                tokens=[],
                time_ms=0,
                error=f"{str(e)}\n{traceback.format_exc()}",
            )

    def run_hydranet_force_hits(
        self,
        prompt: str,
        max_tokens: int = 32,
    ) -> TestResult:
        """Run HydraNet with large cache (force cache hits)."""
        try:
            from ..config import (
                MixtralConfig, ExpertCacheConfig, KVCacheConfig
            )
            from ..model.mixtral import OffloadedMixtral
            from ..model.loader import MixtralWeightLoader
            from transformers import AutoTokenizer

            print("Loading HydraNet (force hits)...")

            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            config = MixtralConfig()

            # Medium cache (should hit most)
            cache_config = ExpertCacheConfig(
                pinned_slots=4,
                hot_slots=2,
                probation_slots=2,
            )
            kv_config = KVCacheConfig()

            model = OffloadedMixtral(
                config=config,
                cache_config=cache_config,
                kv_config=kv_config,
                device=self.device,
                dtype=self.dtype,
            )

            loader = MixtralWeightLoader(
                self.model_path, config, self.device, self.dtype
            )
            weights = loader.load_non_expert_weights()
            model.load_weights(weights)
            loader.register_all_experts(model.expert_cache)
            del weights

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)

            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.0,
                    top_k=1,
                )
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000

            tokens = outputs[0].tolist()
            stats = model.get_cache_stats()

            del model
            torch.cuda.empty_cache()

            return TestResult(
                name="hydranet_force_hits",
                passed=True,
                tokens=tokens,
                time_ms=elapsed,
                stats=stats,
            )

        except Exception as e:
            import traceback
            return TestResult(
                name="hydranet_force_hits",
                passed=False,
                tokens=[],
                time_ms=0,
                error=f"{str(e)}\n{traceback.format_exc()}",
            )

    def run_hydranet_force_misses(
        self,
        prompt: str,
        max_tokens: int = 32,
    ) -> TestResult:
        """Run HydraNet with tiny cache (force cache misses)."""
        try:
            from ..config import (
                MixtralConfig, ExpertCacheConfig, KVCacheConfig
            )
            from ..model.mixtral import OffloadedMixtral
            from ..model.loader import MixtralWeightLoader
            from transformers import AutoTokenizer

            print("Loading HydraNet (force misses)...")

            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            config = MixtralConfig()

            # Tiny cache (force misses)
            cache_config = ExpertCacheConfig(
                pinned_slots=1,
                hot_slots=0,
                probation_slots=1,
            )
            kv_config = KVCacheConfig()

            model = OffloadedMixtral(
                config=config,
                cache_config=cache_config,
                kv_config=kv_config,
                device=self.device,
                dtype=self.dtype,
            )

            loader = MixtralWeightLoader(
                self.model_path, config, self.device, self.dtype
            )
            weights = loader.load_non_expert_weights()
            model.load_weights(weights)
            loader.register_all_experts(model.expert_cache)
            del weights

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)

            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.0,
                    top_k=1,
                )
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000

            tokens = outputs[0].tolist()
            stats = model.get_cache_stats()

            del model
            torch.cuda.empty_cache()

            return TestResult(
                name="hydranet_force_misses",
                passed=True,
                tokens=tokens,
                time_ms=elapsed,
                stats=stats,
            )

        except Exception as e:
            import traceback
            return TestResult(
                name="hydranet_force_misses",
                passed=False,
                tokens=[],
                time_ms=0,
                error=f"{str(e)}\n{traceback.format_exc()}",
            )

    def run_all(self, max_tokens: int = 32):
        """Run all tests and compare results."""
        print("=" * 70)
        print("HYDRANET SMOKE TEST")
        print("=" * 70)
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Dtype: {self.dtype}")
        print(f"Max tokens: {max_tokens}")
        print("=" * 70)

        for prompt_idx, prompt in enumerate(self.prompts):
            print(f"\n{'='*70}")
            print(f"PROMPT {prompt_idx + 1}: {prompt[:50]}...")
            print("=" * 70)

            # Run all tests
            results = {
                "reference": self.run_hf_reference(prompt, max_tokens),
                "no_offload": self.run_hydranet_no_offload(prompt, max_tokens),
                "force_hits": self.run_hydranet_force_hits(prompt, max_tokens),
                "force_misses": self.run_hydranet_force_misses(prompt, max_tokens),
            }

            # Print results
            print(f"\n{'Test':<25} {'Status':<10} {'Time (ms)':<12} {'Tokens':<10}")
            print("-" * 60)

            for name, result in results.items():
                status = "PASS" if result.passed else "FAIL"
                print(f"{name:<25} {status:<10} {result.time_ms:>10.1f} "
                      f"{len(result.tokens):<10}")

                if result.error:
                    print(f"  ERROR: {result.error[:100]}")

            # Compare against reference
            if results["reference"].passed:
                print("\nComparison to HF reference:")
                ref_tokens = results["reference"].tokens

                for name in ["no_offload", "force_hits", "force_misses"]:
                    if results[name].passed:
                        cmp = compare_outputs(ref_tokens, results[name].tokens)
                        if cmp.match:
                            print(f"  {name}: MATCH")
                        else:
                            print(f"  {name}: MISMATCH at position {cmp.first_mismatch_idx}")
                            print(f"    Expected: {cmp.reference_token}")
                            print(f"    Got: {cmp.test_token}")
                    else:
                        print(f"  {name}: SKIPPED (test failed)")

            # Print cache stats
            print("\nCache Statistics:")
            for name in ["no_offload", "force_hits", "force_misses"]:
                if results[name].stats:
                    stats = results[name].stats
                    print(f"  {name}:")
                    print(f"    Hit rate: {stats.get('hit_rate', 0):.1%}")
                    print(f"    Hits: {stats.get('total_hits', 0)}")
                    print(f"    Misses: {stats.get('total_misses', 0)}")
                    print(f"    Evictions: {stats.get('total_evictions', 0)}")

            self.results.extend(results.values())

        # Final summary
        self.print_summary()

    def print_summary(self):
        """Print final test summary."""
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print(f"Tests passed: {passed}/{total}")

        if passed < total:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.error[:100] if r.error else 'Unknown error'}")


def main():
    parser = argparse.ArgumentParser(description="HydraNet Mixtral smoke test")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to Mixtral-8x7B checkpoint",
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
        help="Data type",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max tokens to generate",
    )
    args = parser.parse_args()

    test = SmokeTest(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
    )
    test.run_all(max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
