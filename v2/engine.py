"""
HydraNet v2.1 Inference Engine.

Top-level API for Mixtral inference with expert offloading.

Usage:
    engine = InferenceEngine.from_pretrained(
        mixtral_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
        draft_path="mistralai/Mistral-7B-Instruct-v0.2",
    )

    output = engine.generate("What is the meaning of life?", max_tokens=256)
"""

import torch
from typing import Optional, Dict, List
from pathlib import Path

from .config import (
    MixtralConfig,
    DraftConfig,
    RuntimeConfig,
    ExpertCacheConfig,
    KVCacheConfig,
    VRAMBudget,
    TopKMode,
)
from .model.loader import MixtralWeightLoader
from .model.mixtral import OffloadedMixtral
from .cache import ExpertCacheManager, KVPageManager
from .spec.draft import ResidentDraftModel
from .spec.verifier import SpecDecoder


class InferenceEngine:
    """
    Main inference engine for HydraNet v2.1.

    Features:
    - Mixtral-8x7B with expert offloading
    - Speculative decoding with Mistral-7B draft
    - Per-layer expert caching with sticky LFU
    - KV cache paging for long contexts
    - Top-K switching (Top-2 prefill / Top-1 decode)
    """

    def __init__(
        self,
        config: RuntimeConfig,
        verifier: OffloadedMixtral,
        draft: Optional[ResidentDraftModel] = None,
        tokenizer=None,
    ):
        self.config = config
        self.verifier = verifier
        self.draft = draft
        self.tokenizer = tokenizer

        # Speculative decoder (if draft available)
        if draft and config.enable_spec_decoding:
            self.spec_decoder = SpecDecoder(
                verifier=verifier,
                draft=draft,
                draft_length=config.draft_length,
            )
        else:
            self.spec_decoder = None

        # Device
        self.device = torch.device(config.device)

    @classmethod
    def from_pretrained(
        cls,
        mixtral_path: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        draft_path: Optional[str] = "mistralai/Mistral-7B-Instruct-v0.2",
        config: Optional[RuntimeConfig] = None,
        device: str = "cuda",
        dtype: str = "float16",
    ) -> "InferenceEngine":
        """
        Load models from HuggingFace checkpoints.

        Args:
            mixtral_path: Path to Mixtral-8x7B weights
            draft_path: Path to Mistral-7B weights (None to disable spec decoding)
            config: Runtime configuration
            device: Target device
            dtype: Data type ("float16" or "bfloat16")

        Returns:
            Configured InferenceEngine
        """
        if config is None:
            config = RuntimeConfig()

        torch_device = torch.device(device)
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

        # Model configs
        mixtral_config = MixtralConfig()
        draft_config = DraftConfig() if draft_path else None

        # Check VRAM budget
        budget = VRAMBudget()
        print(f"\n{budget}")
        print(f"\nTarget VRAM: {config.total_vram_gb} GB")

        if not budget.fits_in(config.total_vram_gb):
            print(f"WARNING: Budget ({budget.total_gb:.1f} GB) exceeds target VRAM")
            print("Consider disabling speculative decoding or reducing cache size")

        # Load verifier (Mixtral)
        print(f"\n{'='*60}")
        print(f"Loading Mixtral-8x7B from {mixtral_path}")
        print(f"{'='*60}")

        loader = MixtralWeightLoader(
            model_path=mixtral_path,
            config=mixtral_config,
            device=torch_device,
            dtype=torch_dtype,
        )

        # Memory estimates
        mem = loader.estimate_memory()
        print(f"\nMemory estimates:")
        print(f"  Experts (RAM): {mem['experts_ram_gb']:.1f} GB")
        print(f"  Attention (GPU): {mem['attention_gpu_gb']:.1f} GB")
        print(f"  Embeddings (GPU): {mem['embeddings_gpu_gb']:.2f} GB")

        # Create verifier model
        verifier = OffloadedMixtral(
            config=mixtral_config,
            cache_config=config.expert_cache,
            kv_config=config.kv_cache,
            device=torch_device,
            dtype=torch_dtype,
        )

        # Load non-expert weights to GPU
        print("\nLoading attention weights to GPU...")
        non_expert_weights = loader.load_non_expert_weights()
        verifier.load_weights(non_expert_weights)
        del non_expert_weights

        # Register experts in RAM
        print("\nRegistering experts in RAM...")
        loader.register_all_experts(verifier.expert_cache)

        # Load draft model if enabled
        draft = None
        if draft_path and config.enable_spec_decoding:
            print(f"\n{'='*60}")
            print(f"Loading Mistral-7B draft from {draft_path}")
            print(f"{'='*60}")

            draft = ResidentDraftModel.from_pretrained(
                model_path=draft_path,
                device=torch_device,
                dtype=torch_dtype,
            )

        # Load tokenizer
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(mixtral_path)
            print(f"\nTokenizer loaded: {tokenizer.__class__.__name__}")
        except ImportError:
            print("\nWARNING: transformers not installed, tokenizer unavailable")

        print(f"\n{'='*60}")
        print("HydraNet v2.1 ready!")
        print(f"{'='*60}")

        return cls(
            config=config,
            verifier=verifier,
            draft=draft,
            tokenizer=tokenizer,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        use_spec_decoding: Optional[bool] = None,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            use_spec_decoding: Override spec decoding setting

        Returns:
            Generated text
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available. Install transformers.")

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)

        # Generate
        use_spec = use_spec_decoding if use_spec_decoding is not None else (
            self.spec_decoder is not None
        )

        if use_spec and self.spec_decoder:
            output_ids = self.spec_decoder.generate(input_ids, max_new_tokens=max_tokens)
        else:
            output_ids = self.verifier.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return output_text

    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_tokens: int = 256,
        temperature: float = 0.8,
        use_spec_decoding: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens from input IDs.

        Args:
            input_ids: (batch=1, seq) input token IDs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_spec_decoding: Whether to use speculative decoding

        Returns:
            (batch, seq + generated) output token IDs
        """
        input_ids = input_ids.to(self.device)

        if use_spec_decoding and self.spec_decoder:
            return self.spec_decoder.generate(input_ids, max_new_tokens=max_tokens)
        else:
            return self.verifier.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

    def set_topk_mode(self, mode: TopKMode):
        """Set Top-K mode (DECODE=1, PREFILL=2, FORCED_TOP2=2)."""
        self.verifier.set_topk_mode(mode)

    def add_landmark(self, start_pos: int, end_pos: int, landmark_type: str = "tool_call"):
        """Add KV cache landmark for long-context preservation."""
        self.verifier.kv_cache.add_landmark(start_pos, end_pos, landmark_type)

    def clear_caches(self):
        """Clear all caches (expert + KV)."""
        self.verifier.kv_cache.clear()
        self.verifier.reset_cache_stats()
        if self.spec_decoder:
            self.spec_decoder.reset()

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        stats = {
            "expert_cache": self.verifier.get_cache_stats(),
            "kv_cache": self.verifier.kv_cache.get_stats(),
        }

        if self.spec_decoder:
            stats["spec_decoding"] = self.spec_decoder.get_stats()

        return stats

    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()

        print(f"\n{'='*60}")
        print("HydraNet Statistics")
        print(f"{'='*60}")

        # Expert cache
        ec = stats["expert_cache"]
        print(f"\nExpert Cache:")
        print(f"  Overall hit rate: {ec['hit_rate']:.1%}")
        print(f"  Per-layer hit rate: min={ec.get('layer_hit_rate_min', 0):.1%} "
              f"med={ec.get('layer_hit_rate_median', 0):.1%} "
              f"max={ec.get('layer_hit_rate_max', 0):.1%}")
        print(f"  Total hits: {ec['total_hits']}")
        print(f"  Total misses: {ec['total_misses']}")
        print(f"  Evictions: {ec['total_evictions']}")
        print(f"  Promotions: {ec['total_promotions']}")
        print(f"  Avg load time: {ec['avg_load_time_ms']:.2f} ms")
        print(f"  Miss burst events (>{ec.get('miss_burst_threshold', 8)} layers): "
              f"{ec.get('miss_burst_events', 0)}")
        print(f"  Active sequences: {ec.get('active_sequences', 0)}")

        # KV cache
        kv = stats["kv_cache"]
        print(f"\nKV Cache:")
        print(f"  Context length: {kv['total_length']}")
        print(f"  VRAM window: {kv['vram_window']}")
        print(f"  RAM pages: {kv['total_ram_pages']}")
        print(f"  Landmarks: {kv['landmark_count']}")

        # Spec decoding
        if "spec_decoding" in stats:
            sd = stats["spec_decoding"]
            print(f"\nSpeculative Decoding:")
            print(f"  Acceptance rate: {sd['acceptance_rate']:.1%}")
            print(f"  Total drafted: {sd['total_drafted']}")
            print(f"  Total accepted: {sd['total_accepted']}")


def create_engine(
    mixtral_path: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    draft_path: str = "mistralai/Mistral-7B-Instruct-v0.2",
    enable_spec_decoding: bool = True,
    vram_gb: float = 16.0,
) -> InferenceEngine:
    """
    Convenience function to create inference engine.

    Args:
        mixtral_path: Path to Mixtral weights
        draft_path: Path to draft model weights
        enable_spec_decoding: Whether to enable speculative decoding
        vram_gb: Available VRAM in GB

    Returns:
        Configured InferenceEngine
    """
    config = RuntimeConfig(
        mixtral_path=mixtral_path,
        draft_path=draft_path,
        total_vram_gb=vram_gb,
        enable_spec_decoding=enable_spec_decoding,
    )

    return InferenceEngine.from_pretrained(
        mixtral_path=mixtral_path,
        draft_path=draft_path if enable_spec_decoding else None,
        config=config,
    )
