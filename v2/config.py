"""Configuration classes for HydraNet v2.1."""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
from enum import Enum


class EvictionPolicy(Enum):
    """Expert cache eviction policy."""
    LRU = "lru"
    LFU = "lfu"
    STICKY_LFU = "sticky_lfu"  # Default: probation tier for new entries


class TopKMode(Enum):
    """Router top-k selection mode."""
    PREFILL = "prefill"  # Top-2
    DECODE = "decode"    # Top-1
    FORCED_TOP2 = "forced_top2"  # Always Top-2 (quality over speed)


@dataclass
class MixtralConfig:
    """
    Mixtral-8x7B-Instruct-v0.1 architecture config.

    These values are fixed by the pretrained weights.
    """
    # Model architecture (frozen - matches HuggingFace weights)
    hidden_dim: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    num_kv_heads: int = 8  # GQA: 4 queries per KV head
    head_dim: int = 128
    intermediate_dim: int = 14336  # FFN hidden
    vocab_size: int = 32000

    # MoE config (frozen)
    num_experts: int = 8
    experts_per_token: int = 2  # Mixtral is always top-2

    # RoPE (frozen)
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768

    # Normalization (frozen)
    rms_norm_eps: float = 1e-5

    # Derived values
    @property
    def expert_params(self) -> int:
        """Parameters per expert (gate + up + down projections)."""
        return 3 * self.hidden_dim * self.intermediate_dim

    @property
    def expert_size_bytes(self) -> int:
        """Expert size in bytes (fp16)."""
        return self.expert_params * 2

    @property
    def expert_size_mb(self) -> float:
        """Expert size in MB."""
        return self.expert_size_bytes / (1024 * 1024)

    @property
    def total_experts(self) -> int:
        """Total experts across all layers."""
        return self.num_layers * self.num_experts

    @property
    def total_params_b(self) -> float:
        """Total parameters in billions."""
        # Rough estimate: embeddings + attention + experts
        embed = self.vocab_size * self.hidden_dim
        attn_per_layer = 4 * self.hidden_dim * self.hidden_dim  # Q, K, V, O
        experts_per_layer = self.num_experts * self.expert_params
        router_per_layer = self.hidden_dim * self.num_experts

        total = embed + self.num_layers * (attn_per_layer + experts_per_layer + router_per_layer)
        return total / 1e9


@dataclass
class DraftConfig:
    """
    Mistral-7B-Instruct-v0.2 (draft model) config.

    Must use v0.2 for tokenizer v1 alignment with Mixtral.
    """
    # Model architecture (frozen - matches HuggingFace weights)
    hidden_dim: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_dim: int = 14336
    vocab_size: int = 32000

    # RoPE (frozen)
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768

    # Normalization (frozen)
    rms_norm_eps: float = 1e-5

    # Speculative decoding settings
    draft_length: int = 4  # Tokens to draft before verification

    @property
    def size_gb(self) -> float:
        """Model size in GB (fp16)."""
        params = self.vocab_size * self.hidden_dim  # embed
        params += self.num_layers * 4 * self.hidden_dim * self.hidden_dim  # attn
        params += self.num_layers * 3 * self.hidden_dim * self.intermediate_dim  # ffn
        return params * 2 / 1e9


@dataclass
class ExpertCacheConfig:
    """Per-layer expert cache configuration."""

    # Slot allocation per layer
    pinned_slots: int = 1  # Never evicted, highest-frequency expert
    hot_slots: int = 1     # Sticky, evicted only under pressure
    probation_slots: int = 1  # New entries, promoted on second hit

    # Dynamic reallocation
    enable_dynamic_slots: bool = True
    realloc_interval_tokens: int = 512  # Check every N tokens
    miss_rate_threshold: float = 0.3  # Reallocate if miss rate exceeds this
    max_slots_per_layer: int = 4  # Hard cap even with reallocation

    # Eviction policy
    eviction_policy: EvictionPolicy = EvictionPolicy.STICKY_LFU

    # Cache affinity (router bias toward cached experts)
    enable_affinity_bonus: bool = True
    affinity_bonus: float = 0.1  # Added to routing score
    affinity_entropy_threshold: float = 0.5  # Only apply when router uncertain

    # Prefetch settings
    enable_prefetch: bool = True  # Enable async prefetching

    @property
    def slots_per_layer(self) -> int:
        """Total slots per layer."""
        return self.pinned_slots + self.hot_slots + self.probation_slots


@dataclass
class KVCacheConfig:
    """KV cache paging configuration."""

    # VRAM window
    vram_window_tokens: int = 4096  # Recent tokens in VRAM
    use_int8_kv: bool = True  # INT8 quantization for KV cache

    # RAM paging
    page_size_tokens: int = 256  # Tokens per RAM page
    max_ram_pages: int = 128  # ~32K tokens in RAM

    # Landmark pinning (for tool calls / agentic workloads)
    enable_landmarks: bool = True
    landmark_prefix: str = "<tool_call>"  # Token sequence that triggers pinning
    max_landmarks: int = 8  # Max pinned landmark pages

    @property
    def total_context(self) -> int:
        """Maximum supported context length."""
        return self.vram_window_tokens + self.page_size_tokens * self.max_ram_pages


@dataclass
class RuntimeConfig:
    """Top-level runtime configuration."""

    # Model paths (HuggingFace format or local)
    mixtral_path: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    draft_path: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # Device settings
    device: str = "cuda"
    dtype: str = "float16"  # "float16" or "bfloat16"

    # VRAM budget
    total_vram_gb: float = 16.0
    vram_headroom_gb: float = 1.5  # Reserved for PyTorch overhead, fragmentation

    # Component configs
    expert_cache: ExpertCacheConfig = field(default_factory=ExpertCacheConfig)
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)

    # Top-K switching
    default_topk_mode: TopKMode = TopKMode.DECODE  # Top-1 for decode by default
    allow_runtime_topk_switch: bool = True  # Can switch via API

    # Speculative decoding
    enable_spec_decoding: bool = True
    draft_length: int = 4

    # Prefetch settings
    enable_prefetch: bool = True
    prefetch_both_during_prefill: bool = True  # Prefetch both top-2 candidates
    num_cuda_streams: int = 2

    # Microbatch pipelining
    enable_microbatch_pipeline: bool = True
    microbatch_size: int = 4  # Tokens per microbatch during prefill

    # Resource limits
    reserved_cpu_threads: int = 2
    reserved_ram_gb: float = 6.0

    @property
    def available_vram_gb(self) -> float:
        """VRAM available for model after headroom."""
        return self.total_vram_gb - self.vram_headroom_gb

    @property
    def expert_vram_budget_gb(self) -> float:
        """
        VRAM budget for expert cache.

        Budget breakdown (16GB total, 1.5GB headroom = 14.5GB available):
        - Draft model (resident): ~7GB
        - Attention weights: ~1.5GB
        - KV cache: ~1GB (4K int8)
        - Expert cache: ~4GB (3 slots/layer * 32 layers * ~40MB/expert)
        - Activations: ~0.5GB
        """
        # With draft resident, we have limited expert budget
        return 4.0


@dataclass
class VRAMBudget:
    """Detailed VRAM allocation breakdown."""

    # Fixed allocations
    draft_model_gb: float = 7.0
    attention_weights_gb: float = 1.5
    embeddings_gb: float = 0.25
    activations_gb: float = 0.5

    # Variable allocations
    kv_cache_gb: float = 1.0  # Depends on context length
    expert_cache_gb: float = 4.0  # 3 slots/layer

    # Safety margin
    headroom_gb: float = 1.5

    @property
    def total_gb(self) -> float:
        """Total VRAM usage."""
        return (
            self.draft_model_gb +
            self.attention_weights_gb +
            self.embeddings_gb +
            self.activations_gb +
            self.kv_cache_gb +
            self.expert_cache_gb +
            self.headroom_gb
        )

    def fits_in(self, vram_gb: float) -> bool:
        """Check if budget fits in available VRAM."""
        return self.total_gb <= vram_gb

    def __str__(self) -> str:
        return f"""VRAM Budget:
  Draft model:      {self.draft_model_gb:.1f} GB
  Attention:        {self.attention_weights_gb:.1f} GB
  Embeddings:       {self.embeddings_gb:.2f} GB
  Activations:      {self.activations_gb:.1f} GB
  KV cache:         {self.kv_cache_gb:.1f} GB
  Expert cache:     {self.expert_cache_gb:.1f} GB
  Headroom:         {self.headroom_gb:.1f} GB
  ─────────────────────────
  TOTAL:            {self.total_gb:.1f} GB"""
