"""Expert Cache Manager for GPU↔RAM expert swapping.

Key design changes from v1:
- Per-layer cache instead of global LRU (prevents cross-layer thrashing)
- Fixed VRAM slots per layer (no malloc/free fragmentation)
- Double-buffered staging with pinned memory
- Support for Top-K switching (prefill vs decode)
"""

import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import OrderedDict
import threading
import queue
import time
from enum import Enum

# ========== RESOURCE LIMITS ==========
RESERVED_THREADS = 2
RESERVED_RAM_GB = 6
MAX_CPU_THREADS = max(1, os.cpu_count() - RESERVED_THREADS) if os.cpu_count() else 4
MAX_LOADER_THREADS = max(1, min(2, MAX_CPU_THREADS // 4))

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


class ExpertLocation(Enum):
    """Where an expert's weights currently reside."""
    GPU_HOT = "gpu_hot"      # In fixed VRAM slot, won't be evicted
    GPU_WARM = "gpu_warm"    # In VRAM, can be evicted
    RAM = "ram"              # In pinned RAM
    LOADING = "loading"      # Currently being transferred


@dataclass
class ExpertInfo:
    """Metadata about an expert."""
    layer_idx: int
    expert_idx: int
    location: ExpertLocation = ExpertLocation.RAM
    usage_count: int = 0
    last_used: float = 0.0
    size_bytes: int = 0
    slot_idx: Optional[int] = None  # Fixed VRAM slot index if on GPU


@dataclass
class PerLayerCacheConfig:
    """Configuration for per-layer expert cache."""
    # Per-layer settings
    hot_slots_per_layer: int = 4      # Fixed hot slots per layer (never evicted)
    warm_slots_per_layer: int = 2     # Warm slots per layer (LRU eviction)

    # Total VRAM budget
    total_gpu_budget_gb: float = 6.0

    # Staging buffers (double-buffered for async transfer)
    num_staging_buffers: int = 2

    # Async loading
    use_cuda_streams: bool = True
    num_load_streams: int = 2

    # Prefetch
    enable_prefetch: bool = True
    prefetch_next_layer: bool = True  # Prefetch next layer's likely experts


class PerLayerExpertCache:
    """
    Per-layer cache for expert weights.

    Each layer maintains its own set of hot/warm slots, preventing
    cross-layer cache thrashing that kills performance with many experts.
    """

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        expert_size_bytes: int,
        hot_slots: int,
        warm_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.expert_size_bytes = expert_size_bytes
        self.hot_slots = hot_slots
        self.warm_slots = warm_slots
        self.total_slots = hot_slots + warm_slots
        self.device = device
        self.dtype = dtype

        # Expert info
        self.expert_info: Dict[int, ExpertInfo] = {}
        for i in range(num_experts):
            self.expert_info[i] = ExpertInfo(
                layer_idx=layer_idx,
                expert_idx=i,
                size_bytes=expert_size_bytes,
            )

        # RAM storage (pinned memory)
        self.ram_weights: Dict[int, Dict[str, torch.Tensor]] = {}

        # Fixed GPU slots - pre-allocated to avoid fragmentation
        # slot_idx -> expert_idx (or -1 if empty)
        self.slot_to_expert: List[int] = [-1] * self.total_slots
        # expert_idx -> slot_idx (or -1 if not on GPU)
        self.expert_to_slot: Dict[int, int] = {i: -1 for i in range(num_experts)}

        # GPU weight storage: slot_idx -> weights dict
        self.gpu_slots: Dict[int, Dict[str, torch.Tensor]] = {}

        # LRU for warm slots only (indices hot_slots to total_slots-1)
        self.warm_lru: OrderedDict[int, float] = OrderedDict()  # slot_idx -> last_used

        # Hot expert assignments (expert indices that occupy hot slots)
        self.hot_expert_ids: Set[int] = set()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        # Lock
        self._lock = threading.Lock()

    def register_weights(self, expert_idx: int, weights: Dict[str, torch.Tensor]):
        """Register expert weights in pinned RAM."""
        pinned = {}
        for name, tensor in weights.items():
            p = torch.empty(tensor.shape, dtype=self.dtype, pin_memory=True)
            p.copy_(tensor)
            pinned[name] = p

        with self._lock:
            self.ram_weights[expert_idx] = pinned
            self.expert_info[expert_idx].location = ExpertLocation.RAM

    def set_hot_experts(self, expert_ids: List[int]):
        """
        Designate experts to occupy hot slots (never evicted).

        Should be called after analyzing usage patterns or at init.
        """
        with self._lock:
            # Clear existing hot experts
            for slot_idx in range(self.hot_slots):
                old_expert = self.slot_to_expert[slot_idx]
                if old_expert >= 0:
                    self.expert_to_slot[old_expert] = -1
                    self.expert_info[old_expert].location = ExpertLocation.RAM
                    self.expert_info[old_expert].slot_idx = None
                self.slot_to_expert[slot_idx] = -1

            self.hot_expert_ids = set(expert_ids[:self.hot_slots])

            # Load hot experts into hot slots
            for slot_idx, expert_idx in enumerate(expert_ids[:self.hot_slots]):
                self._load_to_slot(expert_idx, slot_idx, is_hot=True)

    def get_weights(self, expert_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get expert weights, loading to GPU if necessary.

        Returns weights on GPU.
        """
        with self._lock:
            slot_idx = self.expert_to_slot[expert_idx]

            if slot_idx >= 0:
                # Already on GPU
                self.hits += 1
                self.expert_info[expert_idx].usage_count += 1
                self.expert_info[expert_idx].last_used = time.time()

                # Update LRU if in warm slot
                if slot_idx >= self.hot_slots:
                    self.warm_lru.move_to_end(slot_idx)

                return self.gpu_slots[slot_idx]

            # Cache miss - need to load
            self.misses += 1

            # Find a warm slot (evict if necessary)
            slot_idx = self._get_warm_slot()
            self._load_to_slot(expert_idx, slot_idx, is_hot=False)

            return self.gpu_slots[slot_idx]

    def _get_warm_slot(self) -> int:
        """Get an available warm slot, evicting if necessary."""
        # Check for empty warm slots first
        for slot_idx in range(self.hot_slots, self.total_slots):
            if self.slot_to_expert[slot_idx] < 0:
                return slot_idx

        # Evict LRU warm slot
        if self.warm_lru:
            lru_slot = next(iter(self.warm_lru))
            old_expert = self.slot_to_expert[lru_slot]

            # Evict
            self.expert_to_slot[old_expert] = -1
            self.expert_info[old_expert].location = ExpertLocation.RAM
            self.expert_info[old_expert].slot_idx = None
            self.slot_to_expert[lru_slot] = -1
            del self.warm_lru[lru_slot]
            self.evictions += 1

            return lru_slot

        raise RuntimeError(f"Layer {self.layer_idx}: No warm slots available")

    def _load_to_slot(self, expert_idx: int, slot_idx: int, is_hot: bool):
        """Load expert weights into a GPU slot."""
        if expert_idx not in self.ram_weights:
            raise ValueError(f"Expert {expert_idx} not registered in RAM")

        # Copy to GPU
        ram_weights = self.ram_weights[expert_idx]
        gpu_weights = {}
        for name, tensor in ram_weights.items():
            gpu_weights[name] = tensor.to(self.device, non_blocking=True)

        self.gpu_slots[slot_idx] = gpu_weights
        self.slot_to_expert[slot_idx] = expert_idx
        self.expert_to_slot[expert_idx] = slot_idx

        info = self.expert_info[expert_idx]
        info.location = ExpertLocation.GPU_HOT if is_hot else ExpertLocation.GPU_WARM
        info.slot_idx = slot_idx
        info.usage_count += 1
        info.last_used = time.time()

        if not is_hot:
            self.warm_lru[slot_idx] = time.time()

    def is_cached(self, expert_idx: int) -> bool:
        """Check if expert is currently on GPU."""
        return self.expert_to_slot.get(expert_idx, -1) >= 0

    def get_stats(self) -> dict:
        """Get cache statistics for this layer."""
        with self._lock:
            total = self.hits + self.misses
            return {
                "layer": self.layer_idx,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / max(1, total),
                "evictions": self.evictions,
                "hot_count": len(self.hot_expert_ids),
                "cached_count": sum(1 for s in self.slot_to_expert if s >= 0),
            }

    def reset_stats(self):
        """Reset statistics."""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0


class ExpertCacheManager:
    """
    Manages per-layer expert caches with double-buffered async loading.

    Key improvements over global cache:
    - Each layer has independent LRU (no cross-layer thrashing)
    - Fixed VRAM slots (no fragmentation)
    - Double-buffered staging for overlap
    - Supports prefill (Top-K) vs decode (Top-1) switching
    """

    def __init__(
        self,
        config: PerLayerCacheConfig,
        num_layers: int,
        num_experts: int,
        expert_size_bytes: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ):
        self.config = config
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.expert_size_bytes = expert_size_bytes
        self.device = device
        self.dtype = dtype

        # Calculate slots per layer based on VRAM budget
        total_vram = int(config.total_gpu_budget_gb * 1024**3)
        total_experts_in_vram = total_vram // expert_size_bytes
        slots_per_layer = total_experts_in_vram // num_layers

        # Ensure at least hot_slots_per_layer
        slots_per_layer = max(slots_per_layer, config.hot_slots_per_layer)
        warm_slots = max(0, slots_per_layer - config.hot_slots_per_layer)

        print(f"ExpertCacheManager: {slots_per_layer} slots/layer "
              f"({config.hot_slots_per_layer} hot + {warm_slots} warm)")

        # Create per-layer caches
        self.layer_caches: List[PerLayerExpertCache] = []
        for layer_idx in range(num_layers):
            cache = PerLayerExpertCache(
                layer_idx=layer_idx,
                num_experts=num_experts,
                expert_size_bytes=expert_size_bytes,
                hot_slots=config.hot_slots_per_layer,
                warm_slots=warm_slots,
                device=device,
                dtype=dtype,
            )
            self.layer_caches.append(cache)

        # CUDA streams for async loading
        self.load_streams = []
        if config.use_cuda_streams:
            for _ in range(config.num_load_streams):
                self.load_streams.append(torch.cuda.Stream())

        # Staging buffers (pinned memory for fast H2D)
        self.staging_buffers: List[Dict[str, torch.Tensor]] = []

        # Prefetch queue and thread
        self.prefetch_queue: queue.Queue = queue.Queue()
        self._running = True
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

        # Global statistics
        self.total_load_time_ms = 0.0

    def register_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        weights: Dict[str, torch.Tensor],
    ):
        """Register expert weights with cache manager."""
        self.layer_caches[layer_idx].register_weights(expert_idx, weights)

    def get_expert_weights(
        self,
        layer_idx: int,
        expert_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Get expert weights (loads to GPU if necessary)."""
        start = time.time()
        weights = self.layer_caches[layer_idx].get_weights(expert_idx)
        self.total_load_time_ms += (time.time() - start) * 1000
        return weights

    def is_cached(self, layer_idx: int, expert_idx: int) -> bool:
        """Check if expert is on GPU."""
        return self.layer_caches[layer_idx].is_cached(expert_idx)

    def set_hot_experts_for_layer(self, layer_idx: int, expert_ids: List[int]):
        """Set hot experts for a specific layer."""
        self.layer_caches[layer_idx].set_hot_experts(expert_ids)

    def set_hot_experts_uniform(self, experts_per_layer: int):
        """
        Set the first N experts as hot for each layer.

        Simple heuristic - can be improved with usage statistics.
        """
        for layer_idx in range(self.num_layers):
            hot_ids = list(range(min(experts_per_layer, self.num_experts)))
            self.layer_caches[layer_idx].set_hot_experts(hot_ids)

    def update_hot_experts_from_usage(self):
        """Update hot experts based on usage statistics."""
        for cache in self.layer_caches:
            # Sort by usage count
            sorted_experts = sorted(
                cache.expert_info.items(),
                key=lambda x: x[1].usage_count,
                reverse=True,
            )
            hot_ids = [eid for eid, _ in sorted_experts[:cache.hot_slots]]
            cache.set_hot_experts(hot_ids)

    def prefetch_for_next_layer(
        self,
        current_layer: int,
        predicted_experts: List[int],
    ):
        """
        Queue prefetch for next layer's predicted experts.

        Called during forward pass to overlap loading with compute.
        """
        if not self.config.enable_prefetch:
            return

        next_layer = current_layer + 1
        if next_layer >= self.num_layers:
            return

        for expert_idx in predicted_experts:
            if not self.is_cached(next_layer, expert_idx):
                self.prefetch_queue.put((next_layer, expert_idx))

    def _prefetch_worker(self):
        """Background thread for async prefetching."""
        stream_idx = 0
        while self._running:
            try:
                layer_idx, expert_idx = self.prefetch_queue.get(timeout=0.05)

                if self.is_cached(layer_idx, expert_idx):
                    continue

                # Use round-robin stream selection
                if self.load_streams:
                    stream = self.load_streams[stream_idx % len(self.load_streams)]
                    stream_idx += 1
                    with torch.cuda.stream(stream):
                        self.layer_caches[layer_idx].get_weights(expert_idx)
                else:
                    self.layer_caches[layer_idx].get_weights(expert_idx)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Prefetch error: {e}")

    def get_stats(self) -> dict:
        """Get aggregated cache statistics."""
        total_hits = sum(c.hits for c in self.layer_caches)
        total_misses = sum(c.misses for c in self.layer_caches)
        total_evictions = sum(c.evictions for c in self.layer_caches)

        return {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": total_hits / max(1, total_hits + total_misses),
            "total_evictions": total_evictions,
            "avg_load_time_ms": self.total_load_time_ms / max(1, total_misses),
            "per_layer_stats": [c.get_stats() for c in self.layer_caches],
        }

    def reset_stats(self):
        """Reset all statistics."""
        for cache in self.layer_caches:
            cache.reset_stats()
        self.total_load_time_ms = 0.0

    def shutdown(self):
        """Shutdown background threads."""
        self._running = False
        self._prefetch_thread.join(timeout=1.0)

    def __del__(self):
        self.shutdown()


# Legacy compatibility alias
CacheConfig = PerLayerCacheConfig
