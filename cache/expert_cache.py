"""
Per-layer expert cache with fixed VRAM slots.

Key design:
- Each layer has independent cache (prevents cross-layer thrashing)
- Fixed slots: pinned (never evict) + hot (sticky) + probation (new entries)
- Sticky LFU: new experts enter probation, promoted on second hit
- Async H2D via CUDA streams
"""

import torch
import torch.cuda
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, NamedTuple
from collections import OrderedDict
from enum import Enum
import threading
import queue
import time

from ..config import ExpertCacheConfig, MixtralConfig


class SlotTier(Enum):
    """Which tier a slot belongs to."""
    PINNED = "pinned"      # Never evicted
    HOT = "hot"            # Sticky, evicted under pressure
    PROBATION = "probation"  # New entries, promoted on second hit


@dataclass
class SlotInfo:
    """Metadata about a VRAM slot."""
    slot_idx: int
    tier: SlotTier
    expert_id: Optional[Tuple[int, int]] = None  # (layer_idx, expert_idx) or None
    hit_count: int = 0
    last_access: float = 0.0
    pinned_until: Optional[float] = None  # For spec decoding pin protection


@dataclass
class ExpertMetadata:
    """Tracking info for each expert across all layers."""
    layer_idx: int
    expert_idx: int
    frequency: int = 0  # Total accesses
    last_access: float = 0.0
    in_vram: bool = False
    slot_idx: Optional[int] = None


class PerLayerCache:
    """
    Expert cache for a single transformer layer.

    Slot layout:
    - slots[0:pinned_slots]: Pinned tier (highest frequency, never evicted)
    - slots[pinned_slots:pinned_slots+hot_slots]: Hot tier (sticky LFU)
    - slots[...:total_slots]: Probation tier (new entries)

    Promotion policy (Sticky LFU):
    - New experts enter probation
    - On second hit in probation -> promote to hot
    - Hot experts only evicted when all probation slots full
    """

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        expert_size_bytes: int,
        config: ExpertCacheConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.expert_size_bytes = expert_size_bytes
        self.config = config
        self.device = device
        self.dtype = dtype

        # Slot boundaries
        self.pinned_slots = config.pinned_slots
        self.hot_slots = config.hot_slots
        self.probation_slots = config.probation_slots
        self.total_slots = config.slots_per_layer

        # Slot metadata
        self.slots: List[SlotInfo] = []
        for i in range(self.pinned_slots):
            self.slots.append(SlotInfo(slot_idx=i, tier=SlotTier.PINNED))
        for i in range(self.pinned_slots, self.pinned_slots + self.hot_slots):
            self.slots.append(SlotInfo(slot_idx=i, tier=SlotTier.HOT))
        for i in range(self.pinned_slots + self.hot_slots, self.total_slots):
            self.slots.append(SlotInfo(slot_idx=i, tier=SlotTier.PROBATION))

        # expert_idx -> SlotInfo (if cached)
        self.expert_to_slot: Dict[int, SlotInfo] = {}

        # GPU tensor storage: slot_idx -> {gate_proj, up_proj, down_proj}
        self.gpu_weights: Dict[int, Dict[str, torch.Tensor]] = {}

        # RAM storage (pinned memory): expert_idx -> {gate_proj, up_proj, down_proj}
        self.ram_weights: Dict[int, Dict[str, torch.Tensor]] = {}

        # Statistics
        self.hits = 0
        self.misses = 0
        self.promotions = 0
        self.evictions = 0

        # Adaptive pinning: track expert frequency and pin top-K
        self.expert_freq: Dict[int, float] = {}  # expert_idx -> decayed frequency
        self.adaptive_pins: Set[int] = set()  # currently pinned expert IDs
        self.freq_decay = 0.995  # decay per access (slower decay for longer memory)
        self.pin_count = 1  # how many experts to pin per layer

        # Thread safety
        self._lock = threading.RLock()

        # Packed INT4 mode (optional)
        self.packed_store = None
        self.use_packed_mode = False
        self.gpu_int4_slots: Dict[int, "GpuExpertSlot"] = {}  # slot_idx -> GpuExpertSlot
        self._packed_hidden_dim = 0
        self._packed_intermediate_dim = 0
        self._packed_group_size = 0

    def register_expert(self, expert_idx: int, weights: Dict[str, torch.Tensor]):
        """
        Register expert weights in pinned RAM.

        Args:
            expert_idx: Expert index within this layer (0 to num_experts-1)
            weights: Dict with 'gate_proj', 'up_proj', 'down_proj' tensors
        """
        pinned = {}
        # Only use pinned memory if CUDA is available
        use_pinned = torch.cuda.is_available() and self.device.type == "cuda"

        for name, tensor in weights.items():
            if use_pinned:
                # Allocate pinned memory for fast H2D
                p = torch.empty(tensor.shape, dtype=self.dtype, pin_memory=True)
                p.copy_(tensor)
            else:
                # CPU mode - just copy tensor
                p = tensor.to(dtype=self.dtype).clone()
            pinned[name] = p

        with self._lock:
            self.ram_weights[expert_idx] = pinned

    def get_weights(self, expert_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get expert weights on GPU, loading if necessary.

        Returns:
            Dict with 'gate_proj', 'up_proj', 'down_proj' on GPU
        """
        import os
        _debug = os.environ.get("DEBUG_CACHE", "0") == "1"

        with self._lock:
            # Check if already cached
            if expert_idx in self.expert_to_slot:
                slot = self.expert_to_slot[expert_idx]
                if _debug and self.layer_idx == 0:
                    print(f"      [L0] HIT expert {expert_idx} in slot {slot.slot_idx}")
                self._record_hit(slot)
                # Re-lookup slot after record_hit - promotion may have moved it
                slot = self.expert_to_slot[expert_idx]

                # Check if fp16 weights exist (may have been cleared)
                if slot.slot_idx in self.gpu_weights:
                    return self.gpu_weights[slot.slot_idx]

                # fp16 was cleared, re-dequantize from INT4
                if self.use_packed_mode and slot.slot_idx in self.gpu_int4_slots:
                    gpu_slot = self.gpu_int4_slots[slot.slot_idx]
                    gpu_weights = gpu_slot.dequantize()
                    self.gpu_weights[slot.slot_idx] = gpu_weights
                    return gpu_weights

                # Fallback: reload from RAM (shouldn't happen in packed mode)
                self._load_to_slot(expert_idx, slot)
                return self.gpu_weights[slot.slot_idx]

            # Cache miss
            self.misses += 1
            self._record_expert_access(expert_idx)  # Track frequency for pinning

            if _debug and self.layer_idx == 0:
                print(f"      [L0] MISS expert {expert_idx}, cached: {list(self.expert_to_slot.keys())}")

            # Find a slot (may evict)
            slot = self._allocate_slot(expert_idx)

            # Load from RAM to GPU
            self._load_to_slot(expert_idx, slot)

            return self.gpu_weights[slot.slot_idx]

    def _record_hit(self, slot: SlotInfo):
        """Record a cache hit and handle promotion."""
        self.hits += 1
        slot.hit_count += 1
        slot.last_access = time.time()

        # Track frequency for adaptive pinning
        if slot.expert_id is not None:
            _, expert_idx = slot.expert_id
            self._record_expert_access(expert_idx)

        # Sticky LFU: promote from probation to hot on second hit
        if slot.tier == SlotTier.PROBATION and slot.hit_count >= 2:
            self._try_promote(slot)

    def _record_expert_access(self, expert_idx: int):
        """Track expert access frequency with decay."""
        # Decay all frequencies slightly
        for eid in self.expert_freq:
            self.expert_freq[eid] *= self.freq_decay
        # Increment this expert
        self.expert_freq[expert_idx] = self.expert_freq.get(expert_idx, 0.0) + 1.0

    def refresh_adaptive_pins(self, debug: bool = False):
        """
        Update adaptive pins based on recent frequency.

        Pins the top-K most frequent experts (if they're currently cached).
        Called periodically by the cache manager.
        """
        if not self.expert_freq or self.pin_count == 0:
            return

        # Find top-K experts by frequency
        sorted_experts = sorted(self.expert_freq.items(), key=lambda x: -x[1])
        top_experts = [eid for eid, _ in sorted_experts[:self.pin_count]]

        # Only pin if currently cached (don't force load)
        new_pins = set()
        for eid in top_experts:
            if eid in self.expert_to_slot:
                new_pins.add(eid)

        if debug and self.layer_idx == 0:
            print(f"  [L0 pins] top freq: {[(e, f'{f:.1f}') for e, f in sorted_experts[:4]]}, "
                  f"cached: {list(self.expert_to_slot.keys())}, pins: {new_pins}")

        self.adaptive_pins = new_pins

    def _try_promote(self, slot: SlotInfo):
        """Try to promote expert from probation to hot tier."""
        # Find empty hot slot
        for hot_slot in self.slots[self.pinned_slots:self.pinned_slots + self.hot_slots]:
            if hot_slot.expert_id is None:
                self._swap_slots(slot, hot_slot)
                self.promotions += 1
                return

        # All hot slots full - find LFU hot slot
        lfu_hot = min(
            (s for s in self.slots[self.pinned_slots:self.pinned_slots + self.hot_slots]
             if s.expert_id is not None and s.pinned_until is None),
            key=lambda s: (s.hit_count, s.last_access),
            default=None
        )

        # Only promote if we have more hits than LFU hot
        if lfu_hot and slot.hit_count > lfu_hot.hit_count:
            self._swap_slots(slot, lfu_hot)
            self.promotions += 1

    def _swap_slots(self, src: SlotInfo, dst: SlotInfo):
        """Swap contents of two slots (for promotion/demotion)."""
        # Swap GPU weights (fp16 dequantized)
        src_weights = self.gpu_weights.get(src.slot_idx)
        dst_weights = self.gpu_weights.get(dst.slot_idx)

        if src_weights:
            self.gpu_weights[dst.slot_idx] = src_weights
        elif dst.slot_idx in self.gpu_weights:
            del self.gpu_weights[dst.slot_idx]

        if dst_weights:
            self.gpu_weights[src.slot_idx] = dst_weights
        elif src.slot_idx in self.gpu_weights:
            del self.gpu_weights[src.slot_idx]

        # Swap GPU INT4 slots
        src_int4 = self.gpu_int4_slots.get(src.slot_idx)
        dst_int4 = self.gpu_int4_slots.get(dst.slot_idx)

        if src_int4:
            self.gpu_int4_slots[dst.slot_idx] = src_int4
        elif dst.slot_idx in self.gpu_int4_slots:
            del self.gpu_int4_slots[dst.slot_idx]

        if dst_int4:
            self.gpu_int4_slots[src.slot_idx] = dst_int4
        elif src.slot_idx in self.gpu_int4_slots:
            del self.gpu_int4_slots[src.slot_idx]

        # Update expert mappings
        src_expert = src.expert_id
        dst_expert = dst.expert_id

        if src_expert:
            src_layer, src_idx = src_expert
            self.expert_to_slot[src_idx] = dst
        if dst_expert:
            dst_layer, dst_idx = dst_expert
            self.expert_to_slot[dst_idx] = src

        # Swap metadata (but keep tier)
        src_tier, dst_tier = src.tier, dst.tier
        src.expert_id, dst.expert_id = dst.expert_id, src.expert_id
        src.hit_count, dst.hit_count = dst.hit_count, src.hit_count
        src.last_access, dst.last_access = dst.last_access, src.last_access
        src.tier, dst.tier = src_tier, dst_tier

    def _allocate_slot(self, expert_idx: int) -> SlotInfo:
        """
        Find a slot for new expert, evicting if necessary.

        Allocation priority (fill ALL empty slots before evicting):
        1. Empty probation slot
        2. Empty hot slot
        3. Empty pinned slot
        4. LFU probation slot (evict)
        5. LFU hot slot (evict)
        """
        # 1. Empty probation slot
        for slot in self.slots[self.pinned_slots + self.hot_slots:]:
            if slot.expert_id is None:
                return slot

        # 2. Empty hot slot (check before evicting!)
        for slot in self.slots[self.pinned_slots:self.pinned_slots + self.hot_slots]:
            if slot.expert_id is None:
                return slot

        # 3. Empty pinned slot
        for slot in self.slots[:self.pinned_slots]:
            if slot.expert_id is None:
                return slot

        # Helper: check if slot holds an adaptively pinned expert
        def is_evictable(s: SlotInfo) -> bool:
            if s.pinned_until is not None and s.pinned_until >= time.time():
                return False
            if s.expert_id is not None:
                _, eid = s.expert_id
                if eid in self.adaptive_pins:
                    return False
            return True

        # 4. LFU probation slot (evict) - skip adaptive pins
        lfu_probation = min(
            (s for s in self.slots[self.pinned_slots + self.hot_slots:]
             if is_evictable(s)),
            key=lambda s: (s.hit_count, s.last_access),
            default=None
        )
        if lfu_probation:
            self._evict(lfu_probation)
            return lfu_probation

        # 5. LFU hot slot (evict) - skip adaptive pins
        lfu_hot = min(
            (s for s in self.slots[self.pinned_slots:self.pinned_slots + self.hot_slots]
             if is_evictable(s)),
            key=lambda s: (s.hit_count, s.last_access),
            default=None
        )
        if lfu_hot:
            self._evict(lfu_hot)
            return lfu_hot

        raise RuntimeError(f"Layer {self.layer_idx}: No evictable slots available")

    def _evict(self, slot: SlotInfo):
        """Evict expert from slot."""
        import os
        _debug = os.environ.get("DEBUG_CACHE", "0") == "1"

        if slot.expert_id is None:
            return

        _, expert_idx = slot.expert_id
        if _debug and self.layer_idx == 0:
            print(f"      [L0] EVICT expert {expert_idx} from slot {slot.slot_idx}")
        del self.expert_to_slot[expert_idx]

        # Free fp16 weights immediately to prevent memory accumulation
        if slot.slot_idx in self.gpu_weights:
            del self.gpu_weights[slot.slot_idx]

        # Clear cached fp16 in GPU INT4 slot if using packed mode
        if self.use_packed_mode and slot.slot_idx in self.gpu_int4_slots:
            self.gpu_int4_slots[slot.slot_idx]._cached_fp16 = None

        # Keep GPU INT4 memory allocated (reuse buffer)
        slot.expert_id = None
        slot.hit_count = 0
        slot.last_access = 0.0
        self.evictions += 1

    def add_slot(self, tier: SlotTier = SlotTier.PROBATION) -> bool:
        """
        Add a new slot to this layer's cache.

        Returns True if slot was added, False if at max capacity.
        """
        with self._lock:
            if self.total_slots >= self.config.max_slots_per_layer:
                return False

            new_idx = self.total_slots
            new_slot = SlotInfo(slot_idx=new_idx, tier=tier)
            self.slots.append(new_slot)
            self.total_slots += 1

            # Update tier counts
            if tier == SlotTier.PINNED:
                self.pinned_slots += 1
            elif tier == SlotTier.HOT:
                self.hot_slots += 1
            else:
                self.probation_slots += 1

            return True

    def remove_slot(self) -> bool:
        """
        Remove a slot from this layer's cache.

        Removes the least valuable slot (empty > probation LFU > hot LFU).
        Returns True if slot was removed, False if at minimum (1 slot).
        """
        with self._lock:
            if self.total_slots <= 1:
                return False

            # Find slot to remove (prefer empty, then LFU)
            target_slot = None

            # 1. Empty probation slot
            for slot in reversed(self.slots[self.pinned_slots + self.hot_slots:]):
                if slot.expert_id is None:
                    target_slot = slot
                    break

            # 2. Empty hot slot
            if target_slot is None:
                for slot in reversed(self.slots[self.pinned_slots:self.pinned_slots + self.hot_slots]):
                    if slot.expert_id is None:
                        target_slot = slot
                        break

            # 3. LFU probation slot
            if target_slot is None and self.probation_slots > 0:
                probation_slots = [s for s in self.slots[self.pinned_slots + self.hot_slots:]
                                   if s.expert_id is not None]
                if probation_slots:
                    target_slot = min(probation_slots, key=lambda s: (s.hit_count, s.last_access))

            # 4. LFU hot slot (last resort)
            if target_slot is None and self.hot_slots > 0:
                hot_slots = [s for s in self.slots[self.pinned_slots:self.pinned_slots + self.hot_slots]
                             if s.expert_id is not None]
                if hot_slots:
                    target_slot = min(hot_slots, key=lambda s: (s.hit_count, s.last_access))

            if target_slot is None:
                return False

            # Evict if occupied
            if target_slot.expert_id is not None:
                self._evict(target_slot)

            # Remove slot
            tier = target_slot.tier
            slot_idx = target_slot.slot_idx
            self.slots.remove(target_slot)
            self.total_slots -= 1

            # Update tier counts
            if tier == SlotTier.PINNED:
                self.pinned_slots -= 1
            elif tier == SlotTier.HOT:
                self.hot_slots -= 1
            else:
                self.probation_slots -= 1

            # Reindex remaining slots
            for i, slot in enumerate(self.slots):
                if slot.slot_idx != i:
                    old_idx = slot.slot_idx
                    slot.slot_idx = i
                    # Update GPU weight references
                    if old_idx in self.gpu_weights:
                        self.gpu_weights[i] = self.gpu_weights.pop(old_idx)
                    if old_idx in self.gpu_int4_slots:
                        self.gpu_int4_slots[i] = self.gpu_int4_slots.pop(old_idx)
                        self.gpu_int4_slots[i].slot_idx = i

            # Clean up orphaned slot data
            for idx in list(self.gpu_weights.keys()):
                if idx >= self.total_slots:
                    del self.gpu_weights[idx]
            for idx in list(self.gpu_int4_slots.keys()):
                if idx >= self.total_slots:
                    del self.gpu_int4_slots[idx]

            return True

    def get_slot_count(self) -> int:
        """Get current number of slots."""
        return self.total_slots

    def _load_to_slot(self, expert_idx: int, slot: SlotInfo):
        """Load expert from pinned RAM (or packed store) to GPU slot."""
        if self.use_packed_mode:
            self._load_to_slot_packed(expert_idx, slot)
        else:
            self._load_to_slot_fp16(expert_idx, slot)

    def _load_to_slot_fp16(self, expert_idx: int, slot: SlotInfo):
        """Load fp16 expert from pinned RAM to GPU slot."""
        if expert_idx not in self.ram_weights:
            raise ValueError(f"Expert {expert_idx} not registered in RAM")

        ram_weights = self.ram_weights[expert_idx]

        # Async copy to GPU (caller should sync if needed)
        gpu_weights = {}
        for name, tensor in ram_weights.items():
            gpu_weights[name] = tensor.to(self.device, non_blocking=True)

        self.gpu_weights[slot.slot_idx] = gpu_weights
        slot.expert_id = (self.layer_idx, expert_idx)
        slot.hit_count = 1
        slot.last_access = time.time()
        self.expert_to_slot[expert_idx] = slot

    def _load_to_slot_packed(self, expert_idx: int, slot: SlotInfo):
        """Load INT4 expert from packed store, dequantize on GPU."""
        from .packed_expert_store import GpuExpertSlot

        # Get or create GPU INT4 slot
        if slot.slot_idx not in self.gpu_int4_slots:
            self.gpu_int4_slots[slot.slot_idx] = GpuExpertSlot(
                slot_idx=slot.slot_idx,
                hidden_dim=self._packed_hidden_dim,
                intermediate_dim=self._packed_intermediate_dim,
                group_size=self._packed_group_size,
                device=self.device,
            )

        gpu_slot = self.gpu_int4_slots[slot.slot_idx]

        # Load from packed store to GPU slot (via pinned staging)
        self.packed_store.load_to_gpu_slot(
            layer_idx=self.layer_idx,
            expert_idx=expert_idx,
            gpu_slot=gpu_slot,
            stream=None,  # Sync for now, could use stream
        )

        # Dequantize and store as fp16 weights
        # (This is the on-GPU dequant path - no RAM fp16 intermediate)
        gpu_weights = gpu_slot.dequantize()
        self.gpu_weights[slot.slot_idx] = gpu_weights

        slot.expert_id = (self.layer_idx, expert_idx)
        slot.hit_count = 1
        slot.last_access = time.time()
        self.expert_to_slot[expert_idx] = slot

    def pin_expert(self, expert_idx: int, duration_sec: float = 1.0):
        """
        Temporarily pin expert to prevent eviction (for spec decoding).

        Args:
            expert_idx: Expert to pin
            duration_sec: How long to pin (seconds)
        """
        with self._lock:
            if expert_idx in self.expert_to_slot:
                slot = self.expert_to_slot[expert_idx]
                slot.pinned_until = time.time() + duration_sec

    def is_cached(self, expert_idx: int) -> bool:
        """Check if expert is in VRAM."""
        return expert_idx in self.expert_to_slot

    def prefetch_int4_slot(self, expert_idx: int):
        """
        Schedule async load for a missing expert. May evict if needed.

        Does NOT record hit/miss stats - those are tracked in get_int4_slot.
        Returns the GpuExpertSlot (may still be loading).
        """
        with self._lock:
            # If already cached, just return it
            if expert_idx in self.expert_to_slot:
                slot = self.expert_to_slot[expert_idx]
                return self.gpu_int4_slots[slot.slot_idx]

            # Cache miss - allocate slot (may evict) and schedule async load
            slot = self._allocate_slot(expert_idx)
            self._load_int4_to_slot(expert_idx, slot)

            return self.gpu_int4_slots[slot.slot_idx]

    def get_int4_slot(self, expert_idx: int):
        """
        Get INT4 slot ready for compute.

        Always records hit/miss stats (the single source of truth).
        If prefetch was called first, waits for completion.
        """
        with self._lock:
            # Check if already cached (or prefetched)
            if expert_idx in self.expert_to_slot:
                slot = self.expert_to_slot[expert_idx]
                gpu_slot = self.gpu_int4_slots[slot.slot_idx]

                # Record hit
                self._record_hit(slot)

                gpu_slot.wait_ready()  # GPU-side wait if still loading
                return gpu_slot

            # Cache miss - load now
            self.misses += 1
            self._record_expert_access(expert_idx)

            slot = self._allocate_slot(expert_idx)
            self._load_int4_to_slot(expert_idx, slot)

            gpu_slot = self.gpu_int4_slots[slot.slot_idx]
            gpu_slot.wait_ready()
            return gpu_slot

    def _load_int4_to_slot(self, expert_idx: int, slot):
        """Load INT4 expert from packed store (async H2D on memcpy stream)."""
        from .packed_expert_store import GpuExpertSlot

        # Get or create GPU INT4 slot
        if slot.slot_idx not in self.gpu_int4_slots:
            self.gpu_int4_slots[slot.slot_idx] = GpuExpertSlot(
                slot_idx=slot.slot_idx,
                hidden_dim=self._packed_hidden_dim,
                intermediate_dim=self._packed_intermediate_dim,
                group_size=self._packed_group_size,
                device=self.device,
            )

        gpu_slot = self.gpu_int4_slots[slot.slot_idx]

        # Load from packed store to GPU slot (async H2D on memcpy stream)
        # Issue async H2D copy (records to gpu_slot.ready_event)
        self.packed_store.load_to_gpu_slot(
            layer_idx=self.layer_idx,
            expert_idx=expert_idx,
            gpu_slot=gpu_slot,
            stream=None,  # Uses memcpy_stream by default
        )

        # Update slot metadata (but don't create fp16 weights)
        slot.expert_id = (self.layer_idx, expert_idx)
        slot.hit_count = 1
        slot.last_access = time.time()
        self.expert_to_slot[expert_idx] = slot

    def get_stats(self) -> Dict:
        """Get cache statistics for this layer."""
        with self._lock:
            total = self.hits + self.misses
            return {
                "layer": self.layer_idx,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / max(1, total),
                "promotions": self.promotions,
                "evictions": self.evictions,
                "cached_count": len(self.expert_to_slot),
                "pinned_count": sum(1 for s in self.slots[:self.pinned_slots] if s.expert_id),
                "hot_count": sum(1 for s in self.slots[self.pinned_slots:self.pinned_slots+self.hot_slots] if s.expert_id),
                "probation_count": sum(1 for s in self.slots[self.pinned_slots+self.hot_slots:] if s.expert_id),
            }

    def reset_stats(self):
        """Reset statistics."""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.promotions = 0
            self.evictions = 0

    def clear_fp16_cache(self):
        """
        Clear cached fp16 weights to free GPU memory.

        Call after processing a layer to free the dequantized weights.
        The INT4 data stays on GPU for fast re-dequant if needed.

        NOTE: This is a lightweight operation - no gc/sync in hot path.
        Memory will be reclaimed lazily by CUDA allocator.
        """
        with self._lock:
            # Skip if nothing to clear (common in fused INT4 path)
            if not self.gpu_weights:
                return

            # Clear the fp16 weights dict
            # Delete each tensor explicitly
            for slot_idx, weights_dict in list(self.gpu_weights.items()):
                for name in list(weights_dict.keys()):
                    del weights_dict[name]
                del self.gpu_weights[slot_idx]

            # Also clear the cached fp16 in GpuExpertSlots
            for slot in self.gpu_int4_slots.values():
                if slot._cached_fp16 is not None:
                    for name in list(slot._cached_fp16.keys()):
                        del slot._cached_fp16[name]
                    slot._cached_fp16 = None

            # NOTE: No gc.collect() or torch.cuda.synchronize() here!
            # These are extremely expensive (30-60ms each) and would add
            # ~2 seconds per token when called 32 times.
            # CUDA allocator will reuse memory on next allocation.

    def clear_expert_fp16(self, expert_idx: int):
        """
        Clear fp16 cache for a specific expert.

        Keeps INT4 data for fast re-dequant on next access.
        """
        with self._lock:
            if expert_idx not in self.expert_to_slot:
                return

            slot = self.expert_to_slot[expert_idx]

            # Clear from gpu_weights dict
            if slot.slot_idx in self.gpu_weights:
                weights_dict = self.gpu_weights[slot.slot_idx]
                for name in list(weights_dict.keys()):
                    del weights_dict[name]
                del self.gpu_weights[slot.slot_idx]

            # Clear cached fp16 in GpuExpertSlot
            if self.use_packed_mode and slot.slot_idx in self.gpu_int4_slots:
                gpu_slot = self.gpu_int4_slots[slot.slot_idx]
                if gpu_slot._cached_fp16 is not None:
                    for name in list(gpu_slot._cached_fp16.keys()):
                        del gpu_slot._cached_fp16[name]
                    gpu_slot._cached_fp16 = None

    def set_packed_mode(
        self,
        store,
        hidden_dim: int,
        intermediate_dim: int,
        group_size: int,
    ):
        """
        Enable packed INT4 mode for this layer cache.

        Args:
            store: PackedExpertStore instance
            hidden_dim: Model hidden dimension
            intermediate_dim: Expert intermediate dimension
            group_size: Quantization group size
        """
        self.packed_store = store
        self.use_packed_mode = True
        self._packed_hidden_dim = hidden_dim
        self._packed_intermediate_dim = intermediate_dim
        self._packed_group_size = group_size


class ExpertCacheManager:
    """
    Manages per-layer expert caches across all transformer layers.

    Features:
    - Per-layer LRU (prevents cross-layer thrashing)
    - Dynamic slot reallocation based on miss rates
    - Async prefetch via CUDA streams
    - Pin protection for speculative decoding
    """

    def __init__(
        self,
        model_config: MixtralConfig,
        cache_config: ExpertCacheConfig,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ):
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = device
        self.dtype = dtype

        # Calculate expert size
        self.expert_size_bytes = model_config.expert_size_bytes

        # Create per-layer caches with non-uniform slot allocation
        # Middle layers tend to have more routing diversity (churnier)
        self.layer_caches: List[PerLayerCache] = []
        slot_allocation = self._compute_slot_allocation(
            model_config.num_layers,
            cache_config.slots_per_layer,
        )

        for layer_idx in range(model_config.num_layers):
            # Create config override for this layer's slot count
            layer_config = ExpertCacheConfig(
                pinned_slots=0,
                hot_slots=slot_allocation[layer_idx] // 2,
                probation_slots=slot_allocation[layer_idx] - slot_allocation[layer_idx] // 2,
                enable_dynamic_slots=cache_config.enable_dynamic_slots,
                realloc_interval_tokens=cache_config.realloc_interval_tokens,
                miss_rate_threshold=cache_config.miss_rate_threshold,
                max_slots_per_layer=cache_config.max_slots_per_layer,
                eviction_policy=cache_config.eviction_policy,
                enable_affinity_bonus=cache_config.enable_affinity_bonus,
                affinity_bonus=cache_config.affinity_bonus,
                affinity_entropy_threshold=cache_config.affinity_entropy_threshold,
                enable_prefetch=cache_config.enable_prefetch,
            )
            cache = PerLayerCache(
                layer_idx=layer_idx,
                num_experts=model_config.num_experts,
                expert_size_bytes=self.expert_size_bytes,
                config=layer_config,
                device=device,
                dtype=dtype,
            )
            self.layer_caches.append(cache)

        # CUDA streams for async loading (only if CUDA available)
        self.load_streams: List[torch.cuda.Stream] = []
        if torch.cuda.is_available() and device.type == "cuda":
            for _ in range(2):  # 2 streams for double-buffering
                self.load_streams.append(torch.cuda.Stream())

        # Prefetch queue
        self.prefetch_queue: queue.Queue = queue.Queue()
        self._running = True
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker, daemon=True
        )
        self._prefetch_thread.start()

        # Dynamic reallocation tracking
        self.tokens_since_realloc = 0
        self._slot_allocations = slot_allocation.copy()

        # Adaptive pinning tracking
        self.tokens_since_pin_refresh = 0
        self.pin_refresh_interval = 16  # Refresh pins every N decode tokens

        # Packed INT4 store (optional, for packed mode)
        self.packed_store = None
        self.use_packed_mode = False

        # Per-sequence LFU tracking for multi-sequence stability
        # seq_id -> {layer_idx -> {expert_idx -> decayed_count}}
        self.sequence_lfu: Dict[int, Dict[int, Dict[int, float]]] = {}
        self.lfu_decay_rate = 0.95  # Decay per token step
        self.active_sequence_id: Optional[int] = None

        # Miss burst tracking
        self.miss_burst_count = 0  # Misses in current token step
        self.miss_burst_threshold = 8  # Flag if > this many layers miss
        self.total_burst_events = 0

        # Load time tracking
        self.total_load_time_ms = 0.0

    def _compute_slot_allocation(
        self, num_layers: int, base_slots: int
    ) -> List[int]:
        """
        Compute slot allocation across layers.

        Profile-informed heuristic: L0-11, L14-31 get base_slots+1.
        This adds +30 slots (~2.7 GB) - maximum before OOM on 16GB GPU.

        Note: Slot swapping doesn't help because "low-miss" layers are
        low-miss *because* of their slots, not inherently stable routing.
        Use adaptive pinning instead to improve hit rate.
        """
        allocation = []
        for layer_idx in range(num_layers):
            if layer_idx < 12 or layer_idx >= 14:
                allocation.append(base_slots + 1)
            else:
                allocation.append(base_slots)
        return allocation

    def set_packed_store(self, store):
        """
        Attach a PackedExpertStore for INT4 packed mode.

        In packed mode, experts are loaded directly from the store's mmap'd
        INT4 blobs and dequantized on GPU, bypassing the fp16 RAM storage.

        Args:
            store: PackedExpertStore instance
        """
        from .packed_expert_store import PackedExpertStore, GpuExpertSlot, BlobLayout

        self.packed_store = store
        self.use_packed_mode = True

        # Create GPU expert slots for each layer cache
        hidden_dim = store.model_config["hidden_dim"]
        intermediate_dim = store.model_config["intermediate_dim"]
        group_size = store.quant_config["group_size"]

        for layer_cache in self.layer_caches:
            layer_cache.set_packed_mode(
                store=store,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                group_size=group_size,
            )

    def register_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        weights: Dict[str, torch.Tensor],
    ):
        """Register expert weights with cache manager."""
        self.layer_caches[layer_idx].register_expert(expert_idx, weights)

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

    def prefetch_int4_slot(self, layer_idx: int, expert_idx: int):
        """
        Schedule async load for expert (for overlap with compute).

        Does NOT wait for completion - use get_int4_slot when ready to compute.
        """
        return self.layer_caches[layer_idx].prefetch_int4_slot(expert_idx)

    def get_int4_slot(self, layer_idx: int, expert_idx: int):
        """
        Get INT4 slot ready for compute.

        Loads expert if not cached, waits for completion (GPU-side wait).
        Returns the GpuExpertSlot with INT4 buffers.
        """
        start = time.time()
        slot = self.layer_caches[layer_idx].get_int4_slot(expert_idx)
        self.total_load_time_ms += (time.time() - start) * 1000
        return slot

    def clear_layer_fp16(self, layer_idx: int):
        """
        Clear fp16 cache for a specific layer.

        Call after processing each layer to free GPU memory.
        Keeps INT4 data for fast re-dequant.
        """
        self.layer_caches[layer_idx].clear_fp16_cache()

    def clear_expert_fp16(self, layer_idx: int, expert_idx: int):
        """
        Clear fp16 cache for a specific expert.

        Call immediately after using an expert to minimize peak memory.
        Keeps INT4 data for fast re-dequant on next access.
        """
        self.layer_caches[layer_idx].clear_expert_fp16(expert_idx)

    def prefetch(self, layer_idx: int, expert_indices: List[int]):
        """
        Queue experts for async prefetch.

        Called during layer N compute to prefetch layer N experts
        (layer-local overlap, not cross-layer prediction).
        """
        for expert_idx in expert_indices:
            if not self.is_cached(layer_idx, expert_idx):
                self.prefetch_queue.put((layer_idx, expert_idx))

    def pin_experts(
        self,
        expert_ids: List[Tuple[int, int]],
        duration_sec: float = 1.0,
    ):
        """
        Pin experts to prevent eviction during spec decoding verification.

        Args:
            expert_ids: List of (layer_idx, expert_idx) tuples
            duration_sec: Pin duration
        """
        for layer_idx, expert_idx in expert_ids:
            self.layer_caches[layer_idx].pin_expert(expert_idx, duration_sec)

    def maybe_reallocate_slots(self, token_count: int = 1):
        """
        Check if dynamic slot reallocation or pin refresh is needed.

        Called periodically during inference.
        """
        # Refresh adaptive pins periodically
        self.tokens_since_pin_refresh += token_count
        if self.tokens_since_pin_refresh >= self.pin_refresh_interval:
            self.tokens_since_pin_refresh = 0
            self.refresh_adaptive_pins()

        # Dynamic slot reallocation (if enabled)
        if not self.cache_config.enable_dynamic_slots:
            return

        self.tokens_since_realloc += token_count
        if self.tokens_since_realloc < self.cache_config.realloc_interval_tokens:
            return

        self.tokens_since_realloc = 0
        self._do_reallocation()

    def refresh_adaptive_pins(self):
        """
        Refresh adaptive pins for all layers based on recent frequency.

        Call this periodically (e.g., every 64-256 tokens) to update
        which experts are protected from eviction.
        """
        import os
        debug = os.environ.get("DEBUG_PINS", "0") == "1"
        for cache in self.layer_caches:
            cache.refresh_adaptive_pins(debug=debug)

    def _do_reallocation(self):
        """
        Reallocate slots from low-miss to high-miss layers.

        Algorithm:
        1. Collect stats from all layers
        2. Find high-miss layers (miss rate > threshold) as recipients
        3. Find low-miss layers (miss rate < threshold/2) with >1 slot as donors
        4. Actually transfer slots from donors to recipients
        5. Log reallocation for debugging
        """
        import os
        debug = os.environ.get("DEBUG_REALLOC", "0") == "1"

        stats = [cache.get_stats() for cache in self.layer_caches]

        # Find donor (low miss rate) and recipient (high miss rate) layers
        # Sort recipients by miss rate (highest first) and donors by miss rate (lowest first)
        donors = []
        recipients = []

        miss_threshold = self.cache_config.miss_rate_threshold
        low_miss_threshold = miss_threshold / 2

        for s in stats:
            layer_idx = s["layer"]
            miss_rate = 1 - s["hit_rate"]
            current_slots = self.layer_caches[layer_idx].get_slot_count()

            if miss_rate > miss_threshold:
                # High miss rate - wants more slots
                if current_slots < self.cache_config.max_slots_per_layer:
                    recipients.append((layer_idx, miss_rate))
            elif miss_rate < low_miss_threshold:
                # Low miss rate - can donate slots
                if current_slots > 1:
                    donors.append((layer_idx, miss_rate))

        # Sort: recipients by miss rate (highest first), donors by miss rate (lowest first)
        recipients.sort(key=lambda x: -x[1])
        donors.sort(key=lambda x: x[1])

        if debug and (donors or recipients):
            print(f"[Realloc] Recipients (high miss): {[(l, f'{m:.1%}') for l, m in recipients[:5]]}")
            print(f"[Realloc] Donors (low miss): {[(l, f'{m:.1%}') for l, m in donors[:5]]}")

        # Transfer slots
        transfers = 0
        max_transfers_per_interval = 2  # Limit churn

        for (recipient_idx, _), (donor_idx, _) in zip(recipients, donors):
            if transfers >= max_transfers_per_interval:
                break

            donor_cache = self.layer_caches[donor_idx]
            recipient_cache = self.layer_caches[recipient_idx]

            # Check constraints
            if donor_cache.get_slot_count() <= 1:
                continue
            if recipient_cache.get_slot_count() >= self.cache_config.max_slots_per_layer:
                continue

            # Do the transfer
            if donor_cache.remove_slot() and recipient_cache.add_slot():
                self._slot_allocations[donor_idx] -= 1
                self._slot_allocations[recipient_idx] += 1
                transfers += 1

                if debug:
                    print(f"[Realloc] Moved slot: L{donor_idx} ({donor_cache.get_slot_count()+1}->{donor_cache.get_slot_count()}) "
                          f"-> L{recipient_idx} ({recipient_cache.get_slot_count()-1}->{recipient_cache.get_slot_count()})")

        if debug and transfers > 0:
            print(f"[Realloc] Total transfers: {transfers}")

    def _prefetch_worker(self):
        """Background thread for async prefetching."""
        stream_idx = 0
        while self._running:
            try:
                layer_idx, expert_idx = self.prefetch_queue.get(timeout=0.01)

                if self.is_cached(layer_idx, expert_idx):
                    continue

                # Use CUDA stream if available, otherwise just load directly
                if self.load_streams:
                    stream = self.load_streams[stream_idx % len(self.load_streams)]
                    stream_idx += 1
                    with torch.cuda.stream(stream):
                        self.layer_caches[layer_idx].get_weights(expert_idx)
                else:
                    # No CUDA streams (CPU mode) - load synchronously
                    self.layer_caches[layer_idx].get_weights(expert_idx)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Prefetch error: {e}")

    # ========== Per-Sequence LFU Tracking ==========

    def register_sequence(self, seq_id: int):
        """
        Register a new sequence for LFU tracking.

        Call when a new sequence starts generation.
        """
        if seq_id not in self.sequence_lfu:
            self.sequence_lfu[seq_id] = {
                layer_idx: {} for layer_idx in range(self.model_config.num_layers)
            }

    def unregister_sequence(self, seq_id: int):
        """Remove a completed sequence from tracking."""
        if seq_id in self.sequence_lfu:
            del self.sequence_lfu[seq_id]

    def set_active_sequence(self, seq_id: int):
        """Set the currently active sequence for cache affinity."""
        self.active_sequence_id = seq_id
        if seq_id not in self.sequence_lfu:
            self.register_sequence(seq_id)

    def record_expert_use(
        self,
        seq_id: int,
        layer_idx: int,
        expert_idx: int,
    ):
        """
        Record expert usage for sequence LFU.

        Call after each expert is used during forward pass.
        """
        if seq_id not in self.sequence_lfu:
            self.register_sequence(seq_id)

        lfu = self.sequence_lfu[seq_id][layer_idx]
        lfu[expert_idx] = lfu.get(expert_idx, 0.0) + 1.0

    def decay_sequence_lfu(self, seq_id: int):
        """
        Apply decay to sequence LFU counts.

        Call once per token step to decay historical usage.
        """
        if seq_id not in self.sequence_lfu:
            return

        for layer_lfu in self.sequence_lfu[seq_id].values():
            for expert_idx in list(layer_lfu.keys()):
                layer_lfu[expert_idx] *= self.lfu_decay_rate
                # Remove near-zero entries
                if layer_lfu[expert_idx] < 0.01:
                    del layer_lfu[expert_idx]

    def get_sequence_hot_experts(
        self,
        seq_id: int,
        layer_idx: int,
        top_k: int = 4,
    ) -> List[int]:
        """
        Get the hot experts for a sequence at a specific layer.

        Returns:
            List of expert indices sorted by usage (most used first)
        """
        if seq_id not in self.sequence_lfu:
            return []

        lfu = self.sequence_lfu[seq_id].get(layer_idx, {})
        sorted_experts = sorted(lfu.items(), key=lambda x: -x[1])
        return [expert_idx for expert_idx, _ in sorted_experts[:top_k]]

    def start_token_step(self):
        """
        Called at the start of each token generation step.

        Resets per-step tracking like miss burst count.
        """
        self.miss_burst_count = 0

    def end_token_step(self):
        """
        Called at the end of each token generation step.

        Checks for miss bursts and decays LFU.
        """
        if self.miss_burst_count > self.miss_burst_threshold:
            self.total_burst_events += 1

        # Decay LFU for active sequence
        if self.active_sequence_id is not None:
            self.decay_sequence_lfu(self.active_sequence_id)

    def record_miss(self, layer_idx: int):
        """Record a cache miss for burst tracking."""
        self.miss_burst_count += 1

    # ========== Statistics ==========

    def get_stats(self) -> Dict:
        """Get aggregated cache statistics."""
        total_hits = sum(c.hits for c in self.layer_caches)
        total_misses = sum(c.misses for c in self.layer_caches)
        total_evictions = sum(c.evictions for c in self.layer_caches)
        total_promotions = sum(c.promotions for c in self.layer_caches)

        # Per-layer hit rates
        per_layer_stats = [c.get_stats() for c in self.layer_caches]
        layer_hit_rates = [s["hit_rate"] for s in per_layer_stats]

        return {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": total_hits / max(1, total_hits + total_misses),
            "total_evictions": total_evictions,
            "total_promotions": total_promotions,
            "avg_load_time_ms": self.total_load_time_ms / max(1, total_misses),
            "per_layer_stats": per_layer_stats,
            # New metrics
            "layer_hit_rate_min": min(layer_hit_rates) if layer_hit_rates else 0.0,
            "layer_hit_rate_median": sorted(layer_hit_rates)[len(layer_hit_rates)//2] if layer_hit_rates else 0.0,
            "layer_hit_rate_max": max(layer_hit_rates) if layer_hit_rates else 0.0,
            "miss_burst_events": self.total_burst_events,
            "active_sequences": len(self.sequence_lfu),
        }

    def reset_stats(self):
        """Reset all statistics."""
        for cache in self.layer_caches:
            cache.reset_stats()
        self.total_load_time_ms = 0.0
        self.total_burst_events = 0
        self.miss_burst_count = 0

    def shutdown(self):
        """Shutdown background threads."""
        self._running = False
        if self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)

    def __del__(self):
        self.shutdown()
