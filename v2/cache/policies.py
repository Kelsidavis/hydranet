"""
Cache eviction policies.

Implements various eviction strategies:
- LRU: Least Recently Used
- LFU: Least Frequently Used
- Sticky LFU: LFU with probation tier for new entries
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TypeVar, Generic
from collections import OrderedDict
import time
import heapq

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Generic cache entry with tracking metadata."""
    key: T
    frequency: int = 0
    last_access: float = field(default_factory=time.time)
    inserted_at: float = field(default_factory=time.time)
    in_probation: bool = True  # For sticky LFU


class EvictionTracker:
    """
    Tracks access patterns for eviction decisions.

    Used to collect statistics and make informed eviction choices.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.access_history: List[int] = []  # Expert IDs
        self.frequency: Dict[int, int] = {}  # ID -> count in window

    def record_access(self, expert_id: int):
        """Record an access."""
        self.access_history.append(expert_id)
        self.frequency[expert_id] = self.frequency.get(expert_id, 0) + 1

        # Trim old history
        if len(self.access_history) > self.window_size:
            old_id = self.access_history.pop(0)
            self.frequency[old_id] -= 1
            if self.frequency[old_id] <= 0:
                del self.frequency[old_id]

    def get_top_k(self, k: int) -> List[int]:
        """Get top-k most frequent IDs."""
        sorted_ids = sorted(
            self.frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [id for id, _ in sorted_ids[:k]]

    def get_frequency(self, expert_id: int) -> int:
        """Get frequency count for an ID."""
        return self.frequency.get(expert_id, 0)


class LRUPolicy:
    """
    Least Recently Used eviction policy.

    Simple and effective for many workloads.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[int, CacheEntry] = OrderedDict()

    def access(self, key: int) -> bool:
        """
        Record access. Returns True if was already cached.
        """
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key].last_access = time.time()
            self.cache[key].frequency += 1
            return True

        # New entry
        entry = CacheEntry(key=key, frequency=1)
        self.cache[key] = entry
        self.cache.move_to_end(key)

        return False

    def evict(self) -> Optional[int]:
        """
        Evict LRU entry if over capacity.

        Returns evicted key or None.
        """
        if len(self.cache) <= self.capacity:
            return None

        # Pop oldest (first item)
        key, _ = self.cache.popitem(last=False)
        return key

    def remove(self, key: int):
        """Remove specific key."""
        if key in self.cache:
            del self.cache[key]

    def contains(self, key: int) -> bool:
        """Check if key is cached."""
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)


class LFUPolicy:
    """
    Least Frequently Used eviction policy.

    Better for workloads with stable hot set.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[int, CacheEntry] = {}
        self.min_freq = 0
        # freq -> OrderedDict of keys (for tie-breaking by recency)
        self.freq_buckets: Dict[int, OrderedDict[int, None]] = {}

    def access(self, key: int) -> bool:
        """Record access. Returns True if was already cached."""
        if key in self.cache:
            entry = self.cache[key]
            old_freq = entry.frequency

            # Remove from old frequency bucket
            del self.freq_buckets[old_freq][key]
            if not self.freq_buckets[old_freq]:
                del self.freq_buckets[old_freq]
                if self.min_freq == old_freq:
                    self.min_freq += 1

            # Add to new frequency bucket
            entry.frequency += 1
            entry.last_access = time.time()
            new_freq = entry.frequency

            if new_freq not in self.freq_buckets:
                self.freq_buckets[new_freq] = OrderedDict()
            self.freq_buckets[new_freq][key] = None

            return True

        # New entry
        entry = CacheEntry(key=key, frequency=1)
        self.cache[key] = entry

        if 1 not in self.freq_buckets:
            self.freq_buckets[1] = OrderedDict()
        self.freq_buckets[1][key] = None
        self.min_freq = 1

        return False

    def evict(self) -> Optional[int]:
        """Evict LFU entry if over capacity."""
        if len(self.cache) <= self.capacity:
            return None

        # Get LFU bucket, remove oldest in that bucket
        bucket = self.freq_buckets[self.min_freq]
        key, _ = bucket.popitem(last=False)

        if not bucket:
            del self.freq_buckets[self.min_freq]

        del self.cache[key]
        return key

    def remove(self, key: int):
        """Remove specific key."""
        if key not in self.cache:
            return

        entry = self.cache[key]
        freq = entry.frequency

        del self.freq_buckets[freq][key]
        if not self.freq_buckets[freq]:
            del self.freq_buckets[freq]

        del self.cache[key]

    def contains(self, key: int) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)


class StickyLFU:
    """
    Sticky LFU with probation tier.

    Design:
    - New entries start in probation
    - Second access promotes to main tier (sticky)
    - Eviction prefers probation tier
    - Main tier uses LFU for eviction

    This prevents cache pollution from one-time accesses.
    """

    def __init__(
        self,
        main_capacity: int,
        probation_capacity: int,
    ):
        self.main_capacity = main_capacity
        self.probation_capacity = probation_capacity

        # Main cache: promoted entries (sticky)
        self.main: Dict[int, CacheEntry] = {}
        self.main_freq_buckets: Dict[int, OrderedDict[int, None]] = {}
        self.main_min_freq = 1

        # Probation cache: new entries
        self.probation: OrderedDict[int, CacheEntry] = OrderedDict()

    def access(self, key: int) -> bool:
        """
        Record access.

        Returns True if was already cached (hit).
        """
        # Check main cache
        if key in self.main:
            self._increment_main_freq(key)
            return True

        # Check probation
        if key in self.probation:
            # Second hit - promote to main
            entry = self.probation.pop(key)
            entry.in_probation = False
            self._add_to_main(key, entry)
            return True

        # New entry - add to probation
        entry = CacheEntry(key=key, frequency=1, in_probation=True)
        self.probation[key] = entry
        self.probation.move_to_end(key)

        return False

    def _increment_main_freq(self, key: int):
        """Increment frequency of main cache entry."""
        entry = self.main[key]
        old_freq = entry.frequency

        # Remove from old bucket
        del self.main_freq_buckets[old_freq][key]
        if not self.main_freq_buckets[old_freq]:
            del self.main_freq_buckets[old_freq]
            if self.main_min_freq == old_freq:
                self.main_min_freq = old_freq + 1

        # Add to new bucket
        entry.frequency += 1
        entry.last_access = time.time()
        new_freq = entry.frequency

        if new_freq not in self.main_freq_buckets:
            self.main_freq_buckets[new_freq] = OrderedDict()
        self.main_freq_buckets[new_freq][key] = None

    def _add_to_main(self, key: int, entry: CacheEntry):
        """Add entry to main cache."""
        entry.frequency = 2  # Promoted with 2 hits
        entry.last_access = time.time()

        self.main[key] = entry

        if 2 not in self.main_freq_buckets:
            self.main_freq_buckets[2] = OrderedDict()
        self.main_freq_buckets[2][key] = None

        if not self.main_freq_buckets.get(self.main_min_freq):
            self.main_min_freq = 2

    def evict(self) -> Optional[int]:
        """
        Evict if over capacity.

        Priority:
        1. Evict from probation (FIFO)
        2. Evict from main (LFU)

        Returns evicted key or None.
        """
        total = len(self.main) + len(self.probation)
        total_capacity = self.main_capacity + self.probation_capacity

        if total <= total_capacity:
            return None

        # Try probation first
        if self.probation:
            key, _ = self.probation.popitem(last=False)
            return key

        # Evict from main (LFU)
        if self.main and self.main_min_freq in self.main_freq_buckets:
            bucket = self.main_freq_buckets[self.main_min_freq]
            key, _ = bucket.popitem(last=False)

            if not bucket:
                del self.main_freq_buckets[self.main_min_freq]
                # Find new min_freq
                if self.main_freq_buckets:
                    self.main_min_freq = min(self.main_freq_buckets.keys())

            del self.main[key]
            return key

        return None

    def remove(self, key: int):
        """Remove specific key."""
        if key in self.probation:
            del self.probation[key]
        elif key in self.main:
            freq = self.main[key].frequency
            del self.main_freq_buckets[freq][key]
            if not self.main_freq_buckets[freq]:
                del self.main_freq_buckets[freq]
            del self.main[key]

    def contains(self, key: int) -> bool:
        """Check if key is cached."""
        return key in self.main or key in self.probation

    def is_promoted(self, key: int) -> bool:
        """Check if key is in main (promoted) tier."""
        return key in self.main

    def get_main_keys(self) -> Set[int]:
        """Get all keys in main tier."""
        return set(self.main.keys())

    def get_probation_keys(self) -> Set[int]:
        """Get all keys in probation tier."""
        return set(self.probation.keys())

    def __len__(self) -> int:
        return len(self.main) + len(self.probation)

    def stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "main_count": len(self.main),
            "probation_count": len(self.probation),
            "main_capacity": self.main_capacity,
            "probation_capacity": self.probation_capacity,
        }
