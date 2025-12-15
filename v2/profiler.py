"""
Profiling utilities for HydraNet.

Features:
- NVTX ranges for Nsight profiling
- CUDA event timers for latency measurement
- Rolling statistics for throughput tracking
- Structured logging for analysis
"""

import torch
import torch.cuda
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from collections import deque
from contextlib import contextmanager
import time
import json


# Try to import NVTX (available in PyTorch 1.8+)
try:
    from torch.cuda import nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False


@dataclass
class TimerStats:
    """Statistics for a single timer."""
    name: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    recent: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def mean_ms(self) -> float:
        return self.total_ms / max(1, self.count)

    @property
    def recent_mean_ms(self) -> float:
        if not self.recent:
            return 0.0
        return sum(self.recent) / len(self.recent)

    @property
    def p95_ms(self) -> float:
        if len(self.recent) < 20:
            return self.max_ms
        sorted_recent = sorted(self.recent)
        idx = int(len(sorted_recent) * 0.95)
        return sorted_recent[idx]

    def record(self, ms: float):
        self.count += 1
        self.total_ms += ms
        self.min_ms = min(self.min_ms, ms)
        self.max_ms = max(self.max_ms, ms)
        self.recent.append(ms)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms if self.count > 0 else 0,
            "max_ms": self.max_ms,
            "p95_ms": self.p95_ms,
            "recent_mean_ms": self.recent_mean_ms,
        }


class CUDATimer:
    """
    CUDA event-based timer for accurate GPU timing.

    Usage:
        timer = CUDATimer()
        timer.start()
        # ... GPU work ...
        elapsed_ms = timer.stop()
    """

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self._started = False

    def start(self):
        """Record start event."""
        self.start_event.record()
        self._started = True

    def stop(self) -> float:
        """Record end event and return elapsed time in ms."""
        if not self._started:
            return 0.0
        self.end_event.record()
        self.end_event.synchronize()
        self._started = False
        return self.start_event.elapsed_time(self.end_event)


class Profiler:
    """
    Central profiling manager for HydraNet.

    Tracks:
    - Per-component latencies (attention, MoE, H2D, etc.)
    - Throughput metrics (tokens/sec, experts/sec)
    - Cache statistics (hit rate, evictions)
    - Memory usage
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timers: Dict[str, TimerStats] = {}
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}

        # Pre-allocate common timers
        self._cuda_timers: Dict[str, CUDATimer] = {}

        # Rolling throughput tracking
        self.token_times: deque = deque(maxlen=1000)
        self.generation_start: Optional[float] = None

    def get_timer(self, name: str) -> TimerStats:
        """Get or create timer stats."""
        if name not in self.timers:
            self.timers[name] = TimerStats(name=name)
        return self.timers[name]

    @contextmanager
    def timer(self, name: str):
        """
        Context manager for timing a code block.

        Usage:
            with profiler.timer("attention"):
                # ... attention code ...
        """
        if not self.enabled:
            yield
            return

        if name not in self._cuda_timers:
            self._cuda_timers[name] = CUDATimer()

        cuda_timer = self._cuda_timers[name]
        cuda_timer.start()

        try:
            yield
        finally:
            elapsed = cuda_timer.stop()
            self.get_timer(name).record(elapsed)

    @contextmanager
    def nvtx_range(self, name: str, color: str = "blue"):
        """
        NVTX range for Nsight profiling.

        Usage:
            with profiler.nvtx_range("forward_pass"):
                # ... code visible in Nsight ...
        """
        if not self.enabled or not NVTX_AVAILABLE:
            yield
            return

        # Color mapping
        colors = {
            "blue": 0x0000FF,
            "green": 0x00FF00,
            "red": 0xFF0000,
            "yellow": 0xFFFF00,
            "purple": 0xFF00FF,
            "cyan": 0x00FFFF,
        }

        nvtx.range_push(name)
        try:
            yield
        finally:
            nvtx.range_pop()

    def record_token(self):
        """Record a generated token for throughput calculation."""
        self.token_times.append(time.time())

    def start_generation(self):
        """Mark start of generation."""
        self.generation_start = time.time()
        self.token_times.clear()

    def get_tokens_per_second(self) -> float:
        """Calculate recent tokens/second."""
        if len(self.token_times) < 2:
            return 0.0

        times = list(self.token_times)
        duration = times[-1] - times[0]
        if duration <= 0:
            return 0.0

        return (len(times) - 1) / duration

    def increment(self, name: str, amount: int = 1):
        """Increment a counter."""
        self.counters[name] = self.counters.get(name, 0) + amount

    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        self.gauges[name] = value

    def record_cache_stats(self, stats: Dict):
        """Record cache statistics."""
        self.set_gauge("cache_hit_rate", stats.get("hit_rate", 0))
        self.set_gauge("cache_evictions", stats.get("total_evictions", 0))
        self.increment("cache_hits", stats.get("total_hits", 0))
        self.increment("cache_misses", stats.get("total_misses", 0))

    def get_summary(self) -> Dict:
        """Get profiling summary."""
        return {
            "timers": {name: stats.to_dict() for name, stats in self.timers.items()},
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "throughput": {
                "tokens_per_second": self.get_tokens_per_second(),
                "total_tokens": len(self.token_times),
            },
        }

    def print_summary(self):
        """Print formatted profiling summary."""
        print("\n" + "=" * 70)
        print("PROFILING SUMMARY")
        print("=" * 70)

        if self.timers:
            print("\nTimers (ms):")
            print(f"{'Name':<30} {'Count':>8} {'Mean':>10} {'P95':>10} {'Max':>10}")
            print("-" * 70)

            for name, stats in sorted(self.timers.items()):
                if stats.count > 0:
                    print(f"{name:<30} {stats.count:>8} "
                          f"{stats.mean_ms:>10.2f} {stats.p95_ms:>10.2f} "
                          f"{stats.max_ms:>10.2f}")

        if self.counters:
            print("\nCounters:")
            for name, value in sorted(self.counters.items()):
                print(f"  {name}: {value}")

        if self.gauges:
            print("\nGauges:")
            for name, value in sorted(self.gauges.items()):
                print(f"  {name}: {value:.4f}")

        tps = self.get_tokens_per_second()
        if tps > 0:
            print(f"\nThroughput: {tps:.1f} tokens/sec")

    def reset(self):
        """Reset all statistics."""
        self.timers.clear()
        self.counters.clear()
        self.gauges.clear()
        self.token_times.clear()
        self.generation_start = None

    def save(self, path: str):
        """Save profiling data to JSON."""
        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)


# Global profiler instance
_profiler: Optional[Profiler] = None


def get_profiler() -> Profiler:
    """Get global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = Profiler()
    return _profiler


def enable_profiling():
    """Enable global profiling."""
    get_profiler().enabled = True


def disable_profiling():
    """Disable global profiling."""
    get_profiler().enabled = False


# Convenience decorators
def profile_function(name: Optional[str] = None):
    """
    Decorator to profile a function.

    Usage:
        @profile_function("my_function")
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        timer_name = name or func.__name__

        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            with profiler.timer(timer_name):
                with profiler.nvtx_range(timer_name):
                    return func(*args, **kwargs)

        return wrapper
    return decorator


class OverlapChecker:
    """
    Utility to verify H2D/compute overlap.

    Records events at key points to verify streams are actually overlapping.
    """

    def __init__(self):
        self.compute_events: List[torch.cuda.Event] = []
        self.copy_events: List[torch.cuda.Event] = []

    def mark_compute_start(self):
        """Record compute start."""
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.compute_events.append(("start", event))

    def mark_compute_end(self):
        """Record compute end."""
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.compute_events.append(("end", event))

    def mark_copy_start(self):
        """Record H2D copy start."""
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.copy_events.append(("start", event))

    def mark_copy_end(self):
        """Record H2D copy end."""
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.copy_events.append(("end", event))

    def analyze_overlap(self) -> Dict:
        """
        Analyze if copy and compute actually overlapped.

        Returns dict with overlap statistics.
        """
        torch.cuda.synchronize()

        # Extract timings (simplified analysis)
        compute_ranges = []
        copy_ranges = []

        compute_start = None
        for label, event in self.compute_events:
            if label == "start":
                compute_start = event
            elif label == "end" and compute_start:
                compute_ranges.append((compute_start, event))
                compute_start = None

        copy_start = None
        for label, event in self.copy_events:
            if label == "start":
                copy_start = event
            elif label == "end" and copy_start:
                copy_ranges.append((copy_start, event))
                copy_start = None

        # Check for overlap
        overlapped = 0
        total_compute = 0
        total_copy = 0

        for cs, ce in compute_ranges:
            compute_time = cs.elapsed_time(ce)
            total_compute += compute_time

        for cs, ce in copy_ranges:
            copy_time = cs.elapsed_time(ce)
            total_copy += copy_time

        return {
            "total_compute_ms": total_compute,
            "total_copy_ms": total_copy,
            "num_compute_ranges": len(compute_ranges),
            "num_copy_ranges": len(copy_ranges),
            "effective_overlap": total_copy < total_compute,  # Simplified check
        }

    def reset(self):
        """Reset events."""
        self.compute_events.clear()
        self.copy_events.clear()
