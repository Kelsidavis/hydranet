"""
KV cache with paging and landmark pinning.

Design:
- VRAM window: Most recent N tokens (fast access)
- RAM pages: Older tokens paged to RAM
- Landmark pinning: Pin important pages (tool calls, system prompt)

Phase 1 (SimpleKVCache): Preallocated fp16, no paging, fixed max length.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import OrderedDict
import threading
import time

from ..config import KVCacheConfig, MixtralConfig, GLM4AirConfig


# =============================================================================
# Phase 1: Simple Preallocated KV Cache (no paging, no int8)
# =============================================================================

class SimpleKVCache:
    """
    Simple preallocated KV cache for efficient autoregressive generation.

    Memory layout per layer:
        K: [batch, num_kv_heads, max_seq_len, head_dim]
        V: [batch, num_kv_heads, max_seq_len, head_dim]

    Supports optional INT8 quantization (50% memory savings):
        - Per-position, per-head absmax scaling
        - Quantize on update, dequantize on get
        - Minimal quality impact for attention

    Usage:
        cache = SimpleKVCache.from_model_config(config, max_seq_len=2048)

        # In attention forward:
        # Prefill (is_prefill=True):
        cache.update(layer_idx, k, v, start_pos=0)
        cache.set_len(prompt_len)
        k_full, v_full = cache.get(layer_idx)

        # Decode (is_prefill=False):
        cache.update(layer_idx, k_new, v_new, start_pos=cache.cur_len)
        # After all layers: cache.advance(1)
        k_full, v_full = cache.get(layer_idx)
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        batch_size: int = 1,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
        use_int8: bool = False,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.use_int8 = use_int8

        # Current sequence length (tokens filled so far)
        self.cur_len = 0

        # Storage dtype: int8 for quantized, fp16 otherwise
        storage_dtype = torch.int8 if use_int8 else dtype

        # Preallocate K, V buffers for all layers
        # Shape: [batch, num_kv_heads, max_seq_len, head_dim]
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []

        # Scales for INT8 dequantization (per-position, per-head)
        # Shape: [batch, num_kv_heads, max_seq_len, 1]
        self.k_scale: List[torch.Tensor] = []
        self.v_scale: List[torch.Tensor] = []

        for _ in range(num_layers):
            k = torch.zeros(
                batch_size,
                num_kv_heads,
                max_seq_len,
                head_dim,
                device=device,
                dtype=storage_dtype,
            )
            v = torch.zeros(
                batch_size,
                num_kv_heads,
                max_seq_len,
                head_dim,
                device=device,
                dtype=storage_dtype,
            )
            self.k_cache.append(k)
            self.v_cache.append(v)

            if use_int8:
                k_s = torch.ones(
                    batch_size, num_kv_heads, max_seq_len, 1,
                    device=device, dtype=torch.float16,
                )
                v_s = torch.ones(
                    batch_size, num_kv_heads, max_seq_len, 1,
                    device=device, dtype=torch.float16,
                )
                self.k_scale.append(k_s)
                self.v_scale.append(v_s)

    def _quantize_int8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to INT8 with per-position, per-head scaling."""
        # tensor: [batch, num_heads, seq_len, head_dim]
        absmax = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / 127.0
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale.to(torch.float16)

    def _dequantize_int8(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize INT8 tensor back to fp16."""
        return quantized.to(self.dtype) * scale

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> None:
        """
        Write K, V to cache at specified position (in-place).

        Args:
            layer_idx: Which layer's cache to update
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
            start_pos: Position to write at (0 for prefill, cur_len for decode)
        """
        seq_len = k.shape[2]
        end_pos = start_pos + seq_len

        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Sequence length {end_pos} exceeds max {self.max_seq_len}"
            )

        if self.use_int8:
            # Quantize and store
            k_q, k_s = self._quantize_int8(k)
            v_q, v_s = self._quantize_int8(v)
            self.k_cache[layer_idx][:, :, start_pos:end_pos, :] = k_q
            self.v_cache[layer_idx][:, :, start_pos:end_pos, :] = v_q
            self.k_scale[layer_idx][:, :, start_pos:end_pos, :] = k_s
            self.v_scale[layer_idx][:, :, start_pos:end_pos, :] = v_s
        else:
            # Direct fp16 store
            self.k_cache[layer_idx][:, :, start_pos:end_pos, :] = k
            self.v_cache[layer_idx][:, :, start_pos:end_pos, :] = v

    def get(
        self,
        layer_idx: int,
        length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get K, V views from cache up to specified length.

        Args:
            layer_idx: Which layer's cache to read
            length: How many positions to return (default: cur_len)

        Returns:
            k: [batch, num_kv_heads, length, head_dim] in fp16
            v: [batch, num_kv_heads, length, head_dim] in fp16
        """
        if length is None:
            length = self.cur_len

        if self.use_int8:
            # Dequantize on read
            k_q = self.k_cache[layer_idx][:, :, :length, :]
            v_q = self.v_cache[layer_idx][:, :, :length, :]
            k_s = self.k_scale[layer_idx][:, :, :length, :]
            v_s = self.v_scale[layer_idx][:, :, :length, :]
            return (
                self._dequantize_int8(k_q, k_s),
                self._dequantize_int8(v_q, v_s),
            )
        else:
            return (
                self.k_cache[layer_idx][:, :, :length, :],
                self.v_cache[layer_idx][:, :, :length, :],
            )

    def set_len(self, length: int) -> None:
        """Set current sequence length (after prefill)."""
        self.cur_len = length

    def advance(self, num_tokens: int = 1) -> None:
        """Advance current position by num_tokens (after decode step)."""
        self.cur_len += num_tokens

    def reset(self) -> None:
        """Reset cache for new sequence."""
        self.cur_len = 0
        # Note: buffers not zeroed - they'll be overwritten

    def memory_mb(self) -> float:
        """Return total memory used by cache in MB."""
        if self.use_int8:
            # INT8 K, V: 1 byte each
            kv_bytes = (
                2  # K + V
                * self.batch_size
                * self.num_kv_heads
                * self.max_seq_len
                * self.head_dim
                * 1  # int8
            )
            # Scales: fp16, one per head per position
            scale_bytes = (
                2  # K + V scales
                * self.batch_size
                * self.num_kv_heads
                * self.max_seq_len
                * 1  # single value
                * 2  # fp16
            )
            per_layer_bytes = kv_bytes + scale_bytes
        else:
            per_layer_bytes = (
                2  # K + V
                * self.batch_size
                * self.num_kv_heads
                * self.max_seq_len
                * self.head_dim
                * 2  # fp16
            )
        return (per_layer_bytes * self.num_layers) / (1024 * 1024)

    @classmethod
    def from_model_config(
        cls,
        model_config: Union[MixtralConfig, GLM4AirConfig],
        max_seq_len: int = 2048,
        batch_size: int = 1,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
        use_int8: bool = False,
    ) -> "SimpleKVCache":
        """Create KV cache from model config (Mixtral or GLM4).

        Args:
            model_config: Model configuration
            max_seq_len: Maximum sequence length
            batch_size: Batch size
            device: Device for cache tensors
            dtype: Data type for fp16 mode (ignored if use_int8=True)
            use_int8: Use INT8 quantization (50% memory savings)

        Returns:
            SimpleKVCache instance
        """
        return cls(
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            use_int8=use_int8,
        )


# =============================================================================
# Phase 2+: Paged KV Cache with Landmarks (existing implementation)
# =============================================================================


@dataclass
class KVPage:
    """A page of KV cache in RAM."""
    page_id: int
    layer_idx: int
    start_pos: int  # Starting token position
    end_pos: int    # Ending token position (exclusive)

    # Tensors (in RAM, not pinned for memory efficiency)
    k_cache: torch.Tensor  # (page_size, num_kv_heads, head_dim)
    v_cache: torch.Tensor  # (page_size, num_kv_heads, head_dim)

    # Metadata
    is_landmark: bool = False
    landmark_type: Optional[str] = None  # "system", "tool_call", etc.
    last_access: float = field(default_factory=time.time)


@dataclass
class Landmark:
    """A pinned position range in KV cache."""
    start_pos: int
    end_pos: int
    landmark_type: str
    created_at: float = field(default_factory=time.time)


class LandmarkTracker:
    """
    Tracks and manages KV cache landmarks.

    Landmarks are important context positions that should be preserved:
    - System prompt
    - Tool call outputs
    - Key conversation turns
    """

    def __init__(self, max_landmarks: int = 8):
        self.max_landmarks = max_landmarks
        self.landmarks: List[Landmark] = []
        self._lock = threading.Lock()

    def add_landmark(
        self,
        start_pos: int,
        end_pos: int,
        landmark_type: str = "generic",
    ):
        """Add a landmark range."""
        with self._lock:
            landmark = Landmark(
                start_pos=start_pos,
                end_pos=end_pos,
                landmark_type=landmark_type,
            )

            if len(self.landmarks) >= self.max_landmarks:
                # Remove oldest non-system landmark
                for i, lm in enumerate(self.landmarks):
                    if lm.landmark_type != "system":
                        del self.landmarks[i]
                        break
                else:
                    # All system landmarks, remove oldest
                    self.landmarks.pop(0)

            self.landmarks.append(landmark)

    def is_landmark_position(self, pos: int) -> bool:
        """Check if position is within any landmark."""
        with self._lock:
            return any(lm.start_pos <= pos < lm.end_pos for lm in self.landmarks)

    def get_landmark_ranges(self) -> List[Tuple[int, int]]:
        """Get all landmark ranges."""
        with self._lock:
            return [(lm.start_pos, lm.end_pos) for lm in self.landmarks]

    def clear(self):
        """Clear all landmarks."""
        with self._lock:
            self.landmarks.clear()


class PerLayerKVCache:
    """
    KV cache for a single layer with paging support.

    Memory layout:
    - VRAM: Recent tokens (window_size)
    - RAM: Older tokens in pages
    """

    def __init__(
        self,
        layer_idx: int,
        config: KVCacheConfig,
        model_config: Union[MixtralConfig, GLM4AirConfig],
        landmark_tracker: LandmarkTracker,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.layer_idx = layer_idx
        self.config = config
        self.model_config = model_config
        self.landmark_tracker = landmark_tracker
        self.device = device
        self.dtype = dtype

        # Dimensions
        self.num_kv_heads = model_config.num_kv_heads
        self.head_dim = model_config.head_dim
        self.window_size = config.vram_window_tokens
        self.page_size = config.page_size_tokens

        # KV dtype (int8 for compression)
        self.kv_dtype = torch.int8 if config.use_int8_kv else dtype

        # VRAM buffer for recent tokens
        # Shape: (window_size, num_kv_heads, head_dim)
        self.k_vram: Optional[torch.Tensor] = None
        self.v_vram: Optional[torch.Tensor] = None

        # Scale factors for int8 quantization
        self.k_scale: Optional[torch.Tensor] = None
        self.v_scale: Optional[torch.Tensor] = None

        # RAM pages: page_id -> KVPage
        self.ram_pages: Dict[int, KVPage] = {}
        self.next_page_id = 0

        # Position tracking
        self.vram_start_pos = 0  # First position in VRAM window
        self.total_length = 0    # Total cached length

        # Lock
        self._lock = threading.RLock()

    def allocate_vram(self):
        """Pre-allocate VRAM buffer."""
        with self._lock:
            self.k_vram = torch.zeros(
                (self.window_size, self.num_kv_heads, self.head_dim),
                dtype=self.kv_dtype,
                device=self.device,
            )
            self.v_vram = torch.zeros(
                (self.window_size, self.num_kv_heads, self.head_dim),
                dtype=self.kv_dtype,
                device=self.device,
            )

            if self.config.use_int8_kv:
                self.k_scale = torch.ones(
                    (self.window_size, self.num_kv_heads, 1),
                    dtype=torch.float16,
                    device=self.device,
                )
                self.v_scale = torch.ones(
                    (self.window_size, self.num_kv_heads, 1),
                    dtype=torch.float16,
                    device=self.device,
                )

    def append(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
    ):
        """
        Append new KV entries.

        Args:
            k: (seq_len, num_kv_heads, head_dim) - key tensor
            v: (seq_len, num_kv_heads, head_dim) - value tensor
            positions: (seq_len,) - position indices
        """
        if self.k_vram is None:
            self.allocate_vram()

        with self._lock:
            seq_len = k.shape[0]
            start_pos = positions[0].item()

            # Check if we need to page out
            vram_used = self.total_length - self.vram_start_pos
            if vram_used + seq_len > self.window_size:
                tokens_to_page = vram_used + seq_len - self.window_size
                self._page_out(tokens_to_page)

            # Quantize if using int8
            if self.config.use_int8_kv:
                k_quant, k_scale = self._quantize_int8(k)
                v_quant, v_scale = self._quantize_int8(v)
            else:
                k_quant, k_scale = k, None
                v_quant, v_scale = v, None

            # Write to VRAM window
            write_start = (start_pos - self.vram_start_pos) % self.window_size
            write_end = write_start + seq_len

            if write_end <= self.window_size:
                # Contiguous write
                self.k_vram[write_start:write_end] = k_quant
                self.v_vram[write_start:write_end] = v_quant
                if k_scale is not None:
                    self.k_scale[write_start:write_end] = k_scale
                    self.v_scale[write_start:write_end] = v_scale
            else:
                # Wrap around
                first_chunk = self.window_size - write_start
                self.k_vram[write_start:] = k_quant[:first_chunk]
                self.v_vram[write_start:] = v_quant[:first_chunk]
                self.k_vram[:write_end - self.window_size] = k_quant[first_chunk:]
                self.v_vram[:write_end - self.window_size] = v_quant[first_chunk:]

            self.total_length = max(self.total_length, start_pos + seq_len)

    def _quantize_int8(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to int8 with per-head scaling."""
        # Per-head absmax scaling
        absmax = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        scale = absmax / 127.0
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale.to(torch.float16)

    def _dequantize_int8(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize from int8."""
        return quantized.to(self.dtype) * scale

    def _page_out(self, num_tokens: int):
        """Page out oldest tokens from VRAM to RAM."""
        # Round up to page boundary
        num_pages = (num_tokens + self.page_size - 1) // self.page_size
        tokens_to_page = num_pages * self.page_size

        for _ in range(num_pages):
            page_start = self.vram_start_pos
            page_end = page_start + self.page_size

            # Check if this is a landmark page
            is_landmark = self.landmark_tracker.is_landmark_position(page_start)

            # Extract page data
            vram_idx_start = page_start % self.window_size
            vram_idx_end = vram_idx_start + self.page_size

            if vram_idx_end <= self.window_size:
                k_page = self.k_vram[vram_idx_start:vram_idx_end].cpu()
                v_page = self.v_vram[vram_idx_start:vram_idx_end].cpu()
            else:
                # Handle wrap
                k_page = torch.cat([
                    self.k_vram[vram_idx_start:],
                    self.k_vram[:vram_idx_end - self.window_size]
                ]).cpu()
                v_page = torch.cat([
                    self.v_vram[vram_idx_start:],
                    self.v_vram[:vram_idx_end - self.window_size]
                ]).cpu()

            # Create RAM page
            page = KVPage(
                page_id=self.next_page_id,
                layer_idx=self.layer_idx,
                start_pos=page_start,
                end_pos=page_end,
                k_cache=k_page,
                v_cache=v_page,
                is_landmark=is_landmark,
            )
            self.ram_pages[self.next_page_id] = page
            self.next_page_id += 1

            self.vram_start_pos = page_end

        # Evict old non-landmark pages if too many
        self._evict_old_pages()

    def _evict_old_pages(self):
        """Evict oldest non-landmark pages if over limit."""
        max_pages = self.config.max_ram_pages

        while len(self.ram_pages) > max_pages:
            # Find oldest non-landmark page
            oldest_id = None
            oldest_time = float('inf')

            for page_id, page in self.ram_pages.items():
                if not page.is_landmark and page.last_access < oldest_time:
                    oldest_time = page.last_access
                    oldest_id = page_id

            if oldest_id is None:
                # All pages are landmarks, remove oldest landmark
                oldest_id = min(self.ram_pages.keys())

            del self.ram_pages[oldest_id]

    def get_kv(
        self,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get KV for given positions.

        Returns:
            k, v tensors on GPU, dequantized if needed
        """
        with self._lock:
            seq_len = positions.shape[0]
            k_out = torch.empty(
                (seq_len, self.num_kv_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            v_out = torch.empty_like(k_out)

            for i, pos in enumerate(positions.tolist()):
                if pos >= self.vram_start_pos:
                    # In VRAM window
                    idx = (pos - self.vram_start_pos) % self.window_size
                    if self.config.use_int8_kv:
                        k_out[i] = self._dequantize_int8(
                            self.k_vram[idx], self.k_scale[idx]
                        )
                        v_out[i] = self._dequantize_int8(
                            self.v_vram[idx], self.v_scale[idx]
                        )
                    else:
                        k_out[i] = self.k_vram[idx]
                        v_out[i] = self.v_vram[idx]
                else:
                    # Find in RAM pages
                    k_val, v_val = self._fetch_from_ram(pos)
                    k_out[i] = k_val.to(self.device)
                    v_out[i] = v_val.to(self.device)

            return k_out, v_out

    def _fetch_from_ram(self, pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch KV from RAM page."""
        for page in self.ram_pages.values():
            if page.start_pos <= pos < page.end_pos:
                page.last_access = time.time()
                idx = pos - page.start_pos
                return page.k_cache[idx], page.v_cache[idx]

        raise ValueError(f"Position {pos} not found in KV cache")

    @property
    def length(self) -> int:
        """Current cached length."""
        return self.total_length

    def clear(self):
        """Clear all cached data."""
        with self._lock:
            self.ram_pages.clear()
            self.vram_start_pos = 0
            self.total_length = 0
            self.next_page_id = 0


class KVPageManager:
    """
    Manages KV cache across all layers.

    Features:
    - Paging from VRAM to RAM for long contexts
    - INT8 quantization for VRAM efficiency
    - Landmark pinning for agentic workloads
    """

    def __init__(
        self,
        model_config: Union[MixtralConfig, GLM4AirConfig],
        kv_config: KVCacheConfig,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ):
        self.model_config = model_config
        self.kv_config = kv_config
        self.device = device
        self.dtype = dtype

        # Landmark tracker (shared across layers)
        self.landmark_tracker = LandmarkTracker(
            max_landmarks=kv_config.max_landmarks
        )

        # Per-layer caches
        self.layer_caches: List[PerLayerKVCache] = []
        for layer_idx in range(model_config.num_layers):
            cache = PerLayerKVCache(
                layer_idx=layer_idx,
                config=kv_config,
                model_config=model_config,
                landmark_tracker=self.landmark_tracker,
                device=device,
                dtype=dtype,
            )
            self.layer_caches.append(cache)

    def allocate(self):
        """Pre-allocate VRAM for all layers."""
        for cache in self.layer_caches:
            cache.allocate_vram()

    def append(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
    ):
        """Append KV to a layer's cache."""
        self.layer_caches[layer_idx].append(k, v, positions)

    def get_kv(
        self,
        layer_idx: int,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get KV from a layer's cache."""
        return self.layer_caches[layer_idx].get_kv(positions)

    def add_landmark(
        self,
        start_pos: int,
        end_pos: int,
        landmark_type: str = "tool_call",
    ):
        """Add a landmark range to preserve."""
        self.landmark_tracker.add_landmark(start_pos, end_pos, landmark_type)

    def clear(self):
        """Clear all caches."""
        for cache in self.layer_caches:
            cache.clear()
        self.landmark_tracker.clear()

    @property
    def length(self) -> int:
        """Current cached length (same for all layers)."""
        if self.layer_caches:
            return self.layer_caches[0].length
        return 0

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_ram_pages = sum(len(c.ram_pages) for c in self.layer_caches)
        landmark_count = len(self.landmark_tracker.landmarks)

        return {
            "total_length": self.length,
            "vram_window": self.kv_config.vram_window_tokens,
            "total_ram_pages": total_ram_pages,
            "landmark_count": landmark_count,
            "max_context": self.kv_config.total_context,
        }
