"""
Packed Expert Store for INT4 weights with one-blob-per-expert layout.

File format:
- experts.bin: All experts concatenated as contiguous blobs
- experts.idx: JSON index with offsets, sizes, metadata

Runtime flow:
1. PackedExpertStore memory-maps experts.bin (or loads to RAM)
2. On cache miss: copy blob to pinned staging buffer
3. Async H2D copy to GPU slot
4. GPU slot provides dequantized fp16 weights for compute

Benefits:
- Single file handle (no open/close overhead per expert)
- Contiguous blob per expert (optimal for PCIe4 DMA)
- Memory-mapped access (OS handles paging)
- Pinned staging ring buffer for double-buffered H2D
"""

import torch
import mmap
import os
import json
import struct
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import threading
import queue

# Suppress torch.frombuffer warning for read-only buffers (we only read from them)
warnings.filterwarnings(
    "ignore",
    message=r"The given buffer is not writable.*",
    category=UserWarning,
    module=r"torch\..*",
)

from ..config import MixtralConfig


@dataclass
class ExpertIndex:
    """Index entry for a single expert."""
    layer_idx: int
    expert_idx: int
    offset: int  # Byte offset in experts.bin
    size: int    # Blob size in bytes
    hidden_dim: int
    intermediate_dim: int
    group_size: int


@dataclass
class BlobLayout:
    """Memory layout within a single expert blob."""
    # Packed INT4 weights (uint8)
    gate_packed_offset: int = 0
    gate_packed_size: int = 0

    # Scales (fp16)
    gate_scales_offset: int = 0
    gate_scales_size: int = 0

    up_packed_offset: int = 0
    up_packed_size: int = 0
    up_scales_offset: int = 0
    up_scales_size: int = 0

    down_packed_offset: int = 0
    down_packed_size: int = 0
    down_scales_offset: int = 0
    down_scales_size: int = 0

    @classmethod
    def from_dims(cls, hidden_dim: int, intermediate_dim: int, group_size: int) -> "BlobLayout":
        """Calculate layout from dimensions."""
        num_groups_hidden = hidden_dim // group_size
        num_groups_intermediate = intermediate_dim // group_size

        layout = cls()

        # Gate projection: [intermediate_dim, hidden_dim]
        layout.gate_packed_offset = 0
        layout.gate_packed_size = intermediate_dim * (hidden_dim // 2)  # INT4 packed
        layout.gate_scales_offset = layout.gate_packed_size
        layout.gate_scales_size = intermediate_dim * num_groups_hidden * 2  # fp16

        # Up projection: [intermediate_dim, hidden_dim]
        layout.up_packed_offset = layout.gate_scales_offset + layout.gate_scales_size
        layout.up_packed_size = intermediate_dim * (hidden_dim // 2)
        layout.up_scales_offset = layout.up_packed_offset + layout.up_packed_size
        layout.up_scales_size = intermediate_dim * num_groups_hidden * 2

        # Down projection: [hidden_dim, intermediate_dim]
        layout.down_packed_offset = layout.up_scales_offset + layout.up_scales_size
        layout.down_packed_size = hidden_dim * (intermediate_dim // 2)
        layout.down_scales_offset = layout.down_packed_offset + layout.down_packed_size
        layout.down_scales_size = hidden_dim * num_groups_intermediate * 2

        return layout

    @property
    def total_size(self) -> int:
        """Total blob size in bytes."""
        return self.down_scales_offset + self.down_scales_size


class GpuExpertSlot:
    """
    Holds a single expert on GPU with INT4 weights.

    Provides on-demand fp16 dequantization for compute.
    """

    def __init__(
        self,
        slot_idx: int,
        hidden_dim: int,
        intermediate_dim: int,
        group_size: int,
        device: torch.device,
    ):
        self.slot_idx = slot_idx
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.group_size = group_size
        self.device = device

        # Current expert loaded
        self.layer_idx: Optional[int] = None
        self.expert_idx: Optional[int] = None

        # Pre-allocated GPU buffers for INT4 weights
        self.gate_packed = torch.empty(
            intermediate_dim, hidden_dim // 2,
            dtype=torch.uint8, device=device
        )
        self.gate_scales = torch.empty(
            intermediate_dim, hidden_dim // group_size,
            dtype=torch.float16, device=device
        )

        self.up_packed = torch.empty(
            intermediate_dim, hidden_dim // 2,
            dtype=torch.uint8, device=device
        )
        self.up_scales = torch.empty(
            intermediate_dim, hidden_dim // group_size,
            dtype=torch.float16, device=device
        )

        self.down_packed = torch.empty(
            hidden_dim, intermediate_dim // 2,
            dtype=torch.uint8, device=device
        )
        self.down_scales = torch.empty(
            hidden_dim, intermediate_dim // group_size,
            dtype=torch.float16, device=device
        )

        # Cached dequantized fp16 weights (optional, for fp16 mode)
        self._cached_fp16: Optional[Dict[str, torch.Tensor]] = None

        # Reusable async load completion event (for overlap)
        # Pre-allocated to avoid per-load allocation overhead
        if device.type == "cuda":
            self.ready_event: torch.cuda.Event = torch.cuda.Event()
        else:
            self.ready_event: Optional[torch.cuda.Event] = None

    def load_from_pinned(
        self,
        pinned_buffer: torch.Tensor,
        layout: BlobLayout,
        layer_idx: int,
        expert_idx: int,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        """
        Async load from pinned staging buffer.

        Args:
            pinned_buffer: Contiguous pinned memory with expert blob
            layout: Memory layout within blob
            layer_idx, expert_idx: Expert identification
            stream: CUDA stream for async copy
        """
        self.layer_idx = layer_idx
        self.expert_idx = expert_idx
        self._cached_fp16 = None  # Invalidate cache

        # View pinned buffer as raw bytes
        pinned_bytes = pinned_buffer.view(-1)

        # Copy each tensor with proper reshaping
        def copy_tensor(dst, offset, size, dtype, shape):
            src_flat = pinned_bytes[offset:offset + size].view(dtype)
            src = src_flat.view(shape)
            dst.copy_(src, non_blocking=True)

        # Handle CPU vs CUDA context
        if self.device.type == "cuda":
            ctx = torch.cuda.stream(stream) if stream else torch.cuda.device(self.device)
            ctx_manager = ctx
        else:
            # CPU: no context manager needed
            from contextlib import nullcontext
            ctx_manager = nullcontext()

        with ctx_manager:
            # Gate projection
            copy_tensor(
                self.gate_packed,
                layout.gate_packed_offset,
                layout.gate_packed_size,
                torch.uint8,
                (self.intermediate_dim, self.hidden_dim // 2)
            )
            copy_tensor(
                self.gate_scales,
                layout.gate_scales_offset,
                layout.gate_scales_size,
                torch.float16,
                (self.intermediate_dim, self.hidden_dim // self.group_size)
            )

            # Up projection
            copy_tensor(
                self.up_packed,
                layout.up_packed_offset,
                layout.up_packed_size,
                torch.uint8,
                (self.intermediate_dim, self.hidden_dim // 2)
            )
            copy_tensor(
                self.up_scales,
                layout.up_scales_offset,
                layout.up_scales_size,
                torch.float16,
                (self.intermediate_dim, self.hidden_dim // self.group_size)
            )

            # Down projection
            copy_tensor(
                self.down_packed,
                layout.down_packed_offset,
                layout.down_packed_size,
                torch.uint8,
                (self.hidden_dim, self.intermediate_dim // 2)
            )
            copy_tensor(
                self.down_scales,
                layout.down_scales_offset,
                layout.down_scales_size,
                torch.float16,
                (self.hidden_dim, self.intermediate_dim // self.group_size)
            )

    def dequantize(self) -> Dict[str, torch.Tensor]:
        """
        Dequantize INT4 weights to fp16.

        Returns dict with 'gate_proj', 'up_proj', 'down_proj' as fp16 tensors.
        """
        if self._cached_fp16 is not None:
            return self._cached_fp16

        self._cached_fp16 = {
            "gate_proj": self._dequant_tensor(
                self.gate_packed, self.gate_scales,
                self.intermediate_dim, self.hidden_dim
            ),
            "up_proj": self._dequant_tensor(
                self.up_packed, self.up_scales,
                self.intermediate_dim, self.hidden_dim
            ),
            "down_proj": self._dequant_tensor(
                self.down_packed, self.down_scales,
                self.hidden_dim, self.intermediate_dim
            ),
        }
        return self._cached_fp16

    def _dequant_tensor(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        out_features: int,
        in_features: int,
    ) -> torch.Tensor:
        """Dequantize INT4 packed tensor to fp16."""
        # Unpack int4 values
        even = (packed & 0x0F).to(torch.int8) - 8  # Low nibble
        odd = ((packed >> 4) & 0x0F).to(torch.int8) - 8  # High nibble

        # Interleave back to original order
        unpacked = torch.empty(
            out_features, in_features,
            dtype=torch.float16,
            device=self.device,
        )
        unpacked[:, 0::2] = even.to(torch.float16)
        unpacked[:, 1::2] = odd.to(torch.float16)

        # Apply per-group scales
        num_groups = in_features // self.group_size
        unpacked_grouped = unpacked.view(out_features, num_groups, self.group_size)
        scales_expanded = scales.unsqueeze(-1)

        dequantized = unpacked_grouped * scales_expanded
        # Use contiguous() to ensure independent storage (no view refs)
        return dequantized.view(out_features, in_features).contiguous()

    def size_bytes(self) -> int:
        """Total GPU memory used by this slot."""
        packed = self.gate_packed.numel() + self.up_packed.numel() + self.down_packed.numel()
        scales = (self.gate_scales.numel() + self.up_scales.numel() + self.down_scales.numel()) * 2
        return packed + scales

    def wait_ready(self):
        """
        GPU-side wait for async load completion.

        Does NOT block CPU - only makes current CUDA stream wait on the copy.
        Event is reusable - no need to clear after wait.
        """
        if self.ready_event is not None:
            torch.cuda.current_stream().wait_event(self.ready_event)


class PinnedStagingRing:
    """
    Event-safe pinned staging slots for async H2D overlap.

    Uses free list + inflight event tracking to ensure staging buffers
    are not reused until async H2D copies complete.
    """

    def __init__(
        self,
        num_slots: int,
        slot_size: int,
        device: torch.device,
    ):
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.device = device

        # Allocate pinned memory slots
        use_pinned = torch.cuda.is_available() and device.type == "cuda"
        self.slots: List[torch.Tensor] = []
        for _ in range(num_slots):
            if use_pinned:
                slot = torch.empty(slot_size, dtype=torch.uint8, pin_memory=True)
            else:
                slot = torch.empty(slot_size, dtype=torch.uint8)
            self.slots.append(slot)

        self._lock = threading.Lock()
        self._free = list(range(num_slots))  # Available slot indices
        self._inflight: List[Tuple[torch.cuda.Event, int]] = []  # (event, idx)

    def _reclaim_completed(self):
        """Move completed inflight slots back to free list."""
        if not self._inflight:
            return
        keep = []
        for ev, idx in self._inflight:
            if ev.query():
                self._free.append(idx)
            else:
                keep.append((ev, idx))
        self._inflight = keep

    def acquire(self) -> Tuple[int, torch.Tensor]:
        """Get next available staging slot, waiting if necessary."""
        with self._lock:
            self._reclaim_completed()
            if not self._free:
                # All staging slots busy - wait for oldest to complete
                # (rare backpressure; increase num_slots if this happens often)
                ev, idx = self._inflight.pop(0)
                ev.synchronize()
                self._free.append(idx)

            idx = self._free.pop()
            return idx, self.slots[idx]

    def release(self, idx: int, ev: torch.cuda.Event):
        """Mark slot as inflight until event completes."""
        with self._lock:
            self._inflight.append((ev, idx))


class PackedExpertStore:
    """
    Loads and serves packed INT4 experts from disk.

    Supports:
    - Memory-mapped file access
    - Pinned staging ring buffer
    - Integration with GPU slot manager
    """

    def __init__(
        self,
        index_path: Path,
        bin_path: Optional[Path] = None,
        device: torch.device = torch.device("cuda"),
        use_mmap: bool = True,
        num_staging_slots: int = 4,  # Balance between overlap and memory
    ):
        self.device = device
        self.use_mmap = use_mmap

        # Load index
        with open(index_path) as f:
            index_data = json.load(f)

        self.format_version = index_data["format_version"]
        self.quant_config = index_data["quant_config"]
        self.model_config = index_data["model_config"]

        # Build index lookup: (layer_idx, expert_idx) -> ExpertIndex
        self.index: Dict[Tuple[int, int], ExpertIndex] = {}
        for entry in index_data["experts"]:
            key = (entry["layer_idx"], entry["expert_idx"])
            self.index[key] = ExpertIndex(
                layer_idx=entry["layer_idx"],
                expert_idx=entry["expert_idx"],
                offset=entry["offset"],
                size=entry["size"],
                hidden_dim=self.model_config["hidden_dim"],
                intermediate_dim=self.model_config["intermediate_dim"],
                group_size=self.quant_config["group_size"],
            )

        # Calculate blob layout
        self.layout = BlobLayout.from_dims(
            hidden_dim=self.model_config["hidden_dim"],
            intermediate_dim=self.model_config["intermediate_dim"],
            group_size=self.quant_config["group_size"],
        )

        # Open binary file
        if bin_path is None:
            bin_path = index_path.parent / "experts.bin"

        self.bin_path = bin_path
        self._file = None
        self._mmap = None

        if use_mmap:
            self._file = open(bin_path, "rb")
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            # Hint kernel to read-ahead sequentially
            try:
                self._mmap.madvise(mmap.MADV_SEQUENTIAL)
            except (AttributeError, OSError):
                pass  # madvise not available on all platforms
        else:
            # Load entire file to RAM as memoryview for zero-copy slicing
            with open(bin_path, "rb") as f:
                self._data_bytes = f.read()
                self._data = memoryview(self._data_bytes)

        # Initialize staging ring buffer
        self.staging = PinnedStagingRing(
            num_slots=num_staging_slots,
            slot_size=self.layout.total_size,
            device=device,
        )

        # Dedicated stream for async H2D copies (overlap with compute)
        if torch.cuda.is_available() and device.type == "cuda":
            self.memcpy_stream = torch.cuda.Stream(device=device)
        else:
            self.memcpy_stream = None

        # Statistics
        self.load_count = 0
        self.total_bytes_loaded = 0

    def prefault_expert(self, layer_idx: int, expert_idx: int):
        """
        Hint kernel to prefault pages for an expert.

        Call this before you need the expert to reduce page fault latency.
        Non-blocking - just issues madvise hint.
        """
        key = (layer_idx, expert_idx)
        if key not in self.index or not self.use_mmap:
            return

        entry = self.index[key]
        try:
            # MADV_WILLNEED hints kernel to page in this range
            self._mmap.madvise(mmap.MADV_WILLNEED, entry.offset, entry.size)
        except (AttributeError, OSError):
            pass  # madvise not available or failed

    def get_expert_blob(
        self,
        layer_idx: int,
        expert_idx: int,
    ) -> Tuple[bytes, BlobLayout]:
        """
        Get raw expert blob from storage.

        Args:
            layer_idx: Layer index
            expert_idx: Expert index within layer

        Returns:
            (blob_bytes, layout) tuple
        """
        key = (layer_idx, expert_idx)
        if key not in self.index:
            raise ValueError(f"Expert ({layer_idx}, {expert_idx}) not in index")

        entry = self.index[key]

        self.load_count += 1
        self.total_bytes_loaded += entry.size

        # Return slice directly - no copy! Caller must not hold reference long.
        if self.use_mmap:
            return self._mmap[entry.offset:entry.offset + entry.size], self.layout
        else:
            return self._data[entry.offset:entry.offset + entry.size], self.layout

    def load_to_staging(
        self,
        layer_idx: int,
        expert_idx: int,
    ) -> Tuple[int, torch.Tensor, BlobLayout]:
        """
        Load expert blob to pinned staging buffer.

        Returns:
            (staging_slot_idx, staging_buffer, layout)
        """
        blob_view, layout = self.get_expert_blob(layer_idx, expert_idx)
        slot_idx, staging = self.staging.acquire()

        # Copy to staging buffer - pure torch, zero-copy view of source
        # blob_view is memoryview (mmap or RAM-backed), blob_t is a view (don't cache it)
        blob_t = torch.frombuffer(blob_view, dtype=torch.uint8)
        n = blob_t.numel()
        staging.view(torch.uint8)[:n].copy_(blob_t)

        return slot_idx, staging, layout

    def load_to_gpu_slot(
        self,
        layer_idx: int,
        expert_idx: int,
        gpu_slot: GpuExpertSlot,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Optional[torch.cuda.Event]:
        """
        Load expert from storage to GPU slot with async H2D.

        Args:
            layer_idx, expert_idx: Expert identification
            gpu_slot: Target GPU slot
            stream: CUDA stream for async H2D (defaults to memcpy_stream)

        Returns:
            CUDA event that signals when gpu_slot is ready, or None if CPU-only.
        """
        staging_idx, staging, layout = self.load_to_staging(layer_idx, expert_idx)

        # Default to memcpy stream for overlap
        if stream is None:
            stream = self.memcpy_stream

        # Issue async H2D copy
        gpu_slot.load_from_pinned(
            pinned_buffer=staging,
            layout=layout,
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            stream=stream,
        )

        # Record completion on gpu_slot's reusable event and defer staging reuse
        if stream is not None and gpu_slot.ready_event is not None:
            gpu_slot.ready_event.record(stream)
            self.staging.release(staging_idx, gpu_slot.ready_event)
            return gpu_slot.ready_event
        else:
            # CPU mode - no event needed, staging can be reused immediately
            return None

    def get_stats(self) -> Dict:
        """Get store statistics."""
        return {
            "load_count": self.load_count,
            "total_bytes_loaded": self.total_bytes_loaded,
            "total_mb_loaded": self.total_bytes_loaded / (1024 * 1024),
        }

    def close(self):
        """Close file handles."""
        if hasattr(self, '_mmap') and self._mmap is not None:
            try:
                self._mmap.close()
            except (ValueError, OSError):
                pass  # Already closed
            self._mmap = None

        if hasattr(self, '_file') and self._file is not None:
            try:
                self._file.close()
            except (ValueError, OSError):
                pass  # Already closed
            self._file = None

    def __del__(self):
        self.close()


def pack_to_single_blob(
    input_dir: Path,
    output_dir: Path,
    model_config: MixtralConfig,
    group_size: int = 128,
) -> Dict:
    """
    Pack individual expert files into single experts.bin + experts.idx.

    Args:
        input_dir: Directory with individual expert .bin files
        output_dir: Where to write experts.bin and experts.idx
        model_config: Model configuration
        group_size: Quantization group size

    Returns:
        Index metadata dict
    """
    from ..preprocess.pack_weights import load_packed_expert

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate blob layout
    layout = BlobLayout.from_dims(
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        group_size=group_size,
    )

    index_data = {
        "format_version": "2.0",
        "quant_config": {
            "bits": 4,
            "group_size": group_size,
            "symmetric": True,
        },
        "model_config": {
            "num_layers": model_config.num_layers,
            "num_experts": model_config.num_experts,
            "hidden_dim": model_config.hidden_dim,
            "intermediate_dim": model_config.intermediate_dim,
        },
        "experts": [],
    }

    bin_path = output_dir / "experts.bin"
    current_offset = 0

    print(f"Packing experts to single blob: {bin_path}")

    with open(bin_path, "wb") as bin_file:
        for layer_idx in range(model_config.num_layers):
            for expert_idx in range(model_config.num_experts):
                # Find input file
                expert_file = input_dir / f"expert_L{layer_idx:02d}_E{expert_idx:02d}.bin"
                if not expert_file.exists():
                    raise FileNotFoundError(f"Missing expert file: {expert_file}")

                # Load and repack
                packed = load_packed_expert(expert_file)

                # Write as contiguous blob
                blob = bytearray(layout.total_size)

                # Write packed weights
                gate_packed_bytes = packed.gate_proj_packed.numpy().tobytes()
                blob[layout.gate_packed_offset:layout.gate_packed_offset + layout.gate_packed_size] = gate_packed_bytes

                gate_scales_bytes = packed.gate_proj_scales.numpy().tobytes()
                blob[layout.gate_scales_offset:layout.gate_scales_offset + layout.gate_scales_size] = gate_scales_bytes

                up_packed_bytes = packed.up_proj_packed.numpy().tobytes()
                blob[layout.up_packed_offset:layout.up_packed_offset + layout.up_packed_size] = up_packed_bytes

                up_scales_bytes = packed.up_proj_scales.numpy().tobytes()
                blob[layout.up_scales_offset:layout.up_scales_offset + layout.up_scales_size] = up_scales_bytes

                down_packed_bytes = packed.down_proj_packed.numpy().tobytes()
                blob[layout.down_packed_offset:layout.down_packed_offset + layout.down_packed_size] = down_packed_bytes

                down_scales_bytes = packed.down_proj_scales.numpy().tobytes()
                blob[layout.down_scales_offset:layout.down_scales_offset + layout.down_scales_size] = down_scales_bytes

                # Write blob
                bin_file.write(blob)

                # Record index entry
                index_data["experts"].append({
                    "layer_idx": layer_idx,
                    "expert_idx": expert_idx,
                    "offset": current_offset,
                    "size": layout.total_size,
                })

                current_offset += layout.total_size

                if (layer_idx * model_config.num_experts + expert_idx + 1) % 32 == 0:
                    loaded = layer_idx * model_config.num_experts + expert_idx + 1
                    total = model_config.num_layers * model_config.num_experts
                    print(f"  Packed {loaded}/{total} experts")

    # Write index
    index_path = output_dir / "experts.idx"
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

    total_size = current_offset
    print(f"\nDone! Total size: {total_size / 1e9:.2f} GB")
    print(f"  Index: {index_path}")
    print(f"  Binary: {bin_path}")

    return index_data
