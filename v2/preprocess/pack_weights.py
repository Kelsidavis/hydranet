"""
Offline expert weight packing and quantization.

Converts HuggingFace fp16 Mixtral weights to HydraNet-native INT4 format.

Format: INT4 with per-group scales (group_size=128)
- Weights packed as int4 pairs in uint8
- Scales stored as fp16 per group
- Zero-point assumed 8 (symmetric around 0)

This runs ONCE offline, then runtime only loads packed weights.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import struct


@dataclass
class QuantConfig:
    """Quantization configuration."""
    bits: int = 4
    group_size: int = 128
    symmetric: bool = True  # Symmetric quantization (zero_point = 8 for int4)

    @property
    def zero_point(self) -> int:
        """Zero point for symmetric int4."""
        return 2 ** (self.bits - 1) if self.symmetric else 0


def compute_expert_checksum(weights: Dict[str, torch.Tensor]) -> str:
    """
    Compute lightweight checksum for expert weights.

    Uses sum of absolute values (fast, catches transposition/corruption).
    Returns hex string for JSON serialization.
    """
    total = 0.0
    for name in ["gate_proj", "up_proj", "down_proj"]:
        if name in weights:
            total += weights[name].float().abs().sum().item()
    # Convert to hex representation of float bits
    import struct
    packed = struct.pack('d', total)
    return packed.hex()


def verify_expert_checksum(weights: Dict[str, torch.Tensor], expected: str) -> bool:
    """Verify expert weights match expected checksum."""
    actual = compute_expert_checksum(weights)
    return actual == expected


@dataclass
class PackedExpert:
    """
    Packed expert weights in HydraNet-native format.

    All tensors are contiguous and ready for H2D copy.
    """
    layer_idx: int
    expert_idx: int

    # Packed INT4 weights (stored as uint8, 2 values per byte)
    gate_proj_packed: torch.Tensor  # [intermediate_dim, hidden_dim // 2]
    up_proj_packed: torch.Tensor    # [intermediate_dim, hidden_dim // 2]
    down_proj_packed: torch.Tensor  # [hidden_dim, intermediate_dim // 2]

    # Per-group scales (fp16)
    gate_proj_scales: torch.Tensor  # [intermediate_dim, hidden_dim // group_size]
    up_proj_scales: torch.Tensor    # [intermediate_dim, hidden_dim // group_size]
    down_proj_scales: torch.Tensor  # [hidden_dim, intermediate_dim // group_size]

    # Metadata for kernel dispatch
    hidden_dim: int
    intermediate_dim: int
    group_size: int

    # Checksum for validation (catches mis-indexing)
    checksum: str = ""

    def size_bytes(self) -> int:
        """Total size in bytes."""
        packed_size = (
            self.gate_proj_packed.numel() +
            self.up_proj_packed.numel() +
            self.down_proj_packed.numel()
        )
        scale_size = (
            self.gate_proj_scales.numel() +
            self.up_proj_scales.numel() +
            self.down_proj_scales.numel()
        ) * 2  # fp16
        return packed_size + scale_size

    def to_pinned(self) -> "PackedExpert":
        """Move to pinned memory for fast H2D."""
        return PackedExpert(
            layer_idx=self.layer_idx,
            expert_idx=self.expert_idx,
            gate_proj_packed=self.gate_proj_packed.pin_memory(),
            up_proj_packed=self.up_proj_packed.pin_memory(),
            down_proj_packed=self.down_proj_packed.pin_memory(),
            gate_proj_scales=self.gate_proj_scales.pin_memory(),
            up_proj_scales=self.up_proj_scales.pin_memory(),
            down_proj_scales=self.down_proj_scales.pin_memory(),
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            group_size=self.group_size,
        )


class ExpertWeightPacker:
    """
    Packs fp16 expert weights into INT4 format.

    Usage:
        packer = ExpertWeightPacker(QuantConfig())
        packed = packer.pack_expert(layer_idx, expert_idx, weights)
        packer.save(packed, output_dir)
    """

    def __init__(self, config: QuantConfig):
        self.config = config
        self.group_size = config.group_size
        self.zero_point = config.zero_point

    def quantize_tensor(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize fp16 tensor to INT4 with per-group scales.

        Args:
            tensor: [out_features, in_features] fp16 weight matrix

        Returns:
            packed: [out_features, in_features // 2] uint8 (2 int4 per byte)
            scales: [out_features, in_features // group_size] fp16
        """
        out_features, in_features = tensor.shape
        assert in_features % self.group_size == 0, \
            f"in_features {in_features} must be divisible by group_size {self.group_size}"

        # Reshape for per-group quantization
        num_groups = in_features // self.group_size
        tensor_grouped = tensor.view(out_features, num_groups, self.group_size)

        # Compute per-group scales (absmax / 7 for int4 symmetric)
        absmax = tensor_grouped.abs().amax(dim=-1, keepdim=True)
        scales = absmax / 7.0  # 7 = 2^3 - 1 for int4 symmetric [-7, 7]
        scales = scales.clamp(min=1e-10)  # Avoid division by zero

        # Quantize to int4 range [-8, 7] -> stored as [0, 15]
        quantized = (tensor_grouped / scales).round().clamp(-8, 7)
        quantized = (quantized + 8).to(torch.uint8)  # Shift to [0, 15]

        # Reshape back
        quantized = quantized.view(out_features, in_features)
        scales = scales.squeeze(-1)  # [out_features, num_groups]

        # Pack two int4 values into one uint8
        # Low nibble = even indices, high nibble = odd indices
        assert in_features % 2 == 0
        packed = torch.zeros(
            out_features, in_features // 2,
            dtype=torch.uint8,
            device=tensor.device,
        )

        even = quantized[:, 0::2]  # [out, in//2]
        odd = quantized[:, 1::2]   # [out, in//2]
        packed = (odd << 4) | even  # Pack: high nibble = odd, low nibble = even

        return packed.contiguous(), scales.to(torch.float16).contiguous()

    def pack_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        weights: Dict[str, torch.Tensor],
    ) -> PackedExpert:
        """
        Pack a single expert's weights.

        Args:
            layer_idx: Layer index
            expert_idx: Expert index within layer
            weights: Dict with 'gate_proj', 'up_proj', 'down_proj' fp16 tensors

        Returns:
            PackedExpert with quantized weights
        """
        gate_proj = weights["gate_proj"]  # [intermediate, hidden]
        up_proj = weights["up_proj"]      # [intermediate, hidden]
        down_proj = weights["down_proj"]  # [hidden, intermediate]

        hidden_dim = gate_proj.shape[1]
        intermediate_dim = gate_proj.shape[0]

        # Compute checksum BEFORE quantization (from original fp16)
        checksum = compute_expert_checksum(weights)

        # Quantize each projection
        gate_packed, gate_scales = self.quantize_tensor(gate_proj)
        up_packed, up_scales = self.quantize_tensor(up_proj)
        down_packed, down_scales = self.quantize_tensor(down_proj)

        return PackedExpert(
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            gate_proj_packed=gate_packed,
            up_proj_packed=up_packed,
            down_proj_packed=down_packed,
            gate_proj_scales=gate_scales,
            up_proj_scales=up_scales,
            down_proj_scales=down_scales,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            group_size=self.group_size,
            checksum=checksum,
        )

    def dequantize_tensor(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        out_features: int,
        in_features: int,
    ) -> torch.Tensor:
        """
        Dequantize INT4 tensor back to fp16 (for verification).

        Args:
            packed: [out_features, in_features // 2] uint8
            scales: [out_features, in_features // group_size] fp16
            out_features, in_features: Original dimensions

        Returns:
            [out_features, in_features] fp16 tensor
        """
        # Unpack int4 values
        even = (packed & 0x0F).to(torch.int8) - 8  # Low nibble
        odd = ((packed >> 4) & 0x0F).to(torch.int8) - 8  # High nibble

        # Interleave back
        unpacked = torch.zeros(
            out_features, in_features,
            dtype=torch.int8,
            device=packed.device,
        )
        unpacked[:, 0::2] = even
        unpacked[:, 1::2] = odd

        # Reshape for group dequantization
        num_groups = in_features // self.group_size
        unpacked_grouped = unpacked.view(out_features, num_groups, self.group_size)
        scales_expanded = scales.unsqueeze(-1)  # [out, groups, 1]

        # Dequantize
        dequantized = unpacked_grouped.to(torch.float16) * scales_expanded
        return dequantized.view(out_features, in_features)


def pack_mixtral_experts(
    model_path: str,
    output_dir: str,
    config: Optional[QuantConfig] = None,
) -> Dict[str, any]:
    """
    Pack all Mixtral experts to HydraNet-native format.

    This is the main offline preprocessing function.

    Args:
        model_path: Path to HuggingFace Mixtral checkpoint
        output_dir: Where to save packed weights
        config: Quantization config

    Returns:
        Metadata dict with sizes and checksums
    """
    from ..model.loader import MixtralWeightLoader
    from ..config import MixtralConfig

    if config is None:
        config = QuantConfig()

    model_config = MixtralConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize loader and packer
    loader = MixtralWeightLoader(
        model_path=model_path,
        config=model_config,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    packer = ExpertWeightPacker(config)

    metadata = {
        "format_version": "1.0",
        "quant_config": {
            "bits": config.bits,
            "group_size": config.group_size,
            "symmetric": config.symmetric,
        },
        "model_config": {
            "num_layers": model_config.num_layers,
            "num_experts": model_config.num_experts,
            "hidden_dim": model_config.hidden_dim,
            "intermediate_dim": model_config.intermediate_dim,
        },
        "experts": [],
    }

    total_size = 0

    print(f"Packing {model_config.total_experts} experts to INT4...")
    print(f"Output: {output_path}")

    for layer_idx, expert_idx, weights in loader.iter_experts():
        # Pack expert
        packed = packer.pack_expert(layer_idx, expert_idx, weights)

        # Save as single .bin file per expert
        expert_file = output_path / f"expert_L{layer_idx:02d}_E{expert_idx:02d}.bin"

        with open(expert_file, "wb") as f:
            # Header: layer_idx, expert_idx, hidden_dim, intermediate_dim, group_size
            f.write(struct.pack(
                "IIIII",
                layer_idx,
                expert_idx,
                packed.hidden_dim,
                packed.intermediate_dim,
                packed.group_size,
            ))

            # Write packed weights and scales
            f.write(packed.gate_proj_packed.numpy().tobytes())
            f.write(packed.gate_proj_scales.numpy().tobytes())
            f.write(packed.up_proj_packed.numpy().tobytes())
            f.write(packed.up_proj_scales.numpy().tobytes())
            f.write(packed.down_proj_packed.numpy().tobytes())
            f.write(packed.down_proj_scales.numpy().tobytes())

        expert_size = packed.size_bytes()
        total_size += expert_size

        metadata["experts"].append({
            "layer": layer_idx,
            "expert": expert_idx,
            "file": expert_file.name,
            "size_bytes": expert_size,
            "checksum": packed.checksum,  # For validation at load time
        })

        if (layer_idx * model_config.num_experts + expert_idx + 1) % 32 == 0:
            loaded = layer_idx * model_config.num_experts + expert_idx + 1
            print(f"  Packed {loaded}/{model_config.total_experts} experts "
                  f"({total_size / 1e9:.2f} GB)")

    # Save metadata
    metadata["total_size_bytes"] = total_size
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Total packed size: {total_size / 1e9:.2f} GB")
    print(f"  Original fp16 size: {model_config.total_experts * model_config.expert_size_bytes / 1e9:.2f} GB")
    print(f"  Compression: {model_config.total_experts * model_config.expert_size_bytes / total_size:.1f}x")

    return metadata


def load_packed_expert(
    expert_file: Path,
) -> PackedExpert:
    """
    Load a single packed expert from disk.

    Args:
        expert_file: Path to expert .bin file

    Returns:
        PackedExpert ready for cache manager
    """
    with open(expert_file, "rb") as f:
        # Read header
        header = struct.unpack("IIIII", f.read(20))
        layer_idx, expert_idx, hidden_dim, intermediate_dim, group_size = header

        # Calculate sizes
        num_groups_hidden = hidden_dim // group_size
        num_groups_intermediate = intermediate_dim // group_size

        gate_packed_size = intermediate_dim * (hidden_dim // 2)
        gate_scales_size = intermediate_dim * num_groups_hidden * 2  # fp16
        up_packed_size = gate_packed_size
        up_scales_size = gate_scales_size
        down_packed_size = hidden_dim * (intermediate_dim // 2)
        down_scales_size = hidden_dim * num_groups_intermediate * 2

        # Read tensors
        gate_packed = torch.from_numpy(
            np.frombuffer(f.read(gate_packed_size), dtype=np.uint8)
        ).view(intermediate_dim, hidden_dim // 2)

        gate_scales = torch.from_numpy(
            np.frombuffer(f.read(gate_scales_size), dtype=np.float16)
        ).view(intermediate_dim, num_groups_hidden)

        up_packed = torch.from_numpy(
            np.frombuffer(f.read(up_packed_size), dtype=np.uint8)
        ).view(intermediate_dim, hidden_dim // 2)

        up_scales = torch.from_numpy(
            np.frombuffer(f.read(up_scales_size), dtype=np.float16)
        ).view(intermediate_dim, num_groups_hidden)

        down_packed = torch.from_numpy(
            np.frombuffer(f.read(down_packed_size), dtype=np.uint8)
        ).view(hidden_dim, intermediate_dim // 2)

        down_scales = torch.from_numpy(
            np.frombuffer(f.read(down_scales_size), dtype=np.float16)
        ).view(hidden_dim, num_groups_intermediate)

    return PackedExpert(
        layer_idx=layer_idx,
        expert_idx=expert_idx,
        gate_proj_packed=gate_packed.contiguous(),
        up_proj_packed=up_packed.contiguous(),
        down_proj_packed=down_packed.contiguous(),
        gate_proj_scales=gate_scales.contiguous(),
        up_proj_scales=up_scales.contiguous(),
        down_proj_scales=down_scales.contiguous(),
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        group_size=group_size,
    )
