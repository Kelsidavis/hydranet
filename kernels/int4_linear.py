"""
INT4 Linear layer for non-expert weights.

Stores weights in INT4 with per-group scales, dequantizes on-the-fly.
Reduces VRAM by ~75% compared to fp16 with some quality loss.

Use for attention projections and other non-critical weights.
Keep lm_head in higher precision for output quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .int4_gemm import int4_linear


class Int4Linear(nn.Module):
    """
    Linear layer with INT4 weights and fp16 computation.

    Quantization: per-group absmax scaling (default group_size=128)
    Forward: dequant to fp16 in Triton kernel -> matmul -> add bias

    Memory: weight is int4 (0.5 bytes) + scale per group (2 bytes / group_size)
    vs fp16: 2 bytes per weight
    Savings: ~75% for large layers
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Ensure in_features is divisible by group_size
        assert in_features % group_size == 0, f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        # Ensure in_features is even for packing
        assert in_features % 2 == 0, f"in_features ({in_features}) must be even for INT4 packing"

        num_groups = in_features // group_size

        # INT4 packed weights: (out_features, in_features // 2) as uint8
        self.register_buffer(
            "weight_packed",
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8, device=device)
        )
        # Per-group scales: (out_features, num_groups) as fp16
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, num_groups, dtype=torch.float16, device=device)
        )

        # Bias stays fp16
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16, device=device))
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def quantize_weight(
        weight: torch.Tensor,
        group_size: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize fp16/fp32 weight to INT4 with per-group scales.

        Args:
            weight: (out_features, in_features) in fp16/fp32
            group_size: number of elements per quantization group

        Returns:
            weight_packed: (out_features, in_features // 2) in uint8
            scale: (out_features, num_groups) in fp16
        """
        out_features, in_features = weight.shape
        assert in_features % group_size == 0
        num_groups = in_features // group_size

        # Reshape to (out_features, num_groups, group_size)
        weight_grouped = weight.view(out_features, num_groups, group_size)

        # Per-group absmax
        absmax = weight_grouped.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
        # Scale to map [-absmax, absmax] to [-8, 7] (INT4 range)
        scale = (absmax / 7.0).squeeze(2).to(torch.float16)

        # Quantize to INT4 range [-8, 7], then shift to [0, 15] for packing
        weight_int4 = (weight_grouped / absmax * 7.0).round().clamp(-8, 7).to(torch.int8)
        weight_uint4 = (weight_int4 + 8).to(torch.uint8)  # Shift to [0, 15]

        # Reshape back to (out_features, in_features)
        weight_uint4 = weight_uint4.view(out_features, in_features)

        # Pack two INT4 values into one uint8
        # Even indices go to low nibble, odd indices go to high nibble
        weight_packed = (weight_uint4[:, 0::2] & 0xF) | ((weight_uint4[:, 1::2] & 0xF) << 4)

        return weight_packed, scale

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
        device: Optional[torch.device] = None,
    ) -> "Int4Linear":
        """
        Convert a fp16/fp32 Linear layer to Int4Linear.

        Args:
            linear: nn.Linear to convert
            group_size: quantization group size
            device: target device (default: same as input)
        """
        device = device or linear.weight.device
        in_features = linear.in_features

        # Pad in_features to be divisible by group_size if needed
        if in_features % group_size != 0:
            # For simplicity, fall back to a compatible group size
            group_size = 64
            if in_features % group_size != 0:
                group_size = 32
                if in_features % group_size != 0:
                    raise ValueError(f"in_features ({in_features}) must be divisible by a power-of-2 group size")

        int4_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            device=device,
        )

        # Quantize weights
        weight_packed, scale = cls.quantize_weight(linear.weight.data, group_size)
        int4_linear.weight_packed.copy_(weight_packed)
        int4_linear.weight_scale.copy_(scale)

        # Copy bias
        if linear.bias is not None:
            int4_linear.bias.data.copy_(linear.bias.data.to(torch.float16))

        return int4_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with on-the-fly INT4 dequantization.

        Args:
            x: (*, in_features) input tensor

        Returns:
            (*, out_features) output tensor
        """
        out = int4_linear(x, self.weight_packed, self.weight_scale, self.group_size)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, dtype=int4, group_size={self.group_size}"


def convert_model_to_int4(
    model: nn.Module,
    skip_layers: Optional[list[str]] = None,
    group_size: int = 128,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Convert Linear layers in a model to INT4.

    Args:
        model: Model to convert
        skip_layers: List of layer name patterns to skip (e.g., ["lm_head", "embed"])
        group_size: Quantization group size
        device: Target device

    Returns:
        Model with INT4 weights (modified in place)
    """
    skip_layers = skip_layers or []

    def should_skip(name: str) -> bool:
        return any(skip in name for skip in skip_layers)

    converted = 0
    skipped = 0

    for name, module in list(model.named_modules()):
        if should_skip(name):
            skipped += 1
            continue

        if isinstance(module, nn.Linear):
            # Check if conversion is possible
            if module.in_features % 2 != 0:
                print(f"  Skipping {name}: in_features ({module.in_features}) not even")
                skipped += 1
                continue

            # Find a compatible group size
            gs = group_size
            while gs > 1 and module.in_features % gs != 0:
                gs = gs // 2
            if gs < 32:
                print(f"  Skipping {name}: no compatible group size for in_features={module.in_features}")
                skipped += 1
                continue

            # Get parent module and attr name
            parts = name.rsplit(".", 1)
            if len(parts) == 1:
                parent = model
                attr = parts[0]
            else:
                parent = model.get_submodule(parts[0])
                attr = parts[1]

            # Convert
            int4_layer = Int4Linear.from_float(module, group_size=gs, device=device)
            setattr(parent, attr, int4_layer)
            converted += 1

    print(f"  Converted {converted} layers to INT4, skipped {skipped}")
    return model


def estimate_int4_memory(model: nn.Module, group_size: int = 128) -> dict:
    """
    Estimate memory usage with INT4 vs fp16.

    Returns:
        Dict with fp16_mb, int4_mb, savings_mb
    """
    fp16_bytes = 0
    int4_bytes = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        fp16_bytes += numel * 2  # fp16 = 2 bytes

        # INT4 + scale overhead
        if param.dim() == 2:
            out_features, in_features = param.shape
            # INT4 packed: 0.5 bytes per weight
            int4_bytes += numel // 2
            # Scales: fp16 per group
            num_groups = (in_features + group_size - 1) // group_size
            int4_bytes += out_features * num_groups * 2
        else:
            # Small tensors stay fp16
            int4_bytes += numel * 2

    return {
        "fp16_mb": fp16_bytes / 1e6,
        "int4_mb": int4_bytes / 1e6,
        "savings_mb": (fp16_bytes - int4_bytes) / 1e6,
    }
