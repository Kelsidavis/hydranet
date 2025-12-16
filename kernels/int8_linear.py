"""
INT8 Linear layer for non-expert weights.

Stores weights in INT8 with per-channel scales, dequantizes to fp16 on forward.
Reduces VRAM by ~50% compared to fp16 with minimal quality loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Int8Linear(nn.Module):
    """
    Linear layer with INT8 weights and fp16 computation.

    Quantization: per-output-channel absmax scaling
    Forward: dequant to fp16 -> matmul -> add bias

    Memory: weight is int8 (1 byte) + scale per output channel (2 bytes)
    vs fp16: 2 bytes per weight
    Savings: ~50% for large layers
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # INT8 weights: (out_features, in_features)
        self.register_buffer(
            "weight_int8",
            torch.zeros(out_features, in_features, dtype=torch.int8, device=device)
        )
        # Per-output-channel scales: (out_features,)
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, dtype=torch.float16, device=device)
        )

        # Bias stays fp16
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16, device=device))
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize fp16/fp32 weight to INT8 with per-channel scales.

        Args:
            weight: (out_features, in_features) in fp16/fp32

        Returns:
            weight_int8: (out_features, in_features) in int8
            scale: (out_features,) in fp16
        """
        # Per-output-channel absmax
        absmax = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = (absmax / 127.0).squeeze(1).to(torch.float16)

        # Quantize
        weight_int8 = (weight / absmax * 127.0).round().clamp(-128, 127).to(torch.int8)

        return weight_int8, scale

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        device: Optional[torch.device] = None,
    ) -> "Int8Linear":
        """
        Convert a fp16/fp32 Linear layer to Int8Linear.

        Args:
            linear: nn.Linear to convert
            device: target device (default: same as input)
        """
        device = device or linear.weight.device

        int8_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=device,
        )

        # Quantize weights
        weight_int8, scale = cls.quantize_weight(linear.weight.data)
        int8_linear.weight_int8.copy_(weight_int8)
        int8_linear.weight_scale.copy_(scale)

        # Copy bias
        if linear.bias is not None:
            int8_linear.bias.data.copy_(linear.bias.data.to(torch.float16))

        return int8_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with on-the-fly dequantization.

        Args:
            x: (*, in_features) input tensor

        Returns:
            (*, out_features) output tensor
        """
        # Dequantize weight: int8 * scale -> fp16
        # weight_int8: (out, in), scale: (out,)
        weight_fp16 = self.weight_int8.to(x.dtype) * self.weight_scale.unsqueeze(1)

        # Standard linear
        return F.linear(x, weight_fp16, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, dtype=int8"


class Int8Embedding(nn.Module):
    """
    Embedding layer with INT8 weights.

    For GLM4's 151k vocab × 4096 dim embedding:
    - fp16: 1.2 GB
    - int8: 0.6 GB
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # INT8 weights: (num_embeddings, embedding_dim)
        self.register_buffer(
            "weight_int8",
            torch.zeros(num_embeddings, embedding_dim, dtype=torch.int8, device=device)
        )
        # Per-row scales: (num_embeddings,)
        self.register_buffer(
            "weight_scale",
            torch.ones(num_embeddings, dtype=torch.float16, device=device)
        )

    @classmethod
    def from_float(
        cls,
        embedding: nn.Embedding,
        device: Optional[torch.device] = None,
    ) -> "Int8Embedding":
        """Convert fp16/fp32 Embedding to Int8Embedding."""
        device = device or embedding.weight.device

        int8_emb = cls(
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
            device=device,
        )

        # Quantize per-row
        weight = embedding.weight.data
        absmax = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = (absmax / 127.0).squeeze(1).to(torch.float16)
        weight_int8 = (weight / absmax * 127.0).round().clamp(-128, 127).to(torch.int8)

        int8_emb.weight_int8.copy_(weight_int8)
        int8_emb.weight_scale.copy_(scale)

        return int8_emb

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Embedding lookup with dequantization.

        Args:
            input: (*, ) integer indices

        Returns:
            (*, embedding_dim) embeddings in fp16
        """
        # Gather int8 embeddings
        emb_int8 = F.embedding(input, self.weight_int8)  # (*, dim) int8

        # Gather scales
        scales = F.embedding(input, self.weight_scale.unsqueeze(1))  # (*, 1)

        # Dequantize
        return emb_int8.to(torch.float16) * scales

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, dtype=int8"


def convert_model_to_int8(
    model: nn.Module,
    skip_layers: Optional[list[str]] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Convert all Linear and Embedding layers in a model to INT8.

    Args:
        model: Model to convert
        skip_layers: List of layer name patterns to skip (e.g., ["lm_head"])
        device: Target device

    Returns:
        Model with INT8 weights (modified in place)
    """
    skip_layers = skip_layers or []

    def should_skip(name: str) -> bool:
        return any(skip in name for skip in skip_layers)

    # Convert Linear layers
    for name, module in list(model.named_modules()):
        if should_skip(name):
            continue

        if isinstance(module, nn.Linear):
            # Get parent module and attr name
            parts = name.rsplit(".", 1)
            if len(parts) == 1:
                parent = model
                attr = parts[0]
            else:
                parent = model.get_submodule(parts[0])
                attr = parts[1]

            # Convert
            int8_linear = Int8Linear.from_float(module, device=device)
            setattr(parent, attr, int8_linear)

        elif isinstance(module, nn.Embedding):
            parts = name.rsplit(".", 1)
            if len(parts) == 1:
                parent = model
                attr = parts[0]
            else:
                parent = model.get_submodule(parts[0])
                attr = parts[1]

            int8_emb = Int8Embedding.from_float(module, device=device)
            setattr(parent, attr, int8_emb)

    return model


def estimate_int8_memory(model: nn.Module) -> dict:
    """
    Estimate memory usage with INT8 vs fp16.

    Returns:
        Dict with fp16_mb, int8_mb, savings_mb
    """
    fp16_bytes = 0
    int8_bytes = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        fp16_bytes += numel * 2  # fp16 = 2 bytes

        # INT8 + scale overhead
        if param.dim() >= 2:
            # Scale per output channel
            int8_bytes += numel * 1 + param.shape[0] * 2
        else:
            # Small tensors stay fp16
            int8_bytes += numel * 2

    return {
        "fp16_mb": fp16_bytes / 1e6,
        "int8_mb": int8_bytes / 1e6,
        "savings_mb": (fp16_bytes - int8_bytes) / 1e6,
    }
