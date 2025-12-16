"""
Triton kernels for optimized operations.

Implemented:
- expert_mlp: Fused SwiGLU MLP with INT4 weights
- int4_gemm: On-the-fly INT4 dequant GEMM (no fp16 materialization)
- int8_linear: INT8 linear/embedding layers for non-expert weights

To be implemented:
- Paged attention kernel
- Async H2D copy primitives
- INT8 KV quantization kernels
"""

from .expert_mlp import ExpertMLPTriton, expert_mlp_int4
from .int4_gemm import int4_linear, expert_mlp_int4_fused
from .int8_linear import Int8Linear, Int8Embedding, convert_model_to_int8, estimate_int8_memory
from .int8_gemm import int8_linear_fused, int8_linear
from .int4_linear import Int4Linear, convert_model_to_int4, estimate_int4_memory

__all__ = [
    "ExpertMLPTriton",
    "expert_mlp_int4",
    "int4_linear",
    "expert_mlp_int4_fused",
    "Int8Linear",
    "Int8Embedding",
    "convert_model_to_int8",
    "estimate_int8_memory",
    "int8_linear_fused",
    "int8_linear",
    "Int4Linear",
    "convert_model_to_int4",
    "estimate_int4_memory",
]
