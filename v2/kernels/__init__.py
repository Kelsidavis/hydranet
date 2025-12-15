"""
Triton kernels for optimized operations.

Implemented:
- expert_mlp: Fused SwiGLU MLP with INT4 weights
- int4_gemm: On-the-fly INT4 dequant GEMM (no fp16 materialization)

To be implemented:
- Paged attention kernel
- Async H2D copy primitives
- INT8 KV quantization kernels
"""

from .expert_mlp import ExpertMLPTriton, expert_mlp_int4
from .int4_gemm import int4_linear, expert_mlp_int4_fused

__all__ = [
    "ExpertMLPTriton",
    "expert_mlp_int4",
    "int4_linear",
    "expert_mlp_int4_fused",
]
