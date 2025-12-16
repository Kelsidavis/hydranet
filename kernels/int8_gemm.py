"""
Fused INT8 GEMM with on-the-fly dequantization.

This kernel reads INT8 weights and per-output-channel scales directly,
dequantizing in registers during the matmul. No fp16 weight materialization.

Memory savings: Avoids ~32MB temporary for each 4096x4096 linear layer.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def int8_matmul_kernel(
    # Pointers
    x_ptr,              # [M, K] fp16 input
    w_int8_ptr,         # [N, K] int8 weights
    w_scales_ptr,       # [N] fp16 per-output-channel scales
    bias_ptr,           # [N] fp16 bias (can be null)
    out_ptr,            # [M, N] fp16 output
    # Dimensions
    M, N, K,
    # Strides for x [M, K]
    stride_xm, stride_xk,
    # Strides for weights [N, K]
    stride_wn, stride_wk,
    # Strides for output [M, N]
    stride_om, stride_on,
    # Flags
    HAS_BIAS: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Compute out = x @ W.T + bias where W is stored as INT8 with per-channel scales.

    W_int8[n, k] is the int8 weight value.
    W_scales[n] is the fp16 scale for output channel n.

    Dequantization: W_fp16[n, k] = W_int8[n, k] * scale[n]
    """
    # Program ID with swizzling for better L2 cache utilization
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block starting positions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load scales for this output block [BLOCK_N]
    scale_ptrs = w_scales_ptr + offs_n
    scales = tl.load(scale_ptrs, mask=offs_n < N, other=1.0)

    # Accumulator in fp32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load x block [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load w_int8 block [BLOCK_N, BLOCK_K] and convert to fp16
        w_ptrs = w_int8_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w_int8 = tl.load(w_ptrs, mask=w_mask, other=0)

        # Dequantize: w_fp16 = w_int8 * scale (broadcast scale across K)
        w_fp16 = w_int8.to(tl.float16) * scales[:, None]

        # Accumulate: acc += x @ w.T
        # x is [BLOCK_M, BLOCK_K], w_fp16 is [BLOCK_N, BLOCK_K]
        # We want [BLOCK_M, BLOCK_N], so acc += x @ w_fp16.T
        acc += tl.dot(x, tl.trans(w_fp16))

    # Convert accumulator to fp16
    acc = acc.to(tl.float16)

    # Add bias if present
    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n
        bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def int8_linear_fused(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """
    Fused INT8 linear: y = x @ W.T * scale + bias

    Args:
        x: [*, K] input tensor (fp16)
        weight_int8: [N, K] INT8 weights
        weight_scale: [N] per-output-channel scales (fp16)
        bias: [N] optional bias (fp16)

    Returns:
        [*, N] output tensor (fp16)
    """
    # Flatten batch dimensions
    orig_shape = x.shape
    x_flat = x.view(-1, x.shape[-1])
    M, K = x_flat.shape
    N = weight_int8.shape[0]

    # Ensure contiguous and correct dtype
    x_flat = x_flat.contiguous()
    weight_int8 = weight_int8.contiguous()
    weight_scale = weight_scale.contiguous()

    # Output tensor
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    int8_matmul_kernel[grid](
        x_flat, weight_int8, weight_scale,
        bias if bias is not None else x_flat,  # dummy pointer if no bias
        out,
        M, N, K,
        x_flat.stride(0), x_flat.stride(1),
        weight_int8.stride(0), weight_int8.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=(bias is not None),
    )

    # Restore batch dimensions
    return out.view(*orig_shape[:-1], N)


# Convenience wrapper matching Int8Linear interface
def int8_linear(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """Drop-in replacement for Int8Linear.forward()"""
    return int8_linear_fused(x, weight_int8, weight_scale, bias)
