"""
Fused INT4 GEMM with on-the-fly dequantization.

This kernel reads INT4 packed weights and per-group scales directly,
dequantizing in registers during the matmul. No fp16 weight materialization.

Memory savings: ~350MB per expert (the fp16 weight buffer is eliminated)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def int4_matmul_kernel(
    # Pointers
    x_ptr,              # [M, K] fp16 input
    w_packed_ptr,       # [N, K//2] uint8 packed INT4 weights
    w_scales_ptr,       # [N, K//group_size] fp16 scales
    out_ptr,            # [M, N] fp16 output
    # Dimensions
    M, N, K,
    # Quantization
    group_size,
    # Strides for x
    stride_xm, stride_xk,
    # Strides for packed weights [N, K//2]
    stride_wn, stride_wk,
    # Strides for scales [N, K//group_size]
    stride_sn, stride_sk,
    # Strides for output
    stride_om, stride_on,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Compute out = x @ W.T where W is stored as INT4 with per-group scales.

    W_packed[n, k//2] contains two INT4 values for W[n, k] and W[n, k+1].
    W_scales[n, k//group_size] is the fp16 scale for that group.

    Dequantization: W_fp16[n, k] = (int4_value - 8) * scale[n, k//group_size]
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

    # Accumulator in fp32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load x block [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load packed weights [BLOCK_N, BLOCK_K//2]
        # Each byte has 2 INT4 values
        offs_k_packed = offs_k // 2
        w_ptrs = w_packed_ptr + offs_n[:, None] * stride_wn + offs_k_packed[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (offs_k_packed[None, :] < K // 2)
        w_packed = tl.load(w_ptrs, mask=w_mask, other=0)

        # Unpack INT4: determine which nibble based on k position
        k_is_odd = (offs_k % 2) == 1
        # Low nibble for even k, high nibble for odd k
        w_int4 = tl.where(
            k_is_odd[None, :],
            (w_packed >> 4) & 0xF,
            w_packed & 0xF
        )
        # Convert unsigned 0-15 to signed -8 to 7
        w_int = (w_int4.to(tl.int32) - 8).to(tl.float16)

        # Load scales [BLOCK_N, num_groups_in_block]
        offs_group = offs_k // group_size
        s_ptrs = w_scales_ptr + offs_n[:, None] * stride_sn + offs_group[None, :] * stride_sk
        s_mask = (offs_n[:, None] < N) & (offs_group[None, :] < tl.cdiv(K, group_size))
        scales = tl.load(s_ptrs, mask=s_mask, other=1.0)

        # Dequantize: w_fp16 = w_int * scale
        w = w_int * scales

        # Matmul: acc += x @ w.T
        # x: [BLOCK_M, BLOCK_K], w: [BLOCK_N, BLOCK_K]
        # Result: [BLOCK_M, BLOCK_N]
        acc += tl.dot(x, tl.trans(w))

    # Convert to fp16 and store
    out = acc.to(tl.float16)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, out, mask=out_mask)


def int4_linear(
    x: torch.Tensor,           # [M, K] or [batch, seq, K] fp16
    w_packed: torch.Tensor,    # [N, K//2] uint8
    w_scales: torch.Tensor,    # [N, K//group_size] fp16
    group_size: int = 128,
) -> torch.Tensor:
    """
    INT4 linear layer with on-the-fly dequantization.

    Computes: out = x @ W.T
    where W is stored as INT4 with per-group scales.

    No fp16 weight materialization - dequant happens in the kernel.
    """
    # Handle batched input
    orig_shape = x.shape
    if x.dim() == 3:
        batch, seq, K = x.shape
        x = x.view(batch * seq, K)

    M, K = x.shape
    N = w_packed.shape[0]

    # Validate shapes
    assert w_packed.shape[1] == K // 2, f"w_packed shape {w_packed.shape} vs K={K}"
    assert w_scales.shape == (N, K // group_size), f"w_scales shape {w_scales.shape} vs ({N}, {K // group_size})"

    # Allocate output
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Grid
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    # Launch kernel
    int4_matmul_kernel[grid](
        x, w_packed, w_scales, out,
        M, N, K,
        group_size,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        w_scales.stride(0), w_scales.stride(1),
        out.stride(0), out.stride(1),
    )

    # Restore batch dimensions
    if len(orig_shape) == 3:
        out = out.view(batch, seq, N)

    return out


@triton.jit
def fused_silu_mul_kernel(
    gate_ptr, up_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SiLU(gate) * up."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    gate = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask).to(tl.float32)

    # SiLU: x * sigmoid(x)
    silu_gate = gate * tl.sigmoid(gate)
    result = silu_gate * up

    tl.store(out_ptr + offs, result.to(tl.float16), mask=mask)


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(gate) * up operation."""
    assert gate.shape == up.shape
    out = torch.empty_like(gate)
    n_elements = gate.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    fused_silu_mul_kernel[grid](
        gate, up, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def expert_mlp_int4_fused(
    x: torch.Tensor,              # [num_tokens, hidden_dim] fp16
    gate_packed: torch.Tensor,    # [intermediate_dim, hidden_dim//2] uint8
    gate_scales: torch.Tensor,    # [intermediate_dim, hidden_dim//group_size] fp16
    up_packed: torch.Tensor,      # [intermediate_dim, hidden_dim//2] uint8
    up_scales: torch.Tensor,      # [intermediate_dim, hidden_dim//group_size] fp16
    down_packed: torch.Tensor,    # [hidden_dim, intermediate_dim//2] uint8
    down_scales: torch.Tensor,    # [hidden_dim, intermediate_dim//group_size] fp16
    group_size: int = 128,
) -> torch.Tensor:
    """
    Fused SwiGLU expert MLP with INT4 weights.

    Computes: out = down(silu(gate(x)) * up(x))

    NO fp16 weight materialization - all dequant is on-the-fly in each GEMM.
    Peak memory: only the activation tensors (~3MB for 9 tokens × 14336 intermediate)

    Args:
        x: Input [num_tokens, hidden_dim]
        gate_packed, gate_scales: Gate projection INT4 weights
        up_packed, up_scales: Up projection INT4 weights
        down_packed, down_scales: Down projection INT4 weights
        group_size: Quantization group size

    Returns:
        Output [num_tokens, hidden_dim]
    """
    # gate(x) -> [num_tokens, intermediate_dim]
    gate_out = int4_linear(x, gate_packed, gate_scales, group_size)

    # up(x) -> [num_tokens, intermediate_dim]
    up_out = int4_linear(x, up_packed, up_scales, group_size)

    # silu(gate) * up -> [num_tokens, intermediate_dim]
    hidden = fused_silu_mul(gate_out, up_out)

    # Free intermediates eagerly
    del gate_out, up_out

    # down(hidden) -> [num_tokens, hidden_dim]
    out = int4_linear(hidden, down_packed, down_scales, group_size)

    return out
