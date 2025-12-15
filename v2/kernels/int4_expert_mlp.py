"""
Fused INT4 Expert MLP Triton kernel.

Dequantizes INT4 weights on-the-fly during GEMM, eliminating
the ~350MB fp16 weight materialization spike.

SwiGLU MLP: out = down(silu(gate(x)) * up(x))
"""

import torch
import triton
import triton.language as tl


@triton.jit
def int4_matmul_kernel(
    # Input
    x_ptr,           # [M, K] fp16
    # Packed weights
    w_packed_ptr,    # [N, K//2] uint8 (INT4 packed)
    w_scales_ptr,    # [N, K//group_size] fp16
    # Output
    out_ptr,         # [M, N] fp16
    # Dimensions
    M, N, K,
    group_size: tl.constexpr,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sn, stride_sk,
    stride_om, stride_on,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    INT4 matrix multiply with on-the-fly dequantization.

    Computes: out[m, n] = sum_k(x[m, k] * dequant(w[n, k]))
    where dequant unpacks INT4 and applies per-group scales.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute block starting positions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load x block [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float16)

        # Load packed weights [BLOCK_N, BLOCK_K//2]
        # Each byte contains 2 INT4 values
        offs_k_packed = offs_k // 2
        w_ptrs = w_packed_ptr + offs_n[:, None] * stride_wn + offs_k_packed[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (offs_k_packed[None, :] < K // 2)
        w_packed = tl.load(w_ptrs, mask=w_mask, other=0).to(tl.uint8)

        # Unpack INT4: even indices get low nibble, odd get high nibble
        is_odd = (offs_k % 2) == 1
        w_int4 = tl.where(
            is_odd[None, :],
            (w_packed >> 4) & 0xF,
            w_packed & 0xF
        )
        # Convert to signed: 0-15 -> -8 to 7
        w_int = w_int4.to(tl.int8) - 8

        # Load scales [BLOCK_N, num_groups_in_block]
        # Each group of `group_size` elements shares a scale
        offs_group = offs_k // group_size
        s_ptrs = w_scales_ptr + offs_n[:, None] * stride_sn + offs_group[None, :] * stride_sk
        s_mask = (offs_n[:, None] < N) & (offs_group[None, :] < K // group_size)
        scales = tl.load(s_ptrs, mask=s_mask, other=1.0).to(tl.float16)

        # Dequantize: w_fp16 = w_int * scale
        w_fp16 = w_int.to(tl.float16) * scales

        # Accumulate: out += x @ w.T
        # x: [BLOCK_M, BLOCK_K], w: [BLOCK_N, BLOCK_K] -> out: [BLOCK_M, BLOCK_N]
        acc += tl.dot(x, tl.trans(w_fp16)).to(tl.float32)

    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


def int4_linear(
    x: torch.Tensor,           # [M, K] fp16
    w_packed: torch.Tensor,    # [N, K//2] uint8
    w_scales: torch.Tensor,    # [N, K//group_size] fp16
    group_size: int = 128,
) -> torch.Tensor:
    """
    INT4 linear layer with on-the-fly dequantization.

    Args:
        x: Input tensor [M, K] or [batch, seq, K]
        w_packed: Packed INT4 weights [N, K//2]
        w_scales: Per-group scales [N, K//group_size]
        group_size: Quantization group size

    Returns:
        Output tensor [M, N] or [batch, seq, N]
    """
    # Handle batched input
    orig_shape = x.shape
    if x.dim() == 3:
        batch, seq, K = x.shape
        x = x.view(batch * seq, K)

    M, K = x.shape
    N = w_packed.shape[0]

    assert w_packed.shape == (N, K // 2), f"w_packed shape mismatch: {w_packed.shape} vs ({N}, {K // 2})"
    assert w_scales.shape == (N, K // group_size), f"w_scales shape mismatch: {w_scales.shape} vs ({N}, {K // group_size})"

    # Output
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Block sizes (tuned for common sizes)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64  # Must be multiple of group_size for correct scale indexing

    # Grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
    int4_matmul_kernel[grid](
        x, w_packed, w_scales, out,
        M, N, K,
        group_size,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        w_scales.stride(0), w_scales.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # Restore batch dimensions
    if len(orig_shape) == 3:
        out = out.view(batch, seq, N)

    return out


@triton.jit
def silu_mul_kernel(
    gate_ptr,  # [M, N] fp16
    up_ptr,    # [M, N] fp16
    out_ptr,   # [M, N] fp16
    M, N,
    stride_m, stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused SiLU(gate) * up."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    gate = tl.load(gate_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)

    # SiLU: x * sigmoid(x)
    silu_gate = gate * tl.sigmoid(gate)
    out = silu_gate * up

    tl.store(out_ptr + ptrs, out.to(tl.float16), mask=mask)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(gate) * up operation."""
    assert gate.shape == up.shape
    out = torch.empty_like(gate)

    M, N = gate.shape[-2], gate.shape[-1]
    if gate.dim() == 3:
        M = gate.shape[0] * gate.shape[1]
        gate = gate.view(M, N)
        up = up.view(M, N)
        out = out.view(M, N)

    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    silu_mul_kernel[grid](
        gate, up, out, M, N,
        gate.stride(-2), gate.stride(-1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    return out


def expert_mlp_int4(
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

    No fp16 weight materialization - dequant happens on-the-fly in each GEMM.

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
    hidden = silu_mul(gate_out, up_out)

    # down(hidden) -> [num_tokens, hidden_dim]
    out = int4_linear(hidden, down_packed, down_scales, group_size)

    return out


def expert_mlp_int4_from_slot(
    x: torch.Tensor,
    gpu_slot,  # GpuExpertSlot with INT4 buffers
    group_size: int = 128,
) -> torch.Tensor:
    """
    Expert MLP using INT4 weights directly from a GpuExpertSlot.

    This is the main entry point for the MoE layer - reads INT4 buffers
    from the slot without ever materializing fp16 weights.
    """
    return expert_mlp_int4(
        x,
        gpu_slot.gate_packed, gpu_slot.gate_scales,
        gpu_slot.up_packed, gpu_slot.up_scales,
        gpu_slot.down_packed, gpu_slot.down_scales,
        group_size,
    )
