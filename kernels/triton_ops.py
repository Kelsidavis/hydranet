"""Triton kernels for HydraNet MoE operations."""

import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _expert_forward_kernel(
    # Input pointers
    x_ptr,
    gate_ptr,
    up_ptr,
    down_ptr,
    # Output pointer
    out_ptr,
    # Dimensions
    batch_size,
    hidden_dim,
    intermediate_dim,
    # Strides
    stride_x_batch,
    stride_x_hidden,
    stride_gate_inter,
    stride_gate_hidden,
    stride_up_inter,
    stride_up_hidden,
    stride_down_hidden,
    stride_down_inter,
    stride_out_batch,
    stride_out_hidden,
    # Block sizes
    BLOCK_BATCH: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_INTER: tl.constexpr,
):
    """
    Fused SwiGLU expert forward kernel.

    Computes: out = down(silu(gate(x)) * up(x))
    """
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_hidden = tl.program_id(1)

    # Batch and hidden offsets
    batch_offs = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    hidden_offs = pid_hidden * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)

    # Masks
    batch_mask = batch_offs < batch_size
    hidden_mask = hidden_offs < hidden_dim

    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_BATCH, BLOCK_HIDDEN), dtype=tl.float32)

    # Process intermediate dimension in blocks
    for inter_start in range(0, intermediate_dim, BLOCK_INTER):
        inter_offs = inter_start + tl.arange(0, BLOCK_INTER)
        inter_mask = inter_offs < intermediate_dim

        # Load input x slice
        x_ptrs = x_ptr + batch_offs[:, None] * stride_x_batch + hidden_offs[None, :] * stride_x_hidden
        x = tl.load(x_ptrs, mask=batch_mask[:, None] & hidden_mask[None, :], other=0.0)

        # Compute gate(x) for this intermediate block
        gate_ptrs = gate_ptr + inter_offs[:, None] * stride_gate_inter + hidden_offs[None, :] * stride_gate_hidden
        gate_weights = tl.load(gate_ptrs, mask=inter_mask[:, None] & hidden_mask[None, :], other=0.0)

        # gate_out = x @ gate_weights.T
        gate_out = tl.dot(x, tl.trans(gate_weights))  # (BLOCK_BATCH, BLOCK_INTER)

        # Apply SiLU activation: silu(x) = x * sigmoid(x)
        gate_out = gate_out * tl.sigmoid(gate_out)

        # Compute up(x) for this intermediate block
        up_ptrs = up_ptr + inter_offs[:, None] * stride_up_inter + hidden_offs[None, :] * stride_up_hidden
        up_weights = tl.load(up_ptrs, mask=inter_mask[:, None] & hidden_mask[None, :], other=0.0)

        up_out = tl.dot(x, tl.trans(up_weights))  # (BLOCK_BATCH, BLOCK_INTER)

        # Element-wise multiply
        intermediate = gate_out * up_out  # (BLOCK_BATCH, BLOCK_INTER)

        # Load down projection weights for this block
        down_ptrs = down_ptr + hidden_offs[:, None] * stride_down_hidden + inter_offs[None, :] * stride_down_inter
        down_weights = tl.load(down_ptrs, mask=hidden_mask[:, None] & inter_mask[None, :], other=0.0)

        # Accumulate: intermediate @ down_weights
        acc += tl.dot(intermediate, tl.trans(down_weights))

    # Store output
    out_ptrs = out_ptr + batch_offs[:, None] * stride_out_batch + hidden_offs[None, :] * stride_out_hidden
    tl.store(out_ptrs, acc.to(tl.float16), mask=batch_mask[:, None] & hidden_mask[None, :])


def expert_forward_triton(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Triton-accelerated expert forward pass.

    Args:
        x: Input tensor (batch, hidden_dim)
        gate_weight: Gate projection (intermediate_dim, hidden_dim)
        up_weight: Up projection (intermediate_dim, hidden_dim)
        down_weight: Down projection (hidden_dim, intermediate_dim)

    Returns:
        Output tensor (batch, hidden_dim)
    """
    batch_size, hidden_dim = x.shape
    intermediate_dim = gate_weight.shape[0]

    # Allocate output
    out = torch.empty((batch_size, hidden_dim), device=x.device, dtype=x.dtype)

    # Block sizes (tuned for RTX 5080)
    BLOCK_BATCH = 32
    BLOCK_HIDDEN = 128
    BLOCK_INTER = 128

    # Grid
    grid = (
        triton.cdiv(batch_size, BLOCK_BATCH),
        triton.cdiv(hidden_dim, BLOCK_HIDDEN),
    )

    # Launch kernel
    _expert_forward_kernel[grid](
        x, gate_weight, up_weight, down_weight, out,
        batch_size, hidden_dim, intermediate_dim,
        x.stride(0), x.stride(1),
        gate_weight.stride(0), gate_weight.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        down_weight.stride(0), down_weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_HIDDEN=BLOCK_HIDDEN,
        BLOCK_INTER=BLOCK_INTER,
    )

    return out


@triton.jit
def _top_k_softmax_kernel(
    # Input/output pointers
    logits_ptr,
    indices_ptr,
    weights_ptr,
    # Dimensions
    num_tokens,
    num_experts,
    top_k,
    # Strides
    stride_logits_token,
    stride_logits_expert,
    stride_indices_token,
    stride_indices_k,
    stride_weights_token,
    stride_weights_k,
    # Block size
    BLOCK_EXPERTS: tl.constexpr,
):
    """
    Fused top-k selection and softmax for routing.
    """
    pid = tl.program_id(0)

    # Load logits for this token
    expert_offs = tl.arange(0, BLOCK_EXPERTS)
    mask = expert_offs < num_experts

    logits_ptrs = logits_ptr + pid * stride_logits_token + expert_offs * stride_logits_expert
    logits = tl.load(logits_ptrs, mask=mask, other=float('-inf'))

    # Find top-k using iterative selection
    selected_indices = tl.zeros((top_k,), dtype=tl.int32)
    selected_logits = tl.zeros((top_k,), dtype=tl.float32)

    remaining_logits = logits
    for k in range(top_k):
        # Find max
        max_val = tl.max(remaining_logits, axis=0)
        max_idx = tl.argmax(remaining_logits, axis=0)

        # Store selection
        # Note: Triton doesn't support dynamic indexing well, so this is simplified
        if k == 0:
            selected_indices = tl.where(tl.arange(0, top_k) == 0, max_idx, selected_indices)
            selected_logits = tl.where(tl.arange(0, top_k) == 0, max_val, selected_logits)
        elif k == 1:
            selected_indices = tl.where(tl.arange(0, top_k) == 1, max_idx, selected_indices)
            selected_logits = tl.where(tl.arange(0, top_k) == 1, max_val, selected_logits)

        # Mask out selected
        remaining_logits = tl.where(expert_offs == max_idx, float('-inf'), remaining_logits)

    # Softmax over selected logits
    max_logit = tl.max(selected_logits, axis=0)
    exp_logits = tl.exp(selected_logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    weights = exp_logits / sum_exp

    # Store results
    k_offs = tl.arange(0, top_k)
    indices_ptrs = indices_ptr + pid * stride_indices_token + k_offs * stride_indices_k
    weights_ptrs = weights_ptr + pid * stride_weights_token + k_offs * stride_weights_k

    tl.store(indices_ptrs, selected_indices, mask=k_offs < top_k)
    tl.store(weights_ptrs, weights, mask=k_offs < top_k)


def top_k_softmax_triton(
    logits: torch.Tensor,
    top_k: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated top-k selection with softmax.

    Args:
        logits: Router logits (num_tokens, num_experts)
        top_k: Number of experts to select

    Returns:
        indices: Selected expert indices (num_tokens, top_k)
        weights: Softmax weights (num_tokens, top_k)
    """
    num_tokens, num_experts = logits.shape

    # Allocate outputs
    indices = torch.empty((num_tokens, top_k), device=logits.device, dtype=torch.int32)
    weights = torch.empty((num_tokens, top_k), device=logits.device, dtype=logits.dtype)

    # Block size (must be >= num_experts)
    BLOCK_EXPERTS = triton.next_power_of_2(num_experts)

    # Launch kernel
    _top_k_softmax_kernel[(num_tokens,)](
        logits, indices, weights,
        num_tokens, num_experts, top_k,
        logits.stride(0), logits.stride(1),
        indices.stride(0), indices.stride(1),
        weights.stride(0), weights.stride(1),
        BLOCK_EXPERTS=BLOCK_EXPERTS,
    )

    return indices.to(torch.int64), weights


@triton.jit
def _scatter_add_kernel(
    # Pointers
    src_ptr,
    indices_ptr,
    out_ptr,
    # Dimensions
    num_elements,
    hidden_dim,
    # Strides
    stride_src_elem,
    stride_src_hidden,
    stride_out_elem,
    stride_out_hidden,
    # Block size
    BLOCK_HIDDEN: tl.constexpr,
):
    """
    Scatter-add kernel for accumulating expert outputs.
    """
    pid_elem = tl.program_id(0)
    pid_hidden = tl.program_id(1)

    # Load destination index
    dest_idx = tl.load(indices_ptr + pid_elem)

    # Hidden dimension offsets
    hidden_offs = pid_hidden * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    mask = hidden_offs < hidden_dim

    # Load source values
    src_ptrs = src_ptr + pid_elem * stride_src_elem + hidden_offs * stride_src_hidden
    src_vals = tl.load(src_ptrs, mask=mask, other=0.0)

    # Atomic add to output
    out_ptrs = out_ptr + dest_idx * stride_out_elem + hidden_offs * stride_out_hidden
    tl.atomic_add(out_ptrs, src_vals, mask=mask)


def scatter_add_triton(
    src: torch.Tensor,
    indices: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """
    Triton-accelerated scatter add.

    Args:
        src: Source tensor (num_elements, hidden_dim)
        indices: Destination indices (num_elements,)
        output_size: Size of output first dimension

    Returns:
        Output tensor (output_size, hidden_dim)
    """
    num_elements, hidden_dim = src.shape

    # Allocate output (zeroed)
    out = torch.zeros((output_size, hidden_dim), device=src.device, dtype=src.dtype)

    BLOCK_HIDDEN = 128
    grid = (num_elements, triton.cdiv(hidden_dim, BLOCK_HIDDEN))

    _scatter_add_kernel[grid](
        src, indices, out,
        num_elements, hidden_dim,
        src.stride(0), src.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_HIDDEN=BLOCK_HIDDEN,
    )

    return out


# Autotuned configurations for different hardware
def get_expert_kernel_config(device_name: str = None) -> dict:
    """Get optimized kernel config for device."""
    if device_name is None:
        device_name = torch.cuda.get_device_name()

    # RTX 5080 specific tuning
    if "5080" in device_name or "50" in device_name:
        return {
            "BLOCK_BATCH": 64,
            "BLOCK_HIDDEN": 128,
            "BLOCK_INTER": 256,
            "num_warps": 8,
            "num_stages": 3,
        }
    # RTX 4090 / 4080
    elif "40" in device_name:
        return {
            "BLOCK_BATCH": 64,
            "BLOCK_HIDDEN": 128,
            "BLOCK_INTER": 128,
            "num_warps": 8,
            "num_stages": 2,
        }
    # Default
    else:
        return {
            "BLOCK_BATCH": 32,
            "BLOCK_HIDDEN": 128,
            "BLOCK_INTER": 128,
            "num_warps": 4,
            "num_stages": 2,
        }


def benchmark_expert_kernel(
    hidden_dim: int = 4096,
    intermediate_dim: int = 6144,
    batch_sizes: list = [1, 8, 32, 128, 512],
    num_runs: int = 100,
) -> dict:
    """Benchmark expert forward kernel."""
    import time

    device = torch.device("cuda")
    results = {}

    for batch_size in batch_sizes:
        # Create test tensors
        x = torch.randn(batch_size, hidden_dim, device=device, dtype=torch.float16)
        gate = torch.randn(intermediate_dim, hidden_dim, device=device, dtype=torch.float16)
        up = torch.randn(intermediate_dim, hidden_dim, device=device, dtype=torch.float16)
        down = torch.randn(hidden_dim, intermediate_dim, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(10):
            _ = expert_forward_triton(x, gate, up, down)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            _ = expert_forward_triton(x, gate, up, down)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        avg_ms = (elapsed / num_runs) * 1000
        results[f"batch_{batch_size}"] = {
            "avg_ms": avg_ms,
            "throughput_tokens_per_sec": batch_size / (avg_ms / 1000),
        }

    return results
