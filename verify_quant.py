#!/usr/bin/env python3
"""Verify INT4 quantization round-trip."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    import torch
    import numpy as np
    from hydranet.v2.config import MixtralConfig
    from hydranet.v2.model.loader import MixtralWeightLoader
    from hydranet.v2.cache.packed_expert_store import PackedExpertStore, GpuExpertSlot, BlobLayout

    device = torch.device("cuda")
    model_path = "/home/k/models/mixtral-8x7b-instruct"
    packed_dir = Path("/home/k/models/mixtral-8x7b-instruct/packed_int4")

    print("=" * 60)
    print("QUANTIZATION VERIFICATION")
    print("=" * 60)

    # Load original fp16 weights for comparison
    config = MixtralConfig()
    loader = MixtralWeightLoader(model_path, config, torch.device("cpu"), torch.float16)

    # Test layer 0, expert 0
    layer_idx, expert_idx = 0, 0
    print(f"\nLoading original fp16 expert ({layer_idx}, {expert_idx})...")
    original_weights = loader.load_expert(layer_idx, expert_idx)

    print(f"  gate_proj: {original_weights['gate_proj'].shape}, {original_weights['gate_proj'].dtype}")
    print(f"  up_proj: {original_weights['up_proj'].shape}")
    print(f"  down_proj: {original_weights['down_proj'].shape}")

    # Load packed store
    print("\nLoading packed store...")
    store = PackedExpertStore(
        index_path=packed_dir / "experts.idx",
        bin_path=packed_dir / "experts.bin",
        device=device,
        use_mmap=True,
    )

    # Create GPU slot and load expert
    print("Loading INT4 expert to GPU and dequantizing...")
    gpu_slot = GpuExpertSlot(
        slot_idx=0,
        hidden_dim=config.hidden_dim,
        intermediate_dim=config.intermediate_dim,
        group_size=store.quant_config["group_size"],
        device=device,
    )

    # Load to staging and then to GPU
    _, staging, layout = store.load_to_staging(layer_idx, expert_idx)
    gpu_slot.load_from_pinned(staging, layout, layer_idx, expert_idx)

    # Dequantize
    dequant_weights = gpu_slot.dequantize()

    # Compare
    print("\nComparing fp16 original vs INT4 dequantized:")

    for name in ["gate_proj", "up_proj", "down_proj"]:
        orig = original_weights[name].to(device)
        deq = dequant_weights[name]

        diff = (orig - deq).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_error = (diff / (orig.abs() + 1e-10)).mean().item()

        # SNR (Signal-to-Noise Ratio)
        signal_power = (orig ** 2).mean().item()
        noise_power = (diff ** 2).mean().item()
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

        print(f"  {name}:")
        print(f"    max_diff: {max_diff:.6f}")
        print(f"    mean_diff: {mean_diff:.6f}")
        print(f"    rel_error: {rel_error:.4%}")
        print(f"    SNR: {snr_db:.1f} dB")

        # Check a few specific values
        print(f"    Sample values:")
        print(f"      orig[0,0]: {orig[0,0].item():.6f}")
        print(f"      deq[0,0]:  {deq[0,0].item():.6f}")
        print(f"      orig[100,500]: {orig[100,500].item():.6f}")
        print(f"      deq[100,500]:  {deq[100,500].item():.6f}")

    # Test a simple forward pass through one expert
    print("\nTesting expert MLP forward pass...")
    x = torch.randn(1, 4, config.hidden_dim, device=device, dtype=torch.float16)

    # Original fp16
    with torch.no_grad():
        gate_orig = original_weights["gate_proj"].to(device)
        up_orig = original_weights["up_proj"].to(device)
        down_orig = original_weights["down_proj"].to(device)

        gate_out_orig = torch.nn.functional.silu(torch.nn.functional.linear(x, gate_orig))
        up_out_orig = torch.nn.functional.linear(x, up_orig)
        out_orig = torch.nn.functional.linear(gate_out_orig * up_out_orig, down_orig)

    # Dequantized INT4
    with torch.no_grad():
        gate_deq = dequant_weights["gate_proj"]
        up_deq = dequant_weights["up_proj"]
        down_deq = dequant_weights["down_proj"]

        gate_out_deq = torch.nn.functional.silu(torch.nn.functional.linear(x, gate_deq))
        up_out_deq = torch.nn.functional.linear(x, up_deq)
        out_deq = torch.nn.functional.linear(gate_out_deq * up_out_deq, down_deq)

    # Compare outputs
    out_diff = (out_orig - out_deq).abs()
    print(f"\nExpert output comparison:")
    print(f"  max_diff: {out_diff.max().item():.6f}")
    print(f"  mean_diff: {out_diff.mean().item():.6f}")
    print(f"  orig mean: {out_orig.abs().mean().item():.6f}")
    print(f"  deq mean: {out_deq.abs().mean().item():.6f}")

    # Relative error
    rel_out_error = (out_diff / (out_orig.abs() + 1e-10)).mean().item()
    print(f"  rel_error: {rel_out_error:.4%}")

    store.close()


if __name__ == "__main__":
    main()
