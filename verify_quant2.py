#!/usr/bin/env python3
"""Verify INT4 dequantization matches between packer and GPU slot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    import torch
    import numpy as np
    from hydranet.v2.config import MixtralConfig
    from hydranet.v2.preprocess.pack_weights import ExpertWeightPacker, QuantConfig, load_packed_expert
    from hydranet.v2.cache.packed_expert_store import PackedExpertStore, GpuExpertSlot, BlobLayout

    device = torch.device("cuda")
    packed_dir = Path("/home/k/models/mixtral-8x7b-instruct/packed_int4")

    print("=" * 60)
    print("DEQUANT VERIFICATION")
    print("=" * 60)

    config = MixtralConfig()
    layer_idx, expert_idx = 0, 0

    # Method 1: Load individual .bin file and dequant with packer
    print("\n1. Loading from individual .bin file...")
    expert_file = packed_dir / f"expert_L{layer_idx:02d}_E{expert_idx:02d}.bin"
    packed_expert = load_packed_expert(expert_file)

    packer = ExpertWeightPacker(QuantConfig(group_size=128))
    gate_packer = packer.dequantize_tensor(
        packed_expert.gate_proj_packed,
        packed_expert.gate_proj_scales,
        config.intermediate_dim,
        config.hidden_dim,
    )
    print(f"  gate_packer[0,0:5]: {gate_packer[0, 0:5]}")

    # Method 2: Load from consolidated store and dequant with GpuExpertSlot
    print("\n2. Loading from consolidated store...")
    store = PackedExpertStore(
        index_path=packed_dir / "experts.idx",
        bin_path=packed_dir / "experts.bin",
        device=device,
        use_mmap=True,
    )

    gpu_slot = GpuExpertSlot(
        slot_idx=0,
        hidden_dim=config.hidden_dim,
        intermediate_dim=config.intermediate_dim,
        group_size=store.quant_config["group_size"],
        device=device,
    )

    _, staging, layout = store.load_to_staging(layer_idx, expert_idx)
    gpu_slot.load_from_pinned(staging, layout, layer_idx, expert_idx)
    dequant_weights = gpu_slot.dequantize()
    gate_gpu = dequant_weights["gate_proj"]
    print(f"  gate_gpu[0,0:5]:    {gate_gpu[0, 0:5].cpu()}")

    # Compare
    gate_packer_gpu = gate_packer.to(device)
    diff = (gate_packer_gpu - gate_gpu).abs()
    print(f"\n  Difference between packer and GPU dequant:")
    print(f"    max_diff: {diff.max().item():.10f}")
    print(f"    mean_diff: {diff.mean().item():.10f}")

    # Check raw packed data matches
    print("\n3. Comparing raw packed data...")

    # From .bin file
    gate_packed_file = packed_expert.gate_proj_packed
    gate_scales_file = packed_expert.gate_proj_scales

    # From GPU slot (loaded from consolidated store)
    gate_packed_gpu = gpu_slot.gate_packed.cpu()
    gate_scales_gpu = gpu_slot.gate_scales.cpu()

    print(f"  gate_packed_file[0,0:5]: {gate_packed_file[0, 0:5]}")
    print(f"  gate_packed_gpu[0,0:5]:  {gate_packed_gpu[0, 0:5]}")
    print(f"  gate_scales_file[0,0:5]: {gate_scales_file[0, 0:5]}")
    print(f"  gate_scales_gpu[0,0:5]:  {gate_scales_gpu[0, 0:5]}")

    packed_match = torch.equal(gate_packed_file, gate_packed_gpu)
    scales_match = torch.allclose(gate_scales_file, gate_scales_gpu, atol=1e-10)
    print(f"\n  packed_match: {packed_match}")
    print(f"  scales_match: {scales_match}")

    if not packed_match:
        diff_packed = (gate_packed_file.int() - gate_packed_gpu.int()).abs()
        print(f"  packed diff max: {diff_packed.max().item()}")
        # Find where differences are
        nonzero = diff_packed.nonzero()
        if len(nonzero) > 0:
            print(f"  First diff at: {nonzero[0]}")
            idx = tuple(nonzero[0].tolist())
            print(f"    file value: {gate_packed_file[idx].item()}")
            print(f"    gpu value:  {gate_packed_gpu[idx].item()}")

    store.close()


if __name__ == "__main__":
    main()
