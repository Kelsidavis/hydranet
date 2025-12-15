#!/usr/bin/env python3
"""
Pack GLM-4.5-Air expert weights to INT4 format.

Creates experts.bin + experts.idx for the PackedExpertStore.

Usage:
    python -m hydranet.v2.preprocess.pack_glm4 \
        --model-path /path/to/GLM-4.5-Air \
        --output-dir /path/to/packed_int4

Memory requirements:
    - ~4GB RAM for packing buffer
    - Streams experts one at a time
"""

import torch
import json
import argparse
from pathlib import Path
from typing import Dict
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hydranet.v2.config import GLM4AirConfig
from hydranet.v2.model.glm4_loader import GLM4WeightLoader
from hydranet.v2.preprocess.pack_weights import ExpertWeightPacker, QuantConfig
from hydranet.v2.cache.packed_expert_store import BlobLayout


def pack_glm4_experts(
    model_path: str,
    output_dir: str,
    group_size: int = 128,
    dtype: torch.dtype = torch.float16,
):
    """
    Pack all GLM4 routed expert weights to INT4 format.

    Creates:
        - experts.bin: All experts concatenated as contiguous blobs
        - experts.idx: JSON index with offsets, sizes, metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GLM4AirConfig()
    loader = GLM4WeightLoader(model_path, config, dtype=dtype)
    packer = ExpertWeightPacker(QuantConfig(group_size=group_size))

    # Calculate blob layout for GLM4 experts
    layout = BlobLayout.from_dims(
        hidden_dim=config.hidden_dim,
        intermediate_dim=config.moe_intermediate_dim,
        group_size=group_size,
    )

    print(f"GLM-4.5-Air Expert Packing")
    print(f"=" * 60)
    print(f"Model path: {model_path}")
    print(f"Output dir: {output_dir}")
    print(f"MoE layers: {config.num_moe_layers}")
    print(f"Experts per layer: {config.num_experts}")
    print(f"Total routed experts: {config.total_routed_experts}")
    print(f"Expert dimensions: {config.hidden_dim} -> {config.moe_intermediate_dim}")
    print(f"Blob size per expert: {layout.total_size / 1e6:.2f} MB")
    print(f"Total packed size: {config.total_routed_experts * layout.total_size / 1e9:.2f} GB")
    print(f"=" * 60)

    # Index data
    index_data = {
        "format_version": "2.0",
        "model_type": "glm4_air",
        "quant_config": {
            "bits": 4,
            "group_size": group_size,
            "symmetric": True,
        },
        "model_config": {
            "num_layers": config.num_moe_layers,
            "num_experts": config.num_experts,
            "hidden_dim": config.hidden_dim,
            "intermediate_dim": config.moe_intermediate_dim,
        },
        "experts": [],
    }

    bin_path = output_dir / "experts.bin"
    current_offset = 0

    print(f"\nPacking experts to: {bin_path}")

    with open(bin_path, "wb") as bin_file:
        for moe_layer_idx, expert_idx, weights in loader.iter_experts():
            # Quantize each projection
            gate_packed, gate_scales = packer.quantize_tensor(weights["gate_proj"])
            up_packed, up_scales = packer.quantize_tensor(weights["up_proj"])
            down_packed, down_scales = packer.quantize_tensor(weights["down_proj"])

            # Create blob
            blob = bytearray(layout.total_size)

            # Write packed weights
            blob[layout.gate_packed_offset:layout.gate_packed_offset + layout.gate_packed_size] = \
                gate_packed.numpy().tobytes()
            blob[layout.gate_scales_offset:layout.gate_scales_offset + layout.gate_scales_size] = \
                gate_scales.numpy().tobytes()

            blob[layout.up_packed_offset:layout.up_packed_offset + layout.up_packed_size] = \
                up_packed.numpy().tobytes()
            blob[layout.up_scales_offset:layout.up_scales_offset + layout.up_scales_size] = \
                up_scales.numpy().tobytes()

            blob[layout.down_packed_offset:layout.down_packed_offset + layout.down_packed_size] = \
                down_packed.numpy().tobytes()
            blob[layout.down_scales_offset:layout.down_scales_offset + layout.down_scales_size] = \
                down_scales.numpy().tobytes()

            # Write blob
            bin_file.write(blob)

            # Record index entry
            index_data["experts"].append({
                "layer_idx": moe_layer_idx,
                "expert_idx": expert_idx,
                "offset": current_offset,
                "size": layout.total_size,
            })

            current_offset += layout.total_size

            # Progress
            total_done = moe_layer_idx * config.num_experts + expert_idx + 1
            if total_done % 256 == 0 or total_done == config.total_routed_experts:
                pct = 100 * total_done / config.total_routed_experts
                print(f"  Packed {total_done}/{config.total_routed_experts} experts ({pct:.1f}%)")

    # Write index
    index_path = output_dir / "experts.idx"
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

    loader.close()

    print(f"\nDone!")
    print(f"  Index: {index_path}")
    print(f"  Binary: {bin_path} ({current_offset / 1e9:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Pack GLM-4.5-Air experts to INT4")
    parser.add_argument("--model-path", required=True, help="Path to GLM-4.5-Air weights")
    parser.add_argument("--output-dir", required=True, help="Output directory for packed weights")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size")
    args = parser.parse_args()

    pack_glm4_experts(
        model_path=args.model_path,
        output_dir=args.output_dir,
        group_size=args.group_size,
    )


if __name__ == "__main__":
    main()
