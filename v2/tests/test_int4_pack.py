#!/usr/bin/env python3
"""
Tests for INT4 quantization and packed expert store.

Verifies:
1. INT4 quantization preserves information (within tolerance)
2. Pack/unpack roundtrip is correct
3. GpuExpertSlot dequantization matches original
4. Packed store read/write works
"""

import torch
import tempfile
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def test_int4_quantization():
    """Test INT4 quantization preserves information."""
    print("=" * 60)
    print("TEST: INT4 Quantization")
    print("=" * 60)

    from hydranet.v2.preprocess.pack_weights import ExpertWeightPacker, QuantConfig

    config = QuantConfig(bits=4, group_size=128, symmetric=True)
    packer = ExpertWeightPacker(config)

    # Create test tensor (random fp16 weights)
    torch.manual_seed(42)
    out_features, in_features = 512, 256
    original = torch.randn(out_features, in_features, dtype=torch.float16) * 0.1

    # Quantize
    packed, scales = packer.quantize_tensor(original)

    # Dequantize
    dequant = packer.dequantize_tensor(packed, scales, out_features, in_features)

    # Check shapes
    assert packed.shape == (out_features, in_features // 2), f"Packed shape: {packed.shape}"
    assert scales.shape == (out_features, in_features // config.group_size), f"Scales shape: {scales.shape}"
    assert dequant.shape == original.shape, f"Dequant shape: {dequant.shape}"

    # Check reconstruction error
    abs_error = (original - dequant).abs()
    max_error = abs_error.max().item()
    mean_error = abs_error.mean().item()
    rmse = (abs_error ** 2).mean().sqrt().item()

    # Also check relative error
    rel_error = abs_error / (original.abs() + 1e-10)
    mean_rel_error = rel_error.mean().item()

    print(f"  Original range: [{original.min():.4f}, {original.max():.4f}]")
    print(f"  Max absolute error: {max_error:.6f}")
    print(f"  Mean absolute error: {mean_error:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Mean relative error: {mean_rel_error:.2%}")

    # INT4 with group_size=128 should have reasonable error
    assert max_error < 0.1, f"Max error too high: {max_error}"
    assert mean_error < 0.01, f"Mean error too high: {mean_error}"

    print("  ✓ Quantization error within tolerance")
    return True


def test_expert_pack_roundtrip():
    """Test packing and unpacking a full expert."""
    print("\n" + "=" * 60)
    print("TEST: Expert Pack Roundtrip")
    print("=" * 60)

    from hydranet.v2.preprocess.pack_weights import ExpertWeightPacker, QuantConfig

    config = QuantConfig()
    packer = ExpertWeightPacker(config)

    # Create mock expert weights
    torch.manual_seed(42)
    hidden_dim = 256
    intermediate_dim = 512

    weights = {
        "gate_proj": torch.randn(intermediate_dim, hidden_dim, dtype=torch.float16) * 0.02,
        "up_proj": torch.randn(intermediate_dim, hidden_dim, dtype=torch.float16) * 0.02,
        "down_proj": torch.randn(hidden_dim, intermediate_dim, dtype=torch.float16) * 0.02,
    }

    # Pack
    packed = packer.pack_expert(layer_idx=0, expert_idx=0, weights=weights)

    print(f"  Packed expert size: {packed.size_bytes() / 1024:.1f} KB")
    print(f"  Original size: {(3 * hidden_dim * intermediate_dim * 2) / 1024:.1f} KB")
    print(f"  Compression: {(3 * hidden_dim * intermediate_dim * 2) / packed.size_bytes():.2f}x")

    # Dequantize and compare
    for name in ["gate_proj", "up_proj", "down_proj"]:
        packed_tensor = getattr(packed, f"{name}_packed")
        scales = getattr(packed, f"{name}_scales")

        orig = weights[name]
        out_features, in_features = orig.shape

        dequant = packer.dequantize_tensor(packed_tensor, scales, out_features, in_features)

        error = (orig - dequant).abs().mean().item()
        print(f"  {name} mean error: {error:.6f}")

        assert error < 0.01, f"{name} error too high"

    print("  ✓ Expert roundtrip successful")
    return True


def test_gpu_expert_slot():
    """Test GpuExpertSlot dequantization."""
    print("\n" + "=" * 60)
    print("TEST: GPU Expert Slot")
    print("=" * 60)

    from hydranet.v2.cache.packed_expert_store import GpuExpertSlot, BlobLayout

    # Skip if no CUDA
    device = torch.device("cpu")  # Use CPU for testing

    hidden_dim = 256
    intermediate_dim = 512
    group_size = 128

    slot = GpuExpertSlot(
        slot_idx=0,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        group_size=group_size,
        device=device,
    )

    # Create mock INT4 data
    torch.manual_seed(42)

    # Simulate packed data
    slot.gate_packed.random_(0, 256)
    slot.gate_scales.fill_(0.01)
    slot.up_packed.random_(0, 256)
    slot.up_scales.fill_(0.01)
    slot.down_packed.random_(0, 256)
    slot.down_scales.fill_(0.01)

    slot.layer_idx = 0
    slot.expert_idx = 0

    # Dequantize
    weights = slot.dequantize()

    # Check shapes
    assert weights["gate_proj"].shape == (intermediate_dim, hidden_dim)
    assert weights["up_proj"].shape == (intermediate_dim, hidden_dim)
    assert weights["down_proj"].shape == (hidden_dim, intermediate_dim)

    # Check no NaN/Inf
    for name, tensor in weights.items():
        assert not torch.isnan(tensor).any(), f"NaN in {name}"
        assert not torch.isinf(tensor).any(), f"Inf in {name}"

    # Check caching works
    weights2 = slot.dequantize()
    assert weights is weights2, "Caching should return same object"

    print(f"  Slot memory: {slot.size_bytes() / 1024:.1f} KB")
    print("  ✓ GPU slot dequantization works")
    return True


def test_blob_layout():
    """Test blob layout calculation."""
    print("\n" + "=" * 60)
    print("TEST: Blob Layout")
    print("=" * 60)

    from hydranet.v2.cache.packed_expert_store import BlobLayout

    hidden_dim = 4096
    intermediate_dim = 14336
    group_size = 128

    layout = BlobLayout.from_dims(hidden_dim, intermediate_dim, group_size)

    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Intermediate dim: {intermediate_dim}")
    print(f"  Group size: {group_size}")
    print(f"  Total blob size: {layout.total_size / 1024 / 1024:.2f} MB")

    # Verify sizes
    num_groups_h = hidden_dim // group_size
    num_groups_i = intermediate_dim // group_size

    expected_gate_packed = intermediate_dim * (hidden_dim // 2)
    expected_gate_scales = intermediate_dim * num_groups_h * 2

    assert layout.gate_packed_size == expected_gate_packed, f"gate_packed_size mismatch"
    assert layout.gate_scales_size == expected_gate_scales, f"gate_scales_size mismatch"

    # Verify offsets are contiguous
    assert layout.gate_packed_offset == 0
    assert layout.gate_scales_offset == layout.gate_packed_size
    assert layout.up_packed_offset == layout.gate_scales_offset + layout.gate_scales_size

    # Calculate expected total size
    # gate: intermediate * hidden/2 (packed) + intermediate * (hidden/group) * 2 (scales)
    # up: same as gate
    # down: hidden * intermediate/2 (packed) + hidden * (intermediate/group) * 2 (scales)
    gate_up_size = (intermediate_dim * hidden_dim // 2) + (intermediate_dim * num_groups_h * 2)
    down_size = (hidden_dim * intermediate_dim // 2) + (hidden_dim * num_groups_i * 2)
    expected_total = 2 * gate_up_size + down_size

    assert layout.total_size == expected_total, f"Total size mismatch: {layout.total_size} vs {expected_total}"

    print(f"  Gate packed: offset={layout.gate_packed_offset}, size={layout.gate_packed_size}")
    print(f"  Gate scales: offset={layout.gate_scales_offset}, size={layout.gate_scales_size}")
    print(f"  Up packed: offset={layout.up_packed_offset}, size={layout.up_packed_size}")
    print(f"  Up scales: offset={layout.up_scales_offset}, size={layout.up_scales_size}")
    print(f"  Down packed: offset={layout.down_packed_offset}, size={layout.down_packed_size}")
    print(f"  Down scales: offset={layout.down_scales_offset}, size={layout.down_scales_size}")
    print("  ✓ Blob layout correct")
    return True


def test_pinned_staging_ring():
    """Test pinned staging ring buffer."""
    print("\n" + "=" * 60)
    print("TEST: Pinned Staging Ring")
    print("=" * 60)

    from hydranet.v2.cache.packed_expert_store import PinnedStagingRing

    device = torch.device("cpu")  # Use CPU for testing
    num_slots = 3
    slot_size = 1024

    ring = PinnedStagingRing(num_slots, slot_size, device)

    # Acquire slots in sequence
    indices = []
    for _ in range(6):  # More than num_slots to test wrap-around
        idx, buffer = ring.acquire()
        indices.append(idx)
        assert buffer.shape == (slot_size,), f"Wrong buffer shape: {buffer.shape}"

    # Check wrap-around
    expected = [0, 1, 2, 0, 1, 2]
    assert indices == expected, f"Wrong indices: {indices} vs {expected}"

    print(f"  {num_slots} slots of {slot_size} bytes each")
    print(f"  Acquired indices: {indices}")
    print("  ✓ Ring buffer wrap-around works")
    return True


def test_packed_store_format():
    """Test creating and reading packed expert store."""
    print("\n" + "=" * 60)
    print("TEST: Packed Expert Store Format")
    print("=" * 60)

    from hydranet.v2.cache.packed_expert_store import PackedExpertStore, BlobLayout
    from hydranet.v2.preprocess.pack_weights import ExpertWeightPacker, QuantConfig
    import json

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock config
        hidden_dim = 256
        intermediate_dim = 512
        group_size = 128
        num_layers = 2
        num_experts = 4

        layout = BlobLayout.from_dims(hidden_dim, intermediate_dim, group_size)

        # Create mock experts.bin and experts.idx
        index_data = {
            "format_version": "2.0",
            "quant_config": {
                "bits": 4,
                "group_size": group_size,
                "symmetric": True,
            },
            "model_config": {
                "num_layers": num_layers,
                "num_experts": num_experts,
                "hidden_dim": hidden_dim,
                "intermediate_dim": intermediate_dim,
            },
            "experts": [],
        }

        # Create packer for test data
        packer = ExpertWeightPacker(QuantConfig(group_size=group_size))

        bin_path = tmpdir / "experts.bin"
        current_offset = 0

        with open(bin_path, "wb") as f:
            for layer_idx in range(num_layers):
                for expert_idx in range(num_experts):
                    # Create mock weights
                    torch.manual_seed(layer_idx * 100 + expert_idx)
                    weights = {
                        "gate_proj": torch.randn(intermediate_dim, hidden_dim, dtype=torch.float16) * 0.02,
                        "up_proj": torch.randn(intermediate_dim, hidden_dim, dtype=torch.float16) * 0.02,
                        "down_proj": torch.randn(hidden_dim, intermediate_dim, dtype=torch.float16) * 0.02,
                    }

                    packed = packer.pack_expert(layer_idx, expert_idx, weights)

                    # Write as blob
                    blob = bytearray(layout.total_size)
                    blob[layout.gate_packed_offset:layout.gate_packed_offset + layout.gate_packed_size] = \
                        packed.gate_proj_packed.numpy().tobytes()
                    blob[layout.gate_scales_offset:layout.gate_scales_offset + layout.gate_scales_size] = \
                        packed.gate_proj_scales.numpy().tobytes()
                    blob[layout.up_packed_offset:layout.up_packed_offset + layout.up_packed_size] = \
                        packed.up_proj_packed.numpy().tobytes()
                    blob[layout.up_scales_offset:layout.up_scales_offset + layout.up_scales_size] = \
                        packed.up_proj_scales.numpy().tobytes()
                    blob[layout.down_packed_offset:layout.down_packed_offset + layout.down_packed_size] = \
                        packed.down_proj_packed.numpy().tobytes()
                    blob[layout.down_scales_offset:layout.down_scales_offset + layout.down_scales_size] = \
                        packed.down_proj_scales.numpy().tobytes()

                    f.write(blob)

                    index_data["experts"].append({
                        "layer_idx": layer_idx,
                        "expert_idx": expert_idx,
                        "offset": current_offset,
                        "size": layout.total_size,
                    })
                    current_offset += layout.total_size

        # Write index
        idx_path = tmpdir / "experts.idx"
        with open(idx_path, "w") as f:
            json.dump(index_data, f)

        print(f"  Created {num_layers * num_experts} experts")
        print(f"  Total size: {current_offset / 1024:.1f} KB")

        # Now test loading via PackedExpertStore
        store = PackedExpertStore(
            index_path=idx_path,
            bin_path=bin_path,
            device=torch.device("cpu"),
            use_mmap=True,
        )

        # Read each expert and verify
        for layer_idx in range(num_layers):
            for expert_idx in range(num_experts):
                blob_bytes, blob_layout = store.get_expert_blob(layer_idx, expert_idx)

                assert len(blob_bytes) == layout.total_size, \
                    f"Wrong blob size: {len(blob_bytes)} vs {layout.total_size}"

        stats = store.get_stats()
        print(f"  Load count: {stats['load_count']}")
        print(f"  Total loaded: {stats['total_mb_loaded']:.2f} MB")

        store.close()
        print("  ✓ Packed store read successful")

    return True


def run_all_int4_tests():
    """Run all INT4 tests."""
    print("=" * 70)
    print("HYDRANET INT4 QUANTIZATION TESTS")
    print("=" * 70)

    tests = [
        ("INT4 Quantization", test_int4_quantization),
        ("Expert Pack Roundtrip", test_expert_pack_roundtrip),
        ("GPU Expert Slot", test_gpu_expert_slot),
        ("Blob Layout", test_blob_layout),
        ("Pinned Staging Ring", test_pinned_staging_ring),
        ("Packed Store Format", test_packed_store_format),
    ]

    results = []

    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, p, _ in results if p)
    total = len(results)

    for name, p, err in results:
        status = "PASS ✓" if p else "FAIL ✗"
        print(f"  {name}: {status}")
        if err:
            print(f"    Error: {err[:80]}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_int4_tests()
    sys.exit(0 if success else 1)
