#!/usr/bin/env python3
"""
Tests for INT4 expert loading paths.

Tests both:
1. Dequant mode: INT4 -> fp16 at startup
2. Packed mode: INT4 stays packed, dequant on GPU

Uses synthetic packed store (2 layers × 2 experts, small dims) for fast CI.
"""

import torch
import pytest
import tempfile
import json
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hydranet.v2.preprocess.pack_weights import ExpertWeightPacker, QuantConfig
from hydranet.v2.cache.packed_expert_store import PackedExpertStore, BlobLayout
from hydranet.v2.cache.expert_cache import ExpertCacheManager, PerLayerCache
from hydranet.v2.config import MixtralConfig, ExpertCacheConfig


def create_synthetic_packed_store(
    output_dir: Path,
    num_layers: int = 2,
    num_experts: int = 2,
    hidden_dim: int = 256,
    intermediate_dim: int = 512,
    group_size: int = 128,
) -> tuple:
    """
    Create a synthetic packed expert store for testing.

    Returns:
        (store_path, original_weights) where original_weights is a dict
        of (layer_idx, expert_idx) -> {gate_proj, up_proj, down_proj}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate random fp16 weights and pack them
    torch.manual_seed(42)
    packer = ExpertWeightPacker(QuantConfig(group_size=group_size))

    original_weights = {}
    layout = BlobLayout.from_dims(hidden_dim, intermediate_dim, group_size)

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

    bin_path = output_dir / "experts.bin"
    current_offset = 0

    with open(bin_path, "wb") as bin_file:
        for layer_idx in range(num_layers):
            for expert_idx in range(num_experts):
                # Generate random fp16 weights
                weights = {
                    "gate_proj": torch.randn(intermediate_dim, hidden_dim, dtype=torch.float16) * 0.02,
                    "up_proj": torch.randn(intermediate_dim, hidden_dim, dtype=torch.float16) * 0.02,
                    "down_proj": torch.randn(hidden_dim, intermediate_dim, dtype=torch.float16) * 0.02,
                }
                original_weights[(layer_idx, expert_idx)] = weights

                # Pack to INT4
                packed = packer.pack_expert(layer_idx, expert_idx, weights)

                # Write as contiguous blob
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

                bin_file.write(blob)

                index_data["experts"].append({
                    "layer_idx": layer_idx,
                    "expert_idx": expert_idx,
                    "offset": current_offset,
                    "size": layout.total_size,
                })

                current_offset += layout.total_size

    # Write index
    idx_path = output_dir / "experts.idx"
    with open(idx_path, "w") as f:
        json.dump(index_data, f, indent=2)

    return output_dir, original_weights


class TestPackedExpertStore:
    """Tests for PackedExpertStore loading."""

    def test_store_loads_correctly(self, tmp_path):
        """Test that PackedExpertStore loads and provides blobs."""
        store_path, _ = create_synthetic_packed_store(tmp_path / "packed")

        store = PackedExpertStore(
            index_path=store_path / "experts.idx",
            bin_path=store_path / "experts.bin",
            device=torch.device("cpu"),
            use_mmap=True,
        )

        # Check metadata
        assert store.model_config["num_layers"] == 2
        assert store.model_config["num_experts"] == 2
        assert store.model_config["hidden_dim"] == 256

        # Check we can load blobs
        for layer_idx in range(2):
            for expert_idx in range(2):
                blob, layout = store.get_expert_blob(layer_idx, expert_idx)
                assert len(blob) == layout.total_size

        store.close()

    def test_store_stats(self, tmp_path):
        """Test that store tracks load statistics."""
        store_path, _ = create_synthetic_packed_store(tmp_path / "packed")

        store = PackedExpertStore(
            index_path=store_path / "experts.idx",
            device=torch.device("cpu"),
        )

        assert store.load_count == 0

        store.get_expert_blob(0, 0)
        store.get_expert_blob(0, 1)

        assert store.load_count == 2
        assert store.total_bytes_loaded > 0

        store.close()


class TestDequantMode:
    """Tests for startup dequant mode (INT4 -> fp16 at load time)."""

    def test_dequant_matches_reference(self, tmp_path):
        """Test that dequantized weights match ExpertWeightPacker.dequantize_tensor()."""
        store_path, original_weights = create_synthetic_packed_store(
            tmp_path / "packed",
            hidden_dim=256,
            intermediate_dim=512,
            group_size=128,
        )

        store = PackedExpertStore(
            index_path=store_path / "experts.idx",
            device=torch.device("cpu"),
        )

        packer = ExpertWeightPacker(QuantConfig(group_size=128))
        hidden_dim = 256
        intermediate_dim = 512
        group_size = 128

        for layer_idx in range(2):
            for expert_idx in range(2):
                blob_bytes, layout = store.get_expert_blob(layer_idx, expert_idx)
                blob = np.frombuffer(blob_bytes, dtype=np.uint8)

                # Parse and dequantize gate_proj
                gate_packed = torch.from_numpy(
                    blob[layout.gate_packed_offset:layout.gate_packed_offset + layout.gate_packed_size].copy()
                ).view(intermediate_dim, hidden_dim // 2)

                gate_scales_bytes = blob[layout.gate_scales_offset:layout.gate_scales_offset + layout.gate_scales_size]
                gate_scales = torch.from_numpy(
                    np.frombuffer(gate_scales_bytes.tobytes(), dtype=np.float16).copy()
                ).view(intermediate_dim, hidden_dim // group_size)

                gate_dequant = packer.dequantize_tensor(
                    gate_packed, gate_scales, intermediate_dim, hidden_dim
                )

                # Should be close to original (quantization error expected)
                original = original_weights[(layer_idx, expert_idx)]["gate_proj"]
                max_diff = (gate_dequant.float() - original.float()).abs().max().item()

                # INT4 quantization should have <10% max error for random weights
                assert max_diff < 0.1, f"Gate proj max diff {max_diff} too large"

        store.close()


class TestPackedMode:
    """Tests for packed runtime mode (INT4 stays in RAM, dequant on GPU)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_packed_mode_cache_integration(self, tmp_path):
        """Test that packed mode integrates with ExpertCacheManager."""
        store_path, _ = create_synthetic_packed_store(
            tmp_path / "packed",
            num_layers=2,
            num_experts=2,
            hidden_dim=256,
            intermediate_dim=512,
            group_size=128,
        )

        store = PackedExpertStore(
            index_path=store_path / "experts.idx",
            device=torch.device("cuda"),
        )

        # Create cache manager
        config = MixtralConfig()
        config.num_layers = 2
        config.num_experts = 2
        config.hidden_dim = 256
        config.intermediate_dim = 512

        cache_config = ExpertCacheConfig(
            pinned_slots=1,
            hot_slots=1,
            probation_slots=1,
        )

        cache = ExpertCacheManager(
            model_config=config,
            cache_config=cache_config,
            device=torch.device("cuda"),
            dtype=torch.float16,
        )

        # Attach packed store
        cache.set_packed_store(store)

        # Request expert weights - should load from packed store
        weights = cache.get_expert_weights(0, 0)

        assert "gate_proj" in weights
        assert "up_proj" in weights
        assert "down_proj" in weights

        assert weights["gate_proj"].device.type == "cuda"
        assert weights["gate_proj"].shape == (512, 256)

        # Should be cached now
        assert cache.is_cached(0, 0)

        cache.shutdown()
        store.close()

    def test_packed_mode_cpu_fallback(self, tmp_path):
        """Test packed mode works on CPU (for CI without GPU)."""
        store_path, _ = create_synthetic_packed_store(
            tmp_path / "packed",
            num_layers=2,
            num_experts=2,
            hidden_dim=256,
            intermediate_dim=512,
            group_size=128,
        )

        store = PackedExpertStore(
            index_path=store_path / "experts.idx",
            device=torch.device("cpu"),
        )

        # Create cache manager on CPU
        config = MixtralConfig()
        config.num_layers = 2
        config.num_experts = 2
        config.hidden_dim = 256
        config.intermediate_dim = 512

        cache_config = ExpertCacheConfig(
            pinned_slots=1,
            hot_slots=1,
            probation_slots=1,
        )

        cache = ExpertCacheManager(
            model_config=config,
            cache_config=cache_config,
            device=torch.device("cpu"),
            dtype=torch.float16,
        )

        # Attach packed store
        cache.set_packed_store(store)

        # Request expert weights
        weights = cache.get_expert_weights(0, 0)

        assert "gate_proj" in weights
        assert weights["gate_proj"].device.type == "cpu"

        cache.shutdown()
        store.close()


class TestCacheEviction:
    """Tests for cache eviction with packed mode."""

    def test_eviction_works_with_packed(self, tmp_path):
        """Test that cache eviction works correctly with packed store."""
        store_path, _ = create_synthetic_packed_store(
            tmp_path / "packed",
            num_layers=1,
            num_experts=4,  # More experts than slots
            hidden_dim=256,
            intermediate_dim=512,
            group_size=128,
        )

        store = PackedExpertStore(
            index_path=store_path / "experts.idx",
            device=torch.device("cpu"),
        )

        # Create cache with limited slots (will force eviction)
        config = MixtralConfig()
        config.num_layers = 1
        config.num_experts = 4
        config.hidden_dim = 256
        config.intermediate_dim = 512

        cache_config = ExpertCacheConfig(
            pinned_slots=1,
            hot_slots=1,
            probation_slots=1,  # Only 3 slots for 4 experts
        )

        cache = ExpertCacheManager(
            model_config=config,
            cache_config=cache_config,
            device=torch.device("cpu"),
            dtype=torch.float16,
        )

        cache.set_packed_store(store)

        # Load all 4 experts (should cause eviction)
        for expert_idx in range(4):
            weights = cache.get_expert_weights(0, expert_idx)
            assert "gate_proj" in weights

        # Check stats show evictions occurred
        stats = cache.get_stats()
        assert stats["total_evictions"] > 0

        cache.shutdown()
        store.close()


class TestBandwidthEstimate:
    """Tests for bandwidth calculation helpers."""

    def test_int4_vs_fp16_size(self):
        """Verify INT4 is ~4x smaller than fp16."""
        hidden_dim = 4096
        intermediate_dim = 14336
        group_size = 128

        # fp16 expert size
        fp16_size = hidden_dim * intermediate_dim * 3 * 2  # 3 projections, 2 bytes

        # INT4 expert size
        layout = BlobLayout.from_dims(hidden_dim, intermediate_dim, group_size)
        int4_size = layout.total_size

        ratio = fp16_size / int4_size
        print(f"\nfp16 size: {fp16_size / 1024 / 1024:.1f} MB")
        print(f"INT4 size: {int4_size / 1024 / 1024:.1f} MB")
        print(f"Compression ratio: {ratio:.2f}x")

        # Should be roughly 3-4x smaller (scales add overhead)
        assert 2.5 < ratio < 4.5, f"Unexpected compression ratio: {ratio}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
