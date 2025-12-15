#!/usr/bin/env python3
"""Test script to verify HydraNet model functionality."""

import os
import torch
import time
import sys
from pathlib import Path

# ========== RESOURCE LIMITS ==========
# Reserve 2 CPU threads and 6GB RAM for system stability
RESERVED_THREADS = 2
RESERVED_RAM_GB = 6
MAX_CPU_THREADS = max(1, os.cpu_count() - RESERVED_THREADS) if os.cpu_count() else 4

torch.set_num_threads(MAX_CPU_THREADS)
torch.set_num_interop_threads(max(1, MAX_CPU_THREADS // 2))
os.environ["OMP_NUM_THREADS"] = str(MAX_CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MAX_CPU_THREADS)
# =====================================

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydranet.model import HydraNetConfig, HydraNetConfigs, HydraNet, create_hydranet


def test_config():
    """Test configuration classes."""
    print("\n" + "="*60)
    print("Testing Configuration")
    print("="*60)

    for name in ["tiny", "small", "medium", "large"]:
        config_fn = getattr(HydraNetConfigs, name)
        config = config_fn()
        print(f"\n{name.upper()}:")
        print(f"  Total params:  {config.total_params_billions:.2f}B")
        print(f"  Active params: {config.active_params_billions:.2f}B")
        print(f"  Expert size:   {config.expert_size_mb:.1f}MB (4-bit)")
        print(f"  KV cache:      {config.kv_cache_size_mb:.0f}MB (fp16)")


def test_model_creation():
    """Test model instantiation."""
    print("\n" + "="*60)
    print("Testing Model Creation")
    print("="*60)

    # Create tiny model for testing
    config = HydraNetConfigs.tiny()
    print(f"\nCreating tiny model ({config.total_params_billions:.2f}B params)...")

    model = HydraNet(config)
    num_params = model.num_parameters()
    print(f"Model created with {num_params:,} parameters")

    return model, config


def test_forward_pass(model: HydraNet, config: HydraNetConfig):
    """Test forward pass."""
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    model.eval()

    # Test input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    print(f"\nInput shape: {input_ids.shape}")

    # Forward pass
    with torch.no_grad():
        start = time.time()
        outputs = model(input_ids)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.time() - start

    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Forward pass time: {elapsed*1000:.2f}ms")

    return outputs


def test_generation(model: HydraNet, config: HydraNetConfig):
    """Test token generation."""
    print("\n" + "="*60)
    print("Testing Generation")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Starting tokens
    input_ids = torch.randint(0, config.vocab_size, (1, 32), device=device)

    print(f"Prompt length: {input_ids.shape[1]} tokens")
    print("Generating 64 tokens...")

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=64,
            temperature=0.8,
            top_k=50,
        )
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.time() - start

    generated = output_ids.shape[1] - input_ids.shape[1]
    tokens_per_sec = generated / elapsed

    print(f"Generated {generated} tokens in {elapsed:.2f}s")
    print(f"Speed: {tokens_per_sec:.1f} tokens/second")


def test_kv_cache(model: HydraNet, config: HydraNetConfig):
    """Test KV cache functionality."""
    print("\n" + "="*60)
    print("Testing KV Cache")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Initial prompt
    input_ids = torch.randint(0, config.vocab_size, (1, 64), device=device)

    # First forward (prefill)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values

    print(f"Initial prompt: {input_ids.shape[1]} tokens")
    print(f"KV cache layers: {len(past_kv)}")
    print(f"KV shape per layer: {past_kv[0][0].shape}")

    # Incremental decode
    new_token = torch.randint(0, config.vocab_size, (1, 1), device=device)

    with torch.no_grad():
        start = time.time()
        outputs = model(new_token, past_key_values=past_kv, use_cache=True)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.time() - start

    print(f"\nIncremental decode time: {elapsed*1000:.2f}ms")
    print(f"Estimated decode speed: {1000/elapsed:.0f} tokens/second")


def test_memory_usage():
    """Test memory usage for different model sizes."""
    print("\n" + "="*60)
    print("Testing Memory Usage")
    print("="*60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return

    device = torch.device("cuda")

    for name in ["tiny", "small"]:  # Don't test larger on limited VRAM
        config_fn = getattr(HydraNetConfigs, name)
        config = config_fn()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        print(f"\n{name.upper()} model:")

        try:
            model = HydraNet(config).to(device).half()

            # Forward pass
            input_ids = torch.randint(0, config.vocab_size, (1, 256), device=device)
            with torch.no_grad():
                _ = model(input_ids)

            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            print(f"  Peak memory: {peak_mb:.0f}MB")

            del model
            torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"  Failed: {e}")


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# HydraNet Model Tests")
    print("#"*60)

    # Test configuration
    test_config()

    # Test model creation
    model, config = test_model_creation()

    # Test forward pass
    test_forward_pass(model, config)

    # Test generation
    test_generation(model, config)

    # Test KV cache
    test_kv_cache(model, config)

    # Test memory usage
    test_memory_usage()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
