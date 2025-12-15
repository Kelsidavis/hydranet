#!/usr/bin/env python3
"""Debug memory usage step by step."""

import torch
import gc
import sys
from pathlib import Path

# Add parent directory to path so 'hydranet' package is found
sys.path.insert(0, str(Path(__file__).parent.parent))

def print_mem(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{label}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def main():
    from hydranet.v2.config import MixtralConfig, ExpertCacheConfig, KVCacheConfig
    from hydranet.v2.model.mixtral import OffloadedMixtral
    from hydranet.v2.model.loader import MixtralWeightLoader
    from hydranet.v2.cache.packed_expert_store import PackedExpertStore

    device = torch.device("cuda")
    dtype = torch.float16
    model_path = "/home/k/models/mixtral-8x7b-instruct"
    packed_dir = Path("/home/k/models/mixtral-8x7b-instruct/packed_int4")

    print("=" * 60)
    print("MEMORY DEBUG")
    print("=" * 60)

    # Calculate memory requirements
    hidden = 4096
    intermediate = 14336
    expert_mb = (hidden * intermediate * 3 * 2) / (1024 * 1024)
    print(f"\nExpert size (fp16): {expert_mb:.1f} MB")
    print(f"GPU total: 16 GB")
    print(f"Available for experts: ~12 GB")
    print(f"Max experts in memory: {12 * 1024 / expert_mb:.0f} (~34)")
    print(f"Per layer (32 layers): ~1 expert/layer recommended")

    torch.cuda.empty_cache()
    gc.collect()
    print_mem("Start")

    # Load config with MINIMAL slot budget
    config = MixtralConfig()

    # Memory budget: 16GB total - 4GB base = ~12GB for experts
    # Each expert ~336 MB, so max ~35 experts total
    # With 32 layers, that's ~1 expert/layer
    # Use probation (evictable) slots, not pinned
    cache_config = ExpertCacheConfig(
        pinned_slots=0,      # No permanent slots
        hot_slots=0,         # No sticky slots
        probation_slots=1,   # 1 evictable slot per layer
        enable_prefetch=False,  # Disable async prefetch to avoid race conditions
    )
    kv_config = KVCacheConfig()

    print(f"\nSlots per layer: {cache_config.slots_per_layer}")
    print(f"Total expert VRAM budget: {32 * cache_config.slots_per_layer * expert_mb / 1024:.1f} GB")

    print("\nCreating model...")
    model = OffloadedMixtral(
        config=config,
        cache_config=cache_config,
        kv_config=kv_config,
        device=device,
        dtype=dtype,
    )
    print_mem("After model init")

    print("\nLoading non-expert weights...")
    loader = MixtralWeightLoader(model_path, config, device, dtype)
    weights = loader.load_non_expert_weights()
    model.load_weights(weights)
    del weights
    gc.collect()
    torch.cuda.empty_cache()
    print_mem("After loading weights")

    print("\nAttaching packed store...")
    packed_store = PackedExpertStore(
        index_path=packed_dir / "experts.idx",
        bin_path=packed_dir / "experts.bin",
        device=device,
        use_mmap=True,
    )
    model.expert_cache.set_packed_store(packed_store)
    print_mem("After packed store")

    print("\nValidating router weights...")
    model.validate_weights(verbose=True)
    print_mem("After validation")

    # Skip manual expert loading - go straight to forward pass
    # With 1 slot per layer, we use top_k=1 (decode mode)
    print("\n--- Testing minimal forward pass (top_k=1) ---")
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)

    gc.collect()
    torch.cuda.empty_cache()
    print_mem("Before forward")

    try:
        with torch.no_grad():
            # Step through layer by layer
            hidden_states = model.embed_tokens(input_ids)
            print_mem("After embed")

            position_ids = torch.arange(4, device=device).unsqueeze(0)
            attention_mask = model._make_causal_mask(4, 4)

            for layer_idx, layer in enumerate(model.layers):
                # Attention only
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
                hidden_states, _ = layer.self_attn(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )
                hidden_states = residual + hidden_states

                # MoE - use top_k=1 for decode mode (1 slot per layer)
                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                hidden_states = layer.moe(hidden_states, top_k=1)
                hidden_states = residual + hidden_states

                # Clear fp16 cache after each layer to free GPU memory
                # (keeps INT4 for fast re-dequant)
                model.expert_cache.clear_layer_fp16(layer_idx)

                if layer_idx % 8 == 0:
                    torch.cuda.synchronize()
                    gc.collect()
                    print_mem(f"After layer {layer_idx}")

            hidden_states = model.norm(hidden_states)
            logits = model.lm_head(hidden_states)
            print_mem("After forward complete")
            print(f"Logits shape: {logits.shape}")
            print("SUCCESS!")

    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM: {e}")
        print_mem("At OOM")

    finally:
        # Cleanup
        model.expert_cache.shutdown()
        packed_store.close()


if __name__ == "__main__":
    main()
