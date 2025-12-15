#!/usr/bin/env python3
"""Test actual text generation with packed INT4 and KV cache."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    import warnings
    # Suppress torch.frombuffer warning for read-only buffers (benign - we only read)
    warnings.filterwarnings("ignore", message="The given buffer is not writable")

    import torch
    from hydranet.v2.config import MixtralConfig, ExpertCacheConfig, KVCacheConfig
    from hydranet.v2.model.mixtral import OffloadedMixtral
    from hydranet.v2.model.loader import MixtralWeightLoader
    from hydranet.v2.cache.packed_expert_store import PackedExpertStore
    from hydranet.v2.cache.kv_cache import SimpleKVCache
    from transformers import AutoTokenizer
    import time

    # Clean up any leftover GPU memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda")
    dtype = torch.float16
    model_path = "/home/k/models/mixtral-8x7b-instruct"
    packed_dir = Path("/home/k/models/mixtral-8x7b-instruct/packed_int4")

    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["packed", "dequant"], default="packed")
    parser.add_argument("--tokens", type=int, default=16, help="Number of tokens to generate")
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable KV cache (slow)")
    parser.add_argument("--top2-decode", action="store_true", help="Use Top-2 for decode (slower)")
    parser.add_argument("--profile", action="store_true", help="Enable per-token profiling")
    parser.add_argument("--slots", type=int, default=3, help="Base slots per layer")
    parser.add_argument("--packed-backend", choices=["ram", "mmap"], default="ram",
                        help="Expert storage backend: ram (fast, +22GB RAM) or mmap (slower, low RAM)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"TEXT GENERATION TEST")
    print(f"  Mode: {args.mode}, KV cache: {'OFF' if args.no_kv_cache else 'ON'}")
    print(f"  Decode top-k: {'2' if args.top2_decode else '1'}")
    print(f"  Base slots/layer: {args.slots}")
    print("=" * 60)

    # Config
    config = MixtralConfig()
    cache_config = ExpertCacheConfig(
        pinned_slots=0,
        hot_slots=args.slots // 2 + args.slots % 2,
        probation_slots=args.slots // 2,
        enable_prefetch=False,
    )
    kv_config = KVCacheConfig()

    # Model
    print("\nLoading model...")
    model = OffloadedMixtral(
        config=config,
        cache_config=cache_config,
        kv_config=kv_config,
        device=device,
        dtype=dtype,
    )

    # Weights
    loader = MixtralWeightLoader(model_path, config, device, dtype)
    weights = loader.load_non_expert_weights()
    model.load_weights(weights)
    del weights
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Base model: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    packed_store = None
    if args.mode == "packed":
        use_mmap = args.packed_backend == "mmap"
        if use_mmap:
            print("Using mmap backend (slower, lower RAM usage)...")
        else:
            print("Loading experts.bin into RAM (one-time startup cost)...")
        load_start = time.perf_counter()
        packed_store = PackedExpertStore(
            index_path=packed_dir / "experts.idx",
            bin_path=packed_dir / "experts.bin",
            device=device,
            use_mmap=use_mmap,
        )
        load_time = time.perf_counter() - load_start
        print(f"  Loaded in {load_time:.1f}s")
        model.expert_cache.set_packed_store(packed_store)

    # KV Cache
    kv_cache = None
    if not args.no_kv_cache:
        kv_cache = SimpleKVCache.from_model_config(
            config, max_seq_len=2048, batch_size=1, device=device, dtype=dtype
        )
        print(f"  KV cache: {kv_cache.memory_mb():.1f} MB")

    # Show slot allocation
    total_slots = sum(lc.total_slots for lc in model.expert_cache.layer_caches)
    slot_dist = {}
    for lc in model.expert_cache.layer_caches:
        slot_dist[lc.total_slots] = slot_dist.get(lc.total_slots, 0) + 1
    slot_str = ", ".join(f"{v}×{k}-slot" for k, v in sorted(slot_dist.items()))
    print(f"  Expert slots: {total_slots} total ({slot_str})")
    print(f"  After setup: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Show VRAM headroom
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"  VRAM headroom: {free_b/1e9:.2f} GB free / {total_b/1e9:.2f} GB total")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model loaded!")

    # Generate
    prompt = "[INST] Hi [/INST]"
    print(f"\nPrompt: {prompt}")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Input tokens: {input_ids.shape[1]}")

    model.eval()
    generated = input_ids.clone()

    # Timing tracking
    token_times = []
    prefill_time = 0
    prefill_stats = None  # Will be set after prefill

    # Reset cache stats before generation
    model.expert_cache.reset_stats()

    print("\nGenerating tokens:")
    with torch.no_grad():
        for i in range(args.tokens):
            t_start = time.perf_counter()

            if kv_cache is not None:
                # Efficient path: feed only new token (except first)
                if i == 0:
                    curr_input = generated  # Prefill: full prompt
                    top_k = 2  # Always Top-2 for prefill
                else:
                    # Reset stats after prefill to measure decode-only hit rate
                    if i == 1:
                        prefill_stats = model.expert_cache.get_stats()
                        model.expert_cache.reset_stats()
                    curr_input = next_token  # Decode: single token
                    top_k = 2 if args.top2_decode else 1  # Top-1 or Top-2 for decode

                logits, _ = model.forward(
                    curr_input,
                    kv_cache=kv_cache,
                    force_top_k=top_k,
                )
            else:
                # Slow path: feed full history each step
                logits, _ = model.forward(
                    generated,
                    past_key_values=None,
                    use_cache=False,
                    force_top_k=2,
                )

            # Greedy: take argmax
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000

            if i == 0:
                prefill_time = elapsed_ms
            else:
                token_times.append(elapsed_ms)

            # Decode and print
            token_str = tokenizer.decode(next_token[0])
            if args.profile:
                print(f"  {i+1}. '{token_str}' ({elapsed_ms:.0f}ms)")
            else:
                print(f"  {i+1}. Token {next_token.item()}: '{token_str}'")

    total_time = prefill_time + sum(token_times)
    new_tokens = generated.shape[1] - input_ids.shape[1]

    # Decode full response
    response = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nFull response: {response}")

    # Timing stats
    print(f"\n=== Timing ===")
    print(f"Prefill ({input_ids.shape[1]} tokens): {prefill_time:.0f}ms")
    if token_times:
        avg_decode = sum(token_times) / len(token_times)
        print(f"Decode ({len(token_times)} tokens): {avg_decode:.0f}ms avg, {min(token_times):.0f}ms min, {max(token_times):.0f}ms max")
        print(f"Decode throughput: {1000/avg_decode:.2f} tok/s")
    print(f"Total: {total_time/1000:.1f}s for {new_tokens} tokens ({new_tokens/(total_time/1000):.2f} tok/s)")

    # Expert cache stats - decode only (after cache is warm)
    decode_stats = model.expert_cache.get_stats()
    print(f"\n=== Expert Cache ===")

    # Show prefill stats if available
    if prefill_stats is not None:
        print(f"Prefill: {prefill_stats['hit_rate']:.1%} hit rate ({prefill_stats['total_hits']} hits, {prefill_stats['total_misses']} misses)")

    # Show decode stats (the meaningful metric - cache is warm)
    print(f"Decode:  {decode_stats['hit_rate']:.1%} hit rate ({decode_stats['total_hits']} hits, {decode_stats['total_misses']} misses)")
    print(f"  Evictions: {decode_stats['total_evictions']}")
    if decode_stats['total_misses'] > 0:
        print(f"  Avg load time: {decode_stats['avg_load_time_ms']:.1f}ms")

    # Per-layer miss rates (identify hot layers for slot reallocation)
    if args.profile:
        layer_stats = decode_stats['per_layer_stats']
        # Show top 8 worst layers by miss count
        sorted_by_misses = sorted(layer_stats, key=lambda s: -s['misses'])
        print(f"\n  Top 8 layers by misses:")
        for s in sorted_by_misses[:8]:
            slots = model.expert_cache.layer_caches[s['layer']].total_slots
            print(f"    L{s['layer']:2d}: {1-s['hit_rate']:4.0%} miss, {s['misses']:3d} misses, {slots} slots")

        # Find best donor candidates (4-slot layers with lowest misses)
        donors = [(s['misses'], s['layer']) for s in layer_stats
                  if model.expert_cache.layer_caches[s['layer']].total_slots == 4]
        donors.sort()
        print(f"\n  Best donor layers (4-slot, lowest misses):")
        for misses, layer in donors[:4]:
            print(f"    L{layer:2d}: {misses:3d} misses")

    print(f"\nFinal GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Cleanup
    model.expert_cache.shutdown()
    if packed_store:
        packed_store.close()


if __name__ == "__main__":
    main()
