#!/usr/bin/env python3
"""Test GLM-4.5-Air text generation with packed INT4 and KV cache."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    import warnings
    warnings.filterwarnings("ignore", message="The given buffer is not writable")

    import torch
    import gc
    import time
    import argparse

    from hydranet.config import GLM4AirConfig, ExpertCacheConfig
    from hydranet.model.glm4 import OffloadedGLM4
    from hydranet.model.glm4_loader import GLM4WeightLoader
    from hydranet.cache.packed_expert_store import PackedExpertStore
    from hydranet.cache.kv_cache import SimpleKVCache
    from hydranet.kernels.int8_linear import convert_model_to_int8, estimate_int8_memory
    from hydranet.kernels.int4_linear import convert_model_to_int4, estimate_int4_memory

    parser = argparse.ArgumentParser(description="GLM-4.5-Air generation test")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to GLM-4.5-Air weights")
    parser.add_argument("--packed-dir", type=str, default=None,
                        help="Path to packed INT4 experts (default: model_path/packed_int4)")
    parser.add_argument("--tokens", type=int, default=16,
                        help="Number of tokens to generate")
    parser.add_argument("--slots", type=int, default=8,
                        help="Base slots per layer (GLM4 has small experts, can use more)")
    parser.add_argument("--packed-backend", choices=["ram", "mmap"], default="ram",
                        help="Expert storage backend")
    parser.add_argument("--no-kv-cache", action="store_true",
                        help="Disable KV cache")
    parser.add_argument("--profile", action="store_true",
                        help="Enable per-token profiling")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                        help="Input prompt")
    parser.add_argument("--debug-weights", action="store_true",
                        help="Debug weight loading")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Limit number of layers (for memory-constrained testing)")
    parser.add_argument("--int8", action="store_true",
                        help="Use INT8 for non-expert weights (saves ~50% VRAM)")
    parser.add_argument("--int4", action="store_true",
                        help="Use INT4 for non-expert weights (saves ~75% VRAM, more quality loss)")
    parser.add_argument("--hybrid", action="store_true",
                        help="Use INT8 for attention, INT4 for MLP (balanced speed/memory)")
    parser.add_argument("--fp16-experts", action="store_true",
                        help="Use FP16 cuBLAS for expert compute (4x faster, but caches dequantized weights)")
    parser.add_argument("--kv-size", type=int, default=2048,
                        help="KV cache max sequence length (default: 2048)")
    parser.add_argument("--speculative", type=int, default=0, metavar="K",
                        help="Enable speculative decoding with K draft tokens (0=disabled)")
    args = parser.parse_args()

    # Clean up GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model_path = Path(args.model_path)
    packed_dir = Path(args.packed_dir) if args.packed_dir else model_path / "packed_int4"

    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("=" * 60)
    print("GLM-4.5-Air GENERATION TEST")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}, dtype: {dtype}")
    print(f"  Slots/layer: {args.slots}")
    print(f"  KV cache: {'OFF' if args.no_kv_cache else 'ON'}")
    print("=" * 60)

    # Config
    config = GLM4AirConfig()

    # Optionally limit layers for memory-constrained testing
    if args.max_layers is not None:
        config.num_layers = min(args.max_layers, config.num_layers)
        print(f"  Limited to {config.num_layers} layers for testing")

    cache_config = ExpertCacheConfig(
        pinned_slots=0,
        hot_slots=args.slots // 2 + args.slots % 2,
        probation_slots=args.slots // 2,
        enable_prefetch=True,  # Enable async prefetch for overlap
    )

    print(f"\nModel config:")
    print(f"  Layers: {config.num_layers} ({config.first_k_dense_replace} dense + {config.num_moe_layers} MoE)")
    print(f"  Experts: {config.num_experts} routed + {config.num_shared_experts} shared")
    print(f"  Top-k: {config.experts_per_token}")
    print(f"  Expert size: {config.expert_size_mb:.2f} MB (INT4)")

    # Initialize model
    print("\nInitializing model...")
    model = OffloadedGLM4(
        config=config,
        cache_config=cache_config,
        device=device,
        dtype=dtype,
    )
    print(f"  After init: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load weights
    print("\nLoading weights...")
    loader = GLM4WeightLoader(model_path, config, device=torch.device("cpu"), dtype=dtype)

    if args.debug_weights:
        # Debug: print model parameter names
        print("\nModel parameter names (first 20):")
        for i, (name, param) in enumerate(model.named_parameters()):
            if i < 20:
                print(f"  {name}: {param.shape}")
            elif i == 20:
                print(f"  ... and {sum(1 for _ in model.named_parameters()) - 20} more")
                break

    # Load non-expert weights
    hf_weights = loader.load_non_expert_weights()

    if args.debug_weights:
        print("\nLoader weight names (first 20):")
        for i, name in enumerate(sorted(hf_weights.keys())):
            if i < 20:
                print(f"  {name}: {hf_weights[name].shape}")
            elif i == 20:
                print(f"  ... and {len(hf_weights) - 20} more")
                break

    # Map HF weight names to model parameter names
    weights = map_weights_to_model(hf_weights, config)
    del hf_weights

    # Load weights into model
    missing, unexpected = load_weights_with_report(model, weights)
    if missing:
        print(f"  Warning: {len(missing)} missing weights")
        if args.debug_weights:
            for m in missing[:10]:
                print(f"    - {m}")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected weights")

    del weights
    gc.collect()
    torch.cuda.empty_cache()

    # Convert to INT4/INT8 if requested (before moving to GPU)
    use_quantized = args.int4 or args.int8 or args.hybrid
    if args.hybrid:
        print("\n  Converting to hybrid INT8/INT4...")
        # INT8 for attention projections (hot path), INT4 for rest
        # First convert everything to INT4
        convert_model_to_int4(model, skip_layers=["lm_head", "embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj"])
        # Then convert attention projections to INT8 (overwrite INT4)
        convert_model_to_int8(model, skip_layers=["lm_head", "shared_expert", "gate", "mlp"])
        gc.collect()
        print("  Hybrid conversion complete (INT8 attention, INT4 MLP)")
    elif args.int4:
        print("\n  Converting to INT4...")
        mem_estimate = estimate_int4_memory(model)
        print(f"    fp16: {mem_estimate['fp16_mb']:.0f} MB -> int4: {mem_estimate['int4_mb']:.0f} MB")
        print(f"    Savings: {mem_estimate['savings_mb']:.0f} MB ({mem_estimate['savings_mb']/mem_estimate['fp16_mb']*100:.0f}%)")

        # Skip lm_head and embeddings to preserve output quality
        convert_model_to_int4(model, skip_layers=["lm_head", "embed_tokens"])
        gc.collect()
        print("  INT4 conversion complete")
    elif args.int8:
        print("\n  Converting to INT8...")
        mem_estimate = estimate_int8_memory(model)
        print(f"    fp16: {mem_estimate['fp16_mb']:.0f} MB -> int8: {mem_estimate['int8_mb']:.0f} MB")
        print(f"    Savings: {mem_estimate['savings_mb']:.0f} MB ({mem_estimate['savings_mb']/mem_estimate['fp16_mb']*100:.0f}%)")

        # Skip lm_head to preserve output quality
        convert_model_to_int8(model, skip_layers=["lm_head"])
        gc.collect()
        print("  INT8 conversion complete")

    # Move model to GPU and convert to correct dtype
    print("  Moving model to GPU...")
    if use_quantized:
        # Quantized layers handle their own dtypes, just move to device
        model.to(device=device)
        # Ensure all non-quantized parameters are fp16 (LayerNorms, lm_head, etc.)
        for name, param in model.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(dtype)
        for name, buf in model.named_buffers():
            # Skip quantized weights (they're int8/int4), only convert float32 buffers
            if buf.dtype == torch.float32:
                # Can't modify buffer in-place for some, so skip
                pass
    else:
        model.to(device=device, dtype=dtype)
    torch.cuda.empty_cache()
    print(f"  After weight load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load packed expert store
    if packed_dir.exists() and (packed_dir / "experts.idx").exists():
        print(f"\nLoading packed experts from {packed_dir}...")
        use_mmap = args.packed_backend == "mmap"
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

        # Enable FP16 compute mode if requested (uses cuBLAS instead of INT4 Triton)
        if args.fp16_experts:
            model.expert_cache.use_fp16_compute = True
            print("  FP16 expert compute enabled (dequant + cuBLAS)")
    else:
        print(f"\nWarning: No packed experts at {packed_dir}")
        print("  Run pack_glm4.py first to create INT4 experts")
        packed_store = None

    # KV Cache
    kv_cache = None
    if not args.no_kv_cache:
        kv_cache = SimpleKVCache.from_model_config(
            config, max_seq_len=args.kv_size, batch_size=1, device=device, dtype=dtype
        )
        print(f"  KV cache: {kv_cache.memory_mb():.1f} MB")

    # Tokenizer
    print("\nLoading tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"  Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"  Warning: Could not load tokenizer: {e}")
        print("  Using dummy tokenization")
        tokenizer = None

    # Show final memory
    total_slots = sum(lc.total_slots for lc in model.expert_cache.layer_caches)
    print(f"\n  Expert slots: {total_slots} total")
    print(f"  Final setup: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    free_b, total_b = torch.cuda.mem_get_info()
    print(f"  VRAM headroom: {free_b/1e9:.2f} GB free / {total_b/1e9:.2f} GB total")

    # Generate
    print(f"\n{'=' * 60}")
    print(f"Prompt: {args.prompt}")

    if tokenizer:
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    else:
        # Dummy tokenization
        input_ids = torch.randint(0, config.vocab_size, (1, 10), device=device)

    print(f"Input tokens: {input_ids.shape[1]}")

    model.eval()
    model.expert_cache.reset_stats()

    print(f"\nGenerating {args.tokens} tokens...")
    token_times = []
    prefill_time = 0
    generated = input_ids.clone()
    spec_stats = None

    # Use speculative decoding if requested
    if args.speculative > 0 and kv_cache is not None:
        from hydranet.inference.speculative import speculative_generate
        print(f"  Using speculative decoding with {args.speculative} draft tokens")

        t_start = time.perf_counter()
        generated, spec_stats = speculative_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            kv_cache=kv_cache,
            max_new_tokens=args.tokens,
            num_speculative=args.speculative,
            temperature=0.0,
            verbose=args.profile,
        )
        t_end = time.perf_counter()

        total_time = (t_end - t_start) * 1000
        new_tokens = generated.shape[1] - input_ids.shape[1]
        prefill_time = total_time * 0.1  # Rough estimate
        token_times = [(total_time - prefill_time) / max(1, new_tokens - 1)] * max(0, new_tokens - 1)

    else:
        # Standard autoregressive decoding
        with torch.no_grad():
            for i in range(args.tokens):
                t_start = time.perf_counter()

                if kv_cache is not None:
                    if i == 0:
                        curr_input = generated  # Prefill
                    else:
                        curr_input = next_token  # Decode

                    logits, _ = model.forward(curr_input, kv_cache=kv_cache)

                    # Update KV cache position tracking
                    if i == 0:
                        kv_cache.set_len(curr_input.shape[1])  # After prefill
                    else:
                        kv_cache.advance(1)  # After each decode step
                else:
                    logits, _ = model.forward(generated)

                # Greedy decode
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                t_end = time.perf_counter()
                elapsed_ms = (t_end - t_start) * 1000

                if i == 0:
                    prefill_time = elapsed_ms
                else:
                    token_times.append(elapsed_ms)

                # Decode and print
                if tokenizer:
                    token_str = tokenizer.decode(next_token[0])
                else:
                    token_str = f"<{next_token.item()}>"

                if args.profile:
                    print(f"  {i+1}. '{token_str}' ({elapsed_ms:.0f}ms)")
                else:
                    print(f"  {i+1}. Token {next_token.item()}: '{token_str}'")

    # Stats
    total_time = prefill_time + sum(token_times)
    new_tokens = generated.shape[1] - input_ids.shape[1]

    if tokenizer:
        response = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\nFull response: {response}")

    print(f"\n{'=' * 60}")
    print("TIMING")
    print(f"  Prefill ({input_ids.shape[1]} tokens): {prefill_time:.0f}ms")
    if token_times:
        avg_decode = sum(token_times) / len(token_times)
        print(f"  Decode ({len(token_times)} tokens): {avg_decode:.0f}ms avg")
        print(f"  Decode throughput: {1000/avg_decode:.2f} tok/s")
    print(f"  Total: {total_time/1000:.1f}s for {new_tokens} tokens")

    # Cache stats
    stats = model.expert_cache.get_stats()
    print(f"\nEXPERT CACHE")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Hits: {stats['total_hits']}, Misses: {stats['total_misses']}")
    print(f"  Evictions: {stats['total_evictions']}")
    if stats['total_misses'] > 0:
        print(f"  Avg load time: {stats['avg_load_time_ms']:.1f}ms")
        # Time breakdown estimate
        total_load_ms = stats['total_misses'] * stats['avg_load_time_ms']
        total_decode_ms = sum(token_times) if token_times else 0
        if total_decode_ms > 0:
            print(f"\n  TIME BREAKDOWN (decode only):")
            print(f"    Expert loading: {total_load_ms:.0f}ms ({100*total_load_ms/total_decode_ms:.1f}%)")
            print(f"    Compute: {total_decode_ms - total_load_ms:.0f}ms ({100*(1-total_load_ms/total_decode_ms):.1f}%)")

    # Speculative decoding stats
    if spec_stats is not None:
        print(f"\nSPECULATIVE DECODING")
        print(f"  Acceptance rate: {spec_stats['acceptance_rate']:.1%}")
        print(f"  Avg tokens/step: {spec_stats['avg_tokens_per_step']:.2f}")
        print(f"  Total drafted: {spec_stats['total_drafted']}, accepted: {spec_stats['total_accepted']}")
        print(f"  Draft time: {spec_stats['draft_time_ms']:.1f}ms, Verify time: {spec_stats['verify_time_ms']:.1f}ms")

    print(f"\nFinal GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Cleanup
    model.expert_cache.shutdown()
    if packed_store:
        packed_store.close()
    loader.close()


def map_weights_to_model(hf_weights: dict, config) -> dict:
    """Map HuggingFace weight names to model parameter names."""
    mapped = {}

    for hf_name, tensor in hf_weights.items():
        # Direct mappings
        if hf_name == "embed_tokens.weight":
            mapped["embed_tokens.weight"] = tensor
        elif hf_name == "lm_head.weight":
            mapped["lm_head.weight"] = tensor
        elif hf_name == "norm.weight":
            mapped["norm.weight"] = tensor

        # Layer mappings
        elif hf_name.startswith("layers."):
            parts = hf_name.split(".")
            layer_idx = int(parts[1])

            # Attention
            if "self_attn" in hf_name:
                # layers.0.self_attn.q_proj.weight -> layers.0.self_attn.q_proj.weight
                mapped[hf_name] = tensor

            # Layer norms
            elif "layernorm" in hf_name:
                # layers.0.input_layernorm.weight -> layers.0.input_layernorm.weight
                mapped[hf_name] = tensor

            # MLP (dense layer 0)
            elif "mlp.gate_proj" in hf_name or "mlp.up_proj" in hf_name or "mlp.down_proj" in hf_name:
                # Dense MLP for layer 0
                if layer_idx < config.first_k_dense_replace:
                    mapped[hf_name] = tensor

            # Router
            elif "mlp.router" in hf_name:
                # layers.1.mlp.router.weight -> layers.1.mlp.router.weight
                mapped[hf_name] = tensor

            # Shared expert
            elif "mlp.shared_expert" in hf_name:
                # Map to shared_expert in MoE layer
                mapped[hf_name] = tensor

    return mapped


def load_weights_with_report(model, weights: dict):
    """Load weights and report missing/unexpected."""
    model_params = {name for name, _ in model.named_parameters()}
    weight_names = set(weights.keys())

    missing = model_params - weight_names
    unexpected = weight_names - model_params

    # Load matching weights
    loaded = 0
    for name, param in model.named_parameters():
        if name in weights:
            param.data.copy_(weights[name].to(param.dtype).to(param.device))
            loaded += 1

    print(f"  Loaded {loaded}/{len(model_params)} parameters")
    return list(missing), list(unexpected)


if __name__ == "__main__":
    main()
