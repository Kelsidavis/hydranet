#!/usr/bin/env python3
"""HydraNet CLI - GLM-4.5-Air inference on consumer GPUs."""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        prog="hydranet",
        description="GLM-4.5-Air MoE inference runtime for consumer GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hydranet generate --16gb --prompt "Hello, how are you?"
  hydranet generate --16gb --tokens 100 --prompt "Write a story about:"
  hydranet generate --int4 --slots 15 --prompt "Explain quantum computing"
  hydranet info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", aliases=["gen", "g"], help="Generate text")
    gen_parser.add_argument("--model-path", type=str,
                            default="/media/k/2tb nvme/models/glm4-air",
                            help="Path to GLM-4.5-Air weights")
    gen_parser.add_argument("--packed-dir", type=str, default=None,
                            help="Path to packed INT4 experts")
    gen_parser.add_argument("--prompt", "-p", type=str, default="Hello, how are you?",
                            help="Input prompt")
    gen_parser.add_argument("--tokens", "-n", type=int, default=50,
                            help="Number of tokens to generate")
    gen_parser.add_argument("--slots", type=int, default=8,
                            help="Expert cache slots per layer")

    # Quantization options
    quant_group = gen_parser.add_argument_group("quantization")
    quant_group.add_argument("--int4", action="store_true",
                             help="Use INT4 quantization (74%% memory savings)")
    quant_group.add_argument("--int8", action="store_true",
                             help="Use INT8 quantization (50%% savings)")
    quant_group.add_argument("--hybrid", action="store_true",
                             help="INT8 attention + INT4 MLP")
    quant_group.add_argument("--int8-kv", action="store_true",
                             help="Use INT8 KV cache")
    quant_group.add_argument("--quantize-all", action="store_true",
                             help="Also quantize lm_head (saves 0.6GB)")

    # GPU presets
    preset_group = gen_parser.add_argument_group("GPU presets")
    preset_group.add_argument("--16gb", action="store_true", dest="gpu_16gb",
                              help="Optimized for 16GB GPU (~1.3 tok/s)")
    preset_group.add_argument("--24gb", action="store_true", dest="gpu_24gb",
                              help="Optimized for 24GB GPU")

    # Performance options
    perf_group = gen_parser.add_argument_group("performance")
    perf_group.add_argument("--decode-skip", type=float, default=0.0,
                            help="Skip ratio of MoE layers during decode")
    perf_group.add_argument("--kv-size", type=int, default=2048,
                            help="KV cache max sequence length")
    perf_group.add_argument("--speculative", type=int, default=0,
                            help="Speculative decoding draft tokens")
    perf_group.add_argument("--fp16-experts", action="store_true",
                            help="Use FP16 for expert compute")

    # Debug options
    debug_group = gen_parser.add_argument_group("debug")
    debug_group.add_argument("--profile", action="store_true",
                             help="Enable profiling")
    debug_group.add_argument("--no-kv-cache", action="store_true",
                             help="Disable KV cache")
    debug_group.add_argument("--max-layers", type=int, default=None,
                             help="Limit layers for testing")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system info")
    info_parser.add_argument("--model-path", type=str,
                             default="/media/k/2tb nvme/models/glm4-air",
                             help="Path to model")

    # Benchmark command
    bench_parser = subparsers.add_parser("bench", help="Run benchmarks")
    bench_parser.add_argument("--model-path", type=str,
                              default="/media/k/2tb nvme/models/glm4-air",
                              help="Path to model")
    bench_parser.add_argument("--16gb", action="store_true", dest="gpu_16gb")
    bench_parser.add_argument("--24gb", action="store_true", dest="gpu_24gb")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command in ("generate", "gen", "g"):
        return run_generate(args)
    elif args.command == "info":
        return run_info(args)
    elif args.command == "bench":
        return run_bench(args)

    return 0


def run_generate(args):
    """Run text generation."""
    from config import GLM4AirConfig
    from model.glm4 import OffloadedGLM4
    from model.glm4_loader import GLM4WeightLoader
    from cache.packed_expert_store import PackedExpertStore
    from cache.kv_cache import SimpleKVCache
    from kernels.int8_linear import convert_model_to_int8, estimate_int8_memory
    from kernels.int4_linear import convert_model_to_int4, estimate_int4_memory

    # Apply GPU presets
    if args.gpu_16gb:
        print("\n[16GB GPU preset: INT4, INT8 KV, 15 slots, 25% decode skip]")
        args.int4 = True
        args.int8_kv = True
        if args.slots == 8:
            args.slots = 15
        if args.kv_size == 2048:
            args.kv_size = 1024
        if args.decode_skip == 0.0:
            args.decode_skip = 0.25
    elif args.gpu_24gb:
        print("\n[24GB GPU preset: INT8, INT8 KV, 10 slots]")
        args.int8 = True
        args.int8_kv = True
        if args.slots == 8:
            args.slots = 10

    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda")
    dtype = torch.float16
    model_path = Path(args.model_path)
    packed_dir = Path(args.packed_dir) if args.packed_dir else model_path / "packed_int4"

    print(f"\nLoading GLM-4.5-Air from {model_path}")
    print(f"  Device: {device}, Slots: {args.slots}")

    # Load config
    config = GLM4AirConfig.from_json(model_path / "config.json")
    if args.max_layers:
        config.num_layers = args.max_layers

    # Initialize model
    use_kv_cache = not args.no_kv_cache
    model = OffloadedGLM4(
        config,
        packed_expert_store=None,
        slots_per_layer=args.slots,
        device=device,
        dtype=dtype,
        use_fp16_experts=args.fp16_experts,
        decode_skip_ratio=args.decode_skip,
    )

    # Load weights
    print("  Loading weights...")
    loader = GLM4WeightLoader(model_path, config, device=torch.device("cpu"), dtype=dtype)
    weights = loader.load_full_weights()
    model.load_state_dict(weights, strict=False)
    del weights
    gc.collect()

    # Quantize
    use_quantized = args.int4 or args.int8 or args.hybrid
    if args.hybrid:
        print("  Converting to hybrid INT8/INT4...")
        skip_mlp = ["embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj"]
        if not args.quantize_all:
            skip_mlp.insert(0, "lm_head")
        convert_model_to_int4(model, skip_layers=skip_mlp)
        skip_attn = ["shared_expert", "gate", "mlp"]
        if not args.quantize_all:
            skip_attn.insert(0, "lm_head")
        convert_model_to_int8(model, skip_layers=skip_attn)
    elif args.int4:
        print("  Converting to INT4...")
        skip_layers = ["embed_tokens"] if args.quantize_all else ["lm_head", "embed_tokens"]
        convert_model_to_int4(model, skip_layers=skip_layers)
    elif args.int8:
        print("  Converting to INT8...")
        skip_layers = [] if args.quantize_all else ["lm_head"]
        convert_model_to_int8(model, skip_layers=skip_layers)

    # Move to GPU
    if use_quantized:
        model.to(device=device)
        for name, param in model.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(dtype)
    else:
        model.to(device=device, dtype=dtype)
    torch.cuda.empty_cache()

    # Load packed experts
    print(f"  Loading packed experts from {packed_dir}...")
    expert_store = PackedExpertStore(packed_dir, config, device=device)
    model.set_expert_store(expert_store)

    # Setup KV cache
    kv_cache = None
    if use_kv_cache:
        kv_cache = SimpleKVCache(
            num_layers=config.num_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_seq_len=args.kv_size,
            device=device,
            dtype=dtype,
            use_int8=args.int8_kv,
        )

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Generate
    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.tokens} tokens...\n")

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    model.eval()
    start_time = time.perf_counter()

    with torch.no_grad():
        # Prefill
        logits = model(input_ids, kv_cache=kv_cache)
        if kv_cache:
            kv_cache.set_len(input_ids.shape[1])

        # Decode
        for i in range(args.tokens):
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if kv_cache:
                logits = model(next_token, kv_cache=kv_cache, start_pos=kv_cache.cur_len)
                kv_cache.set_len(kv_cache.cur_len + 1)
            else:
                logits = model(generated)

            # Print token
            token_str = tokenizer.decode(next_token[0])
            print(token_str, end="", flush=True)

    elapsed = time.perf_counter() - start_time
    print(f"\n\n[Generated {args.tokens} tokens in {elapsed:.1f}s ({args.tokens/elapsed:.2f} tok/s)]")

    # Show cache stats
    if hasattr(model, 'cache_manager') and model.cache_manager:
        stats = model.cache_manager.get_stats()
        hit_rate = stats['hits'] / max(1, stats['hits'] + stats['misses']) * 100
        print(f"[Cache hit rate: {hit_rate:.1f}%]")

    return 0


def run_info(args):
    """Show system and model info."""
    print("\n=== HydraNet System Info ===\n")

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_mem:.1f} GB")
    else:
        print("GPU: None (CUDA not available)")

    # PyTorch info
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

    # Model info
    model_path = Path(args.model_path)
    if model_path.exists():
        print(f"\nModel: {model_path}")
        from config import GLM4AirConfig
        config = GLM4AirConfig()
        print(f"  Layers: {config.num_layers}")
        print(f"  Experts: {config.num_experts} routed + 1 shared")
        print(f"  Top-k: {config.experts_per_token}")
        packed_dir = model_path / "packed_int4"
        if packed_dir.exists():
            print(f"  Packed INT4: {packed_dir}")
    else:
        print(f"\nModel path not found: {model_path}")

    # Recommendations
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nRecommended preset:")
        if gpu_mem >= 24:
            print("  hydranet generate --24gb --prompt 'Your prompt'")
        elif gpu_mem >= 15:
            print("  hydranet generate --16gb --prompt 'Your prompt'")
        else:
            print(f"  Warning: {gpu_mem:.0f}GB may be insufficient for GLM-4.5-Air")

    return 0


def run_bench(args):
    """Run quick benchmark."""
    print("\n=== HydraNet Benchmark ===\n")
    print("Running generation benchmark...")

    # Reuse generate with benchmark settings
    args.prompt = "The quick brown fox jumps over the lazy dog."
    args.tokens = 20
    if args.gpu_16gb:
        args.int4 = True
        args.int8_kv = True
        args.slots = 15
        args.kv_size = 1024
        args.decode_skip = 0.25
    elif args.gpu_24gb:
        args.int8 = True
        args.int8_kv = True
        args.slots = 10
        args.kv_size = 2048
        args.decode_skip = 0.0
    else:
        args.int4 = True
        args.int8_kv = True
        args.slots = 8
        args.kv_size = 1024
        args.decode_skip = 0.0

    args.speculative = 0
    args.fp16_experts = False
    args.no_kv_cache = False
    args.max_layers = None
    args.quantize_all = False
    args.packed_dir = None
    args.profile = False

    return run_generate(args)


if __name__ == "__main__":
    sys.exit(main())
