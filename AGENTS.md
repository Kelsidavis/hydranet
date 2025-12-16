# HydraNet - GLM-4.5-Air MoE Inference Runtime

## Project Overview
HydraNet is a high-performance inference runtime for GLM-4.5-Air, a 32B parameter Mixture-of-Experts model with 128 routed experts per layer. Optimized for consumer GPUs (16GB+).

## Key Features
- INT4/INT8 weight quantization (74% memory savings)
- INT8 KV cache (50% cache memory savings)
- Packed INT4 expert storage with lazy GPU cache
- Cross-layer expert prefetching
- Decode layer skipping for speed/quality tradeoff
- Speculative decoding with n-gram drafting

## Directory Structure
```
hydranet/
├── model/
│   ├── glm4.py          # Main model implementation (OffloadedGLM4)
│   └── glm4_loader.py   # Weight loading utilities
├── cache/
│   ├── expert_cache.py  # GPU expert cache with LRU eviction
│   ├── kv_cache.py      # KV cache (FP16/INT8)
│   └── packed_expert_store.py  # INT4 packed expert storage
├── kernels/
│   ├── int4_linear.py   # INT4 quantization and matmul
│   └── int8_linear.py   # INT8 quantization and matmul
└── inference/
    └── speculative.py   # Speculative decoding implementation
```

## Running the Model

### Quick Start (16GB GPU)
```bash
python test_glm4_generate.py --16gb --prompt "Your prompt here"
```

### Manual Configuration
```bash
python test_glm4_generate.py \
  --model-path /path/to/glm4-air \
  --int4 \
  --int8-kv \
  --slots 15 \
  --kv-size 1024 \
  --decode-skip 0.25 \
  --tokens 100 \
  --prompt "Your prompt"
```

### Key Arguments
- `--16gb`: Optimized preset for 16GB GPUs (~1.3 tok/s, 32% cache hit)
- `--24gb`: Preset for 24GB GPUs (INT8, higher quality)
- `--int4`: Use INT4 quantization (74% memory savings)
- `--int8`: Use INT8 quantization (50% savings, better quality)
- `--int8-kv`: Use INT8 KV cache
- `--slots N`: Expert cache slots per layer (more = higher hit rate)
- `--decode-skip RATIO`: Skip routed experts in N% of layers during decode
- `--quantize-all`: Also quantize lm_head (saves 0.6GB, may hurt quality)
- `--speculative K`: Enable speculative decoding with K draft tokens

## Model Weights
Default path: `/media/k/2tb nvme/models/glm4-air`

Required files:
- `*.safetensors` - Model weights
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer
- `packed_int4/` - Pre-packed INT4 experts (auto-generated if missing)

## Performance Expectations (16GB GPU)
- Prefill: ~15s for 7 tokens (expert loading dominated)
- Decode: ~1.3 tok/s with 32% cache hit rate
- VRAM: ~15.4GB with 15 slots

## Development Notes

### Testing Changes
```bash
# Quick test
python test_glm4_generate.py --16gb --tokens 10 --prompt "Test"

# Memory test
python test_glm4_generate.py --16gb --tokens 50 --prompt "Write a story"
```

### Key Classes
- `OffloadedGLM4`: Main model class with expert offloading
- `ExpertCacheManager`: Manages GPU expert cache across layers
- `PackedExpertStore`: Reads INT4 packed experts from disk
- `SimpleKVCache`: KV cache with optional INT8 quantization

### Memory Budget (16GB, INT4)
- CUDA overhead: 1.5 GB
- Embeddings + LM head: 2.5 GB
- Attention layers: 1.2 GB
- Shared experts: 4.1 GB
- KV cache (1024 ctx): 0.06 GB
- Expert cache (15 slots): ~5 GB
- **Total: ~15.4 GB**

## Common Issues

### OOM Errors
- Reduce `--slots` (try 12 or 10)
- Enable `--quantize-all` to save 0.6GB
- Reduce `--kv-size` to 512

### Slow Generation
- Increase `--slots` for better cache hit rate
- Use `--decode-skip 0.25` for 25% speedup
- Ensure model is on fast NVMe for expert loading

### Quality Issues
- Reduce `--decode-skip` (0 for best quality)
- Don't use `--quantize-all`
- Use `--int8` instead of `--int4` if VRAM allows
