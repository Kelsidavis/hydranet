# HydraNet

MoE inference runtime for running large Mixture-of-Experts models on consumer GPUs by **caching experts in VRAM** and **streaming the rest from RAM**. Designed for practical throughput on 16GB cards.

## Supported Models

| Model | Params | Active | Experts | Routing | Branch |
|-------|--------|--------|---------|---------|--------|
| Mixtral-8x7B | 47B | ~13B | 8/layer | Top-2 softmax | `mixtral` |
| GLM-4.5-Air | 112B | ~18B | 128/layer | Top-8 sigmoid | `glm45air` |

## Key Features

- **INT4 quantized experts** — 10x smaller than fp16, fits more in VRAM
- **Per-layer LRU cache** — prevents cross-layer thrashing
- **Async H2D overlap** — prefetch misses while computing hits
- **2-phase hits-first scheduling** — maximize overlap opportunity
- **Fused INT4 kernels** — no fp16 materialization during compute
- **Per-token profiling** — hit rates, load times, bottleneck layers

## Performance (RTX 5080 16GB)

**Mixtral-8x7B** (126 slots, top-1 decode):
- ~5 tok/s decode
- ~65% hit rate
- ~7ms avg load time

**GLM-4.5-Air** (estimated, top-8 decode):
- ~6-10 tok/s decode (experts 10x smaller)
- Higher hit rate potential (more slots fit)
- Real overlap benefit (8 experts/layer)

## Repository Structure

```
hydranet/
├── v2/
│   ├── model/
│   │   ├── mixtral.py      # Mixtral-8x7B model
│   │   ├── glm4.py         # GLM-4.5-Air model
│   │   ├── loader.py       # Mixtral weight loader
│   │   └── glm4_loader.py  # GLM4 weight loader
│   ├── cache/
│   │   ├── expert_cache.py      # Per-layer LRU cache manager
│   │   ├── packed_expert_store.py # INT4 blob storage + staging
│   │   └── kv_cache.py          # KV cache for attention
│   ├── kernels/
│   │   └── int4_gemm.py    # Fused INT4 MLP kernels
│   ├── preprocess/
│   │   ├── pack_weights.py # Mixtral INT4 packer
│   │   └── pack_glm4.py    # GLM4 INT4 packer
│   └── config.py           # Model configs
├── test_generate.py        # Mixtral generation test
└── test_glm4_smoke.py      # GLM4 smoke test
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- ~32GB system RAM (for expert store)
- 16GB+ VRAM recommended

```bash
pip install torch safetensors transformers
```

## Quickstart

### Mixtral-8x7B

```bash
# 1. Download model
huggingface-cli download mistralai/Mixtral-8x7B-Instruct-v0.1 --local-dir /path/to/mixtral

# 2. Pack experts to INT4 (one-time, ~20min)
python -m hydranet.v2.preprocess.pack_weights \
    --model-path /path/to/mixtral \
    --output-dir /path/to/mixtral/packed_int4

# 3. Run generation
python test_generate.py --mode packed --tokens 100 --slots 3
```

### GLM-4.5-Air

```bash
# 1. Download model (~200GB)
huggingface-cli download zai-org/GLM-4.5-Air --local-dir /path/to/glm4

# 2. Pack experts to INT4
python -m hydranet.v2.preprocess.pack_glm4 \
    --model-path /path/to/glm4 \
    --output-dir /path/to/glm4/packed_int4

# 3. Run smoke test
python test_glm4_smoke.py
```

## Configuration

### Slot Budget

More slots = higher hit rate, but more VRAM:

```bash
# Conservative (fits 16GB easily)
python test_generate.py --slots 3  # 126 total slots

# Aggressive (may OOM on 16GB)
python test_generate.py --slots 4  # 158 total slots
```

### Expert Backend

```bash
# RAM backend (fast, uses ~22GB RAM for Mixtral)
python test_generate.py --packed-backend ram

# mmap backend (slower, low RAM usage)
python test_generate.py --packed-backend mmap
```

### Profiling

```bash
python test_generate.py --profile --tokens 50
```

Shows per-layer miss counts to identify bottleneck layers.

## How It Works

1. **Router** selects top-k experts per token (softmax for Mixtral, sigmoid for GLM4)
2. **Cache lookup** — hits return immediately, misses trigger async load
3. **2-phase scheduling** — compute hits while misses load in background
4. **Fused INT4 kernel** — runs expert MLP directly on quantized weights
5. **Staging ring** — pinned memory buffers for non-blocking H2D

### Cache Architecture

```
┌─────────────────────────────────────────────────┐
│                    VRAM                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Slot 0   │  │ Slot 1   │  │ Slot 2   │ ...  │
│  │ Expert 3 │  │ Expert 7 │  │ Expert 1 │      │
│  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────┘
        ▲ async H2D (memcpy stream)
┌─────────────────────────────────────────────────┐
│              Pinned Staging Ring                 │
└─────────────────────────────────────────────────┘
        ▲ memcpy from RAM
┌─────────────────────────────────────────────────┐
│         RAM Expert Store (experts.bin)           │
│         ~22GB Mixtral / ~50GB GLM4              │
└─────────────────────────────────────────────────┘
```

## Metrics to Watch

| Metric | Good | Bad |
|--------|------|-----|
| Decode hit rate | >60% | <40% |
| Avg load time | <10ms | >20ms |
| Decode tok/s | >4 | <2 |
| Evictions/token | <2 | >5 |

## Roadmap

- [x] INT4 packed expert store
- [x] VRAM slot caching with LRU
- [x] Async H2D with CUDA streams
- [x] 2-phase hits-first overlap
- [x] Mixtral-8x7B support
- [x] GLM-4.5-Air support
- [ ] Adaptive slot reallocation
- [ ] Speculative decoding integration
- [ ] Multi-GPU expert sharding

## License

MIT
