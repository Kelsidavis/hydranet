# HydraNet v2.1 - Mixtral Offloading Runtime

Throughput-first MoE inference runtime for consumer hardware.

## Target Hardware
- GPU: RTX 5080 (16GB GDDR7, ~960 GB/s internal, ~25 GB/s PCIe 4.0)
- RAM: 128GB DDR4 (~50 GB/s)
- CPU: Ryzen 3900X (24 threads)

## Model Stack
- **Verifier**: Mixtral-8x7B-Instruct-v0.1 (8 experts/layer, top-2)
- **Draft**: Mistral-7B-Instruct-v0.2 (resident, tokenizer v1 aligned)

## Key Design Decisions
1. **Per-layer expert cache** - 3 slots/layer (pinned/hot/probation), prevents cross-layer thrashing
2. **Sticky LFU with probation** - New experts enter probation, promoted on second hit
3. **Top-2 prefill / Top-1 decode** - Runtime toggleable, halves decode bandwidth
4. **Layer-local prefetch** - Overlap compute with H2D for same layer, microbatch pipelining
5. **KV paging with landmarks** - 4K VRAM window, RAM pages, tool-call pinning
6. **Cache-aware spec decoding** - Pin experts during 4-token draft verification

## Package Structure
```
hydranet/v2/
├── config.py           # MixtralConfig, DraftConfig, RuntimeConfig
├── cache/
│   ├── expert_cache.py # PerLayerCache, ExpertCacheManager
│   ├── kv_cache.py     # KVPageManager, LandmarkTracker
│   └── policies.py     # StickyLFU, eviction policies
├── model/
│   ├── loader.py       # MixtralWeightLoader, weight conversion
│   ├── attention.py    # GQA with paged KV
│   ├── expert.py       # Expert FFN, batched execution
│   ├── router.py       # TopKRouter with affinity bonus
│   └── mixtral.py      # OffloadedMixtral main model
├── spec/
│   ├── draft.py        # ResidentDraftModel
│   └── verifier.py     # CacheAwareVerifier
├── kernels/
│   ├── expert_gemm.py  # Triton fused expert GEMM
│   ├── paged_attn.py   # Paged attention kernel
│   └── copy.py         # Async H2D copy primitives
└── engine.py           # InferenceEngine top-level API
```
