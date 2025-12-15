# HydraNet

HydraNet is an MoE inference runtime for running large expert models on consumer GPUs by **keeping a working set in VRAM** and **streaming the rest from RAM/disk**. It focuses on practical throughput with profiling and cache-aware tuning.

## Highlights
- **INT4 packed experts** with a VRAM slot cache
- **RAM-backed expert store** (optional mmap backend)
- **Pinned staging + non_blocking H2D** for fast transfers
- **Per-token + per-layer profiling** (hit/miss rates, evictions, load time)
- Simple tuning knobs: slots-per-layer, pinning, backend selection

## Repo layout
- `hydranet/v2/model/` — model wrappers + MoE forward path
- `hydranet/v2/cache/` — expert cache, packed store, staging ring
- `hydranet/v2/kernels/` — fused kernels (e.g. INT4 MLP)
- `test_generate.py` — generation + profiling harness

## Requirements
- Python 3.10+ (tested on 3.12)
- PyTorch w/ CUDA recommended
- Enough **system RAM** for `experts.bin` if using `--packed-backend ram`

## Quickstart
```bash
git clone <this-repo>
cd hydranet
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run a generation/profile test
```bash
python test_generate.py --mode packed --tokens 100
python test_generate.py --mode packed --tokens 200 --profile
```

### Packed expert backend
**RAM (fast, +RAM usage):**
```bash
python test_generate.py --mode packed --packed-backend ram
```

**mmap (low RAM, slower):**
```bash
python test_generate.py --mode packed --packed-backend mmap
```

### Slot budget / tuning
```bash
python test_generate.py --mode packed --slots 4 --tokens 200 --profile
```

## How it works (high level)
1. Router picks experts per layer.
2. VRAM cache returns expert weights if present (**hit**).
3. On **miss**, HydraNet loads the packed expert blob from RAM/mmap into pinned staging and transfers it into a VRAM slot.
4. A fused INT4 kernel runs the expert MLP without materializing full fp16 weights.

## What to watch
- Decode throughput (tok/s)
- Decode-only hit rate (warm-cache)
- Avg miss load time
- Per-layer miss counts (bottlenecks)

## Status / roadmap
- ✅ Fast RAM-backed packed expert loading (zero-copy slicing)
- ✅ VRAM slot caching + profiling
- 🟡 Smarter pinning / adaptive policies
- 🟡 Better overlap strategies (batched / top-2 workloads)
- 🟡 More model backends

## License
TBD.
