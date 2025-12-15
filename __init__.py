"""
HydraNet: A Mixture of Experts Language Model for Consumer Hardware

Designed for:
- RTX 5080 (16GB VRAM)
- 128GB RAM
- Efficient expert caching between GPU and RAM
"""

import os

# ========== RESOURCE LIMITS ==========
# Reserve 2 CPU threads and 6GB RAM for system stability
# This prevents the model from using all system resources
RESERVED_THREADS = 2
RESERVED_RAM_GB = 6

MAX_CPU_THREADS = max(1, os.cpu_count() - RESERVED_THREADS) if os.cpu_count() else 4

# Set environment variables before importing torch
os.environ.setdefault("OMP_NUM_THREADS", str(MAX_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(MAX_CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(MAX_CPU_THREADS))

# Calculate available RAM (used by configs, not hard limits)
TOTAL_RAM_GB = 128  # fallback default
AVAILABLE_RAM_GB = TOTAL_RAM_GB - RESERVED_RAM_GB
try:
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('MemTotal:'):
                total_kb = int(line.split()[1])
                TOTAL_RAM_GB = total_kb / (1024 * 1024)
                AVAILABLE_RAM_GB = TOTAL_RAM_GB - RESERVED_RAM_GB
                break
except Exception:
    pass

import torch
torch.set_num_threads(MAX_CPU_THREADS)
torch.set_num_interop_threads(max(1, MAX_CPU_THREADS // 2))
# =====================================

__version__ = "0.1.0"

from .model import (
    HydraNetConfig,
    HydraNetConfigs,
    HydraNet,
    create_hydranet,
)

__all__ = [
    "HydraNetConfig",
    "HydraNetConfigs",
    "HydraNet",
    "create_hydranet",
]
