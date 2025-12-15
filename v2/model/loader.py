"""
Mixtral weight loader.

Loads Mixtral-8x7B weights from HuggingFace format and registers
experts with the cache manager.

Weight mapping (HuggingFace -> HydraNet):
- model.embed_tokens.weight -> embeddings
- model.layers.{i}.self_attn.{q,k,v,o}_proj -> attention
- model.layers.{i}.block_sparse_moe.gate.weight -> router
- model.layers.{i}.block_sparse_moe.experts.{j}.{w1,w2,w3} -> experts
  - w1 = gate_proj
  - w2 = down_proj
  - w3 = up_proj
- model.layers.{i}.{input,post_attention}_layernorm -> norms
- model.norm.weight -> final_norm
- lm_head.weight -> lm_head
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Iterator, Tuple
from safetensors import safe_open
import json
import os

from ..config import MixtralConfig
from ..cache import ExpertCacheManager


class MixtralWeightLoader:
    """
    Loads Mixtral weights from HuggingFace safetensors format.

    Handles:
    - Sharded safetensors files
    - Streaming load to avoid OOM
    - Expert registration with cache manager
    """

    def __init__(
        self,
        model_path: str,
        config: MixtralConfig,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ):
        self.model_path = Path(model_path)
        self.config = config
        self.device = device
        self.dtype = dtype

        # Find safetensor files
        self.shard_files = sorted(self.model_path.glob("*.safetensors"))
        if not self.shard_files:
            raise ValueError(f"No safetensors files found in {model_path}")

        # Load weight index if sharded
        index_path = self.model_path / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                self.weight_index = json.load(f)["weight_map"]
        else:
            # Single file, all weights in one file
            self.weight_index = None

    def _get_shard_for_key(self, key: str) -> Path:
        """Get the shard file containing a weight key."""
        if self.weight_index:
            filename = self.weight_index.get(key)
            if filename:
                return self.model_path / filename
        return self.shard_files[0]

    def load_embedding(self) -> torch.Tensor:
        """Load embedding weights."""
        key = "model.embed_tokens.weight"
        shard = self._get_shard_for_key(key)

        with safe_open(shard, framework="pt") as f:
            weight = f.get_tensor(key)
            return weight.to(dtype=self.dtype)

    def load_lm_head(self) -> torch.Tensor:
        """Load LM head weights."""
        key = "lm_head.weight"
        shard = self._get_shard_for_key(key)

        with safe_open(shard, framework="pt") as f:
            weight = f.get_tensor(key)
            return weight.to(dtype=self.dtype)

    def load_final_norm(self) -> torch.Tensor:
        """Load final layer norm weights."""
        key = "model.norm.weight"
        shard = self._get_shard_for_key(key)

        with safe_open(shard, framework="pt") as f:
            weight = f.get_tensor(key)
            return weight.to(dtype=self.dtype)

    def load_layer_attention(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load attention weights for a layer.

        Returns dict with q_proj, k_proj, v_proj, o_proj weights.
        """
        prefix = f"model.layers.{layer_idx}.self_attn"
        weights = {}

        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            key = f"{prefix}.{name}.weight"
            shard = self._get_shard_for_key(key)

            with safe_open(shard, framework="pt") as f:
                weights[name] = f.get_tensor(key).to(dtype=self.dtype)

        return weights

    def load_layer_norms(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Load layer norm weights for a layer."""
        weights = {}

        for name in ["input_layernorm", "post_attention_layernorm"]:
            key = f"model.layers.{layer_idx}.{name}.weight"
            shard = self._get_shard_for_key(key)

            with safe_open(shard, framework="pt") as f:
                weights[name] = f.get_tensor(key).to(dtype=self.dtype)

        return weights

    def load_router(self, layer_idx: int) -> torch.Tensor:
        """Load router gate weights for a layer."""
        key = f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"
        shard = self._get_shard_for_key(key)

        with safe_open(shard, framework="pt") as f:
            weight = f.get_tensor(key)
            return weight.to(dtype=self.dtype)

    def load_expert(
        self,
        layer_idx: int,
        expert_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Load expert weights.

        Returns dict with gate_proj, up_proj, down_proj.

        HuggingFace naming:
        - w1 = gate_proj (hidden -> intermediate)
        - w2 = down_proj (intermediate -> hidden)
        - w3 = up_proj (hidden -> intermediate)
        """
        prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"

        # Map HF names to our names
        name_map = {
            "w1": "gate_proj",
            "w2": "down_proj",
            "w3": "up_proj",
        }

        weights = {}
        for hf_name, our_name in name_map.items():
            key = f"{prefix}.{hf_name}.weight"
            shard = self._get_shard_for_key(key)

            with safe_open(shard, framework="pt") as f:
                weights[our_name] = f.get_tensor(key).to(dtype=self.dtype)

        return weights

    def iter_experts(self) -> Iterator[Tuple[int, int, Dict[str, torch.Tensor]]]:
        """
        Iterate over all experts, yielding (layer_idx, expert_idx, weights).

        Streams from disk to avoid loading all experts at once.
        """
        for layer_idx in range(self.config.num_layers):
            for expert_idx in range(self.config.num_experts):
                weights = self.load_expert(layer_idx, expert_idx)
                yield layer_idx, expert_idx, weights

    def register_all_experts(self, cache_manager: ExpertCacheManager):
        """
        Register all experts with the cache manager.

        Loads experts one at a time and registers in pinned RAM.
        """
        print(f"Loading {self.config.total_experts} experts from {self.model_path}...")

        for layer_idx, expert_idx, weights in self.iter_experts():
            cache_manager.register_expert(layer_idx, expert_idx, weights)

            if (layer_idx * self.config.num_experts + expert_idx + 1) % 32 == 0:
                loaded = layer_idx * self.config.num_experts + expert_idx + 1
                print(f"  Loaded {loaded}/{self.config.total_experts} experts")

        print(f"All experts registered in RAM")

    def load_non_expert_weights(self) -> Dict[str, torch.Tensor]:
        """
        Load all non-expert weights (to GPU).

        Returns dict with all attention, norm, embedding weights.
        These stay resident on GPU.
        """
        weights = {}

        # Embeddings
        print("Loading embeddings...")
        weights["embed_tokens"] = self.load_embedding()
        weights["lm_head"] = self.load_lm_head()
        weights["final_norm"] = self.load_final_norm()

        # Per-layer attention and norms
        print("Loading attention layers...")
        for layer_idx in range(self.config.num_layers):
            attn = self.load_layer_attention(layer_idx)
            norms = self.load_layer_norms(layer_idx)
            router = self.load_router(layer_idx)

            for name, weight in attn.items():
                weights[f"layers.{layer_idx}.attn.{name}"] = weight

            for name, weight in norms.items():
                weights[f"layers.{layer_idx}.{name}"] = weight

            weights[f"layers.{layer_idx}.router"] = router

            if (layer_idx + 1) % 8 == 0:
                print(f"  Loaded {layer_idx + 1}/{self.config.num_layers} layers")

        return weights

    def estimate_memory(self) -> Dict[str, float]:
        """
        Estimate memory requirements.

        Returns dict with sizes in GB.
        """
        # Expert memory (all in RAM)
        expert_params = self.config.expert_params
        total_expert_params = self.config.total_experts * expert_params
        expert_bytes = total_expert_params * 2  # fp16
        expert_gb = expert_bytes / 1e9

        # Attention memory (on GPU)
        # Q, K, V, O projections per layer
        attn_params_per_layer = 4 * self.config.hidden_dim * self.config.hidden_dim
        total_attn_params = self.config.num_layers * attn_params_per_layer
        attn_bytes = total_attn_params * 2
        attn_gb = attn_bytes / 1e9

        # Embedding + LM head
        embed_params = 2 * self.config.vocab_size * self.config.hidden_dim
        embed_bytes = embed_params * 2
        embed_gb = embed_bytes / 1e9

        # Router weights
        router_params = self.config.num_layers * self.config.hidden_dim * self.config.num_experts
        router_bytes = router_params * 2
        router_gb = router_bytes / 1e9

        return {
            "experts_ram_gb": expert_gb,
            "attention_gpu_gb": attn_gb,
            "embeddings_gpu_gb": embed_gb,
            "router_gpu_gb": router_gb,
            "total_gpu_gb": attn_gb + embed_gb + router_gb,
            "total_ram_gb": expert_gb,
        }


def download_mixtral(
    model_id: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    output_dir: Optional[str] = None,
) -> Path:
    """
    Download Mixtral weights from HuggingFace Hub.

    Requires: pip install huggingface_hub

    Args:
        model_id: HuggingFace model ID
        output_dir: Where to save (default: ~/.cache/huggingface)

    Returns:
        Path to downloaded weights
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("Install huggingface_hub: pip install huggingface_hub")

    print(f"Downloading {model_id}...")
    print("This may take a while (~90GB)...")

    path = snapshot_download(
        model_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.bin", "*.h5"],  # Only safetensors
    )

    return Path(path)
