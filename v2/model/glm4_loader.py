"""
GLM-4.5-Air weight loader.

Loads GLM-4.5-Air weights from HuggingFace format and registers
experts with the cache manager.

Weight mapping (HuggingFace -> HydraNet):
- model.embed_tokens.weight -> embeddings
- model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias} -> attention
- model.layers.{i}.mlp.gate.weight -> router (MoE layers only)
- model.layers.{i}.mlp.experts.{j}.{gate,up,down}_proj.weight -> routed experts
- model.layers.{i}.mlp.shared_experts.{gate,up,down}_proj.weight -> shared expert
- model.layers.{i}.mlp.{gate,up,down}_proj.weight -> dense MLP (layer 0)
- model.layers.{i}.{input,post_attention}_layernorm.weight -> norms
- model.norm.weight -> final_norm
- lm_head.weight -> lm_head
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Iterator, Tuple, Union
from safetensors import safe_open
import json
import os

from ..config import GLM4AirConfig
from ..cache import ExpertCacheManager


class GLM4WeightLoader:
    """
    Loads GLM-4.5-Air weights from HuggingFace safetensors format.

    Handles:
    - Sharded safetensors files
    - Dense vs MoE layers
    - Routed experts + shared experts
    - Streaming load to avoid OOM
    """

    def __init__(
        self,
        model_path: str,
        config: GLM4AirConfig,
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
            self.weight_index = None

        # Cache open file handles for efficiency
        self._open_files: Dict[Path, object] = {}

    def _get_shard_for_key(self, key: str) -> Path:
        """Get the shard file containing a weight key."""
        if self.weight_index:
            filename = self.weight_index.get(key)
            if filename:
                return self.model_path / filename
        return self.shard_files[0]

    def _get_tensor(self, key: str) -> torch.Tensor:
        """Load a tensor from the appropriate shard."""
        shard = self._get_shard_for_key(key)

        # Use cached file handle if available
        if shard not in self._open_files:
            self._open_files[shard] = safe_open(shard, framework="pt")

        return self._open_files[shard].get_tensor(key).to(dtype=self.dtype)

    def _get_tensor_if_exists(self, key: str) -> Optional[torch.Tensor]:
        """Load a tensor if it exists, return None otherwise."""
        try:
            return self._get_tensor(key)
        except (KeyError, Exception):
            return None

    def close(self):
        """Close all open file handles."""
        self._open_files.clear()

    def load_embedding(self) -> torch.Tensor:
        """Load embedding weights."""
        return self._get_tensor("model.embed_tokens.weight")

    def load_lm_head(self) -> torch.Tensor:
        """Load LM head weights."""
        return self._get_tensor("lm_head.weight")

    def load_final_norm(self) -> torch.Tensor:
        """Load final layer norm weights."""
        return self._get_tensor("model.norm.weight")

    def load_layer_attention(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load attention weights for a layer.

        GLM4 has attention bias, so we load both weight and bias.
        """
        prefix = f"model.layers.{layer_idx}.self_attn"
        weights = {}

        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weights[f"{name}.weight"] = self._get_tensor(f"{prefix}.{name}.weight")

            # Load bias if exists
            bias = self._get_tensor_if_exists(f"{prefix}.{name}.bias")
            if bias is not None:
                weights[f"{name}.bias"] = bias

        return weights

    def load_layer_norms(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Load layer norm weights for a layer."""
        weights = {}

        for name in ["input_layernorm", "post_attention_layernorm"]:
            weights[name] = self._get_tensor(f"model.layers.{layer_idx}.{name}.weight")

        return weights

    def load_dense_mlp(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load dense MLP weights (for layer 0).

        GLM4's first layer uses a dense FFN, not MoE.
        """
        prefix = f"model.layers.{layer_idx}.mlp"
        weights = {}

        for name in ["gate_proj", "up_proj", "down_proj"]:
            weights[name] = self._get_tensor(f"{prefix}.{name}.weight")

        return weights

    def load_router(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Load router gate weights for a MoE layer."""
        prefix = f"model.layers.{layer_idx}.mlp.gate"
        weights = {}

        weights["weight"] = self._get_tensor(f"{prefix}.weight")

        # Some versions have e_score_correction_bias
        bias = self._get_tensor_if_exists(f"{prefix}.e_score_correction_bias")
        if bias is not None:
            weights["e_score_correction_bias"] = bias

        return weights

    def load_shared_expert(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load shared expert weights for a MoE layer.

        The shared expert processes all tokens.
        """
        prefix = f"model.layers.{layer_idx}.mlp.shared_experts"
        weights = {}

        for name in ["gate_proj", "up_proj", "down_proj"]:
            key = f"{prefix}.{name}.weight"
            tensor = self._get_tensor_if_exists(key)
            if tensor is not None:
                weights[name] = tensor

        # GLM4 might use fused gate_up_proj for shared expert
        fused = self._get_tensor_if_exists(f"{prefix}.gate_up_proj.weight")
        if fused is not None:
            # Split fused gate_up into separate tensors
            intermediate = fused.shape[0] // 2
            weights["gate_proj"] = fused[:intermediate]
            weights["up_proj"] = fused[intermediate:]

        return weights

    def load_expert(
        self,
        layer_idx: int,
        expert_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Load routed expert weights.

        Returns dict with gate_proj, up_proj, down_proj.
        """
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        weights = {}

        for name in ["gate_proj", "up_proj", "down_proj"]:
            weights[name] = self._get_tensor(f"{prefix}.{name}.weight")

        return weights

    def iter_experts(self) -> Iterator[Tuple[int, int, Dict[str, torch.Tensor]]]:
        """
        Iterate over all routed experts, yielding (moe_layer_idx, expert_idx, weights).

        Streams from disk to avoid loading all experts at once.
        moe_layer_idx is 0-indexed from the first MoE layer.
        """
        for layer_idx in range(self.config.first_k_dense_replace, self.config.num_layers):
            moe_layer_idx = layer_idx - self.config.first_k_dense_replace

            for expert_idx in range(self.config.num_experts):
                weights = self.load_expert(layer_idx, expert_idx)
                yield moe_layer_idx, expert_idx, weights

    def register_all_experts(self, cache_manager: ExpertCacheManager):
        """
        Register all routed experts with the cache manager.

        Loads experts one at a time and registers in pinned RAM.
        """
        total = self.config.total_routed_experts
        print(f"Loading {total} routed experts from {self.model_path}...")

        count = 0
        for moe_layer_idx, expert_idx, weights in self.iter_experts():
            cache_manager.register_expert(moe_layer_idx, expert_idx, weights)
            count += 1

            if count % 128 == 0:
                print(f"  Loaded {count}/{total} experts")

        print(f"All {count} routed experts registered")

    def load_non_expert_weights(self) -> Dict[str, torch.Tensor]:
        """
        Load all non-expert weights (to GPU).

        Returns dict with attention, norm, embedding, router, shared expert weights.
        These stay resident on GPU.
        """
        weights = {}

        # Embeddings
        print("Loading embeddings...")
        weights["embed_tokens.weight"] = self.load_embedding()
        weights["lm_head.weight"] = self.load_lm_head()
        weights["norm.weight"] = self.load_final_norm()

        # Per-layer weights
        print("Loading layers...")
        for layer_idx in range(self.config.num_layers):
            # Attention
            attn = self.load_layer_attention(layer_idx)
            for name, weight in attn.items():
                weights[f"layers.{layer_idx}.self_attn.{name}"] = weight

            # Layer norms
            norms = self.load_layer_norms(layer_idx)
            for name, weight in norms.items():
                weights[f"layers.{layer_idx}.{name}.weight"] = weight

            # MLP: dense or MoE
            is_moe = layer_idx >= self.config.first_k_dense_replace

            if is_moe:
                # Router
                router = self.load_router(layer_idx)
                for name, weight in router.items():
                    weights[f"layers.{layer_idx}.mlp.router.{name}"] = weight

                # Shared expert
                shared = self.load_shared_expert(layer_idx)
                for name, weight in shared.items():
                    weights[f"layers.{layer_idx}.mlp.shared_expert.{name}"] = weight
            else:
                # Dense MLP
                mlp = self.load_dense_mlp(layer_idx)
                for name, weight in mlp.items():
                    weights[f"layers.{layer_idx}.mlp.{name}"] = weight

            if (layer_idx + 1) % 8 == 0:
                print(f"  Loaded {layer_idx + 1}/{self.config.num_layers} layers")

        return weights

    def estimate_memory(self) -> Dict[str, float]:
        """
        Estimate memory requirements.

        Returns dict with sizes in GB.
        """
        cfg = self.config

        # Routed expert memory (all in RAM for offloading)
        expert_params = cfg.expert_params
        total_expert_params = cfg.total_routed_experts * expert_params
        expert_bytes = total_expert_params * 2  # fp16
        expert_gb = expert_bytes / 1e9

        # Shared expert memory (on GPU)
        shared_params = cfg.shared_expert_params * cfg.num_moe_layers
        shared_bytes = shared_params * 2
        shared_gb = shared_bytes / 1e9

        # Attention memory (on GPU)
        # Q + K + V + O projections + biases
        q_params = cfg.hidden_dim * cfg.num_attention_heads * cfg.head_dim
        kv_params = 2 * cfg.hidden_dim * cfg.num_kv_heads * cfg.head_dim
        o_params = cfg.num_attention_heads * cfg.head_dim * cfg.hidden_dim
        attn_params_per_layer = q_params + kv_params + o_params
        total_attn_params = cfg.num_layers * attn_params_per_layer
        attn_bytes = total_attn_params * 2
        attn_gb = attn_bytes / 1e9

        # Embedding + LM head
        embed_params = 2 * cfg.vocab_size * cfg.hidden_dim
        embed_bytes = embed_params * 2
        embed_gb = embed_bytes / 1e9

        # Router weights (per MoE layer)
        router_params = cfg.num_moe_layers * cfg.hidden_dim * cfg.num_experts
        router_bytes = router_params * 2
        router_gb = router_bytes / 1e9

        # Dense MLP (layer 0)
        dense_params = cfg.first_k_dense_replace * 3 * cfg.hidden_dim * cfg.intermediate_dim
        dense_bytes = dense_params * 2
        dense_gb = dense_bytes / 1e9

        return {
            "routed_experts_ram_gb": expert_gb,
            "shared_experts_gpu_gb": shared_gb,
            "attention_gpu_gb": attn_gb,
            "embeddings_gpu_gb": embed_gb,
            "router_gpu_gb": router_gb,
            "dense_mlp_gpu_gb": dense_gb,
            "total_gpu_gb": attn_gb + embed_gb + router_gb + shared_gb + dense_gb,
            "total_ram_gb": expert_gb,
        }


def download_glm4_air(
    model_id: str = "zai-org/GLM-4.5-Air",
    output_dir: Optional[str] = None,
) -> Path:
    """
    Download GLM-4.5-Air weights from HuggingFace Hub.

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
    print("This may take a while (~200GB)...")

    path = snapshot_download(
        model_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.bin", "*.h5", "*.gguf"],  # Only safetensors
    )

    return Path(path)
