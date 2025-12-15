"""HydraNet: Full Mixture of Experts Language Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

from .config import HydraNetConfig
from .attention import RMSNorm, create_causal_mask
from .moe_layer import HydraNetBlock, HydraNetPreTrainedModel


@dataclass
class HydraNetOutput:
    """Output from HydraNet forward pass."""
    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    router_logits: Optional[Tuple[torch.Tensor, ...]] = None
    aux_loss: Optional[torch.Tensor] = None


class HydraNet(HydraNetPreTrainedModel):
    """
    HydraNet: A Mixture of Experts language model optimized for
    consumer hardware with intelligent expert caching.

    Features:
    - Grouped Query Attention (GQA) for efficient KV cache
    - Top-K MoE routing with shared experts
    - Designed for 16GB VRAM + 128GB RAM configuration
    - Supports expert offloading to RAM
    """

    def __init__(self, config: HydraNetConfig):
        super().__init__(config)
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_dim,
        )

        # Transformer blocks
        self.layers = nn.ModuleList([
            HydraNetBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        # LM head (output projection)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Optionally tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Gradient checkpointing flag
        self._gradient_checkpointing = False

        # Initialize weights
        self.apply(self._init_weights)

        # Cache for expert management
        self._expert_cache_manager = None

    def set_expert_cache_manager(self, manager):
        """Set external expert cache manager for inference."""
        self._expert_cache_manager = manager

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
        return_dict: bool = True,
    ) -> Union[HydraNetOutput, Tuple]:
        """
        Forward pass through HydraNet.

        Args:
            input_ids: (batch, seq) token indices
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_values: KV cache from previous steps
            use_cache: Whether to return KV cache
            output_hidden_states: Return intermediate hidden states
            output_router_logits: Return routing decisions
            return_dict: Return HydraNetOutput vs tuple

        Returns:
            HydraNetOutput or tuple with logits and optional extras
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Prepare position IDs
        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                device=input_ids.device,
            ).unsqueeze(0).expand(batch_size, -1)

        # Prepare causal mask
        if attention_mask is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values else 0
            attention_mask = create_causal_mask(
                seq_len,
                past_len=past_len,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        # Storage for outputs
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if output_router_logits else None
        new_key_values = [] if use_cache else None
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device)

        # Process through layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Get past KV for this layer
            past_kv = past_key_values[i] if past_key_values else None

            # Gradient checkpointing
            if self._gradient_checkpointing and self.training:
                hidden_states, present_kv, router_logits = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_kv,
                    use_cache,
                    output_router_logits,
                    use_reentrant=False,
                )
            else:
                hidden_states, present_kv, router_logits = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    return_router_logits=output_router_logits,
                )

            if use_cache:
                new_key_values.append(present_kv)

            if output_router_logits and router_logits is not None:
                all_router_logits += (router_logits,)

            # Accumulate auxiliary loss
            if self.training and layer.aux_loss is not None:
                total_aux_loss = total_aux_loss + layer.aux_loss

        # Final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # LM head
        logits = self.lm_head(hidden_states)

        if not return_dict:
            return (logits, new_key_values, all_hidden_states, all_router_logits)

        return HydraNetOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            past_key_values=new_key_values,
            router_logits=all_router_logits,
            aux_loss=total_aux_loss if self.training else None,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Initial token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-K sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample vs greedy
            eos_token_id: End of sequence token

        Returns:
            Generated token IDs including prompt
        """
        self.eval()
        batch_size = input_ids.shape[0]
        past_key_values = None
        generated = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass (only new tokens if we have cache)
                if past_key_values is not None:
                    curr_input = generated[:, -1:]
                else:
                    curr_input = generated

                outputs = self.forward(
                    curr_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]  # (batch, vocab)

                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature

                # Apply sampling
                if do_sample:
                    next_token = self._sample(logits, top_k=top_k, top_p=top_p)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

                # Check for EOS
                if eos_token_id is not None:
                    if (next_token == eos_token_id).all():
                        break

        return generated

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Sample from logits with top-k and top-p filtering."""
        # Top-K filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative prob above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token

    def get_total_aux_loss(self) -> torch.Tensor:
        """Get sum of all MoE auxiliary losses."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            if layer.aux_loss is not None:
                total = total + layer.aux_loss
        return total

    def get_expert_usage_stats(self) -> dict:
        """Get expert usage statistics across all layers."""
        stats = {}
        for i, layer in enumerate(self.layers):
            usage = layer.moe.router.usage_history.clone()
            stats[f"layer_{i}"] = {
                "usage": usage,
                "total_tokens": usage.sum().item(),
                "active_experts": (usage > 0).sum().item(),
            }
        return stats

    @classmethod
    def from_pretrained(cls, path: str) -> "HydraNet":
        """Load model from saved checkpoint."""
        config = HydraNetConfig.load(f"{path}/config.json")
        model = cls(config)

        # Load weights
        state_dict = torch.load(f"{path}/model.pt", map_location="cpu")
        model.load_state_dict(state_dict)

        return model

    def save_pretrained(self, path: str):
        """Save model to directory."""
        import os
        os.makedirs(path, exist_ok=True)

        # Save config
        self.config.save(f"{path}/config.json")

        # Save weights
        torch.save(self.state_dict(), f"{path}/model.pt")

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Count parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Convenience function
def create_hydranet(size: str = "large") -> HydraNet:
    """Create HydraNet model of specified size."""
    from .config import HydraNetConfigs

    configs = {
        "tiny": HydraNetConfigs.tiny,
        "small": HydraNetConfigs.small,
        "medium": HydraNetConfigs.medium,
        "large": HydraNetConfigs.large,
    }

    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

    config = configs[size]()
    return HydraNet(config)
