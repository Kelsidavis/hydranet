"""Mixture of Experts layer combining router and experts."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import HydraNetConfig
from .attention import GroupedQueryAttention, RMSNorm
from .expert import ExpertContainer, SharedExpert
from .router import TopKRouter, RouterOutput


class MoELayer(nn.Module):
    """
    Single Mixture of Experts layer.

    Combines:
    - Router to select experts
    - Expert container to execute selected experts
    - Optional shared expert for common patterns
    """

    def __init__(
        self,
        config: HydraNetConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Router
        self.router = TopKRouter(config, layer_idx)

        # Experts
        self.experts = ExpertContainer(
            config=config,
            num_experts=config.num_experts,
            layer_idx=layer_idx,
        )

        # Shared expert (always active)
        self.shared_expert = SharedExpert(config) if config.use_shared_expert else None

        # Track auxiliary loss
        self.last_aux_loss = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MoE layer.

        Args:
            hidden_states: (batch, seq, hidden_dim)
            return_router_logits: Whether to return routing decisions

        Returns:
            output: (batch, seq, hidden_dim)
            router_logits: Optional routing logits for analysis
        """
        # Route tokens to experts
        router_output: RouterOutput = self.router(
            hidden_states,
            return_aux_loss=self.training,
        )

        # Execute selected experts
        expert_output = self.experts.forward_expert_parallel(
            hidden_states,
            router_output.expert_indices,
            router_output.expert_weights,
        )

        # Add shared expert output if present
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            expert_output = expert_output + shared_output

        # Store aux loss for training
        self.last_aux_loss = router_output.aux_loss

        if return_router_logits:
            return expert_output, router_output.router_logits
        return expert_output, None

    def get_active_expert_indices(self) -> torch.Tensor:
        """Get indices of experts that were used in last forward."""
        return self.router.usage_history.nonzero().squeeze(-1)


class HydraNetBlock(nn.Module):
    """
    Single transformer block with attention and MoE FFN.

    Architecture:
        x -> LayerNorm -> Attention -> + -> LayerNorm -> MoE -> +
        |______________________________|  |__________________|
    """

    def __init__(
        self,
        config: HydraNetConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Pre-normalization
        self.input_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        # Attention
        self.self_attn = GroupedQueryAttention(config, layer_idx)

        # MoE FFN
        self.moe = MoELayer(config, layer_idx)

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout) if config.hidden_dropout > 0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        return_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[torch.Tensor]]:
        """
        Forward pass through transformer block.

        Args:
            hidden_states: (batch, seq, hidden_dim)
            attention_mask: Causal attention mask
            position_ids: Position indices for RoPE
            past_key_value: KV cache from previous step
            use_cache: Whether to return updated KV cache
            return_router_logits: Whether to return routing decisions

        Returns:
            hidden_states: Output tensor
            present_key_value: Updated KV cache
            router_logits: Optional routing logits
        """
        residual = hidden_states

        # Pre-norm + Attention
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + self.dropout(attn_output)

        # Pre-norm + MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        moe_output, router_logits = self.moe(
            hidden_states,
            return_router_logits=return_router_logits,
        )
        hidden_states = residual + self.dropout(moe_output)

        return hidden_states, present_key_value, router_logits

    @property
    def aux_loss(self) -> torch.Tensor:
        """Get MoE auxiliary loss for this layer."""
        return self.moe.last_aux_loss


class HydraNetPreTrainedModel(nn.Module):
    """Base class for HydraNet models with weight initialization."""

    config_class = HydraNetConfig

    def __init__(self, config: HydraNetConfig):
        super().__init__()
        self.config = config

    def _init_weights(self, module: nn.Module):
        """Initialize weights using config settings."""
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
