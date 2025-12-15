"""Router network for MoE expert selection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass

from .config import HydraNetConfig


@dataclass
class RouterOutput:
    """Output from router forward pass."""
    expert_indices: torch.Tensor    # (batch, seq, top_k) - selected experts
    expert_weights: torch.Tensor    # (batch, seq, top_k) - routing weights
    router_logits: torch.Tensor     # (batch, seq, num_experts) - raw logits
    aux_loss: torch.Tensor          # Auxiliary load balancing loss


class TopKRouter(nn.Module):
    """
    Top-K Router for Mixture of Experts.

    Selects top-k experts for each token based on learned routing weights.
    Includes auxiliary losses for load balancing and preventing router collapse.
    """

    def __init__(
        self,
        config: HydraNetConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        self.top_k = config.experts_per_token
        self.capacity_factor = config.expert_capacity_factor

        # Router gate
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # Loss coefficients
        self.aux_loss_coef = config.router_aux_loss_coef
        self.z_loss_coef = config.router_z_loss_coef

        # For tracking expert usage during inference
        self.register_buffer(
            "expert_usage",
            torch.zeros(self.num_experts, dtype=torch.float),
            persistent=False,
        )

        # Expert usage history for predictive prefetching
        self.register_buffer(
            "usage_history",
            torch.zeros(self.num_experts, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_aux_loss: bool = True,
        top_k_override: Optional[int] = None,
    ) -> RouterOutput:
        """
        Route tokens to top-k experts.

        Args:
            hidden_states: (batch, seq, hidden_dim)
            return_aux_loss: Whether to compute auxiliary loss
            top_k_override: Override top-k for this call (e.g., Top-1 for decode)

        Returns:
            RouterOutput with indices, weights, logits, and aux_loss
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Use override if provided (for prefill/decode switching)
        current_top_k = top_k_override if top_k_override is not None else self.top_k

        # Compute router logits
        router_logits = self.gate(hidden_states)  # (batch, seq, num_experts)

        # Apply softmax to get routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            routing_probs, current_top_k, dim=-1
        )  # Both: (batch, seq, top_k)

        # Normalize weights to sum to 1
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights.to(hidden_states.dtype)

        # Compute auxiliary losses for training
        if return_aux_loss and self.training:
            aux_loss = self._compute_aux_loss(router_logits, expert_indices)
        else:
            aux_loss = torch.tensor(0.0, device=hidden_states.device)

        # Update usage statistics
        if not self.training:
            self._update_usage_stats(expert_indices)

        return RouterOutput(
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            router_logits=router_logits,
            aux_loss=aux_loss,
        )

    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary losses for load balancing.

        Two losses:
        1. Load balancing loss: encourages equal expert utilization
        2. Router z-loss: prevents router logits from growing too large
        """
        num_tokens = router_logits.shape[0] * router_logits.shape[1]

        # Load balancing loss
        # f_i: fraction of tokens routed to expert i
        # P_i: average routing probability to expert i
        # Loss = num_experts * sum(f_i * P_i)

        # Count tokens per expert
        expert_mask = F.one_hot(
            expert_indices, num_classes=self.num_experts
        ).float()  # (batch, seq, top_k, num_experts)
        tokens_per_expert = expert_mask.sum(dim=(0, 1, 2))  # (num_experts,)
        f = tokens_per_expert / (num_tokens * self.top_k)

        # Average routing probability per expert
        routing_probs = F.softmax(router_logits, dim=-1)
        P = routing_probs.mean(dim=(0, 1))  # (num_experts,)

        load_balance_loss = self.num_experts * (f * P).sum()

        # Router z-loss: log(sum(exp(logits)))^2
        # Prevents any single expert from dominating
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = (log_z ** 2).mean()

        total_aux_loss = (
            self.aux_loss_coef * load_balance_loss +
            self.z_loss_coef * z_loss
        )

        return total_aux_loss

    def _update_usage_stats(self, expert_indices: torch.Tensor):
        """Update expert usage statistics for cache management."""
        flat_indices = expert_indices.view(-1)
        for idx in flat_indices.unique():
            count = (flat_indices == idx).sum()
            self.usage_history[idx] += count

    def get_expert_priorities(self) -> torch.Tensor:
        """
        Get expert priority scores for cache management.

        Higher score = more important to keep on GPU.
        """
        # Simple heuristic: usage count (could be more sophisticated)
        return self.usage_history.float()

    def predict_next_experts(
        self,
        hidden_states: torch.Tensor,
        top_n: int = 8,
    ) -> torch.Tensor:
        """
        Predict likely experts for prefetching.

        Uses current hidden states to predict which experts
        will be needed, allowing async prefetch.
        """
        with torch.no_grad():
            router_logits = self.gate(hidden_states)
            # Get top-n most likely experts across all positions
            probs = F.softmax(router_logits, dim=-1)
            avg_probs = probs.mean(dim=(0, 1))  # Average across batch and sequence
            _, top_experts = torch.topk(avg_probs, top_n)
            return top_experts

    def reset_usage_stats(self):
        """Reset usage statistics (call periodically)."""
        self.usage_history.zero_()


class ExpertChoiceRouter(nn.Module):
    """
    Expert Choice routing (alternative to Top-K).

    Instead of tokens choosing experts, experts choose tokens.
    Can provide better load balancing but less flexible.
    """

    def __init__(
        self,
        config: HydraNetConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        self.capacity_factor = config.expert_capacity_factor

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expert-choice routing.

        Each expert selects its top-k tokens based on affinity scores.
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_tokens = batch_size * seq_len

        # Tokens per expert based on capacity
        tokens_per_expert = int(num_tokens * self.capacity_factor / self.num_experts)

        # Compute affinity scores
        router_logits = self.gate(hidden_states.view(num_tokens, -1))
        router_logits = router_logits.t()  # (num_experts, num_tokens)

        # Each expert selects top tokens
        expert_weights, expert_indices = torch.topk(
            router_logits, tokens_per_expert, dim=-1
        )

        # Normalize weights
        expert_weights = F.softmax(expert_weights, dim=-1)

        return expert_indices, expert_weights, router_logits.t()


class SparseRouter(nn.Module):
    """
    Router with hard sparsity for efficiency.

    Uses straight-through estimator during training
    for discrete expert selection.
    """

    def __init__(
        self,
        config: HydraNetConfig,
        layer_idx: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.temperature = temperature

        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        self.top_k = config.experts_per_token

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> RouterOutput:
        """Forward with temperature-scaled softmax."""
        router_logits = self.gate(hidden_states)

        # Temperature scaling
        scaled_logits = router_logits / self.temperature

        # Gumbel-softmax for differentiable discrete selection during training
        if self.training:
            routing_probs = F.gumbel_softmax(scaled_logits, tau=self.temperature, hard=True)
            expert_weights, expert_indices = routing_probs.topk(self.top_k, dim=-1)
        else:
            routing_probs = F.softmax(scaled_logits, dim=-1)
            expert_weights, expert_indices = routing_probs.topk(self.top_k, dim=-1)

        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        return RouterOutput(
            expert_indices=expert_indices,
            expert_weights=expert_weights.to(hidden_states.dtype),
            router_logits=router_logits,
            aux_loss=torch.tensor(0.0, device=hidden_states.device),
        )
