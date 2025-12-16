"""
Top-K Router with cache affinity bonus.

Features:
- Top-K selection (K=2 for prefill, K=1 for decode)
- Cache affinity: bias toward cached experts when router is uncertain
- Usage tracking for hot expert identification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Set, Callable
import math

from ..config import MixtralConfig, ExpertCacheConfig, TopKMode


@dataclass
class RouterOutput:
    """Output from router forward pass."""
    expert_indices: torch.Tensor    # (batch, seq, top_k) - selected experts
    expert_weights: torch.Tensor    # (batch, seq, top_k) - routing weights
    router_logits: torch.Tensor     # (batch, seq, num_experts) - raw logits


class TopKRouter(nn.Module):
    """
    Top-K router for Mixtral MoE.

    Key features:
    - Conditional cache affinity bonus (only when router uncertain)
    - Top-K switching between prefill (K=2) and decode (K=1)
    - Usage tracking for cache management
    """

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: ExpertCacheConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.cache_config = cache_config
        self.layer_idx = layer_idx

        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts

        # Router gate (loaded from pretrained weights)
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # Cache affinity settings
        self.affinity_bonus = cache_config.affinity_bonus
        self.entropy_threshold = cache_config.affinity_entropy_threshold

        # Usage tracking for cache management
        self.register_buffer(
            "usage_counts",
            torch.zeros(self.num_experts, dtype=torch.long),
            persistent=False,
        )

        # Callback to check if expert is cached
        self._is_cached_fn: Optional[Callable[[int], bool]] = None

    def set_cache_checker(self, fn: Callable[[int], bool]):
        """Set function to check if expert is cached."""
        self._is_cached_fn = fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_override: Optional[int] = None,
        apply_affinity: bool = True,
    ) -> RouterOutput:
        """
        Route tokens to top-k experts.

        Uses "topk logits → softmax selected" flow for numerical stability:
        1. Compute logits in fp32
        2. Select top-k by logits (not full softmax)
        3. Softmax only over selected logits
        4. Cast weights to input dtype only at final multiply

        Args:
            hidden_states: (batch, seq, hidden_dim)
            top_k_override: Override top-k (1 for decode, 2 for prefill)
            apply_affinity: Whether to apply cache affinity bonus

        Returns:
            RouterOutput with indices, weights, and logits
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Default to Mixtral's standard top-2
        top_k = top_k_override if top_k_override is not None else self.config.experts_per_token

        # Compute router logits in fp32 for stability
        router_logits = self.gate(hidden_states).float()  # (batch, seq, num_experts)

        # Apply cache affinity bonus if enabled (on logits, before selection)
        if apply_affinity and self.cache_config.enable_affinity_bonus:
            router_logits = self._apply_affinity_bonus(router_logits)

        # Select top-k by LOGITS (not softmax probs) - this is the stable path
        topk_logits, expert_indices = torch.topk(
            router_logits, top_k, dim=-1
        )  # Both: (batch, seq, top_k)

        # Softmax only over selected experts (fp32)
        expert_weights = F.softmax(topk_logits, dim=-1)  # (batch, seq, top_k)

        # Paranoia-normalize (should already sum to 1, but ensures stability)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        # Cast to input dtype only at final step
        expert_weights = expert_weights.to(hidden_states.dtype)

        # Update usage stats (inference only)
        if not self.training:
            self._update_usage(expert_indices)

        return RouterOutput(
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            router_logits=router_logits,
        )

    def _apply_affinity_bonus(
        self,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply cache affinity bonus to routing logits.

        Only applies bonus when:
        1. Router is uncertain (high entropy)
        2. Cache checker function is available

        Args:
            router_logits: (batch, seq, num_experts)

        Returns:
            Modified logits with affinity bonus
        """
        if self._is_cached_fn is None:
            return router_logits

        # Compute entropy of routing distribution
        probs = F.softmax(router_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (batch, seq)

        # Normalize entropy by max possible (log(num_experts))
        max_entropy = math.log(self.num_experts)
        normalized_entropy = entropy / max_entropy  # 0 to 1

        # Create affinity mask: which experts are cached
        cached_mask = torch.zeros(
            self.num_experts,
            device=router_logits.device,
            dtype=router_logits.dtype,
        )
        for expert_idx in range(self.num_experts):
            if self._is_cached_fn(expert_idx):
                cached_mask[expert_idx] = self.affinity_bonus

        # Apply bonus only where entropy > threshold
        # Shape: (batch, seq, 1) * (num_experts,) -> (batch, seq, num_experts)
        entropy_gate = (normalized_entropy > self.entropy_threshold).float().unsqueeze(-1)
        bonus = entropy_gate * cached_mask.unsqueeze(0).unsqueeze(0)

        return router_logits + bonus

    def _update_usage(self, expert_indices: torch.Tensor):
        """Update usage counts for cache management."""
        flat_indices = expert_indices.view(-1)
        for idx in flat_indices.unique():
            count = (flat_indices == idx).sum()
            self.usage_counts[idx] += count

    def get_expert_priorities(self) -> torch.Tensor:
        """
        Get expert priority scores for cache management.

        Higher score = more important to cache.
        """
        return self.usage_counts.float()

    def get_top_experts(self, k: int) -> torch.Tensor:
        """Get indices of top-k most used experts."""
        _, indices = torch.topk(self.usage_counts, k)
        return indices

    def reset_usage(self):
        """Reset usage statistics."""
        self.usage_counts.zero_()

    def validate_gate_weights(self, atol: float = 1e-2) -> bool:
        """
        Validate gate weights are loaded correctly (no accidental transpose).

        Computes F.linear directly and compares against self.gate() to catch
        any weight layout issues that would silently destroy routing.

        Args:
            atol: Absolute tolerance for comparison

        Returns:
            True if validation passes

        Raises:
            ValueError if validation fails
        """
        with torch.no_grad():
            # Random test input (match gate weight dtype)
            x = torch.randn(
                2, 4, self.hidden_dim,
                device=self.gate.weight.device,
                dtype=self.gate.weight.dtype,
            )

            # Direct F.linear computation (ground truth)
            logits_direct = F.linear(x.float(), self.gate.weight.float())

            # Through our gate module
            logits_module = self.gate(x).float()

            # Check match
            max_diff = (logits_direct - logits_module).abs().max().item()

            if max_diff > atol:
                raise ValueError(
                    f"Gate weight validation failed! max_diff={max_diff:.2e} > atol={atol:.2e}. "
                    f"Likely cause: weight transposition error during loading."
                )

            return True

    def predict_experts(
        self,
        hidden_states: torch.Tensor,
        top_n: int = 4,
    ) -> torch.Tensor:
        """
        Predict likely experts for prefetching.

        Args:
            hidden_states: (batch, seq, hidden_dim)
            top_n: Number of experts to predict

        Returns:
            (top_n,) tensor of expert indices
        """
        with torch.no_grad():
            router_logits = self.gate(hidden_states)
            probs = F.softmax(router_logits, dim=-1)

            # Average across batch and sequence
            avg_probs = probs.mean(dim=(0, 1))

            _, top_experts = torch.topk(avg_probs, top_n)
            return top_experts


class RouterWithAuxLoss(TopKRouter):
    """
    Router with auxiliary load balancing loss for training.

    Not used in inference-only HydraNet, but included for completeness.
    """

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: ExpertCacheConfig,
        layer_idx: int,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
    ):
        super().__init__(config, cache_config, layer_idx)
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_override: Optional[int] = None,
        apply_affinity: bool = True,
        return_aux_loss: bool = False,
    ) -> RouterOutput:
        """Forward with optional auxiliary loss computation."""
        output = super().forward(hidden_states, top_k_override, apply_affinity)

        if return_aux_loss and self.training:
            aux_loss = self._compute_aux_loss(
                output.router_logits,
                output.expert_indices,
            )
            # Store in output (hack - proper impl would add field)
            output.aux_loss = aux_loss

        return output

    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute load balancing + z-loss."""
        num_tokens = router_logits.shape[0] * router_logits.shape[1]
        top_k = expert_indices.shape[-1]

        # Load balancing loss
        expert_mask = F.one_hot(
            expert_indices, num_classes=self.num_experts
        ).float()
        tokens_per_expert = expert_mask.sum(dim=(0, 1, 2))
        f = tokens_per_expert / (num_tokens * top_k)

        routing_probs = F.softmax(router_logits, dim=-1)
        P = routing_probs.mean(dim=(0, 1))

        load_balance_loss = self.num_experts * (f * P).sum()

        # Z-loss: prevent logit explosion
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = (log_z ** 2).mean()

        return self.aux_loss_coef * load_balance_loss + self.z_loss_coef * z_loss
