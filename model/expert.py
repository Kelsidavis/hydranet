"""Expert FFN modules for HydraNet MoE."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import HydraNetConfig


class ExpertFFN(nn.Module):
    """
    Single expert feed-forward network using SwiGLU activation.

    Architecture:
        hidden -> gate_proj -> silu
        hidden -> up_proj   ->
        (gate * up) -> down_proj -> hidden

    This is the same architecture used in LLaMA, Mistral, etc.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        # SwiGLU projections
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Track usage statistics for caching decisions
        self.register_buffer("usage_count", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("last_used", torch.tensor(0, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert.

        Args:
            x: (batch * tokens_for_this_expert, hidden_dim)

        Returns:
            output: (batch * tokens_for_this_expert, hidden_dim)
        """
        # SwiGLU: down(silu(gate(x)) * up(x))
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        output = self.down_proj(gate * up)
        return self.dropout(output)

    def update_usage(self, step: int):
        """Update usage statistics for cache management."""
        self.usage_count += 1
        self.last_used.fill_(step)

    @property
    def num_parameters(self) -> int:
        """Total parameters in this expert."""
        return sum(p.numel() for p in self.parameters())

    @property
    def size_bytes(self, dtype_bytes: int = 2) -> int:
        """Size in bytes (default fp16)."""
        return self.num_parameters * dtype_bytes


class ExpertContainer(nn.Module):
    """
    Container for managing multiple experts with efficient execution.

    Supports:
    - Batched execution across experts
    - Expert state serialization for caching
    - Usage tracking for LRU eviction
    """

    def __init__(
        self,
        config: HydraNetConfig,
        num_experts: int,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.layer_idx = layer_idx

        # Create all experts
        self.experts = nn.ModuleList([
            ExpertFFN(
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.expert_intermediate_dim,
                dropout=config.expert_dropout,
            )
            for _ in range(num_experts)
        ])

        # Track which experts are currently on GPU
        self.register_buffer(
            "experts_on_gpu",
            torch.ones(num_experts, dtype=torch.bool),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute experts for given tokens.

        Args:
            hidden_states: (batch, seq, hidden_dim)
            expert_indices: (batch, seq, top_k) - which experts for each token
            expert_weights: (batch, seq, top_k) - weight for each expert

        Returns:
            output: (batch, seq, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        top_k = expert_indices.shape[-1]

        # Flatten for processing
        flat_hidden = hidden_states.view(-1, hidden_dim)  # (batch*seq, hidden)
        flat_indices = expert_indices.view(-1, top_k)      # (batch*seq, top_k)
        flat_weights = expert_weights.view(-1, top_k)      # (batch*seq, top_k)

        # Initialize output
        output = torch.zeros_like(flat_hidden)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            # expert_mask: (batch*seq, top_k) bool tensor
            expert_mask = flat_indices == expert_idx

            if not expert_mask.any():
                continue

            # Get token indices and their k-positions for this expert
            token_indices, k_positions = torch.where(expert_mask)

            if len(token_indices) == 0:
                continue

            # Get hidden states for these tokens
            expert_input = flat_hidden[token_indices]

            # Execute expert
            expert_output = self.experts[expert_idx](expert_input)

            # Get weights for this expert's contribution
            weights = flat_weights[token_indices, k_positions].unsqueeze(-1)

            # Accumulate weighted output
            output.index_add_(0, token_indices, expert_output * weights)

        return output.view(batch_size, seq_len, hidden_dim)

    def forward_expert_parallel(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Alternative forward that groups by expert for better GPU utilization.

        More efficient when many tokens go to the same expert.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        top_k = expert_indices.shape[-1]
        num_tokens = batch_size * seq_len

        # Flatten inputs
        flat_hidden = hidden_states.view(num_tokens, hidden_dim)
        flat_indices = expert_indices.view(num_tokens * top_k)
        flat_weights = expert_weights.view(num_tokens * top_k)

        # Create position mapping: which original token each expert call belongs to
        token_positions = torch.arange(num_tokens, device=hidden_states.device)
        token_positions = token_positions.unsqueeze(1).expand(-1, top_k).reshape(-1)

        # Sort by expert for coalesced execution
        sorted_experts, sort_indices = flat_indices.sort()
        sorted_tokens = token_positions[sort_indices]
        sorted_weights = flat_weights[sort_indices]

        # Find boundaries between experts
        expert_counts = torch.bincount(sorted_experts, minlength=self.num_experts)
        expert_offsets = torch.cumsum(expert_counts, dim=0)
        expert_offsets = torch.cat([torch.tensor([0], device=expert_offsets.device), expert_offsets[:-1]])

        # Initialize output
        output = torch.zeros(num_tokens, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)

        # Execute each expert on its batch of tokens
        for expert_idx in range(self.num_experts):
            start = expert_offsets[expert_idx].item()
            count = expert_counts[expert_idx].item()

            if count == 0:
                continue

            # Get token indices for this expert
            token_idx = sorted_tokens[start:start + count]
            weights = sorted_weights[start:start + count].unsqueeze(-1)

            # Get inputs and execute
            expert_input = flat_hidden[token_idx]
            expert_output = self.experts[expert_idx](expert_input)

            # Accumulate weighted output
            output.index_add_(0, token_idx, expert_output * weights)

        return output.view(batch_size, seq_len, hidden_dim)

    def get_expert_state(self, expert_idx: int) -> dict:
        """Get serialized state of single expert for caching."""
        expert = self.experts[expert_idx]
        return {
            "gate_proj": expert.gate_proj.weight.data.clone(),
            "up_proj": expert.up_proj.weight.data.clone(),
            "down_proj": expert.down_proj.weight.data.clone(),
            "usage_count": expert.usage_count.item(),
            "last_used": expert.last_used.item(),
        }

    def load_expert_state(self, expert_idx: int, state: dict):
        """Load expert state from cache."""
        expert = self.experts[expert_idx]
        expert.gate_proj.weight.data.copy_(state["gate_proj"])
        expert.up_proj.weight.data.copy_(state["up_proj"])
        expert.down_proj.weight.data.copy_(state["down_proj"])
        expert.usage_count.fill_(state["usage_count"])
        expert.last_used.fill_(state["last_used"])


class SharedExpert(nn.Module):
    """
    Shared expert that processes all tokens.

    This expert captures common patterns that don't need specialization,
    reducing the load on routed experts.
    """

    def __init__(self, config: HydraNetConfig):
        super().__init__()
        self.expert = ExpertFFN(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.expert_intermediate_dim,
            dropout=config.expert_dropout,
        )
        # Learnable scale for shared expert contribution
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Process all tokens through shared expert."""
        return self.expert(hidden_states) * self.scale
