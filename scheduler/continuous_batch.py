"""
Continuous batching and group-by-expert scheduling.

Key concepts:
- Continuous batching: Don't wait for all requests to finish
- Microbatch pipelining: Overlap H2D with compute across microbatches
- Group-by-expert: Gather tokens by expert for efficient MLP execution

This is the throughput-first serving loop for agentic workloads.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import deque
import threading
import time
from enum import Enum


class RequestState(Enum):
    """State of a generation request."""
    PENDING = "pending"      # Waiting to be scheduled
    PREFILLING = "prefilling"  # Processing prompt
    GENERATING = "generating"  # Generating tokens
    COMPLETED = "completed"  # Done
    CANCELLED = "cancelled"  # Cancelled by user


@dataclass
class Request:
    """A single generation request."""
    request_id: str
    prompt_ids: torch.Tensor  # (prompt_len,) input token IDs
    max_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9

    # State tracking
    state: RequestState = RequestState.PENDING
    generated_ids: List[int] = field(default_factory=list)
    kv_cache_position: int = 0  # Position in KV cache

    # Timing
    submit_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        """Total tokens processed (prompt + generated)."""
        return len(self.prompt_ids) + len(self.generated_ids)

    @property
    def is_active(self) -> bool:
        """Whether request needs processing."""
        return self.state in (RequestState.PREFILLING, RequestState.GENERATING)


@dataclass
class Batch:
    """
    A batch of requests for processing.

    Supports heterogeneous requests at different stages.
    """
    requests: List[Request]
    batch_id: int

    # Token positions for this step
    input_ids: Optional[torch.Tensor] = None  # (batch, seq)
    position_ids: Optional[torch.Tensor] = None  # (batch, seq)

    # KV cache indices (per request)
    kv_indices: Optional[List[int]] = None

    @property
    def batch_size(self) -> int:
        return len(self.requests)

    @property
    def is_prefill(self) -> bool:
        """True if any request is still prefilling."""
        return any(r.state == RequestState.PREFILLING for r in self.requests)


class ContinuousBatcher:
    """
    Continuous batching scheduler.

    Manages a queue of requests and forms batches for processing.
    Unlike traditional batching, requests can join/leave the batch
    at any step.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_queue_size: int = 64,
        batch_timeout_ms: float = 10.0,  # Max wait before forming batch
    ):
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.batch_timeout_ms = batch_timeout_ms

        # Request management
        self.pending_queue: deque[Request] = deque()
        self.active_requests: Dict[str, Request] = {}  # request_id -> Request
        self.completed_requests: Dict[str, Request] = {}

        # Batch tracking
        self.next_batch_id = 0

        # Synchronization
        self._lock = threading.Lock()
        self._queue_not_empty = threading.Condition(self._lock)

    def submit(self, request: Request) -> bool:
        """
        Submit a new request.

        Returns True if accepted, False if queue full.
        """
        with self._lock:
            if len(self.pending_queue) >= self.max_queue_size:
                return False

            request.state = RequestState.PENDING
            self.pending_queue.append(request)
            self._queue_not_empty.notify()
            return True

    def get_batch(self, timeout_ms: Optional[float] = None) -> Optional[Batch]:
        """
        Get next batch for processing.

        Blocks until batch available or timeout.
        """
        timeout = timeout_ms or self.batch_timeout_ms
        deadline = time.time() + timeout / 1000

        with self._lock:
            while True:
                # Check for work
                batch = self._try_form_batch()
                if batch:
                    return batch

                # Wait for more requests or timeout
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None

                self._queue_not_empty.wait(timeout=remaining)

    def _try_form_batch(self) -> Optional[Batch]:
        """Try to form a batch from pending requests."""
        requests = []

        # Add active requests that need processing
        for req in self.active_requests.values():
            if req.is_active and len(requests) < self.max_batch_size:
                requests.append(req)

        # Add new requests from queue
        while self.pending_queue and len(requests) < self.max_batch_size:
            req = self.pending_queue.popleft()
            req.state = RequestState.PREFILLING
            req.start_time = time.time()
            self.active_requests[req.request_id] = req
            requests.append(req)

        if not requests:
            return None

        batch = Batch(
            requests=requests,
            batch_id=self.next_batch_id,
        )
        self.next_batch_id += 1
        return batch

    def complete_request(self, request_id: str):
        """Mark request as completed."""
        with self._lock:
            if request_id in self.active_requests:
                req = self.active_requests.pop(request_id)
                req.state = RequestState.COMPLETED
                req.end_time = time.time()
                self.completed_requests[request_id] = req

    def cancel_request(self, request_id: str):
        """Cancel a request."""
        with self._lock:
            if request_id in self.active_requests:
                req = self.active_requests.pop(request_id)
                req.state = RequestState.CANCELLED
                self.completed_requests[request_id] = req

    def get_stats(self) -> Dict:
        """Get scheduler statistics."""
        with self._lock:
            return {
                "pending_count": len(self.pending_queue),
                "active_count": len(self.active_requests),
                "completed_count": len(self.completed_requests),
            }


class MicrobatchScheduler:
    """
    Microbatch scheduler for prefill pipelining.

    Splits long prefill sequences into microbatches to enable
    overlapping H2D copies with compute.
    """

    def __init__(
        self,
        microbatch_size: int = 4,  # Tokens per microbatch
        num_prefetch_streams: int = 2,
    ):
        self.microbatch_size = microbatch_size
        self.num_prefetch_streams = num_prefetch_streams

    def split_prefill(
        self,
        input_ids: torch.Tensor,  # (1, seq_len)
    ) -> List[Tuple[torch.Tensor, int]]:
        """
        Split prefill input into microbatches.

        Returns list of (microbatch_ids, start_position).
        """
        seq_len = input_ids.shape[1]
        microbatches = []

        for start in range(0, seq_len, self.microbatch_size):
            end = min(start + self.microbatch_size, seq_len)
            mb_ids = input_ids[:, start:end]
            microbatches.append((mb_ids, start))

        return microbatches

    def get_prefetch_experts(
        self,
        current_mb_experts: List[Tuple[int, int]],  # (layer, expert) pairs
        next_mb_idx: int,
        total_mbs: int,
    ) -> List[Tuple[int, int]]:
        """
        Determine which experts to prefetch for next microbatch.

        Simple heuristic: prefetch same experts as current (locality).
        """
        if next_mb_idx >= total_mbs:
            return []

        # Assume next microbatch uses similar experts
        # Real implementation would use router predictions
        return current_mb_experts


@dataclass
class GatheredTokens:
    """Tokens gathered by expert for efficient MLP execution."""
    expert_idx: int
    layer_idx: int

    # Gathered inputs
    token_indices: torch.Tensor  # (num_tokens,) original positions
    hidden_states: torch.Tensor  # (num_tokens, hidden_dim)
    routing_weights: torch.Tensor  # (num_tokens,) weights

    @property
    def num_tokens(self) -> int:
        return self.token_indices.shape[0]


class GroupByExpertGatherer:
    """
    Groups tokens by their assigned expert for efficient batched MLP.

    Instead of processing token-by-token, we:
    1. Gather all tokens assigned to expert E
    2. Run expert E's MLP once on the batch
    3. Scatter results back

    This is a major throughput win even without Triton kernels.
    """

    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device

    def gather(
        self,
        hidden_states: torch.Tensor,  # (batch*seq, hidden_dim)
        expert_indices: torch.Tensor,  # (batch*seq, top_k)
        expert_weights: torch.Tensor,  # (batch*seq, top_k)
        layer_idx: int,
    ) -> List[GatheredTokens]:
        """
        Gather tokens by expert.

        Args:
            hidden_states: Flattened hidden states
            expert_indices: Expert assignments per token
            expert_weights: Routing weights per token
            layer_idx: Current layer index

        Returns:
            List of GatheredTokens, one per unique expert
        """
        num_tokens, top_k = expert_indices.shape
        gathered = []

        # Find unique experts
        unique_experts = expert_indices.unique().tolist()

        for expert_idx in unique_experts:
            # Find tokens assigned to this expert
            # expert_mask: (num_tokens, top_k) bool
            expert_mask = (expert_indices == expert_idx)

            if not expert_mask.any():
                continue

            # Get token indices and k positions
            token_indices, k_positions = torch.where(expert_mask)

            if len(token_indices) == 0:
                continue

            # Gather hidden states
            gathered_hidden = hidden_states[token_indices]

            # Get routing weights for these tokens
            gathered_weights = expert_weights[token_indices, k_positions]

            gathered.append(GatheredTokens(
                expert_idx=expert_idx,
                layer_idx=layer_idx,
                token_indices=token_indices,
                hidden_states=gathered_hidden,
                routing_weights=gathered_weights,
            ))

        return gathered

    def scatter(
        self,
        gathered_outputs: List[Tuple[GatheredTokens, torch.Tensor]],
        output_shape: Tuple[int, int],  # (num_tokens, hidden_dim)
    ) -> torch.Tensor:
        """
        Scatter expert outputs back to original positions.

        Args:
            gathered_outputs: List of (GatheredTokens, expert_output) pairs
            output_shape: Shape of final output tensor

        Returns:
            (num_tokens, hidden_dim) output tensor
        """
        output = torch.zeros(
            output_shape,
            device=self.device,
            dtype=gathered_outputs[0][1].dtype if gathered_outputs else torch.float16,
        )

        for gathered, expert_out in gathered_outputs:
            # expert_out: (num_gathered, hidden_dim)
            # Apply routing weights
            weighted_out = expert_out * gathered.routing_weights.unsqueeze(-1)

            # Scatter to output positions
            output.index_add_(0, gathered.token_indices, weighted_out)

        return output


class PrefetchSchedule:
    """
    Schedules expert prefetching across microbatches.

    Goal: While computing layer L of microbatch M,
    prefetch experts for layer L of microbatch M+1.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        num_streams: int = 2,
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.num_streams = num_streams

        # Track what's been prefetched
        self.prefetched: Set[Tuple[int, int]] = set()  # (layer, expert)
        self.stream_idx = 0

    def schedule_prefetch(
        self,
        current_layer: int,
        current_experts: List[int],
        next_mb_experts: Optional[List[int]] = None,
    ) -> List[Tuple[int, int, int]]:  # (layer, expert, stream)
        """
        Determine what to prefetch during current layer compute.

        Args:
            current_layer: Layer currently being computed
            current_experts: Experts used in current layer
            next_mb_experts: Predicted experts for next microbatch

        Returns:
            List of (layer, expert, stream_idx) to prefetch
        """
        prefetch_list = []

        # If we know next microbatch's experts, prefetch them
        if next_mb_experts:
            for expert_idx in next_mb_experts:
                key = (current_layer, expert_idx)
                if key not in self.prefetched:
                    stream = self.stream_idx % self.num_streams
                    self.stream_idx += 1
                    prefetch_list.append((current_layer, expert_idx, stream))
                    self.prefetched.add(key)

        return prefetch_list

    def clear(self):
        """Clear prefetch tracking for new sequence."""
        self.prefetched.clear()
        self.stream_idx = 0
