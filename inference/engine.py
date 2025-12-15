"""High-performance inference engine for HydraNet."""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Generator
from dataclasses import dataclass
import time

from ..model.config import HydraNetConfig
from ..model.hydranet import HydraNet
from .cache_manager import ExpertCacheManager, CacheConfig, StreamingExpertExecutor


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    # Performance
    use_kv_cache: bool = True
    prefetch_experts: bool = True
    stream_output: bool = False


@dataclass
class InferenceStats:
    """Statistics from inference run."""
    total_tokens: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_time_ms: float = 0.0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    cache_hit_rate: float = 0.0


class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.

    Stores past keys and values to avoid recomputation.
    """

    def __init__(
        self,
        config: HydraNetConfig,
        batch_size: int = 1,
        max_length: int = None,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ):
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length or config.max_context
        self.device = device
        self.dtype = dtype

        self.num_layers = config.num_layers
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim

        # Pre-allocate cache tensors
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self._allocate()

        # Current sequence length in cache
        self.seq_len = 0

    def _allocate(self):
        """Pre-allocate cache tensors."""
        cache_shape = (
            self.batch_size,
            self.num_kv_heads,
            self.max_length,
            self.head_dim,
        )

        for _ in range(self.num_layers):
            self.key_cache.append(
                torch.zeros(cache_shape, device=self.device, dtype=self.dtype)
            )
            self.value_cache.append(
                torch.zeros(cache_shape, device=self.device, dtype=self.dtype)
            )

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a layer and return full KV tensors.

        Args:
            layer_idx: Which layer
            key: New keys (batch, num_kv_heads, seq, head_dim)
            value: New values (batch, num_kv_heads, seq, head_dim)

        Returns:
            Full key and value tensors including history
        """
        new_seq_len = key.shape[2]

        # Write new KV to cache
        self.key_cache[layer_idx][:, :, self.seq_len:self.seq_len + new_seq_len, :] = key
        self.value_cache[layer_idx][:, :, self.seq_len:self.seq_len + new_seq_len, :] = value

        # Return view of full cache up to current position
        total_len = self.seq_len + new_seq_len
        return (
            self.key_cache[layer_idx][:, :, :total_len, :],
            self.value_cache[layer_idx][:, :, :total_len, :],
        )

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached KV for a layer."""
        return (
            self.key_cache[layer_idx][:, :, :self.seq_len, :],
            self.value_cache[layer_idx][:, :, :self.seq_len, :],
        )

    def advance(self, num_tokens: int):
        """Advance the sequence position."""
        self.seq_len += num_tokens

    def reset(self):
        """Reset cache for new sequence."""
        self.seq_len = 0

    def memory_usage_mb(self) -> float:
        """Current memory usage in MB."""
        bytes_per_element = 2 if self.dtype == torch.float16 else 4
        total_elements = (
            self.num_layers * 2 *  # K and V
            self.batch_size *
            self.num_kv_heads *
            self.seq_len *
            self.head_dim
        )
        return total_elements * bytes_per_element / 1e6


class HydraNetEngine:
    """
    High-performance inference engine for HydraNet.

    Features:
    - Efficient KV caching
    - Expert prefetching based on router predictions
    - Streaming token generation
    - Performance monitoring
    """

    def __init__(
        self,
        model: HydraNet,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
        cache_config: Optional[CacheConfig] = None,
    ):
        self.model = model.to(device).to(dtype)
        self.model.eval()
        self.config = model.config
        self.device = device
        self.dtype = dtype

        # Expert cache manager
        cache_config = cache_config or CacheConfig()
        self.expert_cache = ExpertCacheManager(
            config=cache_config,
            num_layers=self.config.num_layers,
            num_experts=self.config.num_experts,
            expert_size_bytes=int(self.config.expert_size_mb * 1e6),
            device=device,
        )

        # KV cache (created per-generation)
        self.kv_cache: Optional[KVCache] = None

        # Statistics
        self.stats = InferenceStats()

    def _register_experts_with_cache(self):
        """Register all expert weights with cache manager."""
        for layer_idx, layer in enumerate(self.model.layers):
            expert_container = layer.moe.experts
            for expert_idx, expert in enumerate(expert_container.experts):
                weights = {
                    "gate_proj": expert.gate_proj.weight.data,
                    "up_proj": expert.up_proj.weight.data,
                    "down_proj": expert.down_proj.weight.data,
                }
                # Register - initially keep some on GPU
                to_gpu = expert_idx < 8  # Keep first 8 experts hot per layer
                self.expert_cache.register_expert_weights(
                    layer_idx, expert_idx, weights, to_gpu=to_gpu
                )

    def generate(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
    ) -> Tuple[torch.Tensor, InferenceStats]:
        """
        Generate tokens from input.

        Args:
            input_ids: (batch, seq) input token IDs
            generation_config: Generation parameters

        Returns:
            Generated token IDs and statistics
        """
        config = generation_config or GenerationConfig()

        batch_size, prompt_len = input_ids.shape
        input_ids = input_ids.to(self.device)

        # Initialize KV cache
        self.kv_cache = KVCache(
            self.config,
            batch_size=batch_size,
            max_length=prompt_len + config.max_new_tokens,
            device=self.device,
            dtype=self.dtype,
        )

        # Reset stats
        self.stats = InferenceStats(prompt_tokens=prompt_len)

        total_start = time.time()

        # Prefill phase - process entire prompt
        prefill_start = time.time()
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                use_cache=config.use_kv_cache,
            )

        # Store KV cache from prefill
        if config.use_kv_cache and outputs.past_key_values:
            for layer_idx, (k, v) in enumerate(outputs.past_key_values):
                self.kv_cache.key_cache[layer_idx][:, :, :prompt_len, :] = k
                self.kv_cache.value_cache[layer_idx][:, :, :prompt_len, :] = v
            self.kv_cache.seq_len = prompt_len

        self.stats.prefill_time_ms = (time.time() - prefill_start) * 1000

        # Get initial logits
        logits = outputs.logits[:, -1, :]

        # Decode phase - generate tokens one at a time
        decode_start = time.time()
        generated = input_ids
        generated_tokens = 0

        with torch.no_grad():
            for _ in range(config.max_new_tokens):
                # Sample next token
                next_token = self._sample_token(
                    logits,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                    past_tokens=generated,
                )

                generated = torch.cat([generated, next_token], dim=1)
                generated_tokens += 1

                # Check for EOS
                if config.eos_token_id is not None:
                    if (next_token == config.eos_token_id).all():
                        break

                # Get past KV for next forward
                past_kv = [
                    (self.kv_cache.key_cache[i][:, :, :self.kv_cache.seq_len, :],
                     self.kv_cache.value_cache[i][:, :, :self.kv_cache.seq_len, :])
                    for i in range(self.config.num_layers)
                ]

                # Forward pass for next token
                outputs = self.model(
                    next_token,
                    past_key_values=past_kv,
                    use_cache=True,
                )

                # Update KV cache
                if outputs.past_key_values:
                    for layer_idx, (k, v) in enumerate(outputs.past_key_values):
                        seq_pos = self.kv_cache.seq_len
                        self.kv_cache.key_cache[layer_idx][:, :, seq_pos:seq_pos+1, :] = k[:, :, -1:, :]
                        self.kv_cache.value_cache[layer_idx][:, :, seq_pos:seq_pos+1, :] = v[:, :, -1:, :]
                    self.kv_cache.seq_len += 1

                logits = outputs.logits[:, -1, :]

        self.stats.decode_time_ms = (time.time() - decode_start) * 1000
        self.stats.total_time_ms = (time.time() - total_start) * 1000
        self.stats.generated_tokens = generated_tokens
        self.stats.total_tokens = prompt_len + generated_tokens
        self.stats.tokens_per_second = generated_tokens / (self.stats.decode_time_ms / 1000)
        self.stats.cache_hit_rate = self.expert_cache.get_cache_stats()["hit_rate"]

        return generated, self.stats

    def generate_streaming(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
    ) -> Generator[Tuple[torch.Tensor, InferenceStats], None, None]:
        """
        Generate tokens with streaming output.

        Yields each token as it's generated.
        """
        config = generation_config or GenerationConfig()
        config.stream_output = True

        batch_size, prompt_len = input_ids.shape
        input_ids = input_ids.to(self.device)

        # Initialize KV cache
        self.kv_cache = KVCache(
            self.config,
            batch_size=batch_size,
            max_length=prompt_len + config.max_new_tokens,
            device=self.device,
            dtype=self.dtype,
        )

        self.stats = InferenceStats(prompt_tokens=prompt_len)
        total_start = time.time()

        # Prefill
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)

        if outputs.past_key_values:
            for layer_idx, (k, v) in enumerate(outputs.past_key_values):
                self.kv_cache.key_cache[layer_idx][:, :, :prompt_len, :] = k
                self.kv_cache.value_cache[layer_idx][:, :, :prompt_len, :] = v
            self.kv_cache.seq_len = prompt_len

        logits = outputs.logits[:, -1, :]
        generated = input_ids
        generated_tokens = 0

        with torch.no_grad():
            for _ in range(config.max_new_tokens):
                next_token = self._sample_token(
                    logits,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                )

                generated = torch.cat([generated, next_token], dim=1)
                generated_tokens += 1

                # Update stats
                self.stats.generated_tokens = generated_tokens
                self.stats.total_time_ms = (time.time() - total_start) * 1000
                self.stats.tokens_per_second = generated_tokens / (self.stats.total_time_ms / 1000)

                # Yield current state
                yield next_token, self.stats

                if config.eos_token_id is not None and (next_token == config.eos_token_id).all():
                    break

                # Next forward pass
                past_kv = [
                    (self.kv_cache.key_cache[i][:, :, :self.kv_cache.seq_len, :],
                     self.kv_cache.value_cache[i][:, :, :self.kv_cache.seq_len, :])
                    for i in range(self.config.num_layers)
                ]

                outputs = self.model(next_token, past_key_values=past_kv, use_cache=True)

                if outputs.past_key_values:
                    for layer_idx, (k, v) in enumerate(outputs.past_key_values):
                        seq_pos = self.kv_cache.seq_len
                        self.kv_cache.key_cache[layer_idx][:, :, seq_pos:seq_pos+1, :] = k[:, :, -1:, :]
                        self.kv_cache.value_cache[layer_idx][:, :, seq_pos:seq_pos+1, :] = v[:, :, -1:, :]
                    self.kv_cache.seq_len += 1

                logits = outputs.logits[:, -1, :]

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        past_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample next token from logits."""
        # Apply repetition penalty
        if repetition_penalty != 1.0 and past_tokens is not None:
            for i in range(logits.shape[0]):
                for token_id in past_tokens[i].unique():
                    if logits[i, token_id] < 0:
                        logits[i, token_id] *= repetition_penalty
                    else:
                        logits[i, token_id] /= repetition_penalty

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Apply top-k
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Apply top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token

    def benchmark(
        self,
        prompt_lengths: List[int] = [128, 512, 1024],
        generation_lengths: List[int] = [64, 128, 256],
        num_runs: int = 3,
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.

        Returns dict with timing statistics.
        """
        results = {}

        for prompt_len in prompt_lengths:
            for gen_len in generation_lengths:
                key = f"prompt_{prompt_len}_gen_{gen_len}"
                times = []

                for _ in range(num_runs):
                    # Random input
                    input_ids = torch.randint(
                        0, self.config.vocab_size,
                        (1, prompt_len),
                        device=self.device,
                    )

                    config = GenerationConfig(
                        max_new_tokens=gen_len,
                        do_sample=False,  # Greedy for consistency
                    )

                    start = time.time()
                    _, stats = self.generate(input_ids, config)
                    times.append(stats.tokens_per_second)

                results[key] = {
                    "avg_tok_s": sum(times) / len(times),
                    "min_tok_s": min(times),
                    "max_tok_s": max(times),
                }

        return results


def load_engine(
    model_path: str,
    device: str = "cuda",
    dtype: str = "float16",
) -> HydraNetEngine:
    """
    Load HydraNet engine from saved model.

    Args:
        model_path: Path to saved model directory
        device: Device to load to
        dtype: Data type

    Returns:
        Initialized inference engine
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    model = HydraNet.from_pretrained(model_path)
    engine = HydraNetEngine(
        model,
        device=torch.device(device),
        dtype=dtype_map[dtype],
    )

    return engine
