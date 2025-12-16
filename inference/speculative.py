"""
Speculative decoding for MoE models.

Uses n-gram drafting to predict multiple tokens, then verifies in batched forward pass.
Takes advantage of the 17x throughput improvement at batch 16 vs batch 1 for INT4 kernels.

Key insight: Decode is inherently batch=1, but speculative decoding batches verification.
"""

import torch
from collections import defaultdict
from typing import Optional, Tuple, List
import time


class NGramDrafter:
    """
    N-gram based draft token predictor.

    Uses patterns from the current context to predict likely continuations.
    Zero additional memory - just matches against existing token history.
    """

    def __init__(self, n: int = 4, max_context: int = 512):
        """
        Args:
            n: N-gram size (larger = more specific matches, fewer hits)
            max_context: How far back to look for n-gram matches
        """
        self.n = n
        self.max_context = max_context

    def draft(
        self,
        token_ids: torch.Tensor,  # [1, seq_len] current tokens
        num_drafts: int = 4,
    ) -> List[int]:
        """
        Draft next tokens based on n-gram matching.

        Looks for the current n-gram suffix in the context and predicts
        what came after it historically.

        Args:
            token_ids: Current sequence [1, seq_len]
            num_drafts: Number of draft tokens to generate

        Returns:
            List of draft token ids (may be shorter than num_drafts if no matches)
        """
        tokens = token_ids[0].tolist()
        seq_len = len(tokens)

        if seq_len < self.n:
            return []

        drafts = []
        current_suffix = tuple(tokens[-self.n:])

        # Build n-gram index from context
        # Look for where this n-gram appeared before
        context_start = max(0, seq_len - self.max_context)

        for draft_idx in range(num_drafts):
            # Search for current_suffix in history
            match_pos = None
            for i in range(context_start, seq_len - self.n - draft_idx):
                if tuple(tokens[i:i + self.n]) == current_suffix:
                    # Found a match - what came after?
                    next_pos = i + self.n + draft_idx
                    if next_pos < seq_len:
                        match_pos = next_pos
                        break  # Use first match

            if match_pos is not None:
                next_token = tokens[match_pos]
                drafts.append(next_token)
                # Update suffix for next iteration
                current_suffix = (*current_suffix[1:], next_token)
            else:
                # No more matches found
                break

        return drafts


class RepetitionDrafter:
    """
    Draft based on detecting repetitive patterns.

    Looks for repeating sequences (common in MoE outputs) and predicts continuation.
    """

    def __init__(self, max_pattern_len: int = 16):
        self.max_pattern_len = max_pattern_len

    def draft(
        self,
        token_ids: torch.Tensor,
        num_drafts: int = 4,
    ) -> List[int]:
        """Draft by detecting and continuing repetitive patterns."""
        tokens = token_ids[0].tolist()
        seq_len = len(tokens)

        if seq_len < 4:
            return []

        drafts = []

        # Look for repeating patterns of various lengths
        for pattern_len in range(2, min(self.max_pattern_len, seq_len // 2)):
            pattern = tokens[-pattern_len:]

            # Check if this pattern repeated before
            prev_start = seq_len - 2 * pattern_len
            if prev_start >= 0:
                prev_pattern = tokens[prev_start:prev_start + pattern_len]
                if pattern == prev_pattern:
                    # Pattern is repeating! Predict continuation
                    for i in range(num_drafts):
                        idx = i % pattern_len
                        drafts.append(pattern[idx])
                    return drafts

        return drafts


class SpeculativeDecoder:
    """
    Speculative decoding wrapper for any autoregressive model.

    Process:
    1. Generate K draft tokens using drafter (fast, no model calls)
    2. Run model on all K tokens in one batched forward pass
    3. Verify which drafts match model predictions
    4. Accept longest matching prefix, rollback KV cache for rejected tokens

    Speedup comes from batch=K verification vs K individual decode steps.
    """

    def __init__(
        self,
        model,
        tokenizer,
        kv_cache,
        num_speculative: int = 4,
        use_ngram: bool = True,
        use_repetition: bool = True,
        temperature: float = 0.0,  # 0 = greedy
    ):
        """
        Args:
            model: The language model with forward(input_ids, kv_cache)
            tokenizer: Tokenizer for decoding
            kv_cache: KV cache instance (must support set_len for rollback)
            num_speculative: Number of draft tokens to generate
            use_ngram: Enable n-gram based drafting
            use_repetition: Enable repetition-based drafting
            temperature: Sampling temperature (0 = greedy)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache = kv_cache
        self.num_speculative = num_speculative
        self.temperature = temperature

        # Initialize drafters
        self.drafters = []
        if use_ngram:
            self.drafters.append(NGramDrafter(n=4, max_context=512))
        if use_repetition:
            self.drafters.append(RepetitionDrafter(max_pattern_len=16))

        # Stats
        self.stats = {
            'total_accepted': 0,
            'total_drafted': 0,
            'total_steps': 0,
            'draft_time_ms': 0,
            'verify_time_ms': 0,
        }

    def reset_stats(self):
        """Reset statistics."""
        for key in self.stats:
            self.stats[key] = 0

    def get_stats(self) -> dict:
        """Get speculative decoding statistics."""
        stats = self.stats.copy()
        if stats['total_drafted'] > 0:
            stats['acceptance_rate'] = stats['total_accepted'] / stats['total_drafted']
        else:
            stats['acceptance_rate'] = 0.0
        if stats['total_steps'] > 0:
            stats['avg_tokens_per_step'] = (stats['total_accepted'] + stats['total_steps']) / stats['total_steps']
        else:
            stats['avg_tokens_per_step'] = 1.0
        return stats

    def _get_drafts(self, token_ids: torch.Tensor) -> List[int]:
        """Get draft tokens from all drafters, return first non-empty."""
        for drafter in self.drafters:
            drafts = drafter.draft(token_ids, self.num_speculative)
            if drafts:
                return drafts
        return []

    def _sample_token(self, logits: torch.Tensor) -> int:
        """Sample a token from logits."""
        if self.temperature == 0:
            return logits.argmax(dim=-1).item()
        else:
            probs = torch.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1).item()

    @torch.no_grad()
    def generate_step(
        self,
        input_ids: torch.Tensor,  # [1, seq_len] current sequence
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate tokens using speculative decoding.

        Args:
            input_ids: Current token sequence [1, seq_len]

        Returns:
            new_tokens: Tensor of newly generated tokens [1, num_new]
            num_accepted: Number of draft tokens that were accepted
        """
        device = input_ids.device
        self.stats['total_steps'] += 1

        # Step 1: Get draft tokens
        t_draft_start = time.perf_counter()
        draft_tokens = self._get_drafts(input_ids)
        self.stats['draft_time_ms'] += (time.perf_counter() - t_draft_start) * 1000

        if not draft_tokens:
            # No drafts - fall back to standard decode
            t_verify_start = time.perf_counter()

            # Just decode the last token
            if self.kv_cache is not None and self.kv_cache.cur_len > 0:
                # Decode mode: only process last token
                curr_input = input_ids[:, -1:]
            else:
                # Prefill mode
                curr_input = input_ids

            logits, _ = self.model.forward(curr_input, kv_cache=self.kv_cache)

            if self.kv_cache is not None:
                if self.kv_cache.cur_len == 0:
                    self.kv_cache.set_len(input_ids.shape[1])
                else:
                    self.kv_cache.advance(1)

            next_token = self._sample_token(logits[0, -1])
            self.stats['verify_time_ms'] += (time.perf_counter() - t_verify_start) * 1000

            return torch.tensor([[next_token]], device=device), 0

        # Step 2: Create speculative sequence
        # [original_last_token, draft_1, draft_2, ..., draft_k]
        draft_tensor = torch.tensor(draft_tokens, device=device).unsqueeze(0)  # [1, k]
        num_drafts = len(draft_tokens)
        self.stats['total_drafted'] += num_drafts

        # Save KV cache state for potential rollback
        kv_len_before = self.kv_cache.cur_len if self.kv_cache else 0

        # Step 3: Run batched verification
        # We need to run the model on all draft tokens at once
        t_verify_start = time.perf_counter()

        if self.kv_cache is not None and self.kv_cache.cur_len > 0:
            # Decode mode: process draft tokens
            # The KV cache already has context, just process new tokens
            verify_input = draft_tensor  # [1, num_drafts]
        else:
            # First call - need full context + drafts
            verify_input = torch.cat([input_ids, draft_tensor], dim=1)

        logits, _ = self.model.forward(verify_input, kv_cache=self.kv_cache)

        # Update KV cache position for all processed tokens
        if self.kv_cache is not None:
            if kv_len_before == 0:
                self.kv_cache.set_len(input_ids.shape[1] + num_drafts)
            else:
                self.kv_cache.advance(num_drafts)

        self.stats['verify_time_ms'] += (time.perf_counter() - t_verify_start) * 1000

        # Step 4: Verify drafts
        # logits shape: [1, num_drafts, vocab_size]
        # We need to check if model's prediction at position i matches draft[i]
        accepted_tokens = []

        for i in range(num_drafts):
            # Model's prediction for position i
            model_token = self._sample_token(logits[0, i])

            if i < num_drafts and model_token == draft_tokens[i]:
                # Draft matches! Accept it
                accepted_tokens.append(draft_tokens[i])
            else:
                # Mismatch - accept model's token instead and stop
                accepted_tokens.append(model_token)
                break

        num_accepted = len([t for t in accepted_tokens[:-1] if t in draft_tokens])
        self.stats['total_accepted'] += num_accepted

        # Step 5: Rollback KV cache if we rejected some drafts
        tokens_to_keep = len(accepted_tokens)
        tokens_processed = num_drafts

        if tokens_to_keep < tokens_processed and self.kv_cache is not None:
            # Rollback: we processed more tokens than we're keeping
            new_len = kv_len_before + tokens_to_keep
            self.kv_cache.set_len(new_len)

        result = torch.tensor([accepted_tokens], device=device)
        return result, num_accepted


def speculative_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    kv_cache,
    max_new_tokens: int = 32,
    num_speculative: int = 4,
    temperature: float = 0.0,
    verbose: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """
    Generate tokens using speculative decoding.

    Args:
        model: Language model
        tokenizer: Tokenizer
        input_ids: Input token ids [1, seq_len]
        kv_cache: KV cache instance
        max_new_tokens: Maximum tokens to generate
        num_speculative: Draft tokens per step
        temperature: Sampling temperature
        verbose: Print progress

    Returns:
        generated: Full sequence including input [1, total_len]
        stats: Speculative decoding statistics
    """
    decoder = SpeculativeDecoder(
        model=model,
        tokenizer=tokenizer,
        kv_cache=kv_cache,
        num_speculative=num_speculative,
        temperature=temperature,
    )

    generated = input_ids.clone()
    tokens_generated = 0

    # First step: prefill
    if kv_cache is not None:
        with torch.no_grad():
            logits, _ = model.forward(input_ids, kv_cache=kv_cache)
            kv_cache.set_len(input_ids.shape[1])

        next_token = decoder._sample_token(logits[0, -1])
        generated = torch.cat([generated, torch.tensor([[next_token]], device=input_ids.device)], dim=1)
        tokens_generated += 1
        kv_cache.advance(1)

        if verbose:
            token_str = tokenizer.decode([next_token]) if tokenizer else f"<{next_token}>"
            print(f"  1. '{token_str}' (prefill)")

    # Speculative decode loop
    step = 1
    while tokens_generated < max_new_tokens:
        new_tokens, num_accepted = decoder.generate_step(generated)

        generated = torch.cat([generated, new_tokens], dim=1)
        tokens_generated += new_tokens.shape[1]
        step += 1

        if verbose:
            for i, tok in enumerate(new_tokens[0].tolist()):
                token_str = tokenizer.decode([tok]) if tokenizer else f"<{tok}>"
                accepted_str = "✓" if i < num_accepted else ""
                print(f"  {tokens_generated - new_tokens.shape[1] + i + 1}. '{token_str}' {accepted_str}")

        # Check for EOS
        if tokenizer and hasattr(tokenizer, 'eos_token_id'):
            if tokenizer.eos_token_id in new_tokens[0].tolist():
                break

    stats = decoder.get_stats()
    return generated, stats
