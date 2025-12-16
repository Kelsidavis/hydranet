"""
Inference utilities for HydraNet.

- Speculative decoding: N-gram based drafting for batched verification
"""

from .speculative import (
    NGramDrafter,
    RepetitionDrafter,
    SpeculativeDecoder,
    speculative_generate,
)

__all__ = [
    "NGramDrafter",
    "RepetitionDrafter",
    "SpeculativeDecoder",
    "speculative_generate",
]
