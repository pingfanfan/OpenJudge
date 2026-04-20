"""Utilities for Needle-in-a-Haystack style long-context benchmarks.

v0.1 uses a simple character-based approximation for token counting (1 token ≈ 4
English chars). Precise tokenization is deferred to a later plan.
"""
from __future__ import annotations

from pathlib import Path

_CORPUS_FILE = Path(__file__).parent / "corpus" / "niah_corpus.txt"
_CHARS_PER_TOKEN = 4


def approximate_token_count(text: str) -> int:
    """Return approximate token count (chars // 4, English heuristic)."""
    return len(text) // _CHARS_PER_TOKEN


def load_corpus() -> str:
    """Return the bundled filler corpus used to build haystacks."""
    return _CORPUS_FILE.read_text(encoding="utf-8")


def build_haystack(
    *,
    target_tokens: int,
    needle: str,
    needle_depth: float,
) -> str:
    """Generate a haystack of approximately `target_tokens` tokens with `needle`
    inserted at relative position `needle_depth` ∈ [0.0, 1.0].
    """
    if not 0.0 <= needle_depth <= 1.0:
        raise ValueError(f"needle_depth must be in [0, 1], got {needle_depth}")

    corpus = load_corpus()
    target_chars = target_tokens * _CHARS_PER_TOKEN

    filler_parts = []
    filler_len = 0
    while filler_len < target_chars:
        filler_parts.append(corpus)
        filler_len += len(corpus)
    filler = "".join(filler_parts)[:target_chars]

    needle_pos = int(len(filler) * needle_depth)
    while needle_pos < len(filler) and filler[needle_pos] not in " \n":
        needle_pos += 1

    haystack = filler[:needle_pos] + " " + needle + " " + filler[needle_pos:]
    return haystack
