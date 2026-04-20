from prism.utils.haystack import (
    approximate_token_count,
    build_haystack,
    load_corpus,
)


def test_approximate_token_count():
    """1 token ≈ 4 English chars."""
    assert approximate_token_count("") == 0
    assert 90 <= approximate_token_count("a" * 400) <= 110


def test_load_corpus_returns_nonempty_text():
    text = load_corpus()
    assert len(text) > 100
    assert "The" in text or "the" in text


def test_build_haystack_hits_target_length_within_tolerance():
    needle = "The special code is HONEYBADGER."
    target_tokens = 500
    hs = build_haystack(target_tokens=target_tokens, needle=needle, needle_depth=0.5)
    actual = approximate_token_count(hs)
    assert 0.9 * target_tokens <= actual <= 1.1 * target_tokens
    assert needle in hs


def test_build_haystack_needle_at_expected_depth():
    needle = "The magic phrase is PURPLE_ELEPHANT_42."
    hs = build_haystack(target_tokens=400, needle=needle, needle_depth=0.25)
    idx = hs.find(needle)
    assert idx > 0
    relative_depth = idx / len(hs)
    assert 0.15 <= relative_depth <= 0.35, f"needle at {relative_depth:.2f}, expected ~0.25"


def test_build_haystack_rejects_bad_depth():
    import pytest
    with pytest.raises(ValueError, match="depth"):
        build_haystack(target_tokens=100, needle="x", needle_depth=1.5)
    with pytest.raises(ValueError, match="depth"):
        build_haystack(target_tokens=100, needle="x", needle_depth=-0.1)
