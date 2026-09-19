"""Unit tests for ``comparison_extractor``.

Covers the connective patterns (``vs``, ``versus``, ``compared to``,
``difference between``, ``distinguish between``, ``contrast between``,
``comparison of/between``), edge cases (empty, missing connective, leading/
trailing connectives), whitespace normalization, punctuation stripping,
case insensitivity, and multi-word connective splitting via secondary
separators.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

# Ensure the ravana source tree is importable without a full engine boot.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "ravana", "src"),
    os.path.join(_REPO_ROOT, "ravana_ml", "src"),
    os.path.join(_REPO_ROOT, "ravana-v2", "src"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.comparison_extractor import (  # noqa: E402
    _COMPARISON_CONNECTIVES,
    extract_comparison_entities,
    is_comparison_query,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _assert_entities(
    query: str,
    expected_left: Optional[str],
    expected_right: Optional[str],
    msg: str = "",
) -> None:
    result = extract_comparison_entities(query)
    if expected_left is None:
        assert result is None, f"{msg}: expected None, got {result!r}"
        return
    assert result is not None, f"{msg}: expected ({expected_left!r}, {expected_right!r}), got None"
    left, right = result
    assert left == expected_left, f"{msg}: left mismatch — {left!r} != {expected_left!r}"
    assert right == expected_right, f"{msg}: right mismatch — {right!r} != {expected_right!r}"


# ── Basic connectives ─────────────────────────────────────────────────────


def test_vs_basic():
    _assert_entities(
        "jaguar the car vs the animal top speed comparison",
        "jaguar the car",
        "the animal top speed comparison",
        "vs basic",
    )


def test_vs_with_extra_whitespace():
    _assert_entities(
        "python    vs    rust",
        "python",
        "rust",
        "vs extra whitespace",
    )


def test_versus_basic():
    _assert_entities(
        "samsung galaxy s25 ultra versus iphone 16 pro max",
        "samsung galaxy s25 ultra",
        "iphone 16 pro max",
        "versus basic",
    )


def test_compared_to():
    _assert_entities(
        "iphone 16 pro max compared to samsung galaxy s25 ultra camera",
        "iphone 16 pro max",
        "samsung galaxy s25 ultra camera",
        "compared to",
    )


def test_compared_with():
    _assert_entities(
        "rust compared with go performance",
        "rust",
        "go performance",
        "compared with",
    )


def test_difference_between_with_and():
    # "difference between X and Y" — the "and" is the secondary separator.
    _assert_entities(
        "difference between python and rust",
        "python",
        "rust",
        "difference between + and",
    )


def test_difference_between_with_vs():
    # "difference between X vs Y" — the "vs" is the secondary separator.
    _assert_entities(
        "difference between python vs rust",
        "python",
        "rust",
        "difference between + vs",
    )


def test_differences_between_with_and():
    _assert_entities(
        "differences between cats and dogs as pets",
        "cats",
        "dogs as pets",
        "differences between + and",
    )


def test_distinguish_between_with_and():
    _assert_entities(
        "distinguish between a crocodile and an alligator",
        "a crocodile",
        "an alligator",
        "distinguish between + and",
    )


def test_contrast_between_with_and():
    _assert_entities(
        "contrast between functional and imperative programming",
        "functional",
        "imperative programming",
        "contrast between + and",
    )


def test_comparison_of_with_and():
    _assert_entities(
        "comparison of rust and go for web servers",
        "rust",
        "go for web servers",
        "comparison of + and",
    )


def test_comparison_between_with_and():
    _assert_entities(
        "comparison between mac and windows laptops",
        "mac",
        "windows laptops",
        "comparison between + and",
    )


def test_comparison_between_with_vs():
    # "comparison between X vs Y" — vs works as secondary separator too.
    _assert_entities(
        "comparison between mac vs windows laptops",
        "mac",
        "windows laptops",
        "comparison between + vs",
    )


# ── Case insensitivity ───────────────────────────────────────────────────


def test_vs_uppercase():
    _assert_entities(
        "Python VS Rust",
        "Python",
        "Rust",
        "vs uppercase",
    )


def test_versus_mixed_case():
    _assert_entities(
        "iPhone Versus Samsung",
        "iPhone",
        "Samsung",
        "versus mixed case",
    )


def test_compared_to_mixed_case():
    _assert_entities(
        "Tesla Model 3 Compared To BYD Seal",
        "Tesla Model 3",
        "BYD Seal",
        "compared to mixed case",
    )


# ── Punctuation stripping ────────────────────────────────────────────────


def test_trailing_question_mark():
    _assert_entities(
        "python vs rust?",
        "python",
        "rust",
        "trailing question mark",
    )


def test_trailing_period():
    _assert_entities(
        "python vs rust.",
        "python",
        "rust",
        "trailing period",
    )


def test_surrounding_quotes():
    _assert_entities(
        '"python vs rust"',
        "python",
        "rust",
        "surrounding quotes",
    )


# ── Long entity phrases ──────────────────────────────────────────────────


def test_long_entity_phrases():
    _assert_entities(
        "the new macbook pro with m4 max chip vs the dell xps 16 with intel ultra 9",
        "the new macbook pro with m4 max chip",
        "the dell xps 16 with intel ultra 9",
        "long entity phrases",
    )


# ── No connective → None ─────────────────────────────────────────────────


def test_no_connective_returns_none():
    # A simple informational query — no comparison connective.
    _assert_entities("what is gravity", None, None, "no connective")


def test_juxtaposed_nouns_no_connective():
    # Two nouns side by side without a connective — not a comparison.
    _assert_entities("python rust programming", None, None, "juxtaposed nouns")


def test_and_alone_is_not_connective():
    # "and" by itself is NOT a comparison connective.
    _assert_entities("python and rust", None, None, "and alone")


def test_difference_between_without_secondary_separator():
    # "difference between python rust" — no "and"/"vs"/"versus" separator
    # so we can't determine the two entities.
    _assert_entities(
        "difference between python rust",
        None,
        None,
        "difference between without secondary separator",
    )


# ── Empty / degenerate ───────────────────────────────────────────────────


def test_empty_string_returns_none():
    _assert_entities("", None, None, "empty string")


def test_whitespace_only_returns_none():
    _assert_entities("   \t\n  ", None, None, "whitespace only")


def test_single_word_returns_none():
    _assert_entities("python", None, None, "single word")


def test_connective_at_start_returns_none():
    # "vs rust" — no left side.
    _assert_entities("vs rust", None, None, "connective at start")


def test_connective_at_end_returns_none():
    # "python vs" — no right side.
    _assert_entities("python vs", None, None, "connective at end")


def test_connective_alone_returns_none():
    # Just the connective — neither side.
    _assert_entities("vs", None, None, "connective alone")


# ── is_comparison_query ──────────────────────────────────────────────────


def test_is_comparison_query_true_cases():
    assert is_comparison_query("python vs rust")
    assert is_comparison_query("python versus rust")
    assert is_comparison_query("python compared to rust")
    assert is_comparison_query("difference between cats and dogs")
    assert is_comparison_query("comparison of x and y")
    assert is_comparison_query("distinguish between a and b")
    assert is_comparison_query("contrast between x and y")


def test_is_comparison_query_false_cases():
    assert not is_comparison_query("")
    assert not is_comparison_query("what is gravity")
    assert not is_comparison_query("python programming language")
    assert not is_comparison_query("buy a laptop")


# ── Connectives table is non-empty ───────────────────────────────────────


def test_connectives_table_not_empty():
    # A guard: if the table is empty, every test above would trivially pass
    # because every query would return None.  The table must have entries.
    assert len(_COMPARISON_CONNECTIVES) > 0, (
        "_COMPARISON_CONNECTIVES is empty — the extractor cannot match anything"
    )


# ── Parametric: every connective pattern matches at least one query ──────


def test_every_connective_matches_something():
    """Each compiled pattern in the table must actually match a real query.
    A pattern that never matches is dead code (or wrong)."""
    sample_queries = [
        ("vs", "python vs rust"),
        ("versus", "python versus rust"),
        ("compared to", "x compared to y"),
        ("compared with", "x compared with y"),
        ("difference between", "difference between a and b"),
        ("differences between", "differences between a and b"),
        ("comparison of", "comparison of x and y"),
        ("comparison between", "comparison between x and y"),
        ("comparison to", "comparison of x to y"),
        ("distinguish between", "distinguish between a and b"),
        ("contrast between", "contrast between a and b"),
    ]
    for label, query in sample_queries:
        matched = any(p.search(query) for p in _COMPARISON_CONNECTIVES)
        assert matched, f"no pattern matched sample query for {label!r}: {query!r}"


# ── Entry point ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    import traceback

    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
