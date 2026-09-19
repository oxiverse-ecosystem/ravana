"""Dual-entity extraction for comparison queries.

When a query is classified as `comparison` intent, the two comparison entities
must be extracted from the query structure.  This module handles that purely
structurally — splitting on connectives like ``vs``, ``compared to``,
``versus``, ``difference between`` — with no per-query literals.

Public API::

    extract_comparison_entities("jaguar the car vs the animal top speed")
    # -> ("jaguar the car", "the animal top speed")

    extract_comparison_entities("difference between python and rust")
    # -> ("python", "rust")

    extract_comparison_entities("hello world")   # not a comparison query
    # -> None

The function returns ``None`` when the query contains none of the recognised
comparison connectives (so callers can short-circuit) or when one side is
empty (degenerate).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# ── Connectives ──────────────────────────────────────────────────────────
# Connectives that signal a comparison, ordered from most-specific
# (multi-word) to least-specific so that the first match wins deterministically.
#
# Multi-word connectives (``difference between``) are listed before single-word
# ones (``vs``) so the longer pattern wins when both could match.

_COMPARISON_CONNECTIVES: list[re.Pattern[str]] = [
    # Multi-word: difference between X and Y
    re.compile(r"\bdifference\s+between\s+", re.IGNORECASE),
    # Multi-word: differences between X and Y
    re.compile(r"\bdifferences\s+between\s+", re.IGNORECASE),
    # Multi-word: compared to / compared with
    re.compile(r"\bcompared\s+(?:to|with)\s+", re.IGNORECASE),
    # Multi-word: comparison of / comparison between / comparison of X to Y
    re.compile(r"\bcomparison\s+(?:of|between|to|with)\s+", re.IGNORECASE),
    # Multi-word: distinguish between X and Y
    re.compile(r"\bdistinguish\s+between\s+", re.IGNORECASE),
    # Multi-word: contrast between X and Y
    re.compile(r"\bcontrast\s+between\s+", re.IGNORECASE),
    # Single-word: vs (bounded so "a vs b" works but "vessel" does not)
    re.compile(r"\bvs\b", re.IGNORECASE),
    # Single-word: versus
    re.compile(r"\bversus\b", re.IGNORECASE),
]

# When a multi-word connective (``difference between``, ``comparison of``,
# etc.) appears at the *start* of the query, the two entities are the
# phrases on either side of a secondary separator within the remainder.
_SECONDARY_SEPARATORS: list[re.Pattern[str]] = [
    re.compile(r"\s+and\s+", re.IGNORECASE),
    re.compile(r"\s+vs\.?\s+", re.IGNORECASE),
    re.compile(r"\s+versus\s+", re.IGNORECASE),
]

# Whitespace normaliser.
_WS = re.compile(r"\s+")


def _clean_side(raw: str) -> str:
    """Normalise whitespace, strip punctuation fluff."""
    s = raw.strip().strip(".,!?\"'")
    s = _WS.sub(" ", s).strip()
    return s


def extract_comparison_entities(query: str) -> Optional[Tuple[str, str]]:
    """Extract the two comparison entities from a query string.

    Returns ``None`` when:
      * none of the recognised comparison connectives appear in the query, or
      * one of the two sides is empty after cleaning.

    The split position is the *start* of the first matching connective, and
    the connective text itself is not included in either side.

    For multi-word connectives at the start of the query (e.g. *difference
    between X and Y*), the two entities are split on a secondary separator
    (``and``, ``vs``, ``versus``) found after the connective.
    """
    if not query or not query.strip():
        return None

    # Find the earliest-connective match.
    earliest_match: Optional[re.Match[str]] = None

    for pat in _COMPARISON_CONNECTIVES:
        m = pat.search(query)
        if m and (earliest_match is None or m.start() < earliest_match.start()):
            earliest_match = m

    if earliest_match is None:
        return None

    if earliest_match.start() > 0:
        # Connective is in the middle: "X vs Y" / "X compared to Y"
        left_raw = query[: earliest_match.start()]
        right_raw = query[earliest_match.end() :]
    else:
        # Connective is at the start: "difference between X and Y"
        # Look for a secondary separator in the remainder.
        remainder = query[earliest_match.end() :]
        sec_match: Optional[re.Match[str]] = None
        for pat in _SECONDARY_SEPARATORS:
            m = pat.search(remainder)
            if m and (sec_match is None or m.start() < sec_match.start()):
                sec_match = m
        if sec_match is None:
            # No secondary separator — can't split into two entities.
            return None
        left_raw = remainder[: sec_match.start()]
        right_raw = remainder[sec_match.end() :]

    left = _clean_side(left_raw)
    right = _clean_side(right_raw)

    if not left or not right:
        return None

    return (left, right)


def is_comparison_query(query: str) -> bool:
    """Return True if the query contains a recognised comparison connective.

    This is a cheap pre-check — callers can use it to avoid invoking the full
    entity extractor when the query is clearly not a comparison.
    """
    if not query:
        return False
    return any(p.search(query) for p in _COMPARISON_CONNECTIVES)
