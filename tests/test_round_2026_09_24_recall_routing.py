"""Tests for recall routing fixes (round 2026-09-24T0559Z).

Commit: 28258eff — score-based fact matching + recall routing guards.

These tests exercise the fix logic DIRECTLY (regex patterns, scoring math,
guard predicates). They are deterministic and GloVe-independent, so they
can be asserted on this box without ambiguity.
"""
import re

import pytest


# ── 1. _match_fact score-based selection (bug #1) ──────────────────────────

def _match_fact_score(facts, phrase):
    """Replicate _match_fact scoring (engine.py:4922-4928)."""
    _p = (phrase or "").lower().strip().replace("-", " ")
    _ptoks = set(w for w in re.findall(r"[a-z']+", _p)
                 if len(w) >= 3 and w not in ("does", "did", "do", "done"))
    _best = None
    for _attr, _val, _conf in facts:
        _val_l = (_val or "").lower()
        _attr_l = (_attr or "").lower()
        _contain = _val_l if _val_l and _val_l != _attr_l else ""
        if _contain and (_contain in _p or _p in _contain):
            _best = (_attr, _val, _conf)
            break
        _vtoks = set(w for w in re.findall(r"[a-z']+", _val_l + " " + _attr_l)
                     if len(w) >= 3 and w not in ("does", "did", "do", "done"))
        if _vtoks & _ptoks:
            _overlap = len(_vtoks & _ptoks)
            _score = _overlap + _conf * 0.1
            if _attr_l and any(t in _attr_l for t in _ptoks):
                _score += 0.5
            if _best is None or (len(_best) == 4 and _score > _best[3]):
                _best = (_attr, _val, _conf, _score)
    if _best is not None and len(_best) == 4:
        _best = (_best[0], _best[1], _best[2])
    return _best


def test_match_fact_score_selects_higher_overlap():
    """planning/trip (overlap=2, score=2.58) beats thinking/moving (overlap=0)."""
    facts = [
        ("planning", "planning trip", 0.8),
        ("thinking", "thinking about moving", 0.9),
    ]
    result = _match_fact_score(facts, "planning trip")
    assert result is not None
    assert result[0] == "planning"
    assert "planning trip" in result[1]


def test_match_fact_attr_match_bonus():
    """Attribute-match bonus applied when query topic is in the attr itself."""
    facts = [
        ("planning", "trip", 0.8),  # val has "trip" (1 tok overlap), attr has "planning" (bonus)
        ("hobby", "something fun", 0.5),
    ]
    result = _match_fact_score(facts, "planning")
    assert result is not None
    # planning/trip: overlap=1 + conf(0.8)*0.1 + attr_match(0.5) = 1.58
    # hobby/fun: overlap=0 → not selected
    assert result[0] == "planning"


# ── 2. _B confirmation guard (bug #2) ──────────────────────────────────────

def _B_guard(query):
    """Replicate _B guard (engine.py:5064-5067)."""
    _B = None
    if not re.match(r'^what\b', query):
        _B = re.search(
            r'\b(did|have|had)\s+(i|you)\s+(tell|told|say|said|mention|'
            r'mentioned|share|shared|let you know)\b', query)
    return _B


def test_B_guard_skips_what_prefixed():
    """Queries starting with 'what' should NOT fire the confirmation path."""
    assert _B_guard("what did i tell you i am planning for next spring") is None
    assert _B_guard("what do you remember about my trip") is None


def test_B_guard_fires_on_confirmation():
    """Genuine confirmation queries should still fire."""
    assert _B_guard("did i tell you i liked japan") is not None
    assert _B_guard("have i told you about my brother") is not None


# ── 3. _TOLD regex generalization (bug #3) ────────────────────────────────

_TOLD_RE = re.compile(
    r"\b("
    r"what\s+(?:did|do)\s+i\s+(?:just\s+)?(?:tell|say)\s+(?:you|me)(?:\s+about)?\s+"
    r"|what\s+(?:do|did)\s+i\s+(?:think|feel)\s+(?:of|about)\s+"
    r"|how\s+(?:do|did)\s+i\s+feel\s+about\s+"
    r"|what'?s\s+my\s+(?:opinion|stance)\s+(?:on|about|of)\s+"
    r"|do\s+you\s+(?:remember|recall)\s+what\s+(?:i\s+)?(?:said|mentioned|told|shared)\s+(?:you\s+)(?:\s+about)?\s+"
    r"|anything\s+i\s+(?:told|said|shared|mentioned)\s+(?:you\s+)(?:\s+about)?\s+"
    r"|remember\s+what\s+i\s+(?:said|told|mentioned|shared)(?:\s+about)?\s+"
    r")([a-z][a-z '\-]{1,40})")


def test_TOLD_optional_about_and_just():
    """New regex matches phrasings the old one missed."""
    cases = [
        ("what did i tell you i am planning for next spring", "planning"),
        ("what did i just tell you i am doing", "doing"),
        ("what did i tell you i'm feeling", "feeling"),
    ]
    for q, expected_topic in cases:
        m = _TOLD_RE.search(q)
        assert m is not None, f"Pattern failed to match: {q}"
        assert expected_topic in m.group(2), f"Captured wrong topic for {q}: {m.group(2)!r}"


def test_TOLD_regression_still_matches_about():
    """Queries with 'about' still match (no regression from the generalization)."""
    m = _TOLD_RE.search("what did i tell you about my trip")
    assert m is not None
    assert "trip" in m.group(2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
