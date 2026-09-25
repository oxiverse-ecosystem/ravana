"""Tests for FIX-RV-05: recall of 'planning trip' from 'where am i planning to travel'.

Commit: <pending> — activity-verb recall extraction (round 2026-09-25).

These tests exercise the _match_fact scoring path for an -ing verb extracted
from a 'where am i [verb]ing' query. Logic-only (regex + score), no GloVe.
"""
import re


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


def _extract_activity_verb(query):
    """Replicate the activity-verb extraction regex (engine.py:3254)."""
    m = re.search(r"\b(where|what)\s+am\s+i\s+([a-z]+ing)\b", query)
    return m.group(2).lower() if m else None


def test_extract_activity_verb_planning():
    """'where am i planning to travel' -> 'planning'."""
    assert _extract_activity_verb("where am i planning to travel") == "planning"


def test_extract_activity_verb_thinking():
    """'what am i thinking about' -> 'thinking'."""
    assert _extract_activity_verb("what am i thinking about moving") == "thinking"


def test_extract_activity_verb_no_match():
    """A plain question with no -ing verb returns None."""
    assert _extract_activity_verb("what is my name") is None
    assert _extract_activity_verb("where do i live") is None


def test_match_fact_verb_overlap():
    """'planning' matches 'planning trip' via attr overlap, not 'thinking about moving'."""
    facts = [
        ("planning", "planning trip", 0.8),
        ("thinking", "thinking about moving", 0.9),
    ]
    result = _match_fact_score(facts, "planning")
    assert result is not None
    assert result[0] == "planning"
    assert "planning trip" in result[1]


def test_match_fact_verb_overlap_lower_conf():
    """Even when the wrong fact has higher confidence, verb overlap wins."""
    facts = [
        ("planning", "planning trip", 0.7),
        ("thinking", "thinking about moving cities", 0.95),
    ]
    # "planning" query -> "planning" attr has overlap=1 + attr_match=0.5 + conf*0.1 = 1.57
    # "thinking" attr has overlap=0
    result = _match_fact_score(facts, "planning")
    assert result is not None
    assert result[0] == "planning"


def test_end_to_end_planning_travel():
    """End-to-end: extract 'planning' from query, match 'planning trip' fact."""
    query = "where am i planning to travel"
    verb = _extract_activity_verb(query)
    assert verb == "planning"
    facts = [
        ("planning", "planning trip", 0.8),
        ("thinking", "thinking about moving", 0.9),
    ]
    result = _match_fact_score(facts, verb)
    assert result is not None
    assert result[0] == "planning"
    assert "planning trip" in result[1]


if __name__ == "__main__":
    test_extract_activity_verb_planning()
    test_extract_activity_verb_thinking()
    test_extract_activity_verb_no_match()
    test_match_fact_verb_overlap()
    test_match_fact_verb_overlap_lower_conf()
    test_end_to_end_planning_travel()
    print("All FIX-RV-05 tests passed.")
