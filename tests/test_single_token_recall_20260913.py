#!/usr/bin/env python3
"""Regression test for single-content-token agent-own-recall (round 2026-09-13).

The round surfaced a failure: "remind me what you said about privacy earlier"
returned RAVANA's canned boot greeting instead of recalling its actual prior
statement about privacy. Root cause: two bugs in _route_agent_own_recall:

1. "remind" was missing from the _stop set, so it stayed as a candidate token
   and polluted both query candidates AND stored src_tokens.
2. The overlap threshold was hardcoded to >=2, but a single-content-token
   query (e.g. "privacy") can never reach 2 overlaps with stored src_tokens.

Both fixed. These tests verify the fix end-to-end through the REAL engine path.
"""
import os
import sys
import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana", "src"), os.path.join(PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine

SUFFIX = "test_single_token_recall_20260913"


def _eng():
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=SUFFIX)
    return e


def test_single_content_token_recall_succeeds():
    """A recall query with only ONE content token must still match a stored
    reply whose src_tokens contain that token. Previously failed because
    _best_overlap < 2 rejected it."""
    e = _eng()
    e._record_own_reply(
        "do you think privacy is worth the inconvenience",
        "i care deeply about privacy. it is a basic right — i was built to protect it.",
        "privacy")
    # "remind" is now in _stop, so the only candidate is "privacy"
    out = e._route_agent_own_recall("remind me what you said about privacy earlier")
    assert out is not None, "expected recall of privacy statement, got None"
    assert "privacy" in out.lower(), f"expected privacy in reply, got {out!r}"


def test_remind_not_in_src_tokens():
    """The word 'remind' must NOT appear in stored src_tokens — it's a recall
    verb, not content. Otherwise it pollutes the overlap computation."""
    e = _eng()
    e._record_own_reply(
        "remind me what you think about privacy",
        "i care deeply about privacy. it is a basic right.",
        "privacy")
    # Check that "remind" is not in any stored src_tokens
    for _k, _entries in e._own_replies.items():
        for _e in _entries:
            _src = _e.get("src_tokens", [])
            assert "remind" not in _src, f"'remind' leaked into src_tokens: {_src}"


def test_multi_token_recall_still_requires_two_overlaps():
    """Multi-content-token queries must still require >=2 overlap to prevent
    incidental-word collisions (the original DEFECT C/D fix)."""
    e = _eng()
    e._record_own_reply(
        "what do you think about board games",
        "i'm still quite unsettled about who i am, and it's been growing as we talk.",
        "games")
    # "music" shares no real content with "board games" — only 1 token overlap
    # if we're lucky, should NOT match
    out = e._route_agent_own_recall("what did you tell me about music")
    assert out is None, f"expected None for unrelated topic, got {out!r}"


def test_remind_verb_excluded_from_candidates():
    """After fixing _stop, 'remind' must be excluded from query candidates."""
    e = _eng()
    e._record_own_reply(
        "do you think privacy is worth the inconvenience",
        "i care deeply about privacy. it is a basic right — i was built to protect it.",
        "privacy")
    # The query "remind me what you said about privacy" should have
    # candidates = ["privacy"] (remind excluded), overlap = 1, min_overlap = 1
    out = e._route_agent_own_recall("remind me what you said about privacy")
    assert out is not None, "expected recall with single content token, got None"


def test_glove_synonym_fact_match():
    """Pass 3 of _match_fact: a query using a synonym of the stored value
    must still match via GloVe cosine >= 0.7. Stores ('i','fear','terrified
    of deep water') and queries 'afraid' — no literal token overlap, but
    GloVe cosine between the two synonyms clears the 0.70 bar."""
    e = _eng()
    # Inject a fact directly into the PersonalFactStore
    e.user_model.personal_facts.assert_fact(
        "i", "fear", "terrified of deep water", confidence=0.8)
    # Query with a synonym ("afraid") that has NO literal token overlap
    # with the stored value ("terrified of deep water").
    result = e._match_fact("afraid")
    assert result is not None, (
        "expected synonym match via GloVe Pass 3, got None")
    attr, val, conf = result
    assert attr == "fear", f"expected attr='fear', got {attr!r}"
    assert "terrified" in val, f"expected 'terrified' in val, got {val!r}"


def teardown_module(module):
    try:
        e = _eng()
        e.stop_background_learning()
    except Exception:
        pass
