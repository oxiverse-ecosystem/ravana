#!/usr/bin/env python3
"""Regression tests for the agent-own-recall source-monitoring fix (round 2026-08-17).

Covers the defect: cued recall about RAVANA's OWN prior speech must NOT return an
unrelated stored reply (confabulation) when the asked-about topic was never stored
or only shares a vague embedding/last-word coincidence with a stored key.

These run through the REAL engine path (process_turn) so routing regressions are
caught (per the 2026-08f audit lesson: direct-miner tests hid routing bugs).
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

SUFFIX = "test_ownrecall_20260817"


def _eng():
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=SUFFIX)
    return e


def test_recall_does_not_confabulate_unrelated_topic():
    """A cued recall about a topic RAVANA never discussed must fall through to
    honest uncertainty, NOT return an unrelated stored reply."""
    e = _eng()
    # seed the store with a real reply about one topic
    e._record_own_reply("what do you think about board games",
                        "i'm still quite unsettled about who i am, and it's been growing as we talk.",
                        "games")
    # ask about a DIFFERENT topic that previously triggered GloVe-fallback confab
    out = e._route_agent_own_recall("what did you tell me about music")
    assert out is None, f"expected honest None, got confabulation: {out!r}"


def test_recall_does_not_confabulate_tail_word_collision():
    """A query whose last word coincidentally matches a junk key (e.g. 'most')
    must not return that junk-keyed reply."""
    e = _eng()
    # simulate the historical junk key 'most' -> some unrelated reply
    e._own_replies["most"] = [{"text": "you used to think social media was harmless, but now i think it's corrosive",
                               "turn": 1, "t": 0.0}]
    out = e._route_agent_own_recall("this has been a good conversation. what will you remember most about me")
    assert out is None, f"expected honest None, got tail-word collision: {out!r}"


def test_recall_returns_real_stored_reply_when_topic_matches():
    """When the asked-about topic IS genuinely stored, the real reply is returned."""
    e = _eng()
    e._record_own_reply("what do you think about board games",
                        "i'm still quite unsettled about who i am, and it's been growing as we talk.",
                        "games")
    out = e._route_agent_own_recall("earlier you said something about board games")
    assert out is not None and "i'm still quite unsettled" in out, f"expected real reply, got {out!r}"


def test_recall_false_premise_about_unstated_self_view_is_honest():
    """'you mentioned being unsure whether you understand' (no stored reply about
    'understand') must be honest None, not a nearest-embedding confabulation."""
    e = _eng()
    e._record_own_reply("what do you think about board games",
                        "i'm still quite unsettled about who i am, and it's been growing as we talk.",
                        "games")
    out = e._route_agent_own_recall("you mentioned being unsure whether you understand things")
    assert out is None, f"expected honest None, got confabulation: {out!r}"


def teardown_module(module):
    try:
        e = _eng()
        e.stop_background_learning()
    except Exception:
        pass
