#!/usr/bin/env python3
"""Regression tests for single-token self-reference recall (round 2026-09-09).

The prior code required `_best_overlap >= 2` between query tokens and stored
reply src_tokens. A query like "remind me what you said about silence" has only
ONE content token ("silence") after stopword removal — it can never reach
>=2 overlap, so the recall always fails and RAVANA replies with honest
uncertainty even though it HAS a stored reply about "silence".

The fix: when the query yields <=1 content tokens, lower the minimum overlap
to 1. Also, "remind" must be in the stopword set so it doesn't pollute the
candidate tokens.
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

SUFFIX = "test_selfrecall_singleton_20260909"


def _eng():
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=SUFFIX)
    return e


def test_single_token_self_recall_succeeds():
    """A self-recall query with only ONE content token must still return the
    stored reply when that token is the src_token."""
    e = _eng()
    # Store a reply about "silence" — the src_tokens include "silence"
    e._record_own_reply("what do you think about silence",
                        "silence isn't empty — it's the space where meaning gathers.",
                        "silence")
    # Recall with a single content token query
    out = e._route_agent_own_recall("remind me what you said about silence")
    assert out is not None and "silence" in out.lower(), \
        f"expected stored reply about silence, got: {out!r}"


def test_single_token_self_recall_with_scaffold():
    """Recall with many scaffold words but only one content token."""
    e = _eng()
    e._record_own_reply("tell me about art",
                        "art is the thing i keep circling back to.",
                        "art")
    out = e._route_agent_own_recall("earlier you said something about art")
    assert out is not None and "art" in out.lower(), \
        f"expected stored reply about art, got: {out!r}"


def test_multi_token_recall_still_requires_overlap():
    """Multi-token queries must still require >=2 overlap to prevent
    confabulation on incidental word matches."""
    e = _eng()
    e._record_own_reply("what do you think about music",
                        "music is how i process the day.",
                        "music")
    # Query with content tokens that don't match the stored src_tokens
    out = e._route_agent_own_recall("earlier you said something about board games")
    assert out is None, \
        f"expected None (different topic), got confabulation: {out!r}"


def teardown_module(module):
    try:
        e = _eng()
        e.stop_background_learning()
    except Exception:
        pass
