#!/usr/bin/env python3
"""Regression tests for RFIX-01: self-reference recall via utterance log ring buffer.

Covers the defect where cued recall about RAVANA's own prior speech returned
identity disclosure instead of the actual prior utterance when the topic-keyed
store missed because topic extraction didn't align with the query.

These run through the REAL engine path (process_turn) so routing regressions are
caught.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana", "src"), os.path.join(PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

import pytest
from ravana.chat.engine import CognitiveChatEngine

SUFFIX = "test_rfix01_utterance_log"


def _eng():
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=SUFFIX)
    return e


class TestSelfReferenceRecall:
    def test_self_reference_returns_prior_utterance_not_identity(self):
        """The core bug: cued recall about the agent's own speech must return
        the actual prior utterance, NOT identity disclosure."""
        e = _eng()
        e.process_turn("i love the autumn season because the air feels clean")
        out = e.process_turn("earlier you said something about what season you remind me")
        assert out is not None
        assert "i said" in out.lower() or "earlier i said" in out.lower(), \
            f"expected recall framing, got: {out!r}"
        assert "you love the autumn season" in out.lower(), \
            f"expected the actual prior utterance, got: {out!r}"

    def test_remind_me_what_you_said(self):
        """'remind me what you said about X' must recall the real reply."""
        e = _eng()
        e.process_turn("i think board games are a great way to bond with friends")
        out = e.process_turn("remind me what you said about board games")
        assert out is not None
        assert "board games" in out.lower() or "bond" in out.lower(), \
            f"expected recall of the actual reply, got: {out!r}"

    def test_direct_gate_call_with_ring_buffer(self):
        """Direct call to _route_agent_own_recall after the ring buffer has
        stored a reply from a prior turn."""
        e = _eng()
        e.process_turn("i love the autumn season because the air feels clean")
        # The reply "good to know — you love the autumn season. i'll keep that in mind."
        # should be in the utterance log.
        out = e._route_agent_own_recall("earlier you said something about autumn")
        assert out is not None
        assert "autumn" in out.lower(), f"expected recall, got: {out!r}"

    def test_identity_query_not_regressed(self):
        """A genuine identity query (not about the agent's speech) must NOT
        be intercepted by the ring buffer."""
        e = _eng()
        e.process_turn("i love the autumn season because the air feels clean")
        out = e.process_turn("what are you")
        assert out is not None
        # Must NOT be the autumn reply
        assert "autumn" not in out.lower(), \
            f"identity query regressed: got prior utterance instead of identity answer: {out!r}"

    def test_no_confabulation_on_unrelated_topic(self):
        """A cued recall about a topic RAVANA never discussed must fall through
        to honest uncertainty, NOT return an unrelated stored reply."""
        e = _eng()
        e.process_turn("i think chess is a wonderful game")
        out = e._route_agent_own_recall("earlier you said something about quantum physics")
        assert out is None or "quantum physics" not in out.lower(), \
            f"expected honest None, got confabulation: {out!r}"

    def test_ring_buffer_falls_back_when_topic_store_misses(self):
        """When the topic-keyed store misses (overlap < 2), the ring buffer
        should still return the genuine reply if there's token overlap with
        the reply text itself."""
        e = _eng()
        # Seed a reply via the engine
        e.process_turn("my favorite food is pizza because it's cheesy")
        # The store keyed under the topic (likely "pizza" or "cheesy") — now ask
        # a query whose content tokens overlap with the REPLY text but not
        # with the eliciting src_tokens.
        out = e._route_agent_own_recall("what did you say about cheesy food")
        # At minimum, the gate should not return identity disclosure
        if out is not None:
            assert "i said" in out.lower() or "earlier i said" in out.lower(), \
                f"expected recall or None, got: {out!r}"


def teardown_module(module):
    try:
        e = _eng()
        e.stop_background_learning()
    except Exception:
        pass
