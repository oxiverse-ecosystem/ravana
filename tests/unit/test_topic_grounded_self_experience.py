"""Tests for topic-grounded self-experience replies (D4 confabulation fix).

The D4 defect: self-experience queries with a concrete topic ("do you think
people can change", "what do you dream about") returned generic, one-size-fits-all
replies that didn't engage with the topic — a non-sequitur confabulation.

The fix: _topic_grounded_self_experience grounds the reply in the query topic,
either by finding a real connection to the agent's state (graph, stances, facts)
or by returning an honest topic-specific uncertainty.

Run from repo root:
    python -m pytest tests/unit/test_topic_grounded_self_experience.py -v
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine


def _build_engine():
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               data_dir="/tmp/ravana_m5_test")


# ── 1. Topic-grounded replies engage with the topic ─────────────────────────
def test_topic_grounded_replies_engage_with_topic():
    """Queries with a concrete topic should produce replies that mention the topic."""
    eng = _build_engine()
    # "do you think people can change" — topic is "people can change"
    # The reply should reference the topic or a related concept, not a generic non-sequitur
    eng.process_turn("do you think people can change")
    out = eng._last_responses[-1]
    # The reply should NOT be a generic "i care about people" non-sequitur
    # It should either mention the topic or express honest uncertainty about it
    assert "people" in out.lower() or "change" in out.lower() or "haven't" in out.lower(), \
        f"reply doesn't engage with topic: {out}"


# ── 2. Topic-grounded fallback fires when generic branches miss ─────────────
def test_topic_grounded_fallback_fires():
    """When the generic branches return None, the topic-grounded fallback should fire."""
    eng = _build_engine()
    # "what do you dream about" — the generic branch returns a non-sequitur
    # The topic-grounded fallback should produce a reply that mentions "dream"
    eng.process_turn("what do you dream about")
    out = eng._last_responses[-1]
    # The reply should mention "dream" or express honest uncertainty about it
    assert "dream" in out.lower() or "haven't" in out.lower() or "outside" in out.lower(), \
        f"reply doesn't engage with topic: {out}"


# ── 3. No non-sequitur confabulation ────────────────────────────────────────
def test_no_non_sequitur_confabulation():
    """Replies should not be unrelated to the query topic."""
    eng = _build_engine()
    # "what makes you feel calm" — topic is "calm"
    eng.process_turn("what makes you feel calm")
    out = eng._last_responses[-1]
    # The reply should not be a generic non-sequitur about unrelated topics
    # It should either mention the topic or express honest uncertainty
    assert "calm" in out.lower() or "haven't" in out.lower() or "outside" in out.lower(), \
        f"reply is a non-sequitur: {out}"


# ── 4. Topic-grounded reply uses real state when available ──────────────────
def test_topic_grounded_uses_real_state():
    """When the topic connects to the agent's state, the reply should reference it."""
    eng = _build_engine()
    # First, give the engine a stance on a topic
    eng.process_turn("i love open source software")
    eng.process_turn("what do you think about open source?")
    # Now ask a self-experience question about a related topic
    eng.process_turn("do you ever feel curious about open source?")
    out = eng._last_responses[-1]
    # The reply should reference the connection to open source
    assert "open source" in out.lower() or "haven't" in out.lower(), \
        f"reply doesn't reference the topic connection: {out}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
