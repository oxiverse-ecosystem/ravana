"""Test: belief disclosure with perfect-aspect stative verbs is captured.

Round 2026-10-19T0528Z limitation #3: "i have always believed that technology
makes people more isolated" was acked as "noted." — the belief content was lost.

The fix extends _process_self_disclosure_stmt to parse
"i have (always/long/never) believed/thought/felt/held that X" constructions
and store the belief content as a personal fact + stance.
"""
import os
import sys
import pytest

# Set up paths
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(PROJ, "ravana", "src"),
          os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana-v2", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("RAVANA_OFFLINE", "1")


def _make_engine():
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_belief")
    return eng


def test_belief_disclosure_captured():
    """'i have always believed that X' must store the belief content."""
    eng = _make_engine()
    q = "i have always believed that technology makes people more isolated"
    reply = eng.process_turn(q)
    # The belief content must be stored as a personal fact
    store = eng.user_model.personal_facts
    facts = [(s, a, v) for (s, a, v), f in store.facts.items()
             if not f.superseded and s == "i" and a == "belief"]
    assert len(facts) >= 1, f"belief fact not stored; facts={list(store.facts.keys())[:5]}"
    # The belief content must mention the topic
    belief_val = facts[0][2].lower()
    assert "technology" in belief_val or "isolated" in belief_val, \
        f"belief content missing topic: {belief_val!r}"


def test_belief_disclosure_long_thought():
    """'i have long thought Y' must store the belief content."""
    eng = _make_engine()
    q = "i have long thought that silence is actually wonderful"
    reply = eng.process_turn(q)
    store = eng.user_model.personal_facts
    facts = [(s, a, v) for (s, a, v), f in store.facts.items()
             if not f.superseded and s == "i" and a == "belief"]
    assert len(facts) >= 1, "belief fact not stored for 'long thought'"


def test_belief_disclosure_never_felt():
    """'i have never felt that Z' must store the belief content."""
    eng = _make_engine()
    q = "i have never felt that money brings happiness"
    reply = eng.process_turn(q)
    store = eng.user_model.personal_facts
    facts = [(s, a, v) for (s, a, v), f in store.facts.items()
             if not f.superseded and s == "i" and a == "belief"]
    assert len(facts) >= 1, "belief fact not stored for 'never felt'"


def test_belief_disclosure_have_held():
    """'i have held that X' must store the belief content."""
    eng = _make_engine()
    q = "i have held that the mountains are the best kind of teacher"
    reply = eng.process_turn(q)
    store = eng.user_model.personal_facts
    facts = [(s, a, v) for (s, a, v), f in store.facts.items()
             if not f.superseded and s == "i" and a == "belief"]
    assert len(facts) >= 1, "belief fact not stored for 'have held'"


def test_belief_ack_mentions_content():
    """The ack for a belief disclosure must mention the belief content."""
    eng = _make_engine()
    q = "i have always believed that technology makes people more isolated"
    reply = eng.process_turn(q)
    reply_lower = reply.lower()
    # The ack must not be a hollow "noted." or "got it — thanks for telling me."
    assert "noted." not in reply_lower or "believe" in reply_lower, \
        f"hollow ack: {reply!r}"
    assert "got it" not in reply_lower or "believe" in reply_lower, \
        f"degenerate ack: {reply!r}"
