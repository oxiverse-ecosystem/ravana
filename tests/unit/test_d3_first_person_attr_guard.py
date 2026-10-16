"""Regression test for D3 (round 2026-10-12T1042Z): first-person recall
queries that name a USER ATTRIBUTE must NOT be answered from the agent-reply
store.

The _is_first_person_recall regex in _route_agent_own_recall matches the
scaffolding "what did i just tell you about" regardless of what follows. When
the query also names a user attribute ("my dog", "my sister", etc.), the
attribute guard below is skipped (because _is_first_person_recall is True),
and the query reaches the agent-reply store — returning an unrelated reply
from a prior turn (measured T27: "what did i just tell you about my dog" ->
"your sister maya is just got accepted").

The fix excludes user-attribute queries from the first-person recall path so
they fall through to the user-fact recall. This test fails without the fix
(the query returns an unrelated agent reply) and passes with it (the query
returns None from _route_agent_own_recall, letting the user-fact path handle
it).

Run: RAVANA_OFFLINE=1 python -m pytest tests/unit/test_d3_first_person_attr_guard.py -q
"""
import os
os.environ.setdefault("RAVANA_OFFLINE", "1")

import sys
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (_PROJ, f"{_PROJ}/ravana_ml/src", f"{_PROJ}/ravana/src", f"{_PROJ}/ravana-v2/src"):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine


def _new_engine(suffix):
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)


def test_d3_first_person_recall_with_user_attr_does_not_confabulate():
    """'what did i just tell you about my dog' must NOT return an unrelated
    agent reply from a prior turn. The _route_agent_own_recall path must
    return None so the query falls through to the user-fact recall.

    Without the fix, _is_first_person_recall is True for this query (it
    matches "what did i just tell you about"), which causes the attribute
    guard to be skipped. The query then reaches the agent-reply store where
    the content tokens ['just', 'dog'] overlap with a stored reply's tokens
    by 2 >= _min_overlap=2, triggering a confabulation."""
    eng = _new_engine("t_d3_dog")
    # Seed a prior turn whose reply contains both "just" and "dog" so the
    # overlap with the query's content tokens is sufficient to match.
    eng.process_turn("my dog just learned a new trick and he is so proud of himself now")
    # The query must NOT be intercepted by _route_agent_own_recall.
    r = eng._route_agent_own_recall("what did i just tell you about my dog")
    assert r is None, (
        f"first-person recall with user attr 'my dog' must return None "
        f"(fall through to user-fact path), got: {r!r}"
    )


def test_d3_first_person_recall_with_sister_attr_does_not_confabulate():
    """'what did i just tell you about my sister' must NOT return an unrelated
    agent reply."""
    eng = _new_engine("t_d3_sister")
    eng.process_turn("my dog biscuit sleeps on the wedging table")
    r = eng._route_agent_own_recall("what did i just tell you about my sister")
    assert r is None, (
        f"first-person recall with user attr 'my sister' must return None, "
        f"got: {r!r}"
    )


def test_d3_first_person_recall_without_user_attr_still_works():
    """A genuine first-person recall of RAVANA's own speech (no user attribute)
    must still reach the agent-reply store. 'what did you say about art' has
    no user-attribute word and should NOT be blocked by the D3 guard."""
    eng = _new_engine("t_d3_art")
    eng.process_turn("i think art is about paying attention to small things")
    # This is a genuine agent-self recall — should NOT return None.
    r = eng._route_agent_own_recall("what did you say about art")
    # The exact reply depends on the store, but it must NOT be None
    # (the D3 guard must not block genuine agent-self recalls).
    assert r is not None, (
        "genuine agent-self recall 'what did you say about art' must NOT be "
        "blocked by the D3 guard"
    )


def test_d3_first_person_recall_favorite_food_still_works():
    """'what did i just tell you about my favorite food' must still reach the
    agent-reply store (RV-5 backfill). 'favorite food' is NOT in the
    user-attribute word list, so the D3 guard must not block it."""
    eng = _new_engine("t_d3_food")
    eng.process_turn("i told you my favorite food is dosa")
    r = eng._route_agent_own_recall("what did i just tell you about my favorite food")
    assert r is not None, (
        "RV-5 backfill: 'what did i just tell you about my favorite food' "
        "must still reach the agent-reply store"
    )
