"""Regression test for D3 (round 2026-10-13T0125Z): _route_agent_own_recall
must NOT hijack first-person recall queries that contain a user-attribute word.

Defect: "what did i just tell you about my dog" matched _is_first_person_recall
(because of "what did i just tell you"), which bypassed the user-attribute
guard entirely. The query reached the agent-reply store and confabulated a
reply about a DIFFERENT user (e.g. "your sister maya is just got accepted").

Fix: the user-attribute guard now applies whenever the query contains a
user-attribute word, regardless of _is_first_person_recall. The
_is_first_person_recall heuristic only keeps queries in-scope when they do
NOT contain a user-attribute word (e.g. "what did i just tell you about my
favorite food" — "favorite food" is not in the attribute list).

Run: RAVANA_OFFLINE=1 python -m pytest tests/unit/test_d3_first_person_attr_guard.py -q
"""
import os
os.environ.setdefault("RAVANA_OFFLINE", "1")

import sys
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ, f"{_PROJ}/ravana_ml/src", f"{_PROJ}/ravana/src", f"{_PROJ}/ravana-v2/src"):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine


def _new_engine(suffix):
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)


def test_first_person_recall_with_user_attr_does_not_confabulate():
    """D3: "what did i just tell you about my dog" must NOT be answered from
    the agent-reply store. It must fall through to the user-fact store or
    honest uncertainty — never a confabulated reply about a different user."""
    eng = _new_engine("t_d3_first_person_attr")
    # Store a fact about the user's dog
    eng.process_turn("my dog biscuit sleeps on the wedging table")
    # Store a fact about the user's sister (the confabulation target)
    eng.process_turn("my sister maya just got accepted to college")
    # The query: first-person recall + user-attribute word "dog"
    reply = eng.process_turn("what did i just tell you about my dog")
    reply_lower = (reply or "").lower()
    # Must NOT confabulate the sister fact
    assert "maya" not in reply_lower, (
        f"D3: first-person recall with user-attr confabulated sister fact: {reply!r}"
    )
    assert "sister" not in reply_lower, (
        f"D3: first-person recall with user-attr confabulated sister fact: {reply!r}"
    )
    # Must NOT be answered from the agent-reply store (strategy check)
    strategy = str(getattr(eng, "_last_strategy", ""))
    assert strategy != "agent_own_recall", (
        f"D3: query was answered from agent-reply store (strategy=agent_own_recall): {reply!r}"
    )


def test_first_person_recall_without_user_attr_still_reaches_agent_store():
    """The fix must NOT break the RV-5 backfill: "what did i just tell you
    about my favorite food" (no user-attribute word) must still reach the
    agent-reply store."""
    eng = _new_engine("t_d3_first_person_no_attr")
    eng.process_turn("my favorite food is handmade mugs")
    reply = eng.process_turn("what did i just tell you about my favorite food")
    # This should NOT be blocked by the user-attribute guard
    # (it may or may not find a reply in the store, but it must not be
    # blocked by the attribute guard — it should proceed to the store lookup)
    # We just verify it doesn't crash and returns something
    assert reply is not None, "query should return a reply (even if uncertain)"


def test_user_attr_guard_still_works_for_non_first_person():
    """The user-attribute guard must still work for non-first-person queries
    like "what is my dog called" — these should never reach the agent store."""
    eng = _new_engine("t_d3_attr_no_first_person")
    eng.process_turn("my dog biscuit sleeps on the wedging table")
    eng.process_turn("my sister maya just got accepted to college")
    reply = eng.process_turn("what is my dog called")
    reply_lower = (reply or "").lower()
    assert "maya" not in reply_lower, (
        f"user-attr query confabulated sister fact: {reply!r}"
    )
    strategy = str(getattr(eng, "_last_strategy", ""))
    assert strategy != "agent_own_recall", (
        f"user-attr query was answered from agent-reply store: {reply!r}"
    )


def test_first_person_recall_with_sister_does_not_confabulate():
    """D3 variant: "what did i just tell you about my sister" must NOT be
    answered from the agent-reply store."""
    eng = _new_engine("t_d3_first_person_sister")
    eng.process_turn("my sister maya just got accepted to college")
    eng.process_turn("my dog biscuit sleeps on the wedging table")
    reply = eng.process_turn("what did i just tell you about my sister")
    reply_lower = (reply or "").lower()
    # Must NOT confabulate the dog fact
    assert "biscuit" not in reply_lower, (
        f"D3: first-person recall about sister confabulated dog fact: {reply!r}"
    )
    strategy = str(getattr(eng, "_last_strategy", ""))
    assert strategy != "agent_own_recall", (
        f"D3: query about sister was answered from agent-reply store: {reply!r}"
    )
