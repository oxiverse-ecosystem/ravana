"""Regression tests — feature t_3d147353 (round 2026-08-16T1241Z).

User-model AGGREGATION capability. Queries that ask RAVANA to REPORT the
accumulated CONTENT of its model of the user ("what have you picked up about
me", "what stands out about me", "tell me about myself", "everything you know
about me", "describe me", "what do you remember me telling you") must answer
from the REAL stored profile (facts + stances + beliefs) — NOT fall through to
the graceful-uncertainty fallback that produced degenerate output ("i don't
really have a solid grasp on picked far so far", measured T48 of round
2026-08-16T1241Z) and NOT echo a random episodic memory (T60, "what stands out"
returned an unrelated emotional memory).

Root cause (round 2026-08-16T1241Z): these queries were not recognized as a
distinct intent, so they reached the honest-uncertainty path with a garbage
subject token. The fix detects the aggregation intent in `_structured_recall`
and renders from the live durable stores via `_aggregate_user_model`. Every
slot is read from runtime state RAVANA grows autonomously; no authored reply,
no per-topic table, no retraining. Fail-closed: a brand-new user (nothing
stored) returns None so the honest path handles it.

The test fails WITHOUT the capability (the query returns None or degenerate
text) and passes WITH it (the reply contains the real learned facts/stances).
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine

_SUFFIX = "usermodel_agg_16T1241"


def _cleanup():
    try:
        _p = os.path.join("weights", f"ravana_weights{_SUFFIX}.pkl")
        if os.path.exists(_p):
            os.remove(_p)
    except Exception:
        pass


def _teach(eng):
    """Feed RAVANA a real, varied profile: facts + stances + a belief."""
    eng.process_turn("my name is corvin")
    eng.process_turn("i grew up in a village called aldermoor in the hills")
    eng.process_turn("my niece priya is an astronomer who studies pulsars")
    eng.process_turn("i keep a pet fox named vesper who steals my socks")
    eng.process_turn("i love the sea")
    eng.process_turn("i really dislike being put on the spot in meetings")
    eng.process_turn("freedom to read the source matters more to me than convenience, always")


AGG_QUERIES = (
    "what have you picked up about me",
    "what stands out about me",
    "tell me about myself",
    "everything you know about me",
    "describe me",
    "what do you remember me telling you",
    "what's your read on me",
)


def test_aggregation_renders_real_profile():
    """Each aggregation query must surface the REAL learned facts/stances,
    not degenerate uncertainty text and not an unrelated episodic echo."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix=_SUFFIX)
    _teach(eng)
    for q in AGG_QUERIES:
        ans = eng._structured_recall(q)
        assert ans is not None, f"aggregation reply missing for {q!r}"
        low = ans.lower()
        # Real learned facts are present (name + a disclosed fact + a stance).
        assert "corvin" in low, f"reply does not reference learned name: {ans!r}"
        assert ("aldermoor" in low or "priya" in low or "vesper" in low), \
            f"reply does not reference learned facts: {ans!r}"
        # Real learned stance (sea=for, meetings=against) is present.
        assert ("sea" in low and "for" in low) or ("meetings" in low), \
            f"reply does not reference learned stances: {ans!r}"
        # Fail: must NOT be the degenerate uncertainty fallback.
        assert "solid grasp" not in low, \
            f"degenerate uncertainty leaked: {ans!r}"
        # Fail: must NOT dump a verbatim episodic quote.
        assert "you told me earlier" not in low, \
            f"episodic echo leaked: {ans!r}"
        # Regression: mined fragments must not be listed twice (one
        # disclosure can be mined into several overlapping facts; the
        # aggregator must dedupe by value content, not per-entity). A
        # doubled "; you <fact>" segment is a real defect.
        _segs = [s.strip() for s in ans.split(";") if s.strip()]
        assert len(_segs) == len(set(_segs)), \
            f"duplicate fragment in aggregate (dedupe regressed): {ans!r}"
    _cleanup()


def test_aggregation_fails_closed_on_empty_profile():
    """A brand-new user with nothing stored must return None (fail-closed),
    so the honest-uncertainty path answers instead of fabricating content."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix=_SUFFIX)
    for q in ("what have you picked up about me", "tell me about myself"):
        ans = eng._structured_recall(q)
        assert ans is None, f"empty profile should fail closed, got: {ans!r}"
    _cleanup()


def test_non_aggregation_query_unaffected():
    """A plain biographical or meta-identity query must NOT be intercepted by
    the aggregation branch; those paths still work and the aggregator returns
    None for them."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix=_SUFFIX)
    _teach(eng)
    # name query still resolves via the bio path
    ans = eng._structured_recall("what's my name")
    assert ans is not None and "corvin" in ans.lower(), f"name query broke: {ans!r}"
    # aggregation detector itself must not fire for a normal knowledge query
    ans2 = eng._structured_recall("what is a pulsar, exactly?")
    # (either None — handled elsewhere — or a real answer; MUST NOT be the
    #  aggregation summary, which would be wrong for a world-knowledge query)
    assert ans2 is None or "here's what i've picked up about you" not in (ans2 or "").lower(), \
        f"aggregation wrongly fired on knowledge query: {ans2!r}"
    _cleanup()
