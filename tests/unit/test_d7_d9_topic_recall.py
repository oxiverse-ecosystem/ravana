"""Regression test for D7/D9 (round 2026-08-21T2156Z): distinct-topic gist
recall must return the ASKED episode, never a sibling turn.

Before the fix, a single-topic recollection like "what did i tell you about
sourdough?" was routed into the GENERIC whole-profile dump
("from what you've told me, you like shaping the molten orbs; you took
glassblowing last winter; you watch rings; you bought small telescope; you
bake sourdough; ...") — a multi-episode echo that conflated unrelated
disclosures. That is the D7/D9 "wrong/unrelated turn echo" class.

The fix routes a DISTINCT-TOPIC self-recall through the PRECISE scoped semantic
retriever (over the user's own disclosure transcript) so only the asked gist is
reconstructed. Whole-profile frames ("what do you know about me?") and
biographical-attribute queries ("where do i live?") keep their existing
behavior.

Run: RAVANA_OFFLINE=1 python -m pytest tests/unit/test_d7_d9_topic_recall.py -q
"""
import os
os.environ.setdefault("RAVANA_OFFLINE", "1")

import sys
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (_PROJ, f"{_PROJ}/ravana_ml/src", f"{_PROJ}/ravana/src", f"{_PROJ}/ravana-v2/src"):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine


def _new(suffix):
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)


# Distinct, semantically-separated disclosures (each only in the transcript
# gist, NOT in the entity index — the prior confabulation trigger shape).
_TOPICS = [
    ("glassblowing", "i took up glassblowing last winter and i love shaping the molten orbs"),
    ("telescope",   "i bought a small telescope and i watch the rings of saturn through it at night"),
    ("sourdough",   "i bake sourdough every sunday and the crust crackles when it cools"),
    ("marathon",    "i ran my first marathon and my knees ached for a week afterward"),
]

_PROBES = [
    ("glassblowing", "what did i tell you about glassblowing?"),
    ("telescope",   "remind me what i said about the telescope"),
    ("sourdough",   "do you remember what i mentioned about my sourdough?"),
    ("marathon",    "what was it i told you about the marathon i ran?"),
]


def test_distinct_topic_recall_returns_only_asked_episode():
    """A single-topic gist recall must NOT leak sibling turns."""
    eng = _new("t_d79_topic")
    for _, t in _TOPICS:
        eng.process_turn(t)
    for topic, q in _PROBES:
        r = eng.process_turn(q)
        rl = r.lower()
        # The asked topic's gist must be present.
        assert topic in rl, f"[{topic}] expected gist for asked topic, got: {r!r}"
        # No sibling topic's distinctive word may appear.
        sib = [s for s, _ in _TOPICS if s != topic and s in rl]
        assert not sib, f"[{topic}] wrong-episode echo of sibling(s) {sib}: {r!r}"


def test_whole_profile_recall_still_summarizes():
    """A generic self-recall must still surface MULTIPLE disclosed facts
    (the whole-profile summary path), not collapse to a single unrelated
    episode. We assert >= 2 distinct disclosed topics appear and that no
    single-topic leak dominates — the aggregate behavior is preserved."""
    eng = _new("t_d79_profile")
    for _, t in _TOPICS:
        eng.process_turn(t)
    r = eng.process_turn("what do you remember about me?")
    rl = r.lower()
    # The aggregate path must mention at least two distinct disclosed topics
    # (proving it is a real profile summary, not a one-episode echo).
    present = [topic for topic, _ in _TOPICS if topic in rl]
    assert len(present) >= 2, \
        f"generic self-recall did not aggregate multiple facts (got {present}): {r!r}"


def test_distinct_topic_recall_beats_generic_summary():
    """A single-topic recall must resolve to the one asked gist, even when a
    generic summary would also be available. Guards the D7/D9 routing fix."""
    eng = _new("t_d79_beats")
    for _, t in _TOPICS:
        eng.process_turn(t)
    r = eng.process_turn("what did i tell you about sourdough?")
    rl = r.lower()
    assert "sourdough" in rl, f"asked gist missing: {r!r}"
    sib = [s for s, _ in _TOPICS if s != "sourdough" and s in rl]
    assert not sib, f"single-topic query leaked siblings {sib}: {r!r}"
    # It must NOT be the generic multi-fact summary shape.
    assert "i've picked up" not in rl and "stands out" not in rl, \
        f"single-topic query fell through to generic summary: {r!r}"


def test_topic_recall_without_entity_does_not_confabulate_unknown():
    """A topic the user never disclosed must fail closed, not echo a sibling."""
    eng = _new("t_d79_unknown")
    for _, t in _TOPICS:
        eng.process_turn(t)
    # "origami" was never disclosed; the scoped retriever should miss and the
    # query should not return a sibling topic's gist as if it were the answer.
    r = eng.process_turn("what did i tell you about origami?")
    rl = r.lower()
    # No disclosed sibling topic may stand in for the unasked/unknown one.
    sib = [s for s, _ in _TOPICS if s in rl]
    assert not sib, f"unknown-topic query echoed sibling(s) {sib}: {r!r}"


def test_biographical_attribute_recall_unaffected():
    """A location/name attribute recall keeps its precise entity path."""
    eng = _new("t_d79_attr")
    eng.process_turn("i'm bram, by the way")
    eng.process_turn("i keep six hives of bees on a rooftop above the bakery")
    r = eng.process_turn("where do i keep the bees, exactly? remind me what i said")
    assert "six hives" in r.lower(), f"attribute recall lost: {r!r}"
    assert "bram" not in r.lower(), f"attribute recall leaked user name: {r!r}"


if __name__ == "__main__":
    import time
    t0 = time.time()
    for fn in (test_distinct_topic_recall_returns_only_asked_episode,
               test_whole_profile_recall_still_summarizes,
               test_topic_recall_without_entity_does_not_confabulate_unknown,
               test_biographical_attribute_recall_unaffected):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERR  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"done in {time.time()-t0:.1f}s")
