"""Regression tests for D6 (round 2026-08-19T1628Z): the agent-opinion topic
extractor must not leak a first/second-person pronoun as the stance target.

Root cause: _route_self_query's agent-opinion branch stripped a closed-class
list from the query tail but excluded person pronouns + contractions, so a
user-referential opinion frame set the target to the pronoun itself:
  "what do you make of my lutefisk habit" -> "i'm still forming a view on my"
  "how do you feel about me leaving the water" -> "...on me"
  "do you think i'm contradictory" -> "...on i'm"
The topic is the real noun the pronoun modifies, not the pronoun. The fix
extends the exclusion to all person pronouns + contractions and guards the
empty-target fall-through.

FAILS on pre-fix code, PASSES after. Exercises the REAL engine.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJ, f"{_PROJ}\\ravana_ml\\src", f"{_PROJ}\\ravana\\src", f"{_PROJ}\\ravana-v2\\src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest


@pytest.fixture(scope="module")
def engine():
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(
        dim=64, seed=42, baby_mode=True,
        user_suffix="d6_test_" + str(os.getpid()))
    # Prime context.
    eng.process_turn("i live on a houseboat moored in a quiet lagoon.")
    eng.process_turn("i have a pet crow named mica.")
    yield eng
    try:
        eng.stop_background_learning()
    except Exception:
        pass


@pytest.mark.parametrize("query,bad_tail", [
    ("what do you make of my lutefisk habit now?", "my"),
    ("how do you feel about that, me leaving the water?", "me"),
    ("do you think i'm contradictory, or just changing?", "i'm"),
    ("what's your take on my brother tomas's motorcycles?", "my"),
    ("how do you feel about us moving away?", "us"),
])
def test_opinion_target_not_pronoun(engine, query, bad_tail):
    r = engine.process_turn(query)
    assert r is not None, f"empty reply for {query!r}"
    # Only the TOPIC slot (the words after "view on") must not be the pronoun;
    # the reply's opening "i'm" is fine.
    _slot = r.lower().split("view on", 1)[-1]
    assert bad_tail not in _slot, (
        f"pronoun leaked into opinion topic for {query!r}: {r!r}")


def test_opinion_target_resolves_real_topic(engine):
    # The real noun the pronoun modifies must be the topic.
    r = engine.process_turn("what do you make of my lutefisk habit now?")
    assert "lutefisk" in r.lower(), f"expected real topic 'lutefisk' in reply: {r!r}"
    _slot = r.lower().split("view on", 1)[-1]
    assert "my" not in _slot, f"topic resolved to pronoun: {r!r}"
