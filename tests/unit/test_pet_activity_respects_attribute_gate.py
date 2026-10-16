"""Regression: a pet-ACTIVITY branch must not answer an ATTRIBUTE question.

DEFECT (measured, round 2026-10-03T2040Z, clean engine, RAVANA_OFFLINE=1):

    "my dog biscuit sleeps on the wedging table"
    "what is my dog called"
        -> "your dog biscuit sleeps wedging table."

The pet's NAME was in the store the whole time. The question was about the name
and got the animal's ACTIVITY — the right animal, the wrong attribute, stated
confidently. This is the documented D1 confabulation shape, re-entering through
a different door: `_ACT_CUES` in the (1c-pet) activity resolver contains the
generic interrogative "what", so every "what ..." pet question counted as an
activity ask.

The cue set cannot be fixed by growing it — "what" is a QUESTION word, not an
activity word, and every attribute question begins with it. The question is
really about ATTRIBUTE AGREEMENT, which the engine already owns in exactly one
place: `attribute_gate.asks_name_only`. Reusing it there makes the activity and
name branches agree by construction, and it stands down correctly for a genuine
compound ask with no new cue vocabulary.

This file is the RED half: it fails on the pre-fix source and passes after.
"""
import os
import sys

import pytest

PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for _p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src",
           f"{PROJ}\\ravana-v2\\src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("RAVANA_OFFLINE", "1")


@pytest.fixture(scope="module")
def engine():
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="attrgate2040")
    eng.process_turn("my dog biscuit sleeps on the wedging table")
    yield eng
    try:
        eng.stop_background_learning()
    except Exception:
        pass


@pytest.mark.parametrize("query", [
    "what is my dog called",
    "what is my dog's name",
    "what is my dog name",
])
def test_name_query_answers_with_the_name(engine, query):
    """The defect: an activity answered a question about the name.

    NOTE the apostrophe-less contraction ("whats my dog called") is deliberately
    NOT asserted here. Measured separately this round: it reaches an honest
    "noted." fallback rather than a wrong answer, because query normalisation
    does not expand that contraction before the recall gate — a DIFFERENT defect
    (contraction normalisation), not attribute agreement. Folding it in here
    would have meant widening the subject under test to make a test pass.
    """
    reply = (engine.process_turn(query) or "").lower()
    assert "biscuit" in reply, (
        "name query answered without the name: %r" % (reply,))
    assert "sleeps" not in reply and "wedging" not in reply, (
        "activity leaked into a name answer: %r" % (reply,))


def test_genuine_activity_query_still_answers_with_the_activity(engine):
    """The capability this branch exists for must survive the fix."""
    reply = (engine.process_turn("what does my dog do") or "").lower()
    assert "sleeps" in reply, (
        "pet activity recall regressed: %r" % (reply,))


def test_compound_question_still_answers_the_activity(engine):
    """`asks_name_only` stands the gate down for a COMPOUND ask, so a user who
    asks for both must still get the activity. This is why the fix reuses that
    predicate instead of blanket-blocking the branch on the word "name"."""
    reply = (engine.process_turn(
        "what is my dog called and what does it do") or "").lower()
    assert "sleeps" in reply, (
        "compound ask lost its activity answer: %r" % (reply,))


def test_activity_is_still_in_the_store(engine):
    """Guard against 'fixing' the recall by deleting the activity fact. The
    activity must remain available — only the wrong QUESTION may stop using it."""
    facts = engine.user_model.personal_facts.facts
    acts = [v.value for k, v in facts.items() if k[1].endswith("_activity")]
    assert acts, "pet activity fact was removed from the store"