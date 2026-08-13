"""
Regression tests for round 2026-08-13T1136Z — fact/opinion mining corruptions.

Three defects surfaced in an empirical 55-turn chat probe and are fixed in
user_model.mine_personal_facts:

  A) NAME POISONING — "i'm still buzzing" (a predicate state, not a proper
     noun) was stored as a `name` fact (`('i','name','still buzzing')`),
     which then echoed verbatim in the self-summary ("your name is Still
     Buzzing"). Fixed: the name-candidate guard now rejects when ANY token is
     an affect/state word OR when the candidate is a single verb-form token
     (gerund/participle, e.g. "buzzing").

  B) ACTIVITY-FACT GARBAGE — denials / meta-loops were stored as `does`/`
     event` facts: "nothing i build works" -> `('i','does','build works')`,
     "i keep replaying the worst case" -> `('i','does','keep replaying')`.
     Fixed: (i) a `does`/`event` object that is a single verb-form token is
     rejected (process, not possession); (ii) activity/event capture is
     skipped entirely when the utterance carries a negative marker
     ("nothing", "never", "don't", ...), because a denial is not an
     assertion that the user does the thing.

  C) LOST VALUE-VERB COMPARATIVE OPINION — "i think handwritten letters mean
     more than texts" produced NO stance (the opinion miner only handled the
     copula "is more <ADJ> than" shape, not "mean/matter more than"). Fixed:
     added a general value-verb comparative pattern (seed verb vocabulary),
     landing the stance on the WINNER (X), with correct polarity.

These are GENERAL grammatical/structural guards (no per-topic hardcoding).
"""

import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (
    PROJ,
    os.path.join(PROJ, "ravana_ml", "src"),
    os.path.join(PROJ, "ravana", "src"),
):
    sys.path.insert(0, _p)

import pytest
from ravana.chat.engine import CognitiveChatEngine


@pytest.fixture()
def eng():
    e = CognitiveChatEngine(
        dim=64, seed=42, baby_mode=True, user_suffix="test_r0813b"
    )
    # Start from a clean durable store so each test is isolated.
    e.user_model.personal_facts.facts.clear()
    e.user_model.opinions.stances.clear()
    yield e
    e.stop_background_learning()


def _facts(e):
    return {
        (k[0], k[1], k[2]): v
        for k, v in e.user_model.personal_facts.facts.items()
    }


def test_A_name_poisoning_predicate_state_not_stored(eng):
    """'i'm still buzzing' is a state, never a name fact."""
    eng.process_turn("i'm still buzzing after that show")
    facts = _facts(eng)
    assert ("i", "name", "still buzzing") not in facts, (
        "predicate state stored as name fact (identity corruption)"
    )
    # A real name still works.
    eng.process_turn("my name is Meera")
    facts = _facts(eng)
    assert ("i", "name", "meera") in facts


def test_B_denial_not_stored_as_activity(eng):
    """'nothing i build works' must not become a 'does' fact."""
    eng.process_turn("nothing i build works the way it should")
    facts = _facts(eng)
    assert all(k[1] != "does" for k in facts), (
        f"denial leaked into activity facts: {[k for k in facts if k[1]=='does']}"
    )


def test_B_meta_loop_not_stored_as_activity(eng):
    """'i keep replaying the worst case' is a meta-loop, not an activity."""
    eng.process_turn("i keep replaying the worst case in my head")
    facts = _facts(eng)
    assert all(k[1] != "does" for k in facts), (
        f"meta-loop leaked into activity facts: {[k for k in facts if k[1]=='does']}"
    )


def test_B_legit_activity_still_stored(eng):
    """Genuine first-person disclosures still land as 'does' facts."""
    eng.process_turn("i keep homing pigeons on the roof")
    facts = _facts(eng)
    assert ("i", "does", "keep homing pigeons") in facts


def test_C_value_verb_comparative_winner_gets_stance(eng):
    """'handwritten letters mean more than texts' -> +stance on letters."""
    eng.process_turn("i think handwritten letters mean more than texts")
    stances = eng.user_model.opinions.stances
    assert "handwritten letters" in stances, (
        f"value-verb comparative lost: stances={list(stances)}"
    )
    assert stances["handwritten letters"].polarity > 0


def test_C_negative_value_verb_comparative(eng):
    """'a quiet night means less than a good conversation' -> -stance on night."""
    eng.process_turn("i think a quiet night means less than a good conversation")
    stances = eng.user_model.opinions.stances
    assert "quiet night" in stances
    assert stances["quiet night"].polarity < 0
