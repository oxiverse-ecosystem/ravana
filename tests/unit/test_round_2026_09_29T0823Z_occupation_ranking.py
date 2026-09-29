"""
Regression test — round 2026-09-29T0823Z, card t_af1e837d.

DEFECT (LIMITATION:needs-research): "what do i do for a living" returned the
FIRST-INSERTED `does:` fact, not the occupation.

Seed: "i run a pottery studio in kochi" + "i adopted a stray dog yesterday"
    ('i', 'does:run',     'run pottery studio')            <- the OCCUPATION
    ('i', 'does:adopted', 'adopted stray dog yesterday')   <- a one-off act

    Q: what do i do for a living -> "you adopted stray dog yesterday."   (WRONG)
    Q: what do i do for work     -> "you adopted stray dog yesterday."   (WRONG)
    Q: what do i do              -> "you adopted stray dog yesterday."   (WRONG)

The right fact was in the store the whole time. Two selection defects combined:
  1. the _ACT loop returned the first match in dict INSERTION order, so
     whichever activity was mined first won, and
  2. it matched on bare substring (`_verb in _val`) — "do" is a substring of
     "adopted", so the near-empty verb of an occupation query matched the
     one-off act by luck.

FIX: an occupation query now RANKS the activity facts by how much each reads
as a SUSTAINED livelihood (its verb is a livelihood verb, or its object is a
workplace / production unit), using seed lexicons in user_model that RAVANA can
extend online via learn_occupation_role. The answer is still the LIVE fact
value — this selects among real state and authors nothing.

The trigger is TYPE-AGNOSTIC (an occupation noun, or an activity query naming
no content at all), not a branch for one phrasing. A "what's my job" query never
matches the _ACT regex at all, yet is the same class of question, so the bridge
is standalone.

All assertions are STATE-DRIVEN: they read the fact store and the recall
output, never an authored reply string. Every test is RED-capable — each fails
against the pre-fix engine.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.user_model import (
    is_occupation_query,
    is_livelihood_verb,
    is_livelihood_object,
)


def _eng(suffix):
    for f in (f"weights/ravana_weights{suffix}.pkl",
              f"weights/ravana_usermodel{suffix}.pkl"):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)
    e.stop_background_learning()
    return e


def _seed_pottery(suffix):
    e = _eng(suffix)
    e.user_model.mine_personal_facts("i run a pottery studio in kochi")
    e.user_model.mine_personal_facts("i adopted a stray dog yesterday")
    return e


# ── the headline defect ──────────────────────────────────────────────────────

def test_occupation_query_beats_first_inserted_one_off_act():
    e = _seed_pottery("_occ_headline")
    for q in ("what do i do for a living", "what do i do for work"):
        out = e._structured_recall(q)
        assert out is not None, f"{q!r} returned nothing"
        assert "pottery" in out, f"{q!r} -> {out!r}, expected the pottery studio"
        assert "dog" not in out, f"{q!r} -> {out!r}, leaked the one-off act"


def test_content_free_activity_query_ranks_the_occupation():
    # "what do i do" names no content noun — the same question, other shape.
    e = _seed_pottery("_occ_bare")
    out = e._structured_recall("what do i do")
    assert out is not None and "pottery" in out, f"-> {out!r}"


def test_ranking_is_independent_of_mine_order():
    # The defect was insertion-order dependence: reversing the seed order must
    # not change the answer.
    e = _eng("_occ_order")
    e.user_model.mine_personal_facts("i adopted a stray dog yesterday")
    e.user_model.mine_personal_facts("i run a pottery studio in kochi")
    out = e._structured_recall("what do i do for a living")
    assert out is not None and "pottery" in out, f"-> {out!r}"
    assert "dog" not in out, f"-> {out!r}"


# ── the trigger is type-agnostic, not one phrasing ───────────────────────────

def test_occupation_trigger_covers_the_query_class():
    # Every phrasing of the SAME question must be recognised...
    for q in ("what do i do for a living", "what do i do for work",
              "what's my job", "what is my job", "what is my occupation",
              "what's my profession", "what is my career",
              "how do i make a living", "what's my line of work"):
        assert is_occupation_query(q), f"{q!r} not recognised as occupation"
    # ...and a NON-occupation query must not be swept in.
    for q in ("what do i keep", "what do i play", "where do i keep the bees",
              "what do i do for fun", "what do i do tomorrow"):
        assert not is_occupation_query(q), f"{q!r} misread as occupation"


def test_job_phrasing_answers_without_the_verb_do():
    # "what's my job" never matches the _ACT regex (which needs
    # "what do i <verb>"), so the bridge must be reachable without it.
    e = _seed_pottery("_occ_job")
    out = e._structured_recall("what's my job")
    assert out is not None and "pottery" in out, f"-> {out!r}"


# ── generalization beyond the probe's profession ────────────────────────────

def test_generalizes_to_a_different_livelihood():
    e = _eng("_occ_gen")
    e.user_model.mine_personal_facts("i keep bees")
    e.user_model.mine_personal_facts("i teach chemistry at a college")
    out = e._structured_recall("what do i do for a living")
    assert out is not None and "teach" in out, f"-> {out!r}"
    assert "bees" not in out, f"-> {out!r}"


# ── honesty: never name a one-off act as the user's job ─────────────────────

def test_no_livelihood_facts_does_not_confabulate_a_job():
    e = _eng("none")
    e.user_model.mine_personal_facts("i adopted a stray dog yesterday")
    out = e._structured_recall("what do i do for a living")
    assert out is None or "dog" not in out, \
        f"named a one-off act as the occupation: {out!r}"


def test_stated_work_fact_is_the_occupation():
    # "i work as X" is an occupation stated outright; it must win.
    e = _eng("_occ_work")
    e.user_model.mine_personal_facts("i adopted a stray dog yesterday")
    e.user_model.mine_personal_facts("i work as an apiarist")
    e.user_model.mine_personal_facts("i run a pottery studio in kochi")
    out = e._structured_recall("what do i do for a living")
    assert out is not None and "apiarist" in out, f"-> {out!r}"


# ── the seed must be a SEED: growable online, persisted ─────────────────────

def test_occupation_vocabulary_grows_online():
    # A profession the seed lexicons have never seen becomes addressable after
    # the user says it — no code edit, no retrain.
    e = _eng("_occ_grow")
    assert not e.user_model.occupation_verbs()
    e.user_model.mine_personal_facts("i work as an apiarist")
    assert "apiarist" in e.user_model.occupation_verbs()


def test_learned_profession_word_scores_without_being_in_the_seed():
    # The learned word is what does the work: with an empty seed the word does
    # not score, and once learned it does. Proves growth, not a frozen table.
    assert is_livelihood_verb("apiarist") is False
    assert is_livelihood_verb("apiarist", {"apiarist"}) is True
    assert is_livelihood_object("apiarist", {"apiarist"}) is True


def test_learned_profession_wins_over_a_one_off_act():
    e = _eng("_occ_grow2")
    e.user_model.mine_personal_facts("i adopted a stray dog yesterday")
    e.user_model.personal_facts.assert_fact(
        "i", "does:apiarist", "apiarist my own hives", 0.7)
    e.user_model.learn_occupation_role("apiarist")
    out = e._structured_recall("what do i do for a living")
    assert out is not None and "apiarist" in out, f"-> {out!r}"
    assert "dog" not in out, f"-> {out!r}"


def test_learned_occupation_vocabulary_persists():
    e = _eng("_occ_persist")
    e.user_model.learn_occupation_role("apiarist")
    state = e.user_model.get_state()
    assert "apiarist" in state.get("_occupation_verbs", [])
    e2 = _eng("_occ_persist2")
    e2.user_model.set_state(state)
    assert "apiarist" in e2.user_model.occupation_verbs()
    assert is_livelihood_verb("apiarist", e2.user_model.occupation_verbs())


# ── the substring correction, in isolation ──────────────────────────────────

def test_query_verb_does_not_match_inside_another_word():
    # "do" is a substring of "adopted"; a word-boundary verb match is what keeps
    # a one-off act from being selected as the answer to an occupation query.
    e = _seed_pottery("_occ_substr")
    out = e._structured_recall("what do i do")
    assert out is not None and "adopted" not in out, \
        f"verb matched inside a word: {out!r}"
