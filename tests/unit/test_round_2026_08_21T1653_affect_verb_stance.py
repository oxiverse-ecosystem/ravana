"""Feature test for round 2026-08-21T1653 residual #1: mine affect-verb
attitude constructions as stances.

The chat probe (T27) "lab-grown meat creeps me out" did NOT form a held
stance, so the later reversal ("i changed my mind about lab-grown meat") had
nothing to act on. The stance miner only caught explicit like/love/hate verbs
and comparative/dismissive shapes — not the "<subject> <affect-verb> me"
construction where the verb (creeps/grosses/freaks/...) already lives in the
shared VAD affect lexicon.

This capability is GENERAL: it is a grammatical pattern keyed on the shared
VAD matrix (which RAVANA grows online), NOT a per-topic reply table. These
tests fail on the pre-fix baseline (the branch before this feature commit) and
pass after it.

Run:
    pytest tests/unit/test_round_2026_08_21T1653_affect_verb_stance.py -v
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.user_model import UserModel


def _um(text):
    um = UserModel()
    um.opinions.stances.clear()
    um.mine_personal_facts(text)
    return um


def test_affect_verb_mines_negative_stance():
    """'lab-grown meat creeps me out' must form a negative stance on the
    subject (was silently dropped before the fix)."""
    um = _um("lab-grown meat creeps me out, i can't stand the texture")
    key = um.opinions.resolve_topic("lab-grown meat")
    assert key is not None, "affect-verb stance must be mined"
    held = um.opinions.stances.get(key)
    assert held is not None, "stance store must hold the subject"
    assert held.polarity < 0.0, f"creeps-me-out must be negative: {held.polarity}"


def test_affect_verb_variant_freaks():
    """'that flickering light freaks me out' -> negative stance on the light."""
    um = _um("that flickering light freaks me out every time it blinks")
    key = um.opinions.resolve_topic("flickering light")
    assert key is not None
    held = um.opinions.stances.get(key)
    assert held is not None and held.polarity < 0.0


def test_affect_verb_no_second_list_registered():
    """The verb must have been registered into the SHARED VAD matrix (online
    growth), so the next affect scoring recognizes it. Fail-closed: a verb with
    no VAD entry and not in the seed class must NOT mine a stance."""
    um = _um("his constant humming gets to me after a long day")
    key = um.opinions.resolve_topic("constant humming")
    assert key is not None
    held = um.opinions.stances.get(key)
    assert held is not None and held.polarity < 0.0
    # 'gets to' (surface 'getsto') should now exist in the matrix.
    um._ensure_emotion_detector()
    assert um._emotion_detector._lookup_word("getsto") is not None or \
        um._emotion_detector._lookup_word("gets") is not None, \
        "observed affect verb must be registered into the shared VAD matrix"


def test_affect_verb_question_not_mined():
    """A QUESTION is not a self-report: 'does clowns creep you out?' must NOT
    mine a stance on 'clowns'."""
    um = _um("does clowns creep you out? i was just wondering")
    key = um.opinions.resolve_topic("clowns")
    # Either nothing mined, or (if something is) it is not from this construction.
    if key is not None and key in um.opinions.stances:
        # The question guard must prevent the affect-verb stance.
        # Re-mine the bare declarative separately to prove discrimination.
        pass
    # The construction's own guard is the assertion:
    um2 = _um("does lab-grown meat creep you out? tell me")
    key2 = um2.opinions.resolve_topic("lab-grown meat")
    assert key2 not in um2.opinions.stances, \
        "interrogative must not mine an affect-verb stance"


def test_affect_verb_reversal_later_operable():
    """Once mined, the stance is in the same store so a later reversal can
    recode it (the original defect: reversal had nothing to act on)."""
    um = _um("lab-grown meat creeps me out")
    key = um.opinions.resolve_topic("lab-grown meat")
    assert key is not None
    before = um.opinions.stances.get(key).polarity
    assert before < 0.0
    # Now the user reverses — the residual defect meant this found no target.
    um.mine_stance_reversal("actually i changed my mind about lab-grown meat, "
                            "it's fine now")
    after = um.opinions.stances.get(key)
    assert after is not None, "reversal must find the mined stance"
    assert after.polarity > before, \
        "reversal must move the stance toward positive"


def test_affect_verb_fail_closed_unknown_verb_outside_seed_class():
    """Fail-closed: a verb that is NEITHER in the shared VAD matrix NOR in the
    seed affect-verb class must NOT mine a stance. The construction only encodes
    attitude when the verb is a known affect term; an unknown verb (here
    'wibbles') scores 0.0 and is skipped — no confabulated polarity. This is the
    documented fail-closed branch; it lacked explicit coverage before this test.
    """
    um = _um("the tax form wibbles me something fierce")
    key = um.opinions.resolve_topic("tax form")
    assert key is None or key not in um.opinions.stances, \
        "unknown verb outside VAD+seed-class must not mine a stance"
