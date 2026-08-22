"""Regression tests for round 2026-08-21T1653Z stance-reversal + prior-stance recall.

Two real defects were found this round (62-turn chat probe):

Defect 1 (root cause): a genuine WHOLE-ATTITUDE reversal whose recant names
only the salient head of a multiword held stance key ("i take it back about the
cold", where the held stance is "cold weather give") was wrongly rejected as
"scope-narrowing" by the subset guards in mine_stance_reversal, so the stale
+0.95 stance stayed pinned. The narrowing guards must only fire when the recant
carries an explicit scope marker ("only"/"just"/...), not for a bare whole-
attitude reversal.

Defect 2: a query about the user's PAST attitude after a reversal ("did i used
to say i loved cold weather", "what was my original take ... before i flipped")
fell through to world-knowledge and emitted a confident WRONG subject ("loved is
a bit outside what i know"). It must answer from the user's OWN stance store via
the prior_stance episodic trace recorded by reverse_stance / recode_stance_toward.

These tests fail on the pre-fix code and pass after:
- test_whole_attitude_reversal_by_salient_head  (Defect 1)
- test_scope_narrowing_still_protected          (no-regression guard)
- test_prior_stance_recorded_on_reversal        (supports Defect 2)
- test_prior_stance_recall_from_store            (Defect 2, end-to-end)
- test_prior_stance_recall_unstored_is_none     (no world-knowledge leakage)
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.user_model import UserModel
from ravana.chat.engine import CognitiveChatEngine


def _um_with_stance(text):
    um = UserModel()
    um.opinions.stances.clear()
    um.mine_personal_facts(text)
    return um


def test_whole_attitude_reversal_by_salient_head():
    """'i take it back about the cold' must recode the held cold-weather stance
    to the opposite pole (was silently dropped before the fix)."""
    um = _um_with_stance(
        "i really love cold weather, give me a snowy morning over a hot one")
    # the exact key is produced by the miner; resolve it generically
    key = um.opinions.resolve_topic("cold weather")
    assert key is not None, "cold-weather stance must be mined"
    held = um.opinions.stances.get(key)
    assert held is not None and held.polarity > 0.5
    before = held.polarity

    um.mine_personal_facts(
        "actually i take it back about the cold, i would rather have steady heat")
    after = um.opinions.stances.get(key)
    assert after is not None, "held stance must survive the reversal"
    assert after.polarity < 0.0, f"whole-attitude reversal not applied: {after.polarity}"
    assert after.polarity < before, "polarity must move opposite"


def test_scope_narrowing_still_protected():
    """A scope-narrowing recant ('i was wrong about acoustic-ONLY') must NOT
    flip the held attitude (the original corruption case). Regression guard."""
    um = _um_with_stance("i really love acoustic music, it's my favorite thing")
    key = um.opinions.resolve_topic("acoustic music")
    assert key is not None
    held = um.opinions.stances.get(key)
    assert held is not None and held.polarity > 0.5
    before = held.polarity

    um.mine_personal_facts(
        "i was wrong about acoustic-only, i like acoustic but not only acoustic")
    after = um.opinions.stances.get(key)
    # narrowing keeps the attitude; must NOT invert
    assert after is not None
    assert after.polarity > 0.3, f"scope-narrowing wrongly flipped: {after.polarity}"
    assert abs(after.polarity - before) < 0.01 or after.polarity >= before, \
        "scope-narrowing must not change the held polarity"


def test_prior_stance_recorded_on_reversal():
    """reverse_stance must persist the pre-recode opinion as prior_stance so a
    later 'what was my original take' recall can read the user's own history."""
    um = _um_with_stance(
        "i really love cold weather, give me a snowy morning over a hot one")
    key = um.opinions.resolve_topic("cold weather")
    assert key is not None
    um.mine_personal_facts(
        "actually i take it back about the cold, i would rather have steady heat")
    after = um.opinions.stances.get(key)
    assert after is not None
    assert after.prior_polarity is not None, "prior_polarity must be recorded"
    assert after.prior_polarity > 0.5, "prior must be the OLD for-value"
    assert after.prior_stance is not None and "for" in after.prior_stance, \
        f"prior_stance string malformed: {after.prior_stance}"


def test_prior_stance_recall_from_store():
    """End-to-end: a prior/original-stance query answers from the user's own
    stance store, not world-knowledge."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_prior_recall")
    eng.process_turn(
        "i really love cold weather, give me a snowy morning over a hot one")
    eng.process_turn(
        "actually i take it back about the cold, i would rather have steady heat")
    ans = eng._structured_recall("did i used to say i loved cold weather?")
    assert ans is not None, "prior-stance query must be answered from the store"
    assert "used to be" in ans.lower(), f"reply should report the prior: {ans}"
    assert "cold weather" in ans.lower(), f"reply should name the topic: {ans}"
    # Must NOT be the world-knowledge fallthrough phrasing
    assert "outside what i know" not in ans.lower()


def test_prior_stance_recall_unstored_is_none():
    """A prior-stance query about a topic with no held stance must return None
    (fail-closed), so it reaches honest uncertainty instead of confabulating
    from world-knowledge."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_prior_unstored")
    eng.process_turn("i really love cold weather, give me a snowy morning")
    eng.process_turn("actually i take it back about the cold, i prefer heat")
    ans = eng._structured_recall("did i used to love quantum entanglement?")
    assert ans is None, f"unstored topic must not be answered: {ans}"


def test_held_stance_no_reversal_reports_current():
    """A prior-frame query on a held stance that was NEVER reversed reports the
    honest current stance (not a fabricated prior)."""
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_prior_held")
    eng.process_turn("i think drone delivery is wonderful, it gets medicine to villages")
    ans = eng._structured_recall("what was my original take on drone delivery?")
    assert ans is not None
    assert "strongly for" in ans.lower() or "for" in ans.lower()
    assert "used to be" not in ans.lower(), \
        "no reversal recorded -> should NOT claim a prior"
