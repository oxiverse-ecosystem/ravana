"""Regression test for the 'I flipped' stance-reversal gap (round t_6c023144).

The 2026-08-09T1953Z round worker logged a residual: a free-text reversal like
"i flipped, the reef tank is more work than joy" formed a fresh FOR stance on
'reef tank' rather than recoding the held one — because `flipped` was absent
from the retraction-cue set, so mine_stance_reversal never fired a reversal.

This asserts the NEW capability: a first-person change-of-mind verb reverses a
held stance (polarity flips opposite, last_reversal is recorded), while a flip
on a topic with NO held stance is a harmless no-op (no bogus stance created).
"""
import os
import sys
import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.user_model import UserModel


def _seed_stance(text):
    um = UserModel()
    um.opinions.stances.clear()
    um.mine_personal_facts(text)
    return um


def test_flipped_reverses_held_stance():
    """'i flipped, X...' must recode a held stance on X to the opposite pole."""
    um = _seed_stance("i love my reef tank, watching the corals is what i live for")
    before = um.opinions.stances.get("reef tank")
    assert before is not None and before.polarity > 0.5

    um.mine_personal_facts("i flipped, the reef tank is more work than joy")
    after = um.opinions.stances.get("reef tank")
    assert after is not None, "held stance must survive the reversal"
    assert after.polarity < 0.3, f"reversal not applied, polarity={after.polarity}"
    # linked-acknowledgment marker must be recorded
    assert um.opinions.last_reversal is not None
    assert um.opinions.last_reversal[0] == "reef tank"
    assert um.opinions.last_reversal[1] > 0   # old polarity was FOR
    assert um.opinions.last_reversal[2] < 0   # new polarity is AGAINST


def test_flipped_without_held_stance_is_noop():
    """A flip on a topic with no held stance must NOT create a bogus stance."""
    um = UserModel()
    um.opinions.stances.clear()
    um.mine_personal_facts("i flipped on the stock market, it's boring now")
    assert not any("stock market" in k for k in um.opinions.stances.keys()), \
        "flip with no held stance must not create a stance"


def test_softening_flip_relaxes_toward_neutral():
    """'i kind of flipped ... it's not that bad' relaxes, does not hard-invert."""
    um = _seed_stance("i love running, it clears my head")
    assert um.opinions.stances.get("running").polarity > 0.5
    um.mine_personal_facts("i kind of flipped on running, it's not that bad")
    after = um.opinions.stances.get("running")
    assert after is not None
    assert abs(after.polarity) < 0.5, f"softening flip should relax, got {after.polarity}"


def test_change_of_heart_cue_reverses():
    """A 'change of heart' idiom must also trigger reversal of a held stance."""
    um = _seed_stance("i love the new transit plan, it's overdue and brilliant")
    before = um.opinions.stances.get("new transit plan")
    assert before is not None and before.polarity > 0.3
    before_pol = before.polarity  # capture scalar; reverse_stance mutates in place
    um.mine_personal_facts("i've had a change of heart, the new transit plan is a mistake")
    after = um.opinions.stances.get("new transit plan")
    assert after is not None and after.polarity < before_pol
