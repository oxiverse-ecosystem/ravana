"""Regression tests — round 2026-08-14T0608Z.

Unit 2: name-poisoning regression. A bare self-description ("i'm X") where X
is an affect/state word must NOT be stored as the user's NAME. Real names
("soren") must still be captured. No per-name list — the guard draws on the
shared affect/state seed lexicon.

Verified against the chat probe which poisoned name with quiet / obsessed /
gutted (the lexicon was too narrow on this baseline).
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.user_model import UserModel


def _names(text):
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts(text.lower(), run_correction=True)
    return {c for (a, b, c), f in um.personal_facts.facts.items()
            if b == "name" and not getattr(f, "superseded", False)}


def test_state_word_not_stored_as_name():
    for word in ("quiet", "gutted", "obsessed", "devastated", "crushed"):
        names = _names(f"i'm {word}")
        assert names == set(), f"'i'm {word}' should not poison name, got {names}"


def test_real_name_still_captured():
    names = _names("hi, i'm soren")
    assert "soren" in names, f"real name 'soren' not captured, got {names}"


def test_compound_self_description_not_name():
    # "i'm quiet but stubborn" -> 2 tokens, head 'quiet' is affect -> rejected
    names = _names("i'm quiet but stubborn")
    assert names == set(), f"compound state leaked as name: {names}"


def test_name_question_not_poisoned():
    # "who am i" never stores a name
    names = _names("who am i, do you know anything about me yet")
    assert names == set(), f"identity query poisoned name: {names}"
