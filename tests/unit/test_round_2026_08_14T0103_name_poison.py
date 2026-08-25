"""Regression tests for round 2026-08-14T0103Z — Defect A: name poisoning
by descriptor nouns.

A bare first-person copula self-description ("i'm vegetarian", "i'm a
ceramicist", "i'm an atheist", "i'm scottish") must NOT be stored as the
user's NAME. A later "what's my name" must return the real name, never the
descriptor. Real proper-noun names ("i'm mira") must still be captured.

No hardcoded reply; this tests the miner's state, not a generated string.
"""

import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _um():
    sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
    from ravana.chat.user_model import UserModel
    return UserModel()


def test_descriptor_nouns_not_stored_as_name():
    # Each of these self-descriptions must leave user_name empty/None.
    for t in [
        "i'm vegetarian, mostly.",
        "i'm vegan.",
        "i'm a ceramicist.",
        "i'm an atheist.",
        "i'm scottish.",
        "i'm a teacher.",
        "i'm a proud father.",
        "i'm an environmentalist.",
    ]:
        um = _um()
        um.mine_personal_facts(t)
        assert not um.user_name or um.user_name.strip() == "", \
            f"{t!r} poisoned name -> {um.user_name!r}"


def test_real_names_still_captured():
    for t, exp in [("i'm mira.", "Mira"), ("i'm wren.", "Wren"),
                   ("i'm tobias.", "Tobias")]:
        um = _um()
        um.mine_personal_facts(t)
        assert um.user_name == exp, f"{t!r} -> name {um.user_name!r}, expected {exp!r}"


def test_name_recall_not_descriptor():
    # A real name followed by a descriptor must recall the name, not the descriptor.
    um = _um()
    um.mine_personal_facts("i'm mira.")
    um.mine_personal_facts("i'm vegetarian, mostly.")
    assert um.user_name == "Mira", f"name should stay Mira, got {um.user_name!r}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
