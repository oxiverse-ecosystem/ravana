"""Regression tests for title/abbreviation value handling (FIX-RV-feature 2026-10-22).

Defect: equational miner truncated titled names like "dr. rao" to "dr" because
the clause trim split on every ". ". Fix: abbreviation-aware sentence split +
relative-clause trim, with runtime growth of the abbreviation seed set.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.user_model import UserModel, is_abbreviation, learn_abbreviation, _first_value_clause


def _capture(text):
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts(text, run_correction=True)
    return {
        (a, b): f.value
        for (a, b, c), f in um.personal_facts.facts.items()
        if not getattr(f, "superseded", False)
    }


def test_titled_name_not_truncated():
    caps = _capture("my thesis advisor is dr. rao, who studies marine ecosystems.")
    assert ("i", "thesis advisor") in caps, caps
    assert caps[("i", "thesis advisor")] == "dr. rao", caps


def test_titled_name_recall_value():
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts("my mentor is mr. venkatesh, who taught me biology.", run_correction=True)
    caps = {
        (a, b): f.value
        for (a, b, c), f in um.personal_facts.facts.items()
        if not getattr(f, "superseded", False)
    }
    assert any(v == "mr. venkatesh" for v in caps.values()), caps


def test_abbreviation_helper():
    assert is_abbreviation("dr")
    assert is_abbreviation("Dr")
    assert not is_abbreviation("rao")
    assert learn_abbreviation("xyz")
    assert is_abbreviation("xyz")


def test_first_value_clause_relative_clause():
    assert _first_value_clause("dr. rao, who studies marine ecosystems.") == "dr. rao"
    assert _first_value_clause("the blue hour just before the sun. i also keep pigeons") == "the blue hour just before the sun"


def test_first_value_clause_abbreviation_period():
    assert _first_value_clause("dr. rao is great. next sentence.") == "dr. rao is great"


def test_engine_reply_titled_name():
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_title_reply2")
    r = eng.process_turn("my thesis advisor is dr. rao, who studies marine ecosystems.")
    assert "dr. rao" in (r or "").lower(), r
    eng.stop_background_learning()
    eng.save()
