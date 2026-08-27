"""
Regression test for round 2026-08-13T2059Z location-capture fix.

The round added a `_m_loc_based` regex to capture location from
"based in X" / "located in X" / "stationed at X" / "situated on X" forms.
The original gap group only allowed an OPTIONAL determiner + one word between
the subject and the verb, so common interrupted phrasings missed:
  - "i'm a lighthouse keeper based in skye."  (round's own documented example)
  - "she is stationed at diego garcia."
This locks the widened match so the capture survives a multi-word noun phrase
between subject and verb. State-driven (regex seed, stored content head), not
a hardcoded reply string.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.user_model import UserModel


def _capture(text):
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts(text, run_correction=True)
    loc = {c for (a, b, c), f in um.personal_facts.facts.items()
           if a == "i" and b == "location" and not getattr(f, "superseded", False)}
    return um.user_location, loc


def test_interrupted_based_in_captured():
    # round's own example: multi-word NP ("a lighthouse keeper") between i'm and based
    loc, facts = _capture("i'm a lighthouse keeper based in skye.")
    assert loc == "skye", loc
    assert "skye" in facts, facts


def test_stationed_at_multiword_captured():
    loc, facts = _capture("she is stationed at diego garcia.")
    assert loc == "diego garcia", loc
    assert "diego garcia" in facts, facts


def test_simple_based_in_regression():
    loc, facts = _capture("i'm based in skye.")
    assert loc == "skye", loc
    assert "skye" in facts, facts


def test_natural_feature_form_regression():
    loc, facts = _capture("on the isle of skye.")
    assert loc == "skye", loc
    assert "skye" in facts, facts


def test_non_location_not_stored():
    loc, facts = _capture("i'm gutted about the cut.")
    assert loc == ""
    assert not facts
