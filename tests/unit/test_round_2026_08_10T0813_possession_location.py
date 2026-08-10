"""
Regression tests for round 2026-08-10T0813Z, fix D: a named possession's
whereabouts ("the slow coal is moored at bingley") must be captured as a
structured, entity-keyed location fact (so a later correction supersedes it),
not left as a raw episodic echo.
"""
import os
import sys

os.environ["RAVANA_OFFLINE"] = "1"
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJ, os.path.join(_PROJ, "ravana", "src"), os.path.join(_PROJ, "ravana_ml", "src")):
    sys.path.insert(0, _p)

from ravana.chat.user_model import UserModel


def test_possession_location_captured_and_trimmed():
    um = UserModel()
    um.mine_personal_facts("the slow coal is moored at bingley for the winter")
    f = um.personal_facts.get("slow coal", "location")
    assert f is not None, dict(um.personal_facts.facts)
    assert f.value == "bingley", f.value


def test_possession_location_correction_supersedes():
    um = UserModel()
    um.mine_personal_facts("the slow coal is moored at bingley for the winter")
    um.mine_personal_facts("the slow coal is moored at saltaire now")
    facts = [(k, v.value, getattr(v, "superseded", False))
             for k, v in um.personal_facts.facts.items() if k[0] == "slow coal"]
    active = [v for k, v, sup in facts if not sup]
    assert ("saltaire", False) in [(v, sup) for k, v, sup in facts if k[1] == "location"], facts
    # bingley superseded
    assert ("bingley", True) in [(v, sup) for k, v, sup in facts if k[1] == "location"], facts
