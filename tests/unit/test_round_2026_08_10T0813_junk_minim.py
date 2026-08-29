"""
Regression tests for round 2026-08-10T0813Z, fix C: the activity/event miner
must NOT store junk facts from meta-reflection / embedded-question / self-error
clauses ("i lose track of whether i told you", "i mixed them up", "i got
muddled"). Real disclosures ("i build drystone walls") must still be captured.
"""
import os
import sys

os.environ["RAVANA_OFFLINE"] = "1"
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJ, os.path.join(_PROJ, "ravana", "src"), os.path.join(_PROJ, "ravana_ml", "src")):
    sys.path.insert(0, _p)

from ravana.chat.user_model import UserModel


def _mine_and_facts(cases):
    um = UserModel()
    for c in cases:
        um.mine_personal_facts(c)
    out = {}
    for k, v in um.personal_facts.facts.items():
        if k[0] == "i" and (k[1].startswith("does") or k[1].startswith("event")) and not getattr(v, "superseded", False):
            _bucket = "does" if k[1].startswith("does") else "event"
            out.setdefault(_bucket, []).append(v.value)
    return out


def test_meta_reflection_not_stored():
    facts = _mine_and_facts([
        "i lose track of whether i told you",
        "i mixed them up",
        "i got muddled",
        "no, actually wren is my partner's name, the dog is a lurcher called briar - i mixed them up",
    ])
    flat = [v for vals in facts.values() for v in vals]
    assert "lose track" not in flat, flat
    assert "mix" not in flat, flat
    assert "muddled" not in flat, flat
    assert "got muddled" not in flat, flat


def test_real_disclosure_still_captured():
    facts = _mine_and_facts([
        "i build drystone walls in the dales",
        "i keep homing pigeons",
        "i grow air plants on the windowsill",
    ])
    assert "build drystone walls" in facts.get("does", []), facts
    assert "keep homing pigeons" in facts.get("does", []), facts
    assert "grow air plants" in facts.get("does", []), facts


def test_event_disclosure_still_captured():
    facts = _mine_and_facts([
        "i lost a kestrel last week, a female we'd had three months",
        "i dropped the jar on the floor",
    ])
    assert "lost kestrel last week" in facts.get("event", []), facts
    assert "drop jar" in facts.get("event", []), facts
