"""
Regression tests for round 2026-08-10T0813Z, fix A: a pet name re-disclosed in
a SECOND phrasing ("the dog is a lurcher called briar" after "my dog is a
lurcher named wren") must CORRECT the earlier name, not leave the stale one
active. Through the real engine, the recall must return the corrected name.
"""
import os
import sys

os.environ["RAVANA_OFFLINE"] = "1"
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJ, os.path.join(_PROJ, "ravana", "src"), os.path.join(_PROJ, "ravana_ml", "src")):
    sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine


def _fresh_engine(suffix):
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)


def test_pet_redisclosure_corrects_stale_name():
    eng = _fresh_engine("petfix_a")
    eng.process_turn("my dog is a lurcher named wren, she is half-greyhound")
    eng.process_turn(
        "no, actually wren is my partner's name, the dog is a lurcher "
        "called briar - i mixed them up")
    facts = eng.user_model.personal_facts.facts
    assert facts[("i", "dog", "a lurcher named wren")].superseded is True
    assert facts[("i", "dog", "briar")].superseded is False
    # recall must surface the corrected name
    recall = eng.process_turn("what is my dog's name?")
    assert "briar" in recall, recall


def test_pet_redisclosure_does_not_fire_on_first_disclosure():
    eng = _fresh_engine("petfix_b")
    eng.process_turn("my cat is a tabby called milo")
    # a second, DIFFERENT first disclosure of a new species must just store
    eng.process_turn("my dog is a spaniel called rex")
    facts = eng.user_model.personal_facts.facts
    assert ("i", "cat", "a tabby called milo") in facts
    assert ("i", "dog", "a spaniel called rex") in facts
