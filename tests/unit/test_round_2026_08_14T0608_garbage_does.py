"""Regression tests — round 2026-08-14T0608Z.

Unit 3: garbage 'does'/'event' facts from emotion / achieve-comm / framer
verbs. The open-class miner treated ANY word after "i" as the verb, so
"i felt crushed" -> does='felt crushed', "i said X" -> does='said ...',
"i take back what i said" -> does='take back', "i keep now" -> does='keep now'.
These are NOT activities RAVANA learned. Verified against the chat probe which
produced exactly these garbage facts on this baseline.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.user_model import UserModel, _activity_verb_ok


def _does_event(text):
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts(text.lower(), run_correction=True)
    return {
        c for (a, b, c), f in um.personal_facts.facts.items()
        if b in ("does", "event") and not getattr(f, "superseded", False)
    }


def test_emotion_verb_not_activity():
    facts = _does_event("i felt crushed when my torch failed")
    assert not any("felt" in c for c in facts), f"'felt' leaked as activity: {facts}"


def test_say_verb_not_activity():
    facts = _does_event("i said careless ones but meant well")
    assert not any("said" in c for c in facts), f"'said' leaked as activity: {facts}"


def test_take_back_retraction_not_activity():
    facts = _does_event("i take back what i said about drivers")
    assert not any("take back" in c for c in facts), f"'take back' leaked: {facts}"


def test_framer_now_not_in_object():
    facts = _does_event("how many quail do i keep now")
    # 'keep now' must not be stored; a real 'keep' fact (if any) keeps its object
    assert not any(c.strip().endswith("now") for c in facts), f"'now' leaked: {facts}"


def test_real_activity_still_captured():
    facts = _does_event("i build bicycle frames by hand")
    assert any("build" in c and "frame" in c for c in facts), \
        f"real activity 'build frames' lost: {facts}"


def test_verb_ok_helper():
    assert _activity_verb_ok("build") is True
    assert _activity_verb_ok("felt") is False
    assert _activity_verb_ok("said") is False
    # legitimate activity verbs are NOT denied (denying them breaks the
    # correction detector and loses real activities like "took up the cello")
    assert _activity_verb_ok("take") is True
    assert _activity_verb_ok("keep") is True
    assert _activity_verb_ok("won't") is False  # contraction artifact
