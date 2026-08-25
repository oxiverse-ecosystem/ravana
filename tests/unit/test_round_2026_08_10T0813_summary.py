"""
Regression tests for round 2026-08-10T0813Z, fix B: the learned-profile
summary renderers must NOT dump 'event' lived-experiences or raw 'does'/'event'
fact noise into "what have you told me" / "what do you make of me". They must
read as a biography built from the live personal-fact store.
"""
import os
import sys

os.environ["RAVANA_OFFLINE"] = "1"
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJ, os.path.join(_PROJ, "ravana", "src"), os.path.join(_PROJ, "ravana_ml", "src")):
    sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine


def _seed(suffix):
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)
    for t in [
        "i'm caspar",
        "my dog is a lurcher named wren",
        "i build drystone walls in the dales",
        "i volunteer at a kestrel sanctuary",
        "i lost a kestrel last week",
        "i keep coming back to the wall",
        "i wild-swim in the lochs",
        "i repair old sextants",
    ]:
        eng.process_turn(t)
    return eng


def test_summary_skips_event_noise():
    eng = _seed("summ_b1")
    out = eng.process_turn("what have you told me about me?")
    assert "your event is" not in out, out
    # biographical facts present
    assert "Caspar" in out, out
    assert "dog is a lurcher named wren" in out, out


def test_make_of_me_skips_raw_fact_dump():
    eng = _seed("summ_b2")
    out = eng.process_turn("what do you make of me?")
    # no raw 'event:' / 'does:' attribute dumps
    assert "event:" not in out, out
    assert "does:" not in out, out
    # reads as a natural biographical sketch
    assert "caspar" in out.lower(), out
