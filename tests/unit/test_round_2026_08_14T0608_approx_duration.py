"""Regression tests — round 2026-08-14T0608Z (feature extension).

Unit: APPROXIMATE / HUMAN-PHRASED duration mining.

The prior round captured only DIGIT / spelled-1-12 durations ("for eleven
years"). Real speech says "for a decade" / "a few years" / "two decades" /
"several years" — these now land in the 'since' store (block (d) in
mine_personal_facts) and are recalled by the SAME date resolver as explicit
durations. Every answer slot is read from the PersonalFactStore; no hardcoded
reply, no per-phrase table. Seed map (_FUZZY_DUR) is RAVANA-expandable.
"""
import os
import sys
import datetime as _dt

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.user_model import UserModel

_THIS_YEAR = _dt.datetime.now().year


def _captured_since(text):
    um = UserModel()
    um.personal_facts.facts.clear()
    # the engine lowercases user input before mining; mirror that here
    um.mine_personal_facts(text.lower(), run_correction=True)
    return {
        (a, b, c): f.value
        for (a, b, c), f in um.personal_facts.facts.items()
        if b in ("since", "since_age") and not getattr(f, "superseded", False)
    }


def test_decade_mined_as_year():
    facts = _captured_since("i've been brewing beer for a decade, since before the kids")
    yr = [c.split()[-1] for (a, b, c) in facts if b == "since" and "brew" in c]
    assert yr, f"expected since(brew <year>), got {facts}"
    assert int(yr[0]) == _THIS_YEAR - 10, f"decade not resolved to year-10: {yr}"


def test_couple_of_years_mined():
    facts = _captured_since("i've kept a garden for a couple of years now")
    # 'garden' is in the activity verb vocabulary too, so the resolved head is
    # 'garden' (same quirk as the explicit-duration miner) -> garden <year>.
    yr = [c.split()[-1] for (a, b, c) in facts if b == "since" and "garden" in c]
    assert yr, f"expected since(garden <year>), got {facts}"
    assert int(yr[0]) == _THIS_YEAR - 2, f"couple-of-years not resolved: {yr}"


def test_several_years_mined():
    facts = _captured_since("i've been reading sci-fi for several years")
    yr = [c.split()[-1] for (a, b, c) in facts if b == "since" and "read" in c]
    assert yr, f"expected since(read <year>), got {facts}"
    assert int(yr[0]) == _THIS_YEAR - 4, f"several-years not resolved: {yr}"


def test_two_decades_mined():
    facts = _captured_since("i've been teaching music for two decades")
    yr = [c.split()[-1] for (a, b, c) in facts if b == "since" and "teach" in c]
    assert yr, f"expected since(teach <year>), got {facts}"
    assert int(yr[0]) == _THIS_YEAR - 20, f"two-decades not resolved: {yr}"


def test_approx_duration_without_activity_is_not_captured():
    # 'a decade' with NO activity verb before it must NOT create a since fact
    facts = _captured_since("for a decade i've wondered whether to start")
    assert not facts, f"unexpected since fact from duration w/o activity: {facts}"


def test_approx_duration_recall_resolver():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="approx_dur_test_14T0608")
    eng.process_turn("i've been brewing beer for a decade")
    ans = eng._structured_recall("since what year have i brewed beer")
    assert ans is not None and str(_THIS_YEAR - 10) in ans, \
        f"approx-duration date recall failed: {ans!r}"
    ans2 = eng._structured_recall("when did i start brewing beer")
    assert ans2 is not None and str(_THIS_YEAR - 10) in ans2, \
        f"approx-duration date recall failed: {ans2!r}"
    try:
        import os as _os
        _p = _os.path.join("weights", "ravana_weightsapprox_dur_test_14T0608.pkl")
        if _os.path.exists(_p):
            _os.remove(_p)
    except Exception:
        pass
