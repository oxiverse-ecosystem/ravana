"""Regression tests — round 2026-08-14T0608Z.

Unit 1: temporal / date-grounded fact mining + date recall.
Verifies first-person disclosures anchored to a point in time land in the
personal-fact store (attr 'since' / 'since_age') and are recalled correctly
from a LATER query. No hardcoded reply; every answer slot is read from the
PersonalFactStore.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.user_model import UserModel


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


def test_explicit_year_anchor():
    facts = _captured_since("i've been building frames since 2019")
    assert any(b == "since" and "build" in c and "2019" in c
               for (a, b, c) in facts), f"expected since(build 2019), got {facts}"


def test_in_year_anchor():
    facts = _captured_since("i started keeping quail in 2021 after i lost my garden")
    assert any(b == "since" and "keep" in c and "2021" in c
               for (a, b, c) in facts), f"expected since(keep 2021), got {facts}"


def test_relative_duration_resolves_to_year():
    # 'for eleven years' -> current_year - 11
    facts = _captured_since("i've repaired tube amps for eleven years, since before i built frames")
    yr = [c.split()[-1] for (a, b, c) in facts if b == "since" and "repair" in c]
    assert yr, f"expected since(repair <year>), got {facts}"
    assert yr[0].isdigit() and 1900 < int(yr[0]) <= 2026, f"year not normalized: {yr}"


def test_age_anchor_captured():
    facts = _captured_since("i picked up the cello when i was nine, so about twenty years now")
    assert any(b == "since_age" and "pick up" in c for (a, b, c) in facts), \
        f"expected since_age(pick up 9), got {facts}"


def test_nondate_disclosure_does_not_create_since_fact():
    facts = _captured_since("i build bicycle frames by hand in a small workshop")
    assert not facts, f"unexpected since fact from non-date disclosure: {facts}"


def test_date_recall_resolver():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="temporal_test_14T0608")
    eng.process_turn("i've been building frames since 2019")
    eng.process_turn("i started keeping quail in 2021")
    ans = eng._structured_recall("when did i start building frames")
    assert ans is not None and "2019" in ans, f"date recall failed: {ans!r}"
    ans2 = eng._structured_recall("since what year have i kept quail")
    assert ans2 is not None and "2021" in ans2, f"date recall failed: {ans2!r}"
    try:
        import os as _os
        _p = _os.path.join("weights", "ravana_weightstemporal_test_14T0608.pkl")
        if _os.path.exists(_p):
            _os.remove(_p)
    except Exception:
        pass
