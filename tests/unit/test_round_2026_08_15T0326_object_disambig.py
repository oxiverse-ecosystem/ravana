"""Feature test — round 2026-08-15T0326 (object-disambiguated date recall).

The date miner previously stored only the activity VERB HEAD ("build 2019"),
dropping the object, so two activities that share a verb but differ by object
("building frames" vs "building cabinets") tied at recall and the resolver
returned whichever fact it iterated first (the logged limitation #2: "date
resolver tie-break can show wrong activity on overlapping verbs"). This test
locks in the fix: the object is mined and recall disambiguates by it. It also
locks in the display fix for the double-gerund glitch ("buildinging frames").

No hardcoded reply; every answer slot is read from the PersonalFactStore and
realized by morphology. Generalizes to ANY overlapping-verb pair.
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
    um.mine_personal_facts(text.lower(), run_correction=True)
    return {
        (a, b, c): f.value
        for (a, b, c), f in um.personal_facts.facts.items()
        if b in ("since", "since_age") and not getattr(f, "superseded", False)
    }


def _cleanup(suffix):
    try:
        os.remove(os.path.join("weights", f"ravana_weights{suffix}.pkl"))
    except Exception:
        pass


def test_object_mined_into_since_fact():
    facts = _captured_since("i've been building frames since 2019")
    assert any(b == "since" and "frames" in c and "2019" in c
               for (a, b, c) in facts), f"expected since(build frames 2019), got {facts}"


def test_object_mined_with_determiner_stripped():
    # "the cabinets" -> object "cabinets", not empty, not "the"
    facts = _captured_since("i started building the cabinets in 2021")
    assert any(b == "since" and "cabinets" in c and "2021" in c
               for (a, b, c) in facts), f"expected since(build cabinets 2021), got {facts}"


def test_overlapping_verb_disambiguated_by_object():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="objdisam_15T0326")
    try:
        eng.process_turn("i've been building frames since 2019")
        eng.process_turn("i started building cabinets in 2021")
        ans_frames = eng._structured_recall("when did i start building frames")
        ans_cab = eng._structured_recall("since what year have i been building cabinets")
        assert ans_frames is not None and "2019" in ans_frames, f"frames query: {ans_frames!r}"
        assert ans_cab is not None and "2021" in ans_cab, f"cabinets query: {ans_cab!r}"
        assert "frames" in ans_frames, f"frames query returned wrong object: {ans_frames!r}"
        assert "cabinets" in ans_cab, f"cabinets query returned wrong object: {ans_cab!r}"
        # The bug was returning the SAME activity for both; prove they differ.
        assert ans_frames != ans_cab, f"overlapping verbs tied: {ans_frames!r}"
    finally:
        _cleanup("objdisam_15T0326")


def test_no_double_gerund_display():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="nogerund_15T0326")
    try:
        eng.process_turn("i've been building frames since 2019")
        ans = eng._structured_recall("when did i start building frames")
        assert ans is not None, "no answer"
        assert "buildinging" not in ans, f"double-gerund glitch: {ans!r}"
        assert "building frames" in ans, f"expected natural gerund: {ans!r}"
    finally:
        _cleanup("nogerund_15T0326")


def test_verb_only_disclosure_unaffected():
    # Bare activity (no object) still stores the old shape and recalls.
    facts = _captured_since("i've been restoring since 2018")
    assert any(b == "since" and c.strip().endswith("2018") and " " in c
               for (a, b, c) in facts), f"expected bare since(restore 2018), got {facts}"
