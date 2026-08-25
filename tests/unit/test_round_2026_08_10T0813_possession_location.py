"""
Regression tests for round 2026-08-10T0813Z, fix D: a named possession's
whereabouts ("the slow coal is moored at bingley") must be captured as a
structured, entity-keyed location fact (so a later correction supersedes it),
not left as a raw episodic echo.
"""
import os
import re
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


def test_possession_location_leading_hedge_not_new_entity():
    """A correction led by a discourse hedge ('actually the slow coal ...')
    must resolve to the SAME stored entity ('slow coal'), not a fragment
    ('ctually the slow coal'). Regression for a bug where the optional article
    regex glued onto the 'a' inside 'actually'."""
    um = UserModel()
    um.mine_personal_facts("the slow coal is moored at bingley for the winter")
    um.mine_personal_facts("actually the slow coal is moored at saltaire now")
    ents = {k[0] for k in um.personal_facts.facts if k[0] != "i"}
    assert "ctually the slow coal" not in ents, ents
    assert "slow coal" in ents, ents
    active_loc = [(v.value, getattr(v, "superseded", False))
                  for k, v in um.personal_facts.facts.items()
                  if k[0] == "slow coal" and k[1] == "location"]
    assert ("saltaire", False) in active_loc, active_loc
    assert ("bingley", True) in active_loc, active_loc


def _eng():
    from ravana.chat.engine import CognitiveChatEngine
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix="test_entloc_recall")


def test_entity_location_recall_surfaced_not_echo():
    """Limitation #1 (round 2026-08-10T0813Z): a stored entity-keyed location
    must be SURFACED on a 'where is X' query instead of falling through to the
    episodic echo of the raw utterance."""
    eng = _eng()
    eng.process_turn("the slow coal is moored at bingley")
    ans = eng.process_turn("where's the slow coal moored?")
    assert "bingley" in ans, ans
    assert "you told me earlier" not in ans, ans
    # alternate phrasings all resolve to the structured fact
    for q in ("where is the slow coal?", "where's the slow coal?",
              "what is the slow coal's location?"):
        a = eng.process_turn(q)
        assert "bingley" in a, (q, a)
        assert "you told me earlier" not in a, (q, a)


def test_entity_location_recall_multiple_entities():
    eng = _eng()
    eng.process_turn("the slow coal is moored at bingley")
    eng.process_turn("the van is parked in leeds")
    assert "bingley" in eng.process_turn("where's the slow coal moored?")
    assert "leeds" in eng.process_turn("where is the van?")


def test_entity_location_recall_correction_composes():
    """Stored fact correction (supersede) must be reflected in recall."""
    eng = _eng()
    eng.process_turn("the slow coal is moored at bingley")
    eng.process_turn("actually the slow coal is moored at saltaire now")
    ans = eng.process_turn("where's the slow coal moored?")
    assert "saltaire" in ans, ans
    assert "bingley" not in ans, ans


def test_entity_location_recall_unknown_fails_closed():
    """No stored entity location -> honest uncertainty, not a confabulated
    place. ('where is paris' has no entity fact, so the engine must not
    fabricate a bingley-style 'the paris is at X' answer.)"""
    eng = _eng()
    eng.process_turn("the slow coal is moored at bingley")
    ans = eng.process_turn("where is paris?")
    # Fail-closed property (deterministic, wording-independent): the reply must
    # NOT assert a place for paris, and must NOT leak the unrelated stored
    # 'bingley' fact. The engine's exact uncertainty phrasing varies run-to-run
    # ("i don't have a clean definition for paris...", "paris are fuzzy for me
    # ...", "honestly, paris is a bit outside what i know..."), so we assert the
    # NEGATIVE property — no fabricated / leaked location — rather than matching
    # any one phrasing (which would make this test flaky / one-realization-tuned).
    # Verified across 3 independent seeds: none of the three observed replies
    # assert a paris location or mention bingley.
    assert "the paris is at" not in ans.lower(), ans
    assert "bingley" not in ans.lower(), ans
    assert not re.search(
        r"\bparis (?:is|are) (?:in|at|based|located|moored|parked)\b",
        ans.lower()), ans


def test_entity_location_recall_user_location_not_hijacked():
    """The user's own location ('i live in X') is answered by the biographical
    'where do i live' path, not swallowed as an entity whereabouts."""
    eng = _eng()
    eng.process_turn("the slow coal is moored at bingley")
    eng.process_turn("i live in hexham")
    ans = eng.process_turn("where do i live?")
    assert "hexham" in ans, ans
    assert "the slow coal" not in ans, ans

