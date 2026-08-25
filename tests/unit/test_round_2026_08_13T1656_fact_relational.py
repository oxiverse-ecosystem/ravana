"""
Regression tests for round 2026-08-13T1656Z.

Four defects fixed this round, all root-caused from a clean 42-turn in-process
chat and verified live:

  A. NAME POISONING regression — a bare-copula transient state
     ("i'm gutted") was stored as the user's NAME because the affect lexicon
     lacked the word and the guard only checked the HEAD token. Now rejected
     (any-token affect/verb-form check + broadened seed lexicon).

  B. LOCATION not captured from "i'm a sound engineer based in berlin" — the
     location miner only matched live/in/from, so a later "what city am i in"
     echoed the (wrong) name. Now 'based in X' is captured.

  C. GARBAGE `does`/`event` facts — achievement/communication verbs (got/said/
     made/gave/told/...) in the activity/event verb lists stored outcome
     utterances as ('i','does','got artist residency') / ('i','does','said open').
     Those verbs are excluded; only sustained-activity + physical-world-experience
     verbs remain.

  D. RELATIONAL reverse-lookup gap (the 6f generalization) — "my friend wren
     collects vinyl" was never stored as a structured relationship, so "who is
     wren to me" could only echo an unrelated episode. Now the relationship is
     stored under the entity and a type-agnostic reverse resolver answers it
     from the store.

None of these add authored reply prose; every fix is a structural guard or a
seed-vocabulary change, and the answer content comes from durable store state.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.user_model import UserModel
from ravana.chat.engine import CognitiveChatEngine


def _capture(text):
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts(text, run_correction=True)
    return {
        (a, b): f.value
        for (a, b, c), f in um.personal_facts.facts.items()
        if not getattr(f, "superseded", False)
    }


def _entity_facts(text):
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts(text, run_correction=True)
    return {
        (a, b, c): f.value
        for (a, b, c), f in um.personal_facts.facts.items()
        if not getattr(f, "superseded", False)
    }


# ── A. name poisoning ──────────────────────────────────────────────
def test_transient_state_not_stored_as_name():
    caps = _capture("i'm gutted, i failed the exam by two points.")
    assert ("i", "name") not in caps, caps   # gutted must NOT be a name


def test_emotion_word_not_stored_as_name():
    for emo in ("i'm elated about the news.", "i'm devastated by the loss.",
                "i'm over the moon, i got the job."):
        caps = _capture(emo)
        assert ("i", "name") not in caps, (emo, caps)


def test_real_name_still_captured():
    caps = _capture("i'm rin, and i'm a sound engineer based in berlin.")
    assert ("i", "name") in caps, caps
    assert caps[("i", "name")] == "rin", caps


# ── B. location from 'based in' ─────────────────────────────────────
def test_location_captured_from_based_in():
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts("i'm rin, and i'm a sound engineer based in berlin.")
    assert um.user_location == "berlin", um.user_location
    assert ("i", "location") in {
        (a, b): f.value for (a, b, c), f in um.personal_facts.facts.items()
    }


def test_location_live_in_still_works():
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts("i live in a lighthouse on a rock.")
    assert um.user_location, um.user_location


# ── C. no garbage does/event facts from achievement/communication verbs ──
def test_got_outcome_not_stored_as_does():
    caps = _capture("i'm over the moon, i just got the artist residency in lisbon!")
    assert ("i", "does") not in caps, caps
    assert ("i", "event") not in caps, caps


def test_said_outcome_not_stored_as_does():
    caps = _capture("i said open-plan offices help me think.")
    assert ("i", "does") not in caps, caps


def test_genuine_activity_still_captured():
    caps = _capture("i keep three pet raccoons that steal my tomatoes.")
    assert ("i", "does") in caps, caps
    assert "raccoons" in caps[("i", "does")], caps


def test_genuine_event_still_captured():
    caps = _capture("i dropped the vase and it shattered.")
    assert ("i", "event") in caps, caps


# ── D. relational disclosure + reverse lookup ──────────────────────
def test_friend_relationship_stored():
    ef = _entity_facts("my friend wren collects vinyl records and restores turntables.")
    assert ("wren", "relationship", "friend") in ef, ef
    assert any(k[0] == "wren" and k[1] == "does" for k in ef), ef


def test_sibling_relationship_stored():
    ef = _entity_facts("my sister meera is a marine biologist who studies squid.")
    assert ("meera", "relationship", "sister") in ef, ef
    assert any(k[0] == "meera" and k[1] == "role" for k in ef), ef


def test_brother_role_stored():
    ef = _entity_facts("my brother dev works as a paramedic in leeds.")
    assert ("dev", "relationship", "brother") in ef, ef
    assert any(k[0] == "dev" and k[1] == "role" for k in ef), ef


def test_reverse_lookup_who_is_x_to_me():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_rel_0813")
    eng.mine_and_ingest = None  # not used; drive via process_turn
    # ingest relation via the engine's real path
    eng.process_turn("my friend wren collects vinyl records.")
    ans = eng._structured_recall("who is wren to me?")
    assert ans is not None, "reverse lookup must resolve from store"
    assert "friend" in ans, ans
    assert "wren" in ans, ans


def test_reverse_lookup_sister():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_rel_0813b")
    eng.process_turn("my sister meera is a marine biologist who studies squid.")
    ans = eng._structured_recall("who is meera to me?")
    assert ans is not None, "reverse lookup must resolve from store"
    assert "sister" in ans, ans
