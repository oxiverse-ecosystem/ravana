"""
Regression tests for round 2026-08-17T1126Z fixes.

Three defects, all verified live during the round:

1. Garbage `event: find fascinating` — the EVENT miner's "find" verb matched the
   cognitive-affective copula "i find it fascinating" and stored a sentiment
   adjective as a discovery event. Fix: affective-object guard (_AFFECTIVE_OBJECT_ADJ)
   on the event miner's resolved object. A genuine discovery ("i found my keys")
   must still store.

2. Self-profile dump copula bug — the "what do you remember about me" summary
   rendered verb-phrase values with a spurious copula
   ("your brother theo is fixes bicycles"). Fix: drop the copula for verb-phrase
   values in BOTH self-profile renderers (engine_memory.py), mirroring the D7
   cued-recall rule. Content comes from the store; no authored prose.

3. D-B reverse-name resolver runtime crash — the uncommitted D-B generalization
   called a @staticmethod as a bare name (`_strip_entity_from_does`) which raised
   NameError on the pet "does" path. Fix: call it as `self._strip_entity_from_does`.
   Verified `who is Mango?` -> `your pet parrot.` (type-agnostic, no per-entity branch).

All assertions are STATE-DRIVEN: they read the personal-fact store / recall output,
never authored reply strings.
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


def test_affective_copula_not_stored_as_event():
    # "i find it fascinating" is a cognitive-affective copula, not a discovery.
    caps = _capture("i used to think roman history was boring, but now i find it fascinating")
    assert ("i", "event") not in caps, f"garbage event fact stored: {caps}"
    # the discovery sense still works
    caps2 = _capture("i found my lost keys under the couch")
    assert ("i", "event") in caps2, caps2
    assert "found lost keys" in caps2[("i", "event")], caps2


def test_self_profile_dump_drops_copula_for_verb_phrase():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_1126z_copula")
    for s in [
        "my brother Theo fixes bicycles for the neighborhood kids",
        "my grandmother bakes sourdough bread every sunday",
    ]:
        eng.process_turn(s)
    reply = eng.process_turn("what do you remember about me?")
    assert "is fixes bicycles" not in reply, reply
    assert "is bakes sourdough bread" not in reply, reply
    assert "your brother theo fixes bicycles" in reply, reply
    assert "your grandmother bakes sourdough bread" in reply, reply


def test_reverse_name_resolver_pet_does_path():
    # D-B type-agnostic reverse-name: a pet stored under attr='does' with the
    # name buried in the value must resolve via the shared resolver, not crash.
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_1126z_dbp")
    eng.process_turn("i keep a pet parrot named Mango who mimics the microwave beep")
    r = eng.process_turn("who is Mango?")
    assert r is not None, "reverse-name resolver returned None / crashed"
    assert "pet parrot" in r, r


def test_miner_stores_named_relationship_lowercase_name():
    # D7 miner regression (feature t_1a4a3938): the old regex required the
    # disclosed Name to be CAPITALIZED, so a lowercase chat name ("indira") never
    # matched -> the fact was never stored -> open-ended recall had nothing to
    # recall. The token-based fix finds the activity verb by membership, so the
    # named-relationship fact is stored regardless of name casing.
    caps = _capture("my grandmother indira bakes sourdough bread every sunday")
    assert ("i", "grandmother indira") in caps, caps
    assert "bakes sourdough bread" in caps[("i", "grandmother indira")], caps


def test_open_ended_relationship_recall():
    # New capability (feature t_1a4a3938): RAVANA recalls what it knows about a
    # named relationship/person from OPEN-ENDED phrasings, not just a bare "who
    # is X". Every answer slot is read from the live PersonalFactStore; no
    # authored reply string, no per-person table.
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_openrel_recall")
    for s in [
        "my grandmother indira bakes sourdough bread every sunday",
        "my brother theo fixes bicycles for the neighborhood kids",
    ]:
        eng.process_turn(s)
    # relationship-word keyed, open phrasing
    assert "your grandmother indira bakes sourdough bread" in \
        eng._structured_recall("tell me about my grandmother"), "open 'tell me about' failed"
    # interrogative relationship key
    assert "your grandmother indira bakes sourdough bread" in \
        eng._structured_recall("who is my grandmother?"), "interrogative rel failed"
    # activity-asking phrasing
    assert "your grandmother indira bakes sourdough bread" in \
        eng._structured_recall("what does my grandmother do?"), "activity phrasing failed"
    # bare name (no relationship word in query)
    assert "your brother theo fixes bicycles" in \
        eng._structured_recall("who is theo?"), "bare-name recall failed"
    # unknown relative fails closed (honest None, not a fabricated bio)
    assert eng._structured_recall("tell me about my uncle fred") is None, \
        "unknown relative must fail closed"

