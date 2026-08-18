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


def test_miner_stores_nonkin_role_with_irregular_verb():
    # Round 2026-08-17T1730Z: the named-relationship miner previously only
    # accepted blood-kin heads and activity verbs from a narrow inflection set,
    # so "my mentor Dr. Okonkwo taught me..." (mentor is not kin; "taught" is
    # an irregular verb) was DROPPED and a later "who is my mentor?" had nothing
    # to recall. The miner now accepts any relationship head (kin / non-kin
    # ROLE lexicon / runtime-learned via learn_relation) and recognizes
    # irregular activity verbs, so the fact is stored and recallable.
    caps = _capture("my mentor Dr. Okonkwo taught me astronomy when i was a teenager")
    assert ("i", "mentor dr. okonkwo") in caps, caps
    assert "taught astronomy" in caps[("i", "mentor dr. okonkwo")], caps


def test_open_ended_recall_nonkin_role():
    # The shared relationship vocabulary grows from the live disclosure
    # (learn_relation), so the open-ended recaller (engine.py 1c/1d) resolves
    # a NON-kin role ("my mentor") without a per-role branch. Content read
    # from the live store; no authored reply.
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_1730z_role_recall")
    eng.process_turn("my mentor Dr. Okonkwo taught me astronomy when i was a teenager")
    assert "your mentor dr. okonkwo taught astronomy" in \
        eng._structured_recall("who is my mentor?"), "non-kin role recall failed"
    assert "your mentor dr. okonkwo taught astronomy" in \
        eng._structured_recall("tell me about my mentor"), "open 'tell me about' role failed"


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


def test_open_ended_recall_includes_pet_path():
    # Branch (1d) `(c)` in _structured_recall also keys on a stored pet
    # attribute (not the relationship lexicon). A pet disclosed as "my <species>
    # is <name>" mines into the pet-name slot ('cat'/'dog'); an open phrasing
    # ("tell me about my cat") or a bare-name query ("who is my cat?") must
    # resolve it from the live store. Verified live during the docs pass for
    # feature t_1a4a3938. Declarative disclosure ("my cat is pixel") must NOT
    # be hijacked (interrogative-gate, same as relationship recall).
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_openrel_pet")
    eng.process_turn("my cat is pixel")
    eng.process_turn("my dog is biscuit")
    assert "your cat is pixel." in \
        eng._structured_recall("tell me about my cat"), "pet open-phrasing failed"
    assert "your cat is pixel." in \
        eng._structured_recall("who is my cat?"), "pet bare-name failed"
    # declarative disclosure is left to fact-mining, not echoed
    assert eng._structured_recall("my cat is pixel") is None, \
        "declarative pet disclosure must not be hijacked"


def test_garbage_does_facts_rejected():
    # Round 2026-08-17T1730Z: vague first-person disclosures were mined as
    # junk `does`/`event` facts (phrasal/aspectual verb residue, bare
    # timeframes, generic nouns) and later echoed in "what have you learned
    # about me" dumps. The shared _opinion_topic content-adequacy gate rejects
    # objects that resolve to ONLY non-content words, while real activities
    # (which always contain a content noun) survive.
    um = UserModel()
    um.personal_facts.facts.clear()
    for s in [
        "i read that sumo has this whole ritual side i never appreciated.",
        "i keep coming back to tidal energy, i really believe in it.",
        "i started keeping a vinyl record collection, mostly jazz.",
        "i got burned by an open source project last year.",
        "i went last night and my ears are ringing.",
        "i found out a project i cared about got cancelled.",
    ]:
        um.mine_personal_facts(s, run_correction=True)
    junk = {
        f.value for (a, b, c), f in um.personal_facts.facts.items()
        if b in ("does", "event") and not getattr(f, "superseded", False)
    }
    # the aspectual/particle/timeframe/generic residues must be gone
    for bad in ("keep coming back", "started keeping", "got burned",
                "went last night", "found project", "read sumo has"):
        assert bad not in {j.lower() for j in junk}, f"junk fact stored: {junk}"


def test_real_activity_still_captured_after_gate():
    # The content-adequacy gate must NOT swallow genuine activities that
    # contain a real content noun.
    um = UserModel()
    um.personal_facts.facts.clear()
    um.mine_personal_facts("i build bicycle frames by hand.", run_correction=True)
    caps = {
        f.value for (a, b, c), f in um.personal_facts.facts.items()
        if b == "does" and not getattr(f, "superseded", False)
    }
    assert any("build" in c and "frame" in c for c in caps), f"real activity lost: {caps}"
    um2 = UserModel()
    um2.personal_facts.facts.clear()
    um2.mine_personal_facts("i keep homing pigeons.", run_correction=True)
    caps2 = {
        f.value for (a, b, c), f in um2.personal_facts.facts.items()
        if b == "does" and not getattr(f, "superseded", False)
    }
    assert any("pigeon" in c for c in caps2), f"real activity lost: {caps2}"

