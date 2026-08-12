"""
Feature regression test for round 2026-08-12T0613Z limitation T40:

A pet disclosed as a NAME ("my dog's a retriever called wren", "my cat is
ember") was only recallable via the SPECIES noun ("what is my dog's name").
A query that names the pet by its NAME ("who is wren to me?") had no
retrieval path and fell through to a generic self-blurb. This capability is
the REVERSE lookup: index the pet store by VALUE (the name) and answer the
relationship ("your dog is wren").

Generalizable: works for any species known to pet_slots (canonical or
runtime-learned), resolves by the actual disclosed name, honors the
self/other boundary (a third-party's pet is not "to me"), and tracks the
active (non-superseded) fact so a renamed pet is reflected.

No hardcoded reply strings — every answer slot is read live from the
PersonalFactStore.
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


def test_reverse_pet_name_recall_who_is_to_me():
    # Disclose a pet by name via the forward contraction miner (dog's a ...).
    eng = _fresh_engine("petname_a")
    eng.process_turn("my dog's a nova scotia duck tolling retriever called wren")
    # Query by the NAME, not the species.
    recall = eng.process_turn("who is wren to me?")
    assert "wren" in recall, recall
    # The relationship is the species: "your dog is wren".
    assert "your dog is wren" in recall.lower(), recall
    # Must NOT be the generic self-blurb (identity block).
    assert "i'm" not in recall.lower().split("?")[0] or "your dog is wren" in recall.lower(), recall


def test_reverse_pet_name_recall_other_species():
    eng = _fresh_engine("petname_b")
    eng.process_turn("my cat is a maine coon called ember")
    recall = eng.process_turn("what is ember to me?")
    assert "your cat is ember" in recall.lower(), recall


def test_reverse_pet_name_recall_tracks_rename():
    eng = _fresh_engine("petname_c")
    eng.process_turn("i have a dog called wren")
    eng.process_turn("no, my dog is actually called briar now")
    recall = eng.process_turn("who is wren to me?")
    # The old name was superseded by the corrected one; a name-based recall
    # must reflect the ACTIVE fact (briar), not the retired earlier name.
    # An honest answer may surface briar OR deny wren — it must never assert
    # "your dog is wren" once wren has been superseded.
    assert "your dog is wren" not in recall.lower(), recall
    recall2 = eng.process_turn("who is briar to me?")
    assert "your dog is briar" in recall2.lower(), recall2


def test_reverse_pet_name_recall_runtime_learned_species():
    # pet_slots grows its species vocabulary at runtime; a never-seen animal
    # ("axolotl") disclosed with a name must still be recallable by name.
    eng = _fresh_engine("petname_d")
    eng.process_turn("i have an axolotl named nyx")
    recall = eng.process_turn("who is nyx to me?")
    assert "your axolotl is nyx" in recall.lower(), recall


def test_reverse_pet_name_recall_third_party_out_of_scope():
    # A third-party pet ("my sister's cat is mochi") is stored under a
    # non-user subject and must NOT be claimed as the user's on a "to me"
    # query. The branch's self/other boundary (subject != "i" skipped) means
    # the name lookup falls through instead of asserting "your cat is mochi".
    eng = _fresh_engine("petname_e")
    eng.process_turn("my sister's cat is mochi")
    recall = eng.process_turn("who is mochi to me?")
    assert "your cat is mochi" not in recall.lower(), recall


def test_reverse_name_recall_person():
    # Generalization beyond pets (T40): a person disclosed as a relation
    # ("my sister is sarah") must resolve by NAME on the same query shape
    # the pet branch already answered. The label comes from the LIVE
    # attribute ("sister"), not a pet grammar.
    eng = _fresh_engine("petname_person")
    eng.process_turn("my sister is sarah")
    recall = eng.process_turn("who is sarah to me?")
    assert "your sister is sarah" in recall.lower(), recall


def test_reverse_name_recall_possession():
    # A named possession ("my car is the blue one") must similarly resolve by
    # its name. The trigger captures the multi-word name; the label is the
    # possession noun from the fact attribute.
    eng = _fresh_engine("petname_poss")
    eng.process_turn("my car is the blue one")
    recall = eng.process_turn("what is the blue one to me?")
    assert "your car is the blue one" in recall.lower(), recall


def test_reverse_name_recall_profile_fact_not_hijacked():
    # Negative guard for the generalization: a place/profile value fact
    # ("i was born in paris") must NOT be claimed as a named relation on a
    # "who is paris to me?" query — the resolver is entity-scoped (pets /
    # relationship nouns / possession nouns), so profile attributes are
    # skipped and the lookup falls through honestly.
    eng = _fresh_engine("petname_profile")
    eng.process_turn("i was born in paris")
    recall = eng.process_turn("who is paris to me?")
    assert "your born is paris" not in recall.lower(), recall


# ── Round 2026-08-12T1234Z generalization extensions ────────────────────────
# The T40 fix (above) only handled pets stored under subject=="i" and the
# literal "who|what is" regex. This round generalized the resolver to: (a) any
# entity named via a "name" attribute (incl. animals NOT in the species seed,
# e.g. goshawk); (b) apostrophe contractions ("what's X to me?"); (c) the
# "whose <species> is it" possessor query with head-token species overlap.
def test_reverse_name_recall_apostrophe_contraction():
    # "what's" must normalize to "what is" so the same path fires.
    eng = _fresh_engine("petname_apos")
    eng.process_turn("my dog's a retriever called bracken")
    recall = eng.process_turn("what's bracken to me?")
    assert "your dog is bracken" in recall.lower(), recall


def test_reverse_pet_name_recall_species_subject_nonseed():
    # A pet disclosed via the possessive-split form ("my goshawk's name is
    # vesper") is stored as ('goshawk','name','vesper') — subject is the
    # species, which is NOT in the pet_slots seed. The generalized resolver
    # keys off the universal "name" attribute, so a non-seed animal still
    # resolves by name.
    eng = _fresh_engine("petname_goshawk")
    eng.process_turn("my goshawk's name is vesper, she rides the thermals")
    recall = eng.process_turn("who is vesper to me?")
    assert "your goshawk is vesper" in recall.lower(), recall


def test_reverse_whose_species_query():
    # "whose <species> is it" must resolve the possessor from the live store,
    # tolerating head-token species overlap ("hawk" vs stored "goshawk").
    eng = _fresh_engine("petname_whose")
    eng.process_turn("my goshawk's name is vesper")
    recall = eng.process_turn("whose hawk is it?")
    assert "yours" in recall.lower(), recall


def test_reverse_name_recall_person_combined_attr():
    # A person disclosed as a combined relation+name attribute
    # ("my brother cal is the one who taught me") stores ('i','brother cal',desc).
    # The resolver splits the relation head ("brother") and answers by name.
    eng = _fresh_engine("petname_brother")
    eng.process_turn("my brother cal is the one who taught me to read the wind")
    recall = eng.process_turn("who is cal to me?")
    assert "your brother is cal" in recall.lower(), recall


def test_reverse_name_recall_priority_over_relation_collision():
    # When a name collides across entity types (bracken is both the user's dog
    # and, via a faulty inference, a "neighbour"), the resolver must prefer the
    # most entity-like reading (pet) over the relationship noun.
    eng = _fresh_engine("petname_collide")
    eng.process_turn("my dog is a lurcher called bracken")
    eng.process_turn("my neighbour is bracken")
    recall = eng.process_turn("what's bracken to me?")
    assert "your dog is bracken" in recall.lower(), recall


