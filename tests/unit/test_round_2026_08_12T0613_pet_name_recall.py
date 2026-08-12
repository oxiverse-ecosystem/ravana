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

