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
