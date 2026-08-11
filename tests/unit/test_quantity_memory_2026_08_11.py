"""Regression tests for round 2026-08-11T0521Z: structured quantity memory.

The activity/event miner already preserved the COUNT inside the 'does'/'event'
text fact, but there was NO structured count store, so RAVANA could not:

  * SYNTHESIZE a clean count answer for a multi-word noun phrase
    ("how many racing pigeons do i keep" -> "twelve"), because the old
    recall regex only matched a SINGLE noun word and a fixed verb list; and
  * AGGREGATE counts across the store ("how many pets do i have in total"),
    which was simply impossible before.

These tests assert the new QuantityMemory store (on UserModel) captures
(subject, kind, count, noun) from count disclosures and answers both single
lookups (multi-word noun) and aggregation. They fail without the capability
and pass with it. No LLM, no retraining, no per-topic answer table.
"""
import os
import sys

os.environ["RAVANA_OFFLINE"] = "1"
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_PROJ, os.path.join(_PROJ, "ravana", "src"), os.path.join(_PROJ, "ravana_ml", "src")):
    sys.path.insert(0, _p)

from ravana.chat.user_model import UserModel
from ravana.chat.personal_fact_store import number_to_int


def _mine(cases):
    um = UserModel()
    for c in cases:
        um.mine_personal_facts(c)
    return um


def test_count_captured_for_multiword_noun():
    um = _mine(["i keep twelve racing pigeons in the loft"])
    rec = um.quantity_memory.query_count("racing pigeons", kind="keep")
    assert rec is not None, "racing pigeons count should be captured"
    assert rec.count == 12, rec
    assert rec.noun_canonical == "pigeon", rec.noun_canonical  # species-canonical


def test_count_captured_for_varied_verbs_and_nouns():
    um = _mine([
        "i have three cats",
        "i bake two sourdough loaves every week",
        "i lost five hens to the fox last winter",
    ])
    assert um.quantity_memory.query_count("cats", kind="have").count == 3
    assert um.quantity_memory.query_count("sourdough loaves", kind="bake").count == 2
    assert um.quantity_memory.query_count("hens", kind="lose").count == 5


def test_aggregation_across_species():
    um = _mine([
        "i have three cats",
        "i keep two dogs",
        "i raise four hermit crabs",
        "i lost five hens",  # category 'loss' — must NOT inflate the pet total
    ])
    total_pets = um.quantity_memory.aggregate(category="possession")
    # 3 cats + 2 dogs + 4 crabs = 9; hens (loss) excluded
    assert total_pets == 9, total_pets


def test_question_not_stored_as_fact():
    um = _mine(["how many racing pigeons do i keep"])
    assert len(um.quantity_memory.records) == 0, \
        "a count QUESTION must never be seeded as a quantity fact"


def test_number_word_parser():
    assert number_to_int("twelve") == 12
    assert number_to_int("3") == 3
    assert number_to_int("a") == 1
    assert number_to_int("bananas") is None


def test_correction_supersedes_prior_count():
    um = _mine(["i have three cats", "no, i have four cats actually"])
    # both records present, but only one active (non-superseded) for cats/have
    active = [r for r in um.quantity_memory.records.values()
              if r.subject == "i" and r.kind == "have"
              and r.noun_canonical == "cat" and not r.superseded]
    assert len(active) == 1, [(r.count, r.superseded) for r in um.quantity_memory.records.values()]
    assert active[0].count == 4
