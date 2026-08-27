"""Regression test for R1 (round 2026-08-18T1340Z): multi-word / paraphrased
entity cued-recall resolution.

A possession the user named is stored under the ENTITY key
('sourdough starter', 'name', 'doris'), NOT under the "i" profile. A later
recall that PARAPHRASES the entity ("what did i name that sourdough culture on
my counter?") must resolve to the stored entity via cross-lemma GloVe linking
and report its name — it must NOT fall through to an unrelated "i"-scoped name
fact (the documented R1 confabulation: it returned the best-friend's name).

Run: RAVANA_OFFLINE=1 python -m pytest tests/unit/test_r1_entity_link_recall.py -q
"""
import os
os.environ.setdefault("RAVANA_OFFLINE", "1")

import sys
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (_PROJ, f"{_PROJ}/ravana_ml/src", f"{_PROJ}/ravana/src", f"{_PROJ}/ravana-v2/src"):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine


def _new_engine(suffix):
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)


def test_r1_paraphrased_entity_resolves_to_stored_name():
    """'sourdough culture' (paraphrase) must recall 'doris' for 'sourdough starter'."""
    eng = _new_engine("t_r1_paraphrase")
    eng.process_turn("my best friend's name is Tomas and he's a chef in Lisbon")
    eng.process_turn("i keep a sourdough starter i named doris")
    r = eng.process_turn("what did i name that sourdough culture on my counter?")
    assert "doris" in r.lower(), f"expected 'doris', got: {r!r}"
    # must NOT leak the unrelated best-friend fact
    assert "tomas" not in r.lower(), f"confabulated best-friend fact: {r!r}"


def test_r1_best_friend_still_resolves():
    """An adjacent possession query must still resolve to its own fact."""
    eng = _new_engine("t_r1_bestfriend")
    eng.process_turn("my best friend's name is Tomas and he's a chef in Lisbon")
    eng.process_turn("i keep a sourdough starter i named doris")
    r = eng.process_turn("what's my best friend's name?")
    assert "tomas" in r.lower(), f"expected 'tomas', got: {r!r}"


def test_r1_unknown_entity_does_not_confabulate():
    """A possession the user never disclosed must fail CLOSED (honest miss),
    never alias onto a stored entity (the RAVANA confabulation bar)."""
    eng = _new_engine("t_r1_unknown")
    eng.process_turn("i keep a sourdough starter i named doris")
    r = eng.process_turn("what did i name that garden gnome on my shelf?")
    assert "doris" not in r.lower(), f"unknown entity leaked a stored name: {r!r}"


def test_r1_own_name_query_not_hijacked():
    """A generic self-name query must not be answered from a possession fact."""
    eng = _new_engine("t_r1_ownname")
    eng.process_turn("i keep a sourdough starter i named doris")
    r = eng.process_turn("what is my name?")
    assert "doris" not in r.lower(), f"own-name query hijacked by possession: {r!r}"


def test_r1_qualifier_prefixed_query_matches_stored_entity():
    """Round 2026-08-22T0058Z: a recall query that adds a relationship
    qualifier the disclosure omitted must still resolve to the stored entity,
    not fall through to the GloVe cross-entity linker (which mis-linked
    'pet rabbit' to a sibling entity). 'my pet rabbit Nimbus' is stored under
    key 'rabbit'; 'what's my pet rabbit's name' must recall 'nimbus'."""
    eng = _new_engine("t_r1_qual_rabbit")
    eng.process_turn("my little cousin Bea is learning to read")
    eng.process_turn("my pet rabbit Nimbus chews my phone chargers")
    r = eng.process_turn("what's my pet rabbit's name and what does he do?")
    assert "nimbus" in r.lower(), f"expected rabbit 'nimbus', got: {r!r}"
    # must NOT leak the cousin fact
    assert "bea" not in r.lower(), f"qualifier query mis-linked to sibling: {r!r}"


def test_r1_qualifier_on_stored_key_matches_unqualified_query():
    """The symmetric case: the miner auto-prefixed 'little cousin Bea' but the
    user later asks 'my cousin Bea'. The stored key 'little cousin bea' must
    match the unqualified query 'cousin bea'."""
    eng = _new_engine("t_r1_qual_cousin")
    eng.process_turn("my pet rabbit Nimbus chews my phone chargers")
    eng.process_turn("my little cousin Bea is learning to read")
    r = eng.process_turn("what's my cousin Bea into, the reading thing?")
    assert "bea" in r.lower(), f"expected cousin 'bea', got: {r!r}"
    assert "nimbus" not in r.lower(), f"unqualified query mis-linked to pet: {r!r}"
