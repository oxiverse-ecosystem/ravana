"""Capability test: DERIVATION relations ("<thing> is named after <referent>").

Round 2026-09-30T1031Z chat probe, turn 23:

    "the sourdough starter is named after my uncle bartholomew"

stored ('i','uncle bartholomew','bartholomew') — a fact about the USER whose
value merely restates its own key — because the kin miner matched the
possessive INSIDE the prepositional phrase ("my uncle"). The relation class
("named after") had no vocabulary at all, and the subject (the starter) was
never recovered, so recall answered "your uncle bartholomew is bartholomew."

The capability under test is a DERIVATION relation: an owned thing is related
to a REFERENT it is named after / inspired by / based on. It is stored
ENTITY-keyed with the relation as the attribute, so the subject stays the
thing and the relation survives.

Everything asserted here is a property of the SHARED vocabulary and the
miner's structure, not of the probe sentence: an unseen naming verb
("dubbed after") must resolve, and a non-derivation disclosure must not
produce a derivation fact.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "ravana_ml", "src"),
           os.path.join(REPO, "ravana", "src"),
           os.path.join(REPO, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("RAVANA_OFFLINE", "1")

from ravana.chat import derivation_attrs as dv  # noqa: E402


# --------------------------------------------------------------- vocabulary --
def test_seed_vocabulary_resolves_to_a_canonical_relation():
    assert dv.derivation_of("named after") == "named after"
    assert dv.derivation_of("called after") == "named after"
    assert dv.derivation_of("named for") == "named after"
    assert dv.derivation_of("named in memory of") == "named after"
    assert dv.derivation_of("inspired by") == "inspired by"
    assert dv.derivation_of("based on") == "based on"


def test_a_non_derivation_phrase_is_not_a_derivation():
    assert dv.derivation_of("is") is None
    assert dv.derivation_of("the starter") is None
    # a possessive is not an attribution
    assert dv.derivation_of("my uncle") is None


def test_canonical_form_keeps_the_english_verb_not_a_stem():
    """'named after' must not canonicalize to 'name after'.

    A stem-folded key is not recallable: a later question carrying the surface
    verb resolves to a DIFFERENT attribute, so the stored fact is never found.
    The canonical form is the seed key whenever the verb's stem matches a seed
    verb, and a de-inflected stem only for a verb RAVANA is seeing for the
    first time.
    """
    assert dv.derivation_of("named after") == "named after"
    assert dv.derivation_of("called after") == "named after"
    assert dv.derivation_of("christened after") == "named after"
    # unseen verb -> de-inflected, but still the verb it was said with
    canon = dv.derivation_of("dubbed after")
    assert canon is not None
    assert canon.endswith(" after")
    assert "dub" in canon and "dubb" not in canon


def test_runtime_growth_registers_an_unseen_naming_verb():
    """An unseen naming verb becomes addressable without a code change."""
    assert dv.learn_naming_verb("tagged") == "tag"
    assert dv.derivation_of("tagged after") is not None
    # and it stays resolvable for the NEXT disclosure
    assert dv.naming_verb_of("tagging") == "tag"
    assert dv.naming_verb_of("tags") == "tag"


def test_learning_refuses_words_that_cannot_be_naming_verbs():
    for bad in ("is", "the", "my", "of", "", "  "):
        assert dv.learn_naming_verb(bad) is None
    # an ordinary noun is not registered merely by being scanned
    assert dv.learn_naming_verb("starter") is None


def test_learn_does_not_widen_the_class_from_a_failed_lookup():
    """derivation_of must not MEMORISE a phrase that is not a derivation.

    Otherwise a non-derivation read once ('the starter') becomes a derivation
    forever, and the class leaks on every later utterance.
    """
    assert dv.derivation_of("the starter") is None
    assert dv.derivation_of("the starter") is None


def test_render_uses_stored_words_only():
    assert dv.render_derivation("sourdough starter", "named after",
                                "uncle bartholomew") == \
        "your sourdough starter is named after uncle bartholomew"
    assert dv.render_derivation("", "named after", "x") == ""


# ------------------------------------------------------------------- miner ---
@pytest.fixture(scope="module")
def engine():
    from ravana.chat.engine import CognitiveChatEngine
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix="derivcap")


def _facts(eng):
    return eng.user_model.personal_facts.facts


def test_derivation_is_stored_entity_keyed_with_the_relation(engine):
    engine.process_turn("the sourdough starter is named after my uncle bartholomew")
    keys = [k for k in _facts(engine) if "after" in k[1]]
    assert keys, "no derivation fact was stored"
    subj, attr, val = keys[0]
    # the SUBJECT is the owned thing, not the user
    assert subj != "i"
    assert "sourdough" in subj
    # the RELATION survives as the attribute
    assert attr == "named after"
    # the REFERENT keeps its relationship word so the person is recallable
    assert val == "uncle bartholomew"
    # and nothing degenerate was stored about the user for this disclosure
    assert not [k for k in _facts(engine)
                if k[0] == "i" and k[2] in k[1].split()]


def test_no_fact_whose_value_restates_its_own_key(engine):
    for k in _facts(engine):
        subj, attr, val = k
        assert val not in attr.split() or attr.split() != [val], \
            f"degenerate fact {k}"


def test_an_unseen_derivation_verb_is_mined_and_the_vocabulary_grows(engine):
    before = dv.learn_naming_verb("tagged")
    assert before == "tag"
    engine.process_turn("my blue kettle is tagged after my grandmother ines")
    keys = [k for k in _facts(engine) if "kettle" in k[0]]
    assert keys, f"unseen derivation verb was not mined; facts={list(_facts(engine))}"
    subj, attr, val = keys[0]
    assert "kettle" in subj and subj != "i"
    assert val == "grandmother ines"
    assert attr.endswith("after")


def test_a_plain_possession_disclosure_is_not_a_derivation(engine):
    """Guard: the class must not fire on non-attribution sentences."""
    before = dict(
        (k, v.value) for k, v in _facts(engine).items())
    engine.process_turn("my neighbour clara waters the plants on fridays")
    after = dict((k, v.value) for k, v in _facts(engine).items())
    new = [k for k in after if k not in before]
    for k in new:
        assert not k[1].endswith("after"), \
            f"a non-derivation disclosure produced a derivation fact: {k}"


def test_a_derivation_about_the_users_own_name_is_left_to_the_name_miner(engine):
    """'i am named after my grandmother' is about the USER, not an owned thing.

    The derivation miner must decline it so the existing self-naming path owns
    it; storing ('i','named after','grandmother') here would put an owned-thing
    key on the user and collide with the name slot.
    """
    before = set(_facts(engine))
    engine.process_turn("i was named after my grandmother rose")
    new = [k for k in _facts(engine) if k not in before]
    for k in new:
        assert k[1] != "named after", \
            f"the derivation miner stole the user's own naming: {k}"