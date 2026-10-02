"""RED->GREEN: a relationship fact must not store a VALUE contained in its own
ATTRIBUTE (card t_84311c95, round 2026-09-29T0823Z).

SYMPTOM (live round transcript tmp/round_0823z_out.txt, turn 19):

    Q: who is devika
       A: your sister devika handles glazing end is devika handles the glazing end.

The attribute and the value overlap, so the "<rel> is <val>" render duplicates
itself. The store confirmed zero-information facts existed:

    ('i', 'sister devika handles glazing end', 'devika handles the glazing end')
    ('i', "friend last week and i can't stop thinking about it",
              "last week and i can't stop thinking about it")
    ('i', 'partner and i are thinking about moving in together next year',
              'and i are thinking about moving in together next year')

ROOT CAUSE — two coupled defects in UserModel.mine_personal_facts:

  1. The lowercase-name fallback appended EVERY remaining token to the name
     candidate list, so a trailing CLAUSE became the "name".
  2. It never recorded how many tokens the name consumed, so the `_after`
     slice re-read the same span from index 0 — making the VALUE a
     re-slice of its own ATTRIBUTE. The degenerate-fact guard only compared
     the value to the RELATIONSHIP WORD, so a value repeating its own
     attribute slot was stored.

THE INVARIANT IS INFORMATION, NOT VOCABULARY: a fact must say something its
attribute does not already say. That holds for every relationship word and
any name RAVANA later learns.

NOTE ON A PRIOR FAILED ATTEMPT: run 1923 tried to fix this and was REVERTED.
It bounded the name span but DISCARDED the name, so every legitimate
relationship-name fact was lost ("my friend rhea" -> no facts) and it raised
`UnboundLocalError: _skip_words`. The regression tests below pin BOTH halves:
the zero-information facts must be gone AND the names must survive.

Every assertion uses `assert` — a test that RETURNS a bool is always green
under pytest, a defect class this repo has hit before.
"""
import os
import sys

import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.user_model import UserModel  # noqa: E402


def _mine(turns):
    um = UserModel()
    for t in turns:
        um.mine_personal_facts(t)
    return um


def _facts(um):
    """Map attribute -> value.

    The store is keyed by a (subject, attribute, value) triple, so the
    ATTRIBUTE must be read out of the tuple. Joining the whole tuple would
    fold the value into the "key" and make the invariant untestable.
    """
    return {str(k[1]): v.value
            for k, v in (um.personal_facts.facts or {}).items()}


def _value_is_inside_its_own_attribute(facts):
    """The information invariant, as a predicate over the mined store.

    `facts` maps attribute -> value, so a fact is degenerate exactly when the
    value's token sequence already appears as a contiguous run inside its own
    attribute.
    """
    bad = []
    for attr, val in facts.items():
        vt, at = str(val).split(), str(attr).split()
        n = len(vt)
        if not n:
            continue
        if any(at[i:i + n] == vt for i in range(len(at) - n + 1)):
            bad.append((attr, val))
    return bad


# ── the three zero-information facts from the live round ──────────────────
CARD_FACTS = [
    "my sister devika handles the glazing end",
    "my friend messaged me last week and i can't stop thinking about it",
    "my partner and i are thinking about moving in together next year",
]


@pytest.mark.parametrize("disclosure", CARD_FACTS)
def test_no_value_is_contained_in_its_own_attribute(disclosure):
    """The card's own evidence: no mined fact may store a value that is a
    contiguous token-run already inside that same fact's attribute."""
    facts = _facts(_mine([disclosure]))
    offenders = _value_is_inside_its_own_attribute(facts)
    assert not offenders, (
        f"zero-information relationship fact(s) for {disclosure!r}: {offenders}")


# ── the general class: the guard is about INFORMATION, not vocabulary ─────
@pytest.mark.parametrize("disclosure", [
    "my cousin tanvi keeps showing up with random houseplants",
    "my neighbour old mr. halvorsen repainted the entire fence last spring",
    "my roommate jo was awake all night rearranging the kitchen",
    "my friend rhea messaged me last week and i can't stop thinking about it",
])
def test_no_zero_information_fact_for_any_relationship_word(disclosure):
    facts = _facts(_mine([disclosure]))
    offenders = _value_is_inside_its_own_attribute(facts)
    assert not offenders, (
        f"zero-information fact for {disclosure!r}: {offenders}")


# ── REGRESSION GUARD: the prior attempt broke exactly this ───────────────
# Bounding the name span must CLOSE it, never DISCARD the name.
@pytest.mark.parametrize("disclosure,expect_name", [
    ("my friend rhea", "rhea"),
    ("my cousin tanvi", "tanvi"),
    ("my roommate jo", "jo"),
    ("my friend rhea messaged me", "rhea"),
    ("my cousin tanvi keeps showing up with random houseplants", "tanvi"),
    ("my neighbour old mr. halvorsen repainted the entire fence last spring",
     "halvorsen"),
])
def test_relationship_name_facts_are_still_mined(disclosure, expect_name):
    facts = _facts(_mine([disclosure]))
    joined = " ".join(f"{k} {v}" for k, v in facts.items())
    assert expect_name in joined, (
        f"legitimate name {expect_name!r} was dropped for {disclosure!r}: "
        f"{facts}")


# ── a clause after the name must never be stored AS the name ─────────────
# The defect is the clause landing in the NAME slot (the attribute), which
# made the attribute swallow the sentence. The trailing clause itself is the
# user's own content and IS legitimate as a value — dropping it would discard
# something real, which the card explicitly does not ask for.
def test_clause_after_relationship_word_is_not_stored_as_a_name():
    facts = _facts(_mine(
        ["my friend rhea messaged me last week and i can't stop thinking "
         "about it"]))
    assert facts, "the disclosure should still mine something informative"
    for attr, val in facts.items():
        assert "can't stop thinking" not in attr, (
            f"relationship clause stored as the NAME/attribute: {attr!r}")
        # the name must survive as the leading span of the attribute
        assert attr.split(" ")[:2] == ["friend", "rhea"], (
            f"name span not bounded to the name: {attr!r}")
        assert not _value_is_inside_its_own_attribute({attr: val}), (
            f"value repeats its own attribute: {attr!r} -> {val!r}")


# ── the UnboundLocalError probe from the prior failed attempt ────────────
# `_skip_words` was referenced on a path where it was never assigned. This
# disclosure shape must mine without raising.
@pytest.mark.parametrize("disclosure", [
    "my great-aunt Hortense folds a thousand paper cranes every winter",
    "my great-aunt Mariam went to the coast last summer",
    "my aunt rita sings in the church choir on sundays",
])
def test_unrecognized_verb_after_name_does_not_raise(disclosure):
    um = UserModel()          # must not raise UnboundLocalError
    um.mine_personal_facts(disclosure)
    assert isinstance(um.personal_facts.facts, dict)
