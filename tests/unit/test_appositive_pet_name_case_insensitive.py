"""Regression: an appositive possession disclosure written in LOWERCASE must
mine the name, exactly as the capitalised form already does.

DEFECT (measured, round 2026-10-03T2040Z, clean engine, RAVANA_OFFLINE=1):

    "my dog Biscuit sleeps on the wedging table"
        -> ('i','dog','biscuit')                       MINED
    "my dog biscuit sleeps on the wedging table"
        -> {}                                         DROPPED

Nothing about the disclosure differs except capitalisation, and chat users type
lowercase. Because no name was stored, the later cued recall

    "what is my dog called"
        -> "A dog is a domesticated descendant of the wolf, ..."

fell through to a generic encyclopedic definition — RAVANA answered a question
about the USER'S OWN pet with a dictionary entry about the species.

ROOT CAUSE: ravana/src/ravana/chat/user_model.py, the `_APPOSITIVE_PET_PAT`
branch. A lowercase name candidate was accepted only when the word FOLLOWING it
was a copula ("my dog biscuit is ..."). That tests the wrong thing: the guard
exists to stop the name group grabbing the PREDICATE ("my dog likes the park" ->
name "likes"), but "is the next word a copula" is not a test for "is this token a
verb". A predicate-led clause fails the copula test too, so the guard was
simultaneously too strict on real names and by accident right on verbs.

FIX: ask the question the guard actually means, using the vocabulary the module
already owns — `is_verb_phrase` / `is_past_finite_aux` (is this candidate the
predicate?) and `pet_slots.is_function_word` (is it closed-class?). Accept the
lowercase candidate unless it is a verb head or a function word.

This file is the RED half: it fails on the pre-fix source and passes after.
"""
import os
import sys

import pytest

PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for _p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src",
           f"{PROJ}\\ravana-v2\\src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("RAVANA_OFFLINE", "1")


@pytest.fixture(scope="module")
def user_model():
    from ravana.chat.user_model import UserModel
    return UserModel


def _mine(um, text):
    before = set(um.personal_facts.facts)
    um.mine_personal_facts(text)
    return [k for k in um.personal_facts.facts if k not in before]


def test_lowercase_appositive_pet_name_is_mined(user_model):
    """The defect: capitalisation alone decided whether the name survived."""
    um = user_model()
    new = _mine(um, "my dog biscuit sleeps on the wedging table")
    assert ("i", "dog", "biscuit") in new, (
        "lowercase appositive pet name was dropped; got %r" % (new,))


def test_lowercase_and_capitalised_forms_agree(user_model):
    """The two spellings must produce the SAME fact. This is the invariant the
    defect violated: nothing but case distinguished them."""
    a = user_model()
    b = user_model()
    _mine(a, "my cat ember sleeps on the windowsill")
    _mine(b, "my cat Ember sleeps on the windowsill")
    keys_a = {k for k in a.personal_facts.facts if k[0] == "i"}
    keys_b = {k for k in b.personal_facts.facts if k[0] == "i"}
    assert keys_a == keys_b, "case changed the mined facts: %r vs %r" % (keys_a, keys_b)


def test_lowercase_name_still_yields_the_pet_activity(user_model):
    """The activity that follows the name must still be captured, so
    "what does my dog do" has something to answer with."""
    um = user_model()
    new = _mine(um, "my dog biscuit sleeps on the wedging table")
    acts = [k for k in new if k[1].endswith("_activity")]
    assert acts, "pet activity companion fact missing; got %r" % (new,)


@pytest.mark.parametrize("species,name,predicate", [
    ("dog", "biscuit", "sleeps on the wedging table"),
    ("cat", "mochi", "naps on the windowsill"),
    ("ferret", "pim", "hides my car keys"),
    ("parrot", "kiwi", "talks a lot"),
])
def test_lowercase_name_mined_across_species(user_model, species, name, predicate):
    """6f GENERALIZE: the trigger is "my <species> <name> <predicate>", not
    "my dog ...". Every species RAVANA knows must work, seed or runtime-learned.

    The slot is asserted through pet_slots' own resolver, not by string equality
    on the species word: RAVANA canonicalises a species to its base form by
    design ("parrot" -> "bird"), so the test must ask the store where the name
    landed rather than assume the user\'s exact word is the key.
    """
    from ravana.chat import pet_slots
    um = user_model()
    _mine(um, "my %s %s %s" % (species, name, predicate))
    resolved = pet_slots.species_of(species) or pet_slots.learn_species(species)
    assert ("i", pet_slots.slot_for(resolved, 1), name) in um.personal_facts.facts, (
        "%s/%s not mined; store=%r" % (
            species, name,
            {k: getattr(v, "value", v)
             for k, v in um.personal_facts.facts.items()}))


@pytest.mark.parametrize("verb_led_clause", [
    "my dog likes the park",
    "my dog barks loudly",
    "my dog sleeps on the table",
    "my cat chases the laser pointer",
])
def test_predicate_is_never_stored_as_the_name(user_model, verb_led_clause):
    """The hazard the old copula guard was protecting against must stay
    protected: with no name in the disclosure, the PREDICATE must not be
    mistaken for one."""
    um = user_model()
    _mine(um, verb_led_clause)
    for (subj, attr, val) in um.personal_facts.facts:
        if subj == "i" and not attr.endswith("_activity"):
            assert val not in ("likes", "barks", "sleeps", "chases"), (
                "%r stored a predicate as a pet name: %r" % (verb_led_clause, val))


def test_copula_is_never_stored_as_the_name(user_model):
    """'my dog is called biscuit' — the copula must not become the name.
    (The pre-fix guard ACCEPTED this one, via its _COPULA set.)"""
    um = user_model()
    _mine(um, "my dog is called biscuit")
    for (subj, attr, val) in um.personal_facts.facts:
        if subj == "i":
            assert val not in ("is", "was", "are", "were"), (
                "copula stored as a pet name: %r" % ((subj, attr, val),))


def test_function_word_is_never_stored_as_the_name(user_model):
    """A closed-class token ('my dog of the neighbour') is not a name."""
    um = user_model()
    _mine(um, "my dog of the neighbour")
    for (subj, attr, val) in um.personal_facts.facts:
        if subj == "i":
            assert val not in ("of", "the", "a", "an"), (
                "function word stored as a pet name: %r" % ((subj, attr, val),))