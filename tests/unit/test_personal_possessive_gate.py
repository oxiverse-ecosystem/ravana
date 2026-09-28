"""Personal-possessive pre-gate precision (decision_gate).

The gate exists so a query about the USER's own life/possessions routes to
episodic recall instead of web_search (and instead of falling through to
_consult_internal_knowledge and confabulating). It must be PRECISE: a
possessive plus a common noun is not evidence of autobiography.

Three directions are asserted here because the gate previously had NO test
coverage at all, which is how a 685-word enumerated seed shipped green:

  (a) world-knowledge / how-to queries that merely contain a possessive are
      NOT gated (web_search must still be reachable),
  (b) true autobiographical queries ARE gated,
  (c) the vocabulary GROWS at runtime — the deciding seed-vs-hardcoding test.
"""
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_REPO, "ravana", "src"),
           os.path.join(_REPO, "ravana_ml", "src"),
           os.path.join(_REPO, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.agent import decision_gate as _gate  # noqa: E402


# --- (a) must NOT be gated: genuine world-knowledge / how-to queries that
# happen to contain "my"/"your". Every one of these was blocked before the fix.
WORLD_KNOWLEDGE = [
    "how does your brain process language",
    "how do i change a flat tire on my car",
    "why does my computer fan get so loud",
    "what is the best way to clean my laptop keyboard",
    "how long should i charge my phone",
    "what causes a headache behind my eyes",
    "why does my dog keep scratching its ear",
    "how does your team handle code review",
    "what is your favourite programming language",
    # regression guards for the queries that were already allowed
    "what is the capital of france",
    "how does a combustion engine work",
    "explain quantum entanglement",
    "who wrote dune",
    "what is the boiling point of water",
    "how does your liver detoxify alcohol",
]

# --- (b) MUST be gated: the query is about the user's own life/possessions,
# so only episodic recall can answer it and confabulation is the risk.
AUTOBIOGRAPHICAL = [
    "what is wrong with my car",
    "what happened with my dog",
    "what is wrong with my car?",
    "what did i say about my sister",
    "why did my laptop die",
    "where is my passport",
    "who is my manager",
    "what is my rent",
    "what happened to my phone",
    "remind me what you said about my brother",
]


@pytest.mark.parametrize("query", WORLD_KNOWLEDGE)
def test_world_knowledge_query_is_not_gated(query):
    assert _gate._is_personal_possessive_query(query) is False, (
        f"world-knowledge query wrongly gated (web_search unreachable): {query!r}")


@pytest.mark.parametrize("query", AUTOBIOGRAPHICAL)
def test_autobiographical_query_is_gated(query):
    assert _gate._is_personal_possessive_query(query) is True, (
        f"autobiographical query not gated (would confabulate): {query!r}")


# --- (c) the growth path. A seed is only legitimate if RAVANA can extend it
# through experience. A frozen vocabulary is a fixed table wearing a seed's
# clothing, so this asserts the seam actually works.
def test_entity_vocabulary_grows_at_runtime():
    original = set(_gate._PERSONAL_ENTITY_WORDS)
    try:
        assert _gate._is_personal_possessive_query(
            "what is wrong with my zblork") is False, (
            "precondition: an unlearned entity word must not gate by accident")
        _gate.add_personal_entity_words({"zblork"})
        assert "zblork" in _gate._PERSONAL_ENTITY_WORDS
        assert _gate._is_personal_possessive_query(
            "what is wrong with my zblork") is True
    finally:
        _gate._PERSONAL_ENTITY_WORDS.clear()
        _gate._PERSONAL_ENTITY_WORDS.update(original)


def test_growth_does_not_survive_process_restart_frozen():
    """The seed must be a real set in module state, not a literal read from a
    frozen table each call — otherwise growth is nominal. Adding a word twice
    must not create a duplicate entry that survives differently."""
    original = set(_gate._PERSONAL_ENTITY_WORDS)
    try:
        _gate.add_personal_entity_words({"quimble"})
        size_after_first = len(_gate._PERSONAL_ENTITY_WORDS)
        _gate.add_personal_entity_words({"quimble"})
        assert len(_gate._PERSONAL_ENTITY_WORDS) == size_after_first
    finally:
        _gate._PERSONAL_ENTITY_WORDS.clear()
        _gate._PERSONAL_ENTITY_WORDS.update(original)


def test_possessive_is_required():
    """No possessive => never autobiographical, whatever the noun."""
    assert _gate._is_personal_possessive_query("what is wrong with the car") is False
    assert _gate._is_personal_possessive_query("what is wrong with a dog") is False


def test_learned_entity_from_a_disclosure_gates_a_followup_query():
    """End-to-end shape of the growth path: a disclosure establishes a new
    personal entity, and a later query about that entity is then gated."""
    original = set(_gate._PERSONAL_ENTITY_WORDS)
    try:
        disclosure = "my hovercraft has been leaking since march"
        # precondition: an unlearned entity word must not gate by accident
        assert _gate._is_personal_possessive_query(
            "what is wrong with my hovercraft") is False
        learned = _gate.learn_personal_entities_from_disclosure(disclosure)
        assert "hovercraft" in learned, (
            f"disclosure did not teach the entity head noun: {learned}")
        assert _gate._is_personal_possessive_query(
            "what is wrong with my hovercraft") is True
    finally:
        _gate._PERSONAL_ENTITY_WORDS.clear()
        _gate._PERSONAL_ENTITY_WORDS.update(original)


def test_growth_takes_the_head_noun_not_a_modifier():
    """The learned word must be the head of the possessive phrase — the thing
    the user owns — not a pre-modifier that could be anything."""
    original = set(_gate._PERSONAL_ENTITY_WORDS)
    try:
        _gate.learn_personal_entities_from_disclosure(
            "my old beekeeping mentor taught me to read hives")
        assert "mentor" in _gate._PERSONAL_ENTITY_WORDS
        assert _gate._is_personal_possessive_query(
            "what is wrong with my mentor") is True
    finally:
        _gate._PERSONAL_ENTITY_WORDS.clear()
        _gate._PERSONAL_ENTITY_WORDS.update(original)


def test_growth_does_not_learn_from_a_question():
    """A knowledge query must never teach the gate that the world is the
    user's personal property — that would make the gate learn its way back
    into the over-blocking it was fixed for."""
    original = set(_gate._PERSONAL_ENTITY_WORDS)
    try:
        learned = _gate.learn_personal_entities_from_disclosure(
            "what is the history of the roman empire")
        assert learned == set()
    finally:
        _gate._PERSONAL_ENTITY_WORDS.clear()
        _gate._PERSONAL_ENTITY_WORDS.update(original)

