"""Regression tests for FIX-RV-01: topic-extraction garbage breaks recall.

Trailing modifiers/adverbs pollute topic keys when they survive to the
word list.  "cooking earlier" -> "cooking" (not "cooking earlier"), and
"raise kid care ocean" keeps "ocean" (not "raise kid care").

Verifies the trailing-modifier stripper in _ground_query and the
last-N-words truncation that preserves the head noun.
"""
import pytest


@pytest.fixture(scope="module")
def engine():
    from ravana.chat.engine import CognitiveChatEngine
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True)


@pytest.mark.parametrize("q,expected_topic", [
    # Trailing adverb stripped: "cooking earlier" -> "cooking"
    ("i think cooking earlier is better", "cooking"),
    # Head noun at end preserved: "raise kid care ocean" not "raise kid care"
    ("how do you raise a kid to care about the ocean", "kid care ocean"),
    # Regression guard: single-word topic still works
    ("what is the meaning of life", "life"),
])
def test_trailing_modifier_stripped_from_topic(engine, q, expected_topic):
    """Trailing temporal/adverbial modifiers must not pollute the topic key."""
    subj, conf, method = engine._ground_query(q)
    assert subj == expected_topic, (
        f"{q!r} -> {subj!r} (expected {expected_topic!r})")
    assert conf > 0.0, f"confidence too low for {q!r}"


@pytest.mark.parametrize("q,forbidden", [
    ("i think cooking earlier is better", "earlier"),
    ("how do you raise a kid to care about the ocean", "raise"),
])
def test_trailing_modifier_not_in_topic(engine, q, forbidden):
    """The forbidden word must NOT appear in the grounded topic."""
    subj, _, _ = engine._ground_query(q)
    assert forbidden not in subj, (
        f"{q!r} -> {subj!r} still contains forbidden word {forbidden!r}")


def test_topic_extraction_end_to_end(engine):
    """_extract_topic (full pipeline) must also produce clean topics."""
    # Trailing modifier stripped
    subj, _ = engine._extract_topic("i think cooking earlier is better", [])
    assert subj == "cooking", f"got {subj!r}"

    # Head noun preserved
    subj, _ = engine._extract_topic("how do you raise a kid to care about the ocean", [])
    assert "ocean" in subj, f"head noun 'ocean' missing: {subj!r}"
    assert "raise" not in subj, f"leading verb 'raise' leaked: {subj!r}"
