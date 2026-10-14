"""Regression tests for feature t_ad2ff382 — activity-object noun-phrase
preservation in fact storage (round 2026-10-13T0647Z).

The activity/event miners previously resolved the object phrase through
_opinion_topic(), which CUTS at internal determiners (e.g. "the") because
it is designed for stance-key extraction. This caused activity facts to
lose their object: "learning to play the veena" -> "learning play".

The fix: _extract_activity_obj() preserves the full noun phrase for
activity/event fact storage. All tests call the real function — no
mocking, no authored reply strings, no per-topic table.
"""
import os
import sys

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(PROJ, "ravana_ml", "src"))

from ravana.chat.user_model import _extract_activity_obj, UserModel


# ── _extract_activity_obj: full noun-phrase preservation ──────────────────

def test_preserves_full_nounPhrase_with_internal_determiner():
    """The canonical case: 'to play the veena' must keep 'the veena',
    not collapse to just 'play' (the old _opinion_topic behavior)."""
    result = _extract_activity_obj("to play the veena")
    assert result == "play the veena", (
        f"expected 'play the veena', got {result!r}")


def test_preserves_compound_noun_phrase():
    """'a juniper bonsai' must keep both content words, not just 'juniper'."""
    result = _extract_activity_obj("a juniper bonsai")
    assert result == "juniper bonsai", (
        f"expected 'juniper bonsai', got {result!r}")


def test_preserves_noun_phrase_after_leading_determiner():
    """'the reef tank' must keep 'reef tank' after stripping leading 'the'."""
    result = _extract_activity_obj("the reef tank")
    assert result == "reef tank", (
        f"expected 'reef tank', got {result!r}")


def test_preserves_multiword_object_after_particle():
    """'up the frame' must strip leading particle 'up' and keep 'frame'."""
    result = _extract_activity_obj("up the frame")
    assert result == "frame", (
        f"expected 'frame', got {result!r}")


def test_stops_at_preposition():
    """Span must stop at a preposition: 'playing the veena since 2019'
    must not include 'since'."""
    result = _extract_activity_obj("playing the veena since 2019")
    assert result == "playing the veena", (
        f"expected 'playing the veena', got {result!r}")


def test_stops_at_conjunction():
    """Span must stop at a conjunction: 'keep juniper bonsai and play veena'
    must not include 'and'."""
    result = _extract_activity_obj("keep juniper bonsai and play veena")
    assert result == "keep juniper bonsai", (
        f"expected 'keep juniper bonsai', got {result!r}")


def test_strips_leading_infinitive_to():
    """Infinitive 'to' must be stripped: 'to play the veena' -> 'play the veena'."""
    result = _extract_activity_obj("to play the veena")
    assert not result.startswith("to "), (
        f"leading 'to' was not stripped, got {result!r}")


def test_strips_leading_determiners_and_particles():
    """Leading determiners/particles must be stripped before collecting."""
    result = _extract_activity_obj("the reef tank")
    assert result == "reef tank", (
        f"expected 'reef tank', got {result!r}")


def test_returns_empty_for_only_determiners():
    """A phrase with only determiners must return empty string."""
    result = _extract_activity_obj("the")
    assert result == "", (
        f"expected empty string, got {result!r}")


def test_returns_empty_for_empty_input():
    """Empty input must return empty string."""
    result = _extract_activity_obj("")
    assert result == "", (
        f"expected empty string, got {result!r}")


def test_returns_empty_for_only_infinitive_to():
    """Just 'to' must return empty string (nothing real remains)."""
    result = _extract_activity_obj("to")
    assert result == "", (
        f"expected empty string, got {result!r}")


def test_bounded_to_six_tokens():
    """Output must be bounded to 6 tokens to prevent runaway spans."""
    long_phrase = "one two three four five six seven eight nine ten"
    result = _extract_activity_obj(long_phrase)
    assert len(result.split()) <= 6, (
        f"expected at most 6 tokens, got {len(result.split())}: {result!r}")


def test_strips_leading_framer_words():
    """Leading framer words (now, already, still) must be stripped."""
    result = _extract_activity_obj("now building the cabinets")
    assert result == "building the cabinets", (
        f"expected 'building the cabinets', got {result!r}")


def test_skips_internal_framer_words():
    """Internal framer words are skipped but do not stop the span."""
    result = _extract_activity_obj("building now the cabinets")
    # "now" is a framer — skipped, but "the" is kept (internal determiner)
    assert "now" not in result.split(), (
        f"framer 'now' should be skipped, got {result!r}")


# ── Contrast with _opinion_topic (old behavior) ────────────────────────────

def test_contrast_with_opinion_topic_cuts_at_determiner():
    """_opinion_topic cuts at 'the' (returns 'play'), while
    _extract_activity_obj preserves it (returns 'play the veena').
    This test documents WHY the new function was needed."""
    um = UserModel()
    old_result = um._opinion_topic("to play the veena")
    new_result = _extract_activity_obj("to play the veena")
    # Old behavior: cuts at "the" -> "play"
    assert old_result == "play", (
        f"expected old _opinion_topic to return 'play', got {old_result!r}")
    # New behavior: preserves full phrase -> "play the veena"
    assert new_result == "play the veena", (
        f"expected 'play the veena', got {new_result!r}")
    # They must differ — that is the whole point
    assert old_result != new_result, (
        "old and new should differ for phrases with internal determiners")
