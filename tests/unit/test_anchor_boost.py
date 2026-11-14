"""Tests for the anchor-token boost (retrieve-then-elaborate).

The properties worth pinning are the ones that make this a *commitment*
mechanism rather than a generic logit nudge:

* the strength is the MEASURED knee (9.0), not a tunable guess
* only content words are boosted -- boosting function words would push the
  decoder toward the filler register it already over-produces
* nothing to boost yields None, so the caller passes token_boost=None and
  generation is byte-identical to the pre-change behaviour
"""
import pytest

from ravana.learn.anchor_boost import (
    ANCHOR_TOKEN_BOOST,
    anchor_token_boost,
    content_words,
)


class TestMeasuredKnee:
    def test_strength_is_the_measured_knee(self):
        """docs/ANCHOR_TOKEN_BOOST_MEASURED.md: the sweep's threshold is
        between 3.0 and 6.0 and saturates near 9.0."""
        assert ANCHOR_TOKEN_BOOST == pytest.approx(9.0)

    def test_strength_is_above_the_measured_threshold(self):
        assert ANCHOR_TOKEN_BOOST >= 6.0


class TestContentWords:
    def test_function_words_are_dropped(self):
        words = list(content_words("the force of gravity is a pull"))
        assert "the" not in words
        assert "of" not in words
        assert "is" not in words
        assert "a" not in words

    def test_content_words_are_kept(self):
        words = list(content_words("the force of gravity is a pull"))
        assert "force" in words
        assert "gravity" in words
        assert "pull" in words

    def test_special_tokens_are_dropped(self):
        assert "<bos>" not in list(content_words("gravity <bos> <eos>"))

    def test_punctuation_is_stripped(self):
        words = list(content_words("gravity, force."))
        assert words == ["gravity", "force"]

    def test_empty_and_none_yield_nothing(self):
        assert list(content_words("")) == []
        assert list(content_words(None)) == []

    def test_order_is_preserved(self):
        assert list(content_words("zeta alpha beta")) == ["zeta", "alpha", "beta"]


class TestAnchorTokenBoost:
    def test_boosts_content_words_at_the_knee(self):
        w2i = {"gravity": 10, "force": 11, "the": 12}
        boost = anchor_token_boost("Gravity is the force", w2i)
        assert boost == {10: pytest.approx(ANCHOR_TOKEN_BOOST),
                         11: pytest.approx(ANCHOR_TOKEN_BOOST)}
        assert 12 not in boost  # function word

    def test_no_description_yields_none(self):
        assert anchor_token_boost(None, {"a": 1}) is None
        assert anchor_token_boost("", {"a": 1}) is None

    def test_no_vocab_yields_none(self):
        assert anchor_token_boost("gravity", None) is None
        assert anchor_token_boost("gravity", {}) is None

    def test_out_of_vocab_words_are_skipped(self):
        boost = anchor_token_boost("gravity xyzzy", {"gravity": 10})
        assert boost == {10: pytest.approx(ANCHOR_TOKEN_BOOST)}

    def test_all_function_words_yield_none(self):
        """A description made only of closed-class items has no content to
        commit to, so the caller must see None (not an empty dict)."""
        assert anchor_token_boost("the of and", {"the": 1}) is None

    def test_strength_is_overridable(self):
        boost = anchor_token_boost("gravity", {"gravity": 10}, strength=3.0)
        assert boost[10] == pytest.approx(3.0)

    def test_repeated_words_are_not_duplicated(self):
        boost = anchor_token_boost("gravity gravity force", {"gravity": 10,
                                                            "force": 11})
        assert len(boost) == 2
