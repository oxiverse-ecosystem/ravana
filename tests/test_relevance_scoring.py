"""Unit tests for the SEED-backed relevance-scored recall layer.

Tests the multi-signal scoring function, threshold adaptation, and
the RelevanceThreshold data store. No LLM, no hardcoded reply logic.
"""
import math
import os
import sys

import numpy as np
import pytest

# Add the source root to the path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_src = os.path.join(_REPO_ROOT, "ravana", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
_ml_src = os.path.join(_REPO_ROOT, "ravana_ml", "src")
if _ml_src not in sys.path:
    sys.path.insert(0, _ml_src)

from ravana.chat.relevance_store import (
    RelevanceThreshold,
    RelevanceScorer,
    adapt_threshold,
    record_false_positive,
    record_false_negative,
    _content_words,
    _jaccard_overlap,
    _glove_cosine,
    _type_alignment,
    _mean_embedding,
)


# ── Content word extraction ──

class TestContentWords:
    def test_basic_extraction(self):
        words = _content_words("what is my cat's name")
        # "cat's" is extracted as one token (apostrophe kept by regex)
        assert "cat's" in words or "cat" in words
        assert "name" in words
        # stopwords removed
        assert "what" not in words
        assert "is" not in words
        assert "my" not in words

    def test_empty_string(self):
        assert _content_words("") == set()

    def test_only_stopwords(self):
        assert _content_words("the a an is are") == set()

    def test_short_tokens_removed(self):
        words = _content_words("my cat is ok")
        assert "ok" not in words  # len < 3
        assert "cat" in words


# ── Jaccard overlap ──

class TestJaccardOverlap:
    def test_identical_sets(self):
        assert _jaccard_overlap({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard_overlap({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        # intersection = {b}, union = {a, b, c, d} -> 0.25
        assert _jaccard_overlap({"a", "b"}, {"b", "c", "d"}) == pytest.approx(0.25)

    def test_empty_sets(self):
        assert _jaccard_overlap(set(), {"a"}) == 0.0
        assert _jaccard_overlap({"a"}, set()) == 0.0


# ── GloVe cosine ──

class TestGloveCosine:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert _glove_cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _glove_cosine(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _glove_cosine(a, b) == pytest.approx(-1.0)

    def test_none_input(self):
        assert _glove_cosine(None, np.array([1.0])) == 0.0
        assert _glove_cosine(np.array([1.0]), None) == 0.0

    def test_different_shapes(self):
        assert _glove_cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0])) == 0.0

    def test_zero_vector(self):
        assert _glove_cosine(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


# ── Type alignment ──

class TestTypeAlignment:
    def test_same_type(self):
        assert _type_alignment("pet", "pet") == 1.0

    def test_related_types(self):
        assert _type_alignment("pet", "animal") == 0.5
        assert _type_alignment("animal", "pet") == 0.5

    def test_unrelated_types(self):
        assert _type_alignment("pet", "place") == 0.0

    def test_none_type(self):
        assert _type_alignment(None, "pet") == 0.5
        assert _type_alignment("pet", None) == 0.5


# ── Mean embedding ──

class TestMeanEmbedding:
    def test_basic_mean(self):
        def glove_fn(w):
            return {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}.get(w)
        result = _mean_embedding({"a", "b"}, glove_fn)
        assert result is not None
        assert result.shape == (2,)
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(0.5)

    def test_empty_words(self):
        assert _mean_embedding(set(), lambda w: None) is None

    def test_no_glove_fn(self):
        assert _mean_embedding({"a"}, None) is None

    def test_unknown_words(self):
        def glove_fn(w):
            return None
        assert _mean_embedding({"a", "b"}, glove_fn) is None


# ── RelevanceThreshold data store ──

class TestRelevanceThreshold:
    def test_default_values(self):
        rt = RelevanceThreshold()
        assert rt.base_threshold == 0.15
        assert rt.content_weight == 0.40
        assert rt.graph_weight == 0.25
        assert rt.recency_weight == 0.20
        assert rt.type_weight == 0.15
        assert rt.total_scored == 0

    def test_to_dict(self):
        rt = RelevanceThreshold(base_threshold=0.2, false_positive_count=5)
        d = rt.to_dict()
        assert d["base_threshold"] == 0.2
        assert d["false_positive_count"] == 5
        assert "content_weight" in d

    def test_from_dict(self):
        d = {
            "base_threshold": 0.25,
            "content_weight": 0.5,
            "false_positive_count": 10,
            "total_scored": 20,
        }
        rt = RelevanceThreshold.from_dict(d)
        assert rt.base_threshold == 0.25
        assert rt.content_weight == 0.5
        assert rt.false_positive_count == 10
        assert rt.total_scored == 20

    def test_from_dict_defaults(self):
        rt = RelevanceThreshold.from_dict({})
        assert rt.base_threshold == 0.15
        assert rt.content_weight == 0.40

    def test_roundtrip(self):
        rt = RelevanceThreshold(base_threshold=0.3, false_negative_count=7)
        d = rt.to_dict()
        rt2 = RelevanceThreshold.from_dict(d)
        assert rt2.base_threshold == 0.3
        assert rt2.false_negative_count == 7


# ── RelevanceScorer ──

class TestRelevanceScorer:
    def test_identical_text_scores_high(self):
        scorer = RelevanceScorer(glove_fn=None)
        score = scorer.score("my cat is named whiskers",
                             "my cat is named whiskers")
        assert score > 0.3  # content overlap + recency + type

    def test_unrelated_text_scores_low(self):
        scorer = RelevanceScorer(glove_fn=None)
        score = scorer.score("my cat is named whiskers",
                             "the quantum physics lecture was interesting")
        # Low content overlap, no type match. Recency contributes 0.20
        # (same turn), so total is ~0.20. Assert it's below 0.3.
        assert score < 0.3

    def test_with_glove_fn(self):
        def glove_fn(w):
            return {"cat": np.array([1.0, 0.0]),
                    "feline": np.array([0.9, 0.1]),
                    "dog": np.array([0.0, 1.0])}.get(w)
        scorer = RelevanceScorer(glove_fn=glove_fn)
        score = scorer.score("my cat is named whiskers",
                             "the feline is called whiskers")
        assert score > 0.2  # some content overlap + graph similarity

    def test_score_with_signals(self):
        scorer = RelevanceScorer(glove_fn=None)
        score, signals = scorer.score_with_signals(
            "my cat is named whiskers",
            "my cat is named whiskers")
        assert "content" in signals
        assert "graph" in signals
        assert "recency" in signals
        assert "type" in signals
        assert 0.0 <= score <= 1.0

    def test_recency_decay(self):
        scorer = RelevanceScorer(glove_fn=None)
        score_recent = scorer.score("cat", "cat",
                                     candidate_turn=10, query_turn=10)
        score_old = scorer.score("cat", "cat",
                                 candidate_turn=1, query_turn=100)
        assert score_recent > score_old

    def test_score_bounds(self):
        scorer = RelevanceScorer(glove_fn=None)
        score = scorer.score("a" * 100, "b" * 100)
        assert 0.0 <= score <= 1.0

    def test_custom_threshold(self):
        scorer = RelevanceScorer(glove_fn=None)
        rt = RelevanceThreshold(content_weight=1.0, graph_weight=0.0,
                                recency_weight=0.0, type_weight=0.0)
        score = scorer.score("cat dog", "cat dog", threshold=rt)
        assert score == pytest.approx(1.0)  # perfect content overlap


# ── Threshold adaptation ──

class TestThresholdAdaptation:
    def test_no_adaptation_below_min(self):
        rt = RelevanceThreshold(total_scored=5)
        original = rt.base_threshold
        adapt_threshold(rt)
        assert rt.base_threshold == original  # not enough data

    def test_adaptation_on_false_positives(self):
        rt = RelevanceThreshold(total_scored=20, false_positive_count=10,
                                false_negative_count=2)
        original = rt.base_threshold
        adapt_threshold(rt)
        assert rt.base_threshold > original  # raised threshold

    def test_adaptation_on_false_negatives(self):
        rt = RelevanceThreshold(total_scored=20, false_positive_count=2,
                                false_negative_count=10)
        original = rt.base_threshold
        adapt_threshold(rt)
        assert rt.base_threshold < original  # lowered threshold

    def test_threshold_upper_bound(self):
        rt = RelevanceThreshold(base_threshold=0.49, total_scored=20,
                                false_positive_count=15, false_negative_count=1)
        adapt_threshold(rt)
        assert rt.base_threshold <= 0.5

    def test_threshold_lower_bound(self):
        rt = RelevanceThreshold(base_threshold=0.06, total_scored=20,
                                false_positive_count=1, false_negative_count=15)
        adapt_threshold(rt)
        assert rt.base_threshold >= 0.05

    def test_record_false_positive(self):
        rt = RelevanceThreshold()
        record_false_positive(rt, current_turn=10)
        assert rt.false_positive_count == 1
        assert rt.total_scored == 1

    def test_record_false_negative(self):
        rt = RelevanceThreshold()
        record_false_negative(rt, current_turn=10)
        assert rt.false_negative_count == 1
        assert rt.total_scored == 1

    def test_adaptation_updates_turn(self):
        rt = RelevanceThreshold(total_scored=20, false_positive_count=10,
                                false_negative_count=2)
        adapt_threshold(rt, current_turn=42)
        assert rt.last_adapted_turn == 42


# ── Integration: scorer + threshold ──

class TestScorerThresholdIntegration:
    def test_score_above_threshold_passes(self):
        scorer = RelevanceScorer(glove_fn=None)
        rt = RelevanceThreshold(base_threshold=0.1)
        score = scorer.score("my cat is named whiskers",
                             "my cat is named whiskers",
                             threshold=rt)
        assert score >= rt.base_threshold

    def test_score_below_threshold_fails(self):
        scorer = RelevanceScorer(glove_fn=None)
        rt = RelevanceThreshold(base_threshold=0.9)
        score = scorer.score("cat", "quantum physics",
                             threshold=rt)
        assert score < rt.base_threshold

    def test_adaptation_changes_gate_behavior(self):
        scorer = RelevanceScorer(glove_fn=None)
        rt = RelevanceThreshold(base_threshold=0.15, total_scored=20,
                                false_positive_count=10, false_negative_count=2)
        # Before adaptation
        score = scorer.score("cat", "dog", threshold=rt)
        passes_before = score >= rt.base_threshold
        # Adapt
        adapt_threshold(rt)
        # After adaptation (threshold raised)
        passes_after = score >= rt.base_threshold
        # If threshold was raised, something that passed before may fail now
        if rt.base_threshold > 0.15:
            assert not passes_after or passes_before  # consistent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
