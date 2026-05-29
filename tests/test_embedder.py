"""Tests for the LearnedEmbedder — character n-gram + random projection."""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_RAVANA_V2 = os.path.join(_PROJECT_ROOT, "ravana-v2")
if _RAVANA_V2 not in sys.path:
    sys.path.insert(0, _RAVANA_V2)

import numpy as np
import pytest

from core.embedder import LearnedEmbedder


class TestCharNgrams:
    """Test character n-gram extraction."""

    def test_basic_ngrams(self):
        ngrams = LearnedEmbedder._char_ngrams("hello", (3,))
        assert "hel" in ngrams
        assert "ell" in ngrams
        assert "llo" in ngrams
        assert len(ngrams) == 3

    def test_multiple_sizes(self):
        ngrams = LearnedEmbedder._char_ngrams("hello", (3, 4))
        assert "hel" in ngrams
        assert "hell" in ngrams
        assert len(ngrams) == 3 + 2  # 3 trigrams + 2 four-grams

    def test_short_text(self):
        ngrams = LearnedEmbedder._char_ngrams("hi", (3,))
        assert ngrams == []  # too short for trigrams

    def test_whitespace_collapsed(self):
        ngrams = LearnedEmbedder._char_ngrams("he  llo", (3,))
        assert "he " in ngrams or "e l" in ngrams  # whitespace treated as char

    def test_case_insensitive(self):
        ngrams_upper = LearnedEmbedder._char_ngrams("HELLO", (3,))
        ngrams_lower = LearnedEmbedder._char_ngrams("hello", (3,))
        assert set(ngrams_upper) == set(ngrams_lower)


class TestFeatureHash:
    """Test feature hashing of n-grams."""

    def test_output_shape(self):
        embedder = LearnedEmbedder(hash_dim=128)
        ngrams = LearnedEmbedder._char_ngrams("hello world", (3, 4, 5))
        vec = embedder._feature_hash(ngrams)
        assert vec.shape == (128,)

    def test_deterministic(self):
        embedder = LearnedEmbedder()
        ngrams = LearnedEmbedder._char_ngrams("test text", (3, 4, 5))
        v1 = embedder._feature_hash(ngrams)
        v2 = embedder._feature_hash(ngrams)
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_differ(self):
        embedder = LearnedEmbedder()
        ngrams1 = LearnedEmbedder._char_ngrams("the cat sat", (3, 4, 5))
        ngrams2 = LearnedEmbedder._char_ngrams("quantum physics", (3, 4, 5))
        v1 = embedder._feature_hash(ngrams1)
        v2 = embedder._feature_hash(ngrams2)
        assert not np.array_equal(v1, v2)


class TestEncode:
    """Test the full encoding pipeline."""

    def test_output_shape(self):
        embedder = LearnedEmbedder(dim=64)
        vec = embedder.encode("hello world")
        assert vec.shape == (64,)
        assert vec.dtype == np.float32

    def test_normalized(self):
        embedder = LearnedEmbedder(dim=64)
        vec = embedder.encode("some text for testing")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_deterministic(self):
        embedder = LearnedEmbedder(dim=64)
        v1 = embedder.encode("deterministic test")
        v2 = embedder.encode("deterministic test")
        np.testing.assert_array_equal(v1, v2)

    def test_empty_text(self):
        embedder = LearnedEmbedder(dim=64)
        vec = embedder.encode("", importance=0.7, emotional=0.3)
        assert vec.shape == (64,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_salience_signals(self):
        embedder = LearnedEmbedder(dim=64)
        vec_low = embedder.encode("test", importance=0.1, emotional=0.1)
        vec_high = embedder.encode("test", importance=0.9, emotional=0.9)
        # Different importance/emotional should produce different vectors
        assert not np.allclose(vec_low, vec_high, atol=1e-6)

    def test_tags_included(self):
        embedder = LearnedEmbedder(dim=64)
        v1 = embedder.encode("hello", tags="python")
        v2 = embedder.encode("hello", tags="javascript")
        assert not np.allclose(v1, v2, atol=1e-6)

    def test_semantic_similarity(self):
        """Similar texts should be more similar than dissimilar texts."""
        embedder = LearnedEmbedder(dim=64)
        v_cat = embedder.encode("the cat sat on the mat")
        v_dog = embedder.encode("the dog sat on the rug")
        v_phys = embedder.encode("quantum mechanics and relativity")
        # cat/dog should be more similar than cat/physics
        sim_cat_dog = np.dot(v_cat, v_dog)
        sim_cat_phys = np.dot(v_cat, v_phys)
        assert sim_cat_dog > sim_cat_phys, (
            f"Expected cat-dog ({sim_cat_dog:.3f}) > cat-physics ({sim_cat_phys:.3f})"
        )


class TestIDF:
    """Test IDF fitting and weighting."""

    def test_fit_sets_idf(self):
        embedder = LearnedEmbedder()
        assert not embedder.is_fitted
        embedder.fit(["hello world", "world peace", "hello peace"])
        assert embedder.is_fitted

    def test_idf_changes_encoding(self):
        embedder_unfitted = LearnedEmbedder(dim=64, seed=42)
        embedder_fitted = LearnedEmbedder(dim=64, seed=42)
        # "quantum" is rare, "the cat" is frequent — IDF should reweight
        embedder_fitted.fit(["the cat sat on the mat"] * 10 + ["quantum physics"])
        # Encode text with a mix of common and rare n-grams
        v1 = embedder_unfitted.encode("the quantum cat")
        v2 = embedder_fitted.encode("the quantum cat")
        # IDF weighting should change the encoding
        assert not np.allclose(v1, v2, atol=1e-6)

    def test_rare_ngrams_weighted_higher(self):
        """After fitting, rare n-grams should get higher IDF weights."""
        embedder = LearnedEmbedder()
        # "quantum" appears once, "the" appears in every doc
        corpus = [
            "the cat sat on the mat",
            "the dog ran in the park",
            "the bird flew over the tree",
            "quantum mechanics is weird",
        ]
        embedder.fit(corpus)
        # Check that n-grams from "quantum" have higher IDF than "the"
        q_ngrams = LearnedEmbedder._char_ngrams("quantum", (3, 4, 5))
        t_ngrams = LearnedEmbedder._char_ngrams("the", (3,))
        q_weights = [embedder._idf.get(ng, 0) for ng in q_ngrams if ng in embedder._idf]
        t_weights = [embedder._idf.get(ng, 0) for ng in t_ngrams if ng in embedder._idf]
        if q_weights and t_weights:
            assert np.mean(q_weights) > np.mean(t_weights)


class TestSentenceTransformerEmbedder:
    """Test the optional sentence-transformers embedder."""

    def test_fallback_when_unavailable(self):
        from core.embedder import SentenceTransformerEmbedder
        # sentence-transformers likely not installed in test env
        st = SentenceTransformerEmbedder(dim=64)
        # If not available, encode should raise
        if not st.available:
            with pytest.raises(RuntimeError):
                st.encode("test")
