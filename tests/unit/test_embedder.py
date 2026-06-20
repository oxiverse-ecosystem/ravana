"""Tests for ravana_grace.core.embedder."""

import pytest
import numpy as np
from ravana_grace.core.embedder import LearnedEmbedder


class TestLearnedEmbedder:
    def test_init(self):
        e = LearnedEmbedder(dim=64)
        assert e.dim == 64
        assert e.is_fitted is False

    def test_encode_empty_text(self):
        e = LearnedEmbedder(dim=8)
        vec = e.encode("")
        assert vec.shape == (8,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-4

    def test_encode_text(self):
        e = LearnedEmbedder(dim=8)
        vec = e.encode("hello world")
        assert vec.shape == (8,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-4

    def test_encode_with_tags(self):
        e = LearnedEmbedder(dim=8)
        vec = e.encode("hello", tags="greeting")
        assert vec.shape == (8,)

    def test_encode_with_importance(self):
        e = LearnedEmbedder(dim=8)
        vec_high = e.encode("test", importance=0.9)
        vec_low = e.encode("test", importance=0.1)
        # Different importance should produce different vectors
        assert not np.allclose(vec_high, vec_low)

    def test_deterministic_encoding(self):
        e = LearnedEmbedder(dim=64, seed=42)
        v1 = e.encode("hello world")
        v2 = e.encode("hello world")
        assert np.allclose(v1, v2)

    def test_fit(self):
        e = LearnedEmbedder(dim=8)
        corpus = ["hello world", "goodbye world", "hello universe"]
        e.fit(corpus)
        assert e.is_fitted is True

    def test_fit_changes_encoding(self):
        e = LearnedEmbedder(dim=8)
        v_before = e.encode("hello world")
        corpus = ["hello world", "goodbye world", "hello universe"]
        e.fit(corpus)
        v_after = e.encode("hello world")
        # Fitted encoding should differ
        assert not np.allclose(v_before, v_after)

    def test_char_ngrams(self):
        ngrams = LearnedEmbedder._char_ngrams("hi", ns=(3, 4))
        assert len(ngrams) == 0  # "hi" is too short for trigrams

    def test_char_ngrams_longer(self):
        ngrams = LearnedEmbedder._char_ngrams("hello", ns=(3, 4))
        assert len(ngrams) == 3 + 2  # 3 trigrams + 2 4-grams

    def test_fit_twice(self):
        e = LearnedEmbedder(dim=8)
        e.fit(["a", "b"])
        e.fit(["a", "b", "c"])
        assert e.is_fitted is True
        assert e._corpus_size == 3


class TestSentenceTransformerEmbedder:
    def test_init_fallback(self):
        from ravana_grace.core.embedder import SentenceTransformerEmbedder
        st = SentenceTransformerEmbedder(dim=64)
        if not st.available:
            pytest.skip("sentence-transformers not installed")
        assert st.available is True

    def test_encode_not_available(self):
        from ravana_grace.core.embedder import SentenceTransformerEmbedder
        st = SentenceTransformerEmbedder(dim=64)
        if not st.available:
            with pytest.raises(RuntimeError, match="sentence-transformers"):
                st.encode("test")
