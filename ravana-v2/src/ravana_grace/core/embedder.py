"""
LearnedEmbedder — character n-gram + random projection embeddings.

Replaces hash-based encoding with a semantically richer approach:
1. Character n-grams (3,4,5) capture subword/morphological patterns
2. Feature hashing maps n-grams to a fixed sparse vector
3. Random projection reduces dimensionality (Johnson-Lindenstrauss)
4. Optional IDF weighting from corpus statistics

Zero new dependencies — uses only numpy and hashlib.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, List, Optional, Set

import numpy as np


class LearnedEmbedder:
    """Character n-gram + random projection embedder.

    Produces 64-dim vectors that capture subword structure,
    unlike hash-based encodings that are semantically blind.
    """

    def __init__(
        self,
        dim: int = 64,
        ngram_sizes: tuple = (3, 4, 5),
        hash_dim: int = 256,
        seed: int = 42,
    ):
        self.dim = dim
        self.ngram_sizes = ngram_sizes
        self.hash_dim = hash_dim
        self._rng = np.random.RandomState(seed)
        # Fixed random projection matrix (hash_dim -> dim)
        self._projection = self._rng.randn(hash_dim, dim).astype(np.float32)
        self._projection /= np.sqrt(dim)  # scale to preserve norms
        # IDF weights (learned from corpus)
        self._idf: Optional[Dict[str, float]] = None
        self._corpus_size: int = 0

    @staticmethod
    def _char_ngrams(text: str, ns: tuple = (3, 4, 5)) -> List[str]:
        """Extract character n-grams from text."""
        text = text.lower().strip()
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        ngrams = []
        for n in ns:
            for i in range(len(text) - n + 1):
                ngrams.append(text[i : i + n])
        return ngrams

    def _feature_hash(self, ngrams: List[str]) -> np.ndarray:
        """Hash n-grams into a fixed-size sparse vector using feature hashing."""
        vec = np.zeros(self.hash_dim, dtype=np.float32)
        for ng in ngrams:
            h = int(hashlib.md5(ng.encode("utf-8")).hexdigest(), 16)
            pos = h % self.hash_dim
            sign = 1.0 if (h // self.hash_dim) % 2 == 0 else -1.0
            weight = 1.0
            if self._idf is not None:
                # Use IDF weight for this n-gram if available
                weight = self._idf.get(ng, self._idf.get("<unk>", 1.0))
            vec[pos] += sign * weight
        return vec

    def encode(self, text: str, tags: str = "",
               importance: float = 0.5, emotional: float = 0.5) -> np.ndarray:
        """Encode text into a dim-dimensional embedding vector.

        Args:
            text: Primary text content.
            tags: Additional tags to include in encoding.
            importance: Importance signal [0, 1].
            emotional: Emotional salience signal [0, 1].

        Returns:
            L2-normalized float32 vector of shape (dim,).
        """
        combined = (text + " " + tags).strip()
        if not combined:
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[0] = importance
            vec[1] = emotional
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            return vec

        ngrams = self._char_ngrams(combined, self.ngram_sizes)
        sparse = self._feature_hash(ngrams)
        # Random projection: hash_dim -> dim
        dense = sparse @ self._projection  # (dim,)
        # L2 normalize
        norm = np.linalg.norm(dense)
        if norm > 0:
            dense /= norm
        # Blend salience signals (same positions as original hash encoder)
        dense[0] += importance * 0.1
        dense[1] += emotional * 0.1
        # Re-normalize
        norm = np.linalg.norm(dense)
        if norm > 0:
            dense /= norm
        return dense

    def fit(self, corpus: List[str]) -> None:
        """Learn IDF weights from a corpus of texts.

        Args:
            corpus: List of text strings to learn IDF from.
        """
        self._corpus_size = len(corpus)
        doc_freq: Dict[str, int] = {}
        for text in corpus:
            ngrams = set(self._char_ngrams(text, self.ngram_sizes))
            for ng in ngrams:
                doc_freq[ng] = doc_freq.get(ng, 0) + 1
        # IDF = log(N / df), with smoothing
        self._idf = {}
        for ng, df in doc_freq.items():
            self._idf[ng] = math.log((self._corpus_size + 1) / (df + 1)) + 1.0
        # Unknown n-gram weight (for n-grams not in corpus)
        self._idf["<unk>"] = math.log((self._corpus_size + 1) / 1) + 1.0

    @property
    def is_fitted(self) -> bool:
        """Whether IDF weights have been learned."""
        return self._idf is not None


class SentenceTransformerEmbedder:
    """Optional high-quality embedder using sentence-transformers.

    Falls back to LearnedEmbedder if sentence-transformers is not installed.
    """

    def __init__(self, dim: int = 64, model_name: str = "all-MiniLM-L6-v2",
                 seed: int = 42):
        self.dim = dim
        self._model = None
        self._projection = None
        self._seed = seed
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            model_dim = self._model.get_sentence_embedding_dimension()
            rng = np.random.RandomState(seed)
            self._projection = rng.randn(model_dim, dim).astype(np.float32)
            self._projection /= np.sqrt(dim)
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._model is not None

    def encode(self, text: str, tags: str = "",
               importance: float = 0.5, emotional: float = 0.5) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("sentence-transformers not installed")
        combined = (text + " " + tags).strip()
        if not combined:
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[0] = importance
            vec[1] = emotional
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            return vec
        emb = self._model.encode([combined], convert_to_numpy=True)[0]
        vec = emb @ self._projection
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vec[0] += importance * 0.1
        vec[1] += emotional * 0.1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)
