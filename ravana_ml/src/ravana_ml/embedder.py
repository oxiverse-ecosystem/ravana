"""
LearnedEmbedder — character n-gram + random projection embeddings.

Produces 64-dim vectors that capture subword structure, unlike hash-based
encodings that are semantically blind. Zero new dependencies.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

import numpy as np


class LearnedEmbedder:
    """Character n-gram + random projection embedder."""

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
        self._projection = self._rng.randn(hash_dim, dim).astype(np.float32)
        self._projection /= np.sqrt(dim)
        self._idf: Optional[Dict[str, float]] = None
        self._corpus_size: int = 0

    @staticmethod
    def _char_ngrams(text: str, ns: tuple = (3, 4, 5)) -> List[str]:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        ngrams = []
        for n in ns:
            for i in range(len(text) - n + 1):
                ngrams.append(text[i : i + n])
        return ngrams

    def _feature_hash(self, ngrams: List[str]) -> np.ndarray:
        vec = np.zeros(self.hash_dim, dtype=np.float32)
        for ng in ngrams:
            h = int(hashlib.md5(ng.encode("utf-8")).hexdigest(), 16)
            pos = h % self.hash_dim
            sign = 1.0 if (h // self.hash_dim) % 2 == 0 else -1.0
            weight = 1.0
            if self._idf is not None:
                weight = self._idf.get(ng, self._idf.get("<unk>", 1.0))
            vec[pos] += sign * weight
        return vec

    def encode(self, text: str, tags: str = "",
               importance: float = 0.5, emotional: float = 0.5) -> np.ndarray:
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
        dense = sparse @ self._projection
        norm = np.linalg.norm(dense)
        if norm > 0:
            dense /= norm
        dense[0] += importance * 0.1
        dense[1] += emotional * 0.1
        norm = np.linalg.norm(dense)
        if norm > 0:
            dense /= norm
        return dense.astype(np.float32)
