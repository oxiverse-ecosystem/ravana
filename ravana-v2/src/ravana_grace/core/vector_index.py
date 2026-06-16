"""
SharedVectorIndex — fast ANN index for memory vectors.

Pure-numpy flat cosine for small collections (<10k), optional FAISS for larger.
Shared between HumanMemoryEngine and RLM bridge so all memory retrieval
goes through vector similarity instead of string matching.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


class SharedVectorIndex:
    """ANN index mapping memory_id -> vector with cosine similarity search.

    Design:
    - Vectors stored in a dict, matrix rebuilt lazily on dirty flag.
    - search() uses vectorized cosine (matrix @ query_norm) for <10k vectors.
    - FAISS IndexFlatIP used when available and collection >= 64 vectors.
    - batch_search() for multi-cue reconstruction (one matrix multiply).
    """

    def __init__(self, dim: int = 64, use_faiss: bool = False):
        self.dim = dim
        self._vectors: Dict[int, np.ndarray] = {}  # memory_id -> vector
        self._matrix: Optional[np.ndarray] = None  # (N, dim) float32, row-normalized
        self._id_order: List[int] = []  # aligned with matrix rows
        self._dirty: bool = True
        self._use_faiss = use_faiss
        self._faiss_index = None

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add(self, memory_id: int, vector: np.ndarray) -> None:
        """Add a vector for a memory_id. Overwrites if already present."""
        vec = np.asarray(vector, dtype=np.float32).ravel()[: self.dim]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        self._vectors[memory_id] = vec
        self._dirty = True

    def remove(self, memory_id: int) -> None:
        """Remove a memory from the index. No-op if absent."""
        if memory_id in self._vectors:
            del self._vectors[memory_id]
            self._dirty = True

    def update(self, memory_id: int, vector: np.ndarray) -> None:
        """Update vector for an existing memory. Same as add()."""
        self.add(memory_id, vector)

    def clear(self) -> None:
        """Remove all vectors."""
        self._vectors.clear()
        self._matrix = None
        self._id_order = []
        self._faiss_index = None
        self._dirty = False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """Return up to (memory_id, cosine_score) pairs sorted by score desc."""
        if self._dirty:
            self._rebuild()
        if self._matrix is None or len(self._id_order) == 0:
            return []
        k = min(k, len(self._id_order))

        # FAISS path
        if self._use_faiss and self._faiss_index is not None and len(self._id_order) >= 64:
            vec = np.asarray(query_vector, dtype=np.float32).ravel()[: self.dim]
            vec /= np.linalg.norm(vec) + 1e-15
            vec = vec.reshape(1, -1)
            sims, idxs = self._faiss_index.search(vec, k)
            results = []
            for sim, idx in zip(sims[0], idxs[0]):
                if idx >= 0 and float(sim) >= min_score:
                    results.append((self._id_order[idx], float(sim)))
            return results

        # Brute-force cosine: matrix already row-normalized
        q_norm = np.asarray(query_vector, dtype=np.float32).ravel()[: self.dim]
        q_norm = q_norm / (np.linalg.norm(q_norm) + 1e-15)
        sims = self._matrix @ q_norm  # (N,)

        # Partial sort via argpartition — O(N) average
        if k < len(sims):
            top_idx = np.argpartition(sims, -k)[-k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        else:
            top_idx = np.argsort(sims)[::-1]

        return [
            (self._id_order[i], float(sims[i]))
            for i in top_idx
            if sims[i] >= min_score
        ]

    def batch_search(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        min_score: float = 0.0,
    ) -> List[List[Tuple[int, float]]]:
        """Search multiple cue vectors in one matrix multiply.

        Args:
            query_vectors: (Q, dim) array of Q query vectors.
            k: max results per query.

        Returns:
            List of Q result lists, each sorted by score desc.
        """
        if self._dirty:
            self._rebuild()
        if self._matrix is None or len(self._id_order) == 0:
            return [[] for _ in range(len(query_vectors))]

        Q = np.asarray(query_vectors, dtype=np.float32).reshape(-1, self.dim)
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-15)
        # (Q, N) similarity matrix
        sims = Q @ self._matrix.T  # (Q, N)

        results = []
        for row in sims:
            k_eff = min(k, len(self._id_order))
            if k_eff < len(row):
                top_idx = np.argpartition(row, -k_eff)[-k_eff:]
                top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            else:
                top_idx = np.argsort(row)[::-1]
            results.append(
                [
                    (self._id_order[i], float(row[i]))
                    for i in top_idx
                    if row[i] >= min_score
                ]
            )
        return results

    def get_vector(self, memory_id: int) -> Optional[np.ndarray]:
        """Return the stored vector for a memory, or None."""
        return self._vectors.get(memory_id)

    def has(self, memory_id: int) -> bool:
        return memory_id in self._vectors

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        """Rebuild the normalized matrix and optional FAISS index."""
        if not self._vectors:
            self._matrix = None
            self._id_order = []
            self._faiss_index = None
            self._dirty = False
            return

        self._id_order = list(self._vectors.keys())
        mat = np.stack([self._vectors[mid] for mid in self._id_order])  # (N, dim)
        # Row-normalize (vectors should already be normalized, but ensure)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        self._matrix = (mat / norms).astype(np.float32)

        # FAISS index
        if self._use_faiss and len(self._id_order) >= 64:
            try:
                import faiss

                if len(self._id_order) >= 1000:
                    # HNSW for O(log N) approximate NN at scale
                    self._faiss_index = faiss.IndexHNSWFlat(self.dim, 32)
                    self._faiss_index.hnsw.efSearch = 64
                else:
                    self._faiss_index = faiss.IndexFlatIP(self.dim)
                self._faiss_index.add(self._matrix)
            except ImportError:
                self._faiss_index = None
        else:
            self._faiss_index = None

        self._dirty = False

    def rebuild(self) -> None:
        """Public rebuild trigger."""
        self._dirty = True
        self._rebuild()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save index to disk (numpy .npz + JSON sidecar for id mapping)."""
        if self._dirty:
            self._rebuild()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        base, _ = os.path.splitext(path)
        npz_path = base + ".npz"
        json_path = base + "_ids.json"

        if self._matrix is not None and len(self._id_order) > 0:
            np.savez_compressed(npz_path, matrix=self._matrix, dim=np.array([self.dim]))
        with open(json_path, "w") as f:
            json.dump(self._id_order, f)

    def load(self, path: str) -> None:
        """Load index from disk."""
        base, _ = os.path.splitext(path)
        npz_path = base + ".npz"
        json_path = base + "_ids.json"

        if not os.path.exists(json_path):
            return
        with open(json_path, "r") as f:
            self._id_order = json.load(f)
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            self._matrix = data["matrix"]
            if "dim" in data:
                self.dim = int(data["dim"][0])
            # Rebuild vectors dict from matrix
            self._vectors = {
                mid: self._matrix[i] for i, mid in enumerate(self._id_order)
            }
        self._dirty = False
        # Rebuild FAISS if needed
        if self._use_faiss and self._matrix is not None and len(self._id_order) >= 64:
            try:
                import faiss

                if len(self._id_order) >= 1000:
                    self._faiss_index = faiss.IndexHNSWFlat(self.dim, 32)
                    self._faiss_index.hnsw.efSearch = 64
                else:
                    self._faiss_index = faiss.IndexFlatIP(self.dim)
                self._faiss_index.add(self._matrix)
            except ImportError:
                self._faiss_index = None

    # ------------------------------------------------------------------
    # Dunders
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, memory_id: int) -> bool:
        return memory_id in self._vectors

    def __repr__(self) -> str:
        return f"SharedVectorIndex(n={len(self)}, dim={self.dim}, dirty={self._dirty})"
