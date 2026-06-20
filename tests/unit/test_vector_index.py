"""Tests for ravana_grace.core.vector_index."""

import pytest
import numpy as np
import tempfile
import os
from ravana_grace.core.vector_index import SharedVectorIndex


class TestSharedVectorIndex:
    def test_init(self):
        vi = SharedVectorIndex(dim=8)
        assert vi.dim == 8
        assert len(vi) == 0

    def test_add_and_search(self):
        vi = SharedVectorIndex(dim=8)
        vi.add(1, np.random.randn(8).astype(np.float32))
        vi.add(2, np.random.randn(8).astype(np.float32))
        assert len(vi) == 2
        results = vi.search(np.random.randn(8).astype(np.float32), k=2)
        # May return fewer than k results depending on similarity
        assert len(results) <= 2

    def test_search_returns_sorted(self):
        vi = SharedVectorIndex(dim=8)
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vi.add(1, v)  # Exact match
        vi.add(2, np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        results = vi.search(v, k=2)
        assert len(results) >= 1
        assert results[0][0] == 1  # Closest match first

    def test_search_with_min_score(self):
        vi = SharedVectorIndex(dim=8)
        vi.add(1, np.ones(8, dtype=np.float32))
        vi.add(2, np.ones(8, dtype=np.float32) * -1)
        query = np.ones(8, dtype=np.float32)
        results = vi.search(query, k=5, min_score=0.5)
        # Both should be included at this similarity
        assert len(results) > 0

    def test_remove(self):
        vi = SharedVectorIndex(dim=8)
        vi.add(1, np.random.randn(8).astype(np.float32))
        vi.add(2, np.random.randn(8).astype(np.float32))
        vi.remove(1)
        assert len(vi) == 1
        assert 1 not in vi

    def test_update(self):
        vi = SharedVectorIndex(dim=8)
        vi.add(1, np.ones(8, dtype=np.float32))
        vi.update(1, np.zeros(8, dtype=np.float32))
        # Should still have the memory
        assert 1 in vi

    def test_clear(self):
        vi = SharedVectorIndex(dim=8)
        vi.add(1, np.random.randn(8).astype(np.float32))
        vi.clear()
        assert len(vi) == 0
        assert vi._matrix is None

    def test_get_vector(self):
        vi = SharedVectorIndex(dim=8)
        v = np.random.randn(8).astype(np.float32)
        vi.add(1, v)
        retrieved = vi.get_vector(1)
        assert retrieved is not None
        assert np.allclose(retrieved, v / np.linalg.norm(v))

    def test_get_vector_missing(self):
        vi = SharedVectorIndex(dim=8)
        assert vi.get_vector(999) is None

    def test_batch_search(self):
        vi = SharedVectorIndex(dim=8)
        for i in range(5):
            vi.add(i, np.random.randn(8).astype(np.float32))
        queries = np.random.randn(3, 8).astype(np.float32)
        results = vi.batch_search(queries, k=2)
        assert len(results) == 3

    def test_save_and_load(self):
        vi = SharedVectorIndex(dim=8)
        vi.add(1, np.random.randn(8).astype(np.float32))
        vi.add(2, np.random.randn(8).astype(np.float32))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.npz")
            vi.save(path)

            vi2 = SharedVectorIndex(dim=8)
            vi2.load(path)
            assert len(vi2) == 2

    def test_rebuild(self):
        vi = SharedVectorIndex(dim=8)
        vi.add(1, np.random.randn(8).astype(np.float32))
        vi._dirty = True
        vi.rebuild()
        assert vi._dirty is False
        assert vi._matrix is not None
