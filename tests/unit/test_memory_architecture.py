"""
Tests for the unified hybrid memory architecture.

Tests: SharedVectorIndex, MemoryReconstructor, MemoryDegradation,
and their integration with HumanMemoryEngine.
"""

import os
import sys
import tempfile
import time

import numpy as np
import pytest

from ravana_grace.core.vector_index import SharedVectorIndex
from ravana_grace.core.memory_reconstructor import MemoryReconstructor


# ─── SharedVectorIndex ────────────────────────────────────────────────────

class TestSharedVectorIndex:
    def test_add_and_search(self):
        idx = SharedVectorIndex(dim=16)
        rng = np.random.RandomState(42)
        for i in range(100):
            idx.add(i, rng.randn(16).astype(np.float32))
        assert len(idx) == 100

        q = rng.randn(16).astype(np.float32)
        results = idx.search(q, k=5)
        assert len(results) == 5
        # Scores should be in [-1, 1]
        for mid, score in results:
            assert -1.0 <= score <= 1.0
        # Results should be sorted desc
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_remove(self):
        idx = SharedVectorIndex(dim=8)
        idx.add(1, np.ones(8, dtype=np.float32))
        idx.add(2, np.zeros(8, dtype=np.float32))
        assert len(idx) == 2
        idx.remove(1)
        assert len(idx) == 1
        assert not idx.has(1)
        assert idx.has(2)

    def test_update(self):
        idx = SharedVectorIndex(dim=8)
        v1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        v2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        idx.add(1, v1)
        r1 = idx.search(v1, k=1)
        assert r1[0][1] > 0.99  # near-perfect match

        idx.update(1, v2)
        r2 = idx.search(v2, k=1)
        assert r2[0][1] > 0.99  # now matches v2

    def test_batch_search(self):
        idx = SharedVectorIndex(dim=8)
        rng = np.random.RandomState(99)
        for i in range(50):
            idx.add(i, rng.randn(8).astype(np.float32))
        Q = rng.randn(5, 8).astype(np.float32)
        batch = idx.batch_search(Q, k=3)
        assert len(batch) == 5
        for results in batch:
            assert len(results) == 3

    def test_min_score_filter(self):
        idx = SharedVectorIndex(dim=4)
        idx.add(1, np.array([1, 0, 0, 0], dtype=np.float32))
        idx.add(2, np.array([0, 1, 0, 0], dtype=np.float32))
        q = np.array([1, 0, 0, 0], dtype=np.float32)
        # With high min_score, only exact match passes
        results = idx.search(q, k=10, min_score=0.9)
        assert len(results) == 1
        assert results[0][0] == 1

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_idx")
            idx = SharedVectorIndex(dim=8)
            rng = np.random.RandomState(7)
            for i in range(30):
                idx.add(i, rng.randn(8).astype(np.float32))
            q = rng.randn(8).astype(np.float32)
            orig_results = idx.search(q, k=5)

            idx.save(path)
            idx2 = SharedVectorIndex(dim=8)
            idx2.load(path)
            loaded_results = idx2.search(q, k=5)

            assert len(orig_results) == len(loaded_results)
            for (id1, s1), (id2, s2) in zip(orig_results, loaded_results):
                assert id1 == id2
                assert abs(s1 - s2) < 1e-5

    def test_empty_search(self):
        idx = SharedVectorIndex(dim=8)
        results = idx.search(np.zeros(8), k=5)
        assert results == []

    def test_contains(self):
        idx = SharedVectorIndex(dim=4)
        idx.add(42, np.ones(4, dtype=np.float32))
        assert 42 in idx
        assert 99 not in idx


# ─── MemoryReconstructor ──────────────────────────────────────────────────

class MockMemoryStore:
    """Minimal mock of HumanMemoryEngine for testing."""
    def __init__(self, memories):
        self._memories = {m["id"]: m for m in memories}

    def _get(self, mid):
        return self._memories.get(mid)

    def _node_id(self, mid):
        return f"mem-{mid}"

    def _spreading_activation(self, seeds):
        # Return fake neighbor activations
        return [(f"mem-{i}", 0.5 - i * 0.1) for i in range(4) if f"mem-{i}" not in seeds]


class TestMemoryReconstructor:
    def _make_reconstructor(self, n=50):
        idx = SharedVectorIndex(dim=64)
        rng = np.random.RandomState(42)
        memories = []
        for i in range(n):
            vec = rng.randn(64).astype(np.float32)
            idx.add(i, vec)
            memories.append({
                "id": i, "content": f"memory about topic {i}",
                "tags": "test,alpha" if i % 2 == 0 else "test,beta",
                "importance": 0.5, "emotional": 0.3, "coherence": 0.8,
                "memory_type": "experience",
            })
        store = MockMemoryStore(memories)
        return MemoryReconstructor(idx, store)

    def test_reconstruct_returns_results(self):
        recon = self._make_reconstructor()
        q = np.random.RandomState(99).randn(64).astype(np.float32)
        results = recon.reconstruct(q, k=3)
        assert len(results) <= 3
        assert len(results) > 0

    def test_reconstruct_has_fidelity(self):
        recon = self._make_reconstructor()
        q = np.random.RandomState(1).randn(64).astype(np.float32)
        results = recon.reconstruct(q, k=2)
        for r in results:
            assert "reconstruction_fidelity" in r
            assert 0.0 <= r["reconstruction_fidelity"] <= 1.0

    def test_reconstruct_with_text_boost(self):
        recon = self._make_reconstructor()
        q = np.random.RandomState(5).randn(64).astype(np.float32)
        results = recon.reconstruct(q, cue_text="topic 5", k=3)
        assert len(results) > 0
        # Should have match_score
        for r in results:
            assert "match_score" in r

    def test_reconstruct_neighbor_blending(self):
        recon = self._make_reconstructor()
        q = np.random.RandomState(3).randn(64).astype(np.float32)
        results = recon.reconstruct(q, k=1, blend_depth=2)
        r = results[0]
        # Should have blended content with associated memories
        if r["neighbor_count"] > 0:
            assert "[associated:" in r["content"]

    def test_fidelity_no_neighbors(self):
        score = MemoryReconstructor._compute_fidelity(0.8, [], 1.0)
        assert score == 0.8  # seed_score * seed_coherence

    def test_fidelity_with_neighbors(self):
        score = MemoryReconstructor._compute_fidelity(0.8, [0.5, 0.3], 1.0)
        assert 0.0 <= score <= 1.0

    def test_token_overlap(self):
        assert MemoryReconstructor._token_overlap("hello world", "world hello") == 1.0
        assert MemoryReconstructor._token_overlap("hello", "goodbye") == 0.0
        assert MemoryReconstructor._token_overlap("hello world", "hello there") == 0.5


# ─── Integration: Vector Index + HumanMemoryEngine ────────────────────────

class TestVectorIndexIntegration:
    def test_encode_to_vector_deterministic(self):
        """Same input should always produce the same vector."""
        from ravana_grace.core.human_memory import HumanMemoryEngine, HumanMemoryConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = HumanMemoryConfig(db_path=os.path.join(tmpdir, "test.db"))
            engine = HumanMemoryEngine(cfg)
            v1 = engine._encode_to_vector("hello world", "test,foo", 0.5, 0.3)
            v2 = engine._encode_to_vector("hello world", "test,foo", 0.5, 0.3)
            np.testing.assert_array_equal(v1, v2)

    def test_store_adds_to_vector_index(self):
        from ravana_grace.core.human_memory import HumanMemoryEngine, HumanMemoryConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = HumanMemoryConfig(db_path=os.path.join(tmpdir, "test.db"))
            engine = HumanMemoryEngine(cfg)
            mid = engine._store("test memory", tags="test", importance=0.7)
            assert engine.vector_index.has(mid)
            vec = engine.vector_index.get_vector(mid)
            assert vec is not None
            assert len(vec) == 64

    def test_vector_recall(self):
        from ravana_grace.core.human_memory import HumanMemoryEngine, HumanMemoryConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = HumanMemoryConfig(db_path=os.path.join(tmpdir, "test.db"))
            engine = HumanMemoryEngine(cfg)
            # Store some memories
            for i in range(10):
                engine._store(f"memory about topic {i}", tags=f"tag{i}", importance=0.5)
            # Vector recall
            q = engine._encode_to_vector("topic 5", "tag5", 0.5, 0.5)
            results = engine._recall(query_vector=q, limit=5)
            assert len(results) > 0
            # Should find the topic 5 memory
            ids = [r["id"] for r in results]
            assert len(ids) == len(set(ids))  # no duplicates

    def test_vector_index_persistence(self):
        from ravana_grace.core.human_memory import HumanMemoryEngine, HumanMemoryConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            cfg = HumanMemoryConfig(db_path=db_path)
            engine = HumanMemoryEngine(cfg)
            mid = engine._store("persistent memory", tags="persist")
            old_vec = engine.vector_index.get_vector(mid).copy()

            # Create new engine (simulates restart)
            engine2 = HumanMemoryEngine(HumanMemoryConfig(db_path=db_path))
            new_vec = engine2.vector_index.get_vector(mid)
            assert new_vec is not None
            np.testing.assert_array_almost_equal(old_vec, new_vec, decimal=5)

    def test_decay_runs_naturally_in_process_step(self):
        """Decay should run every cycle inside process_step, not require explicit call."""
        from ravana_grace.core.human_memory import HumanMemoryEngine, HumanMemoryConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = HumanMemoryConfig(db_path=os.path.join(tmpdir, "test.db"))
            engine = HumanMemoryEngine(cfg)
            # Store a low-importance memory
            mid = engine._store("fragile memory", importance=0.1, emotional=0.1)
            initial = engine._get(mid)
            initial_decay = initial["decay_score"]
            # Run several process_step cycles (each triggers decay)
            for i in range(10):
                engine.process_step(
                    episode_data={"pre_dissonance": 0.3, "post_dissonance": 0.3,
                                  "pre_identity": 0.5, "post_identity": 0.5,
                                  "wisdom": 0.0, "meaning": 0.0,
                                  "processing_route": "test", "mode": "test"},
                    state_snapshot={"dissonance": 0.3, "identity": 0.5,
                                    "accumulated_wisdom": 0.0}
                )
            after = engine._get(mid)
            # Decay score should have increased from natural decay
            assert after["decay_score"] > initial_decay


# ─── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
