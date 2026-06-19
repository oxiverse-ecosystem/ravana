"""Tests for ravana_grace.core.memory_reconstructor."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from ravana_grace.core.memory_reconstructor import MemoryReconstructor


class TestMemoryReconstructor:
    def test_init(self):
        index = MagicMock()
        store = MagicMock()
        mr = MemoryReconstructor(index, store)
        assert mr.index is index
        assert mr.store is store

    def test_reconstruct_no_candidates(self):
        index = MagicMock()
        index.search.return_value = []
        store = MagicMock()
        mr = MemoryReconstructor(index, store)
        results = mr.reconstruct(np.random.randn(8).astype(np.float32))
        assert results == []

    def test_reconstruct_with_candidates(self):
        index = MagicMock()
        index.search.return_value = [(1, 0.8)]
        store = MagicMock()
        store._get.return_value = {
            "id": 1, "content": "test memory", "tags": "test",
            "importance": 0.5, "emotional": 0.5, "coherence": 1.0,
        }
        store._node_id.return_value = "1"
        store._spreading_activation.return_value = []
        mr = MemoryReconstructor(index, store)
        results = mr.reconstruct(np.random.randn(8).astype(np.float32), k=3)
        assert len(results) > 0

    def test_reconstruct_filters_by_memory_type(self):
        index = MagicMock()
        index.search.return_value = [(1, 0.8), (2, 0.7)]
        store = MagicMock()
        def _get_side_effect(mid):
            if mid == 1:
                return {"id": 1, "content": "episodic", "tags": "",
                        "importance": 0.5, "emotional": 0.5, "coherence": 1.0,
                        "memory_type": "episodic"}
            return None
        store._get.side_effect = _get_side_effect
        store._node_id.return_value = "1"
        store._spreading_activation.return_value = []
        mr = MemoryReconstructor(index, store)
        results = mr.reconstruct(np.random.randn(8).astype(np.float32), k=3, memory_type="semantic")
        assert results == []  # No semantic matches

    def test_token_overlap(self):
        score = MemoryReconstructor._token_overlap("hello world", "hello universe")
        assert score > 0  # "hello" overlaps

    def test_token_overlap_no_overlap(self):
        score = MemoryReconstructor._token_overlap("hello", "goodbye")
        assert score == 0.0

    def test_token_overlap_empty(self):
        assert MemoryReconstructor._token_overlap("", "test") == 0.0

    def test_compute_fidelity_direct(self):
        fid = MemoryReconstructor._compute_fidelity(0.9, [], 1.0)
        assert fid > 0

    def test_compute_fidelity_with_neighbors(self):
        fid = MemoryReconstructor._compute_fidelity(0.5, [0.3, 0.2], 0.8)
        assert fid > 0

    def test_compute_fidelity_low_coherence(self):
        fid = MemoryReconstructor._compute_fidelity(0.9, [], 0.1)
        assert fid < 0.5  # Degraded by low coherence

    def test_blend_memories_no_neighbors(self):
        mr = MemoryReconstructor(MagicMock(), MagicMock())
        seed = {"id": 1, "content": "test", "tags": "a,b", "importance": 0.5, "emotional": 0.5}
        blended = mr._blend_memories(seed, [], [])
        assert blended["content"] == "test"

    def test_blend_memories_with_neighbors(self):
        mr = MemoryReconstructor(MagicMock(), MagicMock())
        seed = {"id": 1, "content": "test seed", "tags": "a", "importance": 0.5, "emotional": 0.5}
        neighbors = [
            {"content": "neighbor1", "tags": "b", "importance": 0.7, "emotional": 0.3},
        ]
        blended = mr._blend_memories(seed, neighbors, [0.8])
        assert "associated" in blended["content"]
        assert "a" in blended.get("reconstructed_tags", "")
        assert "b" in blended.get("reconstructed_tags", "")
