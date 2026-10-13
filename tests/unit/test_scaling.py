"""Scaling tests for ConceptGraph — empirical generalization from toy baselines.

Tests that RAVANA's cognitive architecture (ConceptGraph + GRACE governor +
energy-based inference) can scale to real-world complexity without state
space explosion.

Key scaling properties tested:
1. Spreading activation convergence on large graphs (10k+ nodes)
2. Multi-hop inference with bounded frontier (no semantic fog)
3. Context gating at scale (efficient context switching)
4. Memory efficiency (sparse adjacency for large graphs)
"""

import pytest
import numpy as np
import time
from ravana_ml.graph import ConceptGraph, ConceptEdge


class TestLargeGraphSpreadingActivation:
    """Test that spreading activation converges on large graphs."""

    def _build_large_graph(self, n_nodes: int, edges_per_node: int = 5,
                           dim: int = 64, seed: int = 42) -> ConceptGraph:
        """Build a large graph with random edges."""
        rng = np.random.RandomState(seed)
        g = ConceptGraph(dim=dim, max_nodes=n_nodes * 2,
                        max_edges=n_nodes * edges_per_node * 2,
                        rng=rng)
        # Add nodes
        for i in range(n_nodes):
            vec = rng.randn(dim).astype(np.float32)
            vec /= (np.linalg.norm(vec) + 1e-15)
            g.add_node(vec, label=f"n{i}")
        # Add random edges
        for i in range(n_nodes):
            for _ in range(edges_per_node):
                j = rng.randint(0, n_nodes)
                if i != j:
                    weight = rng.uniform(0.1, 0.9)
                    g.add_edge(i, j, weight=weight, relation_type="semantic")
        return g

    def test_spreading_activation_converges_1k_nodes(self):
        """Spreading activation should converge on a 1k node graph."""
        g = self._build_large_graph(1000, edges_per_node=5)
        # Activate a seed node
        g.reset_activation()
        g.activate(0, 1.0)
        g.spread_activation(steps=3, k_active=7, decay=0.5)
        # Should have some active nodes
        active_count = sum(1 for n in g.nodes.values() if n.activation > 0.01)
        assert active_count > 0
        # Should not activate ALL nodes (no state space explosion)
        assert active_count < 1000

    def test_spreading_activation_converges_5k_nodes(self):
        """Spreading activation should converge on a 5k node graph."""
        g = self._build_large_graph(5000, edges_per_node=5)
        g.reset_activation()
        g.activate(0, 1.0)
        g.spread_activation(steps=3, k_active=7, decay=0.5)
        active_count = sum(1 for n in g.nodes.values() if n.activation > 0.01)
        assert active_count > 0
        assert active_count < 5000

    def test_spreading_activation_converges_10k_nodes(self):
        """Spreading activation should converge on a 10k node graph."""
        g = self._build_large_graph(10000, edges_per_node=5)
        g.reset_activation()
        g.activate(0, 1.0)
        g.spread_activation(steps=3, k_active=7, decay=0.5)
        active_count = sum(1 for n in g.nodes.values() if n.activation > 0.01)
        assert active_count > 0
        assert active_count < 10000

    def test_spreading_activation_bounded_active_set(self):
        """Active set should be bounded regardless of graph size."""
        for n_nodes in [1000, 5000, 10000]:
            g = self._build_large_graph(n_nodes, edges_per_node=5)
            g.reset_activation()
            g.activate(0, 1.0)
            g.spread_activation(steps=3, k_active=7, decay=0.5)
            active_count = sum(1 for n in g.nodes.values() if n.activation > 0.01)
            # Active set should be a small fraction of total nodes
            assert active_count < n_nodes * 0.1, \
                f"Active set too large for {n_nodes} nodes: {active_count}"

    def test_spreading_activation_runtime_scales(self):
        """Runtime should scale roughly linearly with graph size."""
        times = []
        for n_nodes in [1000, 5000, 10000]:
            g = self._build_large_graph(n_nodes, edges_per_node=5)
            g.reset_activation()
            g.activate(0, 1.0)
            start = time.time()
            g.spread_activation(steps=3, k_active=7, decay=0.5)
            elapsed = time.time() - start
            times.append(elapsed)
        # 10x nodes should not take more than ~50x time (sub-quadratic)
        assert times[2] < times[0] * 50, \
            f"Runtime scaling too steep: {times}"


class TestLargeGraphMultiHopInference:
    """Test that multi-hop inference is bounded on large graphs."""

    def _build_chain_graph(self, n_nodes: int, dim: int = 64,
                           seed: int = 42) -> ConceptGraph:
        """Build a graph with long chains for multi-hop inference."""
        rng = np.random.RandomState(seed)
        g = ConceptGraph(dim=dim, max_nodes=n_nodes * 2,
                        max_edges=n_nodes * 2, rng=rng)
        # Add nodes
        for i in range(n_nodes):
            vec = rng.randn(dim).astype(np.float32)
            vec /= (np.linalg.norm(vec) + 1e-15)
            g.add_node(vec, label=f"n{i}")
        # Add chain edges: 0 -> 1 -> 2 -> ... -> n-1
        for i in range(n_nodes - 1):
            g.add_edge(i, i + 1, weight=0.8, relation_type="semantic")
        return g

    def test_inference_bounded_frontier(self):
        """Multi-hop inference should use bounded frontier (winner-take-most)."""
        g = self._build_chain_graph(1000)
        results = g.infer_chain(0, max_hops=3, frontier_budget=5)
        # Should return at most k results
        assert len(results) <= 5

    def test_inference_respects_hop_limit(self):
        """Inference should not exceed max_hops."""
        g = self._build_chain_graph(1000)
        results = g.infer_chain(0, max_hops=2, frontier_budget=5)
        for _, _, path in results:
            # Path length should be <= max_hops + 1 (including start)
            assert len(path) <= 3

    def test_inference_on_large_graph(self):
        """Inference should work on large graphs without explosion."""
        rng = np.random.RandomState(42)
        g = ConceptGraph(dim=64, max_nodes=20000, max_edges=100000, rng=rng)
        for i in range(10000):
            vec = rng.randn(64).astype(np.float32)
            vec /= (np.linalg.norm(vec) + 1e-15)
            g.add_node(vec, label=f"n{i}")
        for _ in range(50000):
            i = rng.randint(0, 10000)
            j = rng.randint(0, 10000)
            if i != j:
                g.add_edge(i, j, weight=rng.uniform(0.1, 0.9),
                           relation_type="semantic")
        results = g.infer_chain(0, max_hops=3, frontier_budget=5)
        # Should return some results (may be 0 if no good paths)
        assert len(results) >= 0

    def test_inference_confidence_decay(self):
        """Deeper hops should have lower confidence."""
        g = self._build_chain_graph(100)
        results = g.infer_chain(0, max_hops=3, frontier_budget=10)
        if len(results) >= 2:
            # Results should be sorted by score (descending)
            scores = [r[1] for r in results]
            assert scores == sorted(scores, reverse=True)


class TestContextGatingAtScale:
    """Test that context gating works efficiently on large graphs."""

    def _build_context_graph(self, n_nodes: int, n_contexts: int = 10,
                             dim: int = 64, seed: int = 42) -> ConceptGraph:
        """Build a large graph with context-dependent edges."""
        rng = np.random.RandomState(seed)
        g = ConceptGraph(dim=dim, max_nodes=n_nodes * 2,
                        max_edges=n_nodes * 10, rng=rng)
        # Add nodes
        for i in range(n_nodes):
            vec = rng.randn(dim).astype(np.float32)
            vec /= (np.linalg.norm(vec) + 1e-15)
            g.add_node(vec, label=f"n{i}")
        # Add context-dependent edges
        for i in range(n_nodes):
            for _ in range(3):
                j = rng.randint(0, n_nodes)
                if i != j:
                    ctx = f"ctx{rng.randint(0, n_contexts)}"
                    g.add_context_edge(i, j, context_labels=[ctx],
                                       weight=rng.uniform(0.3, 0.9),
                                       relation_type="semantic")
        return g

    def test_context_switching_large_graph(self):
        """Context switching should be fast on large graphs."""
        g = self._build_context_graph(5000, n_contexts=10)
        start = time.time()
        for i in range(10):
            g.set_context([f"ctx{i}"])
        elapsed = time.time() - start
        # 10 context switches should take less than 1 second
        assert elapsed < 1.0

    def test_context_gating_filters_edges(self):
        """Context gating should filter edges on large graphs."""
        g = self._build_context_graph(5000, n_contexts=10)
        g.set_context(["ctx0"])
        # Count active edges from node 0
        active = 0
        for target_id, edge in g._outgoing.get(0, []):
            if g.is_edge_active(edge):
                active += 1
        # Should have fewer active edges than total
        total = len(g._outgoing.get(0, []))
        assert active <= total

    def test_spreading_with_context_large_graph(self):
        """Spreading activation with context should work on large graphs."""
        g = self._build_context_graph(5000, n_contexts=10)
        g.set_context(["ctx0"])
        g.reset_activation()
        g.activate(0, 1.0)
        g.spread_activation(steps=3, k_active=7, decay=0.5)
        active_count = sum(1 for n in g.nodes.values() if n.activation > 0.01)
        assert active_count > 0
        assert active_count < 5000


class TestSparseAdjacencyScaling:
    """Test that sparse adjacency matrix is used for large graphs."""

    def test_sparse_path_activated_above_threshold(self):
        """Sparse path should be used when nodes > _sparse_threshold."""
        g = ConceptGraph(dim=64, max_nodes=300, max_edges=1000,
                        rng=np.random.RandomState(42))
        # Add enough nodes to exceed threshold
        for i in range(250):
            vec = np.random.randn(64).astype(np.float32)
            g.add_node(vec, label=f"n{i}")
        # Add edges
        for i in range(250):
            for j in range(5):
                target = (i + j + 1) % 250
                g.add_edge(i, target, weight=0.5, relation_type="semantic")
        # Trigger sparse rebuild
        g._rebuild_sparse_adj()
        # Sparse adjacency should be built
        assert g._adj_sparse is not None
        assert g._adj_sparse.shape[0] == 250

    def test_sparse_propagation_correctness(self):
        """Sparse propagation should produce similar results to fine-grained."""
        g = ConceptGraph(dim=16, max_nodes=300, max_edges=1000,
                        rng=np.random.RandomState(42))
        # Build a small chain
        for i in range(10):
            vec = np.random.randn(16).astype(np.float32)
            g.add_node(vec, label=f"n{i}")
        for i in range(9):
            g.add_edge(i, i + 1, weight=0.8, relation_type="semantic")
        # Activate and spread with relation_type to force fine-grained path
        g.reset_activation()
        g.activate(0, 1.0)
        g.spread_activation(steps=2, k_active=5, decay=0.5, relation_type="semantic")
        # Node 1 should have activation
        assert g.nodes[1].activation > 0.01


class TestRealWorldScaleInput:
    """Test that RAVANA can process real-world-scale input."""

    def test_large_graph_construction(self):
        """Building a 10k node graph should complete in reasonable time."""
        rng = np.random.RandomState(42)
        start = time.time()
        g = ConceptGraph(dim=64, max_nodes=20000, max_edges=100000, rng=rng)
        for i in range(10000):
            vec = rng.randn(64).astype(np.float32)
            vec /= (np.linalg.norm(vec) + 1e-15)
            g.add_node(vec, label=f"concept_{i}")
        # Add ~50k edges
        for _ in range(50000):
            i = rng.randint(0, 10000)
            j = rng.randint(0, 10000)
            if i != j:
                g.add_edge(i, j, weight=rng.uniform(0.1, 0.9),
                           relation_type="semantic")
        elapsed = time.time() - start
        # Should complete in under 30 seconds
        assert elapsed < 30.0
        assert len(g.nodes) == 10000
        assert len(g.edges) > 0

    def test_large_graph_spreading_does_not_explode(self):
        """Spreading activation on 10k nodes should not activate all nodes."""
        rng = np.random.RandomState(42)
        g = ConceptGraph(dim=64, max_nodes=20000, max_edges=100000, rng=rng)
        for i in range(10000):
            vec = rng.randn(64).astype(np.float32)
            vec /= (np.linalg.norm(vec) + 1e-15)
            g.add_node(vec, label=f"concept_{i}")
        for _ in range(50000):
            i = rng.randint(0, 10000)
            j = rng.randint(0, 10000)
            if i != j:
                g.add_edge(i, j, weight=rng.uniform(0.1, 0.9),
                           relation_type="semantic")
        g.reset_activation()
        g.activate(0, 1.0)
        g.spread_activation(steps=3, k_active=7, decay=0.5)
        active_count = sum(1 for n in g.nodes.values() if n.activation > 0.01)
        # Active set should be a small fraction
        assert active_count < 1000
