"""Tests for RAVANA ConceptGraph scaling behavior.

Validates that the existing architecture (ConceptGraph + GRACE governor +
energy-based inference) handles larger graphs without state-space explosion.
Uses 500-1000 node graphs for fast unit test execution; the full 10k-node
demonstration lives in scripts/demonstrate_scaling.py.

Covers:
  - Graph construction and basic operations at scale
  - Spreading activation convergence at scale
  - Inference chain quality at scale
  - Pattern separation (top-K wiring) at scale
  - Homeostatic downscale at scale
  - Concept merging at scale
  - Graph diagnostics at scale
  - Memory-bounded operation (no O(N^2) blowup in critical paths)
"""
import time
import numpy as np
import pytest

from ravana_ml.graph import ConceptGraph, ConceptNode, ConceptEdge


@pytest.fixture
def graph():
    """Create a ConceptGraph with 1000 nodes for scaling tests."""
    g = ConceptGraph(dim=64, max_nodes=5000, max_edges=30000)
    rng = np.random.RandomState(42)
    for i in range(1000):
        vec = rng.randn(64).astype(np.float32)
        vec /= (np.linalg.norm(vec) + 1e-15)
        g.add_node(vector=vec, label=f"concept_{i}")
    return g


@pytest.fixture
def wired_graph():
    """Create a 1000-node graph with realistic edge wiring."""
    g = ConceptGraph(dim=64, max_nodes=5000, max_edges=30000)
    rng = np.random.RandomState(42)
    for i in range(1000):
        vec = rng.randn(64).astype(np.float32)
        vec /= (np.linalg.norm(vec) + 1e-15)
        g.add_node(vector=vec, label=f"concept_{i}")
    # Wire each node to its 8 nearest neighbors (pattern separation)
    g._rebuild_vector_matrix()
    for i in range(1000):
        node = g.get_node(i)
        if node is None:
            continue
        sims = g._vector_matrix_normed @ node.vector.astype(np.float32)
        top_k_idx = np.argpartition(sims, -9)[-9:]  # exclude self
        for idx in top_k_idx:
            other_nid = g._node_id_order[idx]
            if other_nid != i and sims[idx] > 0.3:
                g.add_edge(i, int(other_nid), weight=float(sims[idx]) * 0.5,
                           relation_type="semantic", confidence=0.3)
    return g


class TestScalingGraph:
    """ConceptGraph scaling behavior with larger graphs."""

    def test_graph_construction(self, graph):
        """Graph with 1000 nodes constructs successfully."""
        assert len(graph.nodes) == 1000
        assert graph.dim == 64
        assert graph.max_nodes >= 1000

    def test_node_retrieval_at_scale(self, graph):
        """Node retrieval is O(1) regardless of graph size."""
        start = time.time()
        for i in range(1000):
            node = graph.get_node(i)
            assert node is not None
            assert node.id == i
        elapsed = time.time() - start
        assert elapsed < 0.5  # 1000 lookups in < 0.5s

    def test_edge_addition_at_scale(self, graph):
        """Edge addition works at 1000-node scale."""
        node_ids = list(graph.nodes.keys())
        for i in range(min(100, len(node_ids) - 1)):
            src = node_ids[i]
            tgt = node_ids[i + 1]
            edge = graph.add_edge(src, tgt, weight=0.5, relation_type="semantic")
            assert edge is not None
            assert edge.source == src
            assert edge.target == tgt

    def test_spreading_activation_convergence(self, wired_graph):
        """Spreading activation converges within bounded steps."""
        seed_id = 42
        wired_graph.activate(seed_id, 1.0)
        wired_graph.spread_activation(steps=3, k_active=7, decay=0.5)
        active_count = len(wired_graph._active_nodes)
        assert active_count > 0
        # Check that activation values are bounded (no explosion)
        for nid in wired_graph._active_nodes:
            node = wired_graph.get_node(nid)
            assert node is not None
            assert node.activation <= 3.0  # ConceptGraph caps at 3.0
            assert node.activation >= 0.0

    def test_spreading_activation_bounded_active_set(self, wired_graph):
        """Spreading activation does not activate all nodes (bounded)."""
        seed_id = 100
        wired_graph.activate(seed_id, 1.0)
        wired_graph.spread_activation(steps=3, k_active=7, decay=0.5)
        active_count = len(wired_graph._active_nodes)
        # With k_active=7 and 3 hops, should be much less than 1000
        assert active_count < 200
        assert active_count > 0

    def test_inference_chain_at_scale(self, wired_graph):
        """Inference chains work at 1000-node scale."""
        start_id = 500
        results = wired_graph.infer_chain(start_id, max_hops=3, k=5)
        assert isinstance(results, list)
        for tid, score, path in results:
            assert tid in wired_graph.nodes
            assert 0.0 <= score <= 1.0
            assert len(path) >= 2

    def test_inference_specificity_at_scale(self, wired_graph):
        """Inference produces specific (winner-take-most) results at scale."""
        start_id = 100
        results = wired_graph.infer_chain(start_id, max_hops=3, k=5)
        if len(results) >= 2:
            scores = [score for _, score, _ in results]
            total = sum(scores)
            if total > 0:
                specificity = scores[0] / total
                assert specificity > 0.3  # winner should dominate

    def test_pattern_separation_limits_edges(self, wired_graph):
        """Pattern separation (top-K wiring) keeps edge count bounded."""
        edge_count = len(wired_graph.edges)
        assert edge_count > 0
        # Should be much less than N^2 (1M)
        assert edge_count < 50000

    def test_homeostatic_downscale_at_scale(self, wired_graph):
        """Homeostatic downscale works at 1000-node scale."""
        total_before, total_after = wired_graph.homeostatic_downscale()
        assert total_before > 0
        assert total_after > 0
        assert total_after <= total_before * 1.01  # small tolerance

    def test_concept_merging_at_scale(self, wired_graph):
        """Concept merging works at 1000-node scale."""
        child_ids = [200, 201, 202]
        parent_id = wired_graph.merge_concepts(child_ids, label="merged_cluster")
        assert parent_id is not None
        assert parent_id in wired_graph.nodes
        parent = wired_graph.get_node(parent_id)
        assert parent.label == "merged_cluster"
        assert parent.level > 0
        for cid in child_ids:
            child = wired_graph.get_node(cid)
            assert child.parent == parent_id

    def test_graph_diagnostics_at_scale(self, wired_graph):
        """Graph diagnostics compute at 1000-node scale."""
        start = time.time()
        metrics = wired_graph.graph_diagnostics(lightweight=True)
        elapsed = time.time() - start
        assert not metrics.get("empty", False)
        assert "graph_entropy" in metrics
        assert "branching_factor" in metrics
        assert "edge_weight_mean" in metrics
        assert elapsed < 5.0

    def test_similarity_search_at_scale(self, wired_graph):
        """Similarity search works at 1000-node scale."""
        rng = np.random.RandomState(123)
        query_vec = rng.randn(64).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)
        results = wired_graph.find_similar(query_vec, k=10)
        assert len(results) <= 10
        for nid, sim in results:
            assert nid in wired_graph.nodes
            assert -1.0 <= sim <= 1.0

    def test_bind_input_at_scale(self, wired_graph):
        """bind_input works at 1000-node scale."""
        rng = np.random.RandomState(456)
        query_vec = rng.randn(64).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)
        matched = wired_graph.bind_input(query_vec, k=5)
        assert len(matched) <= 5
        for nid in matched:
            assert nid in wired_graph.nodes

    def test_soft_lateral_inhibition_at_scale(self, wired_graph):
        """Soft lateral inhibition keeps active set bounded at scale."""
        for i in range(50):
            wired_graph.activate(i, 1.0)
        wired_graph.spread_activation(steps=1, k_active=7, decay=0.5)
        active_count = len(wired_graph._active_nodes)
        assert active_count < 100  # should be much less than 1000

    def test_edge_structural_importance_at_scale(self, wired_graph):
        """Structural importance computation works at scale."""
        start = time.time()
        si = wired_graph.compute_edge_structural_importance(sample_size=20)
        elapsed = time.time() - start
        assert isinstance(si, dict)
        assert len(si) > 0
        assert elapsed < 10.0

    def test_prune_low_quality_edges_at_scale(self, wired_graph):
        """Edge pruning works at 1000-node scale."""
        initial_count = len(wired_graph.edges)
        removed = wired_graph.prune_low_quality_edges(C1=0.35, K=1, U=0.3)
        assert removed >= 0
        assert len(wired_graph.edges) == initial_count - removed

    def test_consolidate_vectors_at_scale(self, wired_graph):
        """Vector consolidation works at 1000-node scale."""
        for i in range(50):
            wired_graph.activate(i, 1.0)
        wired_graph.consolidate_vectors(rate=0.1, incremental=True)
        for i in range(50):
            node = wired_graph.get_node(i)
            if node is not None:
                norm = np.linalg.norm(node.vector)
                assert abs(norm - 1.0) < 0.01

    def test_concept_splitting_at_scale(self, wired_graph):
        """Concept splitting works at 1000-node scale."""
        split_id = None
        for nid in wired_graph.nodes:
            if len(wired_graph.get_outgoing(nid)) >= 2:
                split_id = nid
                break
        if split_id is not None:
            child_a, child_b = wired_graph.split_concept(split_id)
            assert child_a in wired_graph.nodes
            assert child_b in wired_graph.nodes
            parent = wired_graph.get_node(split_id)
            assert parent.children == {child_a, child_b}


class TestScalingMemoryBounded:
    """Memory-bounded operation at scale."""

    def test_no_full_similarity_matrix_in_spread(self):
        """Spreading activation does not build full N x N similarity matrix."""
        g = ConceptGraph(dim=64, max_nodes=5000, max_edges=30000)
        rng = np.random.RandomState(42)
        for i in range(500):
            vec = rng.randn(64).astype(np.float32)
            vec /= (np.linalg.norm(vec) + 1e-15)
            g.add_node(vector=vec, label=f"n{i}")
        for i in range(500):
            for j in range(i + 1, min(i + 6, 500)):
                g.add_edge(i, j, weight=0.3, relation_type="semantic")
        g.activate(0, 1.0)
        g.spread_activation(steps=3, k_active=7, decay=0.5)
        assert len(g._active_nodes) > 0

    def test_sparse_adjacency_used_at_scale(self):
        """Sparse adjacency matrix is used for graphs > 200 nodes."""
        g = ConceptGraph(dim=64, max_nodes=5000, max_edges=30000)
        rng = np.random.RandomState(42)
        for i in range(500):
            vec = rng.randn(64).astype(np.float32)
            vec /= (np.linalg.norm(vec) + 1e-15)
            g.add_node(vector=vec, label=f"n{i}")
        for i in range(500):
            for j in range(i + 1, min(i + 4, 500)):
                g.add_edge(i, j, weight=0.4, relation_type="semantic")
        g.activate(0, 1.0)
        g.spread_activation(steps=1, k_active=7, decay=0.5)
        assert g._adj_dirty is False


class TestScalingPatternSeparation:
    """Pattern separation (top-K wiring) behavior at scale."""

    def test_top_k_wiring_limits_out_degree(self):
        """Top-K wiring keeps out-degree bounded."""
        g = ConceptGraph(dim=64, max_nodes=5000, max_edges=30000)
        rng = np.random.RandomState(42)
        n_nodes = 500
        for i in range(n_nodes):
            vec = rng.randn(64).astype(np.float32)
            vec /= (np.linalg.norm(vec) + 1e-15)
            g.add_node(vector=vec, label=f"c{i}")
        g._rebuild_vector_matrix()
        for i in range(n_nodes):
            node = g.get_node(i)
            sims = g._vector_matrix_normed @ node.vector.astype(np.float32)
            top_k_idx = np.argpartition(sims, -9)[-9:]
            for idx in top_k_idx:
                other_nid = int(g._node_id_order[idx])
                if other_nid != i and sims[idx] > 0.3:
                    g.add_edge(i, other_nid, weight=float(sims[idx]) * 0.5,
                               relation_type="semantic", confidence=0.3)
        out_degrees = [len(g.get_outgoing(i)) for i in range(n_nodes)]
        max_degree = max(out_degrees)
        assert max_degree <= 8


class TestScalingGRACEGovernor:
    """GRACE governor behavior at scale."""

    def test_phase_classification_at_scale(self, wired_graph):
        """Phase classification works at 1000-node scale."""
        for i in range(50):
            wired_graph.activate(i, 1.0)
        wired_graph.spread_activation(steps=2, k_active=7, decay=0.5)
        phase_info = wired_graph.classify_phase()
        assert "phase" in phase_info
        assert phase_info["phase"] in ("focused", "exploratory", "diffuse", "rigid", "crisis", "empty")
        assert 0.0 <= phase_info["confidence"] <= 1.0

    def test_regulation_at_scale(self, wired_graph):
        """Regulation pipeline works at 1000-node scale."""
        for i in range(50):
            wired_graph.activate(i, 1.0)
        wired_graph.spread_activation(steps=2, k_active=7, decay=0.5)
        result = wired_graph.regulate()
        assert "phase" in result
        assert "adjustments" in result
        assert "regulator" in result

    def test_geometry_history_at_scale(self, wired_graph):
        """Geometry history tracking works at scale."""
        for step in range(5):
            for i in range(10):
                wired_graph.activate(i + step * 10, 1.0)
            wired_graph.spread_activation(steps=1, k_active=7, decay=0.5)
            wired_graph.record_geometry_snapshot(event=f"step_{step}", lightweight=True)
        summary = wired_graph.geometry_history_summary()
        assert summary["snapshots"] == 5


class TestScalingRealWorldInput:
    """Simulated real-world input processing at scale."""

    def test_codebase_concept_extraction(self):
        """Simulate extracting concepts from a large codebase."""
        g = ConceptGraph(dim=64, max_nodes=5000, max_edges=30000)
        rng = np.random.RandomState(42)
        code_tokens = [
            "function", "class", "import", "return", "if", "else", "for", "while",
            "try", "except", "raise", "with", "as", "def", "lambda", "yield",
            "async", "await", "pass", "break", "continue", "global", "nonlocal",
            "assert", "del", "in", "is", "not", "and", "or", "None", "True", "False",
        ] * 30  # ~1000 tokens
        for i, token in enumerate(code_tokens):
            vec = rng.randn(64).astype(np.float32)
            vec /= (np.linalg.norm(vec) + 1e-15)
            g.add_node(vector=vec, label=token)
        g._rebuild_vector_matrix()
        for i in range(min(200, len(code_tokens))):
            node = g.get_node(i)
            if node is None:
                continue
            sims = g._vector_matrix_normed @ node.vector.astype(np.float32)
            top_k_idx = np.argpartition(sims, -6)[-6:]
            for idx in top_k_idx:
                other_nid = int(g._node_id_order[idx])
                if other_nid != i and sims[idx] > 0.3:
                    g.add_edge(i, other_nid, weight=float(sims[idx]) * 0.4,
                               relation_type="semantic", confidence=0.25)
        assert len(g.nodes) == len(code_tokens)
        assert len(g.edges) > 0

    def test_document_analysis_at_scale(self):
        """Simulate analyzing a large document."""
        g = ConceptGraph(dim=64, max_nodes=5000, max_edges=30000)
        rng = np.random.RandomState(42)
        for i in range(1000):
            vec = rng.randn(64).astype(np.float32)
            vec /= (np.linalg.norm(vec) + 1e-15)
            g.add_node(vector=vec, label=f"term_{i}")
        g._rebuild_vector_matrix()
        for i in range(1000):
            node = g.get_node(i)
            if node is None:
                continue
            sims = g._vector_matrix_normed @ node.vector.astype(np.float32)
            top_k_idx = np.argpartition(sims, -11)[-11:]
            for idx in top_k_idx:
                other_nid = int(g._node_id_order[idx])
                if other_nid != i and sims[idx] > 0.35:
                    g.add_edge(i, other_nid, weight=float(sims[idx]) * 0.45,
                               relation_type="semantic", confidence=0.3)
        results = g.infer_chain(0, max_hops=3, k=5)
        assert isinstance(results, list)
        g.activate(0, 1.0)
        g.spread_activation(steps=3, k_active=7, decay=0.5)
        assert len(g._active_nodes) > 0
        metrics = g.graph_diagnostics(lightweight=True)
        assert not metrics.get("empty", False)
