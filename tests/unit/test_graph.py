"""Comprehensive tests for ravana_ml.graph — ConceptGraph, ConceptNode, ConceptEdge, Binding, Regulator, GeometryHistory."""

import pytest
import numpy as np
import time
from ravana_ml.graph import (
    ConceptNode, ConceptEdge, ConceptBinding, ConceptBindingMap,
    ConceptGraph, CognitiveRegulator, GeometryHistory,
)


class TestConceptNode:
    def test_init(self):
        v = np.random.randn(8).astype(np.float32)
        node = ConceptNode(0, v, label="test")
        assert node.id == 0
        assert node.label == "test"
        assert node.activation == 0.0

    def test_effective_activation(self):
        v = np.random.randn(8).astype(np.float32)
        node = ConceptNode(0, v)
        node.activation = 0.5
        node.fatigue = 0.2
        assert node.effective_activation == pytest.approx(0.4)

    def test_drift_magnitude(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        node = ConceptNode(0, v)
        assert node.drift_magnitude == pytest.approx(0.0)
        node.vector = np.array([0.0, 1.0], dtype=np.float32)
        assert node.drift_magnitude > 0

    def test_decay(self):
        v = np.random.randn(8).astype(np.float32)
        node = ConceptNode(0, v)
        node.activation = 1.0
        node.decay(rate=0.5)
        assert node.activation < 1.0

    def test_record_activation(self):
        v = np.random.randn(8).astype(np.float32)
        node = ConceptNode(0, v)
        node.record_activation()
        assert len(node.activation_history) == 1
        assert node.last_activated > 0

    def test_recency_score(self):
        v = np.random.randn(8).astype(np.float32)
        node = ConceptNode(0, v)
        assert node.recency_score() == 0.0  # Never activated
        node.record_activation()
        assert node.recency_score() > 0.0

    def test_plasticity(self):
        v = np.random.randn(8).astype(np.float32)
        node = ConceptNode(0, v)
        assert node.plasticity == 1.0 - node.stability

    def test_repr(self):
        v = np.random.randn(8).astype(np.float32)
        node = ConceptNode(0, v, label="test")
        r = repr(node)
        assert "Node 0" in r
        assert "test" in r


class TestConceptEdge:
    def test_init(self):
        e = ConceptEdge(0, 1, weight=0.7, confidence=0.8, relation_type="causal")
        assert e.source == 0
        assert e.target == 1
        assert e.weight == 0.7

    def test_effective_weight_excitatory(self):
        e = ConceptEdge(0, 1, weight=0.5)
        assert e.effective_weight == 0.5

    def test_effective_weight_inhibitory(self):
        e = ConceptEdge(0, 1, weight=0.5, edge_type="inhibitory")
        assert e.effective_weight == -0.5

    def test_posterior_mean(self):
        e = ConceptEdge(0, 1, weight=0.5)
        pm = e.posterior_mean
        assert 0 < pm < 1

    def test_posterior_uncertainty(self):
        e = ConceptEdge(0, 1, weight=0.5)
        u = e.posterior_uncertainty
        assert u > 0

    def test_relation_vector_init(self):
        e = ConceptEdge(0, 1, weight=0.5, relation_type="causal")
        assert e.relation_vector is not None
        assert abs(np.linalg.norm(e.relation_vector) - 1.0) < 1e-5

    def test_relation_vector_determinism(self):
        e1 = ConceptEdge(0, 1, weight=0.5, relation_type="causal")
        e2 = ConceptEdge(2, 3, weight=0.5, relation_type="causal")
        assert np.allclose(e1.relation_vector, e2.relation_vector)

    def test_relation_vector_types_different(self):
        e1 = ConceptEdge(0, 1, weight=0.5, relation_type="causal")
        e2 = ConceptEdge(2, 3, weight=0.5, relation_type="temporal")
        assert not np.allclose(e1.relation_vector, e2.relation_vector)

    def test_get_weight_for_agent(self):
        e = ConceptEdge(0, 1, weight=0.5)
        assert e.get_weight_for_agent("global") == 0.5
        e.update_weight_for_agent("user_test", 0.2)
        w = e.get_weight_for_agent("user_test")
        assert 0.5 <= w <= 1.0

    def test_setstate_backward_compat(self):
        state = {'source': 0, 'target': 1, '_weight': 0.5, '_confidence': 0.5, '_edge_type': 'excitatory'}
        e = ConceptEdge.__new__(ConceptEdge)
        e.__setstate__(state)
        assert e.source == 0
        assert e.predicate_token_id == -1  # default

    def test_repr(self):
        e = ConceptEdge(0, 1, weight=0.5)
        assert "Edge 0->1" in repr(e)


class TestConceptBinding:
    def test_init(self):
        b = ConceptBinding(0, 1, confidence=0.9, source="learned")
        assert b.token_id == 0
        assert b.concept_id == 1
        assert b.confidence == 0.9

    def test_reinforce(self):
        b = ConceptBinding(0, 1, confidence=0.5)
        b.reinforce(0.1)
        assert b.confidence > 0.5

    def test_decay(self):
        b = ConceptBinding(0, 1, confidence=0.9)
        b.decay(rate=0.5)
        assert b.decay_score > 0

    def test_strength(self):
        b = ConceptBinding(0, 1, confidence=0.8)
        s = b.strength
        assert s <= 0.8


class TestConceptBindingMap:
    def test_bind(self):
        m = ConceptBindingMap()
        b = m.bind(0, 1, confidence=0.9)
        assert b.concept_id == 1
        assert (0, 1) in m._index

    def test_bind_reinforces(self):
        m = ConceptBindingMap()
        m.bind(0, 1, confidence=0.5)
        m.bind(0, 1, confidence=0.9)
        assert m._index[(0, 1)].confidence > 0.5

    def test_get_concepts(self):
        m = ConceptBindingMap()
        m.bind(0, 1, confidence=0.9)
        m.bind(0, 2, confidence=0.3)
        concepts = m.get_concepts(0, min_confidence=0.5)
        assert len(concepts) == 1
        assert concepts[0].concept_id == 1

    def test_best_concept(self):
        m = ConceptBindingMap()
        m.bind(0, 5, confidence=0.8)
        assert m.best_concept(0) == 5

    def test_best_concept_none(self):
        m = ConceptBindingMap()
        assert m.best_concept(99) is None

    def test_prune(self):
        m = ConceptBindingMap()
        b = m.bind(0, 1, confidence=0.05)
        b.decay(rate=100)  # Force decay
        pruned = m.prune(min_strength=0.5)
        assert pruned == 1

    def test_is_ambiguous(self):
        m = ConceptBindingMap()
        m.bind(0, 1, confidence=0.8)
        m.bind(0, 2, confidence=0.7)
        assert m.is_ambiguous(0, threshold=0.3)

    def test_disambiguate(self):
        from ravana_ml.graph import ConceptGraph
        m = ConceptBindingMap()
        g = ConceptGraph(dim=8, max_nodes=10)
        v1 = g.add_node(vector=np.ones(8, dtype=np.float32))
        v2 = g.add_node(vector=-np.ones(8, dtype=np.float32))
        m.bind(0, v1.id, confidence=0.8)
        m.bind(0, v2.id, confidence=0.8)
        ctx = g.nodes[v1.id].vector
        winner = m.disambiguate(0, ctx, g)
        assert winner is not None

    def test_len(self):
        m = ConceptBindingMap()
        assert len(m) == 0
        m.bind(0, 1)
        assert len(m) == 1


class TestCognitiveRegulator:
    def test_init(self):
        r = CognitiveRegulator()
        assert r.current_phase == "exploratory"

    def test_update_basic(self):
        r = CognitiveRegulator()
        result = r.update({"phase": "exploratory", "confidence": 0.8, "recommendations": {}})
        assert "inhibition_boost" in result
        assert "phase" in result

    def test_status(self):
        r = CognitiveRegulator()
        s = r.status()
        assert s["phase"] == "exploratory"

    def test_meta_adapt(self):
        r = CognitiveRegulator()
        changes = r.meta_adapt(overshoot=0.8, recovery_speed=0.2, oscillation_rate=0.5)
        assert "damping_direction" in changes


class TestGeometryHistory:
    def test_init(self):
        g = GeometryHistory(max_snapshots=10)
        assert g.max_snapshots == 10

    def test_record(self):
        g = GeometryHistory()
        g.record({"graph_entropy": 0.5, "relation_separation": 0.7}, event="test")
        assert len(g.snapshots) == 1

    def test_get_series(self):
        g = GeometryHistory()
        g.record({"graph_entropy": 0.5})
        g.record({"graph_entropy": 0.6})
        s = g.get_series("graph_entropy")
        assert s == [0.5, 0.6]

    def test_detect_trend(self):
        g = GeometryHistory()
        for i in range(30):
            g.record({"graph_entropy": 0.5 + i * 0.01})
        trend = g.detect_trend("graph_entropy", window=20)
        assert trend["slope"] > 0  # Rising trend

    def test_summary(self):
        g = GeometryHistory()
        g.record({"graph_entropy": 0.5})
        s = g.summary()
        assert s["snapshots"] == 1


class TestConceptGraph:
    def test_init(self):
        g = ConceptGraph(dim=16, max_nodes=50)
        assert g.dim == 16
        assert g.max_nodes == 50

    def test_add_node(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n = g.add_node(vector=np.random.randn(8).astype(np.float32), label="test")
        assert n.id in g.nodes
        assert g.nodes[n.id].label == "test"

    def test_add_node_default_vector(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n = g.add_node()
        assert n.id in g.nodes

    def test_get_node(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n = g.add_node(label="test")
        assert g.get_node(n.id) is n
        assert g.get_node(999) is None

    def test_remove_node(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(label="a")
        n2 = g.add_node(label="b")
        g.add_edge(n1.id, n2.id)
        g.remove_node(n1.id)
        assert n1.id not in g.nodes

    def test_add_edge(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(label="a")
        n2 = g.add_node(label="b")
        e = g.add_edge(n1.id, n2.id, weight=0.7, relation_type="causal")
        assert (n1.id, n2.id) in g.edges
        assert e.weight == 0.7
    def test_add_edge_rejects_self_loop(self):
        """A concept must never be wired to itself. add_edge(src, src) must NOT
        insert a self-loop into the adjacency structures (it previously allowed
        the genuine 'oxiverse' self-loop found in saved weights)."""
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(label="oxiverse")
        # Returns an edge object (so callers dereferencing .weight don't crash)
        # but it is NOT stored in the graph.
        e = g.add_edge(n1.id, n1.id, weight=0.4, relation_type="semantic")
        assert e is not None
        assert (n1.id, n1.id) not in g.edges
        # No adjacency entries for the self-loop.
        assert all(t != n1.id for (t, _) in g.get_outgoing(n1.id))
        assert all(s != n1.id for (s, _) in g.get_incoming(n1.id))

    def test_get_edge(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(label="a")
        n2 = g.add_node(label="b")
        g.add_edge(n1.id, n2.id)
        assert g.get_edge(n1.id, n2.id) is not None
        assert g.get_edge(n2.id, n1.id) is None

    def test_remove_edge(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(label="a")
        n2 = g.add_node(label="b")
        g.add_edge(n1.id, n2.id)
        g.remove_edge(n1.id, n2.id)
        assert (n1.id, n2.id) not in g.edges

    def test_activate(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n = g.add_node()
        g.activate(n.id, amount=0.5)
        assert g.nodes[n.id].activation == 0.5

        assert g.nodes[n.id].activation == 0.5

    def test_activate_caps(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n = g.add_node()
        g.activate(n.id, amount=5.0)
        assert g.nodes[n.id].activation <= 3.0

    def test_find_similar(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        v = np.ones(8, dtype=np.float32)
        g.add_node(vector=v, label="target")
        results = g.find_similar(v, k=3)
        assert len(results) >= 1
        nid, sim = results[0]
        assert sim > 0.9

    def test_bind_input(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        v = np.ones(8, dtype=np.float32)
        g.add_node(vector=v, label="target")
        active = g.bind_input(v, k=3)
        assert len(active) > 0

    def test_apply_free_energy(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n = g.add_node()
        g.apply_free_energy(n.id, 3.0)
        assert g.nodes[n.id].prediction_free_energy > 0

    def test_hebbian_update(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        n2 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        g.activate(n1.id, amount=1.0)
        g.activate(n2.id, amount=1.0)
        g.hebbian_update(n1.id, n2.id, 0.5)
        edge = g.get_edge(n1.id, n2.id)
        assert edge is not None

    def test_adjust_vector(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        v = np.ones(8, dtype=np.float32)
        n = g.add_node(vector=v)
        delta = np.random.randn(8).astype(np.float32) * 0.01
        orig = g.nodes[n.id].vector.copy()
        g.adjust_vector(n.id, delta, lr=0.1)
        assert not np.allclose(g.nodes[n.id].vector, orig)

    def test_homeostatic_downscale(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(label="a")
        n2 = g.add_node(label="b")
        g.add_edge(n1.id, n2.id, weight=0.9)
        total_before, total_after = g.homeostatic_downscale()
        assert total_before > 0
        assert total_after > 0

    def test_should_split_false(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n = g.add_node()
        assert not g.should_split(n.id)

    def test_spread_activation_basic(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.ones(8, dtype=np.float32))
        n2 = g.add_node(vector=np.ones(8, dtype=np.float32))
        g.add_edge(n1.id, n2.id, weight=0.8, confidence=0.9, relation_type="causal")
        g.activate(n1.id, amount=1.0)
        g.spread_activation(steps=2, k_active=3, decay=0.5, relation_type="causal")
        assert g.nodes[n2.id].activation > 0

    def test_get_outgoing(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node()
        n2 = g.add_node()
        g.add_edge(n1.id, n2.id)
        edges = g.get_outgoing(n1.id)
        assert len(edges) == 1

    def test_get_incoming(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node()
        n2 = g.add_node()
        g.add_edge(n1.id, n2.id)
        edges = g.get_incoming(n2.id)
        assert len(edges) == 1

    def test_form_inhibitory_edges(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node()
        n2 = g.add_node()
        n3 = g.add_node()
        g.add_edge(n1.id, n2.id, weight=0.5, confidence=0.2)
        g.add_edge(n1.id, n3.id, weight=0.5, confidence=0.2)
        g.activate(n2.id, amount=0.5)
        g.activate(n3.id, amount=0.5)
        g.contradiction_hotspots.add(n1.id)
        g.nodes[n1.id].contradiction_count = 5
        formed = g.form_inhibitory_edges(contradiction_threshold=3)
        assert formed > 0

    def test_split_concept(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        parent = g.add_node(vector=np.ones(8, dtype=np.float32), label="parent")
        g.add_node(vector=np.array([1.0] + [0.0] * 7, dtype=np.float32), label="a")
        g.add_node(vector=np.array([-1.0] + [0.0] * 7, dtype=np.float32), label="b")
        child_a, child_b = g.split_concept(parent.id)
        assert child_a in g.nodes
        assert child_b in g.nodes

    def test_compute_edge_structural_importance(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(label="a")
        n2 = g.add_node(label="b")
        g.add_edge(n1.id, n2.id)
        imp = g.compute_edge_structural_importance()
        assert len(imp) > 0

    def test_update_temporal_context(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n = g.add_node(vector=np.ones(8, dtype=np.float32))
        g.activate(n.id, amount=1.0)
        g.update_temporal_context()
        assert np.any(g.temporal_context != 0)

    def test_temporal_context_similarity(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n = g.add_node(vector=np.ones(8, dtype=np.float32))
        n.record_activation(context_vector=np.ones(8, dtype=np.float32))
        g.temporal_context = np.ones(8, dtype=np.float32)
        g.temporal_context /= np.linalg.norm(g.temporal_context)
        sim = g.temporal_context_similarity(n)
        assert sim > 0
