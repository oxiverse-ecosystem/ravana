"""Tests for ravana_ml.plasticity and ravana_ml.propagation."""

import pytest
import numpy as np
from ravana_ml.graph import ConceptGraph
from ravana_ml.plasticity import HebbianPlasticity, AntiHebbianPlasticity, StructuralPlasticity
from ravana_ml.propagation import PropagationEngine
from ravana_ml.free_energy import FreeEnergyAccumulator


class TestHebbianPlasticity:
    def test_init(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        hp = HebbianPlasticity(graph=g, lr=0.01)
        assert hp.graph is g
        assert hp.lr == 0.01

    def test_update_with_existing_edge(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        n2 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        g.add_edge(n1.id, n2.id, weight=0.5)
        g.activate(n1.id, amount=1.0)
        g.activate(n2.id, amount=1.0)
        hp = HebbianPlasticity(graph=g, lr=0.01)
        delta = hp.update(n1.id, n2.id)
        edge = g.get_edge(n1.id, n2.id)
        assert edge.weight != 0.5  # Should have changed

    def test_update_creates_edge(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        n2 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        g.activate(n1.id, amount=2.0)
        g.activate(n2.id, amount=2.0)
        hp = HebbianPlasticity(graph=g, lr=0.01)
        hp.update(n1.id, n2.id)  # coactivation = 4.0 > 0.3
        edge = g.get_edge(n1.id, n2.id)
        assert edge is not None

    def test_update_returns_zero_no_source(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n2 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        hp = HebbianPlasticity(graph=g, lr=0.01)
        delta = hp.update(999, n2.id)
        assert delta == 0.0


class TestAntiHebbianPlasticity:
    def test_init(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        ap = AntiHebbianPlasticity(graph=g, lr=0.01)
        assert ap.lr == 0.01

    def test_update_weakens_edge(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        n2 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        g.add_edge(n1.id, n2.id, weight=0.8, confidence=0.8)
        g.activate(n1.id, amount=1.0)
        ap = AntiHebbianPlasticity(graph=g, lr=0.01)
        ap.update(n1.id, n2.id, persistent_mismatch=2.0)
        edge = g.get_edge(n1.id, n2.id)
        assert edge is None or edge.weight < 0.8

    def test_update_no_edge_returns_zero(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        g.add_node(vector=np.random.randn(8).astype(np.float32))
        n2 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        ap = AntiHebbianPlasticity(graph=g, lr=0.01)
        delta = ap.update(999, n2.id)
        assert delta == 0.0


class TestStructuralPlasticity:
    def test_init(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        sp = StructuralPlasticity(graph=g)
        assert sp.prune_threshold == 0.05

    def test_step(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        n2 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        g.add_edge(n1.id, n2.id, weight=0.5, confidence=0.02)  # Below prune threshold
        sp = StructuralPlasticity(graph=g, prune_threshold=0.05)
        pruned, formed = sp.step()
        assert pruned == 1  # Edge confidence 0.02 < 0.05 should be pruned

    def test_prune_by_age(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        n2 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        g.add_edge(n1.id, n2.id, weight=0.5, confidence=0.5)
        sp = StructuralPlasticity(graph=g)
        # Set edge timestamp to ancient time
        edge = g.get_edge(n1.id, n2.id)
        edge.timestamp = 0
        pruned = sp.prune_by_age(max_age=1.0)
        assert pruned == 1


class TestPropagationEngine:
    def test_init(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        pe = PropagationEngine(g)
        assert pe.graph is g
        assert pe.propagation_count == 0

    def test_propagate(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.ones(8, dtype=np.float32), label="a")
        n2 = g.add_node(vector=np.ones(8, dtype=np.float32), label="b")
        g.add_edge(n1.id, n2.id, weight=0.5)
        pe = PropagationEngine(g)
        active = pe.propagate(input_vector=np.ones(8, dtype=np.float32), steps=2, k_active=3)
        assert len(active) > 0
        assert pe.propagation_count == 1

    def test_get_activation_vector(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.ones(8, dtype=np.float32))
        n2 = g.add_node(vector=np.ones(8, dtype=np.float32))
        pe = PropagationEngine(g)
        g.activate(n1.id, amount=0.5)
        vec = pe.get_activation_vector([n1.id, n2.id])
        assert vec.shape == (8,)

    def test_measure_coherence(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        n2 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        g.add_edge(n1.id, n2.id, weight=0.7, confidence=0.8)
        pe = PropagationEngine(g)
        coherence = pe.measure_coherence([n1.id, n2.id])
        assert coherence == pytest.approx(0.7 * 0.8)

    def test_measure_coherence_single(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n = g.add_node(vector=np.random.randn(8).astype(np.float32))
        pe = PropagationEngine(g)
        assert pe.measure_coherence([n.id]) == 1.0

    def test_get_prediction(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        n2 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        g.add_edge(n1.id, n2.id, weight=0.8)
        g.activate(n1.id, amount=1.0)
        pe = PropagationEngine(g)
        predictions = pe.get_prediction([n1.id], top_k=3)
        assert len(predictions) > 0
        assert n2.id in predictions

    def test_get_prediction_with_context(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.zeros(8, dtype=np.float32))
        n1.vector[0] = 1.0
        n2 = g.add_node(vector=np.zeros(8, dtype=np.float32))
        n2.vector[0] = 1.0
        g.add_edge(n1.id, n2.id, weight=0.6)
        g.activate(n1.id, amount=1.0)
        pe = PropagationEngine(g)
        ctx = np.zeros(8, dtype=np.float32)
        ctx[0] = 1.0
        predictions = pe.get_prediction([n1.id], top_k=3, context_field=ctx, context_bias=0.5)
        assert len(predictions) > 0

    def test_propagation_no_reset(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        pe = PropagationEngine(g)
        # propagate calls reset_activation internally
        active = pe.propagate(input_vector=np.zeros(8, dtype=np.float32), steps=1, k_active=3)
        # No nodes means empty result
        assert len(active) <= 3


class TestFreeEnergyAccumulator:
    def test_init(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        assert fe.free_energy == 0.0

    def test_accumulate_semantic(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_semantic(1.0, salience=0.5)
        assert fe.semantic_free_energy == 0.5

    def test_accumulate_linguistic(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_linguistic(0.5)
        assert fe.linguistic_free_energy > 0

    def test_accumulate_episodic(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_episodic(0.5, recency=0.3)
        assert fe.episodic_free_energy > 0

    def test_accumulate_contradiction(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_contradiction(3)
        assert fe.contradiction_free_energy > 0

    def test_total_and_normalized(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_semantic(30.0)
        assert fe.total == fe.free_energy
        assert fe.normalized > 0

    def test_decay(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_semantic(10.0)
        fe.decay(rate=0.5)
        assert fe.semantic_free_energy < 10.0

    def test_reset(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_semantic(10.0)
        fe.reset()
        assert fe.free_energy == 0.0

    def test_needs_sleep(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        assert not fe.needs_sleep()
        fe.accumulate_semantic(50.0)
        assert fe.needs_sleep()

    def test_report(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        r = fe.report()
        assert "free_energy" in r
        assert "normalized" in r
        assert "semantic" in r

    def test_history(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        assert len(fe.history) == 0
        fe.accumulate_semantic(1.0)
        assert len(fe.history) > 0

    def test_apply_to_graph(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n = g.add_node(vector=np.random.randn(8).astype(np.float32))
        n.contradiction_count = 5
        fe = FreeEnergyAccumulator(g)
        fe.apply_to_graph()
        assert n.prediction_free_energy > 0
