"""CI-critical tests — core modules that must pass on every push."""

import pytest
import numpy as np

pytestmark = pytest.mark.ci


class TestTensorBasics:
    def test_tensor_import(self):
        from ravana_ml.tensor import StateTensor, RawTensor, tensor as t_utils
        t = StateTensor(np.array([1.0, 2.0]))
        assert isinstance(t, RawTensor)
        assert t.data.shape == (2,)

    def test_state_tensor_salience(self):
        from ravana_ml.tensor import StateTensor
        t = StateTensor(np.array([1.0]), salience=0.8)
        assert t.salience == 0.8

    def test_tensor_string(self):
        from ravana_ml.tensor import tensor
        val = tensor([1.0, 2.0, 3.0])
        assert val.data.shape == (3,)

    def test_tensor_zeros(self):
        from ravana_ml.tensor import RawTensor
        z = RawTensor.zeros(3, 4)
        assert z.shape == (3, 4)
        assert np.all(z.data == 0.0)

    def test_tensor_ops(self):
        from ravana_ml.tensor import RawTensor
        a = RawTensor(np.array([1.0, 2.0]))
        b = RawTensor(np.array([3.0, 4.0]))
        c = a + b
        assert np.allclose(c.data, [4.0, 6.0])


class TestFreeEnergy:
    def test_free_energy_import(self):
        from ravana_ml.free_energy import FreeEnergyAccumulator
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        assert fe.graph is not None
        assert fe.free_energy == 0.0

    def test_free_energy_update(self):
        from ravana_ml.free_energy import FreeEnergyAccumulator
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        initial = fe.free_energy
        fe.accumulate_semantic(0.5, salience=0.3)
        assert fe.free_energy > initial
        assert fe.total == fe.free_energy
        assert 0 <= fe.normalized <= 1.0

    def test_free_energy_decay(self):
        from ravana_ml.free_energy import FreeEnergyAccumulator
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_semantic(10.0)
        fe.decay(rate=0.5)
        assert fe.semantic_free_energy < 10.0
        
    def test_free_energy_report(self):
        from ravana_ml.free_energy import FreeEnergyAccumulator
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        report = fe.report()
        assert "free_energy" in report
        assert "semantic" in report
        assert "normalized" in report
        
    def test_needs_sleep(self):
        from ravana_ml.free_energy import FreeEnergyAccumulator
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        assert not fe.needs_sleep()  # Below threshold (10.0)
        # Accumulate enough: 50 * 0.3 = 15.0 > 10.0
        fe.accumulate_semantic(50.0)
        assert fe.needs_sleep()


class TestGraphBasics:
    def test_graph_import(self):
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        assert g.dim == 8

    def test_graph_add_node(self):
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        n = g.add_node(vector=np.random.randn(8).astype(np.float32), label="test")
        assert n.id in g.nodes
        assert n.label == "test"

    def test_graph_add_edge(self):
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.random.randn(8).astype(np.float32), label="a")
        n2 = g.add_node(vector=np.random.randn(8).astype(np.float32), label="b")
        g.add_edge(n1.id, n2.id, weight=0.5, relation_type="causal")
        e = g.get_edge(n1.id, n2.id)
        assert e is not None
        assert e.weight == 0.5

    def test_graph_find_similar(self):
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        v = np.random.randn(8).astype(np.float32)
        g.add_node(vector=v, label="original")
        results = g.find_similar(v, k=5)
        assert len(results) > 0

    def test_graph_activate(self):
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        n = g.add_node(vector=np.random.randn(8).astype(np.float32), label="test")
        g.activate(n.id, amount=1.0)
        assert g.nodes[n.id].activation > 0

    def test_graph_spread_activation_finegrained(self):
        """Spread activation using relation_type for fine-grained path (small graphs)."""
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.random.randn(8).astype(np.float32), label="a")
        n2 = g.add_node(vector=np.random.randn(8).astype(np.float32), label="b")
        g.add_edge(n1.id, n2.id, weight=0.8, relation_type="causal")
        g.activate(n1.id, amount=1.0)
        # Use relation_type to trigger fine-grained propagation path
        g.spread_activation(steps=2, k_active=3, decay=0.5, relation_type="causal")
        assert g.nodes[n2.id].activation > 0


class TestPlasticityBasics:
    def test_hebbian_import(self):
        from ravana_ml.plasticity import HebbianPlasticity
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        hp = HebbianPlasticity(graph=g, lr=0.01)
        assert hp.lr == 0.01
        assert hp.graph is g

    def test_hebbian_creates_edge_with_coactivation(self):
        from ravana_ml.plasticity import HebbianPlasticity
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        n2 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        hp = HebbianPlasticity(graph=g, lr=0.01)
        g.activate(n1.id, amount=2.0)
        g.activate(n2.id, amount=2.0)
        delta = hp.update(n1.id, n2.id)
        # Coactivation > 0.3 (2.0 * 2.0 = 4.0 > 0.3) should create edge
        edge = g.get_edge(n1.id, n2.id)
        assert edge is not None

    def test_structural_plasticity_step(self):
        from ravana_ml.plasticity import StructuralPlasticity
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        sp = StructuralPlasticity(graph=g, prune_threshold=0.05, form_threshold=0.5)
        pruned, formed = sp.step()
        assert pruned >= 0
        assert formed >= 0


class TestPropagationBasics:
    def test_propagation_import(self):
        from ravana_ml.propagation import PropagationEngine
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        pe = PropagationEngine(g)
        assert pe is not None
        assert pe.propagation_count == 0

    def test_propagation_basic(self):
        from ravana_ml.propagation import PropagationEngine
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.ones(8, dtype=np.float32), label="a")
        n2 = g.add_node(vector=np.ones(8, dtype=np.float32), label="b")
        g.add_edge(n1.id, n2.id, weight=0.5)
        pe = PropagationEngine(g)
        active = pe.propagate(input_vector=np.ones(8, dtype=np.float32), steps=2, k_active=3)
        assert len(active) > 0
        assert pe.propagation_count == 1

    def test_propagation_coherence(self):
        from ravana_ml.propagation import PropagationEngine
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        n2 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        g.add_edge(n1.id, n2.id, weight=0.7, confidence=0.8)
        pe = PropagationEngine(g)
        coherence = pe.measure_coherence([n1.id, n2.id])
        assert coherence > 0

    def test_propagation_prediction(self):
        from ravana_ml.propagation import PropagationEngine
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        n2 = g.add_node(vector=np.random.randn(8).astype(np.float32))
        g.add_edge(n1.id, n2.id, weight=0.8)
        pe = PropagationEngine(g)
        g.activate(n1.id, amount=1.0)
        predictions = pe.get_prediction([n1.id], top_k=5)
        assert len(predictions) > 0
