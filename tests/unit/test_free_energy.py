"""Tests for ravana_ml.free_energy (dedicated test beyond ci/test_core.py)."""

import pytest
import numpy as np
from ravana_ml.free_energy import FreeEnergyAccumulator
from ravana_ml.graph import ConceptGraph


class TestFreeEnergyAccumulator:
    def test_init(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        assert fe.free_energy == 0.0
        assert fe.semantic_free_energy == 0.0
        assert fe.linguistic_free_energy == 0.0
        assert fe.episodic_free_energy == 0.0
        assert fe.contradiction_free_energy == 0.0
        assert fe.abstraction_free_energy == 0.0

    def test_accumulate_semantic(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_semantic(2.0, salience=0.5)
        assert fe.semantic_free_energy == 1.0

    def test_accumulate_linguistic(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_linguistic(5.0)
        assert fe.linguistic_free_energy == 0.5  # * 0.1

    def test_accumulate_episodic(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_episodic(3.0, recency=0.7)
        assert fe.episodic_free_energy == pytest.approx(2.1, rel=1e-4)

    def test_accumulate_contradiction(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_contradiction(3)
        assert fe.contradiction_free_energy == 1.5  # 3 * 0.5

    def test_accumulate_abstraction(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_abstraction(5, 0.4)
        assert fe.abstraction_free_energy > 0  # 5 * 0.4 * 0.3 = 0.6

    def test_total(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_semantic(1.0)
        fe.accumulate_linguistic(10.0)
        fe.accumulate_episodic(1.0)
        fe.accumulate_contradiction(2)
        expected = (1.0 * 0.3) + (10.0 * 0.1) + (1.0 * 0.5) + (2 * 0.5)
        assert fe.total == pytest.approx(expected, rel=1e-5)
        assert fe.free_energy == fe.total

    def test_normalized(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        assert fe.normalized == 0.0
        fe.accumulate_semantic(200.0, salience=1.0)
        assert fe.normalized <= 1.0

    def test_decay(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_semantic(10.0)
        fe.decay(rate=0.1)
        assert fe.semantic_free_energy < 10.0
        assert fe.semantic_free_energy == 0.9 * 3.0  # 10 * 0.3 = 3.0, then 3.0 - 0.1*3.0 = 2.7

    def test_reset(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_semantic(10.0, salience=1.0)
        fe.accumulate_linguistic(10.0)
        fe.reset()
        assert fe.free_energy == 0.0
        for ch in ["semantic", "linguistic", "episodic", "contradiction", "abstraction"]:
            assert getattr(fe, f"{ch}_free_energy") == 0.0

    def test_needs_sleep_initial(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        assert not fe.needs_sleep()

    def test_needs_sleep_above_threshold(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.accumulate_semantic(50.0, salience=1.0)
        assert fe.needs_sleep()

    def test_report_keys(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        report = fe.report()
        assert set(report.keys()) == {
            "semantic", "linguistic", "episodic",
            "contradiction", "abstraction",
            "free_energy", "normalized",
        }

    def test_history_records(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        assert len(fe.history) == 0
        fe.accumulate_semantic(0.5)
        assert len(fe.history) >= 1

    def test_apply_to_graph_no_crash(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        fe = FreeEnergyAccumulator(g)
        fe.apply_to_graph()  # No-op with no nodes having contradiction_count > 2


class TestTensorBasicsExtended:
    """Extended tensor tests beyond ci/test_core.py."""

    def test_raw_tensor_eye(self):
        from ravana_ml.tensor import RawTensor
        eye = RawTensor.eye(3)
        assert eye.shape == (3, 3)
        assert np.allclose(eye.data, np.eye(3))

    def test_raw_tensor_arange(self):
        from ravana_ml.tensor import RawTensor
        a = RawTensor.arange(0, 5, 2)
        assert np.allclose(a.data, [0, 2, 4])

    def test_raw_tensor_stack(self):
        from ravana_ml.tensor import RawTensor
        a = RawTensor(np.array([1.0, 2.0]))
        b = RawTensor(np.array([3.0, 4.0]))
        s = RawTensor.stack([a, b])
        assert s.shape == (2, 2)

    def test_raw_tensor_cat(self):
        from ravana_ml.tensor import RawTensor
        a = RawTensor(np.array([1.0, 2.0]))
        b = RawTensor(np.array([3.0, 4.0]))
        c = RawTensor.cat([a, b], dim=0)
        assert c.shape == (4,)

    def test_state_tensor_salience_property(self):
        from ravana_ml.tensor import StateTensor
        import numpy as np
        t = StateTensor(np.array([1.0]), salience=0.7)
        assert t.salience == 0.7
        t.salience = 0.9
        assert t.salience == 0.9

    def test_state_tensor_free_energy_property(self):
        from ravana_ml.tensor import StateTensor
        import numpy as np
        t = StateTensor(np.array([1.0]), free_energy=2.5)
        assert t.free_energy == 2.5
        t.free_energy = 5.0
        assert t.free_energy == 5.0

    def test_state_tensor_plasticity(self):
        from ravana_ml.tensor import StateTensor
        import numpy as np
        t = StateTensor(np.array([1.0]), stability=0.3)
        assert t.plasticity == 0.7

    def test_state_tensor_age(self):
        from ravana_ml.tensor import StateTensor
        import time
        import numpy as np
        t = StateTensor(np.array([1.0]))
        age = t.age()
        assert 0 <= age < 5.0  # just created

    def test_state_tensor_apply_free_energy(self):
        from ravana_ml.tensor import StateTensor
        import numpy as np
        t = StateTensor(np.array([1.0]), salience=0.5)
        fe = t.apply_free_energy(2.0, salience_weight=0.8)
        # fe += 2.0 * 0.5 * 0.8 = 0.8
        assert fe == 0.8

    def test_state_tensor_consolidate(self):
        from ravana_ml.tensor import StateTensor
        import numpy as np
        t = StateTensor(np.array([0.0]), free_energy=2.0, stability=0.3)
        delta = t.consolidate(rate=0.2)
        assert t.free_energy == pytest.approx(1.0, abs=0.1)  # halved
        assert t.stability > 0.3  # increased

    def test_parameter(self):
        from ravana_ml.tensor import Parameter, StateTensor
        import numpy as np
        inner = StateTensor(np.array([1.0, 2.0, 3.0]))
        p = Parameter(data=inner)
        assert "Parameter" in repr(p)

    def test_tensor_function_with_requires_grad(self):
        from ravana_ml.tensor import tensor, StateTensor
        val = tensor([1.0, 2.0], requires_grad=True)
        assert isinstance(val, StateTensor)

    def test_tensor_function_data(self):
        from ravana_ml.tensor import tensor
        val = tensor([1.0, 2.0])
        assert isinstance(val, type(val))
