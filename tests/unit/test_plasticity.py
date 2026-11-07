"""Tests for the graph-plasticity wiring.

Every mechanism here is the ConceptGraph's OWN rule; these tests check that
the chat engine's wiring reaches it, that the brain's gating is applied
(surprise-gated per turn, rate-limited at sleep), and that the counters and
budget behave.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana_ml", "src"))

from ravana.graph.engine import GraphEngine  # noqa: E402
from ravana.learn.plasticity import (  # noqa: E402
    GraphPlasticity, PlasticityConfig,
)


DIM = 8


def _engine(labels=()):
    ge = GraphEngine(dim=DIM, seed=3)
    for lab in labels:
        n = ge.graph.add_node(vector=np.zeros(DIM, dtype=np.float32), label=lab)
        ge._all_labels[lab] = n.id
    return ge


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-15)


def _link(ge, a, b, weight=0.5, rel="semantic", conf=0.6):
    return ge.graph.add_edge(source=ge._all_labels[a], target=ge._all_labels[b],
                             weight=weight, relation_type=rel, confidence=conf)


def _orthogonal_pair(ge, a, b):
    """Give two nodes orthogonal unit vectors and return them."""
    va = _unit([1.0] + [0.0] * (DIM - 1))
    vb = _unit([0.0, 1.0] + [0.0] * (DIM - 2))
    ge.graph.get_node(ge._all_labels[a]).vector = va
    ge.graph.get_node(ge._all_labels[a]).core_vector = va.copy()
    ge.graph.get_node(ge._all_labels[b]).vector = vb
    ge.graph.get_node(ge._all_labels[b]).core_vector = vb.copy()
    return va, vb


# ── per-turn: surprise gating ────────────────────────────────────────────────

def test_predictable_edge_is_not_updated():
    """An edge whose weight already matches the observed similarity is left
    alone -- plasticity is gated by surprise, not by exposure."""
    ge = _engine(["a", "b"])
    va, vb = _orthogonal_pair(ge, "a", "b")
    e = _link(ge, "a", "b", weight=0.0)      # predicts orthogonality exactly
    gp = GraphPlasticity()
    err = gp.on_edge_traversed(ge.graph, e, va, vb, salience=1.0)
    assert err == 0.0
    assert e.weight == 0.0
    assert gp.stats["edge_updates"] == 0
    assert gp.stats["edge_updates_skipped_predictable"] == 1


def test_surprising_edge_is_corrected_toward_observation():
    ge = _engine(["a", "b"])
    va, vb = _orthogonal_pair(ge, "a", "b")
    e = _link(ge, "a", "b", weight=0.9)      # claims strong relatedness
    gp = GraphPlasticity()
    err = gp.on_edge_traversed(ge.graph, e, va, vb, salience=1.0)
    assert err > 0.05
    assert e.weight < 0.9                      # moved toward the observed 0.0
    assert e.prediction_count == 1
    assert gp.stats["edge_updates"] == 1


def test_correction_is_salience_gated():
    ge = _engine(["a", "b", "c", "d"])
    va, vb = _orthogonal_pair(ge, "a", "b")
    vc, vd = _orthogonal_pair(ge, "c", "d")
    e1 = _link(ge, "a", "b", weight=0.9)
    e2 = _link(ge, "c", "d", weight=0.9)
    gp = GraphPlasticity()

    gp.on_edge_traversed(ge.graph, e1, va, vb, salience=1.0)
    gp.reset_budget()
    gp.on_edge_traversed(ge.graph, e2, vc, vd, salience=0.0)

    # salience 0 -> no movement at all
    assert e2.weight == 0.9
    assert e1.weight < 0.9


def test_update_budget_bounds_a_single_pass():
    ge = _engine([f"n{i}" for i in range(24)])
    va = _unit([1.0] + [0.0] * (DIM - 1))
    vb = _unit([0.0, 1.0] + [0.0] * (DIM - 2))
    edges = []
    for i in range(24):
        ge.graph.get_node(ge._all_labels[f"n{i}"]).vector = va if i % 2 == 0 else vb
        if i % 2 == 0:
            e = _link(ge, f"n{i}", f"n{i+1}", weight=0.9)
            edges.append(e)
    assert len(edges) == 12

    gp = GraphPlasticity(PlasticityConfig(edge_pe_max_updates_per_turn=8))
    gp.reset_budget()
    for e in edges:
        gp.on_edge_traversed(ge.graph, e, va, vb, salience=1.0)
    assert gp.stats["edge_updates"] == 8          # capped


def test_learning_can_be_disabled():
    ge = _engine(["a", "b"])
    va, vb = _orthogonal_pair(ge, "a", "b")
    e = _link(ge, "a", "b", weight=0.9)
    gp = GraphPlasticity(PlasticityConfig(edge_pe_learning=False))
    assert gp.on_edge_traversed(ge.graph, e, va, vb, salience=1.0) == 0.0
    assert e.weight == 0.9


# ── per-turn: prediction error reaches the graph ────────────────────────────

def test_prediction_error_reaches_graph():
    ge = _engine(["a"])
    node = ge.graph.get_node(ge._all_labels["a"])
    node.vector = _unit([1.0] + [0.0] * (DIM - 1))
    actual = _unit([0.0, 1.0] + [0.0] * (DIM - 2))    # orthogonal -> error 1.0

    gp = GraphPlasticity()
    n = gp.on_prediction_error(ge.graph, [node.id], actual)
    assert n == 1
    assert node.contradiction_count >= 1
    assert node.prediction_free_energy > 0.0
    assert gp.stats["prediction_errors_applied"] == 1


def test_empty_prediction_list_is_a_noop():
    ge = _engine(["a"])
    gp = GraphPlasticity()
    assert gp.on_prediction_error(ge.graph, [], np.zeros(DIM, dtype=np.float32)) == 0


# ── path recording -> compression ────────────────────────────────────────────

def test_recorded_paths_are_compressed_at_sleep():
    """Compression creates a shortcut for a repeatedly-used INDIRECT path."""
    ge = _engine(["a", "b", "c"])
    _link(ge, "a", "b")
    _link(ge, "b", "c")
    gp = GraphPlasticity()
    a, c = ge._all_labels["a"], ge._all_labels["c"]

    gp.record_successful_path(ge.graph, a, c)
    assert gp.stats["paths_recorded"] == 1
    # below the min_usage threshold -> nothing to compress yet
    out = gp.sleep_cycle(ge.graph)
    assert out.get("paths_compressed", 0) == 0
    assert ge.graph.get_edge(a, c) is None

    gp.record_successful_path(ge.graph, a, c)
    out = gp.sleep_cycle(ge.graph)
    assert out.get("paths_compressed", 0) >= 1
    assert ge.graph.get_edge(a, c) is not None          # shortcut created


# ── sleep: vector consolidation ─────────────────────────────────────────────

def test_sleep_consolidates_fast_toward_core():
    """The two-timescale rule: the fast active vector moves toward the slow
    core vector (not toward genesis -- drift_magnitude measures genesis)."""
    ge = _engine(["a"])
    node = ge.graph.get_node(ge._all_labels["a"])
    core = _unit([1.0] + [0.0] * (DIM - 1))
    node.core_vector = core.copy()
    node.vector = _unit([0.0, 1.0] + [0.0] * (DIM - 2))   # maximally drifted
    ge.graph._activated_since_sleep.add(node.id)

    before = float(np.dot(node.vector, node.core_vector))
    gp = GraphPlasticity()
    gp.sleep_cycle(ge.graph)
    after = float(np.dot(node.vector, node.core_vector))
    assert after > before
    assert after > 0.0


# ── sleep: contradiction reconciliation ──────────────────────────────────────

def test_sleep_reconciles_contradiction_hotspots():
    """reconcile_contradictions needs an incoming neighbourhood to act on."""
    ge = _engine(["a", "b"])
    _link(ge, "b", "a", weight=0.6)
    node = ge.graph.get_node(ge._all_labels["a"])
    node.contradiction_count = 5
    node.prediction_free_energy = 4.0
    ge.graph.contradiction_hotspots.add(node.id)

    gp = GraphPlasticity()
    out = gp.sleep_cycle(ge.graph)
    assert out.get("contradictions_reconciled", 0) >= 1
    assert node.prediction_free_energy < 4.0


# ── sleep: inhibitory edges ──────────────────────────────────────────────────

def test_sleep_forms_inhibitory_edges_between_competing_targets():
    ge = _engine(["a", "b", "c"])
    _link(ge, "a", "b", weight=0.6)
    _link(ge, "a", "c", weight=0.6)
    node = ge.graph.get_node(ge._all_labels["a"])
    node.contradiction_count = 12          # >= 3*threshold -> mutual inhibition
    node.contradiction_free_energy = 3.0
    node.free_energy_gradient = 1.0
    ge.graph.contradiction_hotspots.add(node.id)

    gp = GraphPlasticity()
    out = gp.sleep_cycle(ge.graph)
    assert out.get("inhibitory_edges_formed", 0) >= 1
    assert ge.graph.get_edge(ge._all_labels["b"], ge._all_labels["c"]) is not None


# ── sleep: concept splitting ─────────────────────────────────────────────────

def test_sleep_splits_a_contradicted_concept():
    ge = _engine(["a", "b", "c"])
    _link(ge, "a", "b", weight=0.6)
    _link(ge, "a", "c", weight=0.6)
    node = ge.graph.get_node(ge._all_labels["a"])
    node.contradiction_count = 8

    n_before = len(ge.graph.nodes)
    gp = GraphPlasticity()
    out = gp.sleep_cycle(ge.graph)
    assert out.get("concepts_split", 0) >= 1
    assert len(ge.graph.nodes) > n_before


def test_split_budget_is_bounded():
    ge = _engine([f"n{i}" for i in range(10)])
    for i in range(9):
        _link(ge, f"n{i}", f"n{i+1}", weight=0.6)
        ge.graph.get_node(ge._all_labels[f"n{i}"]).contradiction_count = 8

    gp = GraphPlasticity(PlasticityConfig(max_splits_per_cycle=2))
    out = gp.sleep_cycle(ge.graph)
    assert out.get("concepts_split", 0) <= 2


def test_no_split_when_signal_is_absent():
    ge = _engine(["a", "b"])
    _link(ge, "a", "b", weight=0.6)
    gp = GraphPlasticity()
    out = gp.sleep_cycle(ge.graph)
    assert out.get("concepts_split", 0) == 0


# ── serialisation ────────────────────────────────────────────────────────────

def test_homeostatic_downscale_is_wired_into_sleep():
    """The SHY downscale must be reached from the sleep path, and it must
    compute the graph's own structural importance (the SI analogue).

    Whether a given edge actually moves is the graph's own scoring decision;
    the wiring claim is that the call happens and the importance is computed.
    """
    ge = _engine(["a", "b", "c"])
    _link(ge, "a", "b", weight=0.6)
    _link(ge, "a", "c", weight=0.6)
    assert not getattr(ge.graph, "_structural_importance_cache", None)

    gp = GraphPlasticity()
    out = gp.sleep_cycle(ge.graph)
    assert "edges_downscaled" in out
    assert getattr(ge.graph, "_structural_importance_cache", None), \
        "sleep did not compute structural importance"
    assert gp.stats["si_edges_protected"] >= 1


def test_structural_protection_shields_critical_edges():
    """With structural protection on, the graph computes its own importance
    and refuses to downscale edges it flags as critical."""
    ge = _engine(["a", "b", "c", "d"])
    _link(ge, "a", "b", weight=0.6)
    _link(ge, "a", "c", weight=0.6)
    _link(ge, "b", "d", weight=0.6)
    _link(ge, "c", "d", weight=0.6)
    before = {k: float(e.weight) for k, e in ge.graph.edges.items()}

    gp = GraphPlasticity()          # structural_protection defaults to 0.5
    out = gp.sleep_cycle(ge.graph)
    assert out.get("si_edges_protected", 0) >= 1
    after = {k: float(e.weight) for k, e in ge.graph.edges.items()}
    # every edge is structurally critical in this small dense graph, so the
    # protection keeps them all at their original weight
    assert all(abs(after[k] - before[k]) < 1e-6 for k in before)


def test_structural_importance_feeds_the_ewc_penalty():
    """The Synaptic-Intelligence analogue must reach `fisher_importance`,
    which is what `hebbian_update`'s EWC penalty actually reads."""
    ge = _engine(["a", "b", "c", "d"])
    _link(ge, "a", "b", weight=0.6)
    _link(ge, "a", "c", weight=0.6)
    _link(ge, "b", "d", weight=0.6)
    _link(ge, "c", "d", weight=0.6)
    for e in ge.graph.edges.values():
        e.fisher_importance = 0.0

    gp = GraphPlasticity()
    out = gp.sleep_cycle(ge.graph)
    assert out.get("si_edges_protected", 0) >= 1
    assert any(float(getattr(e, "fisher_importance", 0.0)) > 0.0
               for e in ge.graph.edges.values())


def test_state_roundtrip():
    gp = GraphPlasticity()
    gp.stats["edge_updates"] = 7
    gp2 = GraphPlasticity()
    gp2.set_state(gp.get_state())
    assert gp2.stats["edge_updates"] == 7


# ── integration: the engine constructs it and uses it on retrieval ───────────

def test_chain_walker_wires_plasticity_on_traversal():
    """_spread_and_collect must reach the graph's own prediction-error rule
    for the edges that actually carried signal."""
    from ravana.chat.chain_walker import ChainWalkerMixin

    class _Stub(ChainWalkerMixin):
        def __init__(self):
            self.ge = _engine(["water", "liquid"])
            self.graph = self.ge.graph
            self._all_labels = self.ge._all_labels
            va = _unit([1.0] + [0.0] * (DIM - 1))
            vb = _unit([0.0, 1.0] + [0.0] * (DIM - 2))
            self.graph.get_node(self._all_labels["water"]).vector = va
            self.graph.get_node(self._all_labels["liquid"]).vector = vb
            # claims strong relatedness between orthogonal concepts
            self.edge = _link(self.ge, "water", "liquid", weight=0.9)

        def _glove_vector(self, w):
            return None

    stub = _Stub()
    before = stub.edge.weight
    stub._spread_and_collect([stub._all_labels["water"]])

    assert stub.edge.weight < before, "traversed surprising edge was not corrected"
    gp = stub._get_graph_plasticity()
    assert gp.stats["edge_updates"] >= 1
    assert gp.stats["paths_recorded"] >= 1


def test_chain_walker_leaves_predictable_edges_alone():
    from ravana.chat.chain_walker import ChainWalkerMixin

    class _Stub(ChainWalkerMixin):
        def __init__(self):
            self.ge = _engine(["water", "liquid"])
            self.graph = self.ge.graph
            self._all_labels = self.ge._all_labels
            va = _unit([1.0] + [0.0] * (DIM - 1))
            vb = _unit([0.0, 1.0] + [0.0] * (DIM - 2))
            self.graph.get_node(self._all_labels["water"]).vector = va
            self.graph.get_node(self._all_labels["liquid"]).vector = vb
            self.edge = _link(self.ge, "water", "liquid", weight=0.0)

        def _glove_vector(self, w):
            return None

    stub = _Stub()
    stub._spread_and_collect([stub._all_labels["water"]])
    assert stub.edge.weight == 0.0
