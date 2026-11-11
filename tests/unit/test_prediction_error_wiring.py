"""Prediction error must reach the graph's free-energy accumulator.

`ConceptGraph.apply_prediction_error` and `apply_free_energy` maintain
`contradiction_hotspots`, which three sleep stages consume:
`form_inhibitory_edges`, hotspot-driven `should_split`, and
`reconcile_contradictions`. `PredictiveCodingLearner.learn_node` updated
`node.prediction_free_energy` and `node.contradiction_count` but never called
`apply_free_energy`, so `contradiction_hotspots` stayed permanently empty and
those stages were unreachable from the live chat path.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana_ml", "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402
from ravana.core.predictive_coding import PredictiveCodingLearner  # noqa: E402
from ravana.learn.plasticity import GraphPlasticity, PlasticityConfig  # noqa: E402


DIM = 64


def _engine(**kw):
    """A fresh engine on an ISOLATED data dir.

    These tests assert properties of a CLEAN graph (no contradiction hotspots,
    no accumulated prediction error). Constructing the engine without a
    ``data_dir`` makes it load the developer's live ``weights/ravana_weights.pkl``
    -- a real, learned snapshot -- which already carries hotspots and
    contradiction counts from actual use. The assertions then fail on state the
    test never created (measured: 2 hotspots and 18 nodes with nonzero
    ``contradiction_count`` in the committed weights). A temp dir gives each
    test the clean graph it is reasoning about.
    """
    import tempfile
    kw.setdefault("data_dir", tempfile.mkdtemp(prefix="ravana_pew_"))
    return CognitiveChatEngine(dim=DIM, seed=7, baby_mode=True, **kw)


def _random_ctx(seed=0):
    v = np.random.RandomState(seed).randn(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------
# 1. learn_node must feed the accumulator.
# --------------------------------------------------------------------------

def test_learn_node_populates_contradiction_hotspots():
    eng = _engine()
    g = eng.graph
    assert len(g.contradiction_hotspots) == 0

    ctx = _random_ctx()
    nid = next(iter(g.nodes))
    node = g.get_node(nid)
    before_fe = float(node.prediction_free_energy)

    # apply_free_energy adds amount * salience * (1 - confidence) and flags a
    # hotspot above 2.0; with salience 0.3 / confidence 0.1 a single error is
    # not enough, so drive it repeatedly -- which is the real condition.
    for _ in range(6):
        eng.predictive_coding_learner.learn_node(nid, ctx, node.vector)

    assert float(node.prediction_free_energy) > before_fe
    # The whole point: the error must land where the sleep stages look.
    assert nid in g.contradiction_hotspots, (
        "learn_node updated node free energy but never called "
        "apply_free_energy, so contradiction_hotspots stays empty and the "
        "inhibitory/split/reconcile sleep stages are unreachable")


def test_repeated_prediction_error_escalates_to_a_hotspot():
    """A concept that keeps being wrong must become a hotspot."""
    eng = _engine()
    g = eng.graph
    ctx = _random_ctx(1)
    nid = next(iter(g.nodes))
    node = g.get_node(nid)

    for _ in range(8):
        eng.predictive_coding_learner.learn_node(nid, ctx, node.vector)

    assert int(node.contradiction_count) > 0
    assert nid in g.contradiction_hotspots


def test_no_hotspot_without_prediction_error():
    """A quiet concept must not be flagged."""
    eng = _engine()
    g = eng.graph
    assert len(g.contradiction_hotspots) == 0
    # Sanity: merely constructing the engine does not create hotspots.
    for nid, node in list(g.nodes.items())[:5]:
        assert float(node.prediction_free_energy) == 0.0


# --------------------------------------------------------------------------
# 2. The sleep stages that consume hotspots must now fire.
# --------------------------------------------------------------------------

def test_reconciliation_fires_on_a_hotspot():
    eng = _engine()
    g = eng.graph
    gp = GraphPlasticity()
    ctx = _random_ctx(2)
    # Drive a node that has INCOMING edges: reconcile_contradictions skips a
    # hotspot with no neighbours (`if not neighbors: continue`), so a leaf node
    # can be flagged but never reconciled.
    nid = None
    for cand, n in g.nodes.items():
        if g.get_incoming(cand):
            nid = cand
            break
    assert nid is not None, "no node with incoming edges to drive"
    node = g.get_node(nid)
    for _ in range(6):
        eng.predictive_coding_learner.learn_node(nid, ctx, node.vector)
    assert nid in g.contradiction_hotspots, (
        f"node {nid} was driven 6 times but not flagged; "
        f"hotspots={sorted(g.contradiction_hotspots)}")

    res = gp.sleep_cycle(g)
    assert res["contradictions_reconciled"] >= 1, res


def test_inhibitory_edges_form_when_targets_are_active():
    """Pathway 1 of form_inhibitory_edges needs target.activation > 0.1,
    which is the state the turn leaves behind before sleep."""
    eng = _engine()
    g = eng.graph
    gp = GraphPlasticity()
    ctx = _random_ctx(3)
    activated = list(g.nodes.keys())[:10]
    for _ in range(6):
        for nid in activated:
            n = g.get_node(nid)
            if n is not None and n.vector is not None:
                eng.predictive_coding_learner.learn_node(nid, ctx, n.vector)
    assert g.contradiction_hotspots

    # Reproduce the turn's activation state (chain_walker.activate).
    for nid in activated:
        g.activate(nid, 1.0)

    res = gp.sleep_cycle(g)
    assert res["inhibitory_edges_formed"] >= 1, res


def test_hotspot_driven_split_is_reachable():
    """Splitting must be reachable from accumulated contradiction, not only
    from vector drift."""
    eng = _engine()
    g = eng.graph
    gp = GraphPlasticity(PlasticityConfig(split_concepts=True))
    ctx = _random_ctx(4)
    activated = list(g.nodes.keys())[:10]
    for _ in range(8):
        for nid in activated:
            n = g.get_node(nid)
            if n is not None and n.vector is not None:
                eng.predictive_coding_learner.learn_node(nid, ctx, n.vector)

    over = [nid for nid in activated
            if int(getattr(g.get_node(nid), "contradiction_count", 0))
            >= gp.config.split_contradiction_threshold]
    assert over, "no node reached the split contradiction threshold"

    res = gp.sleep_cycle(g)
    assert res["concepts_split"] >= 1, res


# --------------------------------------------------------------------------
# 3. The plasticity wrapper's own entry point still works.
# --------------------------------------------------------------------------

def test_on_prediction_error_reaches_the_graph():
    eng = _engine()
    g = eng.graph
    gp = GraphPlasticity()
    ctx = _random_ctx(5)
    nids = list(g.nodes.keys())[:5]

    applied = gp.on_prediction_error(g, nids, ctx)
    assert applied == len(nids)
    assert gp.stats["prediction_errors_applied"] == len(nids)
    for nid in nids:
        assert float(getattr(g.get_node(nid), "contradiction_free_energy", 0.0)) > 0.0


def test_on_prediction_error_is_a_noop_without_nodes():
    eng = _engine()
    gp = GraphPlasticity()
    assert gp.on_prediction_error(eng.graph, [], _random_ctx()) == 0
