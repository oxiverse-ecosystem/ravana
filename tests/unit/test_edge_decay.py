"""Forgetting: SHY-style down-selection with rehearsal rescue.

The graph could previously only strengthen. `ConceptEdge` had no decay
mechanism at all -- unlike `Binding`, which has `decay()`, `last_used` and a
`strength` property. So "retention" was not a variable: every fact ever written
was retained forever, and the calibration question (does effortful retrieval
help?) had nothing to act on, because the effect being studied is *differential
survival under decay*.

The brain does not work that way. The synaptic homeostasis hypothesis
(Tononi & Cirelli) holds that wake produces net potentiation and sleep performs
a selective down-selection: most synapses weaken, reactivated ones are spared.
Ultrastructurally ~80% of synapses shrink proportionally to their size and the
largest are spared (de Vivo et al.), which is why the decay here is
multiplicative rather than subtractive.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana_ml", "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ravana_ml.graph import ConceptGraph  # noqa: E402
from ravana.learn.plasticity import GraphPlasticity, PlasticityConfig  # noqa: E402


DIM = 8


def _graph(n=3):
    g = ConceptGraph(dim=DIM, max_nodes=100, rng=np.random.RandomState(0))
    ids = []
    for i in range(n):
        node = g.add_node(
            vector=np.random.RandomState(i + 1).randn(DIM).astype(np.float32),
            label=f"n{i}")
        ids.append(node.id)
    return g, ids


def _link(g, a, b, conf=0.8):
    return g.add_edge(a, b, weight=0.6, relation_type="is_a", confidence=conf)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Edges forget.
# ─────────────────────────────────────────────────────────────────────────────

def test_an_unused_edge_decays():
    g, (a, b, _) = _graph()
    _link(g, a, b, conf=0.8)
    before = float(g.get_edge(a, b).confidence)

    changed = g.decay_edges(turns_elapsed=10)

    assert changed == 1
    assert float(g.get_edge(a, b).confidence) < before


def test_decay_is_multiplicative_not_subtractive():
    """SHY ultrastructure: synapses shrink proportional to their size, so a
    strong trace loses more absolute confidence than a weak one but keeps its
    rank. A subtractive rule would invert that."""
    g, ids = _graph(4)
    strong = _link(g, ids[0], ids[1], conf=0.9)
    weak = _link(g, ids[0], ids[2], conf=0.4)
    s0, w0 = float(strong.confidence), float(weak.confidence)

    g.decay_edges(turns_elapsed=10)

    s1, w1 = float(strong.confidence), float(weak.confidence)
    assert (s0 - s1) > (w0 - w1), "decay must be proportional to strength"
    assert s1 > w1, "relative ordering must be preserved"


def test_repeated_decay_converges_to_the_floor():
    g, (a, b, _) = _graph()
    _link(g, a, b, conf=0.9)
    # Exponential decay: many cycles to approach the floor, and it must
    # asymptote there rather than cross it.
    prev = float(g.get_edge(a, b).confidence)
    for _ in range(400):
        g.decay_edges(turns_elapsed=10)
        cur = float(g.get_edge(a, b).confidence)
        assert cur <= prev + 1e-12, "decay must be monotone"
        prev = cur
    assert 0.05 <= float(g.get_edge(a, b).confidence) < 0.06


def test_no_decay_without_elapsed_turns():
    g, (a, b, _) = _graph()
    _link(g, a, b, conf=0.8)
    assert g.decay_edges(turns_elapsed=0) == 0
    assert float(g.get_edge(a, b).confidence) == 0.8


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rehearsal rescues.
# ─────────────────────────────────────────────────────────────────────────────

def test_a_recently_rehearsed_edge_is_spared():
    g, (a, b, _) = _graph()
    _link(g, a, b, conf=0.8)
    g.mark_rehearsed(a, b)

    changed = g.decay_edges(turns_elapsed=10)

    assert changed == 0
    assert float(g.get_edge(a, b).confidence) == 0.8


def test_rehearsal_slows_future_decay():
    """Each rescue raises the halflife, so a well-rehearsed trace forgets
    slower -- but is not immortal."""
    def remaining(rehearsals):
        g, (a, b, _) = _graph()
        _link(g, a, b, conf=0.8)
        for _ in range(max(1, rehearsals)):
            g.mark_rehearsed(a, b)
        # Move the stamp out of the protection window so decay can act.
        g.get_edge(a, b).last_rehearsal -= 10 * 3600
        g.decay_edges(turns_elapsed=10)
        return float(g.get_edge(a, b).confidence)

    low = remaining(0)
    high = remaining(6)
    assert high > low, f"rehearsal must slow decay: {high} vs {low}"


def test_rehearsal_does_not_make_a_trace_immortal():
    g, (a, b, _) = _graph()
    _link(g, a, b, conf=0.9)
    for _ in range(20):
        g.mark_rehearsed(a, b)
    g.get_edge(a, b).last_rehearsal -= 10 * 3600
    for _ in range(100):
        g.decay_edges(turns_elapsed=10)
    assert float(g.get_edge(a, b).confidence) < 0.9


def test_mark_rehearsed_on_missing_edge_is_a_noop():
    g, ids = _graph()
    assert g.mark_rehearsed(ids[0], ids[1]) is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Wiring: the sleep pass actually forgets.
# ─────────────────────────────────────────────────────────────────────────────

def test_sleep_cycle_decays_edges():
    g, (a, b, _) = _graph()
    _link(g, a, b, conf=0.8)
    gp = GraphPlasticity()
    before = float(g.get_edge(a, b).confidence)

    res = gp.sleep_cycle(g)

    assert res["edges_decayed"] >= 1, res
    assert float(g.get_edge(a, b).confidence) < before


def test_decay_can_be_disabled():
    g, (a, b, _) = _graph()
    _link(g, a, b, conf=0.8)
    gp = GraphPlasticity(PlasticityConfig(decay_edges=False))
    res = gp.sleep_cycle(g)
    assert res.get("edges_decayed", 0) == 0
    assert float(g.get_edge(a, b).confidence) == 0.8


def test_retrieval_practice_rescues_from_decay():
    """The point of the whole mechanism: a fact that is reviewed survives,
    an identical fact that is not reviewed decays."""
    from ravana.graph.engine import GraphEngine
    from ravana.learn.intake import KnowledgeIntake, Provenance
    from ravana.web.openie import Fact

    def build():
        ge = GraphEngine(dim=DIM, seed=7)
        for lab in ("photosynthesis", "process"):
            node = ge.graph.add_node(
                vector=np.zeros(DIM, dtype=np.float32), label=lab)
            ge._all_labels[lab] = node.id
        ki = KnowledgeIntake(ge=ge)
        ki.intake(Fact("photosynthesis", "is_a", "process", 0.9),
                  Provenance(source_type="web"))
        return ge, ki

    reviewed, ki_r = build()
    ignored, _ = build()

    key = ("photosynthesis", "is_a", "process")
    for _ in range(5):
        ki_r.record_review(key, recalled=True)

    e_r = reviewed.graph.get_edge(reviewed._all_labels["photosynthesis"],
                                  reviewed._all_labels["process"])
    e_i = ignored.graph.get_edge(ignored._all_labels["photosynthesis"],
                                 ignored._all_labels["process"])
    c_r0, c_i0 = float(e_r.confidence), float(e_i.confidence)

    # Sleep: the reviewed trace is spared, the ignored one decays.
    gp = GraphPlasticity()
    gp.sleep_cycle(reviewed.graph)
    gp.sleep_cycle(ignored.graph)

    c_r1, c_i1 = float(e_r.confidence), float(e_i.confidence)
    assert c_r1 >= c_r0, "a rehearsed trace must not decay"
    assert c_i1 < c_i0, "an unrehearsed trace must decay"
