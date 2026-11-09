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


def test_repeated_decay_is_self_retarding():
    """Power-law forgetting: the RELATIVE rate falls as the trace ages.

    This is Jost's second law (1897) and the property that distinguishes a power
    law from an exponential (Wixted & Ebbesen 1991, 1997). An exponential would
    keep losing the same proportion every cycle and eventually cross any floor;
    a power law loses a shrinking proportion and asymptotes instead.

    So the assertion is on the RATIO of successive losses, not on reaching a
    floor: each cycle must remove a smaller fraction than the one before.
    """
    g, (a, b, _) = _graph()
    _link(g, a, b, conf=0.9)
    e = g.get_edge(a, b)
    e._forget_age = 0.0

    losses = []
    for _ in range(12):
        before = float(e.confidence)
        g.decay_edges(turns_elapsed=5)
        after = float(e.confidence)
        losses.append((before - after) / max(before, 1e-9))

    # Monotone decreasing relative loss, and the last is far smaller than the
    # first -- the signature of a power law.
    assert losses[0] > losses[-1], f"relative loss did not fall: {losses}"
    assert losses[-1] < losses[0] * 0.5, (
        f"relative loss barely fell: first={losses[0]:.6f} last={losses[-1]:.6f}")
    # And it never crosses the floor.
    assert float(e.confidence) >= 0.05


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
# 2b. The rate is dynamic, not flat.
# ─────────────────────────────────────────────────────────────────────────────

def test_rate_is_higher_for_a_labile_edge():
    """Fusi's bounded-synapse result: signal decays as q*exp(-q*t), so a
    highly labile synapse forgets faster."""
    g, (a, b, _) = _graph()
    e1 = _link(g, a, b, conf=0.8)
    g._forgetting_base_rate = 0.02
    cold = g._edge_forgetting_rate(e1, competitors=0, structural_importance=0.0)
    for _ in range(6):
        g.mark_rehearsed(a, b)
    warm = g._edge_forgetting_rate(e1, competitors=0, structural_importance=0.0)
    assert cold > warm, "a rehearsed edge must be more rigid (slower decay)"


def test_rate_is_higher_with_more_competitors():
    """Forgetting is driven by competitors sharing a cue, not by time passing
    (Anderson, Bjork & Bjork 1994; Fusi: memories decay because other memories
    overwrite them)."""
    g, ids = _graph(5)
    e = _link(g, ids[0], ids[1], conf=0.8)
    g._forgetting_base_rate = 0.02
    few = g._edge_forgetting_rate(e, competitors=0, structural_importance=0.0)
    many = g._edge_forgetting_rate(e, competitors=9, structural_importance=0.0)
    assert many > few, "more siblings sharing the cue must mean faster decay"


def test_rate_is_lower_for_a_structurally_important_edge():
    """SHY ultrastructure: the largest synapses are the ones spared from
    downscaling."""
    g, (a, b, _) = _graph()
    e = _link(g, a, b, conf=0.8)
    g._forgetting_base_rate = 0.02
    plain = g._edge_forgetting_rate(e, competitors=0, structural_importance=0.0)
    critical = g._edge_forgetting_rate(e, competitors=0, structural_importance=0.9)
    assert critical < plain, "a structurally critical edge must decay slower"


def test_rate_is_lower_for_a_strongly_learned_edge():
    """Degree of learning (Wixted 2022): a higher degree of initial learning
    yields a slower RELATIVE rate of forgetting -- stronger memories decay
    slower proportionally, not just from a higher start."""
    g, (a, b, _) = _graph()
    weak = _link(g, a, b, conf=0.2)
    g._forgetting_base_rate = 0.02
    r_weak = g._edge_forgetting_rate(weak, competitors=0, structural_importance=0.0)
    strong = _link(g, a, b, conf=0.95)
    r_strong = g._edge_forgetting_rate(strong, competitors=0, structural_importance=0.0)
    assert r_strong < r_weak, (
        f"a well-learned trace must forget slower in proportion: "
        f"{r_strong} vs {r_weak}")


def test_older_traces_forget_slower_in_proportion():
    """Jost's second law, end to end: of two traces at equal strength, the
    older loses a smaller FRACTION of itself per cycle."""
    g, ids = _graph(4)
    young = _link(g, ids[0], ids[1], conf=0.8)
    old = _link(g, ids[2], ids[3], conf=0.8)
    young._forget_age = 0.0
    old._forget_age = 200.0

    y0, o0 = float(young.confidence), float(old.confidence)
    g.decay_edges(turns_elapsed=5)
    y_loss = (y0 - float(young.confidence)) / y0
    o_loss = (o0 - float(old.confidence)) / o0

    assert o_loss < y_loss, (
        f"the older trace must forget slower in proportion: "
        f"old={o_loss:.6f} young={y_loss:.6f}")


def test_rates_differ_between_edges_in_one_graph():
    """The whole point: a flat rate cannot distinguish two edges that differ in
    lability, competition and importance."""
    g, ids = _graph(5)
    hub = _link(g, ids[0], ids[1], conf=0.8)
    lonely = _link(g, ids[2], ids[3], conf=0.8)
    # Give the hub many siblings; leave the lonely edge alone.
    for i in range(4):
        _link(g, ids[0], ids[1], conf=0.4) if g.get_edge(ids[0], ids[1]) else None
    g._forgetting_base_rate = 0.02
    r_hub = g._edge_forgetting_rate(hub, competitors=10, structural_importance=0.0)
    r_lonely = g._edge_forgetting_rate(lonely, competitors=0, structural_importance=0.0)
    assert r_hub > r_lonely


def test_interference_actually_changes_outcomes():
    """End-to-end: two identical edges, one under competition and one not,
    must end up at different confidence after the same decay pass."""
    g, ids = _graph(6)
    crowded = _link(g, ids[0], ids[1], conf=0.8)
    quiet = _link(g, ids[2], ids[3], conf=0.8)
    # Add siblings to the crowded edge's source only.
    for tgt in (ids[4], ids[5]):
        g.add_edge(ids[0], tgt, weight=0.5, relation_type="is_a", confidence=0.6)

    g.decay_edges(turns_elapsed=10)

    assert float(crowded.confidence) < float(quiet.confidence), (
        "the edge under retrieval competition must forget faster")


def test_importance_actually_changes_outcomes():
    g, ids = _graph(4)
    important = _link(g, ids[0], ids[1], conf=0.8)
    ordinary = _link(g, ids[2], ids[3], conf=0.8)
    si = {(ids[0], ids[1]): 0.95}

    g.decay_edges(turns_elapsed=10, structural_importance=si)

    assert float(important.confidence) > float(ordinary.confidence)


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
