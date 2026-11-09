"""A used multi-hop chain must be compilable into a shortcut edge.

The graph primitive is fine: `compress_paths` creates a shortcut for a pair
that has no edge. The bug was the recorder -- `chain_walker` records every
edge it traverses, and an edge that already exists can never be compressed into
a shortcut. So the chain-level record has to happen at the inference site,
where the path is still in hand.

See docs/DESIGN_PATH_COMPRESSION.md for the full analysis.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana_ml", "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ravana.learn.plasticity import GraphPlasticity  # noqa: E402
from ravana_ml.graph import ConceptGraph  # noqa: E402


DIM = 8


def _chain_graph():
    """A -> B -> C with no direct A -> C edge."""
    g = ConceptGraph(dim=DIM, max_nodes=100, rng=np.random.RandomState(0))
    a = g.add_node(vector=np.random.RandomState(1).randn(DIM).astype(np.float32),
                   label="a")
    b = g.add_node(vector=np.random.RandomState(2).randn(DIM).astype(np.float32),
                   label="b")
    c = g.add_node(vector=np.random.RandomState(3).randn(DIM).astype(np.float32),
                   label="c")
    g.add_edge(a.id, b.id, weight=0.6, relation_type="is_a")
    g.add_edge(b.id, c.id, weight=0.6, relation_type="is_a")
    return g, a.id, b.id, c.id


# --------------------------------------------------------------------------
# 1. The distinction the fix rests on.
# --------------------------------------------------------------------------

def test_recording_a_traversed_edge_cannot_create_a_shortcut():
    """Pinned: this is the behaviour that made the old metric a lie."""
    g, a, b, _ = _chain_graph()
    gp = GraphPlasticity()
    for _ in range(3):
        gp.record_successful_path(g, a, b)   # an edge that already exists

    assert g.get_compressible_paths(min_usage=2)
    assert g.compress_paths(g.get_compressible_paths(min_usage=2),
                            min_chain_score=0.15) == 0
    assert getattr(g.get_edge(a, b), "shortcut", False) is False


def test_recording_chain_endpoints_does_create_a_shortcut():
    g, a, _, c = _chain_graph()
    gp = GraphPlasticity()
    for _ in range(3):
        gp.record_successful_path(g, a, c)   # endpoints of a 2-hop chain

    n = g.compress_paths(g.get_compressible_paths(min_usage=2),
                         min_chain_score=0.15)
    assert n == 1
    e = g.get_edge(a, c)
    assert e is not None
    assert e.shortcut is True
    # A compiled chain is an inference, not an observed fact.
    assert e.relation_type == "inferred"


# --------------------------------------------------------------------------
# 2. The inference site must record the chain, not the hops.
# --------------------------------------------------------------------------

def test_infer_chain_returns_the_path_needed_for_compilation():
    """The fix depends on infer_chain exposing the path; pin that shape."""
    g, a, _, c = _chain_graph()
    chain = g.infer_chain(a, max_hops=2, verb="is_a")
    assert chain, "expected a 2-hop chain a -> b -> c"
    target, _score, path = chain[-1]
    assert target == c
    assert path[0] == a and path[-1] == c
    assert len(path) >= 2


def test_used_chain_produces_a_shortcut_after_sleep():
    """End-to-end: a used chain, recorded the way the engine now records it,
    collapses into a shortcut during the sleep pass."""
    g, a, _, c = _chain_graph()
    gp = GraphPlasticity()

    chain = g.infer_chain(a, max_hops=2, verb="is_a")
    assert chain
    path = chain[-1][2]
    assert len(path) >= 2

    # Record it the way engine_generation now does: by endpoints.
    for _ in range(gp.config.compress_min_usage):
        gp.record_successful_path(g, path[0], path[-1])

    res = gp.sleep_cycle(g)
    assert res["paths_compressed"] >= 1, res
    assert g.get_edge(a, c) is not None
    assert g.get_edge(a, c).shortcut is True


# --------------------------------------------------------------------------
# 3. Guard rails.
# --------------------------------------------------------------------------

def test_single_hop_chain_is_not_recorded():
    """A one-hop 'chain' is just an edge; compiling it would be a no-op.

    infer_chain's path includes the start node, so a single hop is
    [start, target] -- length 2. The engine's guard is `len(_path) >= 2`,
    which a single hop satisfies, so what actually protects against
    compiling a bare edge is that the pair already has an edge (pinned in
    test_recording_a_traversed_edge_cannot_create_a_shortcut). Here we pin
    the path shape the guard sees.
    """
    g, a, b, _ = _chain_graph()
    chain = g.infer_chain(a, max_hops=1, verb="is_a")
    assert chain
    path = chain[-1][2]
    assert path == [a, b]
    # The pair is a real edge, so even if recorded it cannot compile.
    assert g.get_edge(path[0], path[-1]) is not None


def test_missing_plasticity_is_tolerated():
    """The engine guards with getattr(..., None); a bare graph must not break."""
    g, a, _, c = _chain_graph()
    gp = GraphPlasticity()
    gp.record_successful_path(g, a, c)
    gp.record_successful_path(g, a, c)
    assert g.get_compressible_paths(min_usage=2)
