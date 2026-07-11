"""Phase 2 regression tests: graph pruning + pattern-separation pre-add filter.

Verifies the association-source cleanup that stops off-frame drift
("whales -> deer/pygmy") from polluting the narrative generator's
associations:

1. prune_low_quality_edges removes orphan/noisy SEMANTIC edges that were never
   verified (prediction_count < K) and are tagged co_occurrence / auto_expand
   (or are weak/dormant), while KEEPING verified web_facts and edges that have
   been successfully predicted (prediction_count >= K).
2. auto_expand_concepts applies a top-K nearest-neighbour pre-add filter
   (pattern separation): a new concept is wired only to its closest neighbours,
   so "whale -> deer" (animals cluster in embedding space but deer is not
   whale's nearest) is NOT wired, while "whale -> mammal" (genuinely close) is.

Run from repo root:
    python -m pytest tests/unit/test_phase2_pruning.py -v
"""
import os
import sys
import numpy as np

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.graph.engine import GraphEngine


def _add(ge, a, b, relation_type="semantic", confidence=0.5, kind=None, pc=0):
    """Add a semantic edge and optionally set provenance / prediction_count."""
    e = ge.graph.add_edge(a, b, weight=0.4, relation_type=relation_type,
                           confidence=confidence)
    if e is not None:
        e.prediction_count = pc
        if kind is not None:
            e.source_metadata.update({"edge_kind": kind})
    return e


def test_prune_removes_auto_expand_and_cooccurrence_keeps_webfact():
    ge = GraphEngine(dim=64, seed=7)
    n_whale = ge.graph.add_node(label="whale").id
    n_deer = ge.graph.add_node(label="deer").id
    n_mammal = ge.graph.add_node(label="mammal").id
    n_fact = ge.graph.add_node(label="factnode").id
    n_pred = ge.graph.add_node(label="prednode").id

    _add(ge, n_whale, n_deer, kind="auto_expand", pc=0)      # noise -> prune
    _add(ge, n_whale, n_mammal, kind="co_occurrence", pc=0)   # noise -> prune
    _add(ge, n_whale, n_fact, kind="web_fact", pc=0)         # verified -> KEEP
    _add(ge, n_whale, n_pred, kind="auto_expand", pc=2)      # predicted -> KEEP

    n_before = len(ge.graph.edges)
    pruned = ge.prune_low_quality_edges()
    n_after = len(ge.graph.edges)

    assert pruned == 2, f"expected 2 pruned (auto_expand + co_occurrence), got {pruned}"
    assert n_after == n_before - 2
    assert ge.graph.get_edge(n_whale, n_deer) is None, "auto_expand edge should be pruned"
    assert ge.graph.get_edge(n_whale, n_mammal) is None, "co_occurrence edge should be pruned"
    assert ge.graph.get_edge(n_whale, n_pred) is not None, "predicted edge must survive"
    assert ge.graph.get_edge(n_whale, n_fact) is not None, "web_fact must survive"


def _vec_table():
    """Deterministic 16-dim vectors: mammal is whale's nearest; ~10 distractors
    are closer to whale than deer; deer is only moderately similar (sim>0.4 so
    legacy sim>0.5 wiring WOULD have included it, but top-K excludes it)."""
    rng = np.random.RandomState(0)
    vecs = {}
    whale = np.zeros(16); whale[0] = 1.0
    vecs["whale"] = whale
    mammal = np.zeros(16); mammal[0] = 0.9; mammal[1] = 0.1
    vecs["mammal"] = mammal
    deer = np.zeros(16); deer[0] = 0.5; deer[1] = 0.5
    vecs["deer"] = deer
    vecs["ocean"] = np.array([0.95] + [0.0]*15)
    for i in range(10):  # distractors all closer to whale than deer
        v = np.zeros(16); v[0] = 0.85; v[2 + i % 12] = 0.1 * rng.randn()
        vecs[f"distractor{i}"] = v
    for k in vecs:
        vecs[k] = vecs[k] / (np.linalg.norm(vecs[k]) + 1e-12)
    return vecs


def test_pattern_separation_blocks_whale_deer_keeps_mammal():
    """auto_expand_concepts must wire whale->mammal (genuine neighbour) but NOT
    whale->deer (off-frame animal co-occurrence).

    auto_expand only wires NEW words to PRE-EXISTING graph nodes, so we pre-add
    the neighbours (with deterministic vectors) and then expand "whale".
    """
    ge = GraphEngine(dim=16, seed=3)
    vecs = _vec_table()
    ge._glove_vector = lambda w: vecs.get(w)
    # Pre-add neighbours with vectors so they are "existing" wiring targets.
    for w in list(vecs.keys()):
        if w == "whale":
            continue
        ge.graph.add_node(vector=vecs[w], label=w)
    ge.auto_expand_concepts("whale")  # whale is new -> wires to pre-existing

    whale_nid = ge._concept_keywords.get("whale", [None])[0]
    assert whale_nid is not None, "whale node should exist after auto_expand"
    labels = set()
    for tid, _e in ge.graph.get_outgoing(whale_nid):
        tnode = ge.graph.get_node(tid)
        if tnode and tnode.label:
            labels.add(tnode.label)
    # whale->mammal is a genuine near neighbour (sim~0.99) -> wired
    assert "mammal" in labels, f"whale->mammal should be wired; got {sorted(labels)}"
    # deer is an off-frame animal co-occurrence (sim~0.71): under pattern
    # separation (top-K nearest) it must NOT be wired, even though legacy
    # sim>0.5 would have wired it.
    assert "deer" not in labels, f"whale->deer off-frame edge must be blocked; got {sorted(labels)}"


def test_prune_disabled_is_noop():
    ge = GraphEngine(dim=64, seed=9)
    n1 = ge.graph.add_node(label="aaa").id
    n2 = ge.graph.add_node(label="bbb").id
    _add(ge, n1, n2, kind="auto_expand", pc=0)
    before = len(ge.graph.edges)
    assert ge.prune_low_quality_edges(enabled=False) == 0
    assert len(ge.graph.edges) == before


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
