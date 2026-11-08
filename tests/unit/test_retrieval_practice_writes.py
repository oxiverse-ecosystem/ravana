"""Retrieval practice must modify memory, not just measure it.

Karpicke & Roediger (2008): retrieval is itself the learning event. The
schedule in `KnowledgeIntake` was a pure measurement -- `record_review` updated
the interval and a retention counter, but never touched the graph edge, so a
fact that was successfully recalled again and again was no stronger than one
that was never reviewed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana_ml", "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ravana.graph.engine import GraphEngine  # noqa: E402
from ravana.learn.intake import KnowledgeIntake, Provenance  # noqa: E402
from ravana.web.openie import Fact  # noqa: E402


DIM = 16


def _engine(labels=()):
    ge = GraphEngine(dim=DIM, seed=7)
    for lab in labels:
        node = ge.graph.add_node(vector=np.zeros(DIM, dtype=np.float32),
                                 label=lab)
        ge._all_labels[lab] = node.id
    return ge


def _link(ge, subj, obj, rel="is_a", conf=0.6, weight=0.5):
    a = ge._all_labels[subj]
    b = ge._all_labels[obj]
    return ge.graph.add_edge(source=a, target=b, weight=weight,
                             relation_type=rel, confidence=conf)


def _edge(ge, subj, obj):
    return ge.graph.get_edge(ge._all_labels[subj], ge._all_labels[obj])


def _fact(s, r, o, conf=0.9):
    return Fact(subject=s, relation=r, obj=o, confidence=conf)


# --------------------------------------------------------------------------
# 1. A successful retrieval strengthens the trace.
# --------------------------------------------------------------------------

def test_successful_retrieval_writes_to_the_edge():
    ge = _engine(["photosynthesis", "process"])
    ki = KnowledgeIntake(ge=ge)
    ki.intake(_fact("photosynthesis", "is_a", "process"),
              Provenance(source_type="web"))

    e = _edge(ge, "photosynthesis", "process")
    before = float(e.confidence)
    key = ("photosynthesis", "is_a", "process")
    assert key in ki._review_queue

    ki.record_review(key, recalled=True)

    assert float(e.confidence) > before, (
        "a successful retrieval must strengthen the trace")
    assert ki.stats["review_writes"] == 1
    meta = getattr(e, "source_metadata", None) or {}
    assert meta.get("retrievals") == 1


def test_failed_retrieval_does_not_strengthen():
    """Reactivating a trace that could not be recovered is not evidence the
    trace is correct -- writing on failure would reinforce a wrong value."""
    ge = _engine(["photosynthesis", "process"])
    ki = KnowledgeIntake(ge=ge)
    ki.intake(_fact("photosynthesis", "is_a", "process"),
              Provenance(source_type="web"))

    e = _edge(ge, "photosynthesis", "process")
    before = float(e.confidence)
    ki.record_review(("photosynthesis", "is_a", "process"), recalled=False)

    assert float(e.confidence) == before
    assert ki.stats["review_writes"] == 0
    assert ki.stats["review_failed_no_write"] == 1


# --------------------------------------------------------------------------
# 2. Effort matters: a hard retrieval strengthens more than an easy one.
# --------------------------------------------------------------------------

def test_effortful_retrieval_strengthens_more():
    """Bjork & Bjork (1994) desirable difficulty: the gain is largest when
    retrieval was effortful, i.e. after a long interval."""
    def gain_for_interval(interval):
        ge = _engine(["photosynthesis", "process"])
        ki = KnowledgeIntake(ge=ge)
        ki.intake(_fact("photosynthesis", "is_a", "process"),
                  Provenance(source_type="web"))
        e = _edge(ge, "photosynthesis", "process")
        before = float(e.confidence)
        key = ("photosynthesis", "is_a", "process")
        ki._review_queue[key]["interval"] = interval
        ki.record_review(key, recalled=True)
        return float(e.confidence) - before

    easy = gain_for_interval(1)
    hard = gain_for_interval(16)
    assert hard > easy, (f"effortful retrieval should gain more: {hard} vs {easy}")


def test_repeated_review_converges():
    """Repeated review must converge, not pin the edge at 1.0."""
    ge = _engine(["photosynthesis", "process"])
    ki = KnowledgeIntake(ge=ge)
    ki.intake(_fact("photosynthesis", "is_a", "process"),
              Provenance(source_type="web"))
    e = _edge(ge, "photosynthesis", "process")
    key = ("photosynthesis", "is_a", "process")
    for _ in range(12):
        ki.record_review(key, recalled=True)
    assert float(e.confidence) <= 1.0
    assert float(e.confidence) < 1.0, "review must converge below the ceiling"


# --------------------------------------------------------------------------
# 3. The write is scoped to the reviewed edge.
# --------------------------------------------------------------------------

def test_review_only_touches_the_named_edge():
    ge = _engine(["photosynthesis", "process", "chemical", "light"])
    ki = KnowledgeIntake(ge=ge)
    ki.intake(_fact("photosynthesis", "is_a", "process"),
              Provenance(source_type="web"))
    ki.intake(_fact("photosynthesis", "has_property", "chemical"),
              Provenance(source_type="web"))

    e1 = _edge(ge, "photosynthesis", "process")
    e2 = _edge(ge, "photosynthesis", "chemical")
    b1, b2 = float(e1.confidence), float(e2.confidence)

    ki.record_review(("photosynthesis", "is_a", "process"), recalled=True)

    assert float(e1.confidence) > b1
    assert float(e2.confidence) == b2, "review must not touch other edges"


def test_review_of_absent_edge_is_a_noop():
    ge = _engine(["photosynthesis", "process"])
    ki = KnowledgeIntake(ge=ge)
    ki.intake(_fact("photosynthesis", "is_a", "process"),
              Provenance(source_type="web"))
    # A key that was never scheduled.
    ki.record_review(("photosynthesis", "is_a", "nonexistent"), recalled=True)
    assert ki.stats["review_writes"] == 0


def test_review_without_graph_is_a_noop():
    """A buffer-only intake (no graph) must not crash on review."""
    ki = KnowledgeIntake(ge=None)
    ki._review_queue[("a", "is_a", "b")] = {
        "due": 0, "interval": 1, "reviews": 0, "retention": 0.5}
    ki.record_review(("a", "is_a", "b"), recalled=True)
    assert ki.stats["review_writes"] == 0
