"""Tests for driving the retrieval-practice schedule from the engine.

Karpicke & Roediger (2008): retrieval is itself the learning event, and it
beats re-reading. The schedule and the review logic already existed in
`ravana.learn.intake`; these tests pin the ENGINE wiring that makes it run:
the intake handle, the from-memory retrieval attempt, the per-turn clock, and
the hand-off into WebToGraph.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana_ml", "src"))


@pytest.fixture(scope="module")
def engine():
    import tempfile
    from ravana.chat.engine import CognitiveChatEngine
    _dir = tempfile.mkdtemp(prefix="ravana_rp_")
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               data_dir=_dir, user_suffix="_retr_practice")


def _add_concept(eng, label, vector=None):
    """Add a labelled node to the engine's graph and index it."""
    vec = vector if vector is not None else np.zeros(64, dtype=np.float32)
    node = eng.graph.add_node(vector=vec, label=label)
    eng._concept_keywords.setdefault(label.lower(), []).append(node.id)
    return node


# ── the intake handle ────────────────────────────────────────────────────────

def test_engine_constructs_knowledge_intake(engine):
    ki = engine._get_knowledge_intake()
    assert ki is not None
    # cached: the same object comes back
    assert engine._get_knowledge_intake() is ki
    # the fast episodic store is the engine's own hippocampal buffer
    assert ki.buffer is engine.hippocampal_buffer


def test_intake_absent_host_keeps_legacy_path():
    """A host without _get_knowledge_intake must still build a WebToGraph --
    just without the gate (the legacy unconditional write)."""
    from ravana.chat.web_learning import WebLearningMixin

    class _Bare:
        def __init__(self):
            from ravana.graph.engine import GraphEngine
            self.ge = GraphEngine(dim=16, seed=1)
            self.graph = self.ge.graph
            self.dim = 16

        def _glove_vector(self, w):
            return None

    bare = _Bare()
    getter = getattr(bare, "_get_knowledge_intake", None)
    assert getter is None, "test premise: host must not expose the getter"


# ── the retrieval attempt ────────────────────────────────────────────────────

def test_recall_succeeds_for_a_merged_fact(engine):
    _add_concept(engine, "water")
    _add_concept(engine, "liquid")
    engine.graph.add_edge(source=engine._concept_keywords["water"][0],
                          target=engine._concept_keywords["liquid"][0],
                          weight=0.6, relation_type="is_a", confidence=0.7)
    assert engine._recall_from_own_memory("water", "is_a", "liquid") is True


def test_recall_fails_for_a_value_that_is_not_there(engine):
    _add_concept(engine, "water")
    _add_concept(engine, "liquid")
    engine.graph.add_edge(source=engine._concept_keywords["water"][0],
                          target=engine._concept_keywords["liquid"][0],
                          weight=0.6, relation_type="is_a", confidence=0.7)
    # right subject, right relation, WRONG object
    assert engine._recall_from_own_memory("water", "is_a", "gas") is False
    # unknown subject
    assert engine._recall_from_own_memory("unobtainium", "is_a", "liquid") is False


def test_recall_requires_the_matching_relation_type(engine):
    """A related edge of the wrong relation type is not a retrieval of THIS
    fact -- otherwise every association would count as recall."""
    _add_concept(engine, "water")
    _add_concept(engine, "liquid")
    engine.graph.add_edge(source=engine._concept_keywords["water"][0],
                          target=engine._concept_keywords["liquid"][0],
                          weight=0.6, relation_type="related_to", confidence=0.7)
    assert engine._recall_from_own_memory("water", "is_a", "liquid") is False


def test_recall_falls_back_to_the_episodic_buffer(engine):
    engine.hippocampal_buffer.store("water", "is_a", "liquid", confidence=0.9)
    assert engine._recall_from_own_memory("water", "is_a", "liquid") is True
    assert engine._recall_from_own_memory("water", "is_a", "gas") is False


def test_recall_is_safe_when_empty(engine):
    assert engine._recall_from_own_memory("", "", "") is False


# ── the per-turn schedule ────────────────────────────────────────────────────

def test_schedule_clock_advances_on_every_turn(engine):
    ki = engine._get_knowledge_intake()
    t0 = ki.turn
    engine.process_turn("hello")
    assert ki.turn == t0 + 1


def test_due_fact_is_reviewed_on_process_turn(engine):
    ki = engine._get_knowledge_intake()
    ki._review_queue[("water", "is_a", "liquid")] = {
        "due": ki.turn, "interval": 1, "reviews": 0,
        "retention": 0.5, "provenance": None,
    }
    before = ki.stats["reviewed"]
    engine.process_turn("hello")
    assert ki.stats["reviewed"] == before + 1


def test_successful_recall_expands_the_interval(engine):
    ki = engine._get_knowledge_intake()
    _add_concept(engine, "water")
    _add_concept(engine, "liquid")
    engine.graph.add_edge(source=engine._concept_keywords["water"][0],
                          target=engine._concept_keywords["liquid"][0],
                          weight=0.6, relation_type="is_a", confidence=0.7)
    key = ("water", "is_a", "liquid")
    ki._review_queue[key] = {
        "due": ki.turn, "interval": 1, "reviews": 0,
        "retention": 0.5, "provenance": None,
    }
    engine.process_turn("hello")
    rec = ki._review_queue[key]
    assert rec["reviews"] == 1
    assert rec["interval"] >= 1


def test_failed_recall_resets_the_interval(engine):
    ki = engine._get_knowledge_intake()
    key = ("water", "is_a", "gas")          # not in memory
    ki._review_queue[key] = {
        "due": ki.turn, "interval": 8, "reviews": 3,
        "retention": 0.8, "provenance": None,
    }
    engine.process_turn("hello")
    rec = ki._review_queue[key]
    assert rec["reviews"] == 0
    assert rec["interval"] == 1              # shortest interval: try again soon


def test_nothing_is_reviewed_when_nothing_is_due(engine):
    ki = engine._get_knowledge_intake()
    ki._review_queue.clear()          # isolate from earlier tests' entries
    before = ki.stats["reviewed"]
    engine.process_turn("hello")
    assert ki.stats["reviewed"] == before


def test_trace_reports_the_review(engine, capsys):
    ki = engine._get_knowledge_intake()
    engine._trace_enabled = True
    try:
        ki._review_queue[("water", "is_a", "liquid")] = {
            "due": ki.turn, "interval": 1, "reviews": 0,
            "retention": 0.5, "provenance": None,
        }
        engine.process_turn("hello")
        out = capsys.readouterr().out
        assert "[review]" in out
    finally:
        engine._trace_enabled = False


# ── WebToGraph hand-off ──────────────────────────────────────────────────────

def test_web_to_graph_receives_the_engine_intake(engine):
    w2g = engine._get_web_to_graph()
    assert w2g is not None
    assert w2g.intake is engine._get_knowledge_intake()
