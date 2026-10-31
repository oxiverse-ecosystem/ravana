import pytest
from ravana.chat.self_directed_learning import SelfDirectedLearner, KnowledgeGap


def test_knowledge_gap_to_dict():
    g = KnowledgeGap("gravity", 0.1, 2.0)
    d = g.to_dict()
    assert d["topic"] == "gravity"
    assert d["confidence"] == 0.1


def test_learner_no_engine_cycle():
    l = SelfDirectedLearner(None)
    out = l.run_cycle()
    assert "gaps" in out
    assert "hypotheses" in out
    assert "queries" in out
    assert out["gaps"] == []


def test_form_hypothesis():
    l = SelfDirectedLearner(None)
    g = KnowledgeGap("entropy", 0.2, 1.0)
    h = l.form_hypothesis(g)
    assert h["topic"] == "entropy"
    assert "entropy" in h["claim"]
    assert len(l.hypotheses) == 1


def test_experiment_no_engine():
    l = SelfDirectedLearner(None)
    g = KnowledgeGap("x", 0.1, 0.0)
    r = l.experiment(g)
    assert r["topic"] == "x"
    assert "improved" in r
