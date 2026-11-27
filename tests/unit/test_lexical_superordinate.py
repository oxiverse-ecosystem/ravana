"""Unit tests for the default-off lexical-superordinate transfer primitive.

Pure tests only: no engine boot, no store write, no benchmark import.
"""
from __future__ import annotations

from ravana.learn import lexical_superordinate as ls


def _install_fake(monkeypatch, table):
    monkeypatch.setattr(ls, "path_similarity",
                        lambda a, b: table.get((a, b), table.get((b, a))))


def test_default_off_is_inert(monkeypatch, capsys):
    monkeypatch.delenv(ls.ENV_FLAG, raising=False)
    assert ls.enabled() is False
    assert ls.pick("tree", ["shrub", "petal"], [("a", "b")], 0.2) is None
    assert capsys.readouterr().out == ""


def test_flag_on_with_fake_similarity_emits_unique_winner(monkeypatch):
    monkeypatch.setenv(ls.ENV_FLAG, "1")
    # teaching pair similarity 0.40 -> theta = 0.50
    _install_fake(monkeypatch, {("a", "b"): 0.40,
                                ("tree", "shrub"): 0.70,
                                ("tree", "petal"): 0.20})
    assert ls.pick("tree", ["shrub", "petal"], [("a", "b")]) == "shrub"


def test_strict_margin_abstains(monkeypatch):
    monkeypatch.setenv(ls.ENV_FLAG, "1")
    _install_fake(monkeypatch, {("a", "b"): 0.40,
                                ("tree", "shrub"): 0.52,
                                ("tree", "petal"): 0.50})
    assert ls.pick("tree", ["shrub", "petal"], [("a", "b")]) is None


def test_negated_teaching_absent_criterion_abstains(monkeypatch):
    monkeypatch.setenv(ls.ENV_FLAG, "1")
    _install_fake(monkeypatch, {("tree", "shrub"): 0.90})
    assert ls.pick("tree", ["shrub", "petal"], [], None) is None


def test_oov_options_abstain(monkeypatch):
    monkeypatch.setenv(ls.ENV_FLAG, "1")
    _install_fake(monkeypatch, {("a", "b"): 0.40})
    assert ls.pick("tree", ["zzq_lex_a", "zzq_lex_b"], [("a", "b")]) is None


def test_candidate_order_invariance(monkeypatch):
    monkeypatch.setenv(ls.ENV_FLAG, "1")
    _install_fake(monkeypatch, {("a", "b"): 0.40,
                                ("tree", "shrub"): 0.70,
                                ("tree", "petal"): 0.20})
    first = ls.pick("tree", ["shrub", "petal"], [("a", "b")])
    second = ls.pick("tree", ["petal", "shrub"], [("a", "b")])
    assert first == second == "shrub"


def test_demo_scores_are_subject_independent(monkeypatch):
    monkeypatch.setenv(ls.ENV_FLAG, "1")
    _install_fake(monkeypatch, {("flower", "shrub"): 0.80,
                                ("flower", "petal"): 0.10})
    scores = ls.demo_scores([("seed", "flower")], ["shrub", "petal"])
    assert scores == {"shrub": 0.80, "petal": 0.10}


def test_answer_key_is_not_a_parameter():
    """Public API has no slot for gold/type_sets/world_model/domain metadata."""
    import inspect
    params = inspect.signature(ls.pick).parameters
    assert set(params) == {"subject", "options", "teaching_pairs", "fallback_theta"}
