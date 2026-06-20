"""Tests for ravana_grace.core.occam_layer."""

import pytest
import numpy as np
from ravana_grace.core.occam_layer import (
    OccamLayer, OccamConfig, HypothesisScore, DisciplinedBeliefSystem,
)


class TestOccamConfig:
    def test_defaults(self):
        cfg = OccamConfig()
        assert cfg.lambda_penalty == 0.3
        assert cfg.max_hypotheses == 5
        assert cfg.min_age_to_prune == 20
        assert cfg.prune_threshold == -0.5


class TestOccamLayer:
    def test_init(self):
        ol = OccamLayer()
        assert ol.pruned_hypotheses == []
        assert ol.config is not None

    def test_score_hypothesis_object(self):
        ol = OccamLayer()
        h = type("Hyp", (), {"complexity_score": 0.5, "id": 1})()
        score = ol.score_hypothesis(
            hypothesis=h, explanatory_power=0.8,
            evidence_count=20, age=30,
        )
        assert isinstance(score, HypothesisScore)
        assert score.occam_score < score.raw_score  # Penalty applied

    def test_score_hypothesis_dict(self):
        ol = OccamLayer()
        h = {"complexity_score": 0.3, "id": 2}
        score = ol.score_hypothesis(
            hypothesis=h, explanatory_power=0.7,
            evidence_count=30, age=50,
        )
        assert score.occam_score < score.raw_score

    def test_score_hypothesis_low_evidence(self):
        ol = OccamLayer()
        h = {"complexity_score": 0.9, "id": 3}
        score = ol.score_hypothesis(
            hypothesis=h, explanatory_power=0.8,
            evidence_count=5, age=10,
        )
        # Low evidence = no penalty yet
        assert score.occam_score == score.raw_score

    def test_select_best_hypothesis(self):
        ol = OccamLayer()
        scores = [
            HypothesisScore(hypothesis_id="1", raw_score=0.8, occam_score=0.7,
                          explanatory_power=0.8, complexity=0.3, stability=0.6,
                          evidence_count=10, age=20, penalty_applied=0.1, reason=""),
            HypothesisScore(hypothesis_id="2", raw_score=0.6, occam_score=0.4,
                          explanatory_power=0.6, complexity=0.5, stability=0.6,
                          evidence_count=10, age=20, penalty_applied=0.2, reason=""),
        ]
        best = ol.select_best_hypothesis(scores)
        assert best is not None
        assert best.hypothesis_id == "1"

    def test_select_best_hypothesis_empty(self):
        ol = OccamLayer()
        assert ol.select_best_hypothesis([]) is None

    def test_identify_pruning_candidates(self):
        ol = OccamLayer()
        scores = [
            HypothesisScore(hypothesis_id="1", raw_score=0.8, occam_score=0.7,
                          explanatory_power=0.8, complexity=0.3, stability=0.6,
                          evidence_count=10, age=30, penalty_applied=0.1, reason=""),
            HypothesisScore(hypothesis_id="2", raw_score=0.1, occam_score=-0.6,
                          explanatory_power=0.1, complexity=0.8, stability=0.6,
                          evidence_count=20, age=30, penalty_applied=0.7, reason=""),
        ]
        candidates = ol.identify_pruning_candidates(scores)
        assert "2" in candidates

    def test_detect_overfitting_insufficient_data(self):
        ol = OccamLayer()
        assert ol.detect_overfitting({"id": 1}, [0.5]*5) is False

    def test_detect_overfitting_high_variance(self):
        ol = OccamLayer()
        scores = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
        assert ol.detect_overfitting({"id": 1}, scores) is True

    def test_get_discipline_status(self):
        ol = OccamLayer()
        status = ol.get_discipline_status()
        assert "lambda_penalty" in status
        assert "pruned_count" in status
        assert "overfitting_alerts" in status


class TestDisciplinedBeliefSystem:
    def test_init(self):
        belief = type("B", (), {"hypotheses": [], "current_belief": 0.5, "current_uncertainty": 0.3})()
        gen = type("G", (), {})()
        dbs = DisciplinedBeliefSystem(belief, gen)
        assert dbs.belief is belief
        assert dbs.occam is not None

    def test_score_all_hypotheses_empty(self):
        belief = type("B", (), {"hypotheses": [], "current_belief": 0.5, "current_uncertainty": 0.3})()
        gen = type("G", (), {})()
        dbs = DisciplinedBeliefSystem(belief, gen)
        scores = dbs.score_all_hypotheses(episode=1)
        assert scores == []

    def test_score_all_hypotheses_with_data(self):
        hyp = type("H", (), {
            "id": 1, "boundary_estimate": 0.5, "evidence_count": 20,
            "birth_episode": 1, "confidence": 0.6,
        })()
        belief = type("B", (), {"hypotheses": [hyp], "current_belief": 0.5, "current_uncertainty": 0.1})()
        gen = type("G", (), {})()
        dbs = DisciplinedBeliefSystem(belief, gen)
        scores = dbs.score_all_hypotheses(episode=50)
        assert len(scores) == 1

    def test_should_generate_new_at_capacity(self):
        belief = type("B", (), {"hypotheses": [], "current_belief": 0.5, "current_uncertainty": 0.5})()
        gen = type("G", (), {})()
        dbs = DisciplinedBeliefSystem(belief, gen)
        dbs.pruned_ids = set()
        assert dbs.should_generate_new([], episode=1) is True

    def test_get_status(self):
        belief = type("B", (), {"hypotheses": [], "current_belief": 0.5, "current_uncertainty": 0.3})()
        gen = type("G", (), {})()
        dbs = DisciplinedBeliefSystem(belief, gen)
        status = dbs.get_status()
        assert "total_hypotheses" in status
        assert "pruned_hypotheses" in status
