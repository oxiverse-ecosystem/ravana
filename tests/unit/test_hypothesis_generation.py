"""Tests for ravana_grace.core.hypothesis_generation."""

import pytest
import numpy as np
from ravana_grace.core.hypothesis_generation import (
    HypothesisGenerator, GenerationConfig, GeneratedHypothesis,
    HypothesisType, HypothesisGeneration,
)


class TestGenerationConfig:
    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.kl_plateau_threshold == 0.05
        assert cfg.max_hypotheses == 6
        assert cfg.min_survival_score == 0.2
        assert len(cfg.preferred_types_order) == 6


class TestHypothesisGenerator:
    def test_init(self):
        hg = HypothesisGenerator()
        assert len(hg.hypotheses) == 0
        assert hg.generation_count == 0
        assert hg.last_generation_episode == -100

    def test_monitor_state_no_triggers(self):
        hg = HypothesisGenerator()
        result = hg.monitor_state(
            episode=1, kl_gain=0.5, uncertainty=0.1, dissonance=0.3,
            hypotheses=[],
        )
        assert "triggers_detected" in result
        assert "should_generate" in result

    def test_detect_triggers_kl_plateau(self):
        hg = HypothesisGenerator()
        for i in range(20):
            hg.kl_history.append(0.01)
        triggers = hg._detect_triggers(episode=1)
        assert len(triggers) > 0
        assert any("kl_plateau" in t for t in triggers)

    def test_can_generate_rate_limited(self):
        hg = HypothesisGenerator()
        hg.last_generation_episode = 100
        # 20 < default min_episodes_between_generations, so rate limited
        assert hg._can_generate(episode=120) is False

    def test_can_generate_at_capacity(self):
        hg = HypothesisGenerator()
        hg.hypotheses = {i: None for i in range(6)}
        assert hg._can_generate(episode=1) is False

    def test_generate_hypothesis(self):
        hg = HypothesisGenerator()
        h = hg.generate_hypothesis(episode=1, current_hypotheses=[], triggers=["kl_plateau"])
        assert h is not None
        assert isinstance(h, GeneratedHypothesis)
        assert h.hypothesis_type == HypothesisType.PARAMETRIC_TIME

    def test_generate_hypothesis_rate_limited(self):
        hg = HypothesisGenerator()
        hg.last_generation_episode = 100
        h = hg.generate_hypothesis(episode=105, current_hypotheses=[], triggers=["kl_plateau"])
        # Should be rate-limited (only 5 episodes since last)
        assert h is None

    def test_generate_progresses_through_types(self):
        hg = HypothesisGenerator()
        h1 = hg.generate_hypothesis(episode=100, current_hypotheses=[], triggers=["kl_plateau"])
        assert h1.hypothesis_type == HypothesisType.PARAMETRIC_TIME
        h2 = hg.generate_hypothesis(episode=200, current_hypotheses=[h1], triggers=["kl_plateau"])
        assert h2.hypothesis_type == HypothesisType.PARAMETRIC_STATE

    def test_prune_weak_hypotheses(self):
        hg = HypothesisGenerator()
        h = hg.generate_hypothesis(episode=1, current_hypotheses=[], triggers=["kl_plateau"])
        h.survival_score = 0.01
        h.evidence_count = 30
        pruned = hg._prune_weak_hypotheses(episode=200)
        assert len(pruned) > 0

    def test_get_active_hypotheses(self):
        hg = HypothesisGenerator()
        hg.generate_hypothesis(episode=1, current_hypotheses=[], triggers=["kl_plateau"])
        active = hg.get_active_hypotheses()
        assert len(active) == 1

    def test_get_generation_status(self):
        hg = HypothesisGenerator()
        hg.generate_hypothesis(episode=1, current_hypotheses=[], triggers=["kl_plateau"])
        status = hg.get_generation_status()
        assert "total_generated" in status
        assert "currently_active" in status
        assert status["currently_active"] == 1

    def test_generated_hypothesis_predict(self):
        hg = HypothesisGenerator()
        h = hg.generate_hypothesis(episode=1, current_hypotheses=[], triggers=["kl_plateau"])
        pred = h.predict_boundary(episode=50, context={"dissonance": 0.5})
        assert 0 < pred < 1.0

    def test_update_quality(self):
        hg = HypothesisGenerator()
        h = hg.generate_hypothesis(episode=1, current_hypotheses=[], triggers=["kl_plateau"])
        h.update_quality(prediction_error=0.1, episode=2)
        assert h.evidence_count == 1

    def test_hypothesis_generation_alias(self):
        assert HypothesisGeneration is HypothesisGenerator
