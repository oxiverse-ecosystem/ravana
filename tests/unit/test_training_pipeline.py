"""Tests for ravana_grace.training.pipeline."""

import pytest
from unittest.mock import MagicMock
from ravana_grace.training.pipeline import TrainingPipeline, TrainingConfig


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.total_episodes == 100000
        assert cfg.log_interval == 100
        assert cfg.debug_first_n == 50
        assert cfg.initial_difficulty == 0.3


class TestTrainingPipeline:
    def test_init(self):
        manager = MagicMock()
        pipeline = TrainingPipeline(manager)
        assert pipeline.manager is manager
        assert pipeline.config is not None

    def test_init_with_config(self):
        manager = MagicMock()
        cfg = TrainingConfig(total_episodes=500, log_interval=50)
        pipeline = TrainingPipeline(manager, cfg)
        assert pipeline.config.total_episodes == 500

    def test_compute_difficulty_before_ramp(self):
        manager = MagicMock()
        pipeline = TrainingPipeline(manager)
        difficulty = pipeline._compute_difficulty(episode=0)
        assert difficulty == 0.3

    def test_compute_difficulty_mid_ramp(self):
        manager = MagicMock()
        pipeline = TrainingPipeline(manager)
        difficulty = pipeline._compute_difficulty(episode=25000)
        assert 0.3 < difficulty < 0.9

    def test_compute_difficulty_at_max(self):
        manager = MagicMock()
        pipeline = TrainingPipeline(manager)
        difficulty = pipeline._compute_difficulty(episode=50000)
        assert difficulty == 0.9

    def test_compute_difficulty_after_max(self):
        manager = MagicMock()
        pipeline = TrainingPipeline(manager)
        difficulty = pipeline._compute_difficulty(episode=60000)
        assert difficulty == 0.9

    def test_simulate_outcome_low_difficulty(self):
        manager = MagicMock()
        pipeline = TrainingPipeline(manager)
        # Run multiple times to check it returns bool
        results = [pipeline._simulate_outcome(0.3) for _ in range(50)]
        assert all(isinstance(r, bool) for r in results)

    def test_simulate_outcome_high_difficulty(self):
        manager = MagicMock()
        pipeline = TrainingPipeline(manager)
        results = [pipeline._simulate_outcome(0.9) for _ in range(50)]
        assert all(isinstance(r, bool) for r in results)

    def test_assert_state_valid_pass(self):
        manager = MagicMock()
        manager.state.dissonance = 0.5
        manager.state.identity = 0.5
        manager.state.accumulated_wisdom = 0.5
        manager.governor.config.max_dissonance = 0.95
        manager.governor.config.min_dissonance = 0.15
        manager.governor.config.min_identity = 0.10
        pipeline = TrainingPipeline(manager)
        # Should not raise
        pipeline._assert_state_valid()

    def test_assert_state_valid_breach(self):
        manager = MagicMock()
        manager.state.dissonance = 0.99  # Above max
        manager.state.identity = 0.5
        manager.governor.config.max_dissonance = 0.95
        manager.governor.config.min_dissonance = 0.15
        manager.governor.config.min_identity = 0.10
        pipeline = TrainingPipeline(manager)
        with pytest.raises(AssertionError):
            pipeline._assert_state_valid()

    def test_generate_summary(self):
        manager = MagicMock()
        manager.get_status.return_value = {
            "state": {"dissonance": 0.5, "identity": 0.5},
            "governor": {},
            "resolution": {},
            "identity": {},
        }
        manager.governor.get_clamp_report.return_value = "clamp report"
        manager.governor.get_clamp_metrics.return_value = {"total": 0}
        pipeline = TrainingPipeline(manager)
        summary = pipeline._generate_summary(elapsed=10.5)
        assert "total_episodes" in summary
        assert "final_state" in summary
        assert "clamp_metrics" in summary

    def test_log_progress(self):
        manager = MagicMock()
        manager.state.dissonance = 0.5
        manager.state.identity = 0.5
        manager.state.accumulated_wisdom = 1.0
        manager.governor.config.max_dissonance = 0.95
        pipeline = TrainingPipeline(manager)
        # Should not crash
        pipeline._log_progress(episode=100, record={"mode": "EXPLORE_SAFE"}, difficulty=0.5)
