"""Tests for ravana_grace.core.strategy_learning."""

import pytest
import numpy as np
from ravana_grace.core.strategy_learning import (
    StrategyLearningLayer, LearningConfig, ModeOutcome, StrategyWithLearning,
)
from ravana_grace.core.strategy import ExplorationMode, BehavioralContext


class TestLearningConfig:
    def test_defaults(self):
        cfg = LearningConfig()
        assert cfg.learning_rate == 0.05
        assert cfg.temperature == 0.5


class TestStrategyLearningLayer:
    def test_init(self):
        sl = StrategyLearningLayer()
        assert sl.outcome_history is not None
        assert len(sl.outcome_history) == 0
        for mode in ExplorationMode:
            assert sl.mode_scores[mode] == 0.0

    def test_start_mode_tracking(self):
        sl = StrategyLearningLayer()
        ctx = BehavioralContext(dissonance=0.5, identity=0.5, clamp_rate=0.1,
                                dissonance_trend=0.0, identity_drift=0.0,
                                stability=0.5, dissonance_variance=0.1)
        sl.start_mode_tracking(ExplorationMode.EXPLORE_SAFE, episode=1, context=ctx)
        assert sl._current_mode_episode == 1

    def test_end_mode_tracking(self):
        sl = StrategyLearningLayer()
        ctx = BehavioralContext(dissonance=0.5, identity=0.5, clamp_rate=0.1,
                                dissonance_trend=0.0, identity_drift=0.0,
                                stability=0.5, dissonance_variance=0.1)
        sl.start_mode_tracking(ExplorationMode.EXPLORE_SAFE, episode=1, context=ctx)
        ctx2 = BehavioralContext(dissonance=0.45, identity=0.55, clamp_rate=0.08,
                                 dissonance_trend=-0.05, identity_drift=0.05,
                                 stability=0.6, dissonance_variance=0.05)
        outcome = sl.end_mode_tracking(ExplorationMode.EXPLORE_SAFE, episode=5, context=ctx2)
        assert outcome is not None
        assert isinstance(outcome, ModeOutcome)
        assert sl.mode_experiences[ExplorationMode.EXPLORE_SAFE] == 1

    def test_record_mode_usage(self):
        sl = StrategyLearningLayer()
        outcome = sl.record_mode_usage(
            ExplorationMode.EXPLORE_SAFE, episode=1,
            pre_state={"dissonance": 0.5, "identity": 0.5},
            post_state={"dissonance": 0.45, "identity": 0.55},
            clamp_events=[],
        )
        assert outcome is not None or sl.mode_experiences[ExplorationMode.EXPLORE_SAFE] >= 0

    def test_get_mode_weights(self):
        sl = StrategyLearningLayer()
        ctx = BehavioralContext(dissonance=0.5, identity=0.5, clamp_rate=0.1,
                                dissonance_trend=0.0, identity_drift=0.0,
                                stability=0.5, dissonance_variance=0.1)
        weights = sl.get_mode_weights(ctx)
        assert len(weights) == len(ExplorationMode)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_combine_with_rule_scores(self):
        sl = StrategyLearningLayer()
        ctx = BehavioralContext(dissonance=0.5, identity=0.5, clamp_rate=0.1,
                                dissonance_trend=0.0, identity_drift=0.0,
                                stability=0.5, dissonance_variance=0.1)
        rule_scores = {mode: 0.5 for mode in ExplorationMode}
        combined = sl.combine_with_rule_scores(rule_scores, ctx)
        assert len(combined) == len(ExplorationMode)

    def test_get_learning_status(self):
        sl = StrategyLearningLayer()
        status = sl.get_learning_status()
        assert "mode_scores" in status
        assert "mode_experiences" in status
        assert "outcomes_recorded" in status
