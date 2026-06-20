"""Tests for ravana_grace.core.predictive_world."""

import pytest
import numpy as np
from ravana_grace.core.predictive_world import (
    LearnedWorldModel, WorldModelConfig, PredictedState,
    AnomalyEvent, FalseWorldTester,
)


class TestWorldModelConfig:
    def test_defaults(self):
        cfg = WorldModelConfig()
        assert cfg.input_dim == 6
        assert cfg.hidden_dim == 12
        assert cfg.learning_rate == 0.01
        assert cfg.surprise_threshold_start == 0.1


class TestLearnedWorldModel:
    def test_init(self):
        lwm = LearnedWorldModel()
        assert lwm.W1.shape == (6, 12)
        assert lwm.W2.shape == (12, 3)
        assert lwm.anomaly_count == 0

    def test_predict(self):
        lwm = LearnedWorldModel()
        state = {"dissonance": 0.5, "identity": 0.5, "clamp_rate": 0.1,
                 "dissonance_trend": 0.0, "stability": 0.5}
        pred = lwm.predict(state, mode=0)
        assert isinstance(pred, PredictedState)
        assert 0.15 <= pred.dissonance_pred <= 0.95
        assert 0.10 <= pred.identity_pred <= 0.95
        assert 0 <= pred.clamp_rate_pred <= 1.0
        assert pred.uncertainty >= 0

    def test_observe(self):
        lwm = LearnedWorldModel()
        pre_state = {"dissonance": 0.5, "identity": 0.5, "clamp_rate": 0.1,
                     "dissonance_trend": 0.0, "stability": 0.5}
        post_state = {"dissonance": 0.55, "identity": 0.52, "clamp_rate": 0.08}
        event = lwm.observe(
            episode=1, pre_state=pre_state, mode=0,
            post_state=post_state, actual_boundary=0.95
        )
        assert lwm.baseline_surprise > 0

    def test_get_world_model_status(self):
        lwm = LearnedWorldModel()
        status = lwm.get_world_model_status()
        assert "belief" in status
        assert "surprise_threshold" in status
        assert "prediction_uncertainty" in status
        assert status["confirmed_anomalies"] == 0


class TestFalseWorldTester:
    def test_init(self):
        lwm = LearnedWorldModel()
        fwt = FalseWorldTester(lwm)
        assert fwt.false_patterns_injected == 0
        assert fwt.false_patterns_resisted == 0

    def test_get_resistance_score_initial(self):
        lwm = LearnedWorldModel()
        fwt = FalseWorldTester(lwm)
        assert fwt.get_resistance_score() == 1.0

    def test_inject_false_boundary(self):
        lwm = LearnedWorldModel()
        fwt = FalseWorldTester(lwm)
        result = fwt.inject_false_boundary_shift(episode=1, fake_boundary=0.5)
        assert fwt.false_patterns_injected == 1
        assert isinstance(result, bool)
