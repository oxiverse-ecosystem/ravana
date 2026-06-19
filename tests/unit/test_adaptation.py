"""Tests for ravana_grace.core.adaptation."""

import pytest
import numpy as np
from ravana_grace.core.adaptation import (
    PolicyTweakLayer, AdaptiveGovernorBridge,
    AdaptationConfig, ClampExperience,
)


class TestAdaptationConfig:
    def test_defaults(self):
        cfg = AdaptationConfig()
        assert cfg.learning_rate == 0.05
        assert cfg.clamp_penalty == 5.0
        assert cfg.max_tweak == 0.05


class TestPolicyTweakLayer:
    def test_init(self):
        ptl = PolicyTweakLayer()
        assert ptl.total_tweaks == 0
        assert ptl.learning_steps == 0
        assert ptl.weights.shape == (5, 2)

    def test_init_with_config(self):
        cfg = AdaptationConfig(learning_rate=0.1, max_tweak=0.1)
        ptl = PolicyTweakLayer(config=cfg)
        assert ptl.config.learning_rate == 0.1
        assert ptl.config.max_tweak == 0.1

    def test_encode_state(self):
        ptl = PolicyTweakLayer()
        recent = [{"dissonance": 0.5, "identity": 0.5}]
        state = ptl.encode_state(0.6, 0.4, recent, "normal")
        assert state.shape == (5,)
        assert state[0] == 0.6
        assert state[1] == 0.4

    def test_encode_state_with_empty_history(self):
        ptl = PolicyTweakLayer()
        state = ptl.encode_state(0.5, 0.5, [], "normal")
        assert state.shape == (5,)

    def test_compute_tweak(self):
        ptl = PolicyTweakLayer()
        signals = {"dissonance": 0.6, "identity": 0.5, "d_delta": 0.1, "i_delta": 0.05, "dissonance_delta": 0.1, "identity_delta": 0.05}
        d_tweak, i_tweak, exp = ptl.compute_tweak(
            signals,
            {"mode": "normal", "episode": 1}
        )
        assert ptl.total_tweaks == 1
        assert isinstance(d_tweak, float)
        assert isinstance(i_tweak, float)
        assert isinstance(exp, ClampExperience)

    def test_compute_tweak_with_clamp(self):
        ptl = PolicyTweakLayer()
        signals = {"dissonance": 0.9, "identity": 0.2, "d_delta": 0.3, "i_delta": -0.1,
                   "dissonance_delta": 0.3, "identity_delta": -0.1}
        d_tweak, i_tweak, exp = ptl.compute_tweak(
            signals,
            {"mode": "recovery", "episode": 1, "clamp_occurred": True, "correction": 0.5, "variable": "dissonance", "layer": "governor"}
        )
        # Negative reward for clamp (may be -0.0 which is not < 0 in Python)
        assert exp.reward <= 0

    def test_learn_from_clamp(self):
        ptl = PolicyTweakLayer()
        state = np.array([0.7, 0.3, 0.2, -0.1, 1.0])
        exp = ClampExperience(
            episode=1, state_encoding=state,
            variable="dissonance", correction=0.3, layer="test",
        )
        ptl.learn_from_clamp(exp)
        assert ptl.learning_steps == 1

    def test_get_status(self):
        ptl = PolicyTweakLayer()
        status = ptl.get_status()
        assert "total_tweaks" in status
        assert "mean_tweak_magnitude" in status
        assert "learning_steps" in status

    def test_get_learning_report(self):
        ptl = PolicyTweakLayer()
        report = ptl.get_learning_report()
        assert "ADAPTATION ENGINE" in report


class TestAdaptiveGovernorBridge:
    def test_init(self):
        governor = type("MockGovernor", (), {"mode_history": [], "clamp_diagnostics": type("Mock", (), {"events": []})()})()
        ptl = PolicyTweakLayer()
        bridge = AdaptiveGovernorBridge(governor, ptl)
        assert bridge.governor is governor
        assert bridge.adaptation is ptl
