"""Tests for ravana_grace.core.reality_friction."""

import pytest
import numpy as np
from ravana_grace.core.reality_friction import (
    RealityFrictionLayer, RealityFrictionConfig, FrictionType,
    NoiseConfig, DelayConfig, PartialObsConfig,
    NonStationaryConfig, ResourceConfig, HiddenVariableModel,
    ObservedState, DelayedFeedback, FrictionMetrics, RealityFriction,
)


class TestRealityFrictionConfig:
    def test_defaults(self):
        cfg = RealityFrictionConfig()
        assert cfg.intensity == 0.5
        assert cfg.ramp_up is True
        assert cfg.noise.base_sigma == 0.1
        assert cfg.delay.min_delay == 5
        assert cfg.partial.observable_fraction == 0.8


class TestRealityFrictionLayer:
    def test_init(self):
        rfl = RealityFrictionLayer()
        assert rfl.episode == 0
        assert rfl.hidden_vars is not None
        assert rfl.timeout_count == 0

    def test_get_intensity_initial(self):
        rfl = RealityFrictionLayer()
        assert rfl.get_intensity() == 0.0  # episode=0

    def test_get_intensity_ramped(self):
        rfl = RealityFrictionLayer(config=RealityFrictionConfig(
            ramp_episodes=100, intensity=0.5
        ))
        rfl.episode = 50
        assert rfl.get_intensity() == 0.25  # 50% of ramp

    def test_observe(self):
        rfl = RealityFrictionLayer()
        obs = rfl.observe({"dissonance": 0.5, "identity": 0.5})
        assert isinstance(obs, ObservedState)
        assert obs.episode == 0
        assert obs.noise_level >= 0
        assert obs.confidence > 0

    def test_request_feedback(self):
        rfl = RealityFrictionLayer()
        fb = rfl.request_feedback(
            {"boundary": 0.75}, trigger_episode=0
        )
        # May be None if feedback lost
        if fb is not None:
            assert isinstance(fb, DelayedFeedback)
            assert fb.episode_triggered == 0

    def test_deliver_pending_feedback(self):
        rfl = RealityFrictionLayer()
        fb1 = rfl.request_feedback({"boundary": 0.75}, trigger_episode=0)
        rfl.episode = 1000
        delivered = rfl.deliver_pending_feedback()
        # Some may be delivered
        assert isinstance(delivered, list)

    def test_step(self):
        rfl = RealityFrictionLayer()
        result = rfl.step(
            ravana_belief=0.75, ravana_confidence=0.5,
            true_state={"dissonance": 0.5, "identity": 0.5}
        )
        assert "observation" in result
        assert "metrics" in result
        assert "hidden_effects" in result
        assert rfl.episode == 1

    def test_step_metrics(self):
        rfl = RealityFrictionLayer()
        result = rfl.step(
            ravana_belief=0.75, ravana_confidence=0.9,
            true_state={"dissonance": 0.5, "identity": 0.5}
        )
        metrics = result["metrics"]
        assert isinstance(metrics, FrictionMetrics)
        assert metrics.episode == 1
        assert metrics.belief_drift >= 0

    def test_get_friction_summary_initial(self):
        rfl = RealityFrictionLayer()
        summary = rfl.get_friction_summary()
        assert summary["episodes"] == 0

    def test_get_friction_summary_after_steps(self):
        rfl = RealityFrictionLayer()
        for i in range(10):
            rfl.step(
                ravana_belief=0.75, ravana_confidence=0.5,
                true_state={"dissonance": 0.5 + i*0.01, "identity": 0.5}
            )
        summary = rfl.get_friction_summary()
        assert summary["episodes"] > 0


class TestHiddenVariableModel:
    def test_init(self):
        hvm = HiddenVariableModel()
        assert "ambient_stress" in hvm.hidden_state
        assert "system_drift" in hvm.hidden_state

    def test_evolve(self):
        hvm = HiddenVariableModel()
        hvm.evolve(50, NonStationaryConfig())
        assert abs(hvm.hidden_state["system_drift"]) < 0.5

    def test_compute_effects(self):
        hvm = HiddenVariableModel()
        effects = hvm.compute_effects()
        assert len(effects) == 4

    def test_get_observable_hint(self):
        hvm = HiddenVariableModel()
        hint = hvm.get_observable_hint()
        # 5% chance, may be None
        assert hint is None or hint.startswith("hint:")


class TestFrictionType:
    def test_values(self):
        # FrictionType uses auto(), so values are ints starting at 1
        assert FrictionType.NOISE.value == 1
        assert FrictionType.DELAY.value == 2
        assert FrictionType.PARTIAL.value == 3
        assert FrictionType.NON_STATIONARY.value == 4


class TestRealityFrictionAlias:
    def test_alias(self):
        assert RealityFriction is RealityFrictionLayer
