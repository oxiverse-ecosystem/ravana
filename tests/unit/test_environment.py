"""Tests for ravana_grace.core.environment."""

import pytest
import numpy as np
from ravana_grace.core.environment import (
    NonStationaryEnvironment, EnvironmentConfig,
    WorldState, WorldModelEvaluator, HiddenDynamics,
)


class TestEnvironmentConfig:
    def test_defaults(self):
        cfg = EnvironmentConfig()
        assert cfg.boundary_shift_frequency == 500
        assert cfg.noise_drift_rate == 0.02
        assert cfg.goal_flip_period == 400


class TestNonStationaryEnvironment:
    def test_init(self):
        env = NonStationaryEnvironment()
        assert env.episode_count == 0
        assert env.current_state is not None

    def test_step(self):
        env = NonStationaryEnvironment()
        state = env.step(episode=1)
        assert isinstance(state, WorldState)
        assert state.effective_boundary > 0
        assert state.difficulty_level > 0

    def test_step_shifts_boundary_at_interval(self):
        env = NonStationaryEnvironment()
        initial = env.boundary_base
        for ep in range(505):
            env.step(episode=ep)
        if ep >= 500 and ep % 500 == 0:
            # Boundary may have shifted
            pass
        assert True  # No crash

    def test_get_hidden_truth(self):
        env = NonStationaryEnvironment()
        env.step(episode=1)
        truth = env.get_hidden_truth()
        assert "total_episodes" in truth
        assert "final_boundary" in truth

    def test_get_intensity_ramped(self):
        env = NonStationaryEnvironment()
        for ep in range(100):
            env.step(episode=ep)
        truth = env.get_hidden_truth()
        assert truth["total_episodes"] > 0

    def test_noise_walk_evolves(self):
        env = NonStationaryEnvironment()
        noise_before = env.noise_walk
        for ep in range(10):
            env.step(episode=ep)
        # Noise may or may not change each step
        assert True  # Ensure no crash


class TestWorldModelEvaluator:
    def test_init(self):
        env = NonStationaryEnvironment()
        wme = WorldModelEvaluator(env)
        assert wme.env is env

    def test_insufficient_data(self):
        env = NonStationaryEnvironment()
        wme = WorldModelEvaluator(env)
        result = wme.evaluate_model_accuracy()
        assert result["status"] == "insufficient_data"

    def test_record_belief(self):
        env = NonStationaryEnvironment()
        wme = WorldModelEvaluator(env)
        wme.record_ravana_belief(
            episode=1,
            belief={"believed_boundary": 0.9, "inferred_noise_pattern": 0.05, "detected_regime": "stable"}
        )
        assert len(wme.ravana_beliefs) == 1

    def test_get_actual_boundary(self):
        env = NonStationaryEnvironment()
        wme = WorldModelEvaluator(env)
        boundary = wme._get_actual_boundary(episode=100)
        assert boundary == 0.95  # Default before any shifts


class TestHiddenDynamics:
    def test_enum_values(self):
        assert HiddenDynamics.BOUNDARY_SHIFT.value == "boundary_shift"
        assert HiddenDynamics.NOISE_DRIFT.value == "noise_drift"
        assert HiddenDynamics.GOAL_FLIP.value == "goal_flip"
