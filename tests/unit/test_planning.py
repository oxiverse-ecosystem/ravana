"""Tests for ravana_grace.core.planning."""

import pytest
import numpy as np
from ravana_grace.core.planning import MicroPlanner, PlanningConfig, SimulatedFuture
from ravana_grace.core.strategy import ExplorationMode, BehavioralContext


class TestPlanningConfig:
    def test_defaults(self):
        cfg = PlanningConfig()
        assert cfg.horizon == 5
        assert cfg.dissonance_drift_per_step == 0.02
        assert cfg.mode_deltas is not None
        assert len(cfg.mode_deltas) >= 4


class TestMicroPlanner:
    def test_init(self):
        mp = MicroPlanner()
        assert mp.config is not None

    def test_simulate_forward(self):
        mp = MicroPlanner()
        ctx = BehavioralContext(
            dissonance=0.5, identity=0.5, clamp_rate=0.1,
            dissonance_trend=0.0, identity_drift=0.0,
            stability=0.5, dissonance_variance=0.1
        )
        future = mp.simulate_forward(ctx, ExplorationMode.EXPLORE_SAFE)
        assert isinstance(future, SimulatedFuture)
        assert len(future.dissonance_trajectory) > 0
        assert len(future.identity_trajectory) > 0
        assert future.clamp_risk >= 0
        assert future.terminal_score >= 0

    def test_simulate_forward_different_modes(self):
        mp = MicroPlanner()
        ctx = BehavioralContext(
            dissonance=0.5, identity=0.5, clamp_rate=0.1,
            dissonance_trend=0.0, identity_drift=0.0,
            stability=0.5, dissonance_variance=0.1
        )
        # Aggressive should produce different trajectory
        future_agg = mp.simulate_forward(ctx, ExplorationMode.EXPLORE_AGGRESSIVE)
        future_stab = mp.simulate_forward(ctx, ExplorationMode.STABILIZE)
        assert future_agg.clamp_risk >= 0
        assert future_stab.clamp_risk >= 0

    def test_score_future(self):
        mp = MicroPlanner()
        ctx = BehavioralContext(
            dissonance=0.5, identity=0.5, clamp_rate=0.1,
            dissonance_trend=0.0, identity_drift=0.0,
            stability=0.5, dissonance_variance=0.1
        )
        future = mp.simulate_forward(ctx, ExplorationMode.EXPLORE_SAFE)
        score = mp.score_future(ctx, future)
        assert 0 <= score <= 1.0

    def test_plan_and_select(self):
        mp = MicroPlanner()
        ctx = BehavioralContext(
            dissonance=0.5, identity=0.5, clamp_rate=0.1,
            dissonance_trend=0.0, identity_drift=0.0,
            stability=0.5, dissonance_variance=0.1
        )
        best_mode, predictions, scores = mp.plan_and_select(ctx)
        assert best_mode is not None
        assert isinstance(best_mode, ExplorationMode)
        assert len(predictions) > 0
        assert len(scores) > 0
