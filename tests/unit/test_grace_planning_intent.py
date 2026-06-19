"""Tests for ravana-v2/grace/core: Governor, MicroPlanner, IntentEngine, StrategyLayer, StrategyLearningLayer."""

import pytest
import numpy as np
from ravana_grace.core.governor import Governor, GovernorConfig, CognitiveSignals, RegulationMode, RegulatedOutput
from ravana_grace.core.planning import MicroPlanner, PlanningConfig, SimulatedFuture
from ravana_grace.core.intent import IntentEngine, IntentConfig, SystemObjective, ObjectiveState, IntentAwareStrategy
from ravana_grace.core.strategy import ExplorationMode, BehavioralContext, ModeSelection, StrategyLayer
from ravana_grace.core.strategy_learning import StrategyLearningLayer


# ── Governor Tests ──

class TestGovernorConfig:
    def test_default_config(self):
        cfg = GovernorConfig()
        assert cfg.min_dissonance == 0.15
        assert cfg.max_dissonance == 0.95
        assert cfg.min_identity == 0.10

    def test_custom_config(self):
        cfg = GovernorConfig(min_dissonance=0.2, max_dissonance=0.8)
        assert cfg.min_dissonance == 0.2


class TestRegulationMode:
    def test_values(self):
        assert RegulationMode.NORMAL.value == "normal"
        assert RegulationMode.RECOVERY.value == "recovery"


class TestCognitiveSignals:
    def test_default_signals(self):
        sig = CognitiveSignals()
        assert sig.dissonance_delta == 0.0
        assert sig.source == "unknown"


class TestGovernor:
    def test_default_init(self):
        g = Governor()
        assert g.config.min_dissonance == 0.15

    def test_regulate_normal(self):
        g = Governor()
        sig = CognitiveSignals(dissonance_delta=-0.05, identity_delta=0.02)
        result = g.regulate(current_dissonance=0.5, current_identity=0.5, signals=sig, episode=1)
        assert result.mode == RegulationMode.NORMAL
        assert abs(result.dissonance_delta - (-0.05)) < 1e-6

    def test_regulate_recovery_mode_at_crisis(self):
        g = Governor()
        sig = CognitiveSignals(dissonance_delta=0.1, identity_delta=-0.05)
        result = g.regulate(current_dissonance=0.92, current_identity=0.5, signals=sig, episode=1)
        assert result.mode == RegulationMode.RECOVERY

    def test_get_status(self):
        g = Governor()
        status = g.get_status()
        assert 'mode' in status

    def test_get_health_metrics(self):
        g = Governor()
        metrics = g.get_health_metrics()
        assert 'total_regulation_events' in metrics

    def test_clamp_diagnostics(self):
        g = Governor()
        diag = g.clamp_diagnostics
        assert diag.total_upstream_suggestions == 0


# ── MicroPlanner Tests ──

class TestPlanningConfig:
    def test_default_config(self):
        cfg = PlanningConfig()
        assert cfg.horizon == 5

    def test_mode_deltas_exist(self):
        cfg = PlanningConfig()
        assert ExplorationMode.EXPLORE_AGGRESSIVE in cfg.mode_deltas


class TestMicroPlanner:
    def test_default_init(self):
        mp = MicroPlanner()
        assert mp.config.horizon == 5

    def test_simulate_forward(self):
        mp = MicroPlanner()
        ctx = BehavioralContext(dissonance=0.5, identity=0.5)
        future = mp.simulate_forward(ctx, ExplorationMode.STABILIZE, steps=3)
        assert len(future.dissonance_trajectory) == 4
        assert future.steps == 3

    def test_score_future(self):
        mp = MicroPlanner()
        ctx = BehavioralContext(dissonance=0.5, identity=0.7)
        future = SimulatedFuture(
            dissonance_trajectory=[0.5, 0.48, 0.47],
            identity_trajectory=[0.7, 0.72, 0.73],
            clamp_risk=0.0, terminal_score=0.9, steps=3
        )
        score = mp.score_future(ctx, future)
        assert score > 0.5

    def test_plan_and_select(self):
        mp = MicroPlanner()
        ctx = BehavioralContext(dissonance=0.5, identity=0.5)
        best_mode, predictions, scores = mp.plan_and_select(ctx)
        assert best_mode in list(ExplorationMode)
        assert len(predictions) == len(ExplorationMode)


# ── IntentEngine Tests ──

class TestIntentEngine:
    def test_default_init(self):
        ie = IntentEngine()
        assert ie.episode_count == 0
        assert len(ie.objectives) == 4

    def test_update_state(self):
        ie = IntentEngine()
        ie.update_state({"dissonance": 0.5}, [])
        assert ie.current_context == {"dissonance": 0.5}

    def test_evaluate_outcomes(self):
        ie = IntentEngine()
        ie.evaluate_outcomes(1, {"dissonance": 0.5, "identity": 0.5},
                            {"dissonance": 0.3, "identity": 0.7},
                            ExplorationMode.STABILIZE, 0)
        assert ie.episode_count == 1

    def test_compute_mode_bias(self):
        ie = IntentEngine()
        bias = ie.compute_mode_bias(ExplorationMode.STABILIZE)
        assert -1.0 <= bias <= 1.0

    def test_get_current_intent(self):
        ie = IntentEngine()
        intent = ie.get_current_intent()
        assert 'dominant_objective' in intent
        assert 'objective_weights' in intent


# ── StrategyLayer Tests ──

class TestStrategyLayer:
    def test_default_init(self):
        sl = StrategyLayer()
        assert sl.mode_history == []
        assert sl.current_mode == ExplorationMode.EXPLORE_SAFE

    def test_select_mode(self):
        sl = StrategyLayer()
        ctx = BehavioralContext(dissonance=0.5, identity=0.5)
        selection = sl.select_mode(ctx)
        assert isinstance(selection, ModeSelection)
        assert selection.mode in list(ExplorationMode)

    def test_apply_policy_bias(self):
        sl = StrategyLayer()
        dd, di, info = sl.apply_policy_bias((0.1, 0.05), ExplorationMode.STABILIZE)
        assert isinstance(dd, float)
        assert isinstance(di, float)
        assert 'mode' in info


# ── StrategyLearningLayer Tests ──

class TestStrategyLearningLayer:
    def test_default_init(self):
        sll = StrategyLearningLayer()
        assert len(sll.mode_experiences) == len(ExplorationMode)

    def test_record_mode_usage(self):
        sll = StrategyLearningLayer()
        outcome = sll.record_mode_usage(
            ExplorationMode.STABILIZE, 1,
            {"d": 0.5}, {"d": 0.3}, []
        )
        assert ExplorationMode.STABILIZE in sll.mode_experiences

    def test_get_learning_status(self):
        sll = StrategyLearningLayer()
        status = sll.get_learning_status()
        assert 'outcomes_recorded' in status
        assert 'mode_scores' in status


# ── IntentAwareStrategy Basic Smoke Test ──

class TestIntentAwareStrategy:
    def test_can_be_created(self):
        sl = StrategyLayer()
        sll = StrategyLearningLayer()
        ie = IntentEngine()
        ias = IntentAwareStrategy(sl, sll, ie)
        assert ias.intent is ie
        assert ias.strategy is sl
        assert ias.learning is sll
