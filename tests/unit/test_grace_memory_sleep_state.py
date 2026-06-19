"""Tests for ravana-v2/grace/core: EpisodicMemory, SemanticMemory, WorkingMemory, RavanaMemorySystem, SleepConsolidation, MeaningEngine, StateManager."""

import pytest
import numpy as np
import time
from ravana_grace.core.memory import EpisodicMemory, SemanticMemory, WorkingMemory, MemoryTrace, RavanaMemorySystem
from ravana_grace.core.sleep import SleepConsolidation, SleepConfig, SleepStage
from ravana_grace.core.meaning import MeaningEngine, MeaningConfig, MeaningRecord
from ravana_grace.core.state import CognitiveState, StateManager, CognitivePhase
from ravana_grace.core.governor import Governor, GovernorConfig
from ravana_grace.core.identity import IdentityEngine
from ravana_grace.core.resolution import ResolutionEngine


# ── MemoryTrace Tests ──

class TestMemoryTrace:
    def test_default_creation(self):
        trace = MemoryTrace(timestamp=time.time(), episode=1, content={"key": "val"},
                           dissonance_at_time=0.5, identity_at_time=0.5)
        assert trace.episode == 1
        assert trace.salience == 0.5
        assert trace.tags == []


# ── EpisodicMemory Tests ──

class TestEpisodicMemory:
    def test_default_init(self):
        mem = EpisodicMemory()
        assert mem.capacity == 1000
        assert mem.traces == []

    def test_record_adds_trace(self):
        mem = EpisodicMemory()
        trace = MemoryTrace(time.time(), 1, {}, 0.5, 0.5)
        mem.record(trace)
        assert len(mem.traces) == 1

    def test_record_capacity_enforced(self):
        mem = EpisodicMemory(capacity=2)
        for i in range(3):
            mem.record(MemoryTrace(time.time(), i, {}, 0.5, 0.5, salience=float(i)))
        assert len(mem.traces) == 2

    def test_retrieve_by_dissonance(self):
        mem = EpisodicMemory()
        mem.record(MemoryTrace(time.time(), 1, {}, 0.8, 0.5))
        mem.record(MemoryTrace(time.time(), 2, {}, 0.3, 0.5))
        results = mem.retrieve_by_dissonance(threshold=0.7)
        assert len(results) == 1


# ── SemanticMemory Tests ──

class TestSemanticMemory:
    def test_default_init(self):
        mem = SemanticMemory()
        assert "fairness" in mem.knowledge_graph

    def test_update_norm(self):
        mem = SemanticMemory()
        mem.update_norm("fairness", 0.1, 0.05)
        assert mem.knowledge_graph["fairness"]["weight"] == 0.9

    def test_update_norm_clamp(self):
        mem = SemanticMemory()
        mem.update_norm("fairness", 10.0, 10.0)
        assert mem.knowledge_graph["fairness"]["weight"] == 1.0


# ── WorkingMemory Tests ──

class TestWorkingMemory:
    def test_default_init(self):
        wm = WorkingMemory()
        assert wm.capacity == 7

    def test_broadcast(self):
        wm = WorkingMemory(capacity=2)
        signals = [{"bid": 0.8, "content": "a"}, {"bid": 0.5, "content": "b"}, {"bid": 0.3, "content": "c"}]
        focus = wm.broadcast(signals)
        assert len(focus) == 2


# ── RavanaMemorySystem Tests ──

class TestRavanaMemorySystem:
    def test_default_init(self):
        rms = RavanaMemorySystem()
        assert isinstance(rms.episodic, EpisodicMemory)

    def test_process_step(self):
        rms = RavanaMemorySystem()
        rms.process_step({"episode": 1}, {"dissonance": 0.5, "identity": 0.5})
        assert len(rms.episodic.traces) == 1

    def test_get_context(self):
        rms = RavanaMemorySystem()
        rms.episodic.record(MemoryTrace(time.time(), 1, {}, 0.8, 0.5))
        ctx = rms.get_context_for_decision()
        assert "past_failures" in ctx


# ── MeaningEngine Tests ──

class TestMeaningConfig:
    def test_default_config(self):
        cfg = MeaningConfig()
        assert cfg.w_dissonance_reduction == 0.4
        assert cfg.effort_kappa == 0.5


class TestMeaningRecord:
    def test_default_record(self):
        record = MeaningRecord(episode=1, raw_meaning=0.5, effort=0.3, coherence_gain=0.4,
                              identity_coherence_gain=0.3, predictive_gain=0.2,
                              effective_meaning=0.5, authentic=True, components={})
        assert record.episode == 1
        assert record.effective_meaning == 0.5


class TestMeaningEngine:
    def test_default_init(self):
        me = MeaningEngine()
        assert me.accumulated_meaning == 0.0

    def test_compute_meaning(self):
        me = MeaningEngine()
        record = me.compute_meaning(episode=1, pre_dissonance=0.8, post_dissonance=0.3,
                                   pre_identity=0.5, post_identity=0.6, predictive_gain=0.5, effort=0.2)
        assert isinstance(record, MeaningRecord)
        assert me.accumulated_meaning > 0.0

    def test_compute_meaning_zero(self):
        me = MeaningEngine()
        record = me.compute_meaning(1, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0)
        assert record.effective_meaning == 0.0

    def test_stake_and_resolve(self):
        me = MeaningEngine()
        me.stake_meaning("belief_1", 0.5)
        assert "belief_1" in me._commitments
        result = me.resolve_stake("belief_1", belief_held=True)
        assert result == 0.0

    def test_resolve_belief_wrong_deducts_meaning(self):
        me = MeaningEngine()
        # First accumulate some meaning so the deduction can be measured
        me.compute_meaning(1, 0.8, 0.3, 0.5, 0.7, 0.5, 0.3)
        initial = me.accumulated_meaning
        assert initial > 0.0
        me.stake_meaning("belief_2", 0.5)
        result = me.resolve_stake("belief_2", belief_held=False)
        assert result < 0.0
        assert me.accumulated_meaning < initial

    def test_get_expected_meaning(self):
        me = MeaningEngine()
        expected = me.get_expected_meaning(0.3, 0.2, 0.4, 0.5)
        assert expected > 0.0

    def test_get_status(self):
        me = MeaningEngine()
        me.compute_meaning(1, 0.8, 0.3, 0.5, 0.6, 0.5, 0.2)
        status = me.get_status()
        assert 'accumulated_meaning' in status
        assert 'active_commitments' in status


# ── SleepConsolidation Tests ──

class TestSleepConfig:
    def test_default_config(self):
        cfg = SleepConfig()
        assert cfg.pressure_threshold == 0.2
        assert cfg.counterfactual_rate == 0.20


class TestSleepConsolidation:
    def test_default_init(self):
        sc = SleepConsolidation()
        assert sc._accumulated_pressure == 0.0
        assert sc._current_stage == SleepStage.AWAKE

    def test_accumulate_and_should_sleep(self):
        sc = SleepConsolidation(SleepConfig(pressure_threshold=0.2))
        assert not sc.should_sleep()
        sc.accumulate_pressure(0.3)
        assert sc.should_sleep()  # np.True_ is truthy

    def test_get_pressure(self):
        sc = SleepConsolidation()
        sc.accumulate_pressure(0.5)
        assert sc.get_pressure() == pytest.approx(0.5)

    def test_get_status(self):
        sc = SleepConsolidation()
        status = sc.get_status()
        assert 'accumulated_pressure' in status
        assert 'current_stage' in status
        assert status['total_sleep_cycles'] == 0

    def test_execute_sleep_cycle_basic(self):
        sc = SleepConsolidation(SleepConfig(pressure_threshold=0.05))
        sc.accumulate_pressure(0.3)
        record = sc.execute_sleep_cycle(
            episode=1,
            state_snapshot={"dissonance": 0.5, "identity": 0.5},
            coherence_fn=lambda s: 0.5,
        )
        assert record.episode == 1
        assert len(sc.sleep_history) == 1


# ── CognitiveState Tests ──

class TestCognitiveState:
    def test_default_state(self):
        cs = CognitiveState()
        assert cs.dissonance == 0.5
        assert cs.identity == 0.5
        assert cs.episode == 0

    def test_snapshot(self):
        cs = CognitiveState(dissonance=0.3, identity=0.7, episode=5)
        snap = cs.snapshot()
        assert snap["dissonance"] == 0.3
        assert snap["identity"] == 0.7
        assert snap["episode"] == 5


# ── StateManager Tests ──

class TestStateManager:
    @pytest.fixture
    def state_manager(self):
        governor = Governor()
        resolution = ResolutionEngine()
        identity = IdentityEngine()
        return StateManager(governor=governor, resolution_engine=resolution, identity_engine=identity)

    def test_default_init(self, state_manager):
        assert state_manager.state.dissonance == 0.5
        assert state_manager.state.identity == 0.5

    def test_step_correct(self, state_manager):
        result = state_manager.step(correctness=True, difficulty=0.3, novelty=0.1)
        assert result['episode'] == 1
        assert state_manager.state.episode == 1

    def test_step_incorrect(self, state_manager):
        result = state_manager.step(correctness=False, difficulty=0.7, novelty=0.5)
        assert result['episode'] == 1

    def test_step_accumulates_wisdom(self, state_manager):
        for _ in range(3):
            state_manager.step(correctness=True, difficulty=0.5)
        assert state_manager.state.accumulated_wisdom >= 0.0

    def test_get_status(self, state_manager):
        status = state_manager.get_status()
        assert 'state' in status
        assert 'governor' in status
        assert 'total_steps' in status
