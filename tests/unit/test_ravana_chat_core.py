"""Tests for ravana_chat_src core modules: emotion, identity, meaning, dual_process, gw, meta_cognition, sleep, belief_store."""

import sys, os
_rcs = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ravana_chat_src", "src")
if _rcs not in sys.path:
    sys.path.insert(0, _rcs)

import pytest
import numpy as np
from ravana_chat.core.emotion import VADEmotionEngine, VADConfig, VADState
from ravana_chat.core.identity import IdentityEngine, IdentityConfig, IdentityState
from ravana_chat.core.meaning import MeaningEngine, MeaningConfig, MeaningState
from ravana_chat.core.dual_process import DualProcessController, DualProcessConfig, Route, RouteDecision
from ravana_chat.core.global_workspace import GlobalWorkspace, GWConfig, WorkspaceItem
from ravana_chat.core.meta_cognition import MetaCognition, MetaCognitiveConfig, EpistemicMode
from ravana_chat.core.sleep import SleepConsolidation, SleepConfig
from ravana_chat.core.belief_store import BeliefStore, BeliefConfig, UserBeliefProfile


# ── VADEmotionEngine Tests ──

class TestVADConfig:
    def test_default_config(self):
        cfg = VADConfig()
        assert cfg.eta_valence == 0.3
        assert cfg.baseline_arousal == 0.3


class TestVADState:
    def test_to_dict(self):
        s = VADState(valence=0.5, arousal=0.7, dominance=0.3)
        d = s.to_dict()
        assert d['valence'] == 0.5
        assert d['arousal'] == 0.7
        assert d['dominance'] == 0.3


class TestVADEmotionEngine:
    def test_default_init(self):
        e = VADEmotionEngine()
        assert e.state.valence == 0.0
        assert e.state.arousal == 0.3
        assert e.state.dominance == 0.5

    def test_update_moves_state(self):
        e = VADEmotionEngine()
        e.update(stimulus_valence=1.0, stimulus_arousal=0.8)
        assert e.state.valence > 0.0
        assert e.state.arousal > 0.3

    def test_decay_over_time(self):
        e = VADEmotionEngine()
        e.update(stimulus_arousal=1.0)
        initial = e.state.arousal
        for _ in range(100):
            e.update()
        assert e.state.arousal < initial

    def test_uncertainty_boosts_arousal(self):
        e = VADEmotionEngine()
        e.update(uncertainty=1.0)
        assert e.state.arousal > 0.3

    def test_get_emotional_label(self):
        e = VADEmotionEngine()
        label = e.get_emotional_label()
        assert isinstance(label, str) and len(label) > 0

    def test_high_valence_high_arousal(self):
        e = VADEmotionEngine()
        e.update(stimulus_valence=0.8, stimulus_arousal=0.9)
        label = e.get_emotional_label()
        assert isinstance(label, str) and len(label) > 0


# ── IdentityEngine Tests ──

class TestIdentityConfig:
    def test_default_config(self):
        cfg = IdentityConfig()
        assert cfg.initial_strength == 0.25


class TestIdentityState:
    def test_to_dict(self):
        s = IdentityState(strength=0.5, momentum=0.1)
        d = s.to_dict()
        assert d['strength'] == 0.5
        assert d['momentum'] == 0.1


class TestIdentityEngine:
    def test_default_init(self):
        ie = IdentityEngine()
        assert ie.state.strength == 0.25

    def test_compute_update_success_boosts(self):
        ie = IdentityEngine()
        delta = ie.compute_update(
            resolution_delta=0.5, resolution_success=True,
            regulated_identity_delta=0.1, current_dissonance=0.3,
            resolution_streak=3, correctness=True)
        assert delta > 0.0

    def test_compute_update_failure_penalizes(self):
        ie = IdentityEngine()
        delta = ie.compute_update(
            resolution_delta=0.1, resolution_success=False,
            regulated_identity_delta=-0.1, current_dissonance=0.8,
            resolution_streak=0, correctness=False)
        assert delta < 0.0

    def test_apply_update(self):
        ie = IdentityEngine()
        ie.apply_update(0.1)
        assert ie.state.strength > 0.25

    def test_get_trend(self):
        ie = IdentityEngine()
        ie.apply_update(0.1)
        ie.apply_update(0.1)
        ie.apply_update(0.1)
        trend = ie.get_trend()
        assert trend > 0.0


# ── MeaningEngine Tests ──

class TestMeaningEngine:
    def test_default_init(self):
        me = MeaningEngine()
        assert me.state.accumulated_meaning == 0.0

    def test_compute_meaning_positive(self):
        me = MeaningEngine()
        me.compute_meaning(episode=1, pre_dissonance=0.8, post_dissonance=0.3,
                          pre_identity=0.5, post_identity=0.7, predictive_gain=0.5, effort=0.2)
        assert me.accumulated_meaning > 0.0

    def test_compute_meaning_zero(self):
        me = MeaningEngine()
        me.compute_meaning(1, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5)
        assert me.accumulated_meaning == pytest.approx(0.0, abs=1e-6)


# ── DualProcessController Tests ──

class TestDualProcessController:
    def test_default_init(self):
        dp = DualProcessController()
        assert dp.config.system2_confidence_threshold == 0.25

    def test_low_confidence_triggers_system2(self):
        dp = DualProcessController()
        decision = dp.decide_route(confidence=0.1, novelty=0.1)
        assert decision.route == Route.SYSTEM2_SLOW
        assert decision.reason == "low_confidence"

    def test_high_novelty_triggers_system2(self):
        dp = DualProcessController()
        decision = dp.decide_route(confidence=0.8, novelty=0.8)
        assert decision.route == Route.SYSTEM2_SLOW
        assert decision.reason == "high_novelty"

    def test_high_stakes_triggers_system2(self):
        dp = DualProcessController()
        decision = dp.decide_route(confidence=0.8, novelty=0.1, stakes=0.8)
        assert decision.route == Route.SYSTEM2_SLOW
        assert decision.reason == "high_stakes"

    def test_default_system1(self):
        dp = DualProcessController()
        decision = dp.decide_route(confidence=0.8, novelty=0.1, stakes=0.1)
        assert decision.route == Route.SYSTEM1_FAST

    def test_max_consecutive_system2_forces_system1(self):
        dp = DualProcessController(config=DualProcessConfig(max_consecutive_system2=3))
        for _ in range(3):
            dp.decide_route(confidence=0.1, novelty=0.1)
        decision = dp.decide_route(confidence=0.1, novelty=0.1)
        assert decision.route == Route.SYSTEM1_FAST

    def test_reset(self):
        dp = DualProcessController()
        dp.decide_route(confidence=0.1, novelty=0.1)
        assert dp.consecutive_system2 > 0
        dp.reset()
        assert dp.consecutive_system2 == 0


# ── GlobalWorkspace Tests ──

class TestGlobalWorkspace:
    def test_default_init(self):
        gw = GlobalWorkspace()
        assert gw.config.capacity == 7

    def test_submit_and_compete(self):
        gw = GlobalWorkspace(config=GWConfig(capacity=3, broadcast_threshold=0.3))
        gw.submit_bid("emotion", {"v": 0.5}, urgency=0.8, valence=0.5, episode=1)
        gw.submit_bid("meaning", {"m": 0.3}, urgency=0.2, valence=0.1, episode=1)
        winners = gw.compete()
        assert len(winners) == 1
        assert winners[0].source == "emotion"

    def test_no_winners_below_threshold(self):
        gw = GlobalWorkspace(config=GWConfig(broadcast_threshold=0.5))
        gw.submit_bid("test", {}, urgency=0.1, valence=0.0, episode=1)
        winners = gw.compete()
        assert len(winners) == 0

    def test_get_recent_broadcasts(self):
        gw = GlobalWorkspace()
        assert gw.get_recent_broadcasts() == []


# ── MetaCognition Tests ──

class TestMetaCognition:
    def test_default_init(self):
        mc = MetaCognition()
        assert mc.current_mode == EpistemicMode.EXPLORATORY

    def test_detect_confirmation_bias(self):
        mc = MetaCognition()
        for _ in range(5):
            mc.record_probe(subject="trust", prediction=0.8, actual=0.8, outcome="confirm")
        biases = mc.detect_reasoning_bias(turn=1)['biases']
        assert 'confirmation_bias' in biases

    def test_record_calibration(self):
        mc = MetaCognition()
        mc.record_calibration(0.9, True)
        assert mc.get_calibration_error() == pytest.approx(0.1)
        mc.record_calibration(0.9, False)
        assert mc.get_calibration_error() == pytest.approx(0.5)

    def test_recommend_epistemic_mode(self):
        mc = MetaCognition()
        for _ in range(5):
            mc.record_probe("x", 0.5, 0.5, "confirm")
        mode = mc.recommend_epistemic_mode(1)
        assert mode in list(EpistemicMode)

    def test_record_probe(self):
        mc = MetaCognition()
        mc.record_probe("trust", 0.9, 0.85, "confirm")
        assert len(mc.probe_history) == 1


# ── SleepConsolidation Tests (ravana_chat version) ──

class TestSleepConsolidation:
    def test_default_init(self):
        sc = SleepConsolidation()
        assert sc.metrics["total_sleep_cycles"] == 0

    def test_prune_weak_edges(self):
        from ravana_ml.graph import ConceptGraph
        graph = ConceptGraph(dim=4, max_nodes=100)
        n1 = graph.add_node(vector=np.ones(4, dtype=np.float32), label="a")
        n2 = graph.add_node(vector=np.ones(4, dtype=np.float32), label="b")
        graph.add_edge(n1.id, n2.id, weight=0.05, relation_type="semantic")
        sc = SleepConsolidation(SleepConfig(prune_threshold=0.1))
        pruned = sc._prune_weak_edges(graph, threshold=0.1)
        assert pruned == 1
        assert len(graph.edges) == 0

    def test_normalize_outgoing_weights(self):
        from ravana_ml.graph import ConceptGraph
        graph = ConceptGraph(dim=4, max_nodes=100)
        n1 = graph.add_node(vector=np.ones(4, dtype=np.float32), label="hub")
        n2 = graph.add_node(vector=np.ones(4, dtype=np.float32), label="a")
        n3 = graph.add_node(vector=np.ones(4, dtype=np.float32), label="b")
        graph.add_edge(n1.id, n2.id, weight=0.9, relation_type="semantic")
        graph.add_edge(n1.id, n3.id, weight=0.8, relation_type="semantic")
        sc = SleepConsolidation(SleepConfig(downscaling_budget=1.0))
        sc._normalize_outgoing_weights(graph, budget=1.0)
        total = sum(e.weight for _, e in graph.get_outgoing(n1.id))
        assert total <= 1.0 + 1e-6


# ── BeliefStore Tests ──

class TestBeliefConfig:
    def test_default_config(self):
        cfg = BeliefConfig()
        assert cfg.recency_decay == 0.1


class TestUserBeliefProfile:
    def test_default_profile(self):
        p = UserBeliefProfile(user_id="test")
        assert p.user_id == "test"
        assert p.beliefs == {}
        assert p.contradictions == []


class TestBeliefStore:
    def test_default_init(self):
        bs = BeliefStore()
        assert bs.config.recency_decay == 0.1
        assert bs.current_user is None

    def test_set_user_creates_profile(self):
        bs = BeliefStore()
        bs.set_user("alice")
        assert bs.current_user == "alice"
        assert "alice" in bs.users

    def test_assert_and_query_belief(self):
        bs = BeliefStore()
        bs.set_user("alice")
        bs.assert_belief("trust", "is", "good", confidence=0.9)
        belief = bs.query_belief("trust", "is")
        assert belief is not None
        assert belief[0] == "good"
        assert belief[1] == 0.9

    def test_detect_contradiction(self):
        bs = BeliefStore()
        bs.set_user("alice")
        bs.assert_belief("trust", "is", "good")
        contradiction = bs.detect_contradiction("trust", "is", "bad")
        assert contradiction is not None

    def test_get_all_beliefs_for_subject(self):
        bs = BeliefStore()
        bs.set_user("alice")
        bs.assert_belief("trust", "is", "good")
        bs.assert_belief("trust", "creates", "friendship")
        beliefs = bs.get_all_beliefs_for_subject("trust")
        assert "alice" in beliefs
        assert len(beliefs["alice"]) == 2

    def test_advance_turn(self):
        bs = BeliefStore()
        bs.set_user("alice")
        bs.advance_turn()
        assert bs.global_turn == 1

    def test_get_state_and_set_state(self):
        bs = BeliefStore()
        bs.set_user("alice")
        bs.assert_belief("trust", "is", "good")
        state = bs.get_state()
        assert 'alice' in state['users']
        bs2 = BeliefStore()
        bs2.set_state(state)
        assert bs2.users['alice'].beliefs[("trust", "is")][0] == "good"
