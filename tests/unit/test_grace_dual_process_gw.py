"""Tests for ravana-v2/grace/core: DualProcessController, GlobalWorkspace, MetaCognition."""

import pytest
import numpy as np
from ravana_grace.core.dual_process import DualProcessController, DualProcessConfig, ProcessingRoute, RouteDecision
from ravana_grace.core.global_workspace import GlobalWorkspace, GWConfig, GWContent
from ravana_grace.core.meta_cognition import MetaCognition, MetaCognitiveConfig, EpistemicMode


# ── DualProcessController Tests ──

class TestProcessingRoute:
    def test_values(self):
        assert ProcessingRoute.SYSTEM1_FAST.value == "system1_fast"
        assert ProcessingRoute.SYSTEM2_SLOW.value == "system2_slow"


class TestDualProcessConfig:
    def test_default_config(self):
        cfg = DualProcessConfig()
        assert cfg.system2_confidence_threshold == 0.3
        assert cfg.system2_novelty_threshold == 0.6
        assert cfg.max_consecutive_system2 == 5

    def test_custom_config(self):
        cfg = DualProcessConfig(system2_confidence_threshold=0.5, max_consecutive_system2=3)
        assert cfg.system2_confidence_threshold == 0.5


class TestRouteDecision:
    def test_decision_attributes(self):
        d = RouteDecision(route=ProcessingRoute.SYSTEM1_FAST, confidence=0.8, novelty=0.1,
                         stakes=0.1, reason="default", cognitive_load=0.0, fluency=0.5)
        assert d.route == ProcessingRoute.SYSTEM1_FAST
        assert d.reason == "default"
        assert d.confidence == 0.8


class TestDualProcessController:
    def test_default_init(self):
        dp = DualProcessController()
        assert dp._consecutive_system2 == 0
        assert dp._last_route == ProcessingRoute.SYSTEM1_FAST

    def test_high_confidence_defaults_to_system1(self):
        dp = DualProcessController()
        np.random.seed(42)
        decision = dp.decide_route(confidence=0.8, novelty=0.1, stakes=0.1)
        assert decision.route == ProcessingRoute.SYSTEM1_FAST

    def test_low_confidence_triggers_system2(self):
        dp = DualProcessController()
        np.random.seed(42)
        decision = dp.decide_route(confidence=0.1, novelty=0.1, stakes=0.1)
        # Low confidence + no high novelty or stakes → system2 reasons includes "low_confidence"
        assert decision.route == ProcessingRoute.SYSTEM2_SLOW
        assert "low_confidence" in decision.reason

    def test_high_novelty_triggers_system2(self):
        dp = DualProcessController()
        np.random.seed(42)
        decision = dp.decide_route(confidence=0.8, novelty=0.9, stakes=0.1)
        assert decision.route == ProcessingRoute.SYSTEM2_SLOW
        assert "high_novelty" in decision.reason

    def test_high_stakes_triggers_system2(self):
        dp = DualProcessController()
        np.random.seed(42)
        decision = dp.decide_route(confidence=0.8, novelty=0.1, stakes=0.9)
        assert decision.route == ProcessingRoute.SYSTEM2_SLOW
        assert "high_stakes" in decision.reason

    def test_max_consecutive_system2_forces_system1(self):
        dp = DualProcessController(config=DualProcessConfig(max_consecutive_system2=2))
        dp._consecutive_system2 = 2
        decision = dp.decide_route(confidence=0.1, novelty=0.5, stakes=0.1)
        assert decision.route == ProcessingRoute.SYSTEM1_FAST
        assert "system2_cooldown" in decision.reason

    def test_fluency_heuristic(self):
        dp = DualProcessController()
        decision = dp.decide_route(confidence=0.9, novelty=0.5, stakes=0.5, fluency=0.9)
        assert decision.route == ProcessingRoute.SYSTEM1_FAST
        assert "fluency_heuristic" in decision.reason

    def test_cool_down_after_system2_burst(self):
        dp = DualProcessController(config=DualProcessConfig(max_consecutive_system2=2, system2_cooldown_cycles=2))
        dp._consecutive_system2 = 2
        decision1 = dp.decide_route(confidence=0.1, novelty=0.5, stakes=0.1)
        assert decision1.route == ProcessingRoute.SYSTEM1_FAST
        assert dp._cooldown_counter > 0

    def test_get_system2_rate(self):
        dp = DualProcessController()
        # System 2 is triggered by max_consecutive_system2 limit, then cooldown forces System 1
        # After multiple calls, rate reflects proportion of System 2 decisions
        dp._consecutive_system2 = 0
        # Override hysteresis to 0 for deterministic tests
        dp.config.system1_hysteresis = 0.0
        for _ in range(10):
            dp.decide_route(confidence=0.1, novelty=0.5, stakes=0.1)
        rate = dp.get_system2_rate()
        # With max_consecutive_system2=5, first 5 are system2, then cooldown forces system1 for 3,
        # then remaining 2 could be system2 → rate around 0.5-0.7
        assert 0.3 < rate < 1.0

    def test_get_system2_rate_empty(self):
        dp = DualProcessController()
        assert dp.get_system2_rate() == 0.0

    def test_get_status(self):
        dp = DualProcessController()
        status = dp.get_status()
        assert 'current_route' in status
        assert 'consecutive_system2' in status
        assert 'system2_rate' in status


# ── GlobalWorkspace Tests ──

class TestGWConfig:
    def test_default_config(self):
        cfg = GWConfig()
        assert cfg.capacity == 7
        assert cfg.broadcast_threshold == 0.3


class TestGWContent:
    def test_creation(self):
        import time
        content = GWContent(source="emotion", payload={"v": 0.5}, urgency=0.7, valence=0.5, timestamp=time.time(), episode=1)
        assert content.source == "emotion"
        assert content.urgency == 0.7


class TestGlobalWorkspace:
    def test_default_init(self):
        gw = GlobalWorkspace()
        assert len(gw._bids) == 0
        assert gw.config.capacity == 7

    def test_submit_bid(self):
        gw = GlobalWorkspace()
        gw.submit_bid(source="emotion", payload={"v": 0.5}, urgency=0.7, valence=0.5, episode=1)
        assert len(gw._bids) == 1

    def test_compete_returns_none_when_no_bids(self):
        gw = GlobalWorkspace()
        assert gw.compete() is None

    def test_compete_returns_winner_above_threshold(self):
        gw = GlobalWorkspace()
        gw.submit_bid(source="emotion", payload={"v": 0.5}, urgency=0.8, valence=0.5, episode=1)
        winner = gw.compete()
        assert winner is not None
        assert winner.source == "emotion"

    def test_compete_returns_none_below_threshold(self):
        gw = GlobalWorkspace(config=GWConfig(broadcast_threshold=0.9))
        gw.submit_bid(source="emotion", payload={"v": 0.5}, urgency=0.3, valence=0.1, episode=1)
        winner = gw.compete()
        assert winner is None

    def test_compete_adds_to_buffer(self):
        gw = GlobalWorkspace()
        gw.submit_bid(source="test", payload={"k": "v"}, urgency=0.8, valence=0.5, episode=1)
        gw.compete()
        assert len(gw._buffer) == 1
        assert gw._buffer[0].source == "test"

    def test_get_recent(self):
        gw = GlobalWorkspace()
        gw.submit_bid(source="a", payload={}, urgency=0.8, valence=0.5, episode=1)
        gw.compete()
        gw.submit_bid(source="b", payload={}, urgency=0.8, valence=0.5, episode=2)
        gw.compete()
        recent = gw.get_recent(k=2)
        assert len(recent) == 2

    def test_get_context_vector(self):
        gw = GlobalWorkspace()
        vec = gw.get_context_vector()
        assert len(vec) == 3
        assert vec[1] == 0.3

    def test_get_context_vector_after_broadcasts(self):
        gw = GlobalWorkspace()
        gw.submit_bid(source="emotion", payload={"v": 0.5}, urgency=0.8, valence=0.5, episode=1)
        gw.compete()
        vec = gw.get_context_vector()
        assert vec[0] > 0.0

    def test_get_active_sources(self):
        gw = GlobalWorkspace()
        gw.submit_bid(source="emotion", payload={}, urgency=0.8, valence=0.5, episode=1)
        gw.compete()
        gw.submit_bid(source="meaning", payload={}, urgency=0.8, valence=0.5, episode=2)
        gw.compete()
        sources = gw.get_active_sources()
        assert len(sources) >= 1

    def test_accumulate_pressure(self):
        gw = GlobalWorkspace()
        gw.accumulate_pressure(1.0)
        assert gw._pressure == 1.0

    def test_should_sleep(self):
        gw = GlobalWorkspace()
        assert gw.should_sleep() is False
        gw.accumulate_pressure(15.0)
        assert gw.should_sleep() is True

    def test_get_status(self):
        gw = GlobalWorkspace()
        status = gw.get_status()
        assert 'buffer_size' in status
        assert 'broadcast_count' in status
        assert 'pressure' in status
