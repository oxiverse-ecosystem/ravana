"""Tests for ravana_grace.core.social_epistemology."""

import pytest
from ravana_grace.core.social_epistemology import (
    SocialEpistemologyEngine, SocialEpistemicConfig,
    AgentType, ConflictType, TrustScore, AgentBelief,
    BeliefConflict, ConsensusBelief, SocialEpistemology,
)


class TestSocialEpistemicConfig:
    def test_defaults(self):
        cfg = SocialEpistemicConfig()
        assert cfg.initial_trust == 0.5
        assert cfg.conflict_threshold == 0.15
        assert cfg.min_agents_for_consensus == 3


class TestSocialEpistemologyEngine:
    def test_init(self):
        se = SocialEpistemologyEngine()
        assert se.current_episode == 0
        assert len(se.agents) == 0

    def test_register_agent(self):
        se = SocialEpistemologyEngine()
        ok = se.register_agent("agent1", AgentType.RAVANA)
        assert ok is True
        assert "agent1" in se.agents

    def test_register_expert_agent(self):
        se = SocialEpistemologyEngine()
        ok = se.register_agent("expert1", AgentType.EXPERT)
        assert ok is True
        assert se.trust_scores["expert1"].reliability == 0.7

    def test_register_novice_agent(self):
        se = SocialEpistemologyEngine()
        ok = se.register_agent("novice1", AgentType.NOVICE)
        assert ok is True
        assert se.trust_scores["novice1"].reliability == 0.3

    def test_register_adversary(self):
        se = SocialEpistemologyEngine()
        ok = se.register_agent("adv1", AgentType.ADVERSARY)
        assert ok is True
        assert "adv1" in se.adversarial_agents

    def test_register_max_agents(self):
        se = SocialEpistemologyEngine()
        for i in range(12):
            ok = se.register_agent(f"agent{i}", AgentType.PEER)
        assert ok is False  # Max network size is 10

    def test_update_belief(self):
        se = SocialEpistemologyEngine()
        se.register_agent("agent1", AgentType.PEER)
        result = se.update_belief("agent1", 0.75, 0.6, 0.1)
        assert "conflicts_detected" in result
        assert se.agent_beliefs["agent1"].boundary_estimate == 0.75

    def test_update_belief_unregistered(self):
        se = SocialEpistemologyEngine()
        result = se.update_belief("unknown", 0.75, 0.6, 0.1)
        assert "error" in result

    def test_detect_conflicts(self):
        se = SocialEpistemologyEngine()
        se.register_agent("a1", AgentType.PEER)
        se.register_agent("a2", AgentType.PEER)
        se.update_belief("a1", 0.9, 0.8, 0.1)
        se.update_belief("a2", 0.3, 0.8, 0.1)
        assert len(se.active_conflicts) > 0

    def test_form_consensus_insufficient_agents(self):
        se = SocialEpistemologyEngine()
        se.register_agent("a1", AgentType.PEER)
        consensus = se.form_consensus()
        assert consensus is None

    def test_form_consensus(self):
        se = SocialEpistemologyEngine()
        for i in range(4):
            se.register_agent(f"a{i}", AgentType.PEER)
            se.update_belief(f"a{i}", 0.75, 0.6, 0.1)
        consensus = se.form_consensus()
        assert consensus is not None
        assert isinstance(consensus, ConsensusBelief)
        assert 0 <= consensus.boundary_estimate <= 1.0

    def test_resolve_conflict(self):
        se = SocialEpistemologyEngine()
        se.register_agent("a1", AgentType.PEER)
        se.register_agent("a2", AgentType.PEER)
        se.update_belief("a1", 0.9, 0.8, 0.1)
        se.update_belief("a2", 0.3, 0.8, 0.1)
        conflict_id = list(se.active_conflicts.keys())[0]
        result = se.resolve_conflict(conflict_id, actual_boundary=0.75)
        assert "predicted_boundary" in result
        assert 0 <= result["predicted_boundary"] <= 1.0

    def test_run_adversarial_test_rate_limited(self):
        se = SocialEpistemologyEngine()
        se.last_adversarial_test = 100
        result = se.run_adversarial_test(episode=105)
        assert result.get("skipped") is True

    def test_detect_deception(self):
        se = SocialEpistemologyEngine()
        se.register_agent("a1", AgentType.PEER)
        se.trust_scores["a1"].honesty = 0.1
        alerts = se.detect_deception()
        assert len(alerts) > 0

    def test_step(self):
        se = SocialEpistemologyEngine()
        for i in range(4):
            se.register_agent(f"a{i}", AgentType.PEER)
            se.update_belief(f"a{i}", 0.75 + i*0.05, 0.6, 0.1)
        result = se.step(episode=1)
        assert "consensus_formed" in result
        assert "metrics" in result

    def test_get_status(self):
        se = SocialEpistemologyEngine()
        status = se.get_status()
        assert "num_agents" in status
        assert "trust_scores" in status

    def test_social_epistemology_alias(self):
        assert SocialEpistemology is SocialEpistemologyEngine


class TestTrustScore:
    def test_init(self):
        ts = TrustScore(agent_id="test")
        assert ts.reliability == 0.5
        assert ts.composite_trust < 1.0

    def test_update_reliability(self):
        ts = TrustScore(agent_id="test")
        for _ in range(10):
            ts.update_reliability(0.75, 0.75, episode=1)
        assert ts.reliability > 0.5


class TestAgentBelief:
    def test_init(self):
        ab = AgentBelief(agent_id="a", boundary_estimate=0.75, confidence=0.5, uncertainty=0.2)
        assert ab.boundary_estimate == 0.75

    def test_belief_distance_same(self):
        a = AgentBelief(agent_id="a", boundary_estimate=0.75, confidence=0.5, uncertainty=0.1)
        b = AgentBelief(agent_id="b", boundary_estimate=0.75, confidence=0.5, uncertainty=0.1)
        assert a.belief_distance(b) == 0.0

    def test_belief_distance_different(self):
        a = AgentBelief(agent_id="a", boundary_estimate=0.75, confidence=0.5, uncertainty=0.1)
        b = AgentBelief(agent_id="b", boundary_estimate=0.25, confidence=0.8, uncertainty=0.1)
        assert a.belief_distance(b) > 0
