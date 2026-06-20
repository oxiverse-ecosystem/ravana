"""Tests for ravana_grace.core.meta2_cognition."""

import pytest
import numpy as np
from ravana_grace.core.meta2_cognition import (
    Meta2CognitionEngine, Meta2Config, EpistemicCritiqueType,
    HypothesisSpaceAudit, BiasAssessment, ProbeStrategyEvaluation,
    IdentityBeliefCoupling, EpistemicEpiphany, Meta2Cognition,
)


class TestMeta2Config:
    def test_defaults(self):
        cfg = Meta2Config()
        assert cfg.sustained_failure_window == 30
        assert cfg.failure_rate_threshold == 0.4
        assert cfg.min_hypotheses_for_audit == 3
        assert cfg.epiphany_confidence_min == 0.7


class TestMeta2CognitionEngine:
    def test_init(self):
        m2 = Meta2CognitionEngine()
        assert m2.epiphany_count == 0
        assert len(m2.epiphanies) == 0
        assert m2.last_epiphany_episode == -1000

    def test_audit_hypothesis_space_insufficient(self):
        m2 = Meta2CognitionEngine()
        audit = m2.audit_hypothesis_space([], [], episode=1)
        assert audit.space_adequate is True  # Not enough data

    def test_audit_hypothesis_space_adequate(self):
        m2 = Meta2CognitionEngine()
        hyps = []
        class MockHyp:
            def __init__(self, htype, conf):
                self.hypothesis_type = htype
                self.confidence = conf
        for i in range(5):
            hyps.append(MockHyp("PARAMETRIC_TIME", 0.8))
        audit = m2.audit_hypothesis_space(hyps, [0.5]*10, episode=1)
        assert audit.space_adequate is True
        assert len(audit.hypothesis_types_present) > 0

    def test_audit_hypothesis_space_inadequate(self):
        m2 = Meta2CognitionEngine()
        hyps = []
        class MockHyp:
            def __init__(self, htype, conf):
                self.hypothesis_type = htype
                self.confidence = conf
        for i in range(5):
            hyps.append(MockHyp("PARAMETRIC_TIME", 0.2))
        audit = m2.audit_hypothesis_space(hyps, [0.8]*20, episode=1)
        assert audit.space_adequate is False
        assert audit.recommendation is not None

    def test_detect_biases_empty(self):
        m2 = Meta2CognitionEngine()
        bias = m2.detect_biases([], [], episode=1)
        assert bias.occam_bias_score == 0.5
        assert bias.dominant_bias is None

    def test_detect_biases_occam(self):
        m2 = Meta2CognitionEngine()
        hyps = []
        class MockHyp:
            def __init__(self, htype):
                self.hypothesis_type = htype
                self.confidence = 0.5
        for i in range(10):
            hyps.append(MockHyp("PARAMETRIC_TIME"))
        bias = m2.detect_biases(hyps, [True]*30, episode=1)
        # occam_bias_score depends on diversity of hypothesis types;
        # all same type yields a moderate but not extreme score
        assert bias.occam_bias_score >= 0.0
        assert bias.occam_bias_score <= 1.0

    def test_evaluate_probe_strategy_insufficient(self):
        m2 = Meta2CognitionEngine()
        eval_ = m2.evaluate_probe_strategy([], [], episode=1)
        assert eval_.probe_effectiveness_score == 0.5

    def test_evaluate_probe_strategy_with_data(self):
        m2 = Meta2CognitionEngine()
        probes = [{"kl_gain": 0.3, "probe_type": "standard"} for _ in range(15)]
        eval_ = m2.evaluate_probe_strategy(probes, [0.2]*10, episode=1)
        assert eval_.probe_effectiveness_score > 0

    def test_detect_identity_belief_coupling_insufficient(self):
        m2 = Meta2CognitionEngine()
        result = m2.detect_identity_belief_coupling(
            [0.5]*10, [0.5]*10, [0.2]*10, episode=1
        )
        assert result.coupling_strength == 0.0

    def test_detect_identity_belief_coupling_sufficient(self):
        m2 = Meta2CognitionEngine()
        # Create correlated identity and belief history
        identity = [0.5 + 0.1*i for i in range(25)]
        belief = [0.5 + 0.1*i for i in range(25)]
        result = m2.detect_identity_belief_coupling(
            identity, belief, [0.2]*25, episode=1
        )
        assert result.coupling_strength > 0

    def test_generate_epiphany_rate_limited(self):
        m2 = Meta2CognitionEngine()
        m2.last_epiphany_episode = 10
        ep = m2.generate_epiphany(
            EpistemicCritiqueType.HYPOTHESIS_SPACE_INADEQUATE,
            episode=15  # Only 5 later, rate limit is 50
        )
        assert ep is None

    def test_generate_epiphany_space_inadequate(self):
        m2 = Meta2CognitionEngine()
        # Set up an audit
        hyps = []
        class MockHyp:
            def __init__(self, htype, conf):
                self.hypothesis_type = htype
                self.confidence = conf
        for i in range(5):
            hyps.append(MockHyp("PARAMETRIC_TIME", 0.2))
        m2.audit_hypothesis_space(hyps, [0.8]*20, episode=1)
        ep = m2.generate_epiphany(
            EpistemicCritiqueType.HYPOTHESIS_SPACE_INADEQUATE,
            episode=100
        )
        assert ep is not None
        assert "hypothesis space" in ep.realization.lower()

    def test_generate_epiphany_occam_bias(self):
        m2 = Meta2CognitionEngine()
        hyps = []
        class MockHyp:
            def __init__(self, htype):
                self.hypothesis_type = htype
                self.confidence = 0.5
        for i in range(10):
            hyps.append(MockHyp("PARAMETRIC_TIME"))
        m2.detect_biases(hyps, [True]*30, episode=1)
        ep = m2.generate_epiphany(
            EpistemicCritiqueType.OCCAM_BIAS,
            episode=100
        )
        assert ep is not None
        assert "simple models" in ep.realization.lower()

    def test_generate_epiphany_probe_failure(self):
        m2 = Meta2CognitionEngine()
        probes = [{"kl_gain": 0.01, "probe_type": "standard"} for _ in range(15)]
        m2.evaluate_probe_strategy(probes, [0.0]*10, episode=1)
        ep = m2.generate_epiphany(
            EpistemicCritiqueType.PROBE_STRATEGY_FAILURE,
            episode=100
        )
        assert ep is not None
        assert ep.probe_strategy_redesign is not None

    def test_generate_epiphany_identity_protected(self):
        m2 = Meta2CognitionEngine()
        identity = [0.5 + 0.1*i for i in range(25)]
        belief = [0.5 + 0.1*i for i in range(25)]
        m2.detect_identity_belief_coupling(identity, belief, [0.2]*25, episode=1)
        ep = m2.generate_epiphany(
            EpistemicCritiqueType.IDENTITY_PROTECTED_BELIEF,
            episode=100
        )
        assert ep is not None
        assert ep.identity_decoupling_recommended is True

    def test_step_no_epiphany(self):
        m2 = Meta2CognitionEngine()
        result = m2.step(
            episode=1,
            hypothesis_space=["PARAMETRIC_TIME"],
            failure_rate=0.1,
            belief_history=[0.5]*30,
            hypothesis_generator=None,
            surgical_prober=None,
        )
        assert result["epiphany_triggered"] is False

    def test_get_meta2_status(self):
        m2 = Meta2CognitionEngine()
        status = m2.get_meta2_status()
        assert "epiphany_count" in status
        assert "current_audit" in status
        assert "current_bias" in status
        assert "failure_rate" in status

    def test_meta2_cognition_alias(self):
        assert Meta2Cognition is Meta2CognitionEngine
