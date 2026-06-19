"""Tests for ravana_grace.core.active_epistemology."""

import pytest
import numpy as np
from ravana_grace.core.active_epistemology import (
    ActiveEpistemologyLayer, InformationGainCalculator,
    HypothesisDrivenActionSelector, VoIConfig,
    InformationGainMethod, ActiveEpistemology,
)


class TestVoIConfig:
    def test_defaults(self):
        cfg = VoIConfig()
        assert cfg.info_gain_weight == 0.3
        assert cfg.uncertainty_threshold == 0.15
        assert len(cfg.probe_actions) >= 4


class TestInformationGainCalculator:
    def test_init(self):
        calc = InformationGainCalculator()
        assert calc.config is not None

    def test_calculate_voi_single_hypothesis(self):
        calc = InformationGainCalculator()
        result = calc.calculate_voi({1: {"belief": 0.5, "confidence": 0.6}}, 0.5, 0.2, ["hold_steady"])
        assert result["hold_steady"] == 0.0

    def test_calculate_voi_two_hypotheses(self):
        calc = InformationGainCalculator()
        hyps = {
            1: {"belief": 0.7, "confidence": 0.8},
            2: {"belief": 0.3, "confidence": 0.4},
        }
        result = calc.calculate_voi(hyps, 0.5, 0.3, ["hold_steady", "aggressive_explore"])
        assert len(result) == 2

    def test_should_probe_yes(self):
        calc = InformationGainCalculator()
        assert calc.should_probe_for_info(
            top_two_confidence_gap=0.05,
            episodes_since_last_probe=100,
            current_uncertainty=0.2,
        ) is True

    def test_should_probe_no_gap_too_large(self):
        calc = InformationGainCalculator()
        assert calc.should_probe_for_info(
            top_two_confidence_gap=0.5,
            episodes_since_last_probe=100,
            current_uncertainty=0.2,
        ) is False


class TestHypothesisDrivenActionSelector:
    def test_init(self):
        calc = InformationGainCalculator()
        sel = HypothesisDrivenActionSelector(calc)
        assert sel.probe_count == 0

    def test_select_action_default(self):
        calc = InformationGainCalculator()
        sel = HypothesisDrivenActionSelector(calc)
        hyps = {1: {"id": 1, "belief": 0.7, "confidence": 0.8, "uncertainty": 0.2}}
        action, metadata = sel.select_action(hyps, 0.7, 0.2, episode=1)
        assert isinstance(action, str)

    def test_select_action_probe(self):
        calc = InformationGainCalculator()
        sel = HypothesisDrivenActionSelector(calc)
        sel.last_probe_episode = 0
        hyps = {
            1: {"id": 1, "belief": 0.7, "confidence": 0.6, "uncertainty": 0.1},
            2: {"id": 2, "belief": 0.3, "confidence": 0.55, "uncertainty": 0.2},
        }
        action, metadata = sel.select_action(hyps, 0.5, 0.3, episode=100)
        assert isinstance(action, str)

    def test_get_experiment_summary(self):
        calc = InformationGainCalculator()
        sel = HypothesisDrivenActionSelector(calc)
        summary = sel.get_experiment_summary()
        assert "total_probes" in summary
        assert "experiments_conducted" in summary


class TestActiveEpistemologyLayer:
    def test_init(self):
        class MockBelief:
            def get_belief_state(self):
                return []
            @property
            def current_belief(self):
                return 0.5
        belief = MockBelief()
        layer = ActiveEpistemologyLayer(belief)
        assert layer.config is not None

    def test_act_and_learn_single_hypothesis(self):
        class MockBelief:
            def get_belief_state(self):
                return []
            @property
            def current_belief(self):
                return 0.5
        belief = MockBelief()
        layer = ActiveEpistemologyLayer(belief)
        action, metadata = layer.act_and_learn(episode=1, pre_state={}, mode=None)
        assert metadata["reason"] == "single_hypothesis"

    def test_get_epistemic_status(self):
        class MockBelief:
            def get_belief_state(self):
                return []
            @property
            def current_belief(self):
                return 0.5
        belief = MockBelief()
        layer = ActiveEpistemologyLayer(belief)
        status = layer.get_epistemic_status()
        assert "action_selection" in status
        assert "total_info_gain" in status

    def test_active_epistemology_alias(self):
        assert ActiveEpistemology is ActiveEpistemologyLayer
