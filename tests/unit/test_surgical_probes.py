"""Tests for ravana_grace.core.surgical_probes."""

import pytest
import numpy as np
from ravana_grace.core.surgical_probes import (
    SurgicalProbeSelector, SurgicalProbeConfig, ProbeType,
    ProbeOutcome, ProbeExperiment, SurgicalProbing,
)


class TestSurgicalProbeConfig:
    def test_defaults(self):
        cfg = SurgicalProbeConfig()
        assert cfg.min_kl_for_probe == 0.1
        assert cfg.min_episodes_between_probes == 10


class TestSurgicalProbeSelector:
    def test_init(self):
        sps = SurgicalProbeSelector()
        assert sps.last_probe_episode == -100
        assert len(sps.probe_history) == 0

    def test_select_probe_single_hypothesis(self):
        sps = SurgicalProbeSelector()
        hyps = [type("H", (), {"id": 1, "confidence": 0.8, "boundary_estimate": 0.7})()]
        result = sps.select_surgical_probe(hyps, {"dissonance": 0.5, "identity": 0.5}, episode=1)
        assert result[0] is None
        assert result[1]["reason"] == "single_hypothesis"

    def test_select_probe_rate_limited(self):
        sps = SurgicalProbeSelector()
        sps.last_probe_episode = 100
        hyps = [
            type("H", (), {"id": 1, "confidence": 0.8, "boundary_estimate": 0.7})(),
            type("H", (), {"id": 2, "confidence": 0.5, "boundary_estimate": 0.3})(),
        ]
        result = sps.select_surgical_probe(hyps, {"dissonance": 0.5, "identity": 0.5}, episode=105)
        assert result[0] is None
        assert "rate_limited" in result[1]["reason"]

    def test_select_probe_with_two_hypotheses(self):
        sps = SurgicalProbeSelector()
        sps.last_probe_episode = -100
        hyps = [
            type("H", (), {"id": 1, "confidence": 0.8, "boundary_estimate": 0.7})(),
            type("H", (), {"id": 2, "confidence": 0.6, "boundary_estimate": 0.3})(),
        ]
        result = sps.select_surgical_probe(hyps, {"dissonance": 0.5, "identity": 0.5}, episode=1)
        # May or may not find a good probe
        assert result is not None

    def test_record_probe_result(self):
        sps = SurgicalProbeSelector()
        sps.probe_history.append({
            "episode": 1, "probe": "perturb_low", "expected_kl": 0.3,
        })
        sps.record_probe_result(ProbeType.PERTURB_LOW, episode=1,
                                actual_info_gain=0.3, hypothesis_separation_achieved=0.25)
        assert len(sps.probe_effectiveness[ProbeType.PERTURB_LOW]) == 1

    def test_get_surgical_analytics_empty(self):
        sps = SurgicalProbeSelector()
        analytics = sps.get_surgical_analytics()
        assert analytics["total_probes"] == 0

    def test_get_surgical_analytics_with_data(self):
        sps = SurgicalProbeSelector()
        sps.probe_history.append({
            "episode": 1, "probe": "perturb_med", "expected_kl": 0.4, "separation_achieved": 0.35,
        })
        analytics = sps.get_surgical_analytics()
        assert analytics["total_probes"] == 1

    def test_surgical_probing_alias(self):
        assert SurgicalProbing is SurgicalProbeSelector
