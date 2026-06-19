"""Tests for ravana_grace.core.empathy."""

import pytest
import numpy as np
from ravana_grace.core.empathy import EmpathyEngine, EmpathyConfig, OtherMind


class TestEmpathyConfig:
    def test_defaults(self):
        cfg = EmpathyConfig()
        assert cfg.gp_length_scale == 1.0
        assert cfg.gp_noise_level == 0.1
        assert cfg.max_agents_tracked == 20
        assert cfg.empathy_influence_weight == 0.3


class TestEmpathyEngine:
    def test_init(self):
        ee = EmpathyEngine()
        assert ee.config is not None
        assert len(ee.other_minds) == 0
        assert ee._global_empathy_bias == 0.0

    def test_observe_creates_mind(self):
        ee = EmpathyEngine()
        cues = np.random.randn(10).astype(np.float32)
        mind = ee.observe(agent_id=1, cues=cues)
        assert mind.agent_id == 1
        assert 1 in ee.other_minds

    def test_observe_with_vad_updates_inference(self):
        ee = EmpathyEngine()
        for i in range(10):
            cues = np.random.randn(10).astype(np.float32)
            ee.observe(agent_id=1, cues=cues,
                       observed_vad=(0.3 + i*0.05, 0.5, 0.6))
        mind = ee.other_minds[1]
        v_mean, v_var = mind.inferred_valence
        assert 0 <= v_mean <= 1.0

    def test_infer_emotion_unknown_agent(self):
        ee = EmpathyEngine()
        cues = np.random.randn(10).astype(np.float32)
        mean, std = ee.infer_emotion(agent_id=999, cues=cues)
        assert mean.shape == (3,)
        assert std.shape == (3,)

    def test_infer_emotion_known_agent(self):
        ee = EmpathyEngine()
        for i in range(5):
            cues = np.random.randn(10).astype(np.float32)
            ee.observe(agent_id=1, cues=cues,
                       observed_vad=(0.5, 0.6, 0.7))
        cues = np.random.randn(10).astype(np.float32)
        mean, std = ee.infer_emotion(agent_id=1, cues=cues)
        assert mean.shape == (3,)
        assert std.shape == (3,)

    def test_compute_empathy_distance(self):
        ee = EmpathyEngine()
        own = np.array([0.5, 0.3, 0.5])
        other = np.array([0.5, 0.3, 0.5])
        score = ee.compute_empathy_distance(own, other)
        assert 0 <= score <= 1.0

    def test_compute_empathy_distance_very_different(self):
        ee = EmpathyEngine()
        own = np.array([-1.0, 0.0, 0.0])
        other = np.array([1.0, 1.0, 1.0])
        score = ee.compute_empathy_distance(own, other)
        assert score >= 0  # Lower alignment

    def test_get_trust_score_unknown(self):
        ee = EmpathyEngine()
        assert ee.get_trust_score(999) == 0.5

    def test_get_trust_score_known(self):
        ee = EmpathyEngine()
        cues = np.random.randn(10).astype(np.float32)
        ee.observe(agent_id=1, cues=cues)
        score = ee.get_trust_score(1)
        assert score > 0

    def test_update_trust(self):
        ee = EmpathyEngine()
        cues = np.random.randn(10).astype(np.float32)
        ee.observe(agent_id=1, cues=cues)
        ee.update_trust(agent_id=1, honesty_delta=0.2)
        mind = ee.other_minds[1]
        assert mind.honesty_estimate > 0.5

    def test_get_status(self):
        ee = EmpathyEngine()
        cues = np.random.randn(10).astype(np.float32)
        ee.observe(agent_id=1, cues=cues)
        status = ee.get_status()
        assert "agents_tracked" in status
        assert status["agents_tracked"] == 1
