"""Tests for ravana_chat_src: CognitiveFramework."""

import sys, os
_rcs = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ravana_chat_src", "src")
if _rcs not in sys.path:
    sys.path.insert(0, _rcs)

import pytest
import numpy as np

from ravana_chat.cognitive.framework import CognitiveFramework, FrameworkConfig, FrameworkState


class TestFrameworkConfig:
    def test_default_config(self):
        cfg = FrameworkConfig()
        assert cfg.concept_dim == 64
        assert cfg.max_concepts == 10000
        assert cfg.k_active == 5
        assert cfg.hebbian_lr == 0.03


class TestFrameworkState:
    def test_default_state(self):
        state = FrameworkState()
        assert state.dissonance == 0.5
        assert state.identity == 0.5
        assert state.cycle == 0

    def test_snapshot(self):
        state = FrameworkState(dissonance=0.3, identity=0.7, episode=5)
        snap = state.snapshot()
        assert snap["dissonance"] == 0.3
        assert snap["identity"] == 0.7
        assert snap["episode"] == 5


class TestCognitiveFramework:
    def test_requires_initialize(self):
        fw = CognitiveFramework()
        with pytest.raises(RuntimeError, match="initialize"):
            fw.perceive(FrameworkState(), np.zeros(8))

    def test_initialize(self):
        fw = CognitiveFramework(FrameworkConfig(concept_dim=8, max_concepts=50))
        state = fw.initialize()
        assert isinstance(state, FrameworkState)
        assert fw._initialized

    def test_initialize_creates_cognitive_modules(self):
        fw = CognitiveFramework(FrameworkConfig(concept_dim=8, max_concepts=50))
        fw.initialize()
        assert fw.governor is not None
        assert fw.identity is not None
        assert fw.emotion_engine is not None
        assert fw.sleep_engine is not None
        assert fw.meaning_engine is not None
        assert fw.gw_engine is not None
        assert fw.state_manager is not None

    def test_diagnose_after_initialize(self):
        fw = CognitiveFramework(FrameworkConfig(concept_dim=8, max_concepts=50))
        state = fw.initialize()
        diag = fw.diagnose(state)
        assert "state" in diag
        assert "graph" in diag
        assert "cognitive" in diag

    def test_perceive_predict_learn_cycle(self):
        fw = CognitiveFramework(FrameworkConfig(concept_dim=8, max_concepts=50))
        initial_state = fw.initialize()
        x = np.random.randn(8).astype(np.float32)
        y = np.random.randn(8).astype(np.float32)

        concepts = fw.perceive(initial_state, x)
        assert isinstance(concepts, list)

        predictions = fw.predict(initial_state, concepts)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (8,)

        new_state = fw.learn(initial_state, predictions, y, episode=1)
        assert isinstance(new_state, FrameworkState)

    def test_sleep(self):
        fw = CognitiveFramework(FrameworkConfig(concept_dim=4, max_concepts=50))
        state = fw.initialize()
        state = fw.sleep(state)
        assert isinstance(state, FrameworkState)

    def test_infer_no_state_change(self):
        fw = CognitiveFramework(FrameworkConfig(concept_dim=8, max_concepts=50))
        state = fw.initialize()
        x = np.random.randn(8).astype(np.float32)
        saved = {nid: n.activation for nid, n in fw.graph.nodes.items()}
        result = fw.infer(state, x)
        assert "concepts" in result
        assert "predictions" in result
        assert "confidences" in result
        for nid, act in saved.items():
            if nid in fw.graph.nodes:
                assert fw.graph.nodes[nid].activation == act

    def test_query_concept(self):
        fw = CognitiveFramework(FrameworkConfig(concept_dim=8, max_concepts=50))
        state = fw.initialize()
        if fw.graph.nodes:
            nid = list(fw.graph.nodes.keys())[0]
            result = fw.query(state, nid)
            assert "concept" in result

    def test_sleep_engine_available(self):
        fw = CognitiveFramework(FrameworkConfig(concept_dim=8, max_concepts=50))
        fw.initialize()
        assert fw.sleep_engine_available()
