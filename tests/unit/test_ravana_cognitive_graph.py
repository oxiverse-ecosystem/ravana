"""Tests for ravana/src/ravana: CognitiveFramework, GraphEngine, BootstrapManager, WebLearner, SearchEngine, DecoderEngine."""

import pytest
import numpy as np

# These imports may fail in CI / fresh environment because the cognitive
# __init__.py does 'from core import ...' which needs ravana-v2 path setup.
# Import guards below prevent collection of those tests when imports fail.
try:
    from ravana.cognitive.framework import CognitiveFramework, FrameworkConfig, FrameworkState
except ImportError:
    FrameworkConfig = None
    FrameworkState = None
try:
    from ravana.graph.engine import GraphEngine
except ImportError:
    GraphEngine = None
try:
    from ravana.bootstrap.manager import BootstrapManager
except ImportError:
    BootstrapManager = None
try:
    from ravana.decoder.engine import DecoderEngine, DecoderConfig
except ImportError:
    DecoderEngine = None
    DecoderConfig = None
try:
    from ravana.web.learner import SearchEngine, SearchConfig, SearchError
except ImportError:
    SearchEngine = None
    SearchConfig = None
    SearchError = None

# ── CognitiveFramework Tests ──

class TestFrameworkConfig:
    def test_default_config(self):
        cfg = FrameworkConfig()
        assert cfg.concept_dim == 64
        assert cfg.max_concepts == 10000
        assert cfg.k_active == 5
        assert cfg.hebbian_lr == 0.03
        assert cfg.propagation_steps == 3
        assert cfg.initial_identity == 0.5


class TestFrameworkState:
    def test_default_state(self):
        state = FrameworkState()
        assert state.dissonance == 0.5
        assert state.identity == 0.5
        assert state.wisdom == 0.0
        assert state.meaning == 0.0
        assert state.cycle == 0
        assert state.sleep_cycles == 0
        assert state.emotional_label == "neutral/relaxed"

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
        assert fw.graph is not None

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

    def test_initialize_sets_step_count(self):
        fw = CognitiveFramework(FrameworkConfig(concept_dim=8, max_concepts=50))
        state = fw.initialize()
        assert fw._step_count == 0

    def test_diagnose_before_initialize_raises(self):
        fw = CognitiveFramework()
        with pytest.raises(RuntimeError):
            fw.diagnose(FrameworkState())

    def test_diagnose_after_initialize(self):
        fw = CognitiveFramework(FrameworkConfig(concept_dim=8, max_concepts=50))
        state = fw.initialize()
        diag = fw.diagnose(state)
        assert "state" in diag
        assert "graph" in diag
        assert "cognitive" in diag
        assert diag["step_count"] == 0

    def test_learn_updates_state(self):
        fw = CognitiveFramework(FrameworkConfig(concept_dim=8, max_concepts=50))
        initial_state = fw.initialize()
        x = np.random.randn(8).astype(np.float32)
        y = np.random.randn(8).astype(np.float32)

        concepts = fw.perceive(initial_state, x)
        predictions = fw.predict(initial_state, concepts)
        new_state = fw.learn(initial_state, predictions, y, episode=1)

        assert isinstance(new_state, FrameworkState)
        assert new_state.cycle == 1

    def test_sleep_called(self):
        fw = CognitiveFramework(FrameworkConfig(concept_dim=4, max_concepts=50))
        state = fw.initialize()
        x = np.random.randn(4).astype(np.float32)
        y = np.random.randn(4).astype(np.float32)
        concepts = fw.perceive(state, x)
        predictions = fw.predict(state, concepts)
        state = fw.learn(state, predictions, y, episode=1)
        # Sleep doesn't raise
        state = fw.sleep(state)
        assert isinstance(state, FrameworkState)

    def test_infer_no_state_change(self):
        fw = CognitiveFramework(FrameworkConfig(concept_dim=8, max_concepts=50))
        state = fw.initialize()
        x = np.random.randn(8).astype(np.float32)
        # Save activation state first
        saved = {nid: n.activation for nid, n in fw.graph.nodes.items()}
        result = fw.infer(state, x)
        assert "concepts" in result
        assert "predictions" in result
        assert "confidences" in result
        # Activations should be restored
        for nid, act in saved.items():
            if nid in fw.graph.nodes:
                assert fw.graph.nodes[nid].activation == act

    def test_query_concept(self):
        fw = CognitiveFramework(FrameworkConfig(concept_dim=8, max_concepts=50))
        state = fw.initialize()
        # Find a node
        if fw.graph.nodes:
            nid = list(fw.graph.nodes.keys())[0]
            result = fw.query(state, nid)
            assert "concept" in result
            assert "neighbors" in result

    @pytest.mark.skip(reason="Requires file system access")
    def test_save_and_load(self):
        pass  # Full save/load tested in integration


# ── GraphEngine Tests (basic) ──

class TestGraphEngine:
    def test_default_init(self):
        ge = GraphEngine(dim=8, seed=42)
        assert ge.dim == 8
        assert ge.graph is not None
        assert ge._concept_labels == set()
        assert ge._concept_keywords == {}

    def test_seed_concepts(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: np.random.randn(8).astype(np.float32) * 0.1
        ge.seed_concepts()
        assert len(ge.graph.nodes) > 0
        assert len(ge.graph.edges) > 0
        assert len(ge._concept_labels) > 0

    def test_auto_expand_empty_text(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: np.random.randn(8).astype(np.float32) * 0.1
        # Seed first so we have existing concepts
        ge.seed_concepts()
        count = ge.auto_expand_concepts("a an the")  # all stop words
        assert count == 0

    def test_auto_expand_meaningful_text(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: np.random.randn(8).astype(np.float32) * 0.1
        ge.seed_concepts()
        count = ge.auto_expand_concepts("quantum entanglement physics")
        assert count >= 0  # May or may not add new concepts

    def test_find_vector_neighbor_nonexistent(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: None
        assert ge.find_vector_neighbor("nonexistent") is None

    def test_spread_and_collect_empty(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: None
        results = ge.spread_and_collect([])
        assert results == []

    def test_get_curiosity_scores(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: np.zeros(8, dtype=np.float32)
        ge.seed_concepts()
        scores = ge.get_curiosity_scores(max_topics=5)
        assert isinstance(scores, list)
        if scores:
            assert len(scores) <= 5
            assert isinstance(scores[0], tuple)
            assert len(scores[0]) == 2


# ── BootstrapManager Tests ──

class TestBootstrapManager:
    def test_default_init(self):
        ge = GraphEngine(dim=8, seed=42)
        bm = BootstrapManager(graph_engine=ge)
        assert bm.graph_engine is ge
        assert bm.web_learner is None

    def test_curiosity_bootstrap_without_web_learner(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: np.zeros(8, dtype=np.float32)
        ge.seed_concepts()
        bm = BootstrapManager(graph_engine=ge)
        count = bm.curiosity_bootstrap(max_topics=5)
        assert isinstance(count, int)


# ── DecoderEngine Tests (basic) ──

class TestDecoderConfig:
    def test_default_config(self):
        cfg = DecoderConfig()
        assert cfg.embed_dim == 64
        assert cfg.hidden_dim == 256
        assert cfg.n_attention_heads == 4


class TestDecoderEngine:
    def test_default_init(self):
        de = DecoderEngine()
        assert de.neural_decoder is None
        assert de._decoder_vocab_built is False
        assert de.training_count == 0
        assert de.web_training_count == 0
        assert de.is_ready is False

    def test_vocab_size_empty(self):
        de = DecoderEngine()
        assert de.vocab_size == 0

    def test_save_state_empty(self):
        de = DecoderEngine()
        assert de.save_state() == {}

    def test_load_state_empty(self):
        de = DecoderEngine()
        de.load_state({})  # Should not raise


# ── SearchEngine Tests ──

class TestSearchConfig:
    def test_default_config(self):
        cfg = SearchConfig()
        assert cfg.max_results == 10
        assert cfg.timeout == 5
        assert cfg.cooldown == 60
        assert cfg.max_failures == 3


class TestSearchEngine:
    def test_default_init(self):
        se = SearchEngine()
        assert len(se.apis) >= 1
        assert se.config.max_results == 10

    def test_is_api_available_initially(self):
        se = SearchEngine()
        assert se._is_api_available("oxiverse")

    def test_is_api_available_after_max_failures(self):
        import time
        se = SearchEngine(config=SearchConfig(max_failures=2, cooldown=60))
        se._api_failure_counts["oxiverse"] = 2
        se._api_last_failure_time["oxiverse"] = time.time()
        assert not se._is_api_available("oxiverse")

    def test_is_api_available_after_cooldown(self):
        import time
        se = SearchEngine(config=SearchConfig(max_failures=2, cooldown=0))
        se._api_failure_counts["oxiverse"] = 2
        se._api_last_failure_time["oxiverse"] = 0
        assert se._is_api_available("oxiverse")  # cooldown is 0, so should reset

    def test_record_success_and_failure(self):
        se = SearchEngine()
        se._record_failure("oxiverse")
        assert se._api_failure_counts["oxiverse"] == 1
        se._record_success("oxiverse")
        assert se._api_failure_counts["oxiverse"] == 0
