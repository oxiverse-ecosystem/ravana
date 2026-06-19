"""Tests for ravana_chat_src: BootstrapManager, DecoderEngine, GraphEngine, WebLearner, SearchEngine."""

import sys, os
_rcs = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ravana_chat_src", "src")
if _rcs not in sys.path:
    sys.path.insert(0, _rcs)

import pytest
import numpy as np
import time

from ravana_chat.bootstrap.manager import BootstrapManager
from ravana_chat.decoder.engine import DecoderEngine, DecoderConfig
from ravana_chat.graph.engine import GraphEngine
from ravana_chat.web.learner import WebLearner, SearchEngine, SearchConfig, SearchError


# ── GraphEngine Tests ──

class TestGraphEngine:
    def test_default_init(self):
        ge = GraphEngine(dim=8, seed=42)
        assert ge.dim == 8
        assert ge.graph is not None
        assert ge._concept_labels == set()

    def test_seed_concepts(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: np.random.randn(8).astype(np.float32) * 0.1
        ge.seed_concepts()
        assert len(ge.graph.nodes) > 0
        assert len(ge.graph.edges) > 0
        assert len(ge._concept_labels) > 0

    def test_seed_concepts_sets_concept_pos(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: np.random.randn(8).astype(np.float32) * 0.1
        ge.seed_concepts()
        assert "trust" in ge._concept_pos  # known noun
        assert "good" in ge._concept_pos   # known adj

    def test_auto_expand_empty_text(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: np.random.randn(8).astype(np.float32) * 0.1
        ge.seed_concepts()
        count = ge.auto_expand_concepts("a an the")  # all stop words
        assert count == 0

    def test_bootstrap_domain_concepts(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: np.random.randn(8).astype(np.float32) * 0.1
        ge.bootstrap_domain_concepts()
        # Should add concepts without crashing
        assert ge.graph is not None

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

    def test_find_vector_neighbor_nonexistent(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: None
        assert ge.find_vector_neighbor("nonexistent") is None

    def test_infer_relation_type(self):
        ge = GraphEngine(dim=8, seed=42)
        rel, conf = ge._infer_relation_type("good", "bad", "semantic")
        assert rel == "contrastive"
        assert conf >= 0.85

    def test_correct_relation_types(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: np.random.randn(8).astype(np.float32) * 0.1
        ge.seed_concepts()
        migrated = ge._correct_relation_types()
        assert isinstance(migrated, int)

    def test_recall_hippocampal_missing(self):
        ge = GraphEngine(dim=8, seed=42)
        assert ge.recall_hippocampal("nonexistent") is None

    def test_hippocampal_index_topic(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: None
        ge.hippocampal_index_topic("trust", [], [])
        assert "trust" in ge._topic_list or "trust".lower() in ge._topic_store


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


# ── DecoderEngine Tests ──

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

    def test_web_training_count(self):
        de = DecoderEngine()
        assert de.web_training_count == 0


# ── SearchEngine Tests ──

class TestSearchConfig:
    def test_default_config(self):
        cfg = SearchConfig()
        assert cfg.max_results == 10
        assert cfg.timeout == 5
        assert cfg.cooldown == 60


class TestSearchEngine:
    def test_default_init(self):
        se = SearchEngine()
        assert len(se.apis) >= 1
        assert se.config.max_results == 10

    def test_is_api_available_initially(self):
        se = SearchEngine()
        assert se._is_api_available("oxiverse")

    def test_is_api_available_after_max_failures(self):
        se = SearchEngine(config=SearchConfig(max_failures=2, cooldown=60))
        se._api_failure_counts["oxiverse"] = 2
        se._api_last_failure_time["oxiverse"] = time.time()
        assert not se._is_api_available("oxiverse")

    def test_is_api_available_after_cooldown(self):
        se = SearchEngine(config=SearchConfig(max_failures=2, cooldown=0))
        se._api_failure_counts["oxiverse"] = 2
        se._api_last_failure_time["oxiverse"] = 0
        assert se._is_api_available("oxiverse")

    def test_record_success_and_failure(self):
        se = SearchEngine()
        se._record_failure("oxiverse")
        assert se._api_failure_counts["oxiverse"] == 1
        se._record_success("oxiverse")
        assert se._api_failure_counts["oxiverse"] == 0


# ── WebLearner Tests ──

class TestWebLearner:
    def test_default_init(self):
        ge = GraphEngine(dim=8, seed=42)
        de = DecoderEngine()
        wl = WebLearner(graph_engine=ge, decoder_engine=de, glove_vector_fn=lambda x: None)
        assert wl.graph_engine is ge
        assert wl.decoder_engine is de
        assert wl.search_engine is not None
        assert wl._bg_learning_queue == []
        assert wl._curiosity_drive_enabled is True

    def test_decay_episodic_edges(self):
        ge = GraphEngine(dim=8, seed=42)
        de = DecoderEngine()
        wl = WebLearner(graph_engine=ge, decoder_engine=de, glove_vector_fn=lambda x: None)
        wl._decay_episodic_edges()  # Should not raise

    def test_offline_learn(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: np.random.randn(8).astype(np.float32) * 0.1 if label else None
        de = DecoderEngine()
        wl = WebLearner(graph_engine=ge, decoder_engine=de, glove_vector_fn=ge._glove_vector)
        ge.seed_concepts()
        result = wl._offline_learn("test")
        assert isinstance(result, str)

    def test_update_concept_learning_progress(self):
        ge = GraphEngine(dim=8, seed=42)
        ge._glove_vector = lambda label: np.zeros(8, dtype=np.float32)
        de = DecoderEngine()
        wl = WebLearner(graph_engine=ge, decoder_engine=de, glove_vector_fn=ge._glove_vector)
        wl._update_concept_learning_progress()  # Should not raise

    def test_queue_background_search(self):
        ge = GraphEngine(dim=8, seed=42)
        de = DecoderEngine()
        wl = WebLearner(graph_engine=ge, decoder_engine=de, glove_vector_fn=lambda x: None)
        wl.queue_background_search("trust")
        assert "trust" in wl._bg_learning_queue

    def test_notify_user_active_and_idle(self):
        ge = GraphEngine(dim=8, seed=42)
        de = DecoderEngine()
        wl = WebLearner(graph_engine=ge, decoder_engine=de, glove_vector_fn=lambda x: None)
        wl.notify_user_active()
        wl.notify_user_idle()
