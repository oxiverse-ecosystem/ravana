"""Tests for ravana/src/ravana/web/learner.py."""

import pytest
import time
import numpy as np
from unittest.mock import MagicMock, patch
from ravana.web.learner import SearchEngine, SearchConfig, SearchError, WebLearner


class TestSearchConfig:
    def test_defaults(self):
        cfg = SearchConfig()
        assert cfg.max_results == 10
        assert cfg.timeout == 5
        assert cfg.cooldown == 60
        assert cfg.max_failures == 3

    def test_custom(self):
        cfg = SearchConfig(max_results=5, timeout=10, cooldown=30, max_failures=2)
        assert cfg.max_results == 5
        assert cfg.cooldown == 30
        assert cfg.max_failures == 2


class TestSearchError:
    def test_is_exception(self):
        err = SearchError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"


class TestSearchEngine:
    def test_init(self):
        se = SearchEngine()
        assert len(se.apis) >= 2
        assert se.config.max_results == 10

    def test_is_api_available_initial(self):
        se = SearchEngine()
        assert se._is_api_available("oxiverse") is True

    def test_is_api_available_after_failures(self):
        se = SearchEngine(config=SearchConfig(max_failures=2, cooldown=3600))
        se._api_failure_counts["oxiverse"] = 3
        se._api_last_failure_time["oxiverse"] = time.time()
        assert se._is_api_available("oxiverse") is False

    def test_is_api_available_after_cooldown(self):
        se = SearchEngine(config=SearchConfig(max_failures=2, cooldown=0.01))
        se._api_failure_counts["oxiverse"] = 3
        se._api_last_failure_time["oxiverse"] = time.time() - 1.0
        assert se._is_api_available("oxiverse") is True
        assert se._api_failure_counts["oxiverse"] == 0  # Reset after cooldown

    def test_record_success(self):
        se = SearchEngine()
        se._api_failure_counts["oxiverse"] = 2
        se._record_success("oxiverse")
        assert se._api_failure_counts["oxiverse"] == 0

    def test_record_failure(self):
        se = SearchEngine()
        initial = se._api_failure_counts["oxiverse"]
        se._record_failure("oxiverse")
        assert se._api_failure_counts["oxiverse"] == initial + 1

    def test_search_raises_when_all_apis_down(self):
        se = SearchEngine(config=SearchConfig(timeout=1, cooldown=3600))
        # Mark all APIs as failed
        for name in se._api_failure_counts:
            se._api_failure_counts[name] = 99
            se._api_last_failure_time[name] = time.time()
        with pytest.raises(SearchError):
            se.search("test query")

    def test_parse_duckduckgo_html_empty(self):
        se = SearchEngine()
        results = se._parse_duckduckgo_html("<html></html>", 5)
        assert results == []

    def test_parse_duckduckgo_html_with_bs4_mock(self):
        se = SearchEngine()
        html = '<a class="result__snippet">test snippet</a>'
        results = se._parse_duckduckgo_html(html, 5)
        assert isinstance(results, list)


class TestWebLearner:
    def test_init(self):
        from ravana.graph import GraphEngine
        from ravana.decoder import DecoderEngine
        graph = MagicMock(spec=GraphEngine)
        decoder = MagicMock(spec=DecoderEngine)
        glove_fn = MagicMock(return_value=np.random.randn(64).astype(np.float32))

        wl = WebLearner(graph, decoder, glove_fn, data_dir="/tmp/test_data")
        assert wl.graph_engine is graph
        assert wl.decoder_engine is decoder
        assert wl._curiosity_drive_enabled is True
        assert wl.search_engine is not None

    def test_queue_background_search(self):
        from ravana.graph import GraphEngine
        from ravana.decoder import DecoderEngine
        graph = MagicMock(spec=GraphEngine)
        decoder = MagicMock(spec=DecoderEngine)
        glove_fn = MagicMock(return_value=np.random.randn(64).astype(np.float32))

        wl = WebLearner(graph, decoder, glove_fn, data_dir="/tmp/test_data")
        wl.queue_background_search("test query")
        assert "test query" in wl._bg_learning_queue

    def test_queue_background_search_dedup(self):
        from ravana.graph import GraphEngine
        from ravana.decoder import DecoderEngine
        graph = MagicMock(spec=GraphEngine)
        decoder = MagicMock(spec=DecoderEngine)
        glove_fn = MagicMock(return_value=np.random.randn(64).astype(np.float32))

        wl = WebLearner(graph, decoder, glove_fn, data_dir="/tmp/test_data")
        wl.queue_background_search("test query")
        wl.queue_background_search("test query")
        assert wl._bg_learning_queue.count("test query") == 1

    def test_notify_user_active_idle(self):
        from ravana.graph import GraphEngine
        from ravana.decoder import DecoderEngine
        graph = MagicMock(spec=GraphEngine)
        decoder = MagicMock(spec=DecoderEngine)
        glove_fn = MagicMock(return_value=np.random.randn(64).astype(np.float32))

        wl = WebLearner(graph, decoder, glove_fn, data_dir="/tmp/test_data")
        wl.notify_user_active()
        wl.notify_user_idle()
        assert wl._curiosity_cycles_this_session == 0

    def test_offline_learn(self):
        from ravana.graph import GraphEngine
        from ravana.decoder import DecoderEngine
        graph = MagicMock(spec=GraphEngine)
        graph._concept_labels = set()
        graph._concept_keywords = {}
        graph._topic_list = []
        # _learn_from_text accesses graph.graph.nodes.values() and .items()
        class _MockGraph:
            """Mock ConceptGraph that returns sensible defaults."""
            def __init__(self):
                self.nodes = type('Nodes', (), {
                    'values': lambda self=None: [],
                    'items': lambda self=None: [],
                })()
                self._vectors_dirty = False
                self._vector_matrix_normed = None
                self._node_id_order = []
            def add_node(self, **kwargs):
                return type('Node', (), {'id': 1, 'label': kwargs.get('label', '')})()
            def get_edge(self, *args):
                return None
            def add_edge(self, *args, **kwargs):
                pass
            def get_node(self, *args):
                return type('Node', (), {'vector': None, 'label': ''})()
            def get_outgoing(self, *args):
                return []
            def get_incoming(self, *args):
                return []
            def _rebuild_vector_matrix(self):
                pass
        graph.graph = _MockGraph()
        graph.get_outgoing = MagicMock(return_value=[])
        graph.get_incoming = MagicMock(return_value=[])
        graph._infer_relation_type = MagicMock(return_value=("semantic", 0.5))
        graph._contradiction_map = {}
        graph._dormant_edges = set()
        decoder = MagicMock(spec=DecoderEngine)
        decoder.neural_decoder = None
        decoder._decoder_vocab_built = False
        glove_fn = MagicMock(return_value=np.random.randn(64).astype(np.float32))

        wl = WebLearner(graph, decoder, glove_fn, data_dir="/tmp/test_data")
        wl._curiosity_drive_enabled = False
        result = wl._offline_learn("test query")
        assert isinstance(result, str)

    def test_extract_related_queries(self):
        from ravana.graph import GraphEngine
        from ravana.decoder import DecoderEngine
        graph = MagicMock(spec=GraphEngine)
        graph._concept_labels = set()
        graph._concept_keywords = {"test": []}
        decoder = MagicMock(spec=DecoderEngine)
        decoder.neural_decoder = None
        glove_fn = MagicMock(return_value=np.random.randn(64).astype(np.float32))

        wl = WebLearner(graph, decoder, glove_fn, data_dir="/tmp/test_data")
        related = wl._extract_related_queries("test query")
        assert isinstance(related, list)
