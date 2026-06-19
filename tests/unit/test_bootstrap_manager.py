"""Tests for ravana/src/ravana/bootstrap/manager.py."""

import pytest
from unittest.mock import MagicMock, patch


class TestBootstrapManager:
    def test_init_with_graph_only(self):
        from ravana.bootstrap import BootstrapManager
        from ravana.graph import GraphEngine

        graph = MagicMock(spec=GraphEngine)
        bm = BootstrapManager(graph_engine=graph)
        assert bm.graph_engine is graph
        assert bm.web_learner is None

    def test_init_with_both(self):
        from ravana.bootstrap import BootstrapManager
        from ravana.graph import GraphEngine
        from ravana.web import WebLearner

        graph = MagicMock(spec=GraphEngine)
        web = MagicMock(spec=WebLearner)
        bm = BootstrapManager(graph_engine=graph, web_learner=web)
        assert bm.web_learner is web

    def test_bootstrap_all_calls_seed(self):
        from ravana.bootstrap import BootstrapManager
        from ravana.graph import GraphEngine

        graph = MagicMock(spec=GraphEngine)
        bm = BootstrapManager(graph_engine=graph)
        bm.bootstrap_all()
        graph.seed_concepts.assert_called_once()
        graph.bootstrap_domain_concepts.assert_called_once()

    def test_bootstrap_all_with_web_learner(self):
        from ravana.bootstrap import BootstrapManager
        from ravana.graph import GraphEngine
        from ravana.web import WebLearner

        graph = MagicMock(spec=GraphEngine)
        web = MagicMock(spec=WebLearner)
        web._curiosity_drive_enabled = True
        bm = BootstrapManager(graph_engine=graph, web_learner=web)
        bm.bootstrap_all()
        web._seed_from_graph_curiosity.assert_called_once_with(max_topics=8)

    def test_bootstrap_all_no_curiosity(self):
        from ravana.bootstrap import BootstrapManager
        from ravana.graph import GraphEngine
        from ravana.web import WebLearner

        graph = MagicMock(spec=GraphEngine)
        web = MagicMock(spec=WebLearner)
        web._curiosity_drive_enabled = False
        bm = BootstrapManager(graph_engine=graph, web_learner=web)
        bm.bootstrap_all()
        web._seed_from_graph_curiosity.assert_not_called()

    def test_auto_expand_from_input(self):
        from ravana.bootstrap import BootstrapManager
        from ravana.graph import GraphEngine

        graph = MagicMock(spec=GraphEngine)
        graph.auto_expand_concepts.return_value = 3
        bm = BootstrapManager(graph_engine=graph)
        result = bm.auto_expand_from_input("test text")
        assert result == 3
        graph.auto_expand_concepts.assert_called_once_with("test text")

    def test_curiosity_bootstrap_with_web(self):
        from ravana.bootstrap import BootstrapManager
        from ravana.graph import GraphEngine
        from ravana.web import WebLearner

        graph = MagicMock(spec=GraphEngine)
        web = MagicMock(spec=WebLearner)
        web._seed_from_graph_curiosity.return_value = 5
        bm = BootstrapManager(graph_engine=graph, web_learner=web)
        result = bm.curiosity_bootstrap(max_topics=5)
        assert result == 5
        web._seed_from_graph_curiosity.assert_called_once_with(5)

    def test_curiosity_bootstrap_without_web(self):
        from ravana.bootstrap import BootstrapManager
        from ravana.graph import GraphEngine

        graph = MagicMock(spec=GraphEngine)
        graph.get_curiosity_scores.return_value = [("topic1", 0.8), ("topic2", 0.2), ("topic3", 0.05)]
        bm = BootstrapManager(graph_engine=graph)
        result = bm.curiosity_bootstrap(max_topics=5)
        assert result >= 0

    def test_bootstrap_domain(self):
        from ravana.bootstrap import BootstrapManager
        from ravana.graph import GraphEngine

        graph = MagicMock(spec=GraphEngine)
        bm = BootstrapManager(graph_engine=graph)
        bm.bootstrap_domain()
        graph.bootstrap_domain_concepts.assert_called_once()
