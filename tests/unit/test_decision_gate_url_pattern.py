"""Tests for URL-pattern routing in the decision gate.

Verifies that queries containing URLs route to read_website via
the URL-pattern check (step 0), BEFORE the curiosity/uncertainty path.
"""
import pytest
from unittest.mock import MagicMock
from ravana.agent.decision_gate import decide_tool_use
from ravana.agent.tool_registry import ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry()


def _make_engine(recall=True, uncertainty=0.9, meta_uncertain=True):
    """Mock engine with high curiosity — the condition that was shadowing URLs."""
    engine = MagicMock(spec=[])
    engine._is_recall_query = MagicMock(return_value=recall)
    engine.curiosity_engine = MagicMock()
    engine.curiosity_engine.uncertainty_for = MagicMock(return_value=uncertainty)
    engine.meta_cog = MagicMock()
    engine.meta_cog.current_mode = MagicMock()
    engine.meta_cog.current_mode.value = "UNCERTAIN"
    engine._social_intent = None
    return engine


class TestUrlPatternRouting:
    """The URL pattern check must fire BEFORE curiosity, even when curiosity is maxed."""

    def test_look_up_url_routes_to_read_website(self, registry):
        """'look up https://example.com' — the original failing case."""
        engine = _make_engine(uncertainty=0.9, meta_uncertain=True)
        result = decide_tool_use(engine, "look up https://example.com", registry)
        assert result is not None
        assert result.tool == "read_website"
        assert "url_pattern_detected" in result.reason

    def test_bare_url_routes_to_read_website(self, registry):
        """A bare URL should route to read_website."""
        engine = _make_engine(uncertainty=0.9, meta_uncertain=True)
        result = decide_tool_use(engine, "https://example.com", registry)
        assert result is not None
        assert result.tool == "read_website"

    def test_http_url_routes_to_read_website(self, registry):
        """http:// URLs should also route to read_website."""
        engine = _make_engine(uncertainty=0.9, meta_uncertain=True)
        result = decide_tool_use(engine, "check http://example.com/page", registry)
        assert result is not None
        assert result.tool == "read_website"

    def test_www_url_routes_to_read_website(self, registry):
        """www. prefixed URLs should route to read_website."""
        engine = _make_engine(uncertainty=0.9, meta_uncertain=True)
        result = decide_tool_use(engine, "look at www.example.com", registry)
        assert result is not None
        assert result.tool == "read_website"

    def test_url_with_path_routes_correctly(self, registry):
        """URL with path/query should be extracted fully."""
        engine = _make_engine(uncertainty=0.9, meta_uncertain=True)
        result = decide_tool_use(engine, "https://example.com/path?q=test", registry)
        assert result is not None
        assert result.tool == "read_website"
        assert "https://example.com/path?q=test" in result.arg

    def test_no_url_with_high_curiosity_still_fires_search(self, registry):
        """A knowledge-gap query with NO URL should still fire web_search."""
        engine = _make_engine(uncertainty=0.9, meta_uncertain=True)
        result = decide_tool_use(engine, "what is the capital of France", registry)
        assert result is not None
        assert result.tool == "web_search"

    def test_url_not_in_registry_returns_none(self, registry):
        """If read_website is not in registry, URL check is skipped gracefully."""
        del registry.tools["read_website"]
        engine = _make_engine(uncertainty=0.9, meta_uncertain=True)
        # Without read_website in registry, the curiosity path fires instead
        result = decide_tool_use(engine, "look up https://example.com", registry)
        # Curiosity path fires since read_website isn't available
        assert result is not None
        assert result.tool == "web_search"

    def test_empty_query_returns_none(self, registry):
        engine = _make_engine()
        result = decide_tool_use(engine, "", registry)
        assert result is None

    def test_personal_query_with_url_still_routes_to_read_website(self, registry):
        """URL pattern takes priority even on personal-ish queries."""
        engine = _make_engine(uncertainty=0.9, meta_uncertain=True)
        result = decide_tool_use(engine, "look up https://example.com for me", registry)
        assert result is not None
        assert result.tool == "read_website"

    def test_url_arg_strips_trailing_punctuation(self, registry):
        """URL arg should not include trailing punctuation like periods."""
        engine = _make_engine(uncertainty=0.9, meta_uncertain=True)
        result = decide_tool_use(engine, "look up https://example.com.", registry)
        assert result is not None
        assert result.tool == "read_website"
        assert result.arg == "https://example.com"
