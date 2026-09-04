"""Tests for the noun-heuristic path in the decision gate.

Verifies that git-status queries fire github_cli via the noun-heuristic
path when the social-intent classifier is conservative (returns None/general).
"""
import pytest
from unittest.mock import MagicMock
from ravana.agent.decision_gate import decide_tool_use, add_imperative_verbs, _IMPERATIVE_VERBS
from ravana.agent.tool_registry import ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry()


def _make_engine(conservative=True, recall=False, uncertainty=0.0, meta_uncertain=False):
    """Create a mock engine with configurable cognitive signals."""
    engine = MagicMock(spec=[])
    engine._is_recall_query = MagicMock(return_value=recall)
    engine.curiosity_engine = MagicMock()
    engine.curiosity_engine.uncertainty_for = MagicMock(return_value=uncertainty)
    engine.meta_cog = MagicMock()
    engine.meta_cog.current_mode = MagicMock()
    engine.meta_cog.current_mode.value = "UNCERTAIN" if meta_uncertain else "CERTAIN"
    engine._social_intent = None  # Conservative: no classifier
    return engine


class TestNounHeuristicPath:
    """Tests for the noun-heuristic path (step 2b) in decide_tool_use."""

    def test_git_status_query_fires_github_cli(self, registry):
        """The original failing case: 'show me the last 5 commits in this repo'."""
        engine = _make_engine()
        result = decide_tool_use(engine, "show me the last 5 commits in this repo", registry)
        assert result is not None
        assert result.tool == "github_cli"
        assert "noun_heuristic" in result.reason

    def test_list_branches_fires_github_cli(self, registry):
        """'list all branches' should fire github_cli."""
        engine = _make_engine()
        result = decide_tool_use(engine, "list all branches", registry)
        assert result is not None
        assert result.tool == "github_cli"

    def test_git_status_fires_github_cli(self, registry):
        """'git status' should fire github_cli."""
        engine = _make_engine()
        result = decide_tool_use(engine, "git status", registry)
        assert result is not None
        assert result.tool == "github_cli"

    def test_show_diff_fires_github_cli(self, registry):
        """'show diff' should fire github_cli."""
        engine = _make_engine()
        result = decide_tool_use(engine, "show diff", registry)
        assert result is not None
        assert result.tool == "github_cli"

    def test_show_log_fires_github_cli(self, registry):
        """'show me the log' should fire github_cli."""
        engine = _make_engine()
        result = decide_tool_use(engine, "show me the log", registry)
        assert result is not None
        assert result.tool == "github_cli"

    def test_run_script_fires_run_script(self, registry):
        """'run the test script' should fire run_script."""
        engine = _make_engine()
        result = decide_tool_use(engine, "run the test script", registry)
        assert result is not None
        assert result.tool == "run_script"

    def test_search_fires_web_search(self, registry):
        """'search for python tutorials' should fire web_search."""
        engine = _make_engine()
        result = decide_tool_use(engine, "search for python tutorials", registry)
        assert result is not None
        assert result.tool == "web_search"

    def test_question_does_not_fire(self, registry):
        """Questions should NOT fire the noun-heuristic path."""
        engine = _make_engine()
        result = decide_tool_use(engine, "what is git", registry)
        assert result is None

    def test_question_with_commit_does_not_fire(self, registry):
        """'how do I commit changes' should NOT fire (it's a question)."""
        engine = _make_engine()
        result = decide_tool_use(engine, "how do I commit changes", registry)
        assert result is None

    def test_no_tool_noun_does_not_fire(self, registry):
        """'help me understand this' should NOT fire (no tool noun)."""
        engine = _make_engine()
        result = decide_tool_use(engine, "help me understand this", registry)
        assert result is None

    def test_personal_query_does_not_fire(self, registry):
        """'how are you' should NOT fire (personal/social)."""
        engine = _make_engine()
        result = decide_tool_use(engine, "how are you", registry)
        assert result is None

    def test_create_branch_fires_github_cli(self, registry):
        """'create a new branch' should fire github_cli."""
        engine = _make_engine()
        result = decide_tool_use(engine, "create a new branch", registry)
        assert result is not None
        assert result.tool == "github_cli"

    def test_merge_branch_fires_github_cli(self, registry):
        """'merge this branch' should fire github_cli."""
        engine = _make_engine()
        result = decide_tool_use(engine, "merge this branch", registry)
        assert result is not None
        assert result.tool == "github_cli"

    def test_display_commit_history_fires_github_cli(self, registry):
        """'display the commit history' should fire github_cli."""
        engine = _make_engine()
        result = decide_tool_use(engine, "display the commit history", registry)
        assert result is not None
        assert result.tool == "github_cli"

    def test_check_diff_fires_github_cli(self, registry):
        """'check the diff' should fire github_cli."""
        engine = _make_engine()
        result = decide_tool_use(engine, "check the diff", registry)
        assert result is not None
        assert result.tool == "github_cli"

    def test_show_repository_info_fires_github_cli(self, registry):
        """'show repository info' should fire github_cli."""
        engine = _make_engine()
        result = decide_tool_use(engine, "show repository info", registry)
        assert result is not None
        assert result.tool == "github_cli"

    def test_empty_query_returns_none(self, registry):
        """Empty query should return None."""
        engine = _make_engine()
        result = decide_tool_use(engine, "", registry)
        assert result is None

    def test_runtime_expandable_verbs(self, registry):
        """New verbs can be added at runtime via add_imperative_verbs."""
        verb = "transpile"
        assert verb not in _IMPERATIVE_VERBS
        engine = _make_engine()
        result = decide_tool_use(engine, f"{verb} the script", registry)
        assert result is None

        try:
            add_imperative_verbs({verb})
            assert verb in _IMPERATIVE_VERBS

            result = decide_tool_use(engine, f"{verb} the script", registry)
            assert result is not None
            assert result.tool == "run_script"
        finally:
            _IMPERATIVE_VERBS.discard(verb)
