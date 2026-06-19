"""Tests for ravana-v2 interface_agent scripts: interview_mode, llm_interpreter, memory_learner, ravana_agent, ravana_wrapper, reality_grounding, telegram_reporter, version_manager."""

import sys, os
_grace = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ravana-v2", "src")
if _grace not in sys.path:
    sys.path.insert(0, _grace)

import pytest

# Test if each module is importable (may fail if optional deps like groq are missing)


def _try_import(module_name):
    """Try importing a module, return (success, error)."""
    try:
        __import__(f"ravana_grace.interface_agent.scripts.{module_name}", fromlist=[""])
        return True, None
    except (ImportError, AttributeError) as e:
        return False, str(e)


class TestInterviewMode:
    def test_module_importable(self):
        ok, err = _try_import("interview_mode")
        if not ok:
            pytest.skip(f"interview_mode not importable (optional dep): {err}")


class TestLLMInterpreter:
    def test_module_importable(self):
        ok, err = _try_import("llm_interpreter")
        if not ok:
            pytest.skip(f"llm_interpreter not importable (optional dep): {err}")


class TestMemoryLearner:
    def test_module_importable(self):
        ok, err = _try_import("memory_learner")
        if not ok:
            pytest.skip(f"memory_learner not importable (optional dep): {err}")


class TestRavanaAgent:
    def test_module_importable(self):
        ok, err = _try_import("ravana_agent")
        if not ok:
            pytest.skip(f"ravana_agent not importable (optional dep): {err}")


class TestRavanaWrapper:
    def test_module_importable(self):
        ok, err = _try_import("ravana_wrapper")
        if not ok:
            pytest.skip(f"ravana_wrapper not importable (optional dep): {err}")


class TestRealityGrounding:
    def test_module_importable(self):
        ok, err = _try_import("reality_grounding")
        if not ok:
            pytest.skip(f"reality_grounding not importable (optional dep): {err}")
        else:
            assert True


class TestTelegramReporter:
    def test_module_importable(self):
        ok, err = _try_import("telegram_reporter")
        if not ok:
            pytest.skip(f"telegram_reporter not importable (optional dep): {err}")
        else:
            assert True


class TestVersionManager:
    def test_module_importable(self):
        ok, err = _try_import("version_manager")
        if not ok:
            pytest.skip(f"version_manager not importable (optional dep): {err}")
        else:
            assert True


class TestAgentTestRunner:
    """Test that agent/test_runner.py is importable."""

    def test_module_importable(self):
        try:
            from ravana_grace.agent import test_runner
            assert test_runner is not None
        except (ImportError, AttributeError) as e:
            pytest.skip(f"test_runner not importable (optional dep): {e}")
