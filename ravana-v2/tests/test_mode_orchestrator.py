"""
Tests for the RAVANA mode orchestrator.
"""

from pathlib import Path
import sys
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "agent"))
sys.path.insert(0, str(project_root / "interface_agent" / "scripts"))

from mode_orchestrator import AgentMode, ModeOrchestrator


class FakeVersionManager:
    def __init__(self, db_path: str = ""):
        self.db_path = db_path
        self.queue_calls = []
        self.test_calls = []
        self.summary = {
            "agent_version": "1.2.3",
            "script_count": 4,
            "recent_changes": [],
            "pending_improvements": 0,
            "active_experiments": 0,
            "recent_tests": [],
            "last_updated": "",
        }

    def get_summary(self):
        return self.summary

    def queue_improvement(self, **kwargs):
        self.queue_calls.append(kwargs)

    def record_test(self, **kwargs):
        self.test_calls.append(kwargs)


class FakeRavana:
    def __init__(self):
        self._dissonance = 0.4
        self._identity = 0.6
        self.steps = []

    def get_diagnosis(self):
        return {"dissonance": self._dissonance, "identity": self._identity, "mode": "normal"}

    def get_state_vector(self):
        return {"dissonance": self._dissonance, "identity": self._identity}

    def step(self, correctness: bool, difficulty: float = 0.5, reason: str = ""):
        self.steps.append({"correctness": correctness, "difficulty": difficulty, "reason": reason})
        pre = self._dissonance
        self._dissonance = min(1.0, self._dissonance + 0.1)
        return {
            "pre_dissonance": pre,
            "post_dissonance": self._dissonance,
            "pre_identity": self._identity,
            "post_identity": self._identity,
            "wisdom": 0.0,
            "mode": "normal",
            "resolution": {"full_resolution": correctness},
        }


class FakeGrounding:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def ingest_news(self, ravana_state=None, max_items=5, max_scenarios=3):
        self.calls.append(
            {
                "ravana_state": ravana_state,
                "max_items": max_items,
                "max_scenarios": max_scenarios,
            }
        )
        return {"news_items": self.events, "summary": f"{len(self.events)} events collected"}


def build_orchestrator(summary=None, events=None):
    fake_vm = FakeVersionManager()
    if summary is not None:
        fake_vm.summary = summary
    fake_ravana = FakeRavana()
    fake_grounding = FakeGrounding(events or [])
    return ModeOrchestrator(
        groq_api_key="test-key",
        db_path="/tmp/test-context.db",
        version_manager_cls=lambda db_path: fake_vm,
        ravana_factory=lambda: fake_ravana,
        grounding_factory=lambda: fake_grounding,
    ), fake_vm, fake_ravana, fake_grounding


def test_decide_mode_uses_summary_counts():
    orch, _, _, _ = build_orchestrator(
        summary={
            "agent_version": "1.2.3",
            "script_count": 4,
            "recent_changes": [],
            "pending_improvements": 5,
            "active_experiments": 0,
            "recent_tests": [],
            "last_updated": "",
        }
    )

    assert orch.decide_mode().mode == AgentMode.RESEARCH

    orch, _, _, _ = build_orchestrator(
        summary={
            "agent_version": "1.2.3",
            "script_count": 4,
            "recent_changes": [],
            "pending_improvements": 0,
            "active_experiments": 2,
            "recent_tests": [],
            "last_updated": "",
        }
    )

    assert orch.decide_mode().mode == AgentMode.LEARN

    orch, _, _, _ = build_orchestrator(
        summary={
            "agent_version": "1.2.3",
            "script_count": 4,
            "recent_changes": [],
            "pending_improvements": 0,
            "active_experiments": 0,
            "recent_tests": [{"test_name": "x"}],
            "last_updated": "",
        }
    )

    assert orch.decide_mode().mode == AgentMode.INTERVIEW


def test_research_mode_queues_improvement_from_ingested_news():
    events = [
        {"title": f"AI safety update {i}", "source": "news"}
        for i in range(11)
    ]
    orch, fake_vm, _, fake_grounding = build_orchestrator(events=events)

    result = orch.run_research_mode()

    assert result["mode"] == "research"
    assert result["events"] == 11
    assert result["improvements_queued"] == 1
    assert fake_vm.queue_calls
    assert fake_grounding.calls
    assert result["notes"]["cycle_summary"] == "11 events collected"


def test_learn_mode_tracks_before_and_after_dissonance():
    events = [
        {"title": "Trust and honesty event", "source": "news"},
        {"title": "Risk and exploration event", "source": "news"},
    ]
    orch, _, fake_ravana, _ = build_orchestrator(events=events)

    result = orch.run_learn_mode()

    assert result["mode"] == "learn"
    assert result["events_processed"] == 2
    assert result["results"][0]["D_before"] == 0.4
    assert result["results"][0]["D_after"] == 0.5
    assert result["results"][0]["D_delta"] == pytest.approx(0.1)
    assert fake_ravana.steps[0]["reason"] == "learn:news"
