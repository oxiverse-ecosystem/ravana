"""
Tests for the RAVANA mode orchestrator.
"""

from __future__ import annotations

try:
    from .conftest import import_agent
except ImportError:
    from conftest import import_agent

AgentMode, ModeOrchestrator = import_agent("mode_orchestrator", "AgentMode", "ModeOrchestrator")


class FakeVersionManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.improvements = []
        self.tests = []
        self.summary = {
            "pending_improvements": 0,
            "recent_tests": [],
            "active_experiments": 0,
        }

    def get_summary(self):
        return dict(self.summary)

    def queue_improvement(self, description: str, source: str = "web_search", priority: int = 5):
        self.improvements.append(
            {"description": description, "source": source, "priority": priority}
        )

    def record_test(self, test_name: str, status: str, output: str = "", duration_ms: int = 0):
        self.tests.append(
            {
                "test_name": test_name,
                "status": status,
                "output": output,
                "duration_ms": duration_ms,
            }
        )


class FakeRavana:
    def __init__(self):
        self.state = {
            "dissonance": 0.5,
            "dissonance_ema": 0.5,
            "identity": 0.5,
            "wisdom": 0.0,
            "episode": 0,
            "cycle": 0,
            "resolution_streak": 0,
            "governor_mode": "normal",
        }
        self.steps = []

    def get_state_vector(self):
        return dict(self.state)

    def query(self, _question: str):
        return {
            "cognitive_state": {
                "dissonance": self.state["dissonance"],
                "identity": self.state["identity"],
                "wisdom": self.state["wisdom"],
            },
            "governor": {"mode": self.state["governor_mode"]},
        }

    def step(self, correctness: bool, difficulty: float = 0.5, reason: str = ""):
        self.steps.append(
            {"correctness": correctness, "difficulty": difficulty, "reason": reason}
        )
        if correctness:
            self.state["dissonance"] = max(0.0, self.state["dissonance"] - 0.10)
            self.state["identity"] = min(1.0, self.state["identity"] + 0.02)
        else:
            self.state["dissonance"] = min(1.0, self.state["dissonance"] + 0.12)
            self.state["identity"] = max(0.0, self.state["identity"] - 0.04)
        self.state["wisdom"] = min(1.0, self.state["wisdom"] + (0.03 if correctness else 0.01))
        self.state["episode"] += 1
        self.state["cycle"] += 1
        return {
            "post_dissonance": self.state["dissonance"],
            "post_identity": self.state["identity"],
            "wisdom": self.state["wisdom"],
            "mode": self.state["governor_mode"],
        }


class FakeGrounding:
    def __init__(self):
        self.rss_feeds = [("AI Safety", "https://example.com/rss")]
        self.calls = []

    def ingest_news(self, ravana_state, max_items=5, max_scenarios=3):
        self.calls.append(
            {
                "ravana_state": dict(ravana_state),
                "max_items": max_items,
                "max_scenarios": max_scenarios,
            }
        )
        return {
            "news_items": [
                {"title": "Critical AI safety warning", "source": "Example News"},
                {"title": "Research paper finds improved method", "source": "Example News"},
            ],
            "scenarios": [{"action": "increase scrutiny"}],
            "alignment": {"verdict": "misaligned", "alignment_score": 0.22},
            "workspace_bids": [
                {"source": "news:AI Safety", "urgency": 0.91},
            ],
            "max_pressure": 0.83,
            "summary": "2 articles ingested | 1 scenarios built | alignment=misaligned (0.22)",
        }


def make_orchestrator(version_manager_cls=FakeVersionManager, ravana_factory=FakeRavana, grounding_factory=FakeGrounding):
    return ModeOrchestrator(
        groq_api_key="test-key",
        db_path=":memory:",
        version_manager_cls=version_manager_cls,
        ravana_factory=ravana_factory,
        grounding_factory=grounding_factory,
    )


def test_decide_mode_prioritises_research_when_improvements_queue_is_large():
    orch = make_orchestrator()
    orch.vm.summary["pending_improvements"] = 4

    decision = orch.decide_mode()

    assert decision.mode == AgentMode.RESEARCH
    assert "pending improvements" in decision.reason


def test_decide_mode_prioritises_learn_when_experiments_are_active():
    orch = make_orchestrator()
    orch.vm.summary["active_experiments"] = 2

    decision = orch.decide_mode()

    assert decision.mode == AgentMode.LEARN
    assert "active experiments" in decision.reason


def test_decide_mode_uses_interview_for_recent_tests_and_default():
    orch = make_orchestrator()
    orch.vm.summary["recent_tests"] = [{"test_name": "card_honesty_lied"}]

    decision = orch.decide_mode()

    assert decision.mode == AgentMode.INTERVIEW
    assert "Recent tests" in decision.reason


def test_run_interview_mode_records_all_cards_and_queues_followup_on_failures():
    orch = make_orchestrator()

    result = orch.run_interview_mode()

    assert result["mode"] == "interview"
    assert result["cards_run"] == 4
    assert result["passed"] == 4
    assert result["failed"] == 0
    assert len(result["results"]) == 4
    assert len(orch.vm.tests) == 4
    assert not orch.vm.improvements


def test_run_research_mode_queues_improvement_and_builds_report():
    orch = make_orchestrator()

    result = orch.run_research_mode()
    report = orch.build_telegram_report(
        {
            "mode": "research",
            "duration_s": 1.2,
            "ravana_state": orch._state_snapshot(),
            "result": result,
        }
    )

    assert result["mode"] == "research"
    assert result["events"] == 2
    assert result["improvements_queued"] == 1
    assert orch.vm.improvements
    assert orch.vm.improvements[0]["source"] == "news"
    assert "Events: 2" in report
    assert "Alignment: misaligned" in report
    assert "Top pressure: 0.83" in report


if __name__ == "__main__":
    test_decide_mode_prioritises_research_when_improvements_queue_is_large()
    test_decide_mode_prioritises_learn_when_experiments_are_active()
    test_decide_mode_uses_interview_for_recent_tests_and_default()
    test_run_interview_mode_records_all_cards_and_queues_followup_on_failures()
    test_run_research_mode_queues_improvement_and_builds_report()
    print("mode orchestrator tests passed")
