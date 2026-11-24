"""Regression: schema hypothesis decides on early-return turn paths."""
import os

from ravana.chat.engine import CognitiveChatEngine


class _SchemaEngine:
    process_turn = CognitiveChatEngine.process_turn
    _source_monitor_decision = staticmethod(
        CognitiveChatEngine.__dict__["_source_monitor_decision"])
    _apply_source_monitoring_gate = CognitiveChatEngine._apply_source_monitoring_gate

    def __init__(self, with_hook=True):
        self.turn_count = 0
        self._last_subject = "thaw"
        self._source_gate_ctx = None
        self._pending_source_context = None
        self._last_source_evaluation = None
        self._apply_trusted_edge_choice = None
        if with_hook:
            self._schema_candidate = self._pick

    def _pick(self, raw_input, subject):
        return "mist" if os.environ.get("RAVANA_SCHEMA") == "1" else None

    def _process_turn_impl(self, user_input):
        return "chatter on an early-return path"


def test_schema_hook_decides_on_early_return_path(monkeypatch):
    monkeypatch.setenv("RAVANA_SCHEMA", "1")
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "0")
    assert _SchemaEngine().process_turn("what is true of thaw? Options: a, mist") == "mist"


def test_schema_hook_inert_when_off_or_absent(monkeypatch):
    monkeypatch.setenv("RAVANA_SCHEMA", "0")
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "0")
    assert _SchemaEngine().process_turn("q") == "chatter on an early-return path"
    monkeypatch.setenv("RAVANA_SCHEMA", "1")
    assert _SchemaEngine(with_hook=False).process_turn("q") == "chatter on an early-return path"
