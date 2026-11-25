"""Regression: TRA typed-slot hook fires with only RAVANA_TRA_SLOTS=1."""
import os
from ravana.chat.engine import CognitiveChatEngine


class _TypedEngine:
    process_turn = CognitiveChatEngine.process_turn
    _source_monitor_decision = staticmethod(
        CognitiveChatEngine.__dict__["_source_monitor_decision"])
    _apply_source_monitoring_gate = CognitiveChatEngine._apply_source_monitoring_gate

    def __init__(self, with_hook=True):
        self.turn_count = 0
        self._last_subject = "mist"
        self._source_gate_ctx = None
        self._pending_source_context = None
        self._last_source_evaluation = None
        self._apply_trusted_edge_choice = None
        self._schema_candidate = None
        if with_hook:
            self._typed_slot_candidate = self._pick

    def _pick(self, raw_input, subject):
        return "effect" if os.environ.get("RAVANA_TRA_SLOTS") == "1" else None

    def _process_turn_impl(self, user_input):
        return "chatter on an early-return path"


def test_typed_slot_hook_fires_without_schema_flag(monkeypatch):
    monkeypatch.setenv("RAVANA_TRA_SLOTS", "1")
    monkeypatch.setenv("RAVANA_SCHEMA", "0")
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "0")
    assert _TypedEngine().process_turn("what is true of mist? Options: effect, cause") == "effect"


def test_typed_slot_hook_inert_when_off_or_absent(monkeypatch):
    monkeypatch.setenv("RAVANA_TRA_SLOTS", "0")
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "0")
    assert _TypedEngine().process_turn("q") == "chatter on an early-return path"
    monkeypatch.setenv("RAVANA_TRA_SLOTS", "1")
    assert _TypedEngine(with_hook=False).process_turn("q") == "chatter on an early-return path"
