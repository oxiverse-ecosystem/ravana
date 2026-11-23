"""Regression: source gate fires on early-return process_turn paths."""
import os
from types import SimpleNamespace

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.source_evaluation import (
    ClaimContext, Decision, SourceStatus, TrustClass)


class _EarlyReturnEngine:
    """Duck-typed engine whose impl returns before publishing its ctx."""

    process_turn = CognitiveChatEngine.process_turn
    _apply_source_monitoring_gate = CognitiveChatEngine._apply_source_monitoring_gate
    _source_monitor_decision = staticmethod(
        CognitiveChatEngine.__dict__["_source_monitor_decision"])

    def __init__(self, ctx):
        self.turn_count = 0
        self._pending_source_context = ("Frost creates dew.", ctx)
        self._source_gate_ctx = None
        self._last_source_evaluation = None
        self._last_subject = ""
        self._apply_trusted_edge_choice = None

    def _process_turn_impl(self, user_input):
        return "chatter that never publishes a response context"


def _ctx(trust, token=None):
    return ClaimContext(source=SourceStatus(
        origin_id="source-1", source_type="web_fact", reliability=0.9,
        trust_class=trust, credibility_token=token))


def test_gate_fires_on_early_return_path_credible(monkeypatch):
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "1")
    eng = _EarlyReturnEngine(_ctx(TrustClass.CREDIBLE, "credible"))
    out = eng.process_turn("Did you learn this from a credible source?")
    assert out == "chatter that never publishes a response context"
    assert eng._last_source_evaluation.decision is Decision.ENDORSE
    assert eng._pending_source_context is None and eng._source_gate_ctx is None


def test_gate_fires_on_early_return_path_unknown(monkeypatch):
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "1")
    eng = _EarlyReturnEngine(_ctx(TrustClass.UNKNOWN))
    assert eng.process_turn("Did you learn this from a credible source?") == "N"
    assert eng._last_source_evaluation.decision is Decision.ABSTAIN


def test_no_leak_across_turns_and_flag_off_inert(monkeypatch):
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "1")
    eng = _EarlyReturnEngine(_ctx(TrustClass.CREDIBLE, "credible"))
    eng.process_turn("first")
    out = eng.process_turn("second")
    assert eng._last_source_evaluation.decision is Decision.ENDORSE  # no stale ctx
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "0")
    eng2 = _EarlyReturnEngine(_ctx(TrustClass.UNKNOWN))
    assert eng2.process_turn("q") == "chatter that never publishes a response context"
