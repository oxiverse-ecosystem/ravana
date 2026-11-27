"""Regression tests for the default-off claim-ID prior-act trace.

The trace lets a source-withdrawal decision refer to the engine's OWN earlier
ENDORSE of the same claim, instead of a single global decision slot.  It is
inert unless RAVANA_PRIOR_ACT_TRACE=1.  These tests use the same duck-typed
early-return engine as test_source_gate_early_return.py: no live engine boot,
no weight store, no benchmark code.
"""
import os
from types import SimpleNamespace

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.source_evaluation import (
    ClaimContext, Decision, SourceStatus, TrustClass, claim_act_key)


class _Engine:
    process_turn = CognitiveChatEngine.process_turn
    _apply_source_monitoring_gate = CognitiveChatEngine._apply_source_monitoring_gate
    _source_monitor_decision = staticmethod(
        CognitiveChatEngine.__dict__["_source_monitor_decision"])

    def __init__(self, claim, ctx):
        self._claim = claim
        self._pending_source_context = (claim, ctx)
        self._source_gate_ctx = None
        self._last_source_evaluation = None
        self._source_act_trace = {}
        self._last_subject = ""
        self._apply_trusted_edge_choice = None

    def stage(self, claim, ctx):
        self._pending_source_context = (claim, ctx)

    def _process_turn_impl(self, user_input):
        # An early return: the staged context must still reach the gate.
        return "free-form generated claim"


def credible():
    return ClaimContext(source=SourceStatus(
        origin_id="src-1", source_type="curated", reliability=0.9,
        trust_class=TrustClass.CREDIBLE, credibility_token="credible"))


def invalidated():
    return ClaimContext(source=SourceStatus(
        origin_id="src-1", source_type="curated", trust_class=TrustClass.DISCREDITED),
        invalidated=True, prior_decision=None)


def unknown():
    return ClaimContext(source=SourceStatus(
        origin_id="src-1", trust_class=TrustClass.UNKNOWN))


def test_key_is_text_only_and_item_metadata_invariant():
    a = claim_act_key("The sky is blue.")
    assert a == claim_act_key("The sky is blue.")
    assert a != claim_act_key("The sky is green.")
    assert a.startswith("wfa_") and len(a) == 16
    # Item-side labels never enter the key.
    item = {"claim": "The sky is blue.", "expected_answer": "W",
            "source_status": "credible", "source_id": "forged"}
    assert claim_act_key(item["claim"]) == a


def test_flag_off_is_inert_and_reproduces_no_prior(monkeypatch):
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "1")
    monkeypatch.delenv("RAVANA_PRIOR_ACT_TRACE", raising=False)
    eng = _Engine("Claim A.", credible())
    assert eng.process_turn("q") == "free-form generated claim"
    eng.stage("Claim A.", invalidated())
    out = eng.process_turn("q")
    assert out == "N"
    assert eng._last_source_evaluation.rule == "R_INVALIDATION_NO_PRIOR"
    assert eng._source_act_trace == {}


def test_flag_explicit_zero_is_inert(monkeypatch):
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "1")
    monkeypatch.setenv("RAVANA_PRIOR_ACT_TRACE", "0")
    eng = _Engine("Claim A.", credible())
    eng.process_turn("q")
    eng.stage("Claim A.", invalidated())
    assert eng.process_turn("q") == "N"
    assert eng._source_act_trace == {}


def test_endorse_act_is_recorded_under_claim_key(monkeypatch):
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "1")
    monkeypatch.setenv("RAVANA_PRIOR_ACT_TRACE", "1")
    claim = "Claim A."
    eng = _Engine(claim, credible())
    eng.process_turn("q")
    assert eng._source_act_trace == {claim_act_key(claim): Decision.ENDORSE}
    # Only the decision enum is stored -- no claim text, item label or answer.
    assert all(v is Decision.ENDORSE for v in eng._source_act_trace.values())


def test_invalidated_turn_recollects_prior_endorse(monkeypatch):
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "1")
    monkeypatch.setenv("RAVANA_PRIOR_ACT_TRACE", "1")
    claim = "Claim A."
    eng = _Engine(claim, credible())
    eng.process_turn("q")
    assert eng._last_source_evaluation.decision is Decision.ENDORSE
    eng.stage(claim, invalidated())
    out = eng.process_turn("q")
    assert out == "W"
    assert eng._last_source_evaluation.decision is Decision.WITHDRAW
    assert eng._last_source_evaluation.rule == "R_SOURCE_INVALIDATED"
    # The remembered act is not consumed -- later turns can still recollect it.
    assert eng._source_act_trace[claim_act_key(claim)] is Decision.ENDORSE


def test_no_substitution_without_invalidation(monkeypatch):
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "1")
    monkeypatch.setenv("RAVANA_PRIOR_ACT_TRACE", "1")
    claim = "Claim A."
    eng = _Engine(claim, credible())
    eng.process_turn("q")
    eng.stage(claim, unknown())
    assert eng.process_turn("q") == "N"
    assert eng._last_source_evaluation.rule == "R_UNKNOWN_PROVENANCE"


def test_key_isolation_between_claims(monkeypatch):
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "1")
    monkeypatch.setenv("RAVANA_PRIOR_ACT_TRACE", "1")
    eng = _Engine("Claim A.", credible())
    eng.process_turn("q")
    eng.stage("Claim B.", invalidated())
    assert eng.process_turn("q") == "N"
    assert eng._last_source_evaluation.rule == "R_INVALIDATION_NO_PRIOR"


def test_prior_endorse_supplied_by_caller_is_not_overwritten(monkeypatch):
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "1")
    monkeypatch.setenv("RAVANA_PRIOR_ACT_TRACE", "1")
    claim = "Claim A."
    eng = _Engine(claim, credible())
    eng.process_turn("q")
    ctx = ClaimContext(source=invalidated().source, invalidated=True,
                       prior_decision=Decision.ABSTAIN)
    eng.stage(claim, ctx)
    assert eng.process_turn("q") == "W"   # recollected ENDORSE wins over ABSTAIN


def test_malformed_staged_context_fails_open(monkeypatch):
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "1")
    monkeypatch.setenv("RAVANA_PRIOR_ACT_TRACE", "1")
    eng = _Engine("Claim A.", credible())
    eng.process_turn("q")
    # A non-dataclass staged object would make replace() raise; the wrapper
    # must swallow that and leave the normal gate behaviour intact.
    eng.stage("Claim A.", SimpleNamespace(invalidated=True, prior_decision=None))
    out = eng.process_turn("q")
    assert out in ("N", "?")
    assert eng._source_act_trace[claim_act_key("Claim A.")] is Decision.ENDORSE
