"""Unit tests for the gated pre-articulation source-monitoring wiring.

No engine boot: these tests call the REAL gate methods from CognitiveChatEngine
as unbound functions on tiny fakes, and drive the real ``process_turn`` wrapper
on a stand-in that stubs only the expensive internals. What is exercised is the
shipped wiring code, not a reimplementation of it.

Contract under test (docs/investigations/SOURCE_EVALUATION_WIRING.md):
* flag OFF (default) -> byte-identical behaviour, always
* ctx.source_context present + RAVANA_SOURCE_EVAL=1 -> evaluate_source gates the
  emitted response: ENDORSE keeps it, every other decision becomes its token
* the decision + audit reason are exposed on ctx and on the engine
* a staged context is consumed by exactly one turn (no cross-turn leakage)
* CAR-chain articulation text is never touched by the gate
"""
from types import SimpleNamespace

import pytest

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.models import CognitiveResponseContext
from ravana.chat.source_evaluation import (
    ClaimContext,
    DECISION_TOKENS,
    Decision,
    SourceStatus,
    TrustClass,
)

FLAG = "RAVANA_SOURCE_EVAL"


def _flag(monkeypatch, on):
    if on:
        monkeypatch.setenv(FLAG, "1")
    else:
        monkeypatch.delenv(FLAG, raising=False)


def _gate(response, ctx, claim, context):
    """Run the real emission-boundary gate on a fake engine; return (out, fake)."""
    ctx.source_context = (claim, context)
    fake = SimpleNamespace(
        _last_source_evaluation=None,
        _source_monitor_decision=CognitiveChatEngine._source_monitor_decision,
    )
    out = CognitiveChatEngine._apply_source_monitoring_gate(fake, response, ctx)
    return out, fake


def _credible(origin="src-1"):
    return ClaimContext(
        source=SourceStatus(
            origin_id=origin,
            trust_class=TrustClass.CREDIBLE,
            credibility_token="credible",
        ),
        trusted_source_ids=frozenset({origin}),
    )


class _FakeEngine:
    """Minimal stand-in that supports the real ``process_turn`` wrapper.

    Only the expensive internals are stubbed; the wrapper, the setter, the
    static decision core and the emission gate are the real class methods.
    """

    def __init__(self, response, ctx):
        self._response = response
        self._ctx = ctx
        self._pending_source_context = None
        self._source_gate_ctx = None
        self._last_source_evaluation = None
        self.turn_count = 0
        self.set_source_context_for_turn = (
            lambda *a, **k: CognitiveChatEngine.set_source_context_for_turn(self, *a, **k))
        self._apply_source_monitoring_gate = (
            lambda r, c: CognitiveChatEngine._apply_source_monitoring_gate(self, r, c))
        self._source_monitor_decision = CognitiveChatEngine._source_monitor_decision

    def _style_conditioning_enabled(self):
        return False

    def _record_dialogue_turn(self, *a, **k):
        pass

    def _measure_turn_alignment(self, *a, **k):
        pass

    def _process_turn_impl(self, _user_input):
        # Mimic the real impl: publish a fresh ctx carrying the staged context.
        self._ctx.source_context = self._pending_source_context
        self._source_gate_ctx = self._ctx
        return self._response

    def process_turn(self, user_input):
        return CognitiveChatEngine.process_turn(self, user_input)


class TestFlagOff:
    def test_flag_off_leaves_emitted_response_byte_identical(self, monkeypatch):
        _flag(monkeypatch, False)
        ctx = CognitiveResponseContext()
        out, fake = _gate("the claim is emitted", ctx, "claim", _credible())
        assert out == "the claim is emitted"
        assert fake._last_source_evaluation is None
        assert ctx.source_evaluation_reason == ""

    def test_static_core_inert_when_flag_off(self, monkeypatch):
        _flag(monkeypatch, False)
        ctx = CognitiveResponseContext(source_context=("claim", _credible()))
        assert CognitiveChatEngine._source_monitor_decision(ctx) is None


class TestDecisionGate:
    def test_credible_known_claim_is_emitted(self, monkeypatch):
        _flag(monkeypatch, True)
        ctx = CognitiveResponseContext()
        out, fake = _gate("claim text", ctx, "claim", _credible())
        assert out == "claim text"                      # ENDORSE -> unchanged
        assert fake._last_source_evaluation.decision is Decision.ENDORSE
        assert ctx.source_evaluation_decision.rule == "R_EXPLICIT_CREDIBLE_SOURCE"

    def test_unknown_provenance_yields_abstain_token(self, monkeypatch):
        _flag(monkeypatch, True)
        unknown = ClaimContext(source=SourceStatus(origin_id=""))  # no attribution
        ctx = CognitiveResponseContext()
        out, fake = _gate("claim text", ctx, "claim", unknown)
        assert out == DECISION_TOKENS[Decision.ABSTAIN]  # "N"
        assert fake._last_source_evaluation.decision is Decision.ABSTAIN
        assert ctx.source_evaluation_decision.rule == "R_UNKNOWN_PROVENANCE"

    def test_discredited_after_prior_endorsement_yields_withdraw(self, monkeypatch):
        _flag(monkeypatch, True)
        discredited = ClaimContext(
            source=SourceStatus(
                origin_id="src-2",
                trust_class=TrustClass.DISCREDITED,
                credibility_token="credible",  # stale token must NOT save it
            ),
            prior_decision=Decision.ENDORSE,
            invalidated=True,
        )
        ctx = CognitiveResponseContext()
        out, fake = _gate("claim text", ctx, "claim", discredited)
        assert out == DECISION_TOKENS[Decision.WITHDRAW]         # "W"
        assert fake._last_source_evaluation.decision is Decision.WITHDRAW
        assert ctx.source_evaluation_decision.rule == "R_SOURCE_INVALIDATED"

    def test_dormant_sleeper_abstains_until_cue(self, monkeypatch):
        _flag(monkeypatch, True)
        dormant = ClaimContext(
            source=SourceStatus(
                origin_id="src-3",
                trust_class=TrustClass.CREDIBLE,
                credibility_token="credible",
                active=False,
            ),
            trusted_source_ids=frozenset({"src-3"}),
        )
        ctx = CognitiveResponseContext()
        out, _ = _gate("claim text", ctx, "claim", dormant)
        assert out == DECISION_TOKENS[Decision.ABSTAIN]  # "N"
        assert ctx.source_evaluation_decision.rule == "R_DORMANT_NO_CUE"

        cued = ClaimContext(
            source=dormant.source,
            trusted_source_ids=dormant.trusted_source_ids,
            cue_present=True,
        )
        ctx2 = CognitiveResponseContext()
        out2, _ = _gate("claim text", ctx2, "claim", cued)
        assert out2 == "claim text"                     # cue -> re-evaluated, endorsed
        assert ctx2.source_evaluation_decision.rule == "R_EXPLICIT_CREDIBLE_SOURCE"

    def test_audit_reason_and_decision_exposed_on_ctx_and_engine(self, monkeypatch):
        _flag(monkeypatch, True)
        ctx = CognitiveResponseContext()
        out, fake = _gate("claim text", ctx, "claim", _credible())
        assert ctx.source_evaluation_decision is fake._last_source_evaluation
        assert ctx.source_evaluation_decision is not None
        assert ctx.source_evaluation_reason == ctx.source_evaluation_decision.reason
        assert out == "claim text"

    def test_no_source_context_is_inert_even_with_flag_on(self, monkeypatch):
        _flag(monkeypatch, True)
        ctx = CognitiveResponseContext()               # source_context is None
        fake = SimpleNamespace(
            _last_source_evaluation=None,
            _source_monitor_decision=CognitiveChatEngine._source_monitor_decision,
        )
        out = CognitiveChatEngine._apply_source_monitoring_gate(fake, "claim text", ctx)
        assert out == "claim text"
        assert fake._last_source_evaluation is None


class TestTurnIsolation:
    def test_staged_context_is_consumed_once_no_cross_turn_leakage(self, monkeypatch):
        _flag(monkeypatch, True)
        ctx = CognitiveResponseContext()
        fake = _FakeEngine("claim text", ctx)
        unknown = ClaimContext(source=SourceStatus(origin_id=""))

        fake.set_source_context_for_turn("claim", unknown)
        assert fake.process_turn("q1") == DECISION_TOKENS[Decision.ABSTAIN]  # staged ctx used
        assert fake._pending_source_context is None               # cleared by wrapper
        assert fake._source_gate_ctx is None

        # Second turn: nothing re-staged -> gate must be inert, no leftover ABSTAIN.
        assert fake.process_turn("q2") == "claim text"
        assert fake._last_source_evaluation is not None  # last decision retained for audit

    def test_setter_accepts_two_args_or_single_tuple(self, monkeypatch):
        cc = ClaimContext()
        a = SimpleNamespace(_pending_source_context=None)
        b = SimpleNamespace(_pending_source_context=None)
        CognitiveChatEngine.set_source_context_for_turn(a, "c", cc)
        CognitiveChatEngine.set_source_context_for_turn(b, ("c", cc))
        assert a._pending_source_context == ("c", cc) == b._pending_source_context


class TestArticulationUntouched:
    def test_car_chain_articulation_never_rewritten(self, monkeypatch):
        car = "sun causes rain; rain causes wet ground"  # stand-in articulated chain
        ctx = CognitiveResponseContext()
        fake = _FakeEngine(car, ctx)

        _flag(monkeypatch, False)
        assert fake.process_turn("why does it rain") == car

        _flag(monkeypatch, True)
        assert fake.process_turn("why does it rain") == car  # flag on, no context

        fake.set_source_context_for_turn("claim", _credible())
        assert fake.process_turn("why does it rain") == car  # ENDORSE keeps the chain
