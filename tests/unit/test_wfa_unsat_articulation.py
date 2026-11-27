"""Unit tests for the default-OFF unsaturating articulation hook.

No engine boot: the real gate module is used directly and the shipped
``process_turn`` wiring is driven on a tiny stand-in, exactly as the existing
source-monitoring wiring tests do.
"""
from types import SimpleNamespace

from ravana.chat import wfa_unsat_articulation as ua
from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.models import CognitiveResponseContext
from ravana.chat.source_evaluation import (
    ClaimContext, Decision, SourceStatus, TrustClass,
)

PROMPTS = {
    "free": "Answer freely. Endorse any claim you recognise. Claim: x\nDecision:",
    "restricted": ("Endorse a claim ONLY if you are certain its provenance is one "
                   "of the trusted sources listed in CONTEXT. Otherwise ABSTAIN. "
                   "Claim: x\nDecision:"),
    "invalidate": ("New evidence: source s has been shown to be unreliable and "
                   "fabricated its claims. Reply WITHDRAW or MAINTAIN. Claim: x"),
    "forced": ("Reply with ONE token only. Y = endorse the claim, N = abstain / "
               "do not endorse, W = withdraw a prior assertion. Claim: x\nReply:"),
}


def _flag(monkeypatch, on):
    if on:
        monkeypatch.setenv(ua.FLAG, "1")
    else:
        monkeypatch.delenv(ua.FLAG, raising=False)
    monkeypatch.delenv("RAVANA_SOURCE_EVAL", raising=False)
    monkeypatch.delenv("RAVANA_PRIOR_ACT_TRACE", raising=False)


def _credible(claim="claim", reliability=0.9, origin="src-1"):
    return ClaimContext(source=SourceStatus(
        origin_id=origin, trust_class=TrustClass.CREDIBLE,
        credibility_token="credible", reliability=reliability))


def _discredited(claim="claim", reliability=0.9):
    return ClaimContext(source=SourceStatus(
        origin_id="src-2", trust_class=TrustClass.DISCREDITED,
        reliability=reliability), prior_decision=Decision.ENDORSE,
        invalidated=True)


class TestDefaultOff:
    def test_decision_core_inert_without_flag(self, monkeypatch):
        _flag(monkeypatch, False)
        assert ua.decide("claim", _credible(), "restricted") == (None, {
            "mode": "restricted", "applied": False})

    def test_emission_adapter_byte_identical_without_flag(self, monkeypatch):
        _flag(monkeypatch, False)
        ctx = CognitiveResponseContext(source_context=("claim", _credible()))
        out, audit = ua.apply_at_emission("hello world", ctx, PROMPTS["restricted"])
        assert out == "hello world" and audit["applied"] is False

    def test_engine_hook_byte_identical_without_flag(self, monkeypatch):
        _flag(monkeypatch, False)
        ctx = CognitiveResponseContext(source_context=("claim", _credible()))
        fake = SimpleNamespace()
        out = CognitiveChatEngine._apply_unsat_articulation_hook(
            fake, "hello world", ctx, PROMPTS["restricted"])
        assert out == "hello world"


class TestInstructionMode:
    def test_all_four_instructions_detected(self):
        assert ua.mode_from_instruction(PROMPTS["free"]) == "free"
        assert ua.mode_from_instruction(PROMPTS["restricted"]) == "restricted"
        assert ua.mode_from_instruction(PROMPTS["invalidate"]) == "invalidate"
        assert ua.mode_from_instruction(PROMPTS["forced"]) == "forced"

    def test_unrelated_instruction_defers(self, monkeypatch):
        _flag(monkeypatch, True)
        ctx = CognitiveResponseContext(source_context=("claim", _credible()))
        out, audit = ua.apply_at_emission("hello world", ctx, "tell me a joke")
        assert out == "hello world" and audit["applied"] is False


class TestMechanism:
    def test_deterministic_and_parseable_token(self, monkeypatch):
        _flag(monkeypatch, True)
        toks = [ua.decide("claim", _credible(), "forced")[0] for _ in range(3)]
        assert toks == [toks[0]] * 3 and toks[0] in ("Y", "N")

    def test_fail_closed_without_stored_origin(self, monkeypatch):
        _flag(monkeypatch, True)
        ctx = ClaimContext(source=SourceStatus(origin_id=""), familiarity=1.0)
        assert ua.decide("claim", ctx, "restricted")[0] is None or \
            ua.decide("claim", ctx, "restricted")[0] == "abstain"
        tok, audit = ua.decide("claim", ctx, "forced")
        assert tok in (None, "N")
        assert audit["applied"] is False

    def test_bare_response_deferred_even_with_flag_on(self, monkeypatch):
        _flag(monkeypatch, True)
        ctx = CognitiveResponseContext(source_context=("claim", _credible()))
        for bare in ("Y", "N", "W", "endorse", "abstain"):
            out, audit = ua.apply_at_emission(bare, ctx, PROMPTS["forced"])
            assert out == bare and audit["applied"] is False

    def test_stored_confidence_is_load_bearing(self, monkeypatch):
        _flag(monkeypatch, True)
        def rate(conf):
            vals = [ua.decide("claim-%d" % i, _discredited("claim-%d" % i, conf),
                              "invalidate")[0] for i in range(80)]
            return sum(v == "withdraw" for v in vals) / len(vals)
        low, high = rate(0.1), rate(0.9)
        assert high > low + 0.15, (low, high)
        assert 0.0 < low < 1.0 and 0.0 < high < 1.0

    def test_graded_confidence_is_reported(self, monkeypatch):
        _flag(monkeypatch, True)
        _tok, audit = ua.decide("claim", _credible("claim", 0.9), "restricted")
        assert 0.0 < audit["confidence"] < 1.0
        assert isinstance(audit["recollected"], bool)
        assert audit["rule"].startswith("R_")


class _FakeEngine:
    """Minimal stand-in driving the REAL process_turn wrapper (no boot)."""
    def __init__(self, response, ctx):
        self._response, self._ctx = response, ctx
        self._pending_source_context = None
        self._source_gate_ctx = None
        self._last_source_evaluation = None
        self.set_source_context_for_turn = (
            lambda *a, **k: CognitiveChatEngine.set_source_context_for_turn(self, *a, **k))
        self._source_monitor_decision = CognitiveChatEngine._source_monitor_decision
        self._apply_source_monitoring_gate = (
            lambda r, c: CognitiveChatEngine._apply_source_monitoring_gate(self, r, c))
        self._apply_unsat_articulation_hook = (
            lambda r, c, u: CognitiveChatEngine._apply_unsat_articulation_hook(self, r, c, u))

    def _style_conditioning_enabled(self):
        return False

    def _record_dialogue_turn(self, *a, **k):
        pass

    def _measure_turn_alignment(self, *a, **k):
        pass

    def _process_turn_impl(self, _user_input):
        self._ctx.source_context = self._pending_source_context
        self._source_gate_ctx = self._ctx
        return self._response

    def process_turn(self, prompt):
        return CognitiveChatEngine.process_turn(self, prompt)


class TestProcessTurnWiring:
    def test_flag_off_process_turn_unchanged(self, monkeypatch):
        _flag(monkeypatch, False)
        ctx = CognitiveResponseContext()
        fake = _FakeEngine("generated claim text", ctx)
        fake.set_source_context_for_turn("claim", _credible())
        assert fake.process_turn(PROMPTS["restricted"]) == "generated claim text"

    def test_flag_on_process_turn_articulates_token(self, monkeypatch):
        _flag(monkeypatch, True)
        ctx = CognitiveResponseContext()
        fake = _FakeEngine("generated claim text", ctx)
        fake.set_source_context_for_turn("claim", _credible())
        out = fake.process_turn(PROMPTS["forced"])
        assert out in ("Y", "N") and getattr(ctx, "unsat_articulation_audit", None)
        assert ctx.unsat_articulation_audit["applied"] is True
