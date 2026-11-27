"""Boot-free regression tests for the M1 unified durable-provenance adapter.

No live engine, weight store or benchmark is used. The duck-typed engine
borrows the REAL ``process_turn`` and the REAL emission gate from
``CognitiveChatEngine``; only the expensive inner turn is stubbed. Every
decision is produced by the unmodified ``evaluate_source`` rules.

Run with:  uv run python -m pytest tests/unit/test_durable_provenance.py -q
"""
import json
import os

from ravana.chat import durable_provenance as dp
from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.source_evaluation import (
    ClaimContext, Decision, SourceStatus, TrustClass, claim_act_key,
    context_from_wfa_item, evaluate_source)


class _Engine:
    """Duck-typed early-return engine: production boundary, no boot."""
    process_turn = CognitiveChatEngine.process_turn
    _apply_source_monitoring_gate = CognitiveChatEngine._apply_source_monitoring_gate
    _source_monitor_decision = staticmethod(
        CognitiveChatEngine.__dict__["_source_monitor_decision"])

    def __init__(self, save_path):
        self._save_path = str(save_path)
        self._pending_source_context = None
        self._source_gate_ctx = None
        self._last_source_evaluation = None
        self._source_act_trace = {}
        self._last_subject = "sky"
        self._apply_trusted_edge_choice = None

    def stage(self, claim, ctx):
        self._pending_source_context = (claim, ctx)

    def _process_turn_impl(self, user_input):
        # An early return: the staged context must still reach the gate.
        return "free-form generated claim"


def _credible(origin="src-1"):
    return ClaimContext(source=SourceStatus(
        origin_id=origin, source_type="curated", reliability=0.9,
        trust_class=TrustClass.CREDIBLE, credibility_token="credible"))


def _invalidated(origin="src-1", prior=None):
    return ClaimContext(source=SourceStatus(
        origin_id=origin, source_type="curated", trust_class=TrustClass.DISCREDITED),
        invalidated=True, prior_decision=prior)


def _arms(monkeypatch, durable):
    monkeypatch.setenv("RAVANA_SOURCE_EVAL", "1")
    monkeypatch.delenv("RAVANA_PRIOR_ACT_TRACE", raising=False)
    if durable:
        monkeypatch.setenv("RAVANA_DURABLE_PROVENANCE", "1")
    else:
        monkeypatch.delenv("RAVANA_DURABLE_PROVENANCE", raising=False)


def test_flag_is_strictly_default_off(monkeypatch):
    monkeypatch.delenv("RAVANA_DURABLE_PROVENANCE", raising=False)
    assert dp.enabled() is False
    monkeypatch.setenv("RAVANA_DURABLE_PROVENANCE", "0")
    assert dp.enabled() is False
    monkeypatch.setenv("RAVANA_DURABLE_PROVENANCE", "true")
    assert dp.enabled() is False
    monkeypatch.setenv("RAVANA_DURABLE_PROVENANCE", "1")
    assert dp.enabled() is True


def test_anchor_matches_existing_provenance_anchor_and_is_item_invariant():
    claim = "The sky is blue."
    key = dp.claim_key(claim)
    assert key == claim_act_key(claim) == "wfa_" + __import__("hashlib").sha1(
        claim.encode("utf-8")).hexdigest()[:12]
    assert key != dp.claim_key("The sky is green.")
    item = {"claim": claim, "expected_answer": "W", "source_id": "forged",
            "credibility": 1.0}
    assert dp.claim_key(item["claim"]) == key


def test_flag_off_is_inert_and_writes_no_sidecar(monkeypatch, tmp_path):
    _arms(monkeypatch, durable=False)
    eng = _Engine(tmp_path / "weights.pkl")
    eng.stage("Claim A.", _credible())
    assert eng.process_turn("q") == "free-form generated claim"
    eng.stage("Claim A.", _invalidated())
    assert eng.process_turn("q") == "N"
    assert eng._last_source_evaluation.rule == "R_INVALIDATION_NO_PRIOR"
    assert not (tmp_path / dp.FILE_NAME).exists()


def test_flag_explicit_zero_is_inert(monkeypatch, tmp_path):
    _arms(monkeypatch, durable=False)
    monkeypatch.setenv("RAVANA_DURABLE_PROVENANCE", "0")
    eng = _Engine(tmp_path / "weights.pkl")
    eng.stage("Claim A.", _credible())
    eng.process_turn("q")
    eng.stage("Claim A.", _invalidated())
    assert eng.process_turn("q") == "N"
    assert not (tmp_path / dp.FILE_NAME).exists()


def test_endorse_persists_all_four_aspects(monkeypatch, tmp_path):
    _arms(monkeypatch, durable=True)
    claim = "Claim A."
    eng = _Engine(tmp_path / "weights.pkl")
    eng.stage(claim, _credible())
    assert eng.process_turn("q") == "free-form generated claim"
    assert eng._last_source_evaluation.decision is Decision.ENDORSE

    path = tmp_path / dp.FILE_NAME
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rec = payload["records"][dp.claim_key(claim)]
    # identity + provenance + prior endorsement + negation, all in one record
    assert rec["claim_key"] == dp.claim_key(claim)
    assert rec["claim"] == claim
    assert rec["subject"] == "sky"
    assert rec["subject_key"] == dp.subject_key("sky")
    assert rec["origin_id"] == "src-1"
    assert rec["edge_kind"] == "curated"
    assert rec["trust_class"] == "credible"
    assert rec["prior_decision"] == "ENDORSE"
    assert rec["ever_endorsed"] is True
    assert rec["negated"] is False
    assert "\r\n" not in path.read_text(encoding="utf-8")


def test_cold_reload_restores_record_and_withdraws(monkeypatch, tmp_path):
    """Fresh engine, empty in-memory state, same data dir: a true cold reload."""
    _arms(monkeypatch, durable=True)
    claim = "Claim A."
    first = _Engine(tmp_path / "weights.pkl")
    first.stage(claim, _credible())
    first.process_turn("q")
    before = json.loads((tmp_path / dp.FILE_NAME).read_text(encoding="utf-8"))

    cold = _Engine(tmp_path / "weights.pkl")   # no in-memory trace carried over
    assert not hasattr(cold, "_durable_provenance_store")
    cold.stage(claim, _invalidated())          # no prior decision supplied
    assert cold.process_turn("q") == "W"
    assert cold._last_source_evaluation.decision is Decision.WITHDRAW
    assert cold._last_source_evaluation.rule == "R_SOURCE_INVALIDATED"

    after = json.loads((tmp_path / dp.FILE_NAME).read_text(encoding="utf-8"))
    rec_before = before["records"][dp.claim_key(claim)]
    rec_after = after["records"][dp.claim_key(claim)]
    for field in ("claim_key", "subject_key", "origin_id", "edge_kind",
                  "ever_endorsed"):
        assert rec_after[field] == rec_before[field], field
    assert rec_after["negated"] is True        # negation status now durable
    assert rec_after["prior_decision"] == "WITHDRAW"


def test_withdrawal_survives_a_second_cold_reload(monkeypatch, tmp_path):
    """A later WITHDRAW must not erase the earlier ENDORSE (monotonic bit)."""
    _arms(monkeypatch, durable=True)
    claim = "Claim A."
    e1 = _Engine(tmp_path / "weights.pkl")
    e1.stage(claim, _credible())
    e1.process_turn("q")
    e2 = _Engine(tmp_path / "weights.pkl")
    e2.stage(claim, _invalidated())
    assert e2.process_turn("q") == "W"
    e3 = _Engine(tmp_path / "weights.pkl")
    e3.stage(claim, _invalidated())
    assert e3.process_turn("q") == "W"
    assert e3._last_source_evaluation.decision is Decision.WITHDRAW


def test_no_false_withdrawal_without_prior_act(monkeypatch, tmp_path):
    _arms(monkeypatch, durable=True)
    eng = _Engine(tmp_path / "weights.pkl")
    eng.stage("Never taught.", _invalidated())
    assert eng.process_turn("q") == "N"
    assert eng._last_source_evaluation.rule == "R_INVALIDATION_NO_PRIOR"


def test_key_isolation_between_claims(monkeypatch, tmp_path):
    _arms(monkeypatch, durable=True)
    eng = _Engine(tmp_path / "weights.pkl")
    eng.stage("Claim A.", _credible())
    eng.process_turn("q")
    eng.stage("Claim B.", _invalidated())
    assert eng.process_turn("q") == "N"
    assert eng._last_source_evaluation.rule == "R_INVALIDATION_NO_PRIOR"


def test_provenance_is_first_write_wins_and_negation_is_monotonic(monkeypatch, tmp_path):
    _arms(monkeypatch, durable=True)
    eng = _Engine(tmp_path / "weights.pkl")
    eng.stage("Claim A.", _credible("src-1"))
    eng.process_turn("q")
    eng.stage("Claim A.", _invalidated("src-2"))
    assert eng.process_turn("q") == "W"
    rec = dp.store_for_engine(eng).ledger.get(dp.claim_key("Claim A."))
    assert rec.origin_id == "src-1"       # not clobbered by the later tag
    assert rec.edge_kind == "curated"
    assert rec.trust_class == "credible"
    assert rec.negated is True


def test_item_answer_key_never_enters_the_record(monkeypatch, tmp_path):
    """Changing the item's gold/answer field leaves the record fingerprint alone."""
    _arms(monkeypatch, durable=True)
    base = {"claim": "Claim A.", "source_status": "credible",
            "source_id": "src-1", "expected_answer": "W", "reliability": 0.9}
    ctx_a = context_from_wfa_item(dict(base))
    ctx_b = context_from_wfa_item({**base, "expected_answer": "N", "gold": "WITHDRAW"})
    # Source fields are identical; only the item's answer key differs.
    r1 = dp.record_decision(dp.DurableProvenanceStore(str(tmp_path / "a")),
                            dp.claim_key(base["claim"]), Decision.ENDORSE,
                            claim=base["claim"], source=ctx_a.source)
    r2 = dp.record_decision(dp.DurableProvenanceStore(str(tmp_path / "b")),
                            dp.claim_key(base["claim"]), Decision.ENDORSE,
                            claim=base["claim"], source=ctx_b.source)
    assert dp.claim_key(base["claim"]) == claim_act_key(base["claim"])
    assert r1.fingerprint() == r2.fingerprint()
    assert r1.origin_id == r2.origin_id == "src-1"
    assert r1.edge_kind == r2.edge_kind == "unknown"


def test_evaluator_is_byte_identical_with_flag_off_and_on(monkeypatch):
    """The gate itself is untouched: same context in -> same decision out."""
    ctx = _invalidated(prior=Decision.ENDORSE)
    off = evaluate_source("Claim A.", ctx)
    monkeypatch.setenv("RAVANA_DURABLE_PROVENANCE", "1")
    on = evaluate_source("Claim A.", ctx)
    assert off.as_dict() == on.as_dict()
    assert off.rule == "R_SOURCE_INVALIDATED"


def test_corrupt_sidecar_fails_open(monkeypatch, tmp_path):
    _arms(monkeypatch, durable=True)
    (tmp_path / dp.FILE_NAME).write_text("{not json", encoding="utf-8")
    eng = _Engine(tmp_path / "weights.pkl")
    eng.stage("Claim A.", _credible())
    assert eng.process_turn("q") == "free-form generated claim"
    assert eng._last_source_evaluation.decision is Decision.ENDORSE
    store = dp.store_for_engine(eng)
    assert store.ledger.keys() == [dp.claim_key("Claim A.")]
