#!/usr/bin/env python3
"""WFA E3 -- prior-act re-staging, SCRATCH harness (boot-free check + live driver).

NO production file is edited. Every gate, item, threshold, prompt, expected
decision and grading rule is imported UNMODIFIED from scripts/measure_learn5_wfa
(the production harness). The ONLY injected code is this module's
``e3_run_internal_source_condition``, which is a copy of the production
condition runner with one explicit manipulation:

    at the FIRST (prior-commitment) turn, capture the exact object
    ``engine._last_source_evaluation.decision`` and forward THAT object as
    ``prior_decision`` into the SECOND turn's
    ``context_from_graph_provenance`` call. Nothing is recomputed from item
    fields; the runner never receives an item at all.

``WFA_E3_PRIOR_RESTAGE=0`` disables the capture for a baseline A/B (default 1).
Additive audit rows go to scratch/wfa_e3_staging_audit.jsonl; the production
summary and gates are untouched.

Run modes:
    uv run python scratch/measure_learn5_wfa_e3.py --self-check   # boot-free
    uv run python scratch/measure_learn5_wfa_e3.py                # live (parent)
"""
import hashlib
import inspect
import json
import os
import pickle
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for _p in (os.path.join(PROJ, "scripts"), os.path.join(PROJ, "ravana", "src"),
           PROJ, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import measure_learn5_wfa as w  # noqa: E402  -- UNMODIFIED production harness

AUDIT_PATH = os.path.join(HERE, "wfa_e3_staging_audit.jsonl")
FLAG = "WFA_E3_PRIOR_RESTAGE"
WRITE_AUDIT = True
STAGING_SOURCE = "engine._last_source_evaluation.decision"


def _pickle_sha(obj):
    """Byte-level fingerprint of the forwarded object (no repr/copy)."""
    return hashlib.sha256(pickle.dumps(obj, protocol=4)).hexdigest()


def _append_audit(obs):
    if not WRITE_AUDIT:
        return
    with open(AUDIT_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obs, ensure_ascii=False, default=str) + "\n")


def e3_run_internal_source_condition(engine, claim, condition, source_id,
                                     teach_fn=None, invalidate_fn=None,
                                     prior_decision=None):
    """E3 copy of the production condition runner; no item object in scope.

    Everything outside the E3 block is byte-for-byte the production runner,
    so gates, prompts, grading and the emitted decision audit are unchanged.
    """
    from ravana.chat.source_evaluation import Decision  # lazy, pure module
    if condition not in w.IS_EXPECTED_TOKEN:
        raise ValueError("unknown internal-source condition: %r" % condition)
    enabled = os.environ.get(FLAG, "1") == "1"
    obs = {
        "condition": condition, "claim": claim, "source_id": source_id,
        "decision": None, "token": None, "text": "", "missing": False,
        "teached": 0, "invalidated": 0, "prior_decision": None,
        "prompt": w.source_question_prompt(claim),
        # ---- additive E3 staging audit (read by no gate) ----
        "e3_enabled": enabled,
        "e3_prior_object_source": None,
        "e3_prior_object_id": None,
        "e3_prior_pickle_sha256": None,
        "e3_prior_value": None,
        "e3_prior_is_same_object_at_stage": None,
        "e3_item_fields_read": False,
    }
    if engine is None:
        obs["missing"] = True
        return obs
    reset = getattr(engine, "reset_episodic_state", None)
    if callable(reset):
        try:
            reset()
        except Exception:
            pass

    prior = prior_decision if not enabled else None
    if condition == "unknown":
        obs["teached"] = 0
    else:
        teach = teach_fn or (lambda e, c, s: w.teach_claim_into_graph(e, c, s))
        obs["teached"] = teach(engine, claim, source_id)
        if condition == "discredited":
            # FIRST decision turn: commit while the source is still credible.
            ctx0, _ = w.context_from_graph_provenance(
                engine, claim, source_id=source_id)
            engine.set_source_context_for_turn(claim, ctx0)
            w._is_process_turn(engine, obs["prompt"])
            first = getattr(engine, "_last_source_evaluation", None)
            if enabled and first is not None:
                # E3: forward the exact object -- no `.value` lookup, no
                # `Decision.ENDORSE` constant, no item metadata anywhere.
                prior = first.decision
                obs["e3_prior_object_source"] = STAGING_SOURCE
                obs["e3_prior_object_id"] = id(prior)
                obs["e3_prior_pickle_sha256"] = _pickle_sha(prior)
                obs["e3_prior_value"] = getattr(prior, "value", repr(prior))
            elif enabled:
                obs["e3_prior_object_source"] = "MISSING_ENGINE_AUDIT"
            obs["prior_decision"] = (
                getattr(prior, "value", None) if prior is not None else None)
            invalidate = invalidate_fn or (
                lambda e, s: w.apply_source_invalidation(e, s))
            obs["invalidated"] = invalidate(engine, source_id)

    # SECOND decision turn: stage with the captured object itself.
    ctx, audit = w.context_from_graph_provenance(
        engine, claim, source_id=source_id, prior_decision=prior)
    if enabled:
        obs["e3_prior_is_same_object_at_stage"] = (
            ctx.prior_decision is prior)
    obs.update({"staged_" + k: v for k, v in audit.items()})
    engine.set_source_context_for_turn(claim, ctx)
    text, err = w._is_process_turn(engine, obs["prompt"])
    obs["text"] = text
    obs["error"] = err
    decision = getattr(engine, "_last_source_evaluation", None)
    obs["decision"] = decision.decision.value if decision is not None else None
    obs["token"] = decision.token if decision is not None else None
    obs["rule"] = decision.rule if decision is not None else None
    obs["gate_ok"] = bool(
        decision is not None and
        (decision.decision is Decision.ENDORSE
         or text.strip() == decision.token))
    obs["prompt_label_free"] = w.prompt_is_label_free(obs["prompt"], source_id)
    _append_audit(obs)
    return obs


def install():
    """Patch the module global used by w.main()'s internal-source arm."""
    w.run_internal_source_condition = e3_run_internal_source_condition


# ── Boot-free unit-style checks ───────────────────────────────────────────────

class _SentinelPrior:
    """Unique, picklable stand-in: a copied/recomputed object cannot match."""

    def __init__(self, value="ENDORSE"):
        self.value = value
        self.payload = os.urandom(16)


SENTINEL = _SentinelPrior()


class _ExplodingItem(dict):
    """Raises on any access except the one field the runner may use."""

    def __getitem__(self, key):
        if key != "claim":
            raise AssertionError("item field read during staging: %r" % key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key != "claim":
            raise AssertionError("item field read during staging: %r" % key)
        return super().get(key, default)


class _FakeEngine:
    """In-memory graph holder + turn recorder. No engine boot, no store."""

    def __init__(self, graph, first_prior):
        self.graph = graph
        self._last_source_evaluation = None
        self.staged = []
        self.turns = []
        self.resets = 0
        self.first_prior = first_prior

    def reset_episodic_state(self):
        self.resets += 1

    def set_source_context_for_turn(self, claim, context=None):
        self.staged.append((claim, context))

    def process_turn(self, prompt):
        self.turns.append(prompt)
        if len(self.turns) == 1:
            # E3 unit check supplies a bare prior object; the gate-parity
            # check supplies a full audit record. Accept both shapes.
            self._last_source_evaluation = (
                self.first_prior if hasattr(self.first_prior, "decision")
                else types.SimpleNamespace(decision=self.first_prior))
            return "claim text"
        from ravana.chat.source_evaluation import Decision
        self._last_source_evaluation = types.SimpleNamespace(
            decision=Decision.WITHDRAW, token="W",
            rule="R_SOURCE_INVALIDATED")
        return "W"


def _graph_holder():
    from ravana_ml.graph import ConceptGraph
    return types.SimpleNamespace(graph=ConceptGraph(dim=8))


def _self_check():
    from ravana.chat.source_evaluation import Decision, evaluate_source
    global WRITE_AUDIT
    prev_write, prev_flag = WRITE_AUDIT, os.environ.get(FLAG)
    WRITE_AUDIT = False  # self-check must not write live audit rows
    os.environ[FLAG] = "1"
    checks = {}

    # Tripwire: no engine module (hence no CognitiveChatEngine) may be loaded.
    engine_module_pre = "ravana.chat.engine" in sys.modules
    try:
        items = {it["id"]: it for it in w.build_items()}
        item = items["gg039"]
        claim, sid = item["claim"], w.is_source_id(item["claim"], "discredited")

        # A. identity + byte-identity of the object forwarded at turn 2.
        holder = _graph_holder()
        w._write_asserted_wfa_fact(holder, claim, sid)
        eng = _FakeEngine(holder.graph, SENTINEL)
        obs = e3_run_internal_source_condition(eng, claim, "discredited", sid)
        staged0, staged1 = eng.staged[0][1], eng.staged[1][1]
        checks["A_object_from_engine_audit"] = (
            obs["e3_prior_object_source"] == STAGING_SOURCE and
            obs["e3_prior_value"] == "ENDORSE")
        checks["A_byte_identical_forward"] = (
            obs["e3_prior_object_id"] == id(SENTINEL) and
            obs["e3_prior_pickle_sha256"] == _pickle_sha(SENTINEL) and
            obs["e3_prior_is_same_object_at_stage"] is True and
            staged1.prior_decision is SENTINEL and
            pickle.dumps(staged1.prior_decision, protocol=4) ==
            pickle.dumps(SENTINEL, protocol=4))
        checks["A_first_turn_had_no_prior"] = staged0.prior_decision is None
        checks["A_invalidation_fired"] = obs["invalidated"] >= 1

        # B. no item field can feed the second-turn staging.
        params = inspect.signature(e3_run_internal_source_condition).parameters
        checks["B_no_item_parameter"] = (
            "item" not in params and "claim" in params and
            "condition" in params and "source_id" in params)
        exploding = _ExplodingItem(item)
        _, exploding_audit = w.stage_context_for_item(
            holder, exploding, sid, prior_decision=SENTINEL)
        _, direct_audit = w.context_from_graph_provenance(
            holder, claim, source_id=sid, prior_decision=SENTINEL)
        mutated = dict(item)
        mutated.update({"expected_answer": "X", "source_status": "credible",
                        "source_id": "forged", "reliability": 1.0,
                        "domain": "forged", "prior_decision": Decision.WITHDRAW})
        _, mutated_audit = w.stage_context_for_item(
            holder, mutated, sid, prior_decision=SENTINEL)
        checks["B_item_fields_do_not_change_staging"] = (
            exploding_audit["fingerprint"] == direct_audit["fingerprint"] ==
            mutated_audit["fingerprint"])

        # C. boot-free endpoint control: only the prior object flips N -> W.
        holder2 = _graph_holder()
        w._write_asserted_wfa_fact(holder2, claim, sid)
        w.apply_source_invalidation(holder2, sid)
        ctx_no, audit_no = w.context_from_graph_provenance(
            holder2, claim, source_id=sid, prior_decision=None)
        ctx_yes, audit_yes = w.context_from_graph_provenance(
            holder2, claim, source_id=sid, prior_decision=Decision.ENDORSE)
        d_no, d_yes = evaluate_source(claim, ctx_no), evaluate_source(claim, ctx_yes)
        checks["C_reproduces_ABSTAIN"] = (
            d_no.decision is Decision.ABSTAIN and
            d_no.rule == "R_INVALIDATION_NO_PRIOR")
        checks["C_flips_to_WITHDRAW"] = (
            d_yes.decision is Decision.WITHDRAW and
            d_yes.rule == "R_SOURCE_INVALIDATED")
        checks["C_only_prior_changed"] = (
            audit_no["fingerprint"] == audit_yes["fingerprint"] and
            audit_no["trust_class"] == audit_yes["trust_class"])

        # D. flag-off baseline: staged context carries no prior object.
        os.environ[FLAG] = "0"
        holder3 = _graph_holder()
        w._write_asserted_wfa_fact(holder3, claim, sid)
        eng3 = _FakeEngine(holder3.graph, SENTINEL)
        obs3 = e3_run_internal_source_condition(eng3, claim, "discredited", sid)
        checks["D_flag_off_baseline"] = (
            obs3["prior_decision"] is None and
            eng3.staged[1][1].prior_decision is None)
        checks["D_no_live_engine_boot"] = (
            engine_module_pre or "ravana.chat.engine" not in sys.modules)

        # E. gate preservation: with the SAME engine audit object, the E3 copy
        # emits byte-identical production observation fields for all conditions.
        os.environ[FLAG] = "1"

        def _teach_stub(e, c, s):
            return 1

        def _inv_stub(e, s):
            return 1

        for cond in w.IS_CONDITIONS:
            audit_obj = types.SimpleNamespace(decision=Decision.ENDORSE,
                                              token="Y",
                                              rule="R_EXPLICIT_CREDIBLE_SOURCE")
            ep = _FakeEngine(_graph_holder().graph, audit_obj)
            ee = _FakeEngine(_graph_holder().graph, audit_obj)
            prod = w.run_internal_source_condition(
                ep, claim, cond, sid, teach_fn=_teach_stub,
                invalidate_fn=_inv_stub)
            e3 = e3_run_internal_source_condition(
                ee, claim, cond, sid, teach_fn=_teach_stub,
                invalidate_fn=_inv_stub)
            prod_fields = {k: v for k, v in prod.items()}
            e3_fields = {k: v for k, v in e3.items()
                         if not k.startswith("e3_")}
            if prod_fields != e3_fields:
                checks["E_grading_fields_match_production"] = False
                break
        else:
            checks["E_grading_fields_match_production"] = True
    finally:
        if prev_flag is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = prev_flag
        WRITE_AUDIT = prev_write

    result = {
        "probe": "WFA E3 prior-act re-staging (boot-free unit check)",
        "no_engine_boot": True, "no_store_open": True,
        "no_production_edit": True,
        "checks": checks,
        "verdict": ("E3_SELF_CHECK_OK" if all(checks.values())
                    else "E3_SELF_CHECK_FAILED"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-check" in argv:
        return _self_check()
    install()
    print("WFA E3 scratch harness: %s=%s" %
          (FLAG, os.environ.get(FLAG, "1")))
    print("E3 staging audit: %s" % AUDIT_PATH)
    return w.main(argv)


if __name__ == "__main__":
    sys.exit(main())
