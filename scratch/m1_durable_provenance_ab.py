"""M1 boot-free A/B: unified durable-provenance adapter.

No live engine is booted. The script borrows the REAL ``process_turn`` and the
REAL emission gate from ``CognitiveChatEngine`` onto a duck-typed early-return
engine, so every decision comes from the unmodified ``evaluate_source`` rules.

Arms (identical claims, identical staged contexts, fresh per-claim dirs):
  A: RAVANA_DURABLE_PROVENANCE unset -> no durable substrate (baseline)
  A0: flag explicitly "0"             -> flag-equivalence control
  B: RAVANA_DURABLE_PROVENANCE=1      -> durable substrate ON

Primary endpoint: WITHDRAW rate on an invalidated turn AFTER a cold reload
(fresh engine object, empty in-memory state, same data dir) of a claim whose
credible turn was previously ENDORSEd.

Negative controls: no-false-withdrawal, key isolation, item-answer invariance,
unchanged evaluator, and a substitution counter (must fire only in arm B).

Output: scratch/m1_durable_provenance_ab.json (deterministic, no commit SHA in
the hashed payload) and a per-row stdout audit.
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

from ravana.chat import durable_provenance as dp  # noqa: E402
from ravana.chat.engine import CognitiveChatEngine  # noqa: E402
from ravana.chat.source_evaluation import (  # noqa: E402
    ClaimContext, Decision, SourceStatus, TrustClass, claim_act_key,
    context_from_wfa_item, evaluate_source)


class _Engine:
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
        return "free-form generated claim"


def credible(origin="src-1"):
    return ClaimContext(source=SourceStatus(
        origin_id=origin, source_type="curated", reliability=0.9,
        trust_class=TrustClass.CREDIBLE, credibility_token="credible"))


def invalidated(origin="src-1", prior=None):
    return ClaimContext(source=SourceStatus(
        origin_id=origin, source_type="curated", trust_class=TrustClass.DISCREDITED),
        invalidated=True, prior_decision=prior)


CLAIMS = [
    "Daytime happens because Earth rotates.",
    "Rain follows a drop in pressure.",
    "The mirrors cause a glow.",
    "Cooling produces condensation.",
    "A gale can precede a rainbow.",
    "Thaw is a kind of melt.",
    "Salt dissolves in water.",
    "A seed grows into a tree.",
    "The moon causes tides.",
    "Wood floats on water.",
    "Iron rusts in rain.",
    "A shadow follows an object.",
]

ENV_BACKUP = {}
ENV_KEYS = ("RAVANA_SOURCE_EVAL", "RAVANA_PRIOR_ACT_TRACE",
            "RAVANA_DURABLE_PROVENANCE")


def set_env(durable_mode):
    for k in ENV_KEYS:
        ENV_BACKUP[k] = os.environ.get(k)
    os.environ["RAVANA_SOURCE_EVAL"] = "1"
    os.environ.pop("RAVANA_PRIOR_ACT_TRACE", None)
    if durable_mode == "on":
        os.environ["RAVANA_DURABLE_PROVENANCE"] = "1"
    elif durable_mode == "zero":
        os.environ["RAVANA_DURABLE_PROVENANCE"] = "0"
    else:
        os.environ.pop("RAVANA_DURABLE_PROVENANCE", None)


def restore_env():
    for k, v in ENV_BACKUP.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def substitution_counter():
    orig = dp.DurableProvenanceLedger.apply_to_context
    counter = {"n": 0}

    def wrapped(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        if result:
            counter["n"] += 1
        return result

    dp.DurableProvenanceLedger.apply_to_context = wrapped
    return orig, counter


def run_primary_arm(mode, root):
    set_env(mode)
    orig, counter = substitution_counter()
    rows = []
    try:
        for i, claim in enumerate(CLAIMS):
            d = os.path.join(root, mode, "claim_%02d" % i)
            os.makedirs(d, exist_ok=True)
            save_path = os.path.join(d, "weights.pkl")
            first = _Engine(save_path)
            first.stage(claim, credible())
            first.process_turn("q")
            first_dec = (first._last_source_evaluation.decision.value
                         if first._last_source_evaluation else None)
            # Cold reload: fresh engine, no carried state, same data dir.
            cold = _Engine(save_path)
            assert "_durable_provenance_store" not in cold.__dict__
            cold.stage(claim, invalidated())
            out = cold.process_turn("q")
            dec = cold._last_source_evaluation
            rows.append({
                "claim": claim,
                "claim_key": dp.claim_key(claim),
                "first_decision": first_dec,
                "cold_out": out,
                "cold_decision": dec.decision.value if dec else None,
                "cold_rule": dec.rule if dec else None,
                "sidecar_exists": os.path.exists(os.path.join(d, dp.FILE_NAME)),
                "withdrawn": out == "W",
            })
    finally:
        dp.DurableProvenanceLedger.apply_to_context = orig
        restore_env()
    n = len(rows)
    return {
        "arm": mode,
        "n": n,
        "withdraw": sum(r["withdrawn"] for r in rows),
        "withdraw_rate": round(sum(r["withdrawn"] for r in rows) / n, 4),
        "substitutions": counter["n"],
        "rows": rows,
    }


def run_controls(root):
    out = {}
    # C1 flag equivalence: unset and explicit "0" produce identical rows.
    a = run_primary_arm("off", root)
    z = run_primary_arm("zero", root)
    out["flag_off_equals_explicit_zero"] = (
        [r["cold_out"] for r in a["rows"]] == [r["cold_out"] for r in z["rows"]]
        and a["substitutions"] == z["substitutions"] == 0)

    # C2 no false withdrawal without a prior act.
    set_env("on")
    d = os.path.join(root, "controls", "no_prior")
    os.makedirs(d, exist_ok=True)
    eng = _Engine(os.path.join(d, "weights.pkl"))
    eng.stage("Never taught.", invalidated())
    no_prior_out = eng.process_turn("q")
    out["no_false_withdrawal"] = (
        no_prior_out == "N" and eng._last_source_evaluation.rule
        == "R_INVALIDATION_NO_PRIOR")
    restore_env()

    # C3 key isolation: an endorsed claim A cannot repair claim B.
    set_env("on")
    d = os.path.join(root, "controls", "key_isolation")
    os.makedirs(d, exist_ok=True)
    eng = _Engine(os.path.join(d, "weights.pkl"))
    eng.stage(CLAIMS[0], credible())
    eng.process_turn("q")
    eng.stage("A completely different claim.", invalidated())
    iso_out = eng.process_turn("q")
    out["key_isolation"] = (
        iso_out == "N" and eng._last_source_evaluation.rule
        == "R_INVALIDATION_NO_PRIOR")
    restore_env()

    # C4 item answer key cannot change the durable record fingerprint.
    base = {"claim": "Claim A.", "source_status": "credible",
            "source_id": "src-1", "expected_answer": "W"}
    c1 = context_from_wfa_item(dict(base))
    c2 = context_from_wfa_item({**base, "expected_answer": "N", "gold": "WITHDRAW"})
    r1 = dp.record_decision(dp.DurableProvenanceStore(os.path.join(root, "controls", "i1")),
                            dp.claim_key(base["claim"]), Decision.ENDORSE,
                            claim=base["claim"], source=c1.source)
    r2 = dp.record_decision(dp.DurableProvenanceStore(os.path.join(root, "controls", "i2")),
                            dp.claim_key(base["claim"]), Decision.ENDORSE,
                            claim=base["claim"], source=c2.source)
    out["item_answer_key_invariant"] = r1.fingerprint() == r2.fingerprint()

    # C5 the evaluator itself is unchanged by the flag.
    ctx = invalidated(prior=Decision.ENDORSE)
    eval_off = evaluate_source("Claim A.", ctx).as_dict()
    os.environ["RAVANA_DURABLE_PROVENANCE"] = "1"
    eval_on = evaluate_source("Claim A.", ctx).as_dict()
    os.environ.pop("RAVANA_DURABLE_PROVENANCE", None)
    out["evaluator_unchanged"] = eval_off == eval_on

    # C6 the anchor equals the pre-existing provenance/act anchor.
    out["anchor_matches_existing"] = all(
        dp.claim_key(c) == claim_act_key(c) for c in CLAIMS)
    return out


def main():
    root = tempfile.mkdtemp(prefix="m1_durable_ab_")
    try:
        off = run_primary_arm("off", root)
        on = run_primary_arm("on", root)
        controls = run_controls(root)
        all_controls_ok = all(controls.values())
        result = {
            "instrument": "M1 boot-free durable-provenance A/B",
            "boot_free": True,
            "primary_endpoint": "WITHDRAW on invalidated turn after cold reload",
            "arms": {"off": {"n": off["n"], "withdraw": off["withdraw"],
                             "withdraw_rate": off["withdraw_rate"],
                             "substitutions": off["substitutions"]},
                     "on": {"n": on["n"], "withdraw": on["withdraw"],
                            "withdraw_rate": on["withdraw_rate"],
                            "substitutions": on["substitutions"]}},
            "controls": controls,
            "all_controls_ok": all_controls_ok,
            "positive_effect": on["withdraw"] - off["withdraw"],
            "rows": {"off": off["rows"], "on": on["rows"]},
        }
        payload = json.dumps(result, sort_keys=True, separators=(",", ":"))
        result["manifest_hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        out_path = os.path.join(HERE, "m1_durable_provenance_ab.json")
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print("=== ARM ROWS (off) ===")
        for r in off["rows"]:
            print("off", r["claim_key"], r["first_decision"], "->", r["cold_out"],
                  r["cold_rule"], "sidecar=" + str(r["sidecar_exists"]))
        print("=== ARM ROWS (on) ===")
        for r in on["rows"]:
            print("on ", r["claim_key"], r["first_decision"], "->", r["cold_out"],
                  r["cold_rule"], "sidecar=" + str(r["sidecar_exists"]))
        print("=== SUMMARY ===")
        print(json.dumps({"arms": result["arms"], "controls": controls,
                          "all_controls_ok": all_controls_ok,
                          "positive_effect": result["positive_effect"],
                          "manifest_hash": result["manifest_hash"]}, indent=2))
        ok = (all_controls_ok and off["withdraw"] == 0
              and on["withdraw"] == on["n"] and on["substitutions"] > 0)
        print("VERDICT:", "AB_OK" if ok else "AB_FAIL")
        return 0 if ok else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
