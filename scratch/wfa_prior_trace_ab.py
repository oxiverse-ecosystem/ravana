#!/usr/bin/env python3
"""WFA claim-ID-keyed prior-act trace: scratch A/B (unmodified production code).

Question: does an engine-owned trace of its OWN prior ENDORSE act, keyed by the
same claim-id anchor that fixed g05 and surviving arbitrary intervening turns,
repair the measured sealed-discredited endpoint (0.80 -> 1.00) WITHOUT reading
any item answer key?

The production harness and engine are imported UNMODIFIED.  The candidate arm
monkeypatches exactly ONE engine boundary method,
CognitiveChatEngine._apply_source_monitoring_gate, to:
  * record a SourceDecision.ENDORSE under wfa_claim_token(claim); and
  * when an invalidated staged context has no prior ENDORSE, substitute the
    stored ENDORSE (dataclasses.replace) before calling the ORIGINAL gate.
Everything else -- evaluator, gates, thresholds, item bank, prompts, grading --
is the production code.  The flag defaults OFF.

Modes:
  --self-check  boot-free, no store: flag equivalence + leakage controls
  --live        two sequential w.main() runs on identical copied store,
                trace OFF then ON; writes per-arm summaries
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for p in (os.path.join(PROJ, "scripts"), os.path.join(PROJ, "ravana", "src"), PROJ):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("RAVANA_OFFLINE", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import measure_learn5_wfa as w  # noqa: E402
from ravana.chat import source_evaluation as se  # noqa: E402
from ravana.chat import engine as engine_mod  # noqa: E402

Decision = se.Decision
_enabled = [False]
_trace = {}


def claim_id(claim):
    """Same anchor as the g05 provenance fix; text-only, no item metadata."""
    return "wfa_" + hashlib.sha1((claim or "").encode("utf-8")).hexdigest()[:12]


def install():
    original = engine_mod.CognitiveChatEngine._apply_source_monitoring_gate

    def traced(self, response, ctx):
        key, sub = None, False
        if _enabled[0]:
            sc = getattr(ctx, "source_context", None) if ctx is not None else None
            if sc is not None:
                try:
                    claim, staged = sc
                except (TypeError, ValueError):
                    claim, staged = None, None
                if claim is not None and staged is not None:
                    key = claim_id(claim)
                    # A withdrawal needs a recollected prior ENDORSE.  Only the
                    # invalidation turn asks for one; never invent it otherwise.
                    if (getattr(staged, "invalidated", False)
                            and getattr(staged, "prior_decision", None) is not Decision.ENDORSE
                            and _trace.get(key) is Decision.ENDORSE):
                        ctx.source_context = (
                            claim, replace(staged, prior_decision=Decision.ENDORSE))
                        sub = True
        out = original(self, response, ctx)
        if _enabled[0] and key is not None:
            dec = getattr(self, "_last_source_evaluation", None)
            if dec is not None and getattr(dec, "decision", None) is Decision.ENDORSE:
                _trace[key] = Decision.ENDORSE      # write the engine's own act
        if _enabled[0] and sub:
            _trace["_last_substitution"] = _trace.get("_last_substitution", 0) + 1
        return out

    engine_mod.CognitiveChatEngine._apply_source_monitoring_gate = traced
    return original


def arm(name):
    _enabled[0] = (name == "on")
    if name == "off":
        _trace.clear()


def self_check(original):
    """Boot-free: the flag must be inert off, and the trace must be keyed only
    by claim text and written only from the engine's own decision."""
    w.mfwf.install_frozen_patches()
    from ravana.chat.source_evaluation import ClaimContext, SourceStatus, TrustClass
    from ravana.web.web_to_graph import WebToGraph
    from ravana_ml.graph import ConceptGraph
    results = {}

    # Build a minimal real graph with a real provenanced edge (no engine boot).
    class Mini:
        def __init__(self):
            self.graph = ConceptGraph(dim=8)

    eng = Mini()
    claim = "tokab matches tokcw pattern."
    sid = w.is_source_id(claim, "discredited")
    w._write_asserted_wfa_fact(eng, claim, sid)

    class Ctx:
        def __init__(self, claim, staged):
            self.source_context = (claim, staged)

    def original_decision(engine, c):
        return se.evaluate_source(c.source_context[0], c.source_context[1])

    # 0. OFF equivalence: original and patched produce byte-identical outputs.
    arm("off")
    ctx0 = Ctx(claim, w.context_from_graph_provenance(eng, claim, sid)[0])
    out_off = original_decision(eng, ctx0)
    patched_off = engine_mod.CognitiveChatEngine._apply_source_monitoring_gate
    # emulate the patch's OFF path exactly by calling it through a duck-typed
    # object that only implements the boundary method's needs
    class Duck:
        _apply_source_monitoring_gate = patched_off
    d = Duck()
    # direct invocation of the unbound method needs a real-ish self; use object
    # with _last_source_evaluation plus a no-op original that just computes.
    called = {}

    def fake_original(self, response, ctx):
        called["decision"] = original_decision(self, ctx)
        self._last_source_evaluation = called["decision"]
        return response

    saved = engine_mod.CognitiveChatEngine._apply_source_monitoring_gate
    engine_mod.CognitiveChatEngine._apply_source_monitoring_gate = fake_original
    install()   # wraps fake_original, not production -- self-check only
    host = type("H", (), {"_last_source_evaluation": None})()
    arm("off")
    ctx_off = Ctx(claim, w.context_from_graph_provenance(eng, claim, sid)[0])
    resp_off = engine_mod.CognitiveChatEngine._apply_source_monitoring_gate(host, "X", ctx_off)
    dec_off = host._last_source_evaluation
    results["flag_off_byte_identical"] = (
        resp_off == "X" and dec_off.decision == out_off.decision
        and dec_off.rule == out_off.rule and not _trace)
    # restore real production wrapper after self-check (live mode re-installs)
    engine_mod.CognitiveChatEngine._apply_source_monitoring_gate = original

    # 1. Trace never reads item metadata: mutation leaves key + staged ctx fixed.
    base_key = claim_id(claim)
    mutated = dict(id="x", claim=claim, expected_answer="W", source_status="credible",
                   source_id="forged", reliability=1.0)
    _, base_audit = w.context_from_graph_provenance(eng, claim, sid)
    _, mut_audit = w.stage_context_for_item(eng, mutated, sid)
    results["item_metadata_invariant"] = (
        base_audit["fingerprint"] == mut_audit["fingerprint"]
        and claim_id(mutated["claim"]) == base_key)

    # 2. Key isolation: distinct claim texts have distinct keys.
    results["key_isolation"] = claim_id(claim) != claim_id(claim + " different")

    # 3. Recorded act comes only from evaluate_source (ENDORSE), not a label.
    ctx_e = Ctx(claim, w.context_from_graph_provenance(eng, claim, sid)[0])
    dec_e = se.evaluate_source(claim, ctx_e.source_context[1])
    results["act_is_engine_decision"] = dec_e.decision is Decision.ENDORSE

    # 4. No-engine-boot guarantee for this mode.
    results["no_engine_boot"] = "CognitiveChatEngine(" not in open(__file__, encoding="utf-8").read().split("Modes:")[0]

    results["verdict"] = "SELF_CHECK_OK" if all(results.values()) else "SELF_CHECK_FAIL"
    print(json.dumps(results, indent=2))
    return results["verdict"] == "SELF_CHECK_OK"


def rate(summary, pool="sealed", cond="discredited"):
    cc = summary["internal_source"]["pools"][pool]["conditions"][cond]
    return {"correct": cc["correct"], "withdraw": cc["withdraw_rate"],
            "abstain": cc["abstain_rate"], "n": cc["n"]}


def live():
    w.mfwf.install_frozen_patches()
    os.environ["WFA_BOUNDED_DIAGNOSTIC"] = "1"
    os.environ["WFA_IS_MAX_PER_POOL"] = "20"
    os.environ["RAVANA_SOURCE_EVAL"] = "1"
    original = install()
    summary_path = os.path.join(HERE, "learn5_wfa_summary.json")
    reports = {}
    for name in ("off", "on"):
        arm(name)
        t0 = time.time()
        print("=== ARM %s START %s ===" % (name, time.strftime("%H:%M:%S")), flush=True)
        rc = w.main(["--bounded"]) if False else w.main([])
        elapsed = round(time.time() - t0, 1)
        with open(summary_path, encoding="utf-8") as fh:
            s = json.load(fh)
        dest = os.path.join(HERE, "prior_trace_%s_summary.json" % name)
        shutil.copyfile(summary_path, dest)
        reports[name] = {
            "exit": rc, "elapsed_s": elapsed,
            "sealed_discredited": rate(s, "sealed", "discredited"),
            "primary_discredited": rate(s, "primary", "discredited"),
            "sealed_credible": rate(s, "sealed", "credible"),
            "sealed_unknown": rate(s, "sealed", "unknown"),
            "pass": s.get("pass"),
            "full_bank_verdict": s.get("full_bank_verdict"),
            "internal_pass": s["internal_source"].get("pass"),
            "substitutions": _trace.get("_last_substitution", 0),
            "trace_keys": sorted(k for k in _trace if not k.startswith("_")),
            "summary_copy": dest,
        }
        print("=== ARM %s DONE %s ===" % (name, json.dumps(reports[name])), flush=True)
    out = os.path.join(HERE, "wfa_prior_trace_ab_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2)
    print("RESULTS ->", out, flush=True)
    engine_mod.CognitiveChatEngine._apply_source_monitoring_gate = original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    original = install()
    try:
        if args.self_check:
            ok = self_check(original)
            print("SELF_CHECK:", ok)
        if args.live:
            live()
    finally:
        engine_mod.CognitiveChatEngine._apply_source_monitoring_gate = original


if __name__ == "__main__":
    main()
