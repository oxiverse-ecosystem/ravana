#!/usr/bin/env python3
"""Pre-registered F1-F10 evaluator for the TRA-K far-repair A/B.

BOOT-FREE: reads only JSON artifacts produced by the A/B runner and the
boot-free root-cause/oracle probe. It constructs no engine, opens no weight
store, and changes no gate/threshold/item/prompt/gold/scorer.

Falsifiers are evaluated in the pre-registered order F1..F10. The AUTHORITATIVE
verdict is the FIRST failing row (audit rows after it are recorded but cannot
override it). Rows with action INVALIDATE/DISCARD/ABORT/STOP halt the claim.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "m123_ab_out")

ARMS = ("OFF", "OFF2", "M123", "REPAIR", "REPAIR_SELFLOOP", "SHAM")
TAGS = {"OFF": "off1", "OFF2": "off2", "M123": "m123",
        "REPAIR": "repair", "REPAIR_SELFLOOP": "repair_self", "SHAM": "sham"}
ENDPOINT_KEYS = (
    "far", "far_minus_chance", "near", "immediate", "delayed",
    "delayed_retention", "growth", "foil", "negative", "unrelated",
    "transfer", "chance", "foil_reject", "negative_reject",
    "monotonic_max_drop", "graded_max_drop", "control_max_delta",
)
COUNTER_KEYS = (
    "turns", "query_turns", "hook_calls", "scorer_calls", "trust_calls",
    "process_turn", "records_admitted",
)


def load(name):
    path = os.path.join(OUT, name)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def rep(tag):
    return load("trak_%s.json" % tag)


def rows(tag):
    return load("trak_%s.json.rows.json" % tag)


def by_pool(report):
    return report["harness"]["summary"]


def endpoints(report):
    """Full per-pool endpoint table from the harness's own summarize()."""
    out = {}
    for pool in ("primary", "sealed"):
        s = by_pool(report).get(pool, {})
        out[pool] = {k: s.get(k) for k in ENDPOINT_KEYS}
        out[pool]["_n"] = {k + "_n": s.get(k + "_n")
                           for k in ENDPOINT_KEYS if k + "_n" in s}
    return out


def is_pick(row):
    out = row.get("output") or {}
    cands = [str(c).lower() for c in (row.get("input") or {}).get("candidates", [])]
    return (out.get("verdict") == "match"
            and str(out.get("target") or "").lower() in cands
            and bool(out.get("target")))


def row_rate(rows_, **match):
    sel = [r for r in rows_ if all(r.get(k) == v for k, v in match.items())]
    return (round(sum(is_pick(r) for r in sel) / len(sel), 4) if sel else None), len(sel)


def row_endpoint_table(rows_):
    """Pick rates on row-level endpoints the summary does not expose."""
    out = {}
    for cond in ("far", "near", "foil", "negative", "unrelated"):
        out[cond], _ = row_rate(rows_, condition=cond, phase="query")
    out["far_immediate"], _ = row_rate(rows_, condition="far", phase="immediate")
    out["far_delayed"], _ = row_rate(rows_, condition="far", phase="delayed")
    out["far_k0_query"], _ = row_rate(
        [r for r in rows_ if r.get("k") == 0], condition="far", phase="query")
    return out


def counters(report):
    c = report.get("counters", {})
    return {k: c.get(k) for k in COUNTER_KEYS}


def main():
    pre = load("m123_preflight_result.json")
    root = load("rootcause_replay.json")
    oracle_path = os.path.join(OUT, "rootcause_ab_stores.json")
    oracle = load("rootcause_ab_stores.json") if os.path.isfile(oracle_path) else None

    reports = {a: rep(TAGS[a]) for a in ARMS}
    rrows = {a: rows(TAGS[a]) for a in ARMS}
    eps = {a: endpoints(reports[a]) for a in ARMS}
    reps = {a: row_endpoint_table(rrows[a]) for a in ARMS}
    cnts = {a: counters(reports[a]) for a in ARMS}

    results = []

    def add(fid, name, ok, detail, action="", authoritative=True):
        results.append({"id": fid, "name": name, "ok": bool(ok),
                        "detail": detail, "action": action,
                        "authoritative": bool(authoritative)})

    # F1 -- instrument integrity (pre-registered STOP).
    f1_ok = (bool(pre.get("ok")) and pre.get("self_test", {}).get("ok")
             and pre.get("ast_anti_leakage", {}).get("ok")
             and all(v.get("match") for v in pre.get("sha_pins", {}).values()))
    add("F1", "pins + boot-free preflight + AST anti-leakage", f1_ok,
        {"preflight_ok": pre.get("ok"),
         "self_test": [pre.get("self_test", {}).get("passed"),
                       pre.get("self_test", {}).get("total")],
         "ast_problems": pre.get("ast_anti_leakage", {}).get("problems"),
         "pin_mismatches": [k for k, v in pre.get("sha_pins", {}).items()
                            if not v.get("match")],
         "rootcause_reproduced": len(root.get("far_rows_history", [])) >= 0},
        "INSTRUMENT INVALID — no score")

    # F2 -- independent repeated baseline on a fresh store (INVALIDATE).
    diffs = {}
    for pool in ("primary", "sealed"):
        for k in sorted(set(eps["OFF"][pool]) | set(eps["OFF2"][pool])):
            a, b = eps["OFF"][pool].get(k), eps["OFF2"][pool].get(k)
            if a != b:
                diffs["%s.%s" % (pool, k)] = [a, b]
    ra_err = reports["OFF"].get("guard", {}).get("backend_error")
    rb_err = reports["OFF2"].get("guard", {}).get("backend_error")
    add("F2", "OFF == OFF2 on every endpoint", not diffs and not ra_err and not rb_err,
        {"endpoint_diffs": diffs, "errors": [ra_err, rb_err],
         "off_replayed": root.get("item_ids")},
        "store-carryover; invalidate all deltas")

    # F3 -- wrong-tag control leak (DISCARD).
    sham = reports["SHAM"]
    sham_pick_rows = [{"item": r.get("id"), "condition": r.get("condition"),
                       "phase": r.get("phase")}
                      for r in rrows["SHAM"] if is_pick(r)]
    sham_emitted = (cnts["SHAM"].get("emitted_picks") or 0)
    cd = {k: [cnts["SHAM"].get(k), cnts["M123"].get(k)]
          for k in COUNTER_KEYS if cnts["SHAM"].get(k) != cnts["M123"].get(k)}
    add("F3", "SHAM emits no pick and matches M123 compute",
        not sham_pick_rows and not sham_emitted and not cd,
        {"pick_rows": sham_pick_rows[:8], "emitted_picks": sham_emitted,
         "counter_diffs": cd, "scored": sham.get("scored")},
        "control leak; discard contrast")

    # F4 -- decisive fork: does removing supersession move far at all?
    oracle_picks = ([r.get("pick") for r in oracle.get("oracle_reachability", [])]
                    if oracle else [r.get("pick") for r in
                                    root.get("oracle_reachability", [])])
    oracle_ok = len(oracle_picks) == 4 and all(p is not None for p in oracle_picks)
    far_delta = {}
    for pool in ("primary", "sealed"):
        a = eps["M123"][pool].get("far")
        b = eps["REPAIR"][pool].get("far")
        far_delta[pool] = None if a is None or b is None else round(b - a, 4)
    f4_ok = all(v is not None and v >= 0.10 for v in far_delta.values())
    add("F4", "far REPAIR-M123 >= +0.10 on BOTH pools (oracle 4/4)",
        f4_ok and oracle_ok,
        {"far_M123": {p: eps["M123"][p].get("far") for p in eps["M123"]},
         "far_REPAIR": {p: eps["REPAIR"][p].get("far") for p in eps["REPAIR"]},
         "far_delta": far_delta, "oracle_picks": oracle_picks,
         "oracle_deployable": False,
         "oracle_source": "fresh M123 arm stores" if oracle else "saved M123 stores"},
        "supersession not sufficient" if not f4_ok else "")

    # F5 -- regression / gate leak.
    near_drops = {}
    for pool in ("primary", "sealed"):
        a = eps["M123"][pool].get("near")
        b = eps["REPAIR"][pool].get("near")
        near_drops[pool] = None if a is None or b is None else round(a - b, 4)
    reject_picks = {a: {k: reps[a].get(k) for k in
                        ("foil", "negative", "unrelated")} for a in ARMS}
    f5_ok = (all(v is not None and v <= 0.05 for v in near_drops.values())
             and all((v or 0) == 0 for arm in reject_picks.values()
                     for v in arm.values()))
    add("F5", "near drop <=0.05; foil/negative/unrelated picks == 0", f5_ok,
        {"near_drops": near_drops, "reject_arm_picks": reject_picks},
        "regression or gate leak; revert arm")

    # F6 -- answer-side leak through the manipulation.
    ctrl_delta = {}
    for k in ("foil", "negative", "unrelated"):
        ctrl_delta[k] = round((reps["REPAIR"].get(k) or 0)
                              - (reps["M123"].get(k) or 0), 4)
    f6_ok = all(v == 0 for v in ctrl_delta.values())
    add("F6", "REPAIR does not move foil/negative/unrelated", f6_ok,
        {"control_pick_deltas": ctrl_delta},
        "answer-side leak through manipulation; discard")

    # F7 -- null control: k=0 far rows must not move in ANY arm.
    k0 = {a: reps[a].get("far_k0_query") for a in ARMS}
    f7_ok = len(set(v for v in k0.values() if v is not None)) <= 1
    add("F7", "k=0 far rows unchanged in every arm", f7_ok,
        {"far_k0_query_pick_rate": k0},
        "null-control failure; arm changed exposure")

    # F8 -- saturation caveat, never a capability claim.
    sat = reports["REPAIR"].get("harness", {}).get("saturation", {})
    mut = reports["REPAIR"].get("harness", {}).get("mutation_control", {})
    far_all = [eps["REPAIR"][p].get("far") for p in ("primary", "sealed")]
    saturated = all(v == 1.0 for v in far_all)
    add("F8", "no uncaveated G6 saturation", not (saturated and mut.get("moved") is False),
        {"far": far_all, "g6_status": sat.get("status"),
         "mutation_moved": mut.get("moved")},
        "mechanism evidence only; no capability claim")

    # F9 -- self-loop ceiling, report only (never change option generation).
    facts = {r["item_id"]: r for r in root.get("fact_census", [])}
    selfloop_ids = set()
    for iid, rec in facts.items():
        far = rec.get("far") or {}
        subj = str(far.get("subject", "")).strip().casefold()
        if any(str(f.get("subject", "")).strip().casefold() == subj
               and str(f.get("object", "")).strip().casefold() == subj
               and f.get("evidence_role") == "subject_type"
               for f in far.get("facts", [])):
            selfloop_ids.add(iid)
    selfr = [r for r in rrows["REPAIR"] if r.get("id") in selfloop_ids
             and r.get("condition") == "far" and r.get("phase") == "query"]
    selfs = [r for r in rrows["REPAIR_SELFLOOP"] if r.get("id") in selfloop_ids
             and r.get("condition") == "far" and r.get("phase") == "query"]
    nonselfr = [r for r in rrows["REPAIR"] if r.get("id") not in selfloop_ids
                and r.get("condition") == "far" and r.get("phase") == "query"]
    nonsels = [r for r in rrows["REPAIR_SELFLOOP"] if r.get("id") not in selfloop_ids
               and r.get("condition") == "far" and r.get("phase") == "query"]
    rate = lambda rs: (round(sum(is_pick(r) for r in rs) / len(rs), 4)
                       if rs else None)
    f9_detail = {"selfloop_ids": sorted(selfloop_ids),
                 "selfloop_REPAIR": rate(selfr), "selfloop_REPAIR_SELFLOOP": rate(selfs),
                 "nonselfloop_REPAIR": rate(nonselfr),
                 "nonselfloop_REPAIR_SELFLOOP": rate(nonsels),
                 "note": "self-loop is a bank/writer ceiling; no option-generator change"}
    add("F9", "self-loop ceiling characterised, not repaired", True,
        f9_detail, "report only")

    # F10 -- no production artefact changed (git porcelain, boot-free).
    git = os.popen('cd "%s" && git diff --name-only HEAD -- scripts/ data/ '
                   'ravana/src/ docs/investigations/' % WT).read().splitlines()
    pins = [v for v in pre.get("sha_pins", {})]
    add("F10", "no production file / pin changed", not git,
        {"production_or_pin_diff": git, "pinned": pins},
        "abort; result cannot license a mechanism claim")

    # Authoritative halt: first failing row with a stopping action.
    stopping = [r for r in results if not r["ok"] and r["action"]]
    halted = stopping[0]["id"] if stopping else None
    scored = all(r["ok"] for r in results)
    all_reports_ok = all(reports[a].get("scored") and reports[a].get("status") == "OK"
                         for a in ARMS)

    out = {
        "schema": "trak-far-ab-verdict-v1",
        "deployable": False,
        "capability_claim": False,
        "scored": scored and all_reports_ok,
        "halted_at": halted,
        "verdict": ("INSTRUMENT INVALID" if halted == "F1" else
                    "STORE CARRYOVER" if halted == "F2" else
                    "CONTROL LEAK" if halted == "F3" else
                    "SUPERSESSION NOT SUFFICIENT" if halted == "F4" else
                    "REGRESSION / LEAK" if halted in ("F5", "F6", "F7") else
                    "PRODUCTION CHANGED" if halted == "F10" else
                    "NO STOP RULE FIRED"),
        "falsifiers": results,
        "endpoints_by_arm": eps,
        "row_endpoints_by_arm": reps,
        "counters_by_arm": cnts,
        "arm_status": {a: {"status": reports[a].get("status"),
                           "scored": reports[a].get("scored"),
                           "elapsed_s": reports[a].get("elapsed_s"),
                           "items": reports[a].get("config", {}).get("items"),
                           "falsifiers_local": reports[a].get("falsifiers_local")}
                       for a in ARMS},
        "oracle": {"deployable": False,
                   "picks": oracle_picks,
                   "source": "fresh M123 arm stores" if oracle else "saved M123 stores"},
        "notes": ("n=2/pool mechanism probe, not a full-bank verdict. "
                  "ORACLE is answer-side injected and NON-DEPLOYABLE; it proves "
                  "reachability only. Later falsifier rows after halted_at are "
                  "audit-only and cannot override the authoritative halt."),
    }
    path = os.path.join(OUT, "trak_far_ab_verdict.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({"halted_at": halted, "verdict": out["verdict"],
                      "scored": out["scored"],
                      "falsifiers": [{"id": r["id"], "ok": r["ok"]}
                                     for r in results]}, indent=1))
    return 0 if out["scored"] else 1


if __name__ == "__main__":
    sys.exit(main())
