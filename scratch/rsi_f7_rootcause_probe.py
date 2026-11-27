#!/usr/bin/env python3
"""BOOT-FREE four-way diagnostic for the RSI F7 flat result (M1+M2+M3).

Distinguishes, per recorded arm and per replayed query, which of these is the
blocker:

  1. UNREACHABLE_WIRING    hook never called / flag off / staging precondition
                           never satisfied (no act written, no ledger record).
  2. NO_SUBJECT_EVIDENCE   hook ran but no admitted provenance record for the
                           actual query subject from either channel.
  3. READER_DEFERS         evidence admitted, but the reader emits no pick
                           (label absent from options, negation veto, floor or
                           margin not met).
  4. ARTICULATION_NO_ANSWER hook emitted a pick, but the shipped grader saw an
                           unparseable / no_answer response.

Hard contract: no engine boot, no store write, no network, no gate/threshold/
item/prompt/scoring change, no production edit.  The ORACLE counterfactual in
section 5 relabels one scratch evidence record with a displayed option purely
to prove the reader is live; it is NON-DEPLOYABLE and is never used as a fix.

Usage (from the isolated worktree)::
  .venv/Scripts/python.exe scratch/rsi_f7_rootcause_probe.py
  .venv/Scripts/python.exe scratch/rsi_f7_rootcause_probe.py --all-bank
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(HERE)
for _p in (WT, os.path.join(WT, "scripts"), os.path.join(WT, "ravana", "src"),
           os.path.join(WT, "ravana_ml", "src"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Match the M1+M2+M3 arm's env contract exactly (boot-free).
os.environ["RAVANA_OFFLINE"] = "1"
os.environ["RAVANA_DURABLE_PROVENANCE"] = "1"
os.environ["RAVANA_SUBJECT_TYPE_EVIDENCE"] = "1"
os.environ["RAVANA_M3_GRADED_PROBE"] = "1"

import m123_preflight as pf            # noqa: E402  (unmodified staging primitives)
import m3_graded_probe as m3           # noqa: E402  (unmodified reader)
import measure_learn5_rsi as rsi       # noqa: E402  (unmodified harness)
from ravana.learn import subject_type_evidence as ste  # noqa: E402

REPORTS = os.path.join(HERE, "m123_out")
ARMS = ("OFF", "OFF2", "M1", "M1+M2", "M1+M2+M3", "SHAM")
TAGS = {"OFF": "off1", "OFF2": "off2", "M1": "m1", "M1+M2": "m12",
        "M1+M2+M3": "m123", "SHAM": "sham"}
CONDITIONS = ("near", "far", "foil", "negative")
SELECTED = {"rsi_000", "rsi_001", "rsi_002", "rsi_005"}

# The runner stages this exact prompt shape; no item metadata is read.
QUESTION_TMPL = ("Considering the examples above, what is true of %s? "
                 "Options: %s. Answer briefly.")


def query_surface(subject, candidates):
    return QUESTION_TMPL % (subject, ", ".join(candidates))


def arm_reachability(reports):
    """Read the recorded counters; never re-score."""
    out = {}
    for arm in ARMS:
        path = os.path.join(REPORTS, "rsi_%s.json" % TAGS[arm])
        if not os.path.isfile(path):
            out[arm] = {"missing": True}
            continue
        d = json.load(open(path, encoding="utf-8"))
        c = dict(d.get("counters", {}))
        out[arm] = {
            "status": d.get("status"),
            "hook_calls": c.get("hook_calls", 0),
            "flag_on": c.get("flag_on", 0),
            "flag_off": c.get("flag_off", 0),
            "production_calls": c.get("production_calls", 0),
            "scorer_calls": c.get("scorer_calls", 0),
            "production_pick_defers": c.get("deferred_to_production", 0),
            "staged_m1": c.get("staged_m1", 0),
            "staged_m2": c.get("staged_m2", 0),
            "m1_act_calls": c.get("m1_act_calls", 0),
            "m1_act_writes": c.get("m1_act_writes", 0),
            "records_scanned": c.get("records_scanned", 0),
            "records_admitted": c.get("records_admitted", 0),
            "records_rejected": c.get("records_rejected", 0),
            "veto_hits": c.get("veto_hits", 0),
            "picks": c.get("picks", 0),
            "emitted_picks": c.get("emitted_picks", 0),
            "hook_errors": c.get("hook_errors", 0),
            "last_error": c.get("last_error"),
            "engine_response_unparsed": c.get("engine_response_unparsed", 0),
            "query_turns": c.get("query_turns", 0),
            "endpoints_primary": {k: d["endpoints"]["primary"].get(k)
                                  for k in ("near", "far", "foil", "negative")},
        }
        # Reachability verdict from counters alone.
        a = out[arm]
        if a["hook_calls"] == 0 or (a["scorer_calls"] == 0 and a["picks"] == 0):
            a["reachability"] = "UNREACHABLE_WIRING"
        elif a["records_admitted"] == 0 and a["staged_m2"] == 0 and a["staged_m1"] == 0:
            a["reachability"] = "NO_SUBJECT_EVIDENCE"
        elif a["emitted_picks"] > 0:
            a["reachability"] = "ARTICULATION_NO_ANSWER" if a["engine_response_unparsed"] else "PICK_EMITTED"
        else:
            a["reachability"] = "READER_DEFERS"
    return out


def m1_precondition_audit(item):
    """Why stage_m1_act cannot fire on an RSI item."""
    surfaces = []
    rel = item["relation"]
    for subj, obj in item["demo_pairs"][:1]:
        phrase = rsi.RELATIONS.get(rel, {}).get("fwd", [rel.lower()])[0]
        surfaces.append("%s %s %s." % (subj, phrase, obj))
    q = item["near"]
    surfaces.append(query_surface(q["subject"], q["candidates"]))
    return {
        "item_id": item["item_id"],
        "has_channel_evidence": bool(item.get("channel_evidence")),
        "claim_re_matches": [bool(pf.CLAIM_RE.search(s)) for s in surfaces],
        "sample_surfaces": surfaces,
        # runner.stage_m1_act requires BOTH an allowed_fact whose predicate is
        # is_a/type_of/instance_of AND a CLAIM_RE match on the teaching surface.
        "can_stage_m1": bool(item.get("channel_evidence")) and any(
            pf.CLAIM_RE.search(s) for s in surfaces),
    }


def replay_query(item, condition, snapshot, all_items):
    """Re-stage exactly what the runner staged, then read with the M3 reader."""
    q = item[condition]
    subject, options = q["subject"], [str(c) for c in q["candidates"]]
    prompt = query_surface(subject, options)
    resolved = pf_subject_from_prompt(prompt)
    snap_ev, snap_rej = ste.evidence_for_subject(resolved, snapshot)
    counters = m3.new_counters()
    bundle = pf.stage_arm_evidence(
        resolved, options,
        ledger_records=[],          # M1 acts: none exist on RSI (audited below)
        graph_views=[],             # empty scoped graph views for boot-free replay
        snapshot_evidence=snap_ev,
        negated_pairs=set(),        # present() stages surfaces=[] on queries
        arm="M1+M2+M3", counters=counters)
    decision = m3.read_graded_probe(
        resolved, options, bundle, trust=m3._is_trusted_edge, counters=counters)
    pick = decision.pick
    labels = [str(getattr(ev, "type_name", "")) for ev in snap_ev]
    norm = m3._norm
    opp = {norm(o) for o in options}
    matched = sorted({lab for lab in labels if norm(lab) in opp})
    if not snap_ev:
        cat = "NO_SUBJECT_EVIDENCE"
    elif pick is not None:
        cat = "PICK_EMITTED"
    else:
        cat = "READER_DEFERS"
    return {
        "item_id": item["item_id"], "condition": condition,
        "resolved_subject": resolved, "raw_subject": subject,
        "displayed_options": options,
        "snapshot_type_labels": labels,
        "snapshot_rejections": [getattr(r, "reason", "") for r in snap_rej],
        "labels_matching_displayed_options": matched,
        "candidate_label_coverage": bool(matched),
        "admitted_records": len(decision.admitted),
        "vetoed": list(decision.vetoed),
        "scores": dict(decision.scores),
        "reader_reason": decision.reason,
        "pick": pick,
        "category": cat,
        "counter_delta": {k: counters.get(k, 0) - m3.new_counters().get(k, 0)
                          for k in ("records_admitted", "records_rejected",
                                    "veto_hits", "m1_scanned", "m2_scanned")},
    }


def pf_subject_from_prompt(prompt):
    """Same resolver as runner.subject_from_prompt (import without engine)."""
    m = re.search(
        r"\b(?:what is true of|which option best fits|what is true about|"
        r"best fits)\s+([^?.!,]+)", str(prompt or ""), re.IGNORECASE)
    if m:
        return m.group(1).strip()
    words = str(prompt or "").strip().split()
    return words[-1].strip(" .!?,") if words else ""


def candidate_label_census(items, snapshot):
    """Check whether ANY RSI displayed option is a type label its subject has."""
    total = hits = ev_present = 0
    misses = []
    for it in items:
        for cond in CONDITIONS:
            q = it[cond]
            ev, _ = ste.evidence_for_subject(q["subject"], snapshot)
            labels = {m3._norm(e.type_name) for e in ev}
            opts = {m3._norm(c) for c in q["candidates"]}
            total += 1
            ev_present += int(bool(ev))
            if labels & opts:
                hits += 1
            else:
                misses.append((it["item_id"], cond, q["subject"]))
    return {"rows": total, "rows_with_subject_evidence": ev_present,
            "rows_with_label_in_options": hits,
            "rows_without_label_in_options": len(misses),
            "first_misses": misses[:8]}


def oracle_relabel_control(item, condition, snapshot):
    """NON-DEPLOYABLE upper bound: prove reader emits when label is in options."""
    q = item[condition]
    subject, options = q["subject"], [str(c) for c in q["candidates"]]
    ev, _ = ste.evidence_for_subject(subject, snapshot)
    if not ev:
        return {"applicable": False, "reason": "no_subject_evidence"}
    counters = m3.new_counters()
    bundle = pf.stage_arm_evidence(
        subject, options, ledger_records=[], graph_views=[],
        snapshot_evidence=ev, negated_pairs=set(),
        arm="M1+M2+M3", counters=counters)
    # Replace the label on ONE scratch record with a displayed candidate.
    # This is not a fix: it injects the answer-side vocabulary to test wiring.
    rec = dict(bundle.m2_subject_evidence[0])
    rec["label"] = options[0]
    rec["stamp"] = options[0]
    # Oracle also supplies a stored-confidence value; boot-free replay has no
    # engine graph, so this is part of the NON-DEPLOYABLE upper bound.
    rec["confidence"] = 0.9
    swapped = m3.EvidenceBundle.from_lists(bundle.m1_records, [rec])
    d = m3.read_graded_probe(subject, options, swapped,
                             trust=m3._is_trusted_edge, counters=counters)
    return {"applicable": True, "NON_DEPLOYABLE": True,
            "injected_label": options[0], "pick": d.pick,
            "reason": d.reason, "scores": dict(d.scores)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-bank", action="store_true",
                    help="deprecated/no-op: the census always covers all items")
    args = ap.parse_args(argv)

    bank = rsi.build_items()
    snapshot = ste.load_snapshot()
    # Replay only the 4 items the recorded run actually executed; the label
    # census always covers the full deterministic bank.
    items = [it for it in bank if it["item_id"] in SELECTED]

    result = {
        "schema": "rsi-f7-rootcause-diagnostic-v1",
        "boot": False, "writes": False, "network": False,
        "production_files_changed": 0,
        "gate_threshold_item_prompt_scorer_changed": False,
        "arm_reachability": arm_reachability(REPORTS),
        "m1_precondition_audit": [m1_precondition_audit(it) for it in items[:4]],
        "replayed_queries": [],
        "candidate_label_census": candidate_label_census(bank, snapshot),
        "oracle_relabel_control": oracle_relabel_control(items[0], "near", snapshot),
    }
    for it in items:
        for cond in CONDITIONS:
            result["replayed_queries"].append(
                replay_query(it, cond, snapshot, bank))

    cats = collections.Counter(q["category"] for q in result["replayed_queries"])
    reasons = collections.Counter(q["reader_reason"] for q in result["replayed_queries"])
    result["replay_category_counts"] = dict(cats)
    result["replay_reader_reasons"] = dict(reasons)
    result["verdict"] = (
        "M3 hook reachable; M1 staging impossible on RSI; M2 type labels admitted "
        "but disjoint from the forced-choice object vocabulary -> reader defers "
        "with no_admitted_subject_evidence."
        if cats.get("READER_DEFERS", 0) == len(result["replayed_queries"])
        else "mixed; see per-query categories")

    out = os.path.join(HERE, "rsi_f7_rootcause_result.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, ensure_ascii=False)
    print(json.dumps({
        "out": out,
        "arm_reachability": {a: result["arm_reachability"][a].get("reachability")
                             for a in ARMS},
        "m1_can_stage": [a["can_stage_m1"] for a in result["m1_precondition_audit"]],
        "replay_category_counts": result["replay_category_counts"],
        "replay_reader_reasons": result["replay_reader_reasons"],
        "candidate_label_census": result["candidate_label_census"],
        "oracle_relabel_control": result["oracle_relabel_control"],
        "verdict": result["verdict"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
