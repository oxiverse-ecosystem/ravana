#!/usr/bin/env python3
"""Boot-free TRA-K far-transfer root-cause probe (read-only).

Replays the M3 reader on the SAVED M123 stores and on the frozen bank, with
no engine construction, no store write and no network. Imports the UNMODIFIED
production/staging primitives from scratch/m123_preflight.py and
scratch/m3_graded_probe.py (same code the M123 runner used).

Usage:
  TRAK_CHANNEL_JSON=<parent>/scratch/trak_channel_from_substrate.json \
  PYTHONHASHSEED=0 RAVANA_OFFLINE=1 python scratch/trak_far_rootcause_probe.py
"""
from __future__ import annotations

import collections
import json
import os
import pickle
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(HERE)
PARENT = "C:/Users/Likhith/Documents/projects/ravana"
RUN = "C:/Users/Likhith/Documents/projects/ravana_m123_run"
for p in (WT, os.path.join(WT, "scripts"), os.path.join(WT, "ravana", "src"),
          os.path.join(WT, "ravana_ml", "src"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import m123_preflight as pf
import m3_graded_probe as m3
import measure_tra_k as mtk
import tra_k_items as ti
from ravana.chat import durable_provenance as dp
from ravana.chat.schema_induction import typed_slot_views_from_graph
from ravana.learn import subject_type_evidence as ste

ARM = "M1+M2+M3"
STORE = os.environ.get("TRAK_FAR_STORE", os.path.join(RUN, "scratch", "m123_data", "trak_m123"))
REPORT = os.environ.get("TRAK_FAR_REPORT", os.path.join(PARENT, "scratch", "m123_out", "trak_m123.json"))
NEGATED_CLAIM_RE = re.compile(
    r"\b([A-Za-z][A-Za-z\-]*)\s+(?:is|are|was|were)\s+not\s+(?:a|an)\s+"
    r"([A-Za-z][A-Za-z \-]*?)\s*[.!?]", re.IGNORECASE)


def _negated_pairs(surfaces):
    out = set()
    for s in surfaces or ():
        m = NEGATED_CLAIM_RE.search(str(s))
        if m:
            out.add((m.group(1).strip().casefold(), m.group(2).strip().casefold()))
    return out


def load_store(item_id):
    d = os.path.join(STORE, item_id)
    p = os.path.join(d, "ravana_weights.pkl")
    with open(p, "rb") as fh:
        state = pickle.load(fh)
    graph = state.get("graph")
    views = [{k: getattr(v, k, "") for k in (
        "source", "relation_type", "target", "type_label", "edge_kind",
        "edge_source", "source_type", "confidence")}
        for v in typed_slot_views_from_graph(graph)]
    ledger = dp.DurableProvenanceStore(d).load().to_dict().get("records", {})
    return views, [dict(v) for v in ledger.values()]


def replay(item, condition, views, records, snapshot):
    q = mtk.query_for(item, condition)
    subject, options = str(q["subject"]), [str(c) for c in q["candidates"]]
    os.environ["RAVANA_SUBJECT_TYPE_EVIDENCE"] = "1"
    snap_ev, snap_rej = ste.evidence_for_subject(subject, snapshot)
    surfaces = mtk.teaching_surfaces(item, item["k"], condition, mtk.CHANNEL_ARM)
    counters = collections.Counter()
    bundle = pf.stage_arm_evidence(
        subject, options, ledger_records=records, graph_views=views,
        snapshot_evidence=snap_ev,
        negated_pairs=_negated_pairs(surfaces),
        arm=ARM, counters=counters)
    dec = m3.read_graded_probe(subject, options, bundle)
    staged = {
        "snapshot_labels": [str(getattr(e, "type_name", "")) for e in snap_ev],
        "snapshot_sources": [str(getattr(e, "source", "")) for e in snap_ev],
        "m1_labels": [str(r.get("label", "")) for r in bundle.m1_records],
        "m2_labels": [str(r.get("label", "")) for r in bundle.m2_subject_evidence],
        "m1_conf": [r.get("confidence") for r in bundle.m1_records],
        "m2_conf": [r.get("confidence") for r in bundle.m2_subject_evidence],
    }
    return subject, options, dec, staged, counters


def main():
    os.environ.setdefault("RAVANA_OFFLINE", "1")
    os.environ.setdefault("TRAK_CHANNEL_JSON", os.path.join(
        PARENT, "scratch", "trak_channel_from_substrate.json"))
    items, channel = pf.load_bank()
    rep = json.load(open(REPORT, encoding="utf-8"))
    ids = rep["config"]["items"]
    snapshot = ste.load_snapshot()
    out = {"channel": channel, "item_ids": ids, "rows": [], "bank_census": {},
           "edge_census": []}

    # Saved-store edge census: every typed view on each query subject, with
    # the confidence the reader's view_confidence() actually sees.
    for it in (i for i in items if i["item_id"] in ids):
        views, _ = load_store(it["item_id"])
        for cond in ("near", "far"):
            subj = str(mtk.query_for(it, cond)["subject"])
            scoped = [v for v in views
                      if str(v.get("source", "")).strip().casefold()
                      == subj.strip().casefold()]
            out["edge_census"].append({
                "item_id": it["item_id"], "condition": cond, "subject": subj,
                "views": [{"relation": v.get("relation_type"),
                           "label": v.get("type_label"),
                           "target": v.get("target"),
                           "confidence": v.get("confidence"),
                           "edge_kind": v.get("edge_kind"),
                           "edge_source": v.get("edge_source"),
                           "source_type": v.get("source_type")}
                          for v in scoped],
            })

    # Static replay on the four executed items, saved stores only.
    for it in (i for i in items if i["item_id"] in ids):
        views, records = load_store(it["item_id"])
        for cond in ("near", "far", "foil", "negative"):
            subject, options, dec, staged, counters = replay(
                it, cond, views, records, snapshot)
            q = mtk.query_for(it, cond)
            out["rows"].append({
                "item_id": it["item_id"], "pool": it["pool"], "k": it["k"],
                "condition": cond, "subject": subject, "options": options,
                "display_gold": q.get("gold"), "pick": dec.pick,
                "reason": dec.reason, "scores": dec.scores,
                "vetoed": list(dec.vetoed), "admitted_n": len(dec.admitted),
                "staged": staged, "counters": dict(counters),
            })

    # NON-DEPLOYABLE reachability control: same unmodified reader, same
    # options, same evidence bundle -- only the gold label's confidence is
    # raised (answer-side information injected into the scratch bundle).
    # This proves the hook/scorer/emission path works and the block is the
    # absent stored confidence, not a broken reader or a wrong threshold.
    out["oracle_reachability"] = []
    for row in (r for r in out["rows"] if r["condition"] == "far"):
        it = next(i for i in items if i["item_id"] == row["item_id"])
        views, records = load_store(row["item_id"])
        q = mtk.query_for(it, "far")
        gold = str(q["gold"])
        snap_ev, _ = ste.evidence_for_subject(row["subject"], snapshot)
        surfaces = mtk.teaching_surfaces(it, it["k"], "far", mtk.CHANNEL_ARM)
        bundle = pf.stage_arm_evidence(
            row["subject"], row["options"], ledger_records=records,
            graph_views=views, snapshot_evidence=snap_ev,
            negated_pairs=_negated_pairs(surfaces), arm=ARM,
            counters=collections.Counter())
        for rec in (bundle.m1_records + bundle.m2_subject_evidence):
            if str(rec.get("label", "")).strip().casefold() == gold.casefold():
                rec["confidence"] = 1.0
        dec = m3.read_graded_probe(row["subject"], row["options"], bundle)
        out["oracle_reachability"].append({
            "item_id": row["item_id"], "subject": row["subject"],
            "gold": gold, "pick": dec.pick, "reason": dec.reason,
            "scores": dec.scores, "deployable": False})

    # Full-bank static census: subject-side coverage and option overlap,
    # computed only from declared channel facts / displayed option strings.
    cov = collections.Counter()
    examples = collections.defaultdict(list)
    for it in items:
        for cond in ("near", "far"):
            q = mtk.query_for(it, cond)
            subj = str(q["subject"])
            ev, _ = ste.evidence_for_subject(subj, snapshot)
            labels = {str(getattr(e, "type_name", "")).strip().casefold()
                      for e in ev}
            opts = {str(o).strip().casefold() for o in q["candidates"]}
            gold = str(q.get("gold", "")).strip().casefold()
            cov[cond + "_rows"] += 1
            cov[cond + "_with_subject_evidence"] += bool(ev)
            cov[cond + "_with_label_in_options"] += bool(labels & opts)
            cov[cond + "_gold_in_options"] += bool(gold and gold in opts)
            cov[cond + "_gold_in_subject_labels"] += bool(gold and gold in labels)
            if not (labels & opts) and len(examples[cond + "_no_overlap"]) < 4:
                examples[cond + "_no_overlap"].append({
                    "item_id": it["item_id"], "subject": subj,
                    "labels": sorted(labels)[:6], "options": sorted(opts),
                    "gold": gold})
            pos = ti.admitted_subject_facts(it, subj, ti.SUBJECT_TYPE_ROLE)
            pos_label = str(pos[0].get("object", "")).strip() if pos else ""
            cov[cond + "_positive_selfloop"] += (
                bool(pos_label) and pos_label.casefold() == subj.casefold())
            cov[cond + "_label_count_sum"] += len(labels)
            cov[cond + "_positive_label_in_options"] += (
                bool(pos_label) and pos_label.casefold() in opts)
            mis = [f for f in it.get("channel_evidence", ())
                   if f.get("evidence_role") == ti.MISLEADING_ROLE
                   and str(f.get("subject", "")).strip().casefold()
                   == subj.casefold()]
            cov[cond + "_with_misleading_fact"] += bool(mis)
            if mis:
                cov[cond + "_misleading_label_in_options"] += any(
                    str(f.get("object", "")).strip().casefold() in opts
                    for f in mis)
    out["bank_census"] = {"counts": dict(cov),
                          "examples": {k: v for k, v in examples.items()}}

    # Per-item positive vs misleading channel facts for the four saved items.
    out["fact_census"] = []
    for it in (i for i in items if i["item_id"] in ids):
        rec = {"item_id": it["item_id"]}
        for cond in ("near", "far"):
            subj = str(mtk.query_for(it, cond)["subject"])
            rec[cond] = {"subject": subj, "options": mtk.query_for(it, cond)["candidates"],
                         "facts": [
                             {k: f.get(k) for k in (
                                 "subject", "object", "predicate", "evidence_role",
                                 "sense", "text_evidence_kind", "source",
                                 "source_url", "text_evidence")}
                             for f in it.get("channel_evidence", ())
                             if str(f.get("subject", "")).strip().casefold()
                             == subj.casefold()]}
        out["fact_census"].append(rec)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
