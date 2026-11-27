#!/usr/bin/env python3
"""TRA-K replay-consolidation MECHANISM PROBE.

STATIC / SCRATCH ONLY. This script never boots a CognitiveChatEngine, never
calls process_turn, never saves a store and never makes a network call. It reads
the TRA-K item bank and the ALREADY-RECORDED cold-reload stores from the focused
live run read-only (SQLite ``mode=ro``), imports the UNMODIFIED production
fail-closed scorer, and tests candidate offline replay overlays on top of the
engine's own persisted state.

WHAT IS BEING TESTED
--------------------
Can a provenance-preserving replay/consolidation overlay make typed ``is_a``
channel facts, which are present in the persisted store but not usable by the
fail-closed typed-slot reader, become usable at a held-out forced-choice probe
-- WITHOUT changing any trust gate, threshold, or answer key?

ARMS
----
  OFF        persisted cold-reload views, unchanged (production baseline)
  R1         replay restamp ONLY of stored, taught traces whose earned
             conversation stamp was lost (no new facts, no confidence change)
  R1_CONF    confidence-only sham on the same traces (gate must still refuse)
  R1_WRONG   same replay compute with a deliberately wrong tag (compute parity)
  R2         R1 + bounded, success-gated spaced reactivation of confidence
  ORACLE     NON-DEPLOYABLE upper bound: gold injected into the scorer only

Every deployable view created by R1/R2 must satisfy BOTH provenance conditions:
(a) its (subject,label) pair was actually taught in this item's channel surfaces
    (derived from admitted channel facts, never from type_sets/world_model/gold);
(b) a stored edge for that exact pair exists in THIS item's own DB. Replay
    restores an earned trace; it never fabricates a fact or reads the answer key.

Run:
  TRAK_REPLAY_SOURCE_ROOT=<path to main repo> python \
      scratch/trak_replay_consolidation_probe.py --out scratch/trak_replay_results.json
"""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import glob
import hashlib
import json
import os
import pickle
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCRIPTS = os.path.join(REPO, "scripts")
SRC = os.path.join(REPO, "ravana", "src")
for _p in (SCRIPTS, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The recorded run lives in the parent checkout; this worktree is a sibling.
SOURCE_ROOT = os.environ.get(
    "TRAK_REPLAY_SOURCE_ROOT", os.path.join(os.path.dirname(REPO), "ravana"))
CHANNEL_JSON = os.path.join(SOURCE_ROOT, "scratch", "trak_channel_from_substrate.json")
STORE_ROOT = os.path.join(SOURCE_ROOT, "scratch", "tra_k_harness", "live_data")

os.environ.setdefault("TRAK_CHANNEL_JSON", CHANNEL_JSON)

import tra_k_items as ti                      # noqa: E402  frozen item bank
import measure_tra_k as mtk                   # noqa: E402  pure harness helpers
from ravana.chat.schema_induction import (    # noqa: E402  UNMODIFIED production
    TypedSlotEdgeView, slot_keyed_forced_choice, TRUSTED_EDGE_KINDS,
    TRUSTED_SOURCE, CONGRUENCE_FLOOR, STRICT_MARGIN)

SLOT_WRITER = "tra_typed_slot"                # the production stamp being restored
# Conversation provenance is EARNED at teach time: the harness really asserted
# the admitted channel fact as a user turn. The channel's own Wikidata source
# URL is kept as an audit field; it is never used as the trust source.
EARNED = {"edge_kind": "curated", "edge_source": "conversation",
          "source_type": "user", "relation_type": "is_a",
          "slot_writer": SLOT_WRITER, "slot_label": "type"}
WRONG = {"edge_kind": "unknown", "edge_source": "system",
         "source_type": "system", "relation_type": "is_a",
         "slot_writer": SLOT_WRITER, "slot_label": "type"}


# ── read-only persisted-store census ───────────────────────────────────────
def _decode_metadata(raw):
    """The SQLite mirror stores source_metadata as a pickled-bytes repr."""
    if raw is None:
        return {}
    try:
        blob = ast.literal_eval(raw) if isinstance(raw, str) and raw[:2] in ("b'", 'b"') else raw
        obj = pickle.loads(bytes(blob)) if isinstance(blob, (bytes, bytearray)) else blob
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def census_item_store(db_path):
    """Return (slot_views, stored_pairs) from ONE item's persisted store.

    slot_views: production-shaped dicts for every edge with a typed-slot stamp.
    stored_pairs: (subject_label, target_label) -> list of raw edge records,
    including damaged/episodic traces. No engine, no write, no mutation.
    """
    import sqlite3
    con = sqlite3.connect("file:" + db_path.replace("\\", "/") + "?mode=ro", uri=True)
    try:
        cur = con.cursor()
        labels = {str(i): str(l) for i, l in cur.execute(
            "select id, label from concepts")}
        slot_views, stored = [], collections.defaultdict(list)
        for sid, tid, rel, conf, raw in cur.execute(
                "select source_id, target_id, relation_type, confidence, "
                "source_metadata from edges"):
            md = _decode_metadata(raw)
            src, tgt = labels.get(str(sid), ""), labels.get(str(tid), "")
            rec = {"subject": src, "target": tgt, "relation_type": str(rel or ""),
                   "confidence": float(conf or 0.0), "metadata": md}
            if src and tgt:
                stored[(src.strip().lower(), tgt.strip().lower())].append(rec)
            if md.get("slot_writer") != SLOT_WRITER:
                continue
            slot_views.append({
                "subject": src, "target": tgt,
                "relation_type": str(rel or "").strip().lower(),
                "type_label": str(md.get("type_label") or "").strip(),
                "slot_label": str(md.get("slot_label") or "").strip(),
                "edge_kind": str(md.get("edge_kind") or "").strip().lower(),
                "edge_source": str(md.get("source") or "").strip().lower(),
                "source_type": str(md.get("source_type") or "").strip().lower(),
                "confidence": float(conf or 0.0),
                "channel_source_id": str(md.get("channel_source_id") or ""),
            })
        return slot_views, stored
    finally:
        con.close()


def to_production_views(rows):
    return [TypedSlotEdgeView(
        source=str(v.get("subject") or ""), relation_type=str(v.get("relation_type") or ""),
        target=str(v.get("target") or ""), type_label=str(v.get("type_label") or ""),
        slot_label=str(v.get("slot_label") or ""), edge_kind=str(v.get("edge_kind") or ""),
        edge_source=str(v.get("edge_source") or ""), source_type=str(v.get("source_type") or ""),
        confidence=float(v.get("confidence") or 0.0)) for v in rows]


def is_eligible(v):
    """The production trust rule; we import its constants, never change them."""
    return (str(v.get("relation_type", "")).lower() == "is_a"
            and str(v.get("type_label", "")).strip()
            and str(v.get("type_label", "")).strip().lower()
            == str(v.get("target", "")).strip().lower()
            and str(v.get("edge_kind", "")).lower() in TRUSTED_EDGE_KINDS
            and str(v.get("edge_source", v.get("source", ""))).lower() == TRUSTED_SOURCE
            and str(v.get("source_type", "")).lower() == "user")


# ── teaching-side facts: derived from admitted channel surfaces only ───────
# The harness renders P2c surfaces as "<subject> is a(n) <type label>."
# The harness's own _TAUGHT_SLOT_RE stops at the FIRST content word, which
# truncates source-native multi-word Wikidata labels ("snow storm" -> "snow").
# This probe must parse the FULL object phrase or it would manufacture wrong
# labels and measure its own parser, not the replay mechanism.
_TAUGHT_FULL_RE = re.compile(
    r"^\s*([a-z][a-z\-]*)\s+is\s+(?:a|an)\s+(.+?)\s*\.\s*$", re.I)


def _condition_exclusions(item, condition):
    """Subjects whose channel fact was taught NEGATED in this condition.

    The harness renders P2c for `negative` through `channel_negation_surfaces`
    ("X is not a Y"). A replay journal must not restamp or strengthen a fact the
    teaching explicitly negated. This comes from the harness condition, never
    from the item's gold answer.
    """
    if condition == "negative":
        return {str(mtk.query_for(item, "negative")["subject"]).strip().lower()}
    return set()


def taught_pairs(item, condition="near"):
    """(subject, type_label) pairs the harness ACTUALLY asserted in P2c."""
    excluded = _condition_exclusions(item, condition)
    out = []
    for surface in mtk.channel_teaching_surfaces(item):
        m = _TAUGHT_FULL_RE.match(str(surface).strip())
        if m:
            out.append((m.group(1).strip().lower(),
                        " ".join(m.group(2).lower().split())))
    return sorted(p for p in set(out) if p[0] not in excluded)


def channel_source_for(item, label):
    """Source-native channel provenance, NOT the answer key, for audit only."""
    for fact in ti.ordered_channel_facts(item):
        if str(fact.get("object", "")).strip().lower() == str(label).strip().lower():
            return str(fact.get("source", ""))[:120]
    return ""


# ── replay overlays ────────────────────────────────────────────────────────
def replay_overlays(item, base_views, stored, tag=EARNED, restore_confidence=False,
                    spaced_cycles=0, condition="near", pairs_override=None,
                    boost_override=None, excluded_subjects=None):
    """Apply the candidate mechanism as an in-memory overlay.

    Restores ONLY traces that (a) were taught and (b) are already in this
    item's own store. Never invents a node, edge, label or provenance.
    `pairs_override` / `boost_override` let the recorded-run scope use the
    facts actually supplied to that run instead of the current item bank.
    """
    views = copy.deepcopy(base_views)
    pairs = taught_pairs(item, condition)
    if pairs_override is not None:
        pairs = sorted(set(pairs_override))
    if excluded_subjects:
        pairs = [p for p in pairs if p[0] not in excluded_subjects]
    key_index = {(v["subject"].strip().lower(), v["type_label"].strip().lower()): v
                 for v in views if str(v.get("type_label"))}
    restored, counts = [], collections.Counter()
    for subj, label in pairs:
        key = (subj, label)
        if key in key_index and is_eligible(key_index[key]):
            counts["intact"] += 1
            continue
        traces = stored.get(key, [])
        if not traces:
            counts["absent_no_stored_trace"] += 1
            continue
        # Pick the trace whose object label equals the taught label exactly;
        # fail closed if only a differently-labelled edge is present.
        exact = [t for t in traces if t["target"].strip().lower() == label]
        if not exact:
            counts["absent_no_exact_label"] += 1
            continue
        tr = max(exact, key=lambda t: t["confidence"])
        new = dict(tr["metadata"])  # start from the item's OWN stored record
        new.update(tag)
        new["type_label"] = label
        new["slot_writer"] = SLOT_WRITER
        new["slot_label"] = "type"
        new["channel_source_id"] = channel_source_for(item, label)
        conf = float(tr["confidence"])
        if restore_confidence:
            conf = 1.0                       # sham: threshold-free? no -- gate still binds
        v = {"subject": subj, "target": label,
             "relation_type": new["relation_type"], "type_label": label,
             "slot_label": "type", "edge_kind": new["edge_kind"],
             "edge_source": new["edge_source"], "source_type": new["source_type"],
             "confidence": conf, "channel_source_id": new["channel_source_id"]}
        views.append(v)
        restored.append(v)
        counts["restored"] += 1
    # Optional spaced / sleep-like reactivation: bounded and trace-scoped.
    # Slow-wave/SWR replay preferentially reinstates WEAK traces (Schapiro et
    # al. 2018), so the step applies to every taught trace that is actually in
    # this item's store, including intact edges whose confidence decayed below
    # the congruence floor. It never targets the gold label and never creates
    # a fact that was not taught and stored.
    trajectory = []
    if spaced_cycles:
        eta = 0.25                            # fixed step, pre-registered
        taught = set(boostable_pairs(item, condition)) if boost_override is None \
            else set(boost_override)
        if excluded_subjects:
            taught = {p for p in taught if p[0] not in excluded_subjects}
        for cycle in range(spaced_cycles):
            for v in views:
                if (str(v.get("subject", "")).strip().lower(),
                        str(v.get("type_label", "")).strip().lower()) not in taught:
                    continue
                if v["edge_kind"] not in TRUSTED_EDGE_KINDS:
                    continue
                v["confidence"] = min(
                    1.0, float(v["confidence"]) + eta * (1.0 - float(v["confidence"])))
            trajectory.append(round(
                sum(float(v["confidence"]) for v in views
                    if (str(v.get("subject", "")).strip().lower(),
                        str(v.get("type_label", "")).strip().lower()) in taught)
                / max(1, len(taught)), 4))
    return views, dict(counts), trajectory


# ── scoring ────────────────────────────────────────────────────────────────
def score(item, views, condition, probe=None):
    q = mtk.query_for(item, condition)
    if probe is not None:
        q = {"subject": probe["subject"], "candidates": probe["candidates"],
             "gold": probe["gold"]}
    prod = to_production_views(views)
    res = slot_keyed_forced_choice(
        prod, q["subject"], q["candidates"],
        congruence_floor=CONGRUENCE_FLOOR, strict_margin=STRICT_MARGIN)
    gold = q["gold"]
    # gold is used ONLY here, for scoring; it never enters the replay state.
    should_reject = condition in ("foil", "negative", "unrelated")
    if should_reject:
        correct = res["pick"] is None
    else:
        correct = res["pick"] is not None and str(res["pick"]).lower() == str(gold).lower()
    return {"condition": condition, "subject": q["subject"], "gold": gold,
            "pick": res["pick"], "reason": res["reason"],
            "eligible": res["eligible"], "correct": bool(correct)}


def instrument_controls():
    """Unmodified production scorer must enforce its own fail-closed contract."""
    good = TypedSlotEdgeView(source="thaw", relation_type="is_a", target="effect",
                             type_label="effect", edge_kind="curated",
                             edge_source="conversation", source_type="user", confidence=0.9)
    bad = TypedSlotEdgeView(**dict(good.__dict__, edge_kind="unknown",
                                   edge_source="web"))
    a = slot_keyed_forced_choice([good], "thaw", ["effect", "weather-event"])
    b = slot_keyed_forced_choice([bad], "thaw", ["effect", "weather-event"])
    c = slot_keyed_forced_choice([good], "thaw", ["apple", "banana"])
    d = slot_keyed_forced_choice([good], "unrelated_wumpus", ["effect", "weather-event"])
    return {"trusted_pick": a["pick"], "untrusted_pick": b["pick"],
            "disjoint_pick": c["pick"], "other_subject_pick": d["pick"],
            "ok": (a["pick"] == "effect" and b["pick"] is None
                   and c["pick"] is None and d["pick"] is None)}


def boostable_pairs(item, condition="near"):
    """Pairs eligible for replay strengthening: source-credible channel facts.

    The channel already separates admitted positive assertions from deliberately
    admitted LOW-CREDIBILITY conflicts (`evidence_role='misleading'`). Replay
    strengthens the former only. This is source provenance, not the answer key:
    the same role field is used by the frozen item bank to choose which fact is
    positive, and `channel_overlap()` independently asserts answer-key leak=0.
    """
    excluded = _condition_exclusions(item, condition)
    return sorted({(str(f.get("subject", "")).strip().lower(),
                    " ".join(str(f.get("object", "")).lower().split()))
                   for f in ti.ordered_channel_facts(item)
                   if f.get("evidence_role") != ti.MISLEADING_ROLE
                   and str(f.get("predicate", "")).lower() in ti.ISA_PREDICATES
                   and str(f.get("subject", "")).strip().lower() not in excluded})


def replay_fingerprint(views):
    payload = sorted((str(v.get("subject")), str(v.get("type_label")),
                      str(v.get("edge_kind")), round(float(v.get("confidence")), 6))
                     for v in views)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "trak_replay_results.json"))
    ap.add_argument("--spaced-cycles", type=int, default=3)
    args = ap.parse_args()

    base = ti.build_items()
    loaded = ti.load_channel_evidence(base)
    items = ti.build_items(channel_override=loaded["facts"])
    by_id = {it["item_id"]: it for it in items}
    overlap = ti.channel_overlap(items)
    controls = instrument_controls()

    # Recorded-run facts/probes: the focused live run predates later item-bank
    # edits, so the stores and row inputs are authoritative for THAT run. Gold
    # is used only for scoring; facts come from input.subject_type_facts.
    rows_path0 = os.path.join(os.path.dirname(STORE_ROOT), "tra_k_rows.jsonl")
    rec_facts, rec_probes = collections.defaultdict(list), collections.defaultdict(dict)
    if os.path.exists(rows_path0):
        with open(rows_path0, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                iid = row.get("id")
                inp = row.get("input", {})
                for f in inp.get("subject_type_facts", []):
                    pair = (str(f.get("subject", "")).strip().lower(),
                            str(f.get("type_label", "")).strip().lower())
                    if all(pair) and pair not in rec_facts[iid]:
                        rec_facts[iid].append(pair)
                if row.get("arm") == "channel" and row.get("phase") == "query":
                    rec_probes[iid][row["condition"]] = {
                        "subject": inp.get("query_subject"),
                        "candidates": inp.get("candidates", []),
                        "gold": row.get("gold", {}).get("target")}

    stores = sorted(glob.glob(os.path.join(STORE_ROOT, "tra_k_*", "ravana_weights.db")))
    result = {
        "probe": "trak_replay_consolidation", "static": True, "engine_boot": False,
        "source_root": SOURCE_ROOT, "channel": loaded["path"],
        "channel_external_items": loaded["external_items"],
        "channel_overlap_answer_key_leak": overlap.get("answer_key_leak"),
        "channel_overlap_ok": overlap.get("ok"),
        "instrument_controls": controls,
        "stores_censused": len(stores),
        "production_constants": {"trusted_kinds": sorted(TRUSTED_EDGE_KINDS),
                                 "trusted_source": TRUSTED_SOURCE,
                                 "floor": CONGRUENCE_FLOOR, "margin": STRICT_MARGIN},
        "arms": {}, "per_item": [], "anti_tautology": {},
    }
    if not stores:
        result["error"] = "no recorded cold-reload stores at " + STORE_ROOT
        json.dump(result, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(json.dumps(result, indent=2)[:2000])
        return 2

    arm_totals = {name: collections.Counter() for name in
                  ("OFF", "R1", "R1_CONF", "R1_WRONG", "R2", "ORACLE")}
    for db in stores:
        iid = os.path.basename(os.path.dirname(db))
        item = by_id.get(iid)
        if item is None:
            continue
        base_views, stored = census_item_store(db)
        probes, per_cond_counts, traj_by_cond = {}, {}, {}
        for cond in ("near", "far", "foil", "negative"):
            # Vocabulary-consistent scope: the CURRENT frozen item bank supplies
            # source-native typed labels as options. The persisted stores predate
            # later bank edits, so this is a hypothesis probe, not a suite verdict.
            kw = dict(condition=cond)
            r1, c1, _ = replay_overlays(item, base_views, stored, **kw)
            conf_only, c_conf, _ = replay_overlays(
                item, base_views, stored, restore_confidence=True, **kw)
            wrong, c_wrong, _ = replay_overlays(
                item, base_views, stored, tag=WRONG, **kw)
            r2, c2, traj = replay_overlays(
                item, base_views, stored, spaced_cycles=args.spaced_cycles, **kw)
            rprobe = None  # current-bank typed options; recorded rows use an older vocabulary
            arms = {"OFF": base_views, "R1": r1, "R1_CONF": conf_only,
                    "R1_WRONG": wrong, "R2": r2}
            per_cond_counts[cond] = {"r1": c1, "r2": c2, "wrong": c_wrong}
            traj_by_cond[cond] = traj
            for name, views in arms.items():
                sc = score(item, views, cond, probe=rprobe)
                arm_totals[name][(item["pool"], cond, "n")] += 1
                arm_totals[name][(item["pool"], cond, "correct")] += int(sc["correct"])
            # Non-deployable upper bound: gold supplied to the scorer only.
            q = rprobe or mtk.query_for(item, cond)
            oracle_views = copy.deepcopy(r1)
            if cond in ("near", "far") and q["gold"]:
                oracle_views.append({
                    "subject": q["subject"], "target": q["gold"],
                    "relation_type": "is_a", "type_label": q["gold"],
                    "slot_label": "type", "edge_kind": "curated",
                    "edge_source": "conversation", "source_type": "user",
                    "confidence": 1.0})
            sc = score(item, oracle_views, cond, probe=rprobe)
            arm_totals["ORACLE"][(item["pool"], cond, "n")] += 1
            arm_totals["ORACLE"][(item["pool"], cond, "correct")] += int(sc["correct"])
            probes[cond] = {name: score(item, views, cond, probe=rprobe)["reason"]
                            for name, views in arms.items()}

        # Anti-tautology: mutating the answer key must NOT change replay state.
        mutant = copy.deepcopy(item)
        mutant["unseen_test_words"] = copy.deepcopy(item["unseen_test_words"])
        for cond in ("near", "far"):
            if "gold" in mutant["unseen_test_words"].get(cond, {}):
                mutant["unseen_test_words"][cond]["gold"] = "__MUTATED_GOLD__"
        mutant_views, mutant_counts, _ = replay_overlays(
            mutant, base_views, stored)
        baseline_replay, _, _ = replay_overlays(item, base_views, stored)
        same = replay_fingerprint(baseline_replay) == replay_fingerprint(mutant_views)
        if result["anti_tautology"].get("checked") is None:
            result["anti_tautology"] = {
                "checked": True, "gold_mutation_leaves_replay_state_unchanged": same,
                "replay_counts": per_cond_counts["near"]["r1"]}
        elif not same:
            result["anti_tautology"]["gold_mutation_leaves_replay_state_unchanged"] = False

        result["per_item"].append({
            "item_id": iid, "pool": item["pool"], "k": item["k"],
            "stored_typed_edges": len(base_views),
            "replay_counts_by_condition": per_cond_counts,
            "reactivation_mean_confidence": traj_by_cond,
            "probes": probes})

    # Tie the mechanism probe to the ACTUAL reported failure: the recorded
    # P6 cold-reload / k=8 / near+far rows from the focused live run.
    delayed = {}
    rows_path = os.path.join(os.path.dirname(STORE_ROOT), "tra_k_rows.jsonl")
    if os.path.exists(rows_path):
        stores_by_id = {os.path.basename(os.path.dirname(p)): p for p in stores}
        seen = set()
        with open(rows_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if (row.get("arm") != "channel" or row.get("phase") != "delayed"
                        or row.get("k") != 8
                        or row.get("condition") not in ("near", "far")):
                    continue
                key = (row["id"], row["condition"])
                if key in seen:
                    continue
                seen.add(key)
                item = by_id.get(row["id"])
                db = stores_by_id.get(row["id"])
                if item is None or db is None:
                    continue
                base_v, stored = census_item_store(db)
                cond = row["condition"]
                subj = row["input"]["query_subject"]
                cands = row["input"]["candidates"]
                gold = row["gold"]["target"]
                r_pairs = rec_facts.get(row["id"]) or None
                excl = ({subj.strip().lower()} if cond == "negative" else None)
                kw = dict(condition=cond, pairs_override=r_pairs,
                          boost_override=r_pairs, excluded_subjects=excl)
                r1_v, _, _ = replay_overlays(item, base_v, stored, **kw)
                r2_v, _, _ = replay_overlays(item, base_v, stored,
                                             spaced_cycles=args.spaced_cycles, **kw)
                for name, vv in (("OFF", base_v), ("R1", r1_v), ("R2", r2_v)):
                    res = slot_keyed_forced_choice(to_production_views(vv), subj, cands)
                    bucket = delayed.setdefault(
                        name, {"correct": 0, "abstain": 0, "n": 0, "items": []})
                    bucket["n"] += 1
                    ok = res["pick"] is not None and str(res["pick"]).lower() == str(gold).lower()
                    bucket["correct"] += int(ok)
                    bucket["abstain"] += int(res["pick"] is None)
                    bucket["items"].append({"id": row["id"], "condition": cond,
                                            "subject": subj, "gold": gold,
                                            "pick": res["pick"]})
        for bucket in delayed.values():
            bucket["rate"] = round(bucket["correct"] / bucket["n"], 4) if bucket["n"] else None
    result["recorded_delayed_k8_cold_reload"] = delayed
    result["recorded_delayed_scope_valid_for_typed_read"] = False
    result["recorded_delayed_scope_invalid_reason"] = (
        "The focused live run (commit 0829813) predates the typed-option-space "
        "fix; its recorded delayed options are TRA domain words while the "
        "persisted typed slots carry source-native Wikidata labels. A typed-slot "
        "reader cannot score that option vocabulary, so 0/16 in every arm is an "
        "instrument mismatch, NOT a replay result. Do not cite it as retention.")

    for name, counts in arm_totals.items():
        by_pool = {}
        for pool in ("primary", "sealed"):
            row = {}
            for cond in ("near", "far", "foil", "negative"):
                n = counts[(pool, cond, "n")]
                c = counts[(pool, cond, "correct")]
                row[cond] = {"n": n, "correct": c,
                             "rate": round(c / n, 4) if n else None}
            by_pool[pool] = row
        result["arms"][name] = {
            "deployable": name != "ORACLE", "by_pool": by_pool}

    st = result["stored_typed_edges_summary"] = {
        "items": len(result["per_item"]),
        "total_stored_typed_edges": sum(x["stored_typed_edges"] for x in result["per_item"]),
        "restored_traces": sum(x["replay_counts_by_condition"]["near"]["r1"].get(
            "restored", 0) for x in result["per_item"]),
        "no_stored_trace": sum(x["replay_counts_by_condition"]["near"]["r1"].get(
            "absent_no_stored_trace", 0) for x in result["per_item"]),
    }
    result["verdict"] = {
        "instrument_ok": controls["ok"] and result["channel_overlap_ok"] is True
                         and bool(result["anti_tautology"].get(
                             "gold_mutation_leaves_replay_state_unchanged")),
        "replay_has_substrate": st["restored_traces"] > 0,
        "note": ("static mechanism probe only: no engine boot, no emission. "
                 "An OFF==R1 tie with restored_traces>0 locates the gap "
                 "downstream of storage; a flat oracle locates an input deficit."),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    # concise stdout
    print("stores=%d typed_edges=%d restored=%d no_stored_trace=%d" %
          (st["items"], st["total_stored_typed_edges"], st["restored_traces"],
           st["no_stored_trace"]))
    for name in ("OFF", "R1", "R1_CONF", "R1_WRONG", "R2", "ORACLE"):
        p = result["arms"][name]["by_pool"]["primary"]
        print("%-8s near=%s far=%s foil=%s neg=%s" %
              (name, p["near"]["rate"], p["far"]["rate"],
               p["foil"]["rate"], p["negative"]["rate"]))
    print("controls=", controls["ok"], "anti_tautology=",
          result["anti_tautology"].get("gold_mutation_leaves_replay_state_unchanged"),
          "out=", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
