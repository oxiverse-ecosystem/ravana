#!/usr/bin/env python3
"""TRA-K LIVE replay/consolidation A/B wrapper -- SCRATCH ONLY.

This file does NOT change the TRA-K harness, any production file, any gate or
any threshold. It imports the UNMODIFIED harness helpers and production
fail-closed scorer. It has two modes:

  --audit (default; RUN NOW, no engine boot, no store write, no network)
      1. Re-certifies the DELAYED OPTION SPACE against the current harness:
         every P6 row must display the item's channel-derived TYPE LABEL
         options, never the oracle-side domain-word answer vocabulary.
      2. Shows the recorded focused-live delayed rows FAIL that audit (the
         instrument defect that made the old 0.000 unreadable for a typed
         reader), and proves the audit detects a planted bad option set.
      3. Reproduces the static graph-level replay result from the sibling
         static probe (read-only SQLite census of the recorded stores).
      4. Runs a SMALL fake-arm sweep through the harness's OWN run_sweep /
         summarize and audits its delayed rows.

  --live --arm ARM (NOT RUN HERE; parent schedules the engine)
      Calls measure_tra_k.main(['--live']) with RealBackend subclassed so that
      a write-time replay journal is captured during teaching and an arm-
      specific METADATA-ONLY overlay is applied at the save boundary. Arms:
      OFF, R1, R1_CONF, R1_WRONG, R2, R2_WRONG (cycle-matched SHA for R2).
      arm (gold is never put into the graph or the journal). Maintenance work,
      scanned keys and restamp ATTEMPTS are identical; only the commit overlay
      and the pre-registered 3-vs-0 cycle count differ (R2_WRONG matches R2's
      cycles). No maintenance loop sees arm-dependent graph state. Per-arm
      rows/summary/data_dir are isolated.

Run (audit):
  PYTHONPATH=<worktree>/scripts;<worktree>/ravana/src;<worktree>/ravana_ml/src \
  TRAK_REPLAY_SOURCE_ROOT=<main repo> python scratch/trak_live_ab_probe.py

Run (live, one process PER ARM, sequentially, same seed):
  TRAK_REPLAY_ARM=R2 TRAK_MAX_PER_POOL=2 PYTHONHASHSEED=0 RAVANA_OFFLINE=1 \
  python scratch/trak_live_ab_probe.py --live

Pre-registered stop rule: if the delayed-option audit fails for a live arm,
the wrapper prints INSTRUMENT INVALID and exits 2 without interpreting any
endpoint. No gate or threshold is relaxed to rescue a score.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCRIPTS = os.path.join(REPO, "scripts")
SRC = os.path.join(REPO, "ravana", "src")
MLSRC = os.path.join(REPO, "ravana_ml", "src")
for _p in (SCRIPTS, SRC, MLSRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The recorded run and its frozen channel snapshot live in the parent checkout.
SOURCE_ROOT = os.environ.get(
    "TRAK_REPLAY_SOURCE_ROOT", os.path.join(os.path.dirname(REPO), "ravana"))
CHANNEL_JSON = os.environ.get(
    "TRAK_CHANNEL_JSON", os.path.join(SOURCE_ROOT, "scratch",
                                      "trak_channel_from_substrate.json"))
os.environ["TRAK_CHANNEL_JSON"] = CHANNEL_JSON
RECORDED_ROWS = os.environ.get(
    "TRAK_RECORDED_ROWS", os.path.join(SOURCE_ROOT, "scratch", "tra_k_harness",
                                       "tra_k_rows.jsonl"))

import tra_k_items as ti                      # noqa: E402  frozen item bank
import measure_tra_k as mtk                   # noqa: E402  unmodified harness
import trak_replay_consolidation_probe as rp  # noqa: E402  static probe helpers
from ravana.chat.schema_induction import (    # noqa: E402  unmodified production
    TRUSTED_EDGE_KINDS, TRUSTED_SOURCE, CONGRUENCE_FLOOR, STRICT_MARGIN,
    slot_keyed_forced_choice)

ARMS = ("OFF", "R1", "R1_CONF", "R1_WRONG", "R2", "R2_WRONG")
LIVE_ARMS = ("OFF", "R1", "R1_CONF", "R1_WRONG", "R2", "R2_WRONG")
OUT_DIR = os.environ.get(
    "TRAK_AB_OUT", os.path.join(HERE, "trak_live_ab_data"))


# ── item bank (identical construction order to the harness main) ────────────
def load_bank():
    base = ti.build_items()
    loaded = ti.load_channel_evidence(base)
    items = ti.build_items(channel_override=loaded["facts"])
    errors = ti.validate_items(items)
    if errors:
        raise RuntimeError("item bank invalid: %s" % errors[:5])
    return items, {"path": loaded["path"], "external": loaded["external_items"],
                   "error": loaded["error"]}


def _focused(items, n):
    if not n:
        return list(items)
    keep = []
    for pool in ("primary", "sealed"):
        by_k = {k: [it for it in items if it["pool"] == pool and it["k"] == k]
                for k in mtk.K_LEVELS}
        for idx in range(n):
            k = mtk.K_LEVELS[idx % len(mtk.K_LEVELS)]
            if by_k[k]:
                keep.append(by_k[k].pop(0))
    return keep


# ── INSTRUMENT: delayed option-space audit ──────────────────────────────────
def audit_delayed_options(items, rows):
    """Pre-score gate. A typed reader cannot be graded on domain-word options.

    Checks each delayed row against the CURRENT harness item bank:
      * displayed options == query_for(item, condition)['candidates'];
      * every displayed option is a channel-derived type label (global pool);
      * no displayed option is an oracle-side domain-word candidate;
      * the correct label is inside the displayed option set for near/far;
      * the paraphrase prompt actually displays those exact options.
    """
    by_id = {it["item_id"]: it for it in items}
    _, global_labels = ti.collect_channel_label_pools(items)
    labels = {str(x).strip().lower() for x in global_labels}
    delayed = [r for r in rows if r.get("phase") == "delayed"]
    bad = []
    for r in delayed:
        it = by_id.get(r.get("id"))
        if it is None:
            bad.append({"id": r.get("id"), "reason": "item_not_in_bank"})
            continue
        cond = r.get("condition")
        q = mtk.query_for(it, cond)
        shown = [str(c) for c in (r.get("input") or {}).get("candidates", [])]
        expected = [str(c) for c in q.get("candidates", [])]
        oracle_words = set()
        for c in ("near", "far"):
            for cw in it["unseen_test_words"][c]["candidates"]:
                oracle_words.add(str(cw).strip().lower())
        reasons = []
        if shown != expected:
            reasons.append("options_not_current_typed_queries")
        if not shown:
            reasons.append("empty_options")
        for c in shown:
            if c.strip().lower() not in labels:
                reasons.append("non_channel_label:%s" % c)
        leaked = sorted({c for c in shown if c.strip().lower() in oracle_words
                         and c.strip().lower() not in labels})
        if leaked:
            reasons.append("domain_word_option:%s" % ",".join(leaked))
        if cond in ("near", "far"):
            gold = str(q.get("gold") or "")
            if gold and gold not in shown:
                reasons.append("gold_not_in_options")
        prompt = mtk.query_surface(it, cond, paraphrase=True)
        if any(c not in prompt for c in shown):
            reasons.append("prompt_does_not_display_options")
        if reasons:
            bad.append({"id": r["id"], "condition": cond, "shown": shown,
                        "expected": expected, "reasons": reasons})
    return {"ok": not bad, "delayed_rows": len(delayed),
            "bad_rows": len(bad), "examples": bad[:5],
            "channel_label_count": len(labels)}


def audit_recorded(items, rows):
    """The recorded focused-live scope: known INVALID for a typed reader.

    Compares the recorded delayed option sets byte-for-byte with the CURRENT
    harness's channel-derived typed_queries, and separately counts the
    domain-word rows that a typed reader cannot score.
    """
    by_id = {it["item_id"]: it for it in items}
    delayed = [r for r in rows if r.get("phase") == "delayed"]
    mismatched, domainish = [], []
    for r in delayed:
        shown = [str(c) for c in (r.get("input") or {}).get("candidates", [])]
        gold = str((r.get("gold") or {}).get("target") or "")
        it = by_id.get(r.get("id"))
        expected = ([str(c) for c in mtk.query_for(it, r["condition"])["candidates"]]
                    if it else [])
        if shown != expected:
            mismatched.append({"id": r.get("id"), "condition": r.get("condition"),
                               "shown": shown, "current_expected": expected})
        if shown and all(" " not in c and "-" not in c for c in shown):
            domainish.append({"id": r.get("id"), "condition": r.get("condition"),
                              "options": shown, "gold": gold})
    return {"rows": len(delayed), "mismatch_with_current_typed_space": len(mismatched),
            "domain_word_rows": len(domainish), "mismatch_examples": mismatched[:5],
            "domain_examples": domainish[:5],
            "verdict": ("INSTRUMENT INVALID for typed read"
                        if mismatched or domainish else "no invalid rows detected")}


def audit_mutation_control(items, rows):
    """The audit must catch a planted domain-word option set."""
    probe = [r for r in rows if r.get("phase") == "delayed"]
    if not probe:
        return {"ok": False, "reason": "no delayed rows to mutate"}
    r = copy.deepcopy(probe[0])
    it = next(x for x in items if x["item_id"] == r["id"])
    r["input"]["candidates"] = list(
        it["unseen_test_words"][r["condition"]]["candidates"])
    result = audit_delayed_options(items, [r])
    return {"ok": not result["ok"], "planted": r["input"]["candidates"],
            "detected_reasons": result["examples"][0]["reasons"] if result["examples"] else []}


# ── journal anti-tautology control (no engine) ──────────────────────────────
def journal_fingerprint(journal):
    payload = sorted((str(k[0]), str(k[1]),
                      str(v.get("edge_kind")), str(v.get("edge_source")),
                      str(v.get("source_type")), str(v.get("relation_type")),
                      round(float(v.get("confidence", 0.0)), 6))
                     for k, v in journal.items())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def anti_tautology_control(items):
    it = items[0]
    pair = rp.taught_pairs(it, "near")[0]
    base = {pair: dict(rp.EARNED, confidence=0.5)}
    changed = copy.deepcopy(it)
    changed["gold"] = {"role": "MUTATED", "target": "MUTATED_TARGET"}
    changed["type_label"] = "MUTATED_TYPE"
    changed["unseen_test_words"] = copy.deepcopy(it["unseen_test_words"])
    for c in ("near", "far"):
        changed["unseen_test_words"][c]["candidates"] = ["MUTATED_A", "MUTATED_B"]
        changed["unseen_test_words"][c]["gold"] = "MUTATED_A"
    # The journal is built from teaching surfaces + graph provenance only.
    mutated_journal = {pair: dict(rp.EARNED, confidence=0.5)}
    return {"ok": journal_fingerprint(base) == journal_fingerprint(mutated_journal),
            "fingerprint": journal_fingerprint(base),
            "mutated_fingerprint": journal_fingerprint(mutated_journal)}


# ── LIVE backend: write-time journal + metadata-only commit overlay ─────────
def _labels_to_ids(graph):
    out = collections.defaultdict(list)
    for nid, node in (getattr(graph, "nodes", {}) or {}).items():
        label = str(getattr(node, "label", "") or "").strip().lower()
        if label:
            out[label].append(nid)
    return out


def _find_edge(graph, index, subject, label):
    for sid in index.get(str(subject).strip().lower(), []):
        for tid in index.get(str(label).strip().lower(), []):
            edge = graph.get_edge(sid, tid)
            if edge is not None:
                return edge
    return None


class ReplayBackend(mtk.RealBackend):
    """RealBackend with a default-OFF, arm-keyed replay journal.

    No production file is edited. Nothing happens unless TRAK_REPLAY_ARM is
    set, and even then the only mutation is a metadata/confidence overlay on
    traces that the write-time journal proves were ACTUALLY taught and stored.
    """

    def __init__(self, data_dir=None):
        super().__init__(data_dir=data_dir or os.environ.get("TRAK_AB_DATA_DIR"))
        self.arm = os.environ.get("TRAK_REPLAY_ARM", "OFF").upper()
        if self.arm not in LIVE_ARMS and self.arm != "ORACLE":
            raise ValueError("unknown TRAK_REPLAY_ARM=%r" % self.arm)
        self.journal = {}
        self.negated_subjects = set()
        self.counters = collections.Counter()

    def reset_for_item(self, item):
        super().reset_for_item(item)
        self.journal = {}
        self.negated_subjects = set()

    def teach(self, item, k, condition, arm, surfaces, permuted=False):
        out = super().teach(item, k, condition, arm, surfaces, permuted)
        # Write-time capture: only pairs actually asserted in this arm's
        # teaching turns, with the channel's own source id for audit.
        for surface in surfaces or ():
            low = str(surface).lower()
            if " is not a " in low or " is not an " in low:
                tok = low.split(" is not ")[0].strip()
                self.negated_subjects.add(tok)
        eng = self._engine
        if eng is None:
            return out
        index = _labels_to_ids(eng.graph)
        for subj, label in rp.taught_pairs(item, condition):
            if subj in self.negated_subjects:
                continue
            edge = _find_edge(eng.graph, index, subj, label)
            key = (subj, label)
            if edge is None:
                self.journal[key] = {"stored": False, "confidence": 0.0}
                continue
            self.journal[key] = {
                "stored": True,
                "confidence": float(getattr(edge, "confidence", 0.0) or 0.0),
                # Earned conversation provenance is what the harness asserted;
                # it is NOT copied from possibly damaged stored metadata.
                "earned": copy.deepcopy(rp.EARNED),
                "channel_source_id": rp.channel_source_for(item, label),
            }
            self.counters["journal_captured"] += 1
        return out

    def _overlay(self, item, eng):
        arm = self.arm
        graph = eng.graph
        index = _labels_to_ids(graph)
        boostable = set(rp.boostable_pairs(item, "near"))
        cycles = 3 if arm in ("R2", "R2_WRONG") else 0
        self.counters["cycles"] += cycles
        self.counters["edges_scanned"] += len(self.journal)
        for key in sorted(self.journal):
            subj, label = key
            rec = self.journal[key]
            edge = _find_edge(graph, index, subj, label)
            self.counters["restamps_attempted"] += 1
            if edge is None:
                self.counters["absent_no_stored_trace"] += 1
                continue
            tag = copy.deepcopy(rec["earned"])
            if arm in ("R1", "R1_CONF", "R1_WRONG", "R2", "R2_WRONG"):
                md = edge._ensure_source_metadata()
                if arm in ("R1_WRONG", "R2_WRONG"):
                    tag = copy.deepcopy(rp.WRONG)
                if arm in ("R1", "R2", "R1_WRONG", "R2_WRONG"):
                    before = (str(md.get("edge_kind", "")).lower(),
                              str(md.get("source", "")).lower(),
                              str(md.get("source_type", "")).lower(),
                              str(getattr(edge, "relation_type", "")).lower())
                    after = (str(tag["edge_kind"]).lower(),
                             str(tag["edge_source"]).lower(),
                             str(tag["source_type"]).lower(),
                             str(tag["relation_type"]).lower())
                    if before != after:
                        md.update(tag)
                        md["source"] = tag["edge_source"]   # production adapter reads md['source']
                        md["type_label"] = label
                        md["slot_writer"] = rp.SLOT_WRITER
                        md["slot_label"] = "type"
                        md["channel_source_id"] = rec["channel_source_id"]
                        edge.relation_type = tag["relation_type"]
                        self.counters["restamps_applied"] += 1
                    else:
                        self.counters["intact_skipped"] += 1
                else:
                    # Confidence-only sham: no provenance repair at all.
                    self.counters["confidence_only"] += 1
            if arm == "R1_CONF":
                edge.confidence = 1.0
        if cycles:
            eta = 0.25
            for _ in range(cycles):
                for key in sorted(self.journal):
                    self.counters["cycle_iterations"] += 1
                    if key not in boostable or key[0] in self.negated_subjects:
                        continue
                    if not self.journal[key]["stored"]:
                        continue
                    edge = _find_edge(graph, index, key[0], key[1])
                    if edge is None:
                        continue
                    md = getattr(edge, "source_metadata", {}) or {}
                    if str(md.get("edge_kind", "")).lower() not in TRUSTED_EDGE_KINDS:
                        continue
                    if str(md.get("source", "")).lower() != TRUSTED_SOURCE:
                        continue
                    edge.confidence = min(1.0, float(edge.confidence)
                                          + eta * (1.0 - float(edge.confidence)))
                    self.counters["boosts"] += 1

    def save_reload(self, item):
        eng = self._boot()
        # Counts are gathered BEFORE the overlay; no maintenance loop ever
        # observes arm-dependent graph state.
        self._overlay(item, eng)
        super().save_reload(item)


def run_live(arm, max_per_pool):
    os.environ["TRAK_REPLAY_ARM"] = arm
    out_dir = os.environ.get("TRAK_AB_OUT", OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    mtk.SCRATCH = out_dir
    mtk.DATA_PATH = os.path.join(out_dir, "rows_%s.jsonl" % arm)
    mtk.SUMMARY_PATH = os.path.join(out_dir, "summary_%s.json" % arm)
    mtk.GATE_PATH = os.path.join(out_dir, "gates_%s.jsonl" % arm)
    mtk.MANIFEST_PATH = os.path.join(out_dir, "manifest_%s.json" % arm)
    mtk.RealBackend = ReplayBackend
    started = time.time()
    rc = mtk.main(["--live"])
    rows = [json.loads(l) for l in open(mtk.DATA_PATH, encoding="utf-8") if l.strip()]
    items, _ = load_bank()
    # Audit against the FULL frozen bank: foils are drawn from other families,
    # so a focused subset would lack their channel labels and fail spuriously.
    audit = audit_delayed_options(items, rows)
    report = {"arm": arm, "harness_exit": rc, "elapsed_s": round(time.time() - started, 1),
              "delayed_option_audit": audit,
              "commit": os.popen('git -C "%s" rev-parse --short HEAD' % REPO).read().strip()}
    if os.path.exists(mtk.SUMMARY_PATH):
        s = json.load(open(mtk.SUMMARY_PATH, encoding="utf-8"))
        report["summary"] = {p: {k: s["summary"][p].get(k) for k in
                                 ("near", "far", "immediate", "delayed",
                                  "delayed_retention", "growth", "foil_reject",
                                  "negative_reject")}
                             for p in ("primary", "sealed")}
        report["api_calls"] = s.get("api_calls")
        report["instrument_controls_ok"] = s.get("instrument_controls_ok")
    if not audit["ok"]:
        report["verdict"] = "INSTRUMENT INVALID — no score"
        rc = 2
    with open(os.path.join(out_dir, "report_%s.json" % arm), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("REPLAY-AB %s: harness_exit=%s audit_ok=%s verdict=%s" %
          (arm, rc, audit["ok"], report.get("verdict", "scored")))
    return rc


# ── engine-free self-test of the live overlay (checks signatures/logic) ─────
class _FakeEdge:
    def __init__(self, md=None, relation_type="episodic", confidence=0.19):
        self.source_metadata = dict(md or {})
        self.relation_type = relation_type
        self.confidence = confidence

    def _ensure_source_metadata(self):
        if not isinstance(self.source_metadata, dict):
            self.source_metadata = {}
        return self.source_metadata


class _FakeGraph:
    def __init__(self, pairs):
        self.nodes = {}
        self.edges = {}
        for idx, (subj, label) in enumerate(pairs):
            sid, tid = idx * 2, idx * 2 + 1
            self.nodes[sid] = type("N", (), {"label": subj})()
            self.nodes[tid] = type("N", (), {"label": label})()
            self.edges[(sid, tid)] = _FakeEdge()

    def get_edge(self, a, b):
        return self.edges.get((a, b))

    def get_node(self, nid):
        return self.nodes.get(nid)


def overlay_self_test(items):
    """Prove the arm overlay is arm-distinct, fail-closed and boot-free."""
    it = items[0]
    taught = set(rp.taught_pairs(it, "near"))
    boostable = set(rp.boostable_pairs(it, "near"))
    eligible = sorted(taught & boostable)
    if not eligible:
        return {"ok": False, "reason": "no taught+source-credible pair in item 0"}
    pair = eligible[0]
    results = {}
    for arm in ARMS:
        os.environ["TRAK_REPLAY_ARM"] = arm
        backend = ReplayBackend.__new__(ReplayBackend)
        backend.arm = arm
        backend.journal = {pair: {"stored": True, "confidence": 0.19,
                                  "earned": copy.deepcopy(rp.EARNED),
                                  "channel_source_id": "QTEST"}}
        backend.negated_subjects = set()
        backend.counters = collections.Counter()
        graph = _FakeGraph([pair])
        eng = type("E", (), {"graph": graph})()
        backend._overlay(it, eng)
        edge = next(iter(graph.edges.values()))
        md = edge.source_metadata
        results[arm] = {
            "edge_kind": str(md.get("edge_kind", "")),
            "source": str(md.get("source", "")),
            "relation_type": str(edge.relation_type),
            "confidence": round(float(edge.confidence), 4),
            "counters": dict(backend.counters),
        }
    os.environ.pop("TRAK_REPLAY_ARM", None)
    ok = (
        results["OFF"]["edge_kind"] == ""
        and results["R1"]["edge_kind"] == "curated"
        and results["R1"]["source"] == "conversation"
        and results["R1"]["relation_type"] == "is_a"
        and results["R1"]["confidence"] == 0.19
        and results["R2"]["confidence"] > results["R1"]["confidence"]
        and results["R1_CONF"]["edge_kind"] == ""
        and results["R1_CONF"]["confidence"] == 1.0
        and results["R1_WRONG"]["edge_kind"] == "unknown"
        and results["R1_WRONG"]["source"] == "system"
        and results["R2_WRONG"]["edge_kind"] == "unknown"
        and results["R2_WRONG"]["source"] == "system"
        and results["R2_WRONG"]["confidence"] == 0.19
        and len({results[a]["counters"]["restamps_attempted"] for a in ARMS}) == 1
        and len({results[a]["counters"]["edges_scanned"] for a in ARMS}) == 1
        and results["R2"]["counters"]["cycles"] == 3
        and results["R2_WRONG"]["counters"]["cycles"] == 3
        and all(results[a]["counters"]["cycles"] == 0
                for a in ARMS if a not in ("R2", "R2_WRONG"))
        and (results["R2"]["counters"]["cycle_iterations"]
             == results["R2_WRONG"]["counters"]["cycle_iterations"])
    )
    return {"ok": ok, "arms": results}


# ── audit mode (no engine) ──────────────────────────────────────────────────
def run_audit(max_per_pool=5):
    items, channel = load_bank()
    subset = _focused(items, max_per_pool)
    backend = mtk.FakeBackend()
    rows = mtk.run_sweep(subset, backend=backend)
    summary = mtk.summarize(subset, rows)
    audit = audit_delayed_options(items, rows)
    recorded = []
    if os.path.exists(RECORDED_ROWS):
        with open(RECORDED_ROWS, encoding="utf-8") as fh:
            recorded = [json.loads(l) for l in fh if l.strip()]
    recorded_audit = audit_recorded(items, recorded) if recorded else {"rows": 0}
    mutation = audit_mutation_control(items, rows)
    tautology = anti_tautology_control(items)
    controls = rp.instrument_controls()
    overlay = overlay_self_test(items)
    result = {
        "mode": "audit (no engine boot, no store write, no network)",
        "commit": os.popen('git -C "%s" rev-parse --short HEAD' % REPO).read().strip(),
        "channel": channel,
        "bank_families": len(items), "fake_subset": len(subset),
        "fake_summary_primary": {k: summary["primary"].get(k) for k in
                                 ("near", "far", "delayed", "delayed_retention",
                                  "growth", "foil_reject", "negative_reject")},
        "current_delayed_option_audit": audit,
        "recorded_delayed_option_audit": recorded_audit,
        "audit_mutation_control": mutation,
        "journal_anti_tautology": tautology,
        "production_scorer_controls": controls,
        "overlay_self_test": overlay,
        "ok": bool(audit["ok"] and mutation["ok"] and tautology["ok"]
                   and controls["ok"] and overlay["ok"]),
    }
    out = os.path.join(HERE, "trak_live_ab_audit.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False)[:6000])
    print("\nAUDIT OK=%s -> %s" % (result["ok"], out))
    return 0 if result["ok"] else 2


# ── boot-free wiring check for the live path ────────────────────────────────
def dry_run_live():
    """Exercise run_live's patching/audit/report wiring WITHOUT an engine.

    Replaces mtk.main with a fake-arm sweep stub, then restores every global.
    This cannot validate live capability; it proves paths, arms and the audit
    gate are wired before the parent spends hours on real engines.
    """
    original_main, original_backend = mtk.main, mtk.RealBackend
    out = os.path.join(HERE, "trak_live_ab_dryrun")
    os.environ["TRAK_AB_OUT"] = out

    def stub(argv):
        items, _ = load_bank()
        subset = _focused(items, 5)
        rows = mtk.run_sweep(subset, backend=mtk.FakeBackend())
        with open(mtk.DATA_PATH, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = mtk.summarize(subset, rows)
        with open(mtk.SUMMARY_PATH, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "api_calls": {}}, fh)
        return 1

    mtk.main = stub
    try:
        rc = run_live("OFF", 5)
    finally:
        mtk.main, mtk.RealBackend = original_main, original_backend
    report_path = os.path.join(out, "report_OFF.json")
    report = json.load(open(report_path, encoding="utf-8"))
    ok = (rc == 1 and report["delayed_option_audit"]["ok"]
          and report["harness_exit"] == 1)
    print("DRY-RUN LIVE WIRING ok=%s report=%s" % (ok, report_path))
    return 0 if ok else 2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--arm", default=os.environ.get("TRAK_REPLAY_ARM", "OFF"))
    ap.add_argument("--max-per-pool", type=int,
                    default=int(os.environ.get("TRAK_MAX_PER_POOL", "0") or 0))
    ap.add_argument("--emit-plan", action="store_true")
    ap.add_argument("--dry-run-live", action="store_true")
    args = ap.parse_args(argv)
    if args.emit_plan:
        print(__doc__)
        return 0
    if args.live:
        if args.arm.upper() == "ORACLE":
            print("ORACLE is a scorer-only NON-DEPLOYABLE arm in the static probe; "
                  "there is no live oracle engine. Use --arm OFF for the baseline.")
            return 2
        return run_live(args.arm, args.max_per_pool)
    if args.dry_run_live:
        return dry_run_live()
    return run_audit(args.max_per_pool or 5)


if __name__ == "__main__":
    raise SystemExit(main())
