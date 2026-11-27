#!/usr/bin/env python3
"""TRA-K STAGE-2 scratch-only live A/B wrapper. NO production edits.

Hypothesis (conditional on R1/R2 failing): zero retention is caused by a
WRITE-TIME admission defect -- the channel arm's typed slot edge is not
admitted by the fail-closed reader at the FIRST (immediate) probe, so the
retention denominator is zero. Save-boundary replay cannot repair a probe
that runs before the save. Stage 2 therefore re-encodes the SAME
source-backed assertion at write time until the production reader admits it.

Arms:
  OFF   - unmodified baseline (ab.ReplayBackend, arm OFF)
  S2    - after each channel teaching block, if the production scorer does
          not admit a source-taught slot, re-assert that exact surface once
          and stamp its EARNED provenance. Max 2 re-encodes per (subj,label).
  SHAM  - identical number of extra process_turn calls with a neutral
          surface, no slot write/stamp (compute-matched control).

Never reads item['gold'], unseen_test_words gold, world_model or type_sets.
No threshold, gate, prompt or harness scoring code is changed.
Parent runs this ONLY after reading the stage-2 design doc's falsifiers.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (os.path.join(REPO, "scripts"),
           os.path.join(REPO, "ravana", "src"),
           os.path.join(REPO, "ravana_ml", "src"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import measure_tra_k as mtk                          # noqa: E402
import trak_live_ab_probe as ab                      # noqa: E402
import trak_replay_consolidation_probe as rp         # noqa: E402
from ravana.chat.schema_induction import (           # noqa: E402
    typed_slot_views_from_graph, slot_keyed_forced_choice)

MAX_REENCODES = 2
NEUTRAL = "things happen for many reasons, don't they?"
ARMS = ("OFF", "S2", "SHAM")


class Stage2Backend(ab.ReplayBackend):
    """Write-time slot re-encoding; all mutations are scratch-only."""

    def __init__(self, data_dir=None):
        # ReplayBackend validates TRAK_REPLAY_ARM against its own six arms;
        # disable its save-boundary overlay by construction, then select the
        # stage-2 arm explicitly so teach() can see it.
        os.environ["TRAK_REPLAY_ARM"] = "OFF"
        super().__init__(data_dir=data_dir)
        self.arm = os.environ.get("TRAK_STAGE2_ARM", "OFF").upper()

    def _overlay(self, item, eng):
        self.counters["stage2_save_overlay_disabled"] += 1
        return None

    def reset_for_item(self, item):
        super().reset_for_item(item)
        self._done = collections.Counter()

    def teach(self, item, k, condition, arm, surfaces, permuted=False):
        out = super().teach(item, k, condition, arm, surfaces, permuted)
        mode = self.arm
        if mode not in ("S2", "SHAM"):
            return out
        eng = self._engine
        if eng is None or arm != mtk.CHANNEL_ARM or k < 1:
            return out
        if condition in ("foil", "negative", "unrelated"):
            return out
        pairs = rp.taught_pairs(item, condition)
        options = list(mtk.query_for(item, condition).get("candidates") or [])
        for subj, label in pairs:
            key = (subj, label)
            views = typed_slot_views_from_graph(eng.graph)
            scorer = slot_keyed_forced_choice(views, subj, options)
            self.counters["stage2_scorer_calls"] += 1
            if scorer.get("pick") == label:
                self.counters["stage2_already_admitted"] += 1
                continue
            if self._done[key] >= MAX_REENCODES:
                self.counters["stage2_cap_hit"] += 1
                continue
            self._done[key] += 1
            if mode == "SHAM":
                eng.process_turn(NEUTRAL)
                self.counters["stage2_sham_turns"] += 1
                continue
            # Source-backed re-assertion: the label came from the channel
            # fact already taught in this arm, never from the answer key.
            eng.process_turn("%s is a %s." % (subj, label))
            self.counters["stage2_reencodes"] += 1
            index = ab._labels_to_ids(eng.graph)
            edge = ab._find_edge(eng.graph, index, subj, label)
            if edge is None:
                self.counters["stage2_no_edge"] += 1
                continue
            md = edge._ensure_source_metadata()
            md.update(rp.EARNED)
            md["source"] = rp.EARNED["edge_source"]
            md["type_label"] = label
            md["slot_writer"] = rp.SLOT_WRITER
            md["slot_label"] = "type"
            md["channel_source_id"] = rp.channel_source_for(item, label)
            edge.relation_type = "is_a"
            self.counters["stage2_stamps"] += 1
        return out


def run_live(arm, max_per_pool):
    if arm not in ARMS:
        raise SystemExit("unknown stage-2 arm %r" % arm)
    out_dir = os.environ.get(
        "TRAK_AB_OUT", os.path.join(HERE, "trak_stage2_data", arm))
    data_dir = os.environ.get(
        "TRAK_AB_DATA_DIR", os.path.join(HERE, "trak_stage2_store", arm))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.environ["TRAK_REPLAY_ARM"] = "OFF"     # no save-boundary overlay
    os.environ["TRAK_AB_OUT"] = out_dir
    os.environ["TRAK_AB_DATA_DIR"] = data_dir
    mtk.SCRATCH = out_dir
    mtk.DATA_PATH = os.path.join(out_dir, "rows.jsonl")
    mtk.SUMMARY_PATH = os.path.join(out_dir, "summary.json")
    mtk.GATE_PATH = os.path.join(out_dir, "gates.jsonl")
    mtk.MANIFEST_PATH = os.path.join(out_dir, "manifest.json")
    mtk.RealBackend = Stage2Backend
    started = time.time()
    rc = mtk.main(["--live"])
    report = {"arm": arm, "harness_exit": rc,
              "elapsed_s": round(time.time() - started, 1),
              "commit": os.popen('git -C "%s" rev-parse --short HEAD' % REPO
                                 ).read().strip(),
              "max_per_pool": max_per_pool}
    if os.path.exists(mtk.SUMMARY_PATH):
        s = json.load(open(mtk.SUMMARY_PATH, encoding="utf-8"))
        report["summary"] = {p: {k: s["summary"][p].get(k) for k in
                                 ("near", "far", "immediate", "delayed",
                                  "delayed_retention", "growth",
                                  "foil_reject", "negative_reject")}
                             for p in ("primary", "sealed")}
        report["instrument_controls_ok"] = s.get("instrument_controls_ok")
        report["api_calls"] = s.get("api_calls")
    with open(os.path.join(out_dir, "report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("STAGE2 %s: harness_exit=%s" % (arm, rc))
    return rc


def _self_test():
    """Boot-free check: scorer, arm wiring and overlay no-op; no engine."""
    earned = rp.TypedSlotEdgeView(
        source="blizzard", relation_type="is_a", target="snow storm",
        type_label="snow storm", slot_label="type", edge_kind="curated",
        edge_source="conversation", source_type="user", confidence=0.9)
    wrong = rp.TypedSlotEdgeView(
        source="blizzard", relation_type="is_a", target="snow storm",
        type_label="snow storm", slot_label="type", edge_kind="unknown",
        edge_source="system", source_type="system", confidence=0.9)
    opts = ["snow storm", "party", "intersection", "natural phenomenon"]
    ok = (slot_keyed_forced_choice([earned], "blizzard", opts)["pick"]
          == "snow storm"
          and slot_keyed_forced_choice([wrong], "blizzard", opts)["pick"]
          is None)
    prior = os.environ.get("TRAK_STAGE2_ARM")
    try:
        os.environ["TRAK_STAGE2_ARM"] = "S2"
        backend = Stage2Backend(data_dir=os.path.join(
            HERE, "scratch", "selftest_store"))
        backend._overlay(None, None)
        wired = (backend.arm == "S2"
                 and backend.counters["stage2_save_overlay_disabled"] == 1)
    finally:
        if prior is None:
            os.environ.pop("TRAK_STAGE2_ARM", None)
        else:
            os.environ["TRAK_STAGE2_ARM"] = prior
    ok = bool(ok and wired)
    print("STAGE2 BOOT-FREE SELF-TEST ok=%s scorer=%s wiring=%s"
          % (ok, ok, wired))
    return 0 if ok else 2


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    arm = os.environ.get("TRAK_STAGE2_ARM", "OFF")
    raise SystemExit(run_live(
        arm, int(os.environ.get("TRAK_MAX_PER_POOL", "0") or 0)))
