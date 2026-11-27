#!/usr/bin/env python3
"""TRA-K read-surface hook experiment -- SCRATCH ONLY, NO ENGINE BOOT HERE.

Hypothesis (follow-up to the NEGATIVE replay A/B and the NEGATIVE stage-2
write-time re-encode): the remaining failure is a READ-ADMISSION gap, not a
storage gap. The unmodified production reader
(``_typed_slot_candidate -> typed_slot_views_from_graph -> slot_keyed_forced_choice``)
admits an option only from a whole-graph edge carrying a redundant write-time
``type_label`` stamp. A provenance-trusted ``is_a`` edge whose TARGET NODE LABEL
*is* the source-native type is treated as non-evidence. At the first graded read
the admitted view is empty, the hook fails closed, the engine hedges, the
parser records ``no_answer``, ``immediate == 0`` and ``delayed_retention`` is
``null`` by construction.

This file adds a default-OFF, READ-SURFACE hook that derives its pick ONLY from
provenance-stored typed facts already in the live graph, using the graph node's
own source-native label as the retrieval cue. It does NO writes, issues NO
extra ``process_turn`` calls, and imports the UNMODIFIED production harness and
production admission constants. No gate, threshold, item, prompt, gold or
scoring code is edited.

Arms (one process per arm, isolated store + isolated out dir):
  OFF   - unmodified backend; no hook installed (byte-identical baseline)
  READ  - scratch hook installed on the engine instance at the emission boundary
  SHAM  - same hook, same scan/scorer calls, WRONG-TAG provenance predicate;
          admits nothing (compute- and call-matched control)

Run boot-free self-test first:
  PYTHONPATH=<wt>/scripts;<wt>/ravana/src;<wt>/ravana_ml/src \
  TRAK_READ_SOURCE_ROOT=<main repo> python scratch/trak_read_surface_experiment.py --self-test

Then the parent runs each arm live (NO engine boot in this file):
  TRAK_READ_ARM=READ TRAK_AB_DATA_DIR=.../store/READ TRAK_AB_OUT=.../data/READ \
  TRAK_MAX_PER_POOL=2 PYTHONHASHSEED=0 RAVANA_OFFLINE=1 RAVANA_TRA_SLOTS=1 \
  RAVANA_TRA_K_SLOTS=1 python scratch/trak_read_surface_experiment.py --live
"""
from __future__ import annotations

import argparse
import collections
import inspect
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# The frozen channel snapshot and the two untracked scratch helpers live in the
# parent checkout; they are read-only here (exactly as in the stage-2 probe).
SOURCE_ROOT = os.environ.get(
    "TRAK_READ_SOURCE_ROOT", os.path.join(os.path.dirname(REPO), "ravana"))
os.environ.setdefault("TRAK_REPLAY_SOURCE_ROOT", SOURCE_ROOT)

# Insert in reverse priority order: worktree scripts/src FIRST so the UNMODIFIED
# harness and production primitives come from this worktree, parent scratch
# only for its two untracked helper modules.
for _p in (HERE,
           os.path.join(SOURCE_ROOT, "scratch"),
           os.path.join(REPO, "ravana_ml", "src"),
           os.path.join(REPO, "ravana", "src"),
           os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import measure_tra_k as mtk                      # noqa: E402  UNMODIFIED harness
import trak_live_ab_probe as ab                  # noqa: E402  UNMODIFIED wrapper
from ravana.chat.constants import STOP_WORDS    # noqa: E402  production
from ravana.chat.response_gen import parse_forced_choice  # noqa: E402
from ravana.chat.schema_induction import (       # noqa: E402  production, unchanged
    _is_trusted_edge, CONGRUENCE_FLOOR, STRICT_MARGIN, slots_enabled)

ARMS = ("OFF", "READ", "SHAM")
LAST_BACKEND = {"backend": None}

# Wrong-tag control values mirror the replay A/B's WRONG stamp. No earned edge
# carries them, so the SHAM arm does the SAME work and admits nothing.
WRONG_TAG = {"edge_kind": "unknown", "source": "system", "source_type": "system"}


def wrong_tag_trust(edge) -> bool:
    """Matched-compute SHAM predicate: only a deliberately wrong-tagged edge passes."""
    md = getattr(edge, "source_metadata", None)
    if not isinstance(md, dict):
        return False
    return (str(md.get("edge_kind", "") or "").strip().lower() == WRONG_TAG["edge_kind"]
            and str(md.get("source", "") or "").strip().lower() == WRONG_TAG["source"])


def read_surface_pick(graph, subject, options, *, trust=None):
    """Fail-closed forced-choice pick from PROVENANCE-STORED typed facts only.

    This is a READ adapter. It reads the live graph and the displayed option
    list; it never receives an item, gold target, domain word, ``world_model``
    or ``type_sets``, and it writes nothing. Admission is gate-preserving:

      * relation_type == 'is_a' (the typed-slot relation, unchanged);
      * ``trust(edge)`` -- production ``_is_trusted_edge`` in READ, wrong-tag in
        SHAM (the trust predicate itself is never widened);
      * the SOURCE node label equals the resolved subject;
      * the TARGET node's OWN label is a source-native type label that appears
        in the displayed option set; a stored ``type_label`` stamp, when it
        exists, must AGREE with that label (a contradictory stamp is refused);
      * stored edge confidence clears the PRODUCTION congruence floor and the
        winner clears the PRODUCTION strict margin -- both imported, not re-set.

    Encoding-specificity rationale: the graded cue is a source-native label.
    The stored trace's effective cue is therefore the node label that was
    encoded at write time, not a redundant stamp that may never have been
    written. Recovering that cue at read time is a retrieval-cue change, not a
    threshold change.
    """
    trust = trust or _is_trusted_edge
    s = str(subject or "").strip().lower()
    opt = {}
    for o in (options or ()):
        key = str(o).strip().lower()
        if key:
            opt[key] = str(o)
    scores: dict = {}
    nodes = getattr(graph, "nodes", None) or {}
    edges = getattr(graph, "edges", None) or {}
    pairs = edges.items() if hasattr(edges, "items") else []
    for (_sid, _tid), edge in pairs:
        if edge is None:
            continue
        if str(getattr(edge, "relation_type", "") or "").strip().lower() != "is_a":
            continue
        if not trust(edge):
            continue
        snode = nodes.get(_sid) if isinstance(nodes, dict) else None
        tnode = nodes.get(_tid) if isinstance(nodes, dict) else None
        src = str(getattr(snode, "label", "") or "").strip().lower()
        if src != s:
            continue
        label = str(getattr(tnode, "label", "") or "").strip()
        key = label.lower()
        if not key or key not in opt:
            continue
        md = getattr(edge, "source_metadata", None)
        if isinstance(md, dict):
            stamp = str(md.get("type_label") or "").strip().lower()
            if stamp and stamp != key:
                continue
        try:
            conf = float(getattr(edge, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        cur = scores.setdefault(key, {"score": 0.0, "support": 0})
        cur["score"] = max(cur["score"], conf)
        cur["support"] += 1
    eligible = {k: v for k, v in scores.items() if v["score"] >= CONGRUENCE_FLOOR}
    out = {
        "subject": subject,
        "options": list(options or ()),
        "all_slot_scores": {k: round(v["score"], 4) for k, v in scores.items()},
        "eligible": sorted(eligible),
        "congruence_floor": CONGRUENCE_FLOOR,
        "strict_margin": STRICT_MARGIN,
    }
    if not eligible:
        out.update(pick=None, margin=None,
                   reason="no_trusted_typed_slot_above_floor")
        return out
    ranked = sorted(eligible.items(), key=lambda kv: (-kv[1]["score"], kv[0]))
    top_k, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else None
    margin = round(top["score"] - (second["score"] if second else 0.0), 4)
    out["margin"] = margin
    if margin < STRICT_MARGIN:
        out.update(pick=None, reason="strict_margin_not_met")
        return out
    out.update(pick=opt[top_k], reason="trusted_typed_slot_strict_winner")
    return out


def _resolve_subject(question, subject):
    """Same subject resolution as the production read hook (scratch copy)."""
    subj = subject if (subject and str(subject).strip().lower() in question.lower()) else question
    if subj == question and subject:
        head = str(subject).strip().lower().split()[-1:]
        if head and head[0] in question.lower():
            subj = head[0]
    if subj == question:
        words = [w for w in re.findall(r"[a-z][a-z\-]+", question.lower())
                 if len(w) >= 3 and w not in STOP_WORDS
                 and w not in {"what", "which", "who", "true", "about"}]
        if words:
            subj = words[-1]
    return subj


def install_hook(eng, arm, counters):
    """Install the scratch hook on ONE engine instance. Never for OFF."""
    if arm not in ("READ", "SHAM"):
        counters["hooks_installed_OFF"] += 1
        return False
    original = getattr(eng, "_typed_slot_candidate", None)
    if not callable(original):
        counters["hook_not_installed_no_primitive"] += 1
        return False
    trust = _is_trusted_edge if arm == "READ" else wrong_tag_trust

    def hook(raw_input, subject=None):
        counters["hook_calls"] += 1
        try:
            # Structural deferral: a production-admitted pick always wins.
            pick = original(raw_input, subject)
            if pick is not None:
                counters["deferred_to_production"] += 1
                return str(pick)
            q, opts = parse_forced_choice(raw_input)
            if len(opts) < 2 or not q:
                return None
            subj = _resolve_subject(q, subject)
            if not subj:
                return None
            counters["scorer_calls"] += 1
            result = read_surface_pick(getattr(eng, "graph", None), subj,
                                       opts, trust=trust)
            if result.get("pick") is not None:
                counters["picks"] += 1
            return result.get("pick")
        except Exception as exc:                      # surface, never swallow
            counters["hook_errors"] += 1
            counters["hook_last_error"] = repr(exc)
            return None

    eng._typed_slot_candidate = hook
    counters["hooks_installed"] += 1
    return True


class ReadSurfaceBackend(ab.ReplayBackend):
    """Read-surface A/B backend. All mutations are scratch/instance-local."""

    def __init__(self, data_dir=None):
        # Disable the replay arm's save-boundary overlay by construction; this
        # experiment changes ONLY the read surface, never stored state.
        os.environ["TRAK_REPLAY_ARM"] = "OFF"
        super().__init__(data_dir=data_dir or os.environ.get("TRAK_AB_DATA_DIR"))
        self.arm = os.environ.get("TRAK_READ_ARM", "OFF").upper()
        if self.arm not in ARMS:
            raise ValueError("unknown TRAK_READ_ARM=%r" % self.arm)
        self.rcounters = collections.Counter()
        LAST_BACKEND["backend"] = self

    def _overlay(self, item, eng):
        self.rcounters["save_overlay_disabled"] += 1
        return None

    def _boot(self):
        eng = super()._boot()
        if eng is not None and not getattr(eng, "_read_hook_installed", False):
            if install_hook(eng, self.arm, self.rcounters):
                eng._read_hook_installed = True
        return eng


def run_live(arm, max_per_pool):
    if arm not in ARMS:
        raise SystemExit("unknown read-surface arm %r" % arm)
    out_dir = os.environ.get(
        "TRAK_AB_OUT", os.path.join(HERE, "trak_read_surface_data", arm))
    data_dir = os.environ.get(
        "TRAK_AB_DATA_DIR", os.path.join(HERE, "trak_read_surface_store", arm))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.environ["TRAK_REPLAY_ARM"] = "OFF"
    os.environ["TRAK_READ_ARM"] = arm
    os.environ["TRAK_AB_OUT"] = out_dir
    os.environ["TRAK_AB_DATA_DIR"] = data_dir
    mtk.SCRATCH = out_dir
    mtk.DATA_PATH = os.path.join(out_dir, "rows.jsonl")
    mtk.SUMMARY_PATH = os.path.join(out_dir, "summary.json")
    mtk.GATE_PATH = os.path.join(out_dir, "gates.jsonl")
    mtk.MANIFEST_PATH = os.path.join(out_dir, "manifest.json")
    mtk.RealBackend = ReadSurfaceBackend

    started = time.time()
    rc = mtk.main(["--live"])
    elapsed = round(time.time() - started, 1)
    rows = []
    if os.path.exists(mtk.DATA_PATH):
        with open(mtk.DATA_PATH, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    items, _ = ab.load_bank()
    audit = ab.audit_delayed_options(items, rows)   # full frozen bank, not subset
    report = {
        "arm": arm,
        "harness_exit": rc,
        "elapsed_s": elapsed,
        "max_per_pool": max_per_pool,
        "delayed_option_audit": audit,
        "read_flags": {
            "RAVANA_TRA_SLOTS": os.environ.get("RAVANA_TRA_SLOTS"),
            "TRAK_READ_ARM": os.environ.get("TRAK_READ_ARM"),
        },
        "commit": os.popen('git -C "%s" rev-parse --short HEAD' % REPO
                           ).read().strip(),
    }
    backend = LAST_BACKEND["backend"]
    if backend is not None:
        report["read_surface_counters"] = dict(backend.rcounters)
    if os.path.exists(mtk.SUMMARY_PATH):
        with open(mtk.SUMMARY_PATH, encoding="utf-8") as fh:
            summary = json.load(fh)
        report["summary"] = {
            pool: {k: summary["summary"][pool].get(k) for k in
                   ("near", "far", "immediate", "delayed",
                    "delayed_retention", "growth", "foil_reject",
                    "negative_reject")}
            for pool in ("primary", "sealed")}
        report["instrument_controls_ok"] = summary.get("instrument_controls_ok")
        report["api_calls"] = summary.get("api_calls")
        report["engine_response_unparsed"] = summary.get("engine_response_unparsed")
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("READ-SURFACE %s: harness_exit=%s audit_ok=%s" %
          (arm, rc, audit["ok"]))
    return rc


# ── boot-free self-test (fake graph only; NO engine, NO store, NO network) ──
class _Node:
    def __init__(self, label):
        self.label = label


class _Edge:
    def __init__(self, source, target, confidence, kind="curated",
                 source_name="conversation", source_type="user",
                 relation="is_a", stamp="snow storm"):
        self.relation_type = relation
        self.confidence = confidence
        self.source_metadata = {
            "edge_kind": kind, "source": source_name, "source_type": source_type,
            "type_label": stamp, "slot_writer": "tra_typed_slot",
        }


class _Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add(self, sid, s, tid, t, conf=0.9, stamp="snow storm",
             kind="curated", source="conversation"):
        self.nodes[sid] = _Node(s)
        self.nodes[tid] = _Node(t)
        self.edges[(sid, tid)] = _Edge(s, t, conf, kind=kind,
                                       source_name=source, stamp=stamp)
        return self


def _self_test():
    g = _Graph().add(1, "blizzard", 2, "snow storm", conf=0.9,
                     stamp="")   # no redundant stamp: node label is the cue
    opts = ["snow storm", "party", "intersection", "natural phenomenon"]
    checks = {}

    # Signature: never receives an item / gold / world_model / type_sets.
    params = set(inspect.signature(read_surface_pick).parameters)
    checks["no_answer_key_inputs"] = not ({"item", "gold", "world_model",
                                           "type_sets"} & params)

    trusted = read_surface_pick(g, "blizzard", opts)
    checks["node_label_cue_admits"] = trusted["pick"] == "snow storm"
    checks["production_constants"] = (
        trusted["congruence_floor"] == CONGRUENCE_FLOOR
        and trusted["strict_margin"] == STRICT_MARGIN)

    checks["wrong_tag_sham_defers"] = (
        read_surface_pick(g, "blizzard", opts,
                          trust=wrong_tag_trust)["pick"] is None)
    checks["unrelated_subject_defers"] = (
        read_surface_pick(g, "zeppelin", opts)["pick"] is None)
    checks["below_floor_defers"] = (
        read_surface_pick(_Graph().add(1, "blizzard", 2, "snow storm",
                                       conf=0.19, stamp=""),
                          "blizzard", opts)["pick"] is None)
    checks["contradictory_stamp_defers"] = (
        read_surface_pick(_Graph().add(1, "blizzard", 2, "snow storm",
                                       conf=0.9, stamp="earthquake"),
                          "blizzard", opts)["pick"] is None)
    tie = _Graph().add(1, "blizzard", 2, "snow storm", conf=0.9, stamp="")
    tie.add(1, "blizzard", 3, "party", conf=0.9, stamp="party")
    checks["tie_defers"] = read_surface_pick(tie, "blizzard", opts)["pick"] is None

    # Anti-tautology: score depends on graph provenance only. An item gold field
    # that contradicts the graph cannot move the pick (it is never seen).
    fake_item = {"gold": "party", "world_model": {"x": "earthquake"},
                 "type_sets": {"blizzard": ["earthquake"]}}
    checks["answer_key_invariant"] = (
        read_surface_pick(g, "blizzard", opts)["pick"] == "snow storm"
        and fake_item["gold"] == "party")

    # Flag wiring: production boundary is gated by slots_enabled(). Self-test
    # sets it explicitly and verifies the hook is reachable on that gate.
    old_flag = os.environ.get("RAVANA_TRA_SLOTS")
    os.environ["RAVANA_TRA_SLOTS"] = "1"
    try:
        checks["production_read_flag_on"] = slots_enabled() is True
    finally:
        if old_flag is None:
            os.environ.pop("RAVANA_TRA_SLOTS", None)
        else:
            os.environ["RAVANA_TRA_SLOTS"] = old_flag

    class _FakeEngine:
        def __init__(self):
            self.graph = g

        def _typed_slot_candidate(self, raw_input, subject=None):
            return None   # production primitive defers; scratch hook decides

    question = ("Given the demonstrations so far, which option best fits "
                "blizzard? Options: %s. Reply briefly." % ", ".join(opts))
    c_read, c_sham, c_off = collections.Counter(), collections.Counter(), collections.Counter()
    e_read = _FakeEngine()
    install_hook(e_read, "READ", c_read)
    e_sham = _FakeEngine()
    install_hook(e_sham, "SHAM", c_sham)
    e_off = _FakeEngine()
    install_hook(e_off, "OFF", c_off)
    checks["read_hook_picks"] = e_read._typed_slot_candidate(question) == "snow storm"
    checks["sham_hook_defers"] = e_sham._typed_slot_candidate(question) is None
    checks["off_hook_absent"] = not hasattr(e_off, "_read_hook_installed")
    checks["matched_scorer_calls"] = (
        c_read["scorer_calls"] == c_sham["scorer_calls"] == 1
        and c_read["hook_calls"] == c_sham["hook_calls"])
    checks["no_extra_turns"] = (
        "process_turn" not in inspect.getsource(read_surface_pick)
        and "process_turn" not in inspect.getsource(install_hook))

    ok = all(checks.values())
    print(json.dumps({"self_test": "PASS" if ok else "FAIL",
                      "checks": checks}, indent=2))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="boot-free fake-graph checks only")
    ap.add_argument("--live", action="store_true", help="run one live arm")
    ap.add_argument("--arm", default=os.environ.get("TRAK_READ_ARM", "OFF"))
    ap.add_argument("--max-per-pool", type=int,
                    default=int(os.environ.get("TRAK_MAX_PER_POOL", "0") or 0))
    args = ap.parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.live:
        return run_live(args.arm.upper(), args.max_per_pool)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
