#!/usr/bin/env python3
"""Live A/B runner for the default-off lexical-superordinate hook.

Run ONE arm per process so the flag is the only difference. Each arm gets a
fresh isolated copy of the canonical weight store in its own data_dir. The
production hook is wrapped with a call/emit counter (reachability evidence);
the harness's OWN run_sweep()/summarize() are called with their real keys.

Usage:
  python scratch/lexical_ab_runner.py --bench rsi  --tag off1 --flag off --out <json>
  python scratch/lexical_ab_runner.py --bench rsi  --tag on   --flag on  --out <json>
  python scratch/lexical_ab_runner.py --bench tra  --tag off1 --flag off --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = "C:/Users/Likhith/Documents/projects/ravana"
CANON = os.path.join(MAIN, "weights")
for _p in (WT, os.path.join(WT, "scripts"), os.path.join(WT, "ravana", "src"),
           os.path.join(WT, "ravana_ml/src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ["RAVANA_OFFLINE"] = "1"
os.environ.setdefault("PYTHONHASHSEED", "0")

# ── reachability counter ─────────────────────────────────────────────────────
STATS = {"calls": 0, "emits": 0, "forced_choice_turns": 0, "audit": []}
ORIGINAL = None


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_data_dir(dst):
    """Fresh copy of canonical store into dst (mirrors mlc.prepare_data_dir)."""
    os.makedirs(dst, exist_ok=True)
    picked = {}
    for name in os.listdir(CANON):
        if (name.startswith("ravana_weights.pkl") or name == "ravana_weights.db"
                or name.startswith("junk_")):
            src = os.path.join(CANON, name)
            shutil.copy2(src, os.path.join(dst, name))
            picked[name] = _sha(src)
    return picked


def patch_hook():
    """Wrap the production hook at method entry; count calls and emits."""
    global ORIGINAL
    from ravana.chat.engine import CognitiveChatEngine
    ORIGINAL = CognitiveChatEngine._lexical_superordinate_candidate

    def wrapped(self, raw_input, subject=None, current_response=None):
        STATS["calls"] += 1
        pick = ORIGINAL(self, raw_input, subject, current_response)
        if pick is not None:
            STATS["emits"] += 1
        # Audit only forced-choice turns; recompute with the pure primitives.
        try:
            from ravana.chat.response_gen import (
                parse_forced_choice, lexical_teaching_pairs_from_graph)
            from ravana.learn import lexical_superordinate as lx
            q, opts = parse_forced_choice(raw_input)
            if len(opts) >= 2 and q:
                STATS["forced_choice_turns"] += 1
                pairs = lexical_teaching_pairs_from_graph(getattr(self, "graph", None))
                theta = lx.schema_criterion(pairs, None)
                # Best-effort subject resolution for the audit only (the
                # production hook resolves independently and stays pure).
                subj = subject or getattr(self, "_last_subject", None) or ""
                scores = lx.subject_scores(str(subj), opts)
                choice, reason = lx.decide(scores, theta)
                if len(STATS["audit"]) < 400:
                    STATS["audit"].append({
                        "q": q, "opts": opts, "subject": subj,
                        "pairs_n": len(pairs), "theta": theta,
                        "scores": scores, "recomputed_choice": choice,
                        "recomputed_reason": reason, "emitted_pick": pick,
                    })
        except Exception:
            pass
        return pick

    CognitiveChatEngine._lexical_superordinate_candidate = wrapped
    return True


def run_rsi(arm_dir, out):
    import measure_learning_curve as mlc
    import measure_learn5_rsi as m
    shutil.rmtree(arm_dir, ignore_errors=True)
    store_hashes = seed_data_dir(arm_dir)
    # The harness's own focused subset: 2 items per pool, same code path.
    items = m.build_items()
    pick = []
    for pool in ("primary", "sealed"):
        pick += [it for it in items if it.get("pool") == pool][:2]
    backend = m.RealBackend(data_dir=arm_dir)
    t0 = time.time()
    rows = m.run_sweep(pick, backend)
    secs = time.time() - t0
    summ = m.summarize(pick, rows)
    result = {
        "bench": "rsi", "rows": len(rows), "items": [it["item_id"] for it in pick],
        "store_hashes": store_hashes,
        "api_calls": dict(getattr(backend, "api_calls", {})),
        "boot_count": getattr(backend, "engines_booted", None),
        "engine_graded": getattr(backend, "engine_graded", None),
        "unparsed": getattr(backend, "engine_response_unparsed", None),
        "api_error": getattr(backend, "api_error", None),
        "seconds": round(secs, 1),
        "summary": {p: {"by_k": summ[p]["by_k"],
                        "near": summ[p]["near"], "far": summ[p]["far"],
                        "foil": summ[p]["foil"], "negative": summ[p]["negative"],
                        "neg_n": summ[p].get("negative_n"),
                        "foil_n": summ[p].get("foil_n"),
                        "chance": summ[p]["chance"],
                        "negative_reject": summ[p]["negative_reject"],
                        "counterexample_reject": summ[p]["counterexample_reject"],
                        "monotonic_max_drop": summ[p]["monotonic_max_drop"],
                        "graded_max_drop": summ[p]["graded_max_drop"],
                        "growth": summ[p]["growth_headroom"]}
                    for p in ("primary", "sealed")},
        "reachability": {k: STATS[k] for k in ("calls", "emits", "forced_choice_turns")},
        "audit": STATS["audit"],
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    return result


def run_tra(arm_base, out, max_per_pool=20):
    import measure_tra as t
    items = t.tra_items.build_items()
    seen = {}
    pick = []
    for it in items:
        pool = it["pool"]
        if seen.get(pool, 0) < max_per_pool:
            pick.append(it)
            seen[pool] = seen.get(pool, 0) + 1
    # Fresh copy of canonical store into EACH per-item data_dir.
    store_hashes = {}
    for it in pick:
        d = os.path.join(arm_base, it["item_id"])
        shutil.rmtree(d, ignore_errors=True)
        store_hashes[it["item_id"]] = seed_data_dir(d)
    backend = t.RealBackend(data_dir=arm_base)
    t0 = time.time()
    with t.tra_slots_flag():
        rows = t.run_sweep(pick, backend=backend)
        summ = t.summarize(pick, rows)
        stats = t.statistics(pick, rows, summ)
    secs = time.time() - t0
    result = {
        "bench": "tra", "rows": len(rows),
        "items": [it["item_id"] for it in pick],
        "max_per_pool": max_per_pool, "store_hashes": store_hashes,
        "api_calls": dict(getattr(backend, "api_calls", {})),
        "boot_count": getattr(backend, "engines_booted", None),
        "engine_graded": getattr(backend, "engine_graded", None),
        "unparsed": getattr(backend, "engine_response_unparsed", None),
        "api_error": getattr(backend, "api_error", None),
        "seconds": round(secs, 1),
        "summary": {p: {"transfer": summ[p]["transfer"],
                        "near": summ[p]["near"], "far": summ[p]["far"],
                        "foil": summ[p]["foil"], "negative": summ[p]["negative"],
                        "chance": summ[p]["chance"],
                        "foil_reject": summ[p]["foil_reject"],
                        "negative_reject": summ[p]["negative_reject"],
                        "counterexample_reject": summ[p]["counterexample_reject"],
                        "monotonic_max_drop": summ[p]["monotonic_max_drop"],
                        "graded_max_drop": summ[p]["graded_max_drop"],
                        "role_coverage": summ[p]["role_coverage"]}
                    for p in ("primary", "sealed")},
        "statistics": stats,
        "reachability": {k: STATS[k] for k in ("calls", "emits", "forced_choice_turns")},
        "audit": STATS["audit"],
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=("rsi", "tra"), required=True)
    ap.add_argument("--tag", required=True, help="off1/off2/on")
    ap.add_argument("--flag", choices=("off", "on"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=os.path.join(WT, "scratch", "ab_runs"))
    ap.add_argument("--max-per-pool", type=int, default=20)
    args = ap.parse_args()

    if args.flag == "on":
        os.environ["RAVANA_LEXICAL_SUPERORDINATE"] = "1"
    else:
        os.environ.pop("RAVANA_LEXICAL_SUPERORDINATE", None)

    patch_hook()
    os.makedirs(args.base, exist_ok=True)
    config = {k: os.environ.get(k) for k in (
        "RAVANA_LEXICAL_SUPERORDINATE", "RAVANA_WORDNET_DATA", "RAVANA_OFFLINE",
        "RAVANA_TRA_SLOTS", "RAVANA_SCHEMA", "RAVANA_SOURCE_EVAL", "PYTHONHASHSEED")}
    from ravana.learn import lexical_superordinate as lx
    config["wordnet_available"] = lx.wordnet_available()
    print("CONFIG", json.dumps({**config, "bench": args.bench, "tag": args.tag,
                                "flag": args.flag}), flush=True)

    d = os.path.join(args.base, args.bench + "_" + args.tag)
    if args.bench == "rsi":
        res = run_rsi(d, args.out)
    else:
        res = run_tra(d, args.out, args.max_per_pool)

    print("RESULT", json.dumps({
        "bench": args.bench, "tag": args.tag, "flag": args.flag,
        "reachability": res["reachability"], "rows": res["rows"],
        "api_calls": res["api_calls"], "api_error": res["api_error"],
        "seconds": res["seconds"],
        "summary": {p: {k: res["summary"][p].get(k) for k in (
            "near", "far", "foil", "transfer", "negative_reject", "foil_reject",
            "chance", "growth")} for p in ("primary", "sealed")},
    }), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
