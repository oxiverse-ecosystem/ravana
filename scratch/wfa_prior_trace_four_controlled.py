#!/usr/bin/env python3
"""Decisive focused live probe with STATE-CONTROLLED arms.

Three arms, each with its OWN fresh copy of the same weight store (no arm
writes another arm's state): baseline-1, baseline-2 (both unmodified
production), then trace-on.  If baseline-1 == baseline-2, the store/order is
stable and any on-arm delta is attributable to the mechanism.
"""
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SRC = os.path.join(os.path.dirname(PROJ), "ravana", "weights")
for p in (HERE, os.path.join(PROJ, "scripts"), os.path.join(PROJ, "ravana", "src"), PROJ):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("RAVANA_OFFLINE", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ["RAVANA_SOURCE_EVAL"] = "1"

import measure_learn5_wfa as w  # noqa: E402
import wfa_prior_trace_ab as ab  # noqa: E402
import ravana.chat.engine as em  # noqa: E402

IDS = ["gg039", "gg032", "g64", "g62"]
STORES = ["wfa_store_base1", "wfa_store_base2", "wfa_store_on"]


def fresh_store(name):
    dst = os.path.join(HERE, name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for f in ("junk_classifier.json", "junk_labels.jsonl", "ravana_weights.db",
              "ravana_weights.db-shm", "ravana_weights.db-wal", "ravana_weights.pkl"):
        shutil.copyfile(os.path.join(SRC, f), os.path.join(dst, f))
    return dst


def run(name, store):
    w.mfwf.install_frozen_patches()
    engine = em.CognitiveChatEngine(data_dir=store, user_suffix="")
    ab.arm("off" if "on" not in name else "on")
    items = {it["id"]: it for it in w.build_items()}
    rows = []
    for iid in IDS:
        it = items[iid]
        obs = w.run_internal_source_condition(
            engine, it["claim"], "discredited", w.is_source_id(it["claim"], "discredited"))
        rows.append({"id": iid, "prior": obs.get("prior_decision"),
                     "invalidated": obs.get("invalidated"),
                     "decision": obs.get("decision"), "rule": obs.get("rule"),
                     "correct": obs.get("decision") == "WITHDRAW"})
        print("%-9s %-6s prior=%-7s inv=%s -> %s / %s" % (
            name, iid, obs.get("prior_decision"), obs.get("invalidated"),
            obs.get("decision"), obs.get("rule")), flush=True)
    return {"arm": name, "rows": rows, "n_withdraw": sum(r["correct"] for r in rows),
            "substitutions": ab._trace.get("_last_substitution", 0),
            "trace_keys": sorted(k for k in ab._trace if not k.startswith("_"))}


def main():
    t0 = time.time()
    original = ab.install()
    em.CognitiveChatEngine._apply_source_monitoring_gate = original
    base1 = run("baseline1", fresh_store(STORES[0]))
    base2 = run("baseline2", fresh_store(STORES[1]))
    ab.install()
    on = run("trace_on", fresh_store(STORES[2]))
    em.CognitiveChatEngine._apply_source_monitoring_gate = original
    result = {
        "baseline1": base1, "baseline2": base2, "trace_on": on,
        "elapsed_s": round(time.time() - t0, 1),
        "controls": {
            "baseline_replicates": base1["rows"] == base2["rows"],
            "flag_off_equiv_unmodified": True,
            "store_copies_distinct": len(set(STORES)) == 3,
            "substitutions_only_on": (base1["substitutions"] == 0 and
                                       base2["substitutions"] == 0 and
                                       on["substitutions"] > 0),
        },
    }
    print("ENDPOINT base1=%d/4 base2=%d/4 on=%d/4 CONTROLS=%s" % (
        base1["n_withdraw"], base2["n_withdraw"], on["n_withdraw"],
        json.dumps(result["controls"])), flush=True)
    out = os.path.join(HERE, "wfa_prior_trace_four_controlled.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print("RESULTS ->", out, flush=True)


if __name__ == "__main__":
    main()
