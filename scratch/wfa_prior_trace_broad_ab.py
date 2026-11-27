#!/usr/bin/env python3
"""Broader mechanism A/B: discredited condition only, all 40 gap items, one
shared engine per arm, bank order preserved.  Reports sealed/primary endpoint.
Focused mechanism probe -- NOT a full-harness verdict (credible/unknown arms
and web/PD arms are not run here)."""
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
    ab.arm("off" if name == "off" else "on")
    items = [it for it in w.build_items() if it["kind"] == "gap"]
    rows, t0 = [], time.time()
    for n, it in enumerate(items, 1):
        obs = w.run_internal_source_condition(
            engine, it["claim"], "discredited",
            w.is_source_id(it["claim"], "discredited"))
        f = w.grade_internal_source_condition(obs, "discredited")
        rows.append({"id": it["id"], "pool": it["pool"], "correct": f["correct"],
                     "decision": f["decision"], "rule": f["rule"],
                     "invalidated": f["invalidated"],
                     "fingerprint": f["fingerprint"]})
        if n % 10 == 0:
            print("  %s progress %d/%d (%.1fs)" % (name, n, len(items), time.time() - t0),
                  flush=True)
    rates = {}
    for pool in ("primary", "sealed"):
        sel = [r for r in rows if r["pool"] == pool]
        rates[pool] = {"n": len(sel),
                       "correct": round(sum(r["correct"] for r in sel) / len(sel), 4),
                       "withdraw": round(sum(r["decision"] == "WITHDRAW" for r in sel) / len(sel), 4)}
    return {"arm": name, "elapsed_s": round(time.time() - t0, 1), "rates": rates,
            "failures": [r["id"] for r in rows if not r["correct"]], "rows": rows,
            "substitutions": ab._trace.get("_last_substitution", 0)}


def main():
    original = ab.install()
    em.CognitiveChatEngine._apply_source_monitoring_gate = original
    off = run("off", fresh_store("wfa_store_broad_off"))
    ab.install()
    on = run("on", fresh_store("wfa_store_broad_on"))
    em.CognitiveChatEngine._apply_source_monitoring_gate = original
    result = {"baseline": off, "trace_on": on,
              "controls": {"flag_off_is_unmodified_production": True,
                           "same_store_and_bank_order": True,
                           "substitutions_only_on": (off["substitutions"] == 0 and
                                                      on["substitutions"] > 0)}}
    print("RATES off=%s on=%s" % (json.dumps(off["rates"]), json.dumps(on["rates"])),
          flush=True)
    print("FAILURES off=%s on=%s" % (off["failures"], on["failures"]), flush=True)
    print("CONTROLS", json.dumps(result["controls"]), flush=True)
    out = os.path.join(HERE, "wfa_prior_trace_broad_ab.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print("RESULTS ->", out, flush=True)


if __name__ == "__main__":
    main()
