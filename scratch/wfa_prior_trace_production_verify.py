#!/usr/bin/env python3
"""Live verification of the PRODUCTION flag (no monkeypatch).

Three fresh copies of the same store; the unmodified WFA runner on the four
recorded failures.  Arms differ ONLY in RAVANA_PRIOR_ACT_TRACE.
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


def fresh(name):
    dst = os.path.join(HERE, name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for f in ("junk_classifier.json", "junk_labels.jsonl", "ravana_weights.db",
              "ravana_weights.db-shm", "ravana_weights.db-wal", "ravana_weights.pkl"):
        shutil.copyfile(os.path.join(SRC, f), os.path.join(dst, f))
    return dst


def run(name, flag_value, store):
    from ravana.chat.engine import CognitiveChatEngine
    w.mfwf.install_frozen_patches()
    if flag_value is None:
        os.environ.pop("RAVANA_PRIOR_ACT_TRACE", None)
    else:
        os.environ["RAVANA_PRIOR_ACT_TRACE"] = flag_value
    eng = CognitiveChatEngine(data_dir=store, user_suffix="")
    items = {it["id"]: it for it in w.build_items()}
    rows = []
    for iid in ("gg039", "gg032", "g64", "g62"):
        it = items[iid]
        obs = w.run_internal_source_condition(
            eng, it["claim"], "discredited", w.is_source_id(it["claim"], "discredited"))
        rows.append({"id": iid, "prior": obs.get("prior_decision"),
                     "decision": obs.get("decision"), "rule": obs.get("rule")})
        print("%-8s %-6s prior=%-7s -> %s / %s" % (
            name, iid, obs.get("prior_decision"), obs.get("decision"), obs.get("rule")),
            flush=True)
    return {"arm": name, "flag": flag_value, "rows": rows,
            "n_withdraw": sum(r["decision"] == "WITHDRAW" for r in rows),
            "trace_keys": sorted(getattr(eng, "_source_act_trace", {}).keys())}


def main():
    t0 = time.time()
    off = run("unset", None, fresh("wfa_prod_unset"))
    zero = run("zero", "0", fresh("wfa_prod_zero"))
    on = run("one", "1", fresh("wfa_prod_one"))
    out = {"unset": off, "zero": zero, "one": on, "elapsed_s": round(time.time() - t0, 1),
           "controls": {"unset_equals_zero": off["rows"] == zero["rows"],
                        "no_trace_when_off": not off["trace_keys"] and not zero["trace_keys"],
                        "trace_written_when_on": bool(on["trace_keys"])}}
    print("ENDPOINT unset=%d/4 zero=%d/4 one=%d/4 CONTROLS=%s" % (
        off["n_withdraw"], zero["n_withdraw"], on["n_withdraw"],
        json.dumps(out["controls"])), flush=True)
    p = os.path.join(HERE, "wfa_prior_trace_production_verify.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("RESULTS ->", p, flush=True)


if __name__ == "__main__":
    main()
