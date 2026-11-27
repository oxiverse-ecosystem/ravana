#!/usr/bin/env python3
"""Focused LIVE diagnostic: what does the engine record on the first WFA turn?

One engine boot on a COPIED weight store (production store untouched).  Runs
the UNMODIFIED production runner on 4 recorded failures + 2 sealed successes.
The only instrumentation is a wrapper around the harness's own _is_process_turn
that snapshots engine._last_source_evaluation after each call.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for p in (os.path.join(PROJ, "scripts"), os.path.join(PROJ, "ravana", "src"), PROJ):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("RAVANA_OFFLINE", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ["RAVANA_SOURCE_EVAL"] = "1"

import measure_learn5_wfa as w  # noqa: E402

IDS = ["gg039", "gg032", "g64", "g62", "gg001", "gg002"]


def main():
    w.mfwf.install_frozen_patches()
    from ravana.chat.engine import CognitiveChatEngine
    t0 = time.time()
    engine = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"), user_suffix="")
    print("BOOT_S=%.1f" % (time.time() - t0), flush=True)
    turns = []
    real = w._is_process_turn

    def traced(eng, prompt):
        out = real(eng, prompt)
        dec = getattr(eng, "_last_source_evaluation", None)
        turns.append({
            "prompt_head": (prompt or "")[:45].replace("\n", " "),
            "decision": dec.decision.value if dec is not None else None,
            "rule": dec.rule if dec is not None else None,
            "origin": dec.origin_id if dec is not None else None,
        })
        return out

    w._is_process_turn = traced
    items = {it["id"]: it for it in w.build_items()}
    rows = []
    for iid in IDS:
        it = items.get(iid)
        if it is None:
            print("MISSING ITEM", iid, flush=True)
            continue
        turns.clear()
        obs = w.run_internal_source_condition(
            engine, it["claim"], "discredited",
            w.is_source_id(it["claim"], "discredited"))
        rows.append({
            "id": iid,
            "prior_captured": obs.get("prior_decision"),
            "teached": obs.get("teached"),
            "invalidated": obs.get("invalidated"),
            "decision": obs.get("decision"),
            "rule": obs.get("rule"),
            "token": obs.get("token"),
            "fingerprint": obs.get("staged_fingerprint"),
            "turns": list(turns),
        })
        print("ITEM %s prior=%s inv=%s final=%s/%s turns=%s" % (
            iid, obs.get("prior_decision"), obs.get("invalidated"),
            obs.get("decision"), obs.get("rule"),
            [(t["decision"], t["rule"]) for t in turns]), flush=True)

    out = os.path.join(HERE, "wfa_prior_trace_diag.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "elapsed_s": round(time.time() - t0, 1)}, fh, indent=2)
    print("ELAPSED_S=%.1f -> %s" % (time.time() - t0, out), flush=True)


if __name__ == "__main__":
    main()
