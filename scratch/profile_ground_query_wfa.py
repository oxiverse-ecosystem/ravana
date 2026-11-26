#!/usr/bin/env python3
"""Scratch-only cProfile of THREE representative WFA turns on current main.

No production file is edited. The frozen web/save patches from the WFA harness
are installed before the engine boots, so this probe cannot write the shared
weight store and cannot reach the network.

Run ONLY after the parent WFA focus run has exited (verify with
`ps -ef | grep measure_learn5_wfa`):

  cd C:/Users/Likhith/Documents/projects/ravana_wfa_gqprof
  PYTHONHASHSEED=0 RAVANA_OFFLINE=1 uv run python scratch/profile_ground_query_wfa.py
"""
import io
import json
import os
import pstats
import sys
import time

import cProfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAIN = r"C:\Users\Likhith\Documents\projects\ravana"
for _p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "ravana", "src"), ROOT):
    sys.path.insert(0, _p)
os.environ["RAVANA_OFFLINE"] = "1"

import measure_fronts_web_frozen as mfwf  # noqa: E402

mfwf.install_frozen_patches()  # blocks network + save, exactly like measure_learn5_wfa.main()

import measure_learn5_wfa as W  # noqa: E402
from ravana.chat.engine import CognitiveChatEngine  # noqa: E402


def pick(kind, pool=None):
    for it in W.build_items():
        if it["kind"] == kind and (pool is None or it["pool"] == pool):
            return it
    raise RuntimeError("no WFA item for kind=%r pool=%r" % (kind, pool))


# Three turns spanning the WFA arms: short inclusion, long-context exclusion
# (the shape that stalls in scratch/wfa_e2b_fresh_focused.log), and the
# single-token forced-choice shape that stalls in wfa_e2_fresh_rerun.log.
TURNS = [
    ("inclusion_short", W.endorsement_prompt(pick("gap", "primary"), "inclusion")),
    ("exclusion_long_context", W.endorsement_prompt(pick("gap", "sealed"), "exclusion")),
    ("forced_choice_provenance",
     W.forced_choice_prompt(pick("decoy", "primary"), "exclusion")),
]


def entries(stats, sort_key, limit):
    stats.sort_stats(sort_key)
    rows = []
    for (mod, line, fn), (cc, nc, tt, ct, _) in stats.stats.items():
        rows.append({"file": mod, "line": line, "func": fn,
                     "ncalls": nc, "tottime": round(tt, 6),
                     "cumtime": round(ct, 6)})
    key = "cumtime" if sort_key == "cumulative" else "tottime"
    rows.sort(key=lambda r: -r[key])
    return rows[:limit]


def main():
    print("Booting engine with frozen patches (no save, no network)...", flush=True)
    t_boot = time.time()
    eng = CognitiveChatEngine(data_dir=os.path.join(MAIN, "weights"), user_suffix="")
    boot_s = time.time() - t_boot
    n_nodes = len(getattr(getattr(eng, "graph", None), "nodes", {}) or {})
    n_glove = len(getattr(eng, "_glove_words", []) or [])
    print("boot_s=%.2f  nodes=%d  glove_words=%d" % (
        boot_s, n_nodes, n_glove), flush=True)

    report = {"boot_s": round(boot_s, 3), "nodes": n_nodes,
              "glove_words": n_glove, "turns": []}
    for name, prompt in TURNS:
        print("\n=== TURN %s (prompt_len=%d) ===" % (name, len(prompt)), flush=True)
        pr = cProfile.Profile()
        t0 = time.time()
        pr.enable()
        try:
            resp = eng.process_turn(prompt)
        finally:
            pr.disable()
        wall = time.time() - t0
        dump = os.path.join(HERE, "profile_%s.pstats" % name)
        pr.dump_stats(dump)
        st = pstats.Stats(pr)
        cum = entries(st, "cumulative", 30)
        tot = entries(st, "tottime", 20)
        # Search ALL profile rows, not only the printed top-N.
        all_rows = []
        for (mod, line, fn), (cc, nc, tt, ct, _) in st.stats.items():
            all_rows.append({"file": mod, "line": line, "func": fn,
                             "ncalls": nc, "tottime": round(tt, 6),
                             "cumtime": round(ct, 6)})
        gq = [r for r in all_rows if "_ground_query" in r["func"]]
        act = [r for r in all_rows if "_activate_from_input" in r["func"]]
        txt = io.StringIO()
        pstats.Stats(pr, stream=txt).sort_stats("cumulative").print_stats(50)
        with open(os.path.join(HERE, "profile_%s.txt" % name), "w",
                  encoding="utf-8") as fh:
            fh.write("turn=%s prompt_len=%d wall=%.3fs\n" % (name, len(prompt), wall))
            fh.write("prompt=%r\n\n" % prompt[:2000])
            fh.write(txt.getvalue())
        rec = {
            "name": name,
            "prompt_len": len(prompt),
            "prompt_head": prompt[:240],
            "wall_s": round(wall, 3),
            "response_head": (resp or "")[:240],
            "last_grounding_method": getattr(eng, "_last_grounding_method", None),
            "ground_query_rows": gq,
            "activate_rows": act,
            "top_cumulative": cum[:20],
            "top_tottime": tot[:15],
        }
        report["turns"].append(rec)
        gq_info = [(r["ncalls"], r["cumtime"]) for r in gq[:2]]
        act_info = [(r["ncalls"], r["cumtime"]) for r in act[:2]]
        print("  wall=%.3fs  _ground_query=%s  _activate_from_input=%s" % (
            wall, gq_info, act_info), flush=True)

    with open(os.path.join(HERE, "profile_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nWrote profile_summary.json + profile_*.pstats/.txt in", HERE, flush=True)


if __name__ == "__main__":
    main()
