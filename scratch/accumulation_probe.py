#!/usr/bin/env python3
"""SCRATCH supporting probe: cost growth across repeated URL-heavy WFA turns.

The parent E2C run still timed out after ~1951 lines. This probe measures
whether the newly dominant causal-edge path bloats the in-process graph and
slows later turns, and whether the memoization candidate changes that growth.
12 distinct forced-choice items, two arms, identical seeds, frozen store.
"""
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAIN = r"C:\Users\Likhith\Documents\projects\ravana"
for _p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "ravana", "src"), ROOT):
    sys.path.insert(0, _p)
os.environ["RAVANA_OFFLINE"] = "1"

import measure_fronts_web_frozen as mfwf  # noqa: E402
mfwf.install_frozen_patches()
import measure_learn5_wfa as W  # noqa: E402
from ravana.chat.engine import CognitiveChatEngine  # noqa: E402
from ravana_ml.graph import ConceptEdge  # noqa: E402

ORIGINAL = ConceptEdge.__dict__["_init_relation_vector"].__func__
from ab_relation_vector import CACHED, functional_equivalence  # noqa: E402

N = 12
ITEMS = [it for it in W.build_items() if it["kind"] == "decoy"][:N]
PROMPTS = [W.forced_choice_prompt(it, "exclusion") for it in ITEMS]


def run(label):
    random.seed(0)
    np.random.seed(0)
    ConceptEdge._init_relation_vector = staticmethod(
        CACHED if label == "candidate" else ORIGINAL)
    eng = CognitiveChatEngine(data_dir=os.path.join(MAIN, "weights"), user_suffix="")
    rows = []
    for i, p in enumerate(PROMPTS):
        t0 = time.time()
        eng.process_turn(p)
        rows.append({"turn": i + 1, "wall": round(time.time() - t0, 4),
                     "nodes": len(eng.graph.nodes), "edges": len(eng.graph.edges)})
    return rows


def main():
    assert functional_equivalence()["failures"] == []
    out = {}
    for label in ("baseline", "candidate"):
        rows = run(label)
        out[label] = rows
        print("%s: first=%s last=%s total=%.2fs" % (
            label, rows[0], rows[-1], sum(r["wall"] for r in rows)), flush=True)
    out["identical_graph_counts"] = all(
        (a["nodes"], a["edges"]) == (b["nodes"], b["edges"])
        for a, b in zip(out["baseline"], out["candidate"]))
    print("identical graph counts: %s" % out["identical_graph_counts"], flush=True)
    with open(os.path.join(HERE, "accumulation_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("Wrote accumulation_results.json", flush=True)


if __name__ == "__main__":
    main()
