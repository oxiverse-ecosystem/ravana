#!/usr/bin/env python3
"""Verify the noise distribution of U1's claim-seeded accumulator (boot-free)."""
import os, sys, hashlib, statistics, json
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
for p in (os.path.join(PROJ, "scripts"), os.path.join(PROJ, "ravana", "src"), PROJ):
    if p not in sys.path: sys.path.insert(0, p)
import measure_learn5_wfa as w
import scratch.wfa_unsat_articulation as u
s = json.load(open(os.path.join(PROJ, "..", "ravana", "scratch", "learn5_wfa_summary.json"), encoding="utf-8"))
items = {i["id"]: i for i in w.build_items()}
rows = [r for r in s["pd_audit"] if r["kind"] == "gap" and r["pool"] == "primary"]
for mode in ("invalidate", "restricted"):
    vs = []
    for r in rows:
        seed = int(hashlib.sha256((u.SEED_SALT + "|" + items[r["id"]]["claim"] + "|" + mode).encode()).hexdigest()[:16], 16)
        import random
        rng = random.Random(seed)
        v = sum(u.SIGMA / u.DDM_STEPS**0.5 * rng.gauss(0.0, 1.0) for _ in range(u.DDM_STEPS))
        vs.append(v)
    print(mode, "n", len(vs), "mean", round(statistics.mean(vs), 4),
          "sd", round(statistics.stdev(vs), 4), "P(v>=-0.9)", round(sum(v >= -0.9 for v in vs)/len(vs), 4))
