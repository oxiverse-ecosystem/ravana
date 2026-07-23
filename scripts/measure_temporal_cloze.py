#!/usr/bin/env python3
"""Measure temporal_cloze.solve_cloze on REAL TimeDial data BEFORE wiring.
Top-1 accuracy over 4 shuffled candidates (correct1/correct2 both count;
official TimeDial scores plausibility of both correct options)."""
import os, sys, json, random
_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_proj, "ravana", "src"))
from ravana.core.temporal_cloze import solve_cloze

data = json.load(open(os.path.join(_proj, "data/benchmarks/timedial/test.json"),
                      encoding="utf-8"))
rng = random.Random(42)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
hit1 = hit_any = 0
fails = []
for it in data[:n]:
    dialog = "\n".join(it["conversation"]).replace("<MASK>", "________")
    opts = [it["correct1"], it["correct2"], it["incorrect1"], it["incorrect2"]]
    opts = [o for o in opts if o and o.strip() and o.strip().lower() != "none"]
    gold = {it["correct1"].strip().lower()}
    if it["correct2"] and it["correct2"].strip().lower() != "none":
        gold.add(it["correct2"].strip().lower())
    rng.shuffle(opts)
    idx, score, why = solve_cloze(dialog, opts)
    pick = opts[idx].strip().lower()
    if pick == it["correct1"].strip().lower():
        hit1 += 1
    if pick in gold:
        hit_any += 1
    elif len(fails) < 8:
        fails.append((pick, sorted(gold), why))
print(f"n={n}  top1-correct1={hit1/n:.3f}  top1-any-correct={hit_any/n:.3f}  chance~0.5 (2 of 4 correct)")
print("sample failures (pick | gold | why):")
for p, g, w in fails:
    print("  ", p, "|", g, "|", w)
