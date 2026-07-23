#!/usr/bin/env python3
"""End-to-end smoke: TimeDial cloze cases through a LIVE engine.process_turn,
graded exactly as evaluate_ravana.py grades them."""
import os, sys, json
_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj)
sys.path.insert(0, os.path.join(_proj, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj, "ravana-v2"))
sys.path.insert(0, os.path.join(_proj, "ravana_ml", "src"))
sys.path.insert(0, os.path.join(_proj, "scripts"))
os.environ["RAVANA_SILENT"] = "1"
import importlib
ev = importlib.import_module("evaluate_ravana")
cases = ev._load_timedial(max_cases=20)
engine = ev.restore_from_snapshot()
tot = 0.0
for i, c in enumerate(cases):
    r = engine.process_turn(c["question"])
    s = c["grader"](r)
    tot += s
    print(f"{i+1:2d} score={s:.1f} strat={getattr(engine,'_last_strategy','?'):16s} resp={r[:60]!r} expected={c['expected'][:30]!r}")
print(f"\nTimeDial live 20-case avg: {tot/len(cases):.3f} (was 0.000)")
