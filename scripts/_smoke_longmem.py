#!/usr/bin/env python3
"""End-to-end LongMemEval smoke: first 6 oracle cases through the LIVE engine."""
import os, sys, time
_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj)
sys.path.insert(0, os.path.join(_proj, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj, "scripts"))
import importlib
ev = importlib.import_module("evaluate_ravana")

cases = ev._load_longmemeval_oracle(max_cases=6)
eng = ev.restore_from_snapshot()

tot = hits = 0.0
t0 = time.time()
for c in cases:
    try:
        eng.hippocampal_buffer.facts.clear()
    except Exception:
        pass
    for t in c.get("primer", []):
        eng.process_turn(t)
    r = eng.process_turn(c["question"])
    s = c["grader"](r)
    tot += 1; hits += s
    print(f"[{s:.2f}] Q: {c['question'][:70]}")
    print(f"       E: {str(c['expected'])[:60]}")
    print(f"       R: {(r or '')[:100]}")
print(f"\nelapsed {time.time()-t0:.0f}s")
print(f"longmem smoke: {hits/tot:.3f} over {int(tot)}")
