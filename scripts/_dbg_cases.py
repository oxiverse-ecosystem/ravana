#!/usr/bin/env python3
"""Debug: buffer contents + gate result for specific failing cases."""
import os, sys
_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj)
sys.path.insert(0, os.path.join(_proj, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj, "scripts"))
import importlib
ev = importlib.import_module("evaluate_ravana")
eng = ev.restore_from_snapshot()

def show(case):
    eng.hippocampal_buffer.facts.clear()
    for t in case.get("primer", []):
        eng.process_turn(t)
    print("BUFFER TEXTS:")
    seen = set()
    for fl in eng.hippocampal_buffer.facts.values():
        for f in fl:
            if f.object not in seen:
                seen.add(f.object)
                print("   *", f.object[:110])
    gate = eng._try_fact_reasoning(case["question"])
    print("GATE:", repr(gate)[:200])
    r = eng.process_turn(case["question"])
    print("FULL:", repr(r)[:200])
    print("SCORE:", case["grader"](r))

co = ev._load_memfail_coexisting(max_cases=1)
print("=== coexisting case 0: Q:", co[0]["question"][:80])
show(co[0])

lh = ev._load_memfail_longhop(max_cases=6)
print("\n=== longhop case 1: Q:", lh[1]["question"][:80].replace("\n", " | "))
show(lh[1])
