#!/usr/bin/env python3
"""End-to-end smoke: MemFail cases through a LIVE engine.process_turn."""
import os, sys
_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj)
sys.path.insert(0, os.path.join(_proj, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj, "scripts"))
import importlib
ev = importlib.import_module("evaluate_ravana")

eng = ev.restore_from_snapshot()

def run(name, loader, n=6):
    cases = loader(max_cases=n)
    tot = hits = 0.0
    for c in cases:
        for t in c.get("primer", []):
            eng.process_turn(t)
        r = eng.process_turn(c["question"])
        s = c["grader"](r)
        tot += 1; hits += s
        print(f"  [{s:.2f}] Q:{c['question'][:70]}")
        print(f"         R:{(r or '')[:90]}")
        # fresh buffer between cases
        try:
            eng.hippocampal_buffer.facts.clear()
        except Exception:
            pass
    print(f"{name}: {hits/tot:.3f} over {int(tot)}")

run("coexisting", ev._load_memfail_coexisting)
run("conditional", ev._load_memfail_conditional)
run("longhop", ev._load_memfail_longhop)
run("persona", ev._load_memfail_persona)
