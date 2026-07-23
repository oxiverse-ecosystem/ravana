#!/usr/bin/env python3
"""Debug failing coexisting case 3 (cocktails) and persona Yuki case."""
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
    print("Q:", case["question"][:120])
    print("BUFFER:")
    seen = set()
    for fl in eng.hippocampal_buffer.facts.values():
        for f in fl:
            if f.object not in seen:
                seen.add(f.object); print("   *", f.object[:100])
    print("GATE:", repr(eng._try_fact_reasoning(case["question"]))[:160])

co = ev._load_memfail_coexisting(max_cases=6)
show(co[3])
pe = ev._load_memfail_persona(max_cases=6)
show(pe[2])
show(pe[5])
