#!/usr/bin/env python3
"""Cheap OOM probe: track RSS + len() of candidate stores, no deep traversal."""
import os, sys, gc, psutil
_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj); sys.path.insert(0, os.path.join(_proj, "ravana", "src"))
import importlib
ev = importlib.import_module("evaluate_ravana")
cases = ev._load_memfail(max_cases=200)
eng = ev.restore_from_snapshot()

def rss():
    return psutil.Process().memory_info().rss / 1e6

def lens():
    out = {}
    try: out["epi_index"] = len(eng._episodic_index)
    except Exception: pass
    try: out["hipp_all"] = len(eng.hippocampal_buffer._all_facts)
    except Exception: pass
    try: out["hipp_keys"] = len(eng.hippocampal_buffer.facts)
    except Exception: pass
    try: out["qconcepts"] = len(eng.user_model.query_concepts)
    except Exception: pass
    try: out["edges"] = len(eng.user_model.edge_reactivations)
    except Exception: pass
    try: out["prefs"] = len(str(eng.user_model.preferences))
    except Exception: pass
    try: out["transcript"] = len(getattr(eng, "_episodic_transcript", []))
    except Exception: pass
    try: out["epistemic"] = len(getattr(eng, "_epistemic_new_tags", {}))
    except Exception: pass
    return out

print(f"start rss={rss():.0f}MB cases={len(cases)}")
print("baseline lens:", lens())
for i, c in enumerate(cases):
    try:
        eng.hippocampal_buffer.facts.clear()
    except Exception:
        pass
    for t in c.get("primer", []):
        try:
            eng.process_turn(t)
        except Exception:
            pass
    try:
        eng.process_turn(c["question"])
    except Exception:
        pass
    if (i+1) % 10 == 0:
        gc.collect()
        print(f"case {i+1}: rss={rss():.0f}MB {lens()}")
print("done")
