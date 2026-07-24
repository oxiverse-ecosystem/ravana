#!/usr/bin/env python3
"""Measure RSS + store sizes across LongMemEval priming (edge cap now active)."""
import os, sys, gc, psutil
sys.path.insert(0, "."); sys.path.insert(0, "ravana/src"); sys.path.insert(0, "scripts")
import importlib
ev = importlib.import_module("evaluate_ravana")
eng = ev.restore_from_snapshot()
def rss():
    return psutil.Process().memory_info().rss / 1e6
def sizes():
    return (len(eng.hippocampal_buffer._all_facts), len(eng._episodic_index),
            len(eng.graph.edges), len(getattr(eng, "_pending_learning_queue", [])))
cases = ev._load_longmemeval_oracle(max_cases=50)
print("baseline rss=%.0fMB buf/idx/edges/queue=%s" % (rss(), sizes()), flush=True)
for i, c in enumerate(cases):
    try:
        eng.reset_episodic_state()
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
    gc.collect()
    if (i + 1) % 5 == 0:
        print("case %d rss=%.0fMB buf/idx/edges/queue=%s" % (i + 1, rss(), sizes()), flush=True)
print("done", flush=True)
