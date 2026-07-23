#!/usr/bin/env python3
"""Measure fact_reasoning functions offline on REAL MemFail data (primer
facts as the fact store) before wiring into the engine."""
import os, sys, importlib
_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj)
sys.path.insert(0, os.path.join(_proj, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj, "scripts"))
from ravana.core import fact_reasoning as fr
ev = importlib.import_module("evaluate_ravana")

# ConceptNet isa closure (structural category links)
import pickle
_ont = pickle.load(open(os.path.join(_proj, "data/conceptnet/ont.pkl"), "rb"))
_isa = _ont["isa"]
def isa_parents(w):
    w = w.lower().replace(" ", "_")
    out = set(_isa.get(w, set()))
    if w.endswith("s"):
        out |= set(_isa.get(w[:-1], set()))
    # one more hop
    for p in list(out):
        out |= set(_isa.get(p, set()))
    return out

def run(name, loader, n=50):
    cases = loader(max_cases=n)
    tot = ans = hits = 0.0
    fails = []
    for c in cases:
        tot += 1
        facts = c.get("primer", [])
        q = c["question"]
        r = (fr.select_option(q, facts)
             or fr.conditional_answer(q, facts)
             or fr.enumerate_matching(q, facts, isa_parents=isa_parents)
             or fr.entity_fact_answer(q, facts)
             or fr.missing_entity_abstention(q, facts))
        if r is None:
            r = "i don't have that information."
        s = c["grader"](r)
        hits += s
        if r != "i don't have that information.":
            ans += 1
        if s < 0.5 and len(fails) < 4:
            fails.append((q[:90], c["expected"][:60], r[:90]))
    print(f"{name:14s} n={int(tot):3d}  score={hits/tot:.3f}  answered={ans/tot:.2f}")
    for q, e, r in fails:
        print(f"    Q:{q}\n    E:{e}\n    R:{r}")

run("coexisting", ev._load_memfail_coexisting)
run("conditional", ev._load_memfail_conditional)
run("longhop", ev._load_memfail_longhop)
run("persona", ev._load_memfail_persona)
