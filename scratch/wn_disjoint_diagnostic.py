#!/usr/bin/env python3
"""DIAGNOSTIC ONLY: why did 6 disjoint-option controls emit?"""
import json, os, sys, collections, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import tra_k_items as tk
import nltk
nltk.data.path.append(r"C:/Users/Likhith/AppData/Roaming/nltk_data")
from nltk.corpus import wordnet as wn

def ss(w): return wn.synsets(w, pos="n")
def lcs(a, b):
    best, bd = None, -1
    for x in ss(a):
        for y in ss(b):
            try: l = x.lowest_common_hypernyms(y)[0]
            except Exception: continue
            d = l.max_depth()
            if d > bd: bd, best = d, l
    return best, bd

data = json.load(open(os.path.join(ROOT, "scratch", "lexical_superordinate_results.json"), encoding="utf-8"))
rows = data["rows"]
bad = [r for r in rows if r["struct_disjoint"] is not None]
print("disjoint emits", len(bad))
for r in bad:
    w = r["struct_disjoint"]
    l, d = lcs(r["subject"], w)
    print(r["item_id"], r["subject"], "->", w, "lcs", l.name() if l else None, "depth", d,
          "theta", r["theta"], "scores", {k: v for k, v in r["struct_scores"].items()})

ks = [r for r in rows if r["struct"] is not None]
print("\ngraded emits", len(ks), "lcs depths")
depths = collections.Counter()
for r in ks:
    l, d = lcs(r["subject"], r["struct"])
    depths[d] += 1
print(sorted(depths.items()))

bank = {it["item_id"]: it for it in tk.build_items()}
dd = []
for it in bank.values():
    for a, b in [(e["r1"], e["r2"]) for e in it["taught_exemplars"][:it["k"]]]:
        l, d = lcs(a, b)
        if d >= 0: dd.append(d)
print("demo pair LCS depths: mean", round(statistics.mean(dd), 2),
      "median", statistics.median(dd), "min", min(dd), "max", max(dd))
