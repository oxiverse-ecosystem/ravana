#!/usr/bin/env python3
"""EXPLORATORY ONLY: demo-pair calibration stats and LCS patterns (primary)."""
import collections, json, os, statistics, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import tra_k_items as tk
import nltk
nltk.data.path.append(r"C:/Users/Likhith/AppData/Roaming/nltk_data")
from nltk.corpus import wordnet as wn

rows = json.load(open(os.path.join(ROOT, "scratch", "wn_channel_probe.json"), encoding="utf-8"))
by = collections.defaultdict(list)
for r in rows: by[(r["item"], r["subject"])].append(r)
bank = {it["item_id"]: it for it in tk.build_items()}
prim = [(i, v) for i, v in by.items() if i[0] in {it["item_id"] for it in bank.values() if it["pool"] == "primary"}]
vals = []
for (iid, s), v in prim:
    demo = [e["r2"] for e in bank[iid]["taught_exemplars"][:bank[iid]["k"]]]
    pv = []
    for a, b in zip(demo, demo[1:]):
        x = [wn.path_similarity(x, y) for x in wn.synsets(a, pos="n") for y in wn.synsets(b, pos="n")]
        pv.append(max(x) if x else None)
    good = [z for z in pv if z is not None]
    vals.append(statistics.median(good) if good else None)
good = [v for v in vals if v is not None]
print("demo-pair median path-sim: n", len(good), "mean", round(statistics.mean(good), 4),
      "deciles", [round(x, 3) for x in statistics.quantiles(good, n=10)])

cache = {}
def syn(w):
    if w not in cache: cache[w] = wn.synsets(w, pos="n")
    return cache[w]
def lcs(a, b):
    best, bd = None, -1
    for x in syn(a):
        for y in syn(b):
            try: l = x.lowest_common_hypernyms(y)[0]
            except Exception: continue
            depth = l.max_depth()
            if depth > bd: bd, best = depth, l
    return best
ck, cn = collections.Counter(), collections.Counter()
for (iid, s), v in prim:
    if max((r["path"] or 0) for r in v) < 0.2: continue
    top = max(v, key=lambda r: r["path"] or 0)
    l = lcs(top["subject"], top["cand"])
    (ck if top["key"] else cn)[l.name() if l else None] += 1
print("TOP LCS key", ck.most_common(20))
print("TOP LCS nonkey", cn.most_common(20))
