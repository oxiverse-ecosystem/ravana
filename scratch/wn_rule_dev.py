#!/usr/bin/env python3
"""EXPLORATORY ONLY (development split): compare subject-anchored WordNet
structural route with demo-only WordNet and text-cue controls."""
import collections, json, os, statistics, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import tra_k_items as tk
import nltk
nltk.data.path.append(r"C:/Users/Likhith/AppData/Roaming/nltk_data")
from nltk.corpus import wordnet as wn

DELTA, MARGIN, EPS = 0.10, 0.05, 1e-9
rows = json.load(open(os.path.join(ROOT, "scratch", "wn_channel_probe.json"), encoding="utf-8"))
bank = {it["item_id"]: it for it in tk.build_items()}
prim_ids = {i for i, it in bank.items() if it["pool"] == "primary"}
by = collections.defaultdict(list)
for r in rows: by[r["item"]].append(r)

def best_sim(a, b, metric):
    vals = [metric(x, y) for x in wn.synsets(a, pos="n") for y in wn.synsets(b, pos="n")]
    return max(vals) if vals else None

ALL_DEMO = [best_sim(a, b, wn.path_similarity)
            for it in bank.values()
            for a, b in zip([e["r2"] for e in it["taught_exemplars"][:it["k"]]],
                            [e["r2"] for e in it["taught_exemplars"][:it["k"]]][1:])]
ALL_DEMO = [v for v in ALL_DEMO if v is not None]
GLOBAL_THETA = statistics.median(ALL_DEMO) + DELTA
print('global theta', round(GLOBAL_THETA, 4), 'n demo pairs', len(ALL_DEMO))

def demo_theta(iid):
    it = bank[iid]
    demo = [e["r2"] for e in it["taught_exemplars"][:it["k"]]]
    vals = [best_sim(a, b, wn.path_similarity) for a, b in zip(demo, demo[1:])]
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) + DELTA if vals else GLOBAL_THETA

def decide(scores, theta):
    ranked = sorted(((c, s) for c, s in scores.items() if s is not None), key=lambda x: -x[1])
    if not ranked or theta is None: return None, "no_data"
    if ranked[0][1] < theta: return None, "below_theta"
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if ranked[0][1] - second < MARGIN: return None, "no_margin"
    return ranked[0][0], "emit"

def tally(items, arm):
    emit = cor = 0; reasons = collections.Counter()
    for iid in items:
        v = by[iid]
        if not v: continue
        s = v[0]["subject"]
        theta = demo_theta(iid)
        if arm == "struct":
            scores = {r["cand"]: r["path"] for r in v}
        else:
            demo = [e["r2"] for e in bank[iid]["taught_exemplars"][:bank[iid]["k"]]]
            scores = {r["cand"]: max([best_sim(r["cand"], d, wn.path_similarity) or 0 for d in demo]) if demo else None for r in v}
        p, why = decide(scores, theta); reasons[why] += 1
        if p is not None:
            emit += 1; cor += any(r["cand"] == p and r["key"] for r in v)
    return {"n": len(items), "emit": emit, "correct": cor,
            "precision": round(cor / emit, 4) if emit else None, "reasons": dict(reasons)}

items = sorted(i for i in prim_ids if i in by)
print("primary cases", len(items))
for arm in ("struct", "demo"):
    print(arm, tally(items, arm))
