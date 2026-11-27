#!/usr/bin/env python3
"""EXPLORATORY ONLY (development split): head-noun language cue from the
subject's OWN Wikidata description, admitted by WordNet superordinate structure."""
import collections, json, os, re, statistics, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import tra_k_items as tk
import nltk
nltk.data.path.append(r"C:/Users/Likhith/AppData/Roaming/nltk_data")
from nltk.corpus import wordnet as wn

STOP = frozenset("a an the of in on at to for by with is are was were be been being that which who from as or and not it its this these those one two into about over under than then there their they them he she his her we our you your i me my can may will would should could has have had do does did very more most some any each other another same such only also term used refers referring type kind class something someone".split())
rows = json.load(open(os.path.join(ROOT, "scratch", "wn_channel_probe.json"), encoding="utf-8"))
bank = {it["item_id"]: it for it in tk.build_items()}
prim = {i for i, it in bank.items() if it["pool"] == "primary"}
by = collections.defaultdict(list)
for r in rows: by[r["item"]].append(r)

def head_noun(text):
    words = [w for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) > 3 and w not in STOP]
    for w in reversed(words):
        if wn.synsets(w, pos="n"): return w
    return None

def sim(a, b, metric=wn.path_similarity):
    vals = [metric(x, y) for x in wn.synsets(a, pos="n") for y in wn.synsets(b, pos="n")]
    return max(vals) if vals else None

stats = collections.Counter()
for arm in ("subject", "head", "max_both", "head_gate"):
    emit = cor = 0
    for iid in sorted(i for i in prim if i in by):
        v = by[iid]; subj = v[0]["subject"]
        head = head_noun(v[0]["text"])
        for r in v:
            r["ps"] = r["path"]
            r["hs"] = sim(head, r["cand"]) if head else None
        def decide(scores, theta=0.2667, margin=0.05):
            ranked = sorted(((c, s) for c, s in scores.items() if s is not None), key=lambda x: -x[1])
            if not ranked or ranked[0][1] < theta: return None
            if ranked[0][1] - (ranked[1][1] if len(ranked)>1 else 0) < margin: return None
            return ranked[0][0]
        if arm == "subject": p = decide({r["cand"]: r["ps"] for r in v})
        elif arm == "head": p = decide({r["cand"]: r["hs"] for r in v})
        elif arm == "max_both": p = decide({r["cand"]: max(x for x in (r["ps"], r["hs"]) if x is not None) if (r["ps"] is not None or r["hs"] is not None) else None for r in v})
        else:
            p = decide({r["cand"]: r["hs"] if (r["hs"] is not None and r["hs"] >= 0.25) else r["ps"] for r in v})
        if p:
            emit += 1; cor += any(r["cand"] == p and r["key"] for r in v)
    print(arm, "emit", emit, "correct", cor, "precision", round(cor/emit, 4) if emit else None)
