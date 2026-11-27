#!/usr/bin/env python3
"""EXPLORATORY ONLY (not the pre-registered run): is there any WordNet
subject<->candidate structural signal for the independently derived
Wikidata-type-sharing key? No engine, no store, no network, no production edit.
"""
from __future__ import annotations
import collections, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import tra_k_items  # unmodified, read-only
import nltk
nltk.data.path.append(r"C:/Users/Likhith/AppData/Roaming/nltk_data")
from nltk.corpus import wordnet as wn

STOP = frozenset("a an the of in on at to for by with is are was were be been being that which who from as or and not it its this these those one two into about over under than then there their they them he she his her we our you your i me my can may will would should could has have had do does did very more most some any each other another same such only also term used refers referring type kind class something someone".split())

def norm(t): return re.sub(r"[^a-z0-9 ]", " ", str(t).lower()).strip()

def main():
    sub = json.load(open(os.path.join(ROOT, "scratch", "trak_substrate.json"), encoding="utf-8"))["facts"]
    bys = collections.defaultdict(list)
    for f in sub: bys[f["subject"]].append(f)
    def primary(w): return [f for f in bys.get(w, ()) if f.get("sense") == "primary"]
    def labels(w): return frozenset(norm(f["type_label"]) for f in primary(w)) - {norm(w)}
    def text(w):
        for f in primary(w):
            if f.get("wikidata_property") == "P279" and f.get("text_evidence_kind") == "item_description" and f.get("text_evidence"):
                return norm(f["text_evidence"])
        return ""
    def key(s, cands):
        st = labels(s)
        if not st: return None
        hits = [c for c in cands if labels(c) & st]
        return hits[0] if len(hits) == 1 else None
    bank = tra_k_items.build_items()
    cases = []
    for it in bank:
        for cond in ("near", "far"):
            q = it["unseen_test_words"][cond]
            s, c = q["subject"], tuple(q["candidates"])
            cases.append((it["item_id"], it["pool"], s, c, key(s, c), text(s)))
    graded = [x for x in cases if x[4] is not None]
    print("cases", len(cases), "graded", len(graded))
    # cache synsets
    cache = {}
    def ss(w):
        if w not in cache: cache[w] = wn.synsets(w, pos=wn.NOUN)
        return cache[w]
    def sim(a, b, fn):
        try:
            vals = [fn(x, y) for x in ss(a) for y in ss(b)]
            return max(vals) if vals else None
        except Exception:
            return None
    rows = []
    for iid, pool, s, cands, k, tx in graded:
        for c in cands:
            rows.append({"item": iid, "pool": pool, "subject": s, "cand": c,
                         "key": c == k, "text": tx,
                         "wup": sim(s, c, wn.wup_similarity),
                         "path": sim(s, c, wn.path_similarity),
                         "lch": sim(s, c, wn.lch_similarity)})
    with open(os.path.join(ROOT, "scratch", "wn_channel_probe.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    for metric in ("wup", "path", "lch"):
        for subset_name, subrows in (("all", rows), ("primary", [r for r in rows if r["pool"] == "primary"])):
            hit = [r for r in subrows if r[metric] is not None]
            correct = sum(1 for r in hit if r["key"])
            tot = len(hit)
            # top-1 per case
            bycase = collections.defaultdict(list)
            for r in hit: bycase[(r["item"], r["subject"])].append(r)
            top = sum(1 for v in bycase.values() if max(v, key=lambda r: r[metric])["key"])
            ties = sum(1 for v in bycase.values() if sum(r[metric] == max(x[metric] for x in v) for r in v) > 1)
            pos = [r[metric] for r in hit if r["key"]]; neg = [r[metric] for r in hit if not r["key"]]
            import statistics
            print(metric, subset_name, "n", tot, "key", len(pos), "nonkey", len(neg),
                  "mean_key", round(statistics.mean(pos), 4) if pos else None,
                  "mean_nonkey", round(statistics.mean(neg), 4) if neg else None,
                  "top1_correct", top, "/", len(bycase), "ties", ties)

if __name__ == "__main__":
    main()
