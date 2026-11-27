#!/usr/bin/env python3
"""Boot-free: is there a graded, engine-text-derived familiarity signal?"""
import json, os, sys, statistics, collections
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
for p in (os.path.join(PROJ, "scripts"), os.path.join(PROJ, "ravana", "src"), PROJ):
    if p not in sys.path: sys.path.insert(0, p)
import measure_learn5_wfa as w
items = {it["id"]: it for it in w.build_items()}
s = json.load(open(os.path.join(PROJ, "..", "ravana", "scratch", "learn5_wfa_summary.json"), encoding="utf-8"))
rows = s["pd_audit"]

def feats(r):
    toks = [t for t in w._words(items[r["id"]]["claim"]) if len(t) > 3]
    o = {}
    for tk, lab in (("pd_inclusion_text", "inc"), ("pd_exclusion_text", "exc"), ("pd_forced_choice_text", "fc")):
        txt = (r.get(tk) or "").lower()
        o["ov_" + lab] = round(sum(t in txt for t in toks) / max(1, len(toks)), 3)
        o["len_" + lab] = len(txt.split())
    o["echo_inc"] = bool(r.get("pd_inclusion_echo_marker"))
    return o

for kind in ("gap", "decoy"):
    fs = [feats(r) for r in rows if r["kind"] == kind]
    print(kind, "n", len(fs), {k: round(statistics.mean(f[k] for f in fs), 3) for k in fs[0]})
for lab in ("inc", "exc"):
    vals = [feats(r)["ov_" + lab] for r in rows]
    print(lab, "dist", collections.Counter(vals).most_common(8))
