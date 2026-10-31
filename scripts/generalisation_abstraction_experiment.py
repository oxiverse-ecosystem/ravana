#!/usr/bin/env python3
"""Generalisation and abstraction experiment for RAVANA.

Trains RAVANA on positive-affect expressions about a topic using several
distinct surface forms, then evaluates on UNSEEN constructions about the same
topic. No per-expression rules: the only signal is the shared affect/topic
pattern RAVANA extracts online. No LLM used.
"""
import os, sys, json
_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_proj, os.path.join(_proj, "ravana", "src"), os.path.join(_proj, "ravana_ml", "src"), os.path.join(_proj, "ravana-v2", "src")):
    if p not in sys.path: sys.path.insert(0, p)
os.environ.setdefault("RAVANA_OFFLINE", "1")

from ravana.chat.engine import CognitiveChatEngine

TRAIN = [
    "i love the ocean.",
    "i'm fascinated by the ocean.",
    "the ocean makes me happy.",
    "i enjoy the ocean deeply.",
    "the ocean brings me joy.",
]
EVAL = [
    "do i like the ocean?",
    "am i fond of the ocean?",
    "does the ocean please me?",
    "is the ocean dear to me?",
]

def run():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="genabsexp")
    pre = 0; pre_hits = []
    for q in EVAL:
        r = eng.process_turn(q)
        rep = str(r).lower() if r else ""
        hit = 1 if any(tok in rep for tok in ("ocean", "like", "fond", "yes", "love")) else 0
        pre_hits.append((q, str(r)[:80], hit)); pre += hit
    for t in TRAIN:
        eng.process_turn(t)
    post = 0; post_hits = []
    for q in EVAL:
        r = eng.process_turn(q)
        rep = str(r).lower() if r else ""
        hit = 1 if any(tok in rep for tok in ("ocean", "like", "fond", "yes", "love")) else 0
        post_hits.append((q, str(r)[:80], hit)); post += hit
    eng.save()
    report = {
        "n_train": len(TRAIN), "n_eval": len(EVAL),
        "pre_hits": pre, "post_hits": post,
        "pre_rate": pre/len(EVAL), "post_rate": post/len(EVAL),
        "improvement": (post-pre)/len(EVAL),
        "no_llm": True, "no_per_expression_rules": True,
        "pre": pre_hits, "post": post_hits,
    }
    out = os.path.join(_proj, "docs", "GENERALISATION_ABSTRACTION_EXPERIMENT.md")
    with open(out, "w") as f:
        f.write("# Generalisation and Abstraction Experiment\n\n")
        f.write(f"N_train={len(TRAIN)} N_eval={len(EVAL)} pre={pre}/{len(EVAL)} post={post}/{len(EVAL)} improvement={(post-pre)/len(EVAL):.3f}\n\n")
        f.write("## Pre-training eval\n")
        for q,r,h in pre_hits: f.write(f"- Q: {q}\n  R: {r}\n  hit={h}\n")
        f.write("\n## Post-training eval\n")
        for q,r,h in post_hits: f.write(f"- Q: {q}\n  R: {r}\n  hit={h}\n")
    print(json.dumps(report, indent=2))
    print("REPORT:", out)

if __name__ == "__main__":
    run()
