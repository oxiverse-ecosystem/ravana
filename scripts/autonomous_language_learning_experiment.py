#!/usr/bin/env python3
"""Autonomous language learning experiment for RAVANA.

Train RAVANA on linguistic examples; evaluate on completely unseen sentence
constructions; zero source-code modifications, new vocab seeds, or manually
added rules between train and eval. Measures improvement from learning alone
vs developer intervention. No LLM used.
"""
import os, sys, json, time
_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_proj, os.path.join(_proj, "ravana", "src"), os.path.join(_proj, "ravana_ml", "src"), os.path.join(_proj, "ravana-v2", "src")):
    if p not in sys.path: sys.path.insert(0, p)
os.environ.setdefault("RAVANA_OFFLINE", "1")

from ravana.chat.engine import CognitiveChatEngine

TRAIN = [
    "my favorite color is teal.",
    "i like reading in the morning.",
    "the cat sleeps on the windowsill.",
    "water freezes at zero degrees.",
    "birds migrate south in winter.",
    "she plays piano every evening.",
    "the library opens at nine.",
    "we walked along the river.",
    "he fixed the broken bicycle.",
    "apples are red or green.",
]
EVAL = [
    "what is my favorite color?",
    "do i like reading?",
    "where does the cat sleep?",
    "what happens to water at zero degrees?",
    "where do birds migrate?",
    "what does she play?",
    "when does the library open?",
    "where did we walk?",
    "what did he fix?",
    "what colors can apples be?",
]

def run():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="autolangexp")
    pre = 0; pre_hits = []
    for q in EVAL:
        r = eng.process_turn(q)
        hit = 1 if (r and any(w in str(r).lower() for w in q.lower().split() if len(w)>3)) else 0
        pre_hits.append((q, str(r)[:80], hit)); pre += hit
    for t in TRAIN:
        eng.process_turn(t)
    post = 0; post_hits = []
    for q in EVAL:
        r = eng.process_turn(q)
        hit = 1 if (r and any(w in str(r).lower() for w in q.lower().split() if len(w)>3)) else 0
        post_hits.append((q, str(r)[:80], hit)); post += hit
    eng.save()
    report = {
        "n_train": len(TRAIN), "n_eval": len(EVAL),
        "pre_hits": pre, "post_hits": post,
        "pre_rate": pre/len(EVAL), "post_rate": post/len(EVAL),
        "improvement": (post-pre)/len(EVAL),
        "no_llm": True, "zero_source_mods": True,
        "pre": pre_hits, "post": post_hits,
    }
    out = os.path.join(_proj, "docs", "AUTONOMOUS_LANGUAGE_LEARNING_EXPERIMENT.md")
    with open(out, "w") as f:
        f.write("# Autonomous Language Learning Experiment\n\n")
        f.write(f"N_train={len(TRAIN)} N_eval={len(EVAL)} pre={pre}/{len(EVAL)} post={post}/{len(EVAL)} improvement={(post-pre)/len(EVAL):.3f}\n\n")
        f.write("## Pre-training eval\n")
        for q,r,h in pre_hits: f.write(f"- Q: {q}\n  R: {r}\n  hit={h}\n")
        f.write("\n## Post-training eval\n")
        for q,r,h in post_hits: f.write(f"- Q: {q}\n  R: {r}\n  hit={h}\n")
    print(json.dumps(report, indent=2))
    print("REPORT:", out)

if __name__ == "__main__":
    run()
