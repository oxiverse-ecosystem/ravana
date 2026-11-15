#!/usr/bin/env python3
"""Measure the explanation-chain assembler on the live graph.

Questions:
  1. How many subjects have ANY 2-4 hop causal chain?
  2. What does the adequacy gate say for each (assert / hedge / decline)?
  3. Does the chain text look like an explanation?

READ-ONLY. Usage:
    uv run python scripts/measure_explanation_chains.py
"""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    os.environ["RAVANA_OFFLINE"] = "1"
    from ravana.chat.engine import CognitiveChatEngine
    from ravana.learn.explanation import (
        CAUSAL_RELATIONS, enumerate_chains, grade_chain)

    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    g = eng.graph

    # How many causal edges exist at all?
    causal_edges = 0
    total_edges = 0
    for nid in list(g.nodes.keys() or []):
        for _oid, e in (g.get_outgoing(nid) or []):
            total_edges += 1
            if (getattr(e, "relation_type", "") or "") in CAUSAL_RELATIONS:
                causal_edges += 1
    print(f"[exp] graph nodes={len(g.nodes)} edges={total_edges} "
          f"causal={causal_edges}")

    subjects = sorted((getattr(g.get_node(n), "label", "") or "").lower()
                      for n in list(g.nodes.keys())[:400])
    subjects = [s for s in subjects if s]
    print(f"[exp] subjects scanned: {len(subjects)}")

    found = 0
    verdicts = {"assert": 0, "hedge": 0, "decline": 0}
    samples = []
    for s in subjects:
        nids = eng._concept_keywords.get(s, [])
        if not nids:
            continue
        chains = enumerate_chains(nids[:1], g, direction="bidirectional",
                                  min_steps=2, max_steps=4, limit=4)
        if not chains:
            continue
        found += 1
        best = chains[0]
        grade = grade_chain(best)
        verdicts[grade["verdict"]] += 1
        if len(samples) < 10:
            samples.append((s, best, grade))

    print(f"[exp] subjects with a 2-4 hop causal chain: {found}")
    print(f"[exp] verdicts: {verdicts}")
    print()
    for s, ch, gr in samples:
        print(f"  {s:<16} {' -> '.join(ch.labels)[:62]}")
        print(f"  {'':<16} score={ch.score:.3f} margin={ch.margin:.3f} "
              f"adequacy={gr['adequacy']:.3f} -> {gr['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
