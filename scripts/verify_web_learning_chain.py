#!/usr/bin/env python3
"""Verify the web -> graph -> plan chain end to end.

PR #378 fixed `WebToGraph.RELATION_TO_EDGE["causes"]` -> `"causal"`. A mapping
change is only worth something if the whole chain runs, so this drives the real
classes and asserts each hop:

    web text -> OpenIE -> GraphEngine edge -> relation_type
             -> message_from_best_edge -> message_to_plan -> committed plan

READ-ONLY (builds a throwaway in-memory graph). Usage:
    uv run python scripts/verify_web_learning_chain.py
"""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TEXT = "Smoking causes lung cancer. Water is a liquid."


def main():
    from ravana.graph.engine import GraphEngine
    from ravana.web.web_to_graph import WebToGraph
    from ravana_ml.graph import ConceptEdge
    from ravana.learn.linearization import (
        VACUOUS_RELATIONS, message_to_plan, plan_is_committed)
    from ravana.learn.macroplanning import message_from_best_edge

    g = GraphEngine()
    w = WebToGraph(g, source="verify")
    n = w.learn_text(TEXT, source_url="verify")
    print(f"[chain] facts learned: {n}")

    edges = list(g.graph.edges.values())
    types = {}
    for e in edges:
        t = getattr(e, "relation_type", "?")
        types[t] = types.get(t, 0) + 1
    print(f"[chain] edges: {len(edges)}  relation types: {types}")

    known = set(ConceptEdge.RELATION_DEFAULT_WEIGHTS)
    try:
        from ravana.ontology.graph_typing import TYPED_RELATION_TYPES
        known |= set(TYPED_RELATION_TYPES)
    except Exception:
        pass
    for t in sorted(types):
        print(f"[chain]   {t:<14} recognised: {t in known}")

    # Feed a causal edge through the plan route.
    causal = [e for e in edges if e.relation_type == "causal"]
    if not causal:
        print("[chain] FAIL: no causal edge produced")
        return 1
    e = causal[0]
    print(f"[chain] causal edge weight={getattr(e, 'weight', 0):.2f} "
          f"vacuous={e.relation_type in VACUOUS_RELATIONS}")

    # resolve the target label -- ConceptEdge uses `source`/`target` node ids
    tgt = getattr(e, "target", None)
    node = g.graph.get_node(tgt) if tgt is not None else None
    label = (getattr(node, "label", "") or "") if node is not None else ""
    if not label:
        print("[chain] FAIL: could not resolve causal target label")
        return 1

    cands = [("causal", label, float(getattr(e, "weight", 0.0) or 0.0), 0.8)]
    plan = message_to_plan(message_from_best_edge("smoking", cands),
                           subject="smoking")
    print(f"[chain] plan tokens: {plan.tokens if plan else None}")
    print(f"[chain] committed: {plan_is_committed(plan)}")

    ok = (plan is not None and plan_is_committed(plan)
          and "causal" in known
          and e.relation_type not in VACUOUS_RELATIONS)
    print(f"[chain] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
