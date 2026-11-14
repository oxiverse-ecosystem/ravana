#!/usr/bin/env python3
"""Measure the UNION of the two proposition sources for the plan route.

Source A: a typed graph edge (the original route).
Source B: the stored definition, parsed for its copula (the new route).

Reports, over the eval subject set and over the stored-definition set:
  * coverage of each source alone
  * coverage of the union
  * the groundedness of the union's propositions

READ-ONLY. Usage:
    uv run python scripts/measure_proposition_union.py
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from eval_decoder_groundedness import (  # noqa: E402
    build_engine, discover_subjects, score_on_topic,
    score_unsupported_assertions,
)

from ravana.learn.definition_proposition import proposition_from_definition  # noqa: E402
from ravana.learn.linearization import (  # noqa: E402
    VACUOUS_RELATIONS, message_to_plan, plan_is_committed,
)
from ravana.learn.macroplanning import message_from_best_edge  # noqa: E402


def _neighbours(eng, subject):
    nids = eng._concept_keywords.get(subject.lower(), [])
    neigh = set()
    for nid in nids:
        for tgt, _ed in (eng.graph.get_outgoing(nid) or []):
            tn = eng.graph.get_node(tgt)
            if tn is not None and tn.label:
                neigh.add(tn.label.lower())
    return neigh


def _typed_proposition(eng, subject):
    """Source A: the strongest TYPED edge, resolved to a graph concept."""
    nids = eng._concept_keywords.get(subject.lower(), [])
    if not nids:
        return None
    cands = []
    for tgt, ed in (eng.graph.get_outgoing(nids[0]) or []):
        tn = eng.graph.get_node(tgt)
        lab = (getattr(tn, "label", "") or "").strip()
        if not lab:
            continue
        rel = getattr(ed, "relation_type", "related_to") or "related_to"
        if rel in VACUOUS_RELATIONS:
            continue
        cands.append((rel, lab, float(getattr(ed, "weight", 0.0) or 0.0),
                      float(getattr(ed, "confidence", 0.0) or 0.0)))
    if not cands:
        return None
    return message_to_plan(
        message_from_best_edge(subject, cands), subject=subject)


def _definition_proposition(eng, subject, definition, neigh):
    """Source B: the definition's copula, resolved to a graph concept."""
    p = proposition_from_definition(definition, pos_map=getattr(eng, "_concept_pos", {}))
    if p is None:
        return None
    rel, tgt = p
    tgt2 = tgt if (tgt in neigh or tgt in eng._concept_keywords) else None
    if not tgt2:
        return None
    return message_to_plan(
        message_from_best_edge(subject, [(rel, tgt2, 0.9, 0.8)]),
        subject=subject)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-subjects", type=int, default=None)
    args = ap.parse_args(argv)

    eng, label = build_engine("persisted")
    if eng is None:
        return 1
    defs = getattr(eng, "_definitions", {}) or {}
    pos = getattr(eng, "_concept_pos", {}) or {}

    # Two subject pools: the eval set, and everything with a definition.
    eval_subs = discover_subjects(eng)
    if args.max_subjects:
        eval_subs = eval_subs[:args.max_subjects]
    def_subs = sorted(defs.keys())

    print(f"[union] engine={label}")
    print(f"[union] eval subjects={len(eval_subs)}  definitions={len(def_subs)}")
    print()

    for pool_name, pool in (("EVAL", eval_subs), ("DEFINITIONS", def_subs)):
        a = b = u = 0
        on, un = [], []
        only_a = only_b = both = 0
        for s in pool:
            neigh = _neighbours(eng, s)
            pa = _typed_proposition(eng, s)
            pa = pa if plan_is_committed(pa) else None
            pb = _definition_proposition(eng, s, defs.get(s.lower()), neigh)
            pb = pb if plan_is_committed(pb) else None
            if pa:
                a += 1
            if pb:
                b += 1
            if pa and pb:
                both += 1
            elif pa:
                only_a += 1
            elif pb:
                only_b += 1
            plan = pa or pb
            if not plan:
                continue
            u += 1
            sent = " ".join(plan.tokens)
            on.append(score_on_topic(eng, s, sent)["on_topic_frac"])
            un.append(score_unsupported_assertions(eng, s, sent)
                      .get("unsupported_count", 0))
        n = len(pool)
        print(f"=== {pool_name} (n={n}) ===")
        print(f"  typed-edge route      : {a:>4} ({100*a/n:.1f}%)")
        print(f"  definition route      : {b:>4} ({100*b/n:.1f}%)")
        print(f"  union                 : {u:>4} ({100*u/n:.1f}%)")
        print(f"    both sources        : {both}")
        print(f"    typed edge only     : {only_a}")
        print(f"    definition only     : {only_b}")
        if on:
            print(f"  union on_topic={statistics.mean(on):.4f}  "
                  f"unsupported={statistics.mean(un):.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
