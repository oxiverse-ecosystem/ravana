#!/usr/bin/env python3
"""Independent test of online relation typing.

WHY THIS EXISTS: the earlier accuracy number (75.5% recovery on "known typed
edges") is CIRCULAR. `_correct_relation_types` (chain_walker.py:1479) re-labels
edges at seed time using the SAME prototype mechanism that
`_infer_relation_type` uses, so the labels are partly produced by the thing
being measured. The persisted graph confirms it: isa=0, capable_of=0,
part_of=1 -- essentially no independent ConceptNet-typed edges survive.

So this script scores the mechanism against a HAND-LABELLED set of
(subject, target, relation) triples that no code produced. The labels are
authored here, once, from the plain meaning of the words.

READ-ONLY. Usage:
    uv run python scripts/measure_typing_independent.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from eval_decoder_groundedness import build_engine  # noqa: E402

# Hand-labelled (subject, target, relation_type). Relation vocabulary is
# RAVANA's own. These are ordinary English factual relations, labelled by
# reading the two words -- not derived from any vector or prototype.
LABELLED = [
    # categorize / is_a
    ("dog", "mammal", "is_a"),
    ("robin", "bird", "is_a"),
    ("oak", "tree", "is_a"),
    ("violin", "instrument", "is_a"),
    ("wheat", "grain", "is_a"),
    ("salmon", "fish", "is_a"),
    ("copper", "metal", "is_a"),
    # part_of
    ("wheel", "car", "part_of"),
    ("leaf", "tree", "part_of"),
    ("finger", "hand", "part_of"),
    ("chapter", "book", "part_of"),
    # has_property
    ("ice", "cold", "has_property"),
    ("glass", "fragile", "has_property"),
    ("honey", "sweet", "has_property"),
    ("steel", "strong", "has_property"),
    # capable_of
    ("bird", "fly", "capable_of"),
    ("fish", "swim", "capable_of"),
    # used_for
    ("knife", "cut", "used_for"),
    ("pen", "write", "used_for"),
    # causes
    ("fire", "smoke", "causes"),
    ("rain", "flood", "causes"),
    ("virus", "disease", "causes"),
    ("wind", "erosion", "causes"),
    # temporal
    ("spring", "summer", "temporal"),
    ("seed", "sapling", "temporal"),
    ("lightning", "thunder", "temporal"),
    # analogical
    ("heart", "pump", "analogical"),
    ("brain", "computer", "analogical"),
    ("atom", "solar_system", "analogical"),
    # contrastive
    ("hot", "cold", "contrastive"),
    ("light", "dark", "contrastive"),
    ("war", "peace", "contrastive"),
]


def main():
    eng, label = build_engine("persisted")
    if eng is None:
        return 1
    cw = eng
    infer = getattr(cw, "_infer_relation_type", None)
    if infer is None:
        print("[typing] no _infer_relation_type on chain_walker")
        return 1

    # Which of our relation names does the mechanism even know about?
    protos = getattr(cw, "_relation_prototypes", None) or {}
    if not protos:
        try:
            cw._init_relation_prototypes()
            protos = getattr(cw, "_relation_prototypes", {}) or {}
        except Exception:
            protos = {}
    print(f"[typing] engine={label}")
    print(f"[typing] prototype inventory: {sorted(protos.keys())}")
    print(f"[typing] labelled pairs: {len(LABELLED)}")
    print()

    # Map our labels onto the mechanism's vocabulary so the comparison is fair.
    alias = {"is_a": "isa", "part_of": "part_of", "has_property": "has_property",
             "capable_of": "capable_of", "used_for": "used_for",
             "causes": "causal", "temporal": "temporal",
             "analogical": "analogical", "contrastive": "contrastive"}

    by_rel = {}
    for subj, tgt, rel in LABELLED:
        try:
            pred, conf = infer(subj, tgt)
        except Exception as e:
            pred, conf = f"ERR:{type(e).__name__}", 0.0
        by_rel.setdefault(rel, []).append((subj, tgt, pred, conf))

    print(f"{'relation':<14} {'n':>3} {'correct':>8} {'mean_conf':>10}  examples")
    print("-" * 78)
    tot = cor = 0
    for rel, rows in by_rel.items():
        want = alias.get(rel, rel)
        ok = sum(1 for _, _, p, _ in rows if p == want)
        mc = sum(c for _, _, _, c in rows) / len(rows)
        tot += len(rows)
        cor += ok
        ex = ", ".join(f"{s}->{t}={p}" for s, t, p, _ in rows[:3])
        print(f"{rel:<14} {len(rows):>3} {ok:>4}/{len(rows):<4} {mc:>10.3f}  {ex}")
    print("-" * 78)
    print(f"{'TOTAL':<14} {tot:>3} {cor:>4}/{tot:<4} "
          f"({cor/tot:.3f})" if tot else "no data")
    print()
    print("Labels use the graph's own typed vocabulary (isa, has_property,")
    print("capable_of, used_for, part_of) plus the four learned types.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
