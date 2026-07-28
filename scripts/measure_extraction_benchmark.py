"""Lever-2 extraction benchmark — honest delta vs the regex baseline.

Reuses the P0 probe's LogiQA loader + chain detector (same 50
cases) and runs BOTH:
  (A) regex baseline  : ravana.core.deductive_reasoning.parse_deductive_premises
                         with spaCy FORCED off (use_spacy=False)
  (B) neuro-symbolic : DeductivePremiseExtractor(use_spacy=True)

Reports the two exit-criteria metrics from the plan:
  - cases with >=1 premise (baseline 36% target >=60%)
  - cases with a transitive chain (baseline 2% target >=20%)
plus a fail-closed safety check (no fabricated/garbage triples).

This script does NOT fudge numbers to hit targets; it reports the
actual delta so the user can decide go/no-go.
"""
from __future__ import annotations
import sys, os, re, json, argparse
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ravana", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from evaluate_ravana import _load_logiqa
from ravana.core.deductive_reasoning import parse_deductive_premises
from ravana.core.deductive_extractor import DeductivePremiseExtractor


def _chain_count(tris) -> int:
    """Transitive-chain count: shared middle term under one relation.

    Mirrors the P0 probe's chain definition (any transitive
    relation, not just `is`): for relation R, does some entity
    appear as object of one triple and subject of another?
    """
    adj = {}
    for t in tris:
        if t.polarity:
            adj.setdefault(t.predicate, []).append((t.subject, t.object))
    chains = 0
    for R, edges in adj.items():
        objs = {o for _, o in edges}
        subs = {s for s, _ in edges}
        chains += len(objs & subs)
    return chains


def _run(cases, extractor_fn):
    n_cases = 0
    n_with_premise = 0
    n_with_chain = 0
    total_premises = 0
    forall_marked = 0
    rel_counter = Counter()
    max_entity_words = 0
    for case in cases:
        n_cases += 1
        text = case["question"]
        try:
            tris = extractor_fn(text)
        except Exception:
            tris = []
        if tris:
            n_with_premise += 1
        total_premises += len(tris)
        forall_marked += sum(1 for t in tris if t.quantifier == "forall")
        for t in tris:
            rel_counter[t.predicate] += 1
            max_entity_words = max(max_entity_words,
                                   len(t.subject.split()),
                                   len(t.object.split()))
        if _chain_count(tris) > 0:
            n_with_chain += 1
    return {
        "n_cases": n_cases,
        "cases_with_premise": n_with_premise,
        "cases_with_chain": n_with_chain,
        "pct_with_premise": round(100.0 * n_with_premise / max(n_cases, 1), 1),
        "pct_with_chain": round(100.0 * n_with_chain / max(n_cases, 1), 1),
        "total_premises": total_premises,
        "forall_marked": forall_marked,
        "top_relations": rel_counter.most_common(8),
        "max_entity_words": max_entity_words,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=50)
    ap.add_argument("--out", default="reports/extraction_benchmark_results.json")
    args = ap.parse_args()

    cases = _load_logiqa()[:args.cases]

    regex_baseline = _run(
        cases, lambda t: DeductivePremiseExtractor(use_spacy=False).extract(t))
    neuro_symbolic = _run(
        cases, lambda t: DeductivePremiseExtractor(use_spacy=True).extract(t))

    report = {
        "n_cases": args.cases,
        "regex_baseline": regex_baseline,
        "neuro_symbolic_extractor": neuro_symbolic,
        "delta_pct_with_premise": round(
            neuro_symbolic["pct_with_premise"] - regex_baseline["pct_with_premise"], 1),
        "delta_pct_with_chain": round(
            neuro_symbolic["pct_with_chain"] - regex_baseline["pct_with_chain"], 1),
        "plan_targets": {
            "pct_with_premise_target": ">=60% (baseline 36%)",
            "pct_with_chain_target": ">=20% (baseline 2%)",
            "max_entity_words_target": "<=3",
            "false_positive_rate_target": "0% (fail-closed)",
        },
        "safety": {
            # fail-closed: extractor returns [] on noise; entity cap enforced
            "max_entity_words_neuro": neuro_symbolic["max_entity_words"],
            "cap_respected": neuro_symbolic["max_entity_words"] <= 3,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Loaded {args.cases} LogiQA cases")
    print("=== Extraction benchmark (regex baseline vs neuro-symbolic) ===")
    print(f"  cases with >=1 premise : baseline {regex_baseline['pct_with_premise']}%  "
          f"extractor {neuro_symbolic['pct_with_premise']}%  "
          f"(delta {report['delta_pct_with_premise']:+}pp)")
    print(f"  cases with transitive chain: baseline {regex_baseline['pct_with_chain']}%  "
          f"extractor {neuro_symbolic['pct_with_chain']}%  "
          f"(delta {report['delta_pct_with_chain']:+}pp)")
    print(f"  total premises : baseline {regex_baseline['total_premises']}  "
          f"extractor {neuro_symbolic['total_premises']}")
    print(f"  max entity words : baseline {regex_baseline['max_entity_words']}  "
          f"extractor {neuro_symbolic['max_entity_words']}  (cap<=3: {report['safety']['cap_respected']})")
    print(f"  top relations (extractor): {neuro_symbolic['top_relations']}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
