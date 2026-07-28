"""Phase 2 (section 6.4 plan): ClauseSegmenter go/no-go probe.

GOAL: answer empirically whether prose (LogiQA) can be segmented into
clean SPO chains that the triplet operator could reason over. This is
the question the opencode research study COULD NOT answer — it is a
measurement, not a library call. The probe is throwaway (committed as
evidence, not shipped).

Method (faithful to plan 2.1): for each of N LogiQA cases, take the
CONTEXT + QUESTION text, then:
  1. sentence-split, then split on coordination/contrast markers
     (and/but/however/because/therefore), as the plan specifies.
  2. apply the EXISTING extractors per clause:
       - parse_universal_edges (in_prompt_reasoner)
       - PropositionParser.extract_propositions (with the SAME
         blob-subject filter used in _triplet_mc_answer:
         len(s.split())<=4 and len(o.split())<=4)
  3. measure clean-triples-per-case AND whether premise CHAINS appear
     (a shared middle term linking two triples) — chains are what the
     operator needs; isolated triples cannot produce an inference.

Decision (plan 2.3):
  yield >= ~2 chained premises/case on >= 30% of cases -> BUILD it.
  below that -> DON'T build; commit negative finding and STOP.

We count a "clean triple" as one whose subject and object are each a
short noun phrase (<=4 tokens, no sentence-blob residue) — i.e. the
same quality bar the candidate's matching requires to ever fire.
A "chain" requires two triples sharing a middle term (A-r-B, B-r2-C)
with the same canonical predicate family OR a transitive 'is' link,
since the operator's transitivity gate is what we can actually exploit.
"""
from __future__ import annotations
import sys, os, re, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ravana", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from evaluate_ravana import _load_logiqa
from ravana.core.in_prompt_reasoner import parse_universal_edges
from ravana.core.proposition_parser import PropositionParser
from ravana.core.triplet_inference.canonical import (
    canonical_term, canonical_predicate)

_PARSER = PropositionParser()
_CONJ = re.compile(r"\b(and|but|however|because|therefore|so|thus|"
                  r"whereas|although|though|while)\b", re.IGNORECASE)


def _split_clauses(text: str) -> list:
    # sentence split (simple)
    sents = re.split(r"(?<=[.!?])\s+", text or "")
    clauses = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        # split coordinating/contrastive markers inside the sentence
        parts = _CONJ.split(s)
        # _CONJ.split yields [seg, marker, seg, marker, seg, ...]
        for seg in parts:
            seg = seg.strip().strip(",.!?;:")
            if len(seg.split()) >= 2:
                clauses.append(seg)
    return clauses


def _clean_triples_from_clause(clause: str) -> list:
    out = []
    # (1) universal / instance edges
    universals, instances = parse_universal_edges(clause)
    for a, b in universals + instances:
        if a and b and a != b:
            out.append((canonical_term(a), "is", canonical_term(b)))
    # (2) PropositionParser, with the candidate's blob-subject filter
    for p in (_PARSER.extract_propositions(clause) or []):
        s = canonical_term(getattr(p, "subject", "") or "")
        o = canonical_term(getattr(p, "object", "") or "")
        r = canonical_predicate(getattr(p, "predicate", "") or "")
        if s and o and r and len(s.split()) <= 4 and len(o.split()) <= 4:
            out.append((s, r, o))
    return out


def _shared_middle(triples: list) -> set:
    """Return set of middle terms that link two triples into a chain
    (A-r-B, B-r2-C), restricted to 'is' transitivity chains which the
    operator can actually exploit."""
    by_subj = {}
    by_obj = {}
    for s, r, o in triples:
        if r != "is":
            continue
        by_subj.setdefault(s, []).append((s, r, o))
        by_obj.setdefault(o, []).append((s, r, o))
    mids = set()
    for m in set(by_subj) & set(by_obj):
        mids.add(m)
    return mids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=50)
    ap.add_argument("--out", default="reports/clause_yield_probe.json")
    args = ap.parse_args()

    cases = _load_logiqa(max_cases=args.cases)
    if not cases:
        print("NO LOGIQA CASES LOADED")
        return

    per_case = []
    n_chain_cases = 0
    total_triples = 0
    total_chain_triples = 0
    for i, c in enumerate(cases[:args.cases]):
        q = c["question"]
        # context + question (skip the instruction tail)
        text = q.split("\n\nWhich is the correct answer")[0]
        triples = []
        for clause in _split_clauses(text):
            triples.extend(_clean_triples_from_clause(clause))
        mids = _shared_middle(triples)
        has_chain = len(mids) > 0
        if has_chain:
            n_chain_cases += 1
            total_chain_triples += len(triples)
        total_triples += len(triples)
        per_case.append({
            "idx": i,
            "n_triples": len(triples),
            "has_chain": has_chain,
            "sample": triples[:4],
        })

    n = len(per_case)
    frac_chain = n_chain_cases / n if n else 0
    avg_triples = total_triples / n if n else 0

    result = {
        "n_cases": n,
        "avg_clean_triples_per_case": round(avg_triples, 2),
        "cases_with_chain": n_chain_cases,
        "frac_cases_with_chain": round(frac_chain, 3),
        "decision_threshold": ">=0.30 frac with chain AND >=2 chained "
                              "triples/case",
        "go_build": frac_chain >= 0.30 and (total_chain_triples
                                            / max(n_chain_cases, 1)) >= 2,
        "per_case": per_case[:10],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"=== Clause-yield probe ({n} LogiQA cases) ===")
    print(f"avg clean triples/case : {avg_triples:.2f}")
    print(f"cases with a transitive 'is' chain: {n_chain_cases}/{n} "
          f"({frac_chain:.1%})")
    print(f"GO_BUILD_SEGMENTER: {result['go_build']}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
