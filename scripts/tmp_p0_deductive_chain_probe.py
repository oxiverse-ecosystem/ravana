"""P0 decisive probe (corrected) — can LogiQA premises be extracted into
clean, chainable triples by a LOGICAL-RELATION-AWARE extractor?

Why this differs from scripts/tmp_probe_clause_yield.py:
  The earlier probe counted a "chain" ONLY as a transitive `is` link
  (operators.py:90 `if r != "is": continue`). It found 0/50. That scope
  was correct for the ClauseSegmenter go/no-go, but it is NOT proof that
  LogiQA has no chainable structure at all. LogiQA leans heavily on
  COMPARATIVE relations ("A is taller than B, B is taller than C") and
  POSITIONAL relations ("A is to the left of B") — both transitive, both
  missed by the prior scope.

This probe uses a broader extractor (comparative/positional/partitive-aware)
and counts ANY transitive chain (shared middle term under one relation), plus
reports how many cases yield >=2 usable premises at all (the blobbing floor).

Decision (mirrors the subagent plan exit criteria):
  frac_cases_with_chain >= 0.30  -> BUILD the deductive channel.
  else                              -> bottleneck is extraction, not arch.

Output: reports/deductive_p0_probe.json  (committed as evidence)
"""
from __future__ import annotations
import sys, os, re, json, argparse
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ravana", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from evaluate_ravana import _load_logiqa
from ravana.core.triplet_inference.canonical import canonical_term

# ── clause splitting (sentence + coordinating/contrastive) ─────────────
_CONJ = re.compile(
    r"\b(and|but|however|because|therefore|so|thus|whereas|although|"
    r"though|while|whereas)\b", re.IGNORECASE)

def _split_clauses(text: str) -> list:
    sents = re.split(r"(?<=[.!?])\s+", text or "")
    clauses = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        for seg in _CONJ.split(s):
            seg = seg.strip().strip(",.!?;:\"'")
            if len(seg.split()) >= 2:
                clauses.append(seg)
    return clauses

# ── quantifier / polarity prefix detection ─────────────────────────────
_FORALL = re.compile(r"^\s*(all|every|each|any|the)\b", re.IGNORECASE)
_EXISTS = re.compile(r"^\s*(some|at least one|one of|several|many|a few)\b",
                     re.IGNORECASE)
_NEG = re.compile(r"\b(not|no|never|none|isn'?t|aren'?t|cannot|can'?t|"
                  r"doesn'?t|don'?t|without|except|incorrect|false|wrong)\b",
                  re.IGNORECASE)

# Ordered relation matchers: specific (comparative/positional/partitive)
# BEFORE the generic `is`/`has` grabber, so a clause like "A is taller
# than B" binds to the comparative relation, not a blob object.
_REL_PATTERNS = [
    # comparative (transitive): A is <adj> than B
    (re.compile(r"^(.+?)\s+is\s+(.+?)\s+than\s+(.+)$", re.IGNORECASE),
     lambda m: ("compare:" + re.sub(r"[^a-z]", "", m.group(2).lower()),
                 m.group(1), m.group(3))),
    # positional: A is (to the) left/right/above/below ... of B
    (re.compile(
        r"^(.+?)\s+is\s+(?:to\s+the\s+|on\s+the\s+)?"
        r"(left|right|top|bottom|above|below|in\s*front|behind|"
        r"next\s*to|beside|inside|outside|on\s*top)\b"
        r"(?:\s+of|\s+to)?\s*(.+)$", re.IGNORECASE),
     lambda m: ("loc:" + re.sub(r"[^a-z]", "", m.group(2).lower()),
                 m.group(1), m.group(3))),
    # partitive: A is (a) part of B  /  A contains/has/includes B
    (re.compile(r"^(.+?)\s+is\s+(?:a\s+)?part\s+of\s+(.+)$", re.IGNORECASE),
     lambda m: ("partof", m.group(1), m.group(2))),
    (re.compile(r"^(.+?)\s+(contains|includes|has|holds)\s+(.+)$",
                re.IGNORECASE),
     lambda m: ("has", m.group(1), m.group(3))),
    # generic copula: A is B   (only if both terms stay short => no blob)
    (re.compile(r"^(.+?)\s+is\s+(?:a|an|the)?\s*(.+)$", re.IGNORECASE),
     lambda m: ("is", m.group(1), m.group(2))),
]

_MAX_TOK = 4  # term length cap; above this it's a blob, drop it.

def _extract_triples(clause: str):
    out = []
    q = "none"
    if _FORALL.search(clause):
        q = "forall"
    elif _EXISTS.search(clause):
        q = "exists"
    neg = bool(_NEG.search(clause))
    for pat, fn in _REL_PATTERNS:
        m = pat.match(clause)
        if not m:
            continue
        rel, s_raw, o_raw = fn(m)
        s = canonical_term(s_raw)
        o = canonical_term(o_raw)
        if not s or not o:
            continue
        if len(s.split()) > _MAX_TOK or len(o.split()) > _MAX_TOK:
            continue  # blob guard
        out.append({"s": s, "p": rel, "o": o, "q": q, "neg": neg})
        break  # first (most specific) match wins
    return out

def _chains_in(triples):
    """Return set of middle terms that join two triples into a transitive
    chain under ONE relation (A-p-B, B-p-C). Negated edges don't chain."""
    by_subj, by_obj = {}, {}
    for t in triples:
        if t["neg"]:
            continue
        by_subj.setdefault(t["s"], []).append(t)
        by_obj.setdefault(t["o"], []).append(t)
    mids = set()
    for m in set(by_subj) & set(by_obj):
        # need the two roles to share the same relation
        rels_out = {x["p"] for x in by_subj[m]}
        rels_in = {x["p"] for x in by_obj[m]}
        if rels_out & rels_in:
            mids.add(m)
    return mids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=50)
    ap.add_argument("--out", default="reports/deductive_p0_probe.json")
    args = ap.parse_args()

    cases = _load_logiqa(max_cases=args.cases)
    if not cases:
        print("NO LOGIQA CASES LOADED")
        return

    per = []
    n_chain = 0
    n_prem = 0
    n_forall = 0
    total_prem = 0
    rel_counter = Counter()
    for i, c in enumerate(cases[:args.cases]):
        q = c["question"]
        text = q.split("\n\nWhich is the correct answer")[0]
        triples = []
        for clause in _split_clauses(text):
            triples.extend(_extract_triples(clause))
        triples = [t for t in triples if t["s"] != t["o"]]
        mids = _chains_in(triples)
        if triples:
            n_prem += 1
        total_prem += len(triples)
        for t in triples:
            rel_counter[t["p"]] += 1
            if t["q"] == "forall":
                n_forall += 1
        if mids:
            n_chain += 1
        if i < 8:
            per.append({"idx": i, "n_prem": len(triples),
                        "n_chain": len(mids),
                        "sample": triples[:4]})

    n = len(cases[:args.cases])
    frac_chain = n_chain / n if n else 0
    frac_prem = n_prem / n if n else 0
    result = {
        "n_cases": n,
        "extractor": "logical-relation-aware (comparative/positional/partitive/is)",
        "cases_with_any_premise": n_prem,
        "frac_cases_with_premise": round(frac_prem, 3),
        "cases_with_transitive_chain": n_chain,
        "frac_cases_with_chain": round(frac_chain, 3),
        "total_premises_extracted": total_prem,
        "avg_premises_per_case": round(total_prem / n, 2) if n else 0,
        "forall_premise_count": n_forall,
        "relation_frequency": dict(rel_counter.most_common()),
        "decision_threshold": ">=0.30 frac with transitive chain",
        "go_build": frac_chain >= 0.30,
        "per_case_sample": per,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"=== P0 deductive-chain probe ({n} LogiQA cases) ===")
    print(f"cases with >=1 extracted premise : {n_prem}/{n} ({frac_prem:.1%})")
    print(f"cases with a transitive chain   : {n_chain}/{n} ({frac_chain:.1%})")
    print(f"total premises extracted        : {total_prem}")
    print(f"forall-marked premises          : {n_forall}")
    print(f"top relations                  : {rel_counter.most_common(8)}")
    print(f"GO_BUILD_DEDUCTIVE_CHANNEL     : {result['go_build']}")
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()
