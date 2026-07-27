"""One-time offline builder: extract a compact semantic seed from the public
ConceptNet 5.7 English assertions file.

Output: data/semantic_seed.pkl  — {rel: [(a, b, weight), ...]}  (general world
knowledge, NOT benchmark answers). The engine loads this small file at init to
seed SemanticGraph, giving RAVANA genuine general (semantic) memory so it can
answer advice / common-sense questions it was never explicitly told.

Relation whitelist keeps only cognitively-useful edges (causal, associational,
taxonomic, property). Antonyms are kept too (opposite_of). We filter to single-
token or short English concepts to keep the seed small and the nodes matchable
to query words.

Usage:  python scripts/build_semantic_seed.py
"""

import json
import os
import pickle
import re
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(_PROJ, "data", "conceptnet", "en_assertions.tsv")
OUT = os.path.join(_PROJ, "data", "semantic_seed.pkl")

# ConceptNet relation -> our relation type
_REL_MAP = {
    "/r/IsA": "is_a",
    "/r/PartOf": "part_of",
    "/r/UsedFor": "used_for",
    "/r/HasProperty": "has_property",
    "/r/Causes": "causes",
    "/r/CausesDesire": "causes",
    "/r/RelatedTo": "related_to",
    "/r/CapableOf": "has_property",
    "/r/MotivatedByGoal": "causes",
    "/r/Desires": "has_property",
    "/r/AtLocation": "located_in",
    "/r/LocatedNear": "located_in",
    "/r/DefinedAs": "related_to",
    "/r/SimilarTo": "related_to",
    "/r/SymbolOf": "related_to",
    "/r/Antonym": "opposite_of",
    "/r/DistinctFrom": "opposite_of",
}

# Keep only concepts that are simple English words/phrases (alnum + spaces/_).
_CONCEPT_RE = re.compile(r"^/c/en/([a-z0-9_]+)(?:/n|/v|/a|/r)?$")
_SKIP = {"be", "have", "do", "make", "get", "go", "take", "thing", "something",
         "someone", "person", "people", "entity", "object", "ability",
         "able", "want", "need", "a", "an", "the", "of", "to", "in", "on"}


def _norm(cid: str):
    m = _CONCEPT_RE.match(cid)
    if not m:
        return None
    w = m.group(1).replace("_", " ")
    if w in _SKIP:
        return None
    # drop very long / multi-word noise but allow 1-3 word concepts
    toks = w.split()
    if len(toks) > 3:
        return None
    return w


def main():
    if not os.path.exists(SRC):
        print(f"MISSING {SRC}", file=sys.stderr)
        sys.exit(1)
    out = {rel: [] for rel in set(_REL_MAP.values())}
    seen = {rel: set() for rel in out}
    n = 0
    with open(SRC, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            if n % 5_000_000 == 0:
                print(f"  ...{n:,} lines, {sum(len(v) for v in out.values()):,} edges",
                      file=sys.stderr)
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            rel_raw, subj, obj, meta = parts[0], parts[1], parts[2], parts[3]
            rel = _REL_MAP.get(rel_raw)
            if rel is None:
                continue
            a = _norm(subj)
            b = _norm(obj)
            if not a or not b or a == b:
                continue
            try:
                w = float(json.loads(meta).get("weight", 1.0))
            except Exception:
                w = 1.0
            if w < 1.0:
                continue  # keep only well-supported assertions
            key = (a, b)
            if key in seen[rel]:
                continue
            seen[rel].add(key)
            out[rel].append((a, b, round(w, 3)))
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    total = sum(len(v) for v in out.values())
    print(f"WROTE {OUT}: {total:,} edges across {len(out)} relations",
          file=sys.stderr)
    for rel, v in sorted(out.items(), key=lambda kv: -len(kv[1])):
        print(f"  {rel}: {len(v):,}", file=sys.stderr)


if __name__ == "__main__":
    main()
