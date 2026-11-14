"""Which condition of _sm_response_grounded is doing the withholding?

The gate withholds ~50% of subjects. Reading the conditions is not enough to
know which one binds, so this instrument replicates the gate's decision steps
and counts the first failing condition per subject.

Usage:
    uv run python scripts/diagnose_gate_withholding.py [--limit N]
"""
import argparse
import collections
import os
import re
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CORPUS = os.path.join(PROJ, "data", "corpora", "grounded_descriptions.txt")

# Mirror of the gate's generic-pronoun set (response_gen._sm_response_grounded).
_GENERIC = {
    "other", "others", "the former", "the latter", "former", "latter",
    "one", "ones", "something", "anything", "everything", "nothing",
    "this", "that", "these", "those", "it", "they", "them", "the same",
    "another", "such", "which", "what", "who", "whom", "whose",
}


def load_subjects(path, limit=0):
    subs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "\t" not in line:
                continue
            c = line.split("\t", 1)[0].strip()
            if c:
                subs.append(c)
    return subs[:limit] if limit else subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    os.environ.pop("RAVANA_OFFLINE", None)
    from ravana.chat.engine import CognitiveChatEngine

    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    subs = load_subjects(args.corpus, args.limit)
    print(f"[gate] {len(subs)} subjects")

    fails = collections.Counter()
    for s in subs:
        reason = diagnose(eng, s)
        fails[reason] += 1

    total = sum(fails.values())
    print("\n[gate] first failing condition (n=%d):" % total)
    for reason, n in fails.most_common():
        print("  %-46s %5d  %5.1f%%" % (reason, n, 100.0 * n / max(total, 1)))


def diagnose(eng, subject):
    """Replicate the gate's step-1 decision and name the first failure."""
    if not subject:
        return "no subject"
    if subject in _GENERIC or subject.strip(" .") in _GENERIC:
        return "generic pronoun subject"

    defs = getattr(eng, "_definitions", {})
    srcs = getattr(eng, "_concept_sources", {})
    kws = getattr(eng, "_concept_keywords", {})
    labs = getattr(eng, "_concept_labels", {})

    if subject in defs:
        return "PASS (definition)"
    if subject in srcs:
        return "PASS (concept source)"
    known = subject in kws or subject in labs
    if not known:
        return "no graph node (unknown subject)"
    # D4 branch: needs a GloVe associate >= 0.30
    vec = eng._glove_vector(subject) if hasattr(eng, "_glove_vector") else None
    if vec is None:
        return "PASS (graph node, no embedding)"
    return "graph node present -- needs associate sim >= 0.30 (D4)"


if __name__ == "__main__":
    main()
