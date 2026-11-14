"""What does the decoder actually emit, and does it depend on the subject?

Measures the decoder's output distribution for subjects it has NO stored
definition for (the fallback case), and tests whether the emitted text depends
on the conditioning signal at all.

Usage:
    uv run python scripts/diagnose_decoder_salad.py [--limit N]
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
           os.path.join(PROJ, "ravana-v2", "src"),
           os.path.join(PROJ, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CORPUS = os.path.join(PROJ, "data", "corpora", "grounded_descriptions.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    os.environ.pop("RAVANA_OFFLINE", None)
    from ravana.chat.engine import CognitiveChatEngine
    from eval_decoder_groundedness import build_ctx

    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")

    subs = []
    with open(CORPUS, encoding="utf-8") as fh:
        for line in fh:
            if "\t" in line:
                subs.append(line.split("\t", 1)[0].strip())

    # subjects with NO stored definition -- the fallback case
    novel = [s for s in subs if s not in eng._definitions][:args.limit]
    print(f"[salad] {len(novel)} novel subjects (no stored definition)")

    tok_counts = collections.Counter()
    samples = []
    n_same = 0
    n_total = 0

    for s in novel:
        ctx = build_ctx(eng, s, f"what is {s}")
        prev = getattr(eng, "_disable_grounding_gate", False)
        eng._disable_grounding_gate = True
        try:
            raw = eng._generate_with_situation_model(ctx)
        finally:
            eng._disable_grounding_gate = prev
        if not raw:
            continue
        text = raw[0] or ""
        toks = [w for w in re.findall(r"[a-z']+", text.lower()) if len(w) >= 3]
        tok_counts.update(toks)
        if len(samples) < 8:
            samples.append((s, text[:90]))
        n_total += 1

    print("\n[salad] samples:")
    for s, t in samples:
        print("  %-14s %s" % (s, t))

    print(f"\n[salad] top emitted tokens over {n_total} subjects:")
    for tok, n in tok_counts.most_common(25):
        print("  %-14s %5d" % (tok, n))

    total = sum(tok_counts.values())
    top10 = sum(n for _, n in tok_counts.most_common(10))
    print(f"\n[salad] top-10 tokens are {100.0*top10/max(total,1):.1f}% of all tokens")
    print(f"[salad] distinct tokens: {len(tok_counts)} / {total}")

    # Does the output depend on the subject at all?
    print("\n[salad] subject dependence: two different subjects, same frame")
    if len(novel) >= 2:
        for s in novel[:2]:
            ctx = build_ctx(eng, s, f"what is {s}")
            prev = getattr(eng, "_disable_grounding_gate", False)
            eng._disable_grounding_gate = True
            try:
                raw = eng._generate_with_situation_model(ctx)
            finally:
                eng._disable_grounding_gate = prev
            print("  %-14s %s" % (s, (raw[0] if raw else "")[:80]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
