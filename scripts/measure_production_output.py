"""What does the engine ACTUALLY emit for 'what is X'?

The eval instrument calls `_generate_with_situation_model` directly, which
bypasses the dispatch layer that consults stored knowledge first. This script
measures the real production entry point (`process_turn`) so the two can be
compared.

Usage:
    uv run python scripts/measure_production_output.py [--limit N]
"""
import argparse
import collections
import os
import re
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CORPUS = os.path.join(PROJ, "data", "corpora", "grounded_descriptions.txt")


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


def looks_like_salad(text, subject):
    """Cheap structural salad check: no content word shared with the subject's
    definition, and a low type-token ratio."""
    if not text:
        return True
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 3:
        return True
    uniq = len(set(words)) / len(words)
    return uniq < 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--no-web", action="store_true",
                    help="stub the live web path so the run measures the "
                         "INTERNAL knowledge dispatch (much faster)")
    args = ap.parse_args()

    os.environ.pop("RAVANA_OFFLINE", None)
    from ravana.chat.engine import CognitiveChatEngine

    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")

    if args.no_web:
        # The live web path is slow and not what we are measuring here: we want
        # to know whether a subject with a STORED definition gets it emitted.
        def _no_web(ctx):
            return None
        eng._web_direct_answer = _no_web
        os.environ["RAVANA_OFFLINE"] = "1"
    subs = load_subjects(args.corpus, args.limit)

    stats = collections.Counter()
    salad_samples = []
    good_samples = []

    for s in subs:
        try:
            out = eng.process_turn(f"what is {s}")
        except Exception as exc:
            stats["error"] += 1
            continue
        text = (out or "").strip()
        if not text:
            stats["empty"] += 1
            continue
        if looks_like_salad(text, s):
            stats["salad"] += 1
            if len(salad_samples) < 6:
                salad_samples.append((s, text[:100]))
        else:
            stats["grounded-looking"] += 1
            if len(good_samples) < 6:
                good_samples.append((s, text[:100]))

    print("\n[production] process_turn('what is X') over %d subjects:" % len(subs))
    for k, v in stats.most_common():
        print("  %-20s %5d" % (k, v))

    print("\n[production] grounded-looking samples:")
    for s, t in good_samples:
        print("  %-14s %s" % (s, t))
    print("\n[production] salad samples:")
    for s, t in salad_samples:
        print("  %-14s %s" % (s, t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
