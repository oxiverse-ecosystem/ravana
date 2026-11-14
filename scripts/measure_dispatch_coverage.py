"""How often does production's internal-knowledge dispatch HIT vs MISS?

This is the number that tracks "fluent and factual": for a subject the engine
has a stored definition for, does `consult_internal` return it? And for subjects
it has nothing for, does it fail closed rather than confabulate?

Usage:
    uv run python scripts/measure_dispatch_coverage.py [--limit N]
"""
import argparse
import collections
import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.environ.pop("RAVANA_OFFLINE", None)
    from ravana.chat.engine import CognitiveChatEngine
    from ravana.chat.brain_regions import consult_internal

    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    subs = load_subjects(args.corpus, args.limit)
    print("[dispatch] %d subjects" % len(subs))

    stats = collections.Counter()
    misses = []
    for s in subs:
        ans = consult_internal(s, eng)
        if ans is None:
            stats["MISS"] += 1
            if len(misses) < 15:
                misses.append(s)
        else:
            stats["HIT " + getattr(ans, "source", "?")] += 1

    total = sum(stats.values())
    print("\n[dispatch] consult_internal outcomes (n=%d):" % total)
    for k, v in stats.most_common():
        print("  %-16s %5d  %5.1f%%" % (k, v, 100.0 * v / max(total, 1)))
    print("\n[dispatch] sample MISSes: %s" % ", ".join(misses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
