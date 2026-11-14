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


def looks_like_salad(text, subject, engine=None):
    """Structural salad check for a production answer.

    NOTE (measured 2026-11-14): two earlier heuristics produced false positives
    on CORRECT long definitions -- a type-token-ratio cutoff flagged
    "A moral is a message that is conveyed or a lesson to be learned from a
    story or event...", and the codebase's `_is_word_salad` is not reachable from
    the engine object (it is imported into the response_gen mixin), so a
    hasattr-guarded delegation silently fell through.

    The signal that actually separates a definition from salad: a definition
    states a predicate about its subject. Require the subject (or its head word)
    to appear AND at least one content word beyond the subject.
    """
    if not text:
        return True
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 3:
        return True
    head = re.findall(r"[a-z']+", subject.lower())
    head = [w for w in head if len(w) >= 3]
    if head and not any(w in words for w in head):
        return True
    content = [w for w in words
               if len(w) >= 4 and w not in _SALAD_STOP
               and w not in (head or [])]
    return len(content) < 2


_SALAD_STOP = frozenset({
    "that", "this", "with", "from", "which", "what", "when", "where", "while",
    "their", "there", "these", "those", "about", "into", "than", "then",
    "also", "such", "some", "more", "most", "many", "much", "very", "just",
    "like", "have", "has", "had", "been", "were", "are", "was", "is", "the",
    "and", "for", "not", "but", "may", "can", "its",
})


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
        if looks_like_salad(text, s, eng):
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
