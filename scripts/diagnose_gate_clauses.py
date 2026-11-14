"""Which CLAUSE-LEVEL condition of the grounding gate is doing the withholding?

Step 1 (verified anchor) now passes for ~100% of corpus subjects, so the ~50%
withholding is entirely in the per-sentence monitor. This instrument runs real
turns through `_generate_with_situation_model` and, for each subject, records
whether the decoder's own output survived the gate and -- if not -- which
per-sentence condition would have fired first.

Usage:
    uv run python scripts/diagnose_gate_clauses.py [--limit N]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--limit", type=int, default=80)
    args = ap.parse_args()

    os.environ.pop("RAVANA_OFFLINE", None)
    from ravana.chat.engine import CognitiveChatEngine
    sys.path.insert(0, os.path.join(PROJ, "scripts"))
    from eval_decoder_groundedness import build_ctx

    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    subs = load_subjects(args.corpus, args.limit)

    stats = collections.Counter()
    samples = collections.defaultdict(list)

    for s in subs:
        try:
            ctx = build_ctx(eng, s, f"what is {s}")
            prev = getattr(eng, "_disable_grounding_gate", False)
            eng._disable_grounding_gate = True
            try:
                raw = eng._generate_with_situation_model(ctx)
            finally:
                eng._disable_grounding_gate = prev
            if not raw:
                stats["no decoder output"] += 1
                continue
            raw_text = raw[0] or ""
            if not raw_text.strip():
                stats["empty decoder output"] += 1
                continue

            gated = eng._sm_response_grounded(ctx, raw_text)
            if gated:
                stats["PASSED the gate"] += 1
                continue

            stats["WITHHELD by clause monitor"] += 1
            reason = classify(eng, ctx, s, raw_text)
            stats["  " + reason] += 1
            if len(samples[reason]) < 4:
                samples[reason].append((s, raw_text[:120]))
        except Exception as exc:
            stats["error: %s" % type(exc).__name__] += 1

    print("\n[clause] outcomes (n=%d):" % len(subs))
    for k, v in stats.most_common():
        print("  %-52s %5d" % (k, v))

    for reason, items in samples.items():
        print("\n[clause] %s" % reason)
        for s, txt in items:
            print("  %-16s %s" % (s, txt))


def classify(eng, ctx, subject, text):
    """Name the first per-sentence condition that fails for this text."""
    sl = text.lower().strip()
    words = sl.split()
    if len(words) <= 3:
        return "micro-clause (<=3 words)"
    sw = set(re.findall(r"[a-z']+", sl))
    if subject.lower() not in sl:
        return "reference: subject not mentioned"
    _GLUE = {"is", "are", "was", "were", "means", "does", "makes", "and",
             "with", "to", "from", "of", "the", "a", "an", "in", "on", "for"}
    _VAGUE = {"semantic", "contrastive", "causal", "even", "great", "related",
              "connected", "linked", "means", "matters", "differs", "compared",
              "vs", "going", "pet", "possibly", "maybe", "perhaps"}
    real = [w for w in sw if w not in _GLUE and w not in _VAGUE
            and w not in subject.lower().split()]
    if not real:
        return "vague-only clause (no real content word)"
    return "other clause condition (coherence/conflict/substance)"


if __name__ == "__main__":
    main()
