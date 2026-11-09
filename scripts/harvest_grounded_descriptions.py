#!/usr/bin/env python3
"""Harvest grounded descriptions for the LingGen conditioner.

The promotion gate (`scripts/measure_linggen_ce.py`) scores the decoder with
SENSORIMOTOR CONDITIONING. That conditioning comes from `LingGenConditioner`,
which is fit on (concept, description) pairs -- and it was never trained,
because `data/corpora/grounded_descriptions.txt` did not exist and no script
produced it. `train.py` printed:

    [LingGen] no grounded_descriptions.txt — skip (run harvest first)

So the decoder was ungrounded by construction, which is the most likely reason
it lost the promotion probe (3.80 vs the template path's 3.75, both near the
3.93 uniform baseline).

This script produces that file. It reuses the engine's own knowledge sources
rather than inventing a new one:

  * `engine._definitions` -- already seeded from Wikipedia/ConceptNet by
    `engine_graph.py` at boot (the `[KB] Seeded N definitions` lines).
  * `engine.kb_describe` / `engine.describe_from_cn` -- the same lookups the
    seeder uses, called for any high-value corpus concept not yet covered.

Output format (one pair per line, tab-separated):

    concept<TAB>human description

Usage:
    uv run python scripts/harvest_grounded_descriptions.py
    uv run python scripts/harvest_grounded_descriptions.py --extra 200
"""
import argparse
import os
import re
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJ, os.path.join(PROJ, "ravana", "src"),
          os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

OUT_PATH = os.path.join(PROJ, "data", "corpora", "grounded_descriptions.txt")
CORPUS_PATH = os.path.join(PROJ, "data", "corpora", "teen_seeds.txt")

# A description must be prose, not a citation or navigation fragment. Same
# failure mode as the corpus itself: expand_corpus.py produced 62k lines of
# which only 22.5% was real English, the rest being reference lists
# ("↑ Takai K, Nakamura K, ...") and page furniture.
_JUNK = re.compile(
    r"\b(Email Me|Contact:|Print Page|Login|LibApps|Homepage|Facebook|"
    r"Instagram|RELATED TOPICS|Subjects:|Tags:|Librarian|Copyright|"
    r"All rights reserved|Privacy Policy|Terms of Use)\b", re.I)
_CITE = re.compile(r"^\s*[↑↓]|\(\w+ \d{4}\)|et al\.?\s*$")


def _is_prose(text, min_words=6):
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 20:
        return False
    if _JUNK.search(text) or _CITE.search(text):
        return False
    words = re.findall(r"[a-zA-Z']+", text)
    if len(words) < min_words:
        return False
    # Reject author-initial soup: too many single-letter tokens.
    singles = sum(1 for w in words if len(w) == 1)
    if singles / len(words) > 0.20:
        return False
    # Must read like a sentence, not a reference entry.
    if not re.search(r"\b(is|are|was|were|refers|means|describes|"
                     r"the|a|an|of|to|in)\b", text.lower()):
        return False
    return True


def _corpus_concepts(top_n=150):
    """High-frequency content words from the cleaned corpus."""
    if not os.path.exists(CORPUS_PATH):
        return []
    from collections import Counter
    counts = Counter()
    stop = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
        "how", "its", "may", "new", "now", "old", "see", "two", "who", "boy",
        "did", "she", "use", "her", "way", "many", "oil", "sit", "set", "run",
        "eat", "far", "sea", "eye", "ask", "own", "say", "too", "any", "try",
        "let", "put", "end", "why", "turn", "here", "show", "every", "good",
        "me", "give", "our", "under", "name", "very", "through", "just", "form",
        "sentence", "great", "think", "where", "help", "much", "before", "move",
        "right", "too", "means", "old", "any", "same", "tell", "boy", "follow",
        "came", "want", "show", "also", "around", "farm", "three", "small",
        "set", "put", "end", "does", "another", "well", "large", "must", "big",
        "even", "such", "turn", "because", "went", "read", "need", "land",
        "different", "home", "us", "move", "try", "kind", "hand", "picture",
        "again", "change", "off", "play", "spell", "air", "away", "animal",
        "house", "point", "page", "letter", "mother", "answer", "found",
        "study", "still", "learn", "should", "america", "world",
    }
    with open(CORPUS_PATH, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            for w in re.findall(r"[a-z']+", line.lower()):
                if len(w) >= 4 and w not in stop:
                    counts[w] += 1
    return [w for w, _ in counts.most_common(top_n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", type=int, default=150,
                    help="extra corpus concepts to look up beyond _definitions")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    from ravana.chat.engine import CognitiveChatEngine
    print("Booting engine (seeds _definitions from Wikipedia/ConceptNet)...")
    eng = CognitiveChatEngine(dim=64, seed=7, baby_mode=True)

    pairs = []
    seen = set()

    # 1. Everything already in _definitions.
    defs = getattr(eng, "_definitions", {}) or {}
    for concept, desc in defs.items():
        c = (concept or "").strip().lower()
        if c and c not in seen and _is_prose(desc):
            pairs.append((c, desc.strip()))
            seen.add(c)
    print(f"  {len(pairs)} pairs from engine._definitions")

    # 2. Top corpus concepts not yet covered.
    kb = getattr(eng, "kb_describe", None)
    cn = getattr(eng, "describe_from_cn", None)
    if args.extra > 0 and callable(kb):
        added = 0
        for word in _corpus_concepts(args.extra):
            if word in seen:
                continue
            try:
                desc = kb(word) or (cn(word) if callable(cn) else None)
            except Exception:
                desc = None
            if _is_prose(desc):
                pairs.append((word, desc.strip()))
                seen.add(word)
                added += 1
        print(f"  +{added} pairs from kb_describe/describe_from_cn")

    if not pairs:
        print("  ERROR: no usable pairs harvested", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for concept, desc in pairs:
            fh.write(f"{concept}\t{desc}\n")

    print(f"\nWrote {len(pairs)} grounded pairs -> {args.out}")
    print("Now run:  uv run python scripts/train.py --mode linggen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
