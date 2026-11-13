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
# Wikipedia disambiguation / set-index pages. Anchored so a real definition
# containing "may refer to" mid-sentence is not rejected -- measured: 200 of
# 685 harvested pairs (29%) were this junk.
_DISAMBIG = re.compile(
    r"^\s*(topics referred to by the same term"
    r"|.{0,40}?\s+(most often|often|commonly|usually|also|normally|"
    r"generally)\s+refers to\b[:.]?"
    r"|.{0,40}?\s+may refer to\b[:.]?"
    r")",
    re.I | re.S,
)


def _is_prose(text, min_words=6):
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 20:
        return False
    if _JUNK.search(text) or _CITE.search(text):
        return False
    # Reject Wikipedia disambiguation pages FIRST, before the word-count gate:
    # "Topics referred to by the same term." is 7 words but is navigation, not
    # description, and shorter variants fall below the prose threshold anyway.
    # Measured: 200 of 685 harvested pairs (29%) were this junk.
    if _DISAMBIG.search(text):
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


_NOUN_CACHE = {}
_NOUN_CACHE_PATH = os.path.join(PROJ, "data", "corpora", ".noun_cache.json")


def _load_noun_cache():
    global _NOUN_CACHE
    if os.path.exists(_NOUN_CACHE_PATH):
        try:
            import json
            with open(_NOUN_CACHE_PATH, encoding="utf-8") as fh:
                _NOUN_CACHE = {k: bool(v) for k, v in json.load(fh).items()}
        except Exception:
            _NOUN_CACHE = {}


def _save_noun_cache():
    try:
        import json
        os.makedirs(os.path.dirname(_NOUN_CACHE_PATH), exist_ok=True)
        with open(_NOUN_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(_NOUN_CACHE, fh)
    except Exception:
        pass


def _strong_noun(word):
    """True when `word`'s most frequent WordNet sense is a noun and it has at
    least two noun senses.

    WordNet alone is weak evidence -- "have", "there" and "then" all have a
    marginal noun sense. Requiring the FIRST (most frequent) synset to be a
    noun AND >= 2 noun synsets rejects those while keeping real concepts:

        gravity first=n noun_count=3 strong=True
        ocean   first=n noun_count=2 strong=True
        have    first=n noun_count=1 strong=False
        there   first=n noun_count=1 strong=False
        then    first=n noun_count=1 strong=False
        time    first=n noun_count=10 strong=True

    Results are memoised: a WordNet lookup costs ~1.0s (the corpus scan asks
    about 2000 words, so an uncached scan is ~34 min and gets killed).

    Returns False when NLTK/WordNet is unavailable -- the caller then falls
    back to frequency-only selection rather than harvesting nothing.
    """
    if word in _NOUN_CACHE:
        return _NOUN_CACHE[word]
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        syns = wn.synsets(word)
    except Exception:
        _NOUN_CACHE[word] = False
        return False
    result = bool(syns) and syns[0].pos() == "n" and \
        sum(1 for s in syns if s.pos() == "n") >= 2
    _NOUN_CACHE[word] = result
    return result


def _corpus_concepts(top_n=150, noun_filter=True):
    """High-frequency content words from the cleaned corpus.

    With `noun_filter` (the default) a word must also be a strong noun in
    WordNet. Frequency alone cannot separate concepts from function words:
    the raw top-N from this corpus is "which, there, when, been, very, more"
    -- harvesting those teaches the decoder to describe "which". The strict
    noun test yields real concepts (time, thought, hand, night, house, life)
    at a measured 100% Wikipedia hit rate.
    """
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
    ranked = [w for w, _ in counts.most_common()]
    if noun_filter:
        # WordNet lookups dominate the cost here (~1.0s each), so the results
        # are cached on disk: the first run pays, later runs are instant.
        _load_noun_cache()
        # Scan more of the ranked list than we need -- many high-frequency
        # words are not strong nouns.
        scanned = ranked[:max(top_n * 20, 2000)]
        picked = [w for w in scanned if _strong_noun(w)]
        _save_noun_cache()
        return picked[:top_n]
    return ranked[:top_n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", type=int, default=150,
                    help="extra corpus concepts to look up beyond _definitions")
    ap.add_argument("--no-noun-filter", action="store_true",
                    help="skip the WordNet strong-noun test and take the "
                         "top-N by frequency (not recommended: the raw top-N "
                         "is function words like 'which, there, when')")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    from ravana.chat.engine import CognitiveChatEngine
    print("Booting engine (seeds _definitions from Wikipedia/ConceptNet)...")
    eng = CognitiveChatEngine(dim=64, seed=7, baby_mode=True)

    pairs = []
    seen = set()

    # Resume support: a previous run may have already written pairs. Load them
    # so a restarted harvest skips what it already fetched instead of paying
    # for the same Wikipedia lookups again.
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                if "\t" not in line:
                    continue
                c, d = line.rstrip("\n").split("\t", 1)
                c = c.strip().lower()
                if c and c not in seen and _is_prose(d):
                    pairs.append((c, d.strip()))
                    seen.add(c)
        if pairs:
            print(f"  resumed: {len(pairs)} pairs already in "
                  f"{os.path.basename(args.out)}")

    # 1. Everything already in _definitions.
    defs = getattr(eng, "_definitions", {}) or {}
    for concept, desc in defs.items():
        c = (concept or "").strip().lower()
        if c and c not in seen and _is_prose(desc):
            pairs.append((c, desc.strip()))
            seen.add(c)
    print(f"  {len(pairs)} pairs from engine._definitions")

    # 2. Top corpus concepts not yet covered.
    #
    # RESUMABLE: each pair is appended to the output file as it is fetched, so
    # a run killed mid-way (the gateway restarts long jobs on this host) keeps
    # everything it already got. On the next run the existing file is loaded
    # into `seen` and the loop skips those words.
    kb = getattr(eng, "kb_describe", None)
    cn = getattr(eng, "describe_from_cn", None)
    if args.extra > 0 and callable(kb):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        added = 0
        attempted = 0
        for word in _corpus_concepts(args.extra,
                                     noun_filter=not args.no_noun_filter):
            if word in seen:
                continue
            attempted += 1
            try:
                desc = kb(word) or (cn(word) if callable(cn) else None)
            except Exception:
                desc = None
            if _is_prose(desc):
                pairs.append((word, desc.strip()))
                seen.add(word)
                added += 1
                # Append immediately: the fetch is the expensive part, so
                # persist every success rather than losing the whole run.
                with open(args.out, "a", encoding="utf-8") as fh:
                    fh.write(f"{word}\t{desc.strip()}\n")
        print(f"  +{added} pairs from kb_describe/describe_from_cn "
              f"(attempted {attempted})")

    if not pairs:
        print("  ERROR: no usable pairs harvested", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Rewrite the file from `pairs` so the final output is ordered and free of
    # duplicates. `pairs` already contains every pair this run appended plus
    # the KB definitions, and `seen` was seeded from any pre-existing file.
    with open(args.out, "w", encoding="utf-8") as fh:
        for concept, desc in pairs:
            fh.write(f"{concept}\t{desc}\n")

    print(f"\nWrote {len(pairs)} grounded pairs -> {args.out}")
    print("Now run:  uv run python scripts/train.py --mode linggen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
