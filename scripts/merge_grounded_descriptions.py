#!/usr/bin/env python3
"""Merge grounded-description sources into one conditioner training set.

Two producers exist:
  * `harvest_grounded_corpus(local_books=True)` -- a real sentence containing
    the concept, drawn from the seeded Gutenberg novels. Grounded USAGE.
  * `harvest_grounded_descriptions.py` -- clean Wikipedia/ConceptNet
    definitions from engine._definitions. Grounded DEFINITION.

They write the same file, so running one clobbers the other. This merges them,
dedupes by concept (preferring the definition, which is cleaner prose), and
applies the same prose filter that both the corpus and the conditioner need.

Usage:
    uv run python scripts/merge_grounded_descriptions.py
"""
import os
import re
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "data", "corpora", "grounded_descriptions.txt")
BOOKS = os.path.join(PROJ, "data", "corpora", "gutenberg_descriptions.txt")

_JUNK = re.compile(
    r"\b(Email Me|Contact:|Print Page|Login|LibApps|Homepage|Facebook|"
    r"Instagram|RELATED TOPICS|Subjects:|Tags:|Librarian|Copyright|"
    r"All rights reserved|Privacy Policy|Terms of Use)\b", re.I)


def _is_prose(text, min_words=5):
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 20 or _JUNK.search(text):
        return False
    words = re.findall(r"[a-zA-Z']+", text)
    if len(words) < min_words:
        return False
    singles = sum(1 for w in words if len(w) == 1)
    if singles / len(words) > 0.20:
        return False
    return True


def _load(path):
    pairs = []
    if not os.path.exists(path):
        return pairs
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            concept, desc = line.split("\t", 1)
            concept = concept.strip().lower()
            desc = desc.strip()
            if concept and _is_prose(desc):
                pairs.append((concept, desc))
    return pairs


def main():
    # The book harvest writes grounded_descriptions.txt, and so does the KB
    # harvest -- they clobber each other. gutenberg_descriptions.txt is the
    # preserved copy of the book output, so read THAT as the usage source and
    # the live file as the definition source.
    book_pairs = _load(BOOKS)
    def_pairs = _load(OUT)

    # Prefer definitions (cleaner prose) when a concept appears in both.
    merged = {}
    for c, d in book_pairs:
        merged.setdefault(c, d)
    for c, d in def_pairs:
        merged[c] = d

    if not merged:
        print("  no pairs found in either source", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for concept in sorted(merged):
            fh.write(f"{concept}\t{merged[concept]}\n")

    print(f"  book/usage pairs:   {len(book_pairs)}")
    print(f"  definition pairs:   {len(def_pairs)}")
    print(f"  merged (deduped):   {len(merged)}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
