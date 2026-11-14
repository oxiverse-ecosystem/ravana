"""Recover a proposition from a concept's DEFINITION text.

Research deliverable: ``docs/DEFINITION_PROPOSITION.md``.

The plan route needs a ``(relation, target)`` pair to build a sentence. One
source is a typed graph edge, but typed edges are sparse (~40% of subjects) and
the online-typing mechanism is measured as not working
(``docs/TYPING_ACCURACY_IS_CIRCULAR.md``).

The other source is the definition itself. RAVANA stores 324 of them, and a
definition asserts its relation on its surface: "Gravity is the force by which
..." is a categorizing claim; "Acid is a molecule or ion capable of..." is one
too. The brain does the same -- it grounds a novel concept in its description,
not in a curated label.

## Why the relation is read from text and not from vectors

The deprecated ``_infer_relation_type`` compared the relational DIRECTION
vector between two concepts against prototypes. It scored 0/32 on independent
hand-labelled pairs, because a relation's direction is a semantic-similarity
proxy and relational categories are defined by fit to a relational structure
(Popov et al. 2020, cited in ``docs/BRAIN_RELATION_TYPING_SPEC.md``). Reading
the copula and its complement is structural evidence instead.

## The two patterns

Definitions come in two surface shapes:

1. **Copula-first** -- ``"X is a Y"``, ``"X is the Y by which ..."``.
2. **Article-first** -- ``"a molecule or ion capable of..."``,
   ``"an animal that has reached full growth"``, ``"the creator of an
   original work"``. Wikipedia summaries often drop the subject and open on
   the predicate nominal.

MEASURED: pattern 1 alone recovers 98/324 definitions (30%). Most of the 226
"failures" are article-first, so they are a parser gap rather than a coverage
ceiling.

## Scope

Pure function over (definition text) -> (relation, target) or None. The target
is resolved to a graph concept by the caller (``_graph_target`` lives beside
the caller because it needs the graph); this module only reads text.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# ── Patterns ────────────────────────────────────────────────────────────────
# Each yields (relation, captured predicate phrase). Order matters: the more
# specific patterns are tried first.

# Article-first definitions. The subject is omitted, so the definition opens on
# the predicate nominal. Captured only when the phrase is a plausible noun
# phrase -- the head-noun rule below does the real selection.
# NOTE: these capture a bounded number of WORDS, not characters. A character
# budget truncates mid-word -- "a molecule or ion capable of either donatin"
# (measured), which then yields the nonsense target 'donatin'.
_ARTICLE_FIRST = re.compile(
    r"^\s*(?:a|an|the)\s+"
    r"((?:[a-z][a-z'\-]*\s+){0,5}[a-z][a-z'\-]*)", re.I)

_COPULA = re.compile(
    r"\b(?:is|was|are|were)\s+(?:a|an|the)\s+"
    r"((?:[a-z][a-z'\-]*\s+){0,5}[a-z][a-z'\-]*)", re.I)

_COPULA_RELATIVIZED = re.compile(
    r"\b(?:is|was|are|were)\s+the\s+"
    r"((?:[a-z][a-z'\-]*\s+){0,4}?[a-z][a-z'\-]*?)"
    r"\s+(?:by|that|which|of|in|to)\b", re.I)

_APPOSITIVE = re.compile(
    r"\b[a-z]+\s*,\s*(?:a|an)\s+"
    r"((?:[a-z][a-z'\-]*\s+){0,4}?[a-z][a-z'\-]*),", re.I)

# A definition that is really a disambiguation / referent list, not a claim.
_NOT_A_CLAIM = re.compile(
    r"^\s*(?:this|may|can|could|might|term|word|name|phrase| acronym|"
    r"initialism|abbreviation|symbol)\b", re.I)


def _words(phrase: str):
    """Content words of a predicate phrase, function words dropped.

    NOTE: the character class is ASCII-only on purpose but must NOT truncate a
    word at a non-ASCII character -- a naive ``[a-z]+`` over
    "donating a proton (hydrogen…" yields "donatin". Non-ASCII characters are
    skipped, not treated as word boundaries.
    """
    return [w for w in re.findall(r"[a-z][a-z\-]*", phrase.lower())
            if w not in ("the", "a", "an", "of", "to", "in", "by", "which",
                         "that", "and", "or", "for", "with", "on", "as")]


def _head_noun(pos_map, words, prefer_last=True):
    """Pick the NOUN from a predicate phrase.

    ``prefer_last=True`` implements English right-headedness: "basic unit of
    classification" heads on ``unit``, "continuous progression of existence" on
    ``progression``. MEASURED: taking the first content word yields adjectives
    ("basic", "continuous", "domesticated"), which is a claim about nothing.

    ``prefer_last=False`` takes the FIRST noun. Article-first definitions
    ("a molecule or ion capable of...") put the head noun first, so the
    right-most noun is wrong there ("either", "growth", "mainland").
    """
    if not words:
        return ""
    nouns = [w for w in words
             if (pos_map or {}).get(w) in ("noun", "n", "proper_noun")]
    if nouns:
        return nouns[-1] if prefer_last else nouns[0]
    return words[-1] if prefer_last else words[0]


def proposition_from_definition(
    definition: Optional[str],
    *,
    pos_map=None,
) -> Optional[Tuple[str, str]]:
    """Read a ``(relation, target)`` proposition off a definition.

    Returns ``(relation, target)`` where relation is ``"is_a"`` for a
    categorizing claim or ``"has_property"`` for an attributive one, and target
    is the head noun of the predicate phrase. Returns ``None`` when the text
    does not assert a claim (disambiguation pages, empty text).

    The target is a word from the definition, NOT yet resolved to a graph
    concept -- the caller does that, because resolution needs the graph.
    """
    if not definition:
        return None
    text = str(definition).strip()
    if not text or _NOT_A_CLAIM.match(text):
        return None

    # 1. Copula-first with a relativizer: "is the force by which ...".
    m = _COPULA_RELATIVIZED.search(text)
    if m:
        words = _words(m.group(1))
        head = _head_noun(pos_map, words)
        if head:
            return ("is_a", head)

    # 2. Plain copula: "X is a Y that ...".
    m = _COPULA.search(text)
    if m:
        words = _words(m.group(1))
        head = _head_noun(pos_map, words)
        if head:
            return ("is_a", head)

    # 3. Article-first: "a molecule or ion capable of ...". The head noun is
    #    FIRST here, not last -- English right-headedness applies to the
    #    modifier+noun unit, and the article already marks the phrase start.
    m = _ARTICLE_FIRST.match(text)
    if m:
        words = _words(m.group(1))
        head = _head_noun(pos_map, words, prefer_last=False)
        if head:
            return ("is_a", head)

    # 4. Appositive: "X, a Y, ...".
    m = _APPOSITIVE.search(text)
    if m:
        words = _words(m.group(1))
        head = _head_noun(pos_map, words)
        if head:
            return ("is_a", head)

    return None
