"""Predicate canonicalization — the layer the original plan was missing.

PropositionParser emits surface predicates ("is", "are", "has", "have",
"means", "can"); OpenIE emits verb-phrase relations ("is a", "improves",
"was born in"). Without normalization, evidence for one relation
fragments across surface variants and every profile's n stays near 0 —
the learning loop silently never converges.

Canonicalization here is MORPHOLOGICAL/SYNTACTIC only (copula collapse,
auxiliary stripping, whitespace/number normalization) — it does not
encode any inference knowledge. Semantic equivalence between distinct
predicates (e.g. "parent_of" vs "child_of") is LEARNED as inverse/
composition statistics, not authored here.
"""
from __future__ import annotations

import re

# Copula/auxiliary collapse: pure agreement/tense variants of the same
# relation. This is grammar, not world knowledge.
_COPULA = {"is", "are", "was", "were", "be", "being", "been", "am",
           "is a", "is an", "are a", "are an", "was a", "was an"}
_HAVE = {"has", "have", "had", "having", "has a", "has an", "have a"}
_CAN = {"can", "could", "is able to", "are able to"}

_WS = re.compile(r"\s+")
_ARTICLE_TAIL = re.compile(r"\s+(a|an|the)$")
_ARTICLE_HEAD = re.compile(r"^(a|an|the|my|your|our|their|his|her|its)\s+")
_PUNCT_EDGE = re.compile(r"^[\s\.,;:!?'\"]+|[\s\.,;:!?'\"]+$")


def canonical_term(term: str) -> str:
    """Normalize a subject/object term: lowercase, strip edge punctuation
    and leading determiners ('a man' -> 'man'). PropositionParser keeps
    articles in the object slot; without this, (socrates, is, 'a man')
    never matches a lookup for 'man'. Grammar-only — no semantics."""
    t = (term or "").strip().lower()
    t = _PUNCT_EDGE.sub("", t)
    t = _ARTICLE_HEAD.sub("", t)
    return _WS.sub(" ", t).strip()


def canonical_predicate(predicate: str) -> str:
    """Normalize a surface predicate to a canonical key.

    - lowercase, collapse whitespace, underscores -> spaces
    - collapse copula variants to "is"
    - collapse possession variants to "has"
    - collapse ability variants to "can"
    - strip trailing articles ("consists of the" -> "consists of")
    """
    p = (predicate or "").strip().lower().replace("_", " ")
    p = _WS.sub(" ", p)
    if not p:
        return p
    p = _ARTICLE_TAIL.sub("", p)
    if p in _COPULA:
        return "is"
    if p in _HAVE:
        return "has"
    if p in _CAN:
        return "can"
    return p
