"""
C-lite OpenIE — dependency-free triple extraction from web text.
===============================================================
Part of the unified semantic layer (Phase C-lite: web -> facts -> graph).

Facts do NOT need vector binding (TransE/Bordes 2013; Nickel 2016). They
need a typed graph, which RAVANA already has (ConceptGraph + Hebbian/web
edges). This module extracts (subject, relation, object) assertions from free
text so they can be written as TYPED edges into the existing graph — no new
dimensionality, no HRR (that's deferred N3).

Design: a light, transparent dependency-free extractor. We do NOT pull in a
heavyweight neural OpenIE (ClausIE/OpenIE-5) — the point of C-lite is lowest
risk + exercises shipped infra. We use:
  - sentence segmentation (regex)
  - a verb lexicon that maps surface verbs -> typed relations
    (is-a, has-property, located-in, part-of, causes, related-to)
  - subject/object noun-chunk heuristics (head noun + modifiers)

This is deliberately modest: it captures the dominant fact shapes a curious
agent meets on the web (definitions, properties, locations, meronyms, causes).
Rich compositional extraction is the deferred N3 job (binding).

Grounded in: McClelland et al. 1995 CLS (graph as associative substrate);
Kumaran et al. 2016 (pattern completion); Tse et al. 2007 (schema-accelerated
consolidation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Surface verb -> typed relation. Brain-faithful: these are the relational
# primitives the existing ConceptGraph already supports as relation_type.
VERB_TO_RELATION = {
    # is-a / identity
    "is": "is_a",
    "are": "is_a",
    "was": "is_a",
    "were": "is_a",
    "be": "is_a",
    "means": "is_a",
    "refers": "is_a",
    "define": "is_a",
    "called": "is_a",
    # has-property
    "has": "has_property",
    "have": "has_property",
    "had": "has_property",
    "possess": "has_property",
    "feature": "has_property",
    "measure": "has_property",
    "weigh": "has_property",
    "cost": "has_property",
    # located-in
    "in": "located_in",
    "at": "located_in",
    "near": "located_in",
    "located": "located_in",
    "based": "located_in",
    "found": "located_in",
    # part-of (meronym)
    "part": "part_of",
    "contain": "part_of",
    "contains": "part_of",
    "comprise": "part_of",
    "include": "part_of",
    "consist": "part_of",
    # causes
    "cause": "causes",
    "causes": "causes",
    "produce": "causes",
    "produces": "causes",
    "lead": "causes",
    "leads": "causes",
    "result": "causes",
    "trigger": "causes",
}

# Stop words for noun-chunk heads
STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "for", "with", "on",
    "as", "by", "that", "this", "these", "those", "it", "its", "their",
    "his", "her", "our", "your", "my", "we", "they", "you", "he", "she",
    "is", "are", "was", "were", "be", "been", "being", "which", "who", "what",
    "when", "where", "why", "how", "can", "could", "will", "would", "should",
}

RELATION_LABELS = set(VERB_TO_RELATION.values())

# Words that can never be a real subject/object of a fact (pronouns, deixis,
# interrogatives, copula-complements that aren't nouns).
NON_FACTUAL_HEADS = {
    "you", "your", "yours", "we", "our", "they", "them", "their", "he", "she",
    "him", "her", "it", "its", "i", "me", "my", "us", "this", "that", "these",
    "those", "there", "here", "today", "tomorrow", "yesterday", "now", "then",
    "who", "what", "when", "where", "why", "how", "which", "whom",
}


@dataclass
class Fact:
    """A single extracted (subject, relation, object) assertion."""
    subject: str
    relation: str
    obj: str
    confidence: float
    source_sentence: str = ""

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.relation, self.obj)


def _split_sentences(text: str) -> List[str]:
    # crude but robust segmentation
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _tokenize(sent: str) -> List[str]:
    return re.findall(r"[a-z][a-z0-9'\-]*", sent.lower())


def _chunk(tokens: List[str], exclude: set) -> Optional[str]:
    """Take the head noun + its preceding modifiers, skipping stopwords.
    Returns a normalized multi-word concept label, or None."""
    kept = [t for t in tokens if t not in STOP and t not in exclude]
    if not kept:
        return None
    # drop leading empty determiners already filtered; join modifiers + head
    label = " ".join(kept)
    return label


class OpenIEExtractor:
    """Lightweight, transparent triple extractor (no neural deps)."""

    def extract(self, text: str, max_facts: int = 200) -> List[Fact]:
        facts: List[Fact] = []
        for sent in _split_sentences(text):
            toks = _tokenize(sent)
            # find the first verb we map
            rel = None
            vidx = -1
            for i, t in enumerate(toks):
                if t in VERB_TO_RELATION:
                    rel = VERB_TO_RELATION[t]
                    vidx = i
                    break
            if rel is None or vidx == 0:
                continue
            # reject interrogative frames (no fact in a question)
            if sent.rstrip().endswith("?"):
                continue
            subj_tokens = toks[:vidx]
            obj_tokens = toks[vidx + 1:]
            # remove trailing punctuation tokens
            obj_tokens = [t for t in obj_tokens if t not in {".", "!", "?"}]
            subj = _chunk(subj_tokens, exclude={rel})
            obj = _chunk(obj_tokens, exclude={rel})
            if not subj or not obj:
                continue
            # reject pronouns/deixis/interrogatives as heads (false facts)
            if subj.split()[-1] in NON_FACTUAL_HEADS or obj.split()[-1] in NON_FACTUAL_HEADS:
                continue
            # confidence heuristic: longer/cleaner sentences slightly more可信
            conf = 0.6
            if len(obj_tokens) <= 4:
                conf = 0.7
            facts.append(Fact(subject=subj, relation=rel, obj=obj,
                              confidence=conf, source_sentence=sent))
            if len(facts) >= max_facts:
                break
        return facts
