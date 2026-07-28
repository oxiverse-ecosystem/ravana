"""Neuro-symbolic Premise Extractor (Lever 2).

Bridges unstructured English prose -> clean DeductiveTriples for the
Section 6.5 prefrontal working-memory channel.

Neuroscience grounding (per the relational-reasoning plan):
  The human Left Superior Temporal Gyrus (STG) and Broca's area
  (BA 44/45) parse raw linguistic prose into structured semantic
  frames (Agent, Rel, Patient, Quantifier, Polarity) BEFORE the
  Prefrontal Cortex (dlPFC/rlPFC) integrates them in working
  memory. This module is the STG/Broca analogue: it de-blobs
  multi-word clause fragments ("researchers hypothesized that the
  reason why westernized black people suffer from hypertension")
  into compact noun-phrase heads, then binds relation / quantifier /
  polarity frames onto those heads.

Design consequences (faithful, not a patch):
  1. De-blobbing is DEPENDENCY-AWARE: subject/object are the
     head noun(s) of their NP span (spaCy), not the raw matched
     substring. A clause like "the administrative service area is
     southwest of the cultural area" yields clean heads, not the
     whole preamble sentence.
  2. Relation / quantifier / negation NORMALISERS are REUSED from
     ravana.core.deductive_reasoning (single source of truth) - this
     module adds ONLY the frame-extraction layer, never duplicates
     the logical pattern tables.
  3. Fail-closed on noise: a clause that yields no clean relational
     frame contributes nothing (returns []), exactly like the regex
     path it extends.
  4. Graceful degradation: if spaCy / its model is unavailable,
     it falls back to the regex parser so the channel still runs.

This module is intentionally narrow. It is NOT a general NL reasoner
and does not pretend to cover LogiQA's conditional-rule / argument-
strengthen tasks (those need a different, rule-aware substrate). It
targets the transitive-relational subset the metarule engine can
actually chain (copula, comparative, positional, universal).
"""
from __future__ import annotations

from typing import List, Optional

from ravana.core.deductive_reasoning import (
    DeductiveTriple,
    _canon,
    _EXISTS,
    _FORALL,
    _is_noise_copula,
    _LOC_KEY,
    _NEG,
    _REL_PATTERNS,
)

# --------------------------------------------------------------------------
# spaCy lazy loader (so import of this module never hard-fails if the
# model is absent - we degrade to regex).
# --------------------------------------------------------------------------
_NLP = None
_NLP_OK = None


def _get_nlp():
    """Return a loaded spaCy English pipeline or (None, False)."""
    global _NLP, _NLP_OK
    if _NLP_OK is not None:
        return _NLP, _NLP_OK
    try:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
        _NLP_OK = True
    except Exception:
        _NLP = None
        _NLP_OK = False
    return _NLP, _NLP_OK


# --------------------------------------------------------------------------
# Noun-phrase de-blobbing: collapse an NP span to its head token(s).
# --------------------------------------------------------------------------
def _np_head_surface(span) -> str:
    """Return a compact surface form for an NP span.

    Uses the span's HEAD token (the syntactic nucleus) plus any
    leftmost determiner/article, trimmed to <= _MAX_HEAD_TOK head
    words. This is what kills clause-blobs: "the administrative
    service area" -> "service area" heads, not the whole preamble.
    """
    if span is None or len(span) == 0:
        return ""
    head = span.root
    # Collect the head + its immediate nominal children (compounds,
    # modifiers) to keep multi-word heads like "service area" intact,
    # but DROP relative-clause tails ("... that researchers built").
    toks = [head]
    for c in head.children:
        if c.dep_ in ("compound", "amod", "nummod", "nmod", "poss"):
            toks.append(c)
    # keep only tokens that are actually inside the span
    toks = [t for t in toks if t.i >= span.start and t.i < span.end]
    toks.sort(key=lambda t: t.i)
    words = [t.text for t in toks]
    return " ".join(words)


_MAX_HEAD_TOK = 3  # entity length cap (<=3 content words per entity)


def _clean_entity(raw: str) -> str:
    """Final canonical clean-up of an extracted entity string."""
    c = _canon(raw)
    if not c:
        return ""
    # hard cap: if still blobby (>3 words) it's not a clean entity
    if len(c.split()) > _MAX_HEAD_TOK:
        return ""
    return c


# --------------------------------------------------------------------------
# Clause segmentation (sentence + coordinating / contrastive split).
# Reused logic from the regex parser, kept local so the extractor is
# self-contained for the benchmark script.
# --------------------------------------------------------------------------
import re  # noqa: E402

_CONJ = re.compile(
    r"\b(and|but|however|because|therefore|so|thus|whereas|although|"
    r"though|while)\b", re.IGNORECASE)


def _split_clauses(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text or "")
    clauses: List[str] = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        for seg in _CONJ.split(s):
            seg = seg.strip().strip(",.!?;:\"'")
            if len(seg.split()) >= 2:
                clauses.append(seg)
    if not clauses:
        clauses = [text] if text else []
    return clauses


# --------------------------------------------------------------------------
# Core extractor
# --------------------------------------------------------------------------
class DeductivePremiseExtractor:
    """STG/Broca analogue: prose -> clean DeductiveTriples.

    Primary path uses spaCy dependency parse for NP de-blobbing;
    falls back to the regex parser if spaCy is unavailable.
    """

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        self._nlp, self._spacy_ok = (None, False)
        if use_spacy:
            self._nlp, self._spacy_ok = _get_nlp()

    def extract(self, text: str) -> List[DeductiveTriple]:
        if not text:
            return []
        if self.use_spacy and self._spacy_ok:
            try:
                return self._extract_spacy(text)
            except Exception:
                # any parse failure -> safe regex fallback
                return self._extract_regex(text)
        return self._extract_regex(text)

    # -- spaCy path --------------------------------------------------------
    def _extract_spacy(self, text: str) -> List[DeductiveTriple]:
        nlp = self._nlp
        doc = nlp(text)
        out: List[DeductiveTriple] = []
        for sent in doc.sents:
            clause = sent.text.strip()
            if len(clause.split()) < 2:
                continue
            # Sub-clause split on coordinators inside the sentence.
            for seg in _CONJ.split(clause):
                seg = seg.strip().strip(",.!?;:\"'")
                if len(seg.split()) < 2:
                    continue
                tris = self._clause_to_triples(seg, seg)
                out.extend(tris)
        return out

    def _clause_to_triples(self, clause: str, raw: str) -> List[DeductiveTriple]:
        """Find a relational pattern, then bind subject/object to NP
        head surfaces via a focused spaCy re-parse of the clause."""
        nlp = self._nlp
        sub_doc = nlp(clause)
        out: List[DeductiveTriple] = []
        for pat, fn in _REL_PATTERNS:
            m = pat.match(clause)
            if not m:
                continue
            rel, s_raw, o_raw, rtype, _ = fn(m)
            if rel is None:
                continue  # propositional-noise copula (regex-level filter)
            # Resolve subject/object to clean NP heads.
            s = self._resolve_entity(s_raw, sub_doc)
            o = self._resolve_entity(o_raw, sub_doc)
            if not s or not o or s == o:
                continue
            # Broca's propositional filter on the DE-BLOBBED heads:
            # drops "hypothesized is result" style non-frames.
            if _is_noise_copula(s, o):
                continue
            if len(s.split()) > _MAX_HEAD_TOK or len(o.split()) > _MAX_HEAD_TOK:
                continue
            q = "none"
            if _FORALL.search(clause):
                q = "forall"
            elif _EXISTS.search(clause):
                q = "exists"
            neg = bool(_NEG.search(clause))
            out.append(DeductiveTriple(
                subject=s, predicate=rel, object=o,
                quantifier=q, polarity=not neg, relation_type=rtype))
            break  # most-specific relation wins per clause
        return out

    def _resolve_entity(self, raw: str, doc) -> str:
        """Map a regex-captured raw span to a clean NP-head surface.

        Strategy: find the token span in `doc` matching the raw
        substring, then return its head surface (de-blobbed). If the
        raw string is a bare noun we still canonicalize + cap it.
        """
        raw_stripped = raw.strip().strip(",.!?;:\"'")
        if not raw_stripped:
            return ""
        # Try to locate the substring span in the doc.
        start = doc.text.lower().find(raw_stripped.lower())
        if start == -1:
            # not contiguous (regex captured loosely) -> canonicalize raw
            return _clean_entity(raw_stripped)
        # walk to token boundaries
        end = start + len(raw_stripped)
        toks = [t for t in doc if t.idx >= start and t.idx < end]
        if not toks:
            return _clean_entity(raw_stripped)
        span = doc[toks[0].i : toks[-1].i + 1]
        surf = _np_head_surface(span)
        return _clean_entity(surf) if surf else _clean_entity(raw_stripped)

    # -- regex fallback (unchanged behaviour) ----------------------------
    def _extract_regex(self, text: str) -> List[DeductiveTriple]:
        from ravana.core.deductive_reasoning import parse_deductive_premises
        return parse_deductive_premises(text)


# Convenience module-level function mirroring the parser interface.
def extract_premises(text: str, use_spacy: bool = True) -> List[DeductiveTriple]:
    return DeductivePremiseExtractor(use_spacy=use_spacy).extract(text)
