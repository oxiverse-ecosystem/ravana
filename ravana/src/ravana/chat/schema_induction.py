"""RSI query-time evidence helpers + (later-stage) schema hypotheses.

CORRECTED ROOT CAUSE (parent's live census, 2026-11-23)
------------------------------------------------------
Declarative storage is NOT missing. After two ``process_turn`` calls the graph
already holds trusted causal edges ``gale->rainbow`` and ``mist->thaw`` with
``edge_kind='curated'`` / ``source='conversation'``, written by
``chain_walker.py:878-930`` (``_consolidate_asserted_causal_facts``; called at
``engine.py:10666``) via ``KnowledgeIntake.intake`` with
``Provenance(source_type='user')`` (``intake.py:80`` maps user -> curated).

The observed failure is DOWNSTREAM: the forced-choice query
"...what is true of thaw? Options: sunset, typhoon, mist, hail. Answer
briefly." answered with ``honestly, hail is a bit outside what i know`` --
an OPTION token became the subject and the taught edge was never consulted.

This module supplies pure, fake-testable helpers for the retrieval/articulation
hook. It does NOT import an engine, does NOT boot anything, changes NO gate or
threshold, and is inert unless ``RAVANA_RSI_DECLARATION_STORAGE=1`` is set.

Brain basis: Tse et al. 2007, McClelland et al. 1995, Yassa & Stark 2011,
Gentner 1983, Roediger & Karpicke 2006, Webb/Turney/Dennis 2016. Full citations
and the support-vs-extrapolation ledger are in
``docs/investigations/RSI_DECLARATION_STORAGE.md``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ENV_FLAG",
    "TRUSTED_EDGE_KINDS",
    "TRUSTED_SOURCE",
    "enabled",
    "mask_options",
    "EdgeView",
    "trusted_incident_edges",
    "grounding_from_trusted_edge",
    "Declaration",
    "TrustedTrace",
    "SchemaHypothesis",
    "parse_declaration",
    "induce_schema_hypotheses",
    "admit_for_query",
]

ENV_FLAG = "RAVANA_RSI_DECLARATION_STORAGE"

# Mirrors learn.explanation._hop_provenance + provenance_trusted. These are
# the ALREADY-TRUSTED kinds; nothing here widens them.
TRUSTED_EDGE_KINDS = frozenset(
    {"web_fact", "definition", "curriculum", "curated"})
TRUSTED_SOURCE = "conversation"


def enabled() -> bool:
    """True only when the helper path is explicitly enabled by a parent."""
    try:
        return os.environ.get(ENV_FLAG, "") == "1"
    except Exception:
        return False


# ── Hook 1: keep option tokens out of subject resolution ────────────────────

def mask_options(question: str) -> Tuple[str, List[str], str]:
    """Split a forced-choice prompt into (main, options, masked_text).

    The shipped ``fact_reasoning._split_options`` (fact_reasoning.py:73-104)
    handles letter markers but falls back to ``[;\\n]`` only. The RSI harness
    emits COMMA-separated options, so that fallback yields a single option and
    ``_triplet_mc_answer`` (engine.py:7526) fail-closes; the option tokens are
    then never removed before ``_ground_query`` (engine_generation.py:2029)
    takes the last content words, letting an option win over the real subject.

    Structural fix, no keyword/gate change: mask everything from ``Options:``
    onward for grounding, while returning the options to the answer layer.
    """
    if not isinstance(question, str) or not question.strip():
        return ("", [], "")
    m = re.search(r"\boptions?\s*:\s*", question, re.IGNORECASE)
    if not m:
        return (question.strip(), [], question.strip())
    main = question[:m.start()].strip()
    tail = question[m.end():].strip()
    tail = re.split(r"\banswer\s+briefly\b", tail, flags=re.IGNORECASE)[0]
    tail = tail.strip().rstrip(" .;,")
    parts = [p.strip().rstrip(" .;,") for p in re.split(r"[,;]", tail)
             if p and p.strip()]
    return (main, parts, main)


# ── Hook 2: treat an existing trusted taught edge as grounding evidence ────

@dataclass(frozen=True)
class EdgeView:
    """Minimal read-only shape of a graph edge, usable with fakes."""

    source: str
    relation_type: str
    target: str
    edge_kind: str = ""
    edge_source: str = ""
    confidence: float = 0.0


def trusted_incident_edges(subject: str, edges: Iterable[EdgeView]) -> List[EdgeView]:
    """Return trusted edges incident to ``subject``. Evidence, never a fact.

    A returned edge is an ADMITTED observation: it licences an answer candidate
    but does not itself assert a schema (Webb et al. 2016; skill section 7a).
    """
    if not enabled() or not subject:
        return []
    s = subject.strip().lower()
    out: List[EdgeView] = []
    for e in edges or ():
        if e is None:
            continue
        if e.source.strip().lower() != s and e.target.strip().lower() != s:
            continue
        kind = (e.edge_kind or "").strip().lower()
        src = (e.edge_source or "").strip().lower()
        if kind in TRUSTED_EDGE_KINDS or src == TRUSTED_SOURCE:
            out.append(e)
    return out


def grounding_from_trusted_edge(
    subject: str,
    edges: Iterable[EdgeView],
    candidates: Sequence[str] = (),
    min_confidence: float = 0.0,
) -> Optional[str]:
    """Pick the candidate supported by a trusted taught edge, or ``None``.

    Direction: from ``subject`` take its outgoing trusted edge whose target is
    one of the candidates; from a trusted incoming edge take its source. No
    existing threshold is lowered -- trust is decided by provenance, not score.
    """
    if not enabled():
        return None
    edges_v = trusted_incident_edges(subject, edges)
    if not edges_v:
        return None
    cand = {str(c).strip().lower() for c in (candidates or ()) if str(c).strip()}
    s = subject.strip().lower()
    for e in edges_v:
        if float(e.confidence or 0.0) < float(min_confidence):
            continue
        if e.source.strip().lower() == s and e.target.strip().lower() in cand:
            return str(e.target)
        if e.target.strip().lower() == s and e.source.strip().lower() in cand:
            return str(e.source)
    return None


# ── Later stage (NOT the current bottleneck): unassertable schema hypotheses ─

@dataclass(frozen=True)
class Declaration:
    subject: str
    relation: str
    obj: str
    polarity: int = 1

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.subject.lower(), self.relation, self.obj.lower())


@dataclass(frozen=True)
class TrustedTrace:
    declaration: Declaration
    stamp: Dict[str, Any]
    trace_id: str


@dataclass(frozen=True)
class SchemaHypothesis:
    """Converged relation hypothesis. NEVER an assertable fact (skill 7a)."""

    relation: str
    support: int
    instances: Tuple[Tuple[str, str], ...]
    assertable: bool = False

    def __post_init__(self) -> None:
        if self.assertable:
            object.__setattr__(self, "assertable", False)


def _negated(text: str) -> bool:
    t = " " + text.lower().strip() + " "
    return (" not " in t) or (" never " in t) or ("n't" in t)


def parse_declaration(
    text: str,
    phrase_to_rel: Mapping[str, Tuple[str, int]],
) -> Optional[Declaration]:
    """Canonicalise ``"<s> <phrase> <o>"`` via an injected phrase map."""
    if not isinstance(text, str) or not text.strip() or _negated(text):
        return None
    raw = " ".join(text.strip().lower().rstrip(" ?!.").split())
    hits = [p for p in phrase_to_rel if p and p in raw]
    if not hits:
        return None
    phrase = max(hits, key=len)
    before, sep, after = raw.partition(phrase)
    if not sep:
        return None
    subject, obj = before.strip(), after.strip()
    if not subject or not obj:
        return None
    relation, polarity = phrase_to_rel[phrase]
    if polarity == -1:
        subject, obj = obj, subject
    return Declaration(subject, relation=str(relation), obj=obj,
                       polarity=int(polarity))


def induce_schema_hypotheses(
    traces: Sequence[TrustedTrace],
    min_support: int = 2,
) -> List[SchemaHypothesis]:
    """Aggregate converged traces into *unassertable* schema hypotheses."""
    groups: Dict[str, Dict[str, Any]] = {}
    for tr in traces or ():
        if tr is None or tr.declaration is None:
            continue
        g = groups.setdefault(tr.declaration.relation,
                              {"ids": set(), "pairs": set()})
        g["ids"].add(tr.trace_id)
        g["pairs"].add((tr.declaration.subject.lower(),
                        tr.declaration.obj.lower()))
    out: List[SchemaHypothesis] = []
    for rel, g in sorted(groups.items()):
        if len(g["ids"]) < int(min_support):
            continue
        out.append(SchemaHypothesis(relation=rel, support=len(g["ids"]),
                                    instances=tuple(sorted(g["pairs"])),
                                    assertable=False))
    return out


def admit_for_query(
    hypothesis: SchemaHypothesis,
    observed_traces: Sequence[TrustedTrace],
    min_support: int = 2,
) -> bool:
    """Schemas propose; sources admit. A hypothesis alone never admits."""
    if hypothesis is None or hypothesis.assertable:
        return False
    if hypothesis.support < int(min_support):
        return False
    return any(tr is not None and tr.declaration is not None
               and tr.declaration.relation == hypothesis.relation
               for tr in (observed_traces or ()))
