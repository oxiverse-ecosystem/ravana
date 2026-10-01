"""Agentic tool output is INTERNAL evidence — never a user-facing reply.

The defect (FIX-RV-17): the agentic hands stashed a tool result as
``[agentic:<tool>] <raw payload>`` and the end-of-turn block concatenated it
straight onto RAVANA's reply. The payload was a 1200-character PREFIX of an
IntentForge JSON envelope, so the user received a reply with a JSON document
sliced off mid-string and pasted after it.

Three separate faults lived in that one line, and this module exists to own all
three:

  1. BOUNDARY. A tool result is evidence for cognition. It is parsed into
     structured records here, and the raw payload is dropped. Nothing in this
     module returns text intended for a reply.

  2. RELEVANCE. A search that returns nothing about the query is a non-answer.
     Records are kept only when they cover enough of the query's own content
     terms to be about it. The filter is coverage arithmetic over the query's
     terms — no topic list, no per-topic rules.

  3. HONESTY. An unreachable or unparseable payload becomes a negative result
     (zero usable records), never a fabricated one.

The query-term weighting uses distribution WITHIN the returned result set, so
"what is the capital of france" keeps a document that mentions France, while a
result that only echoes a word shared by most of the other results carries no
information about this query and is dropped.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Marker prefixes the tool registry writes in front of a payload. Parsed as a
# source label, never shown to the user.
_SOURCE_MARKERS = ("web:intentforge", "web:fallback", "site", "script", "git")

_WORD_RE = re.compile(r"[a-z][a-z']{2,}")

# Closed-class function words. A query term that is grammatical scaffolding
# ("when", "was", "there") cannot discriminate between documents, so it never
# counts toward coverage. This is function-word grammar, not topic vocabulary.
_FUNCTION_WORDS = frozenset({
    "and", "are", "but", "can", "could", "did", "does", "for", "from",
    "had", "has", "have", "her", "him", "his", "how", "into", "its",
    "let", "may", "might", "more", "most", "not", "now", "off", "one",
    "our", "out", "own", "she", "should", "some", "such", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "too", "under", "until", "use", "very", "was",
    "were", "what", "when", "where", "which", "while", "who", "whom",
    "whose", "why", "will", "with", "would", "you", "your", "about",
    "after", "again", "against", "all", "also", "any", "because", "before",
    "being", "between", "both", "did", "doing", "down", "during", "each",
    "few", "get", "got", "just", "like", "make", "many", "more", "most",
    "much", "must", "need", "only", "other", "over", "same", "shall",
    "should", "since", "some", "still", "such", "than", "them", "then",
    "these", "they", "this", "those", "through", "told", "took", "very",
    "want", "well", "went", "were", "what", "when", "which", "while",
    "will", "with", "would", "yet", "you", "your",
})


def _terms(text: str) -> List[str]:
    """Content terms of `text`: alphabetic, >=3 chars, not function words."""
    if not text:
        return []
    return [w for w in _WORD_RE.findall(text.lower())
            if w not in _FUNCTION_WORDS]


@dataclass
class EvidenceRecord:
    """One piece of tool evidence, already parsed and already judged relevant.

    `content` is text FOR cognition (graph expansion, source-trust ranking).
    It is never appended to a reply.
    """
    source: str
    title: str = ""
    url: str = ""
    content: str = ""
    authority: Optional[float] = None
    score: Optional[float] = None
    is_local: bool = False
    #: fraction of the query's content terms this record actually covers
    coverage: float = 0.0
    #: per-term information weight, higher = more discriminating
    weights: Dict[str, float] = field(default_factory=dict)

    def as_log(self) -> Dict[str, Any]:
        """A compact, structured log entry — no raw payload, bounded fields."""
        return {
            "source": self.source,
            "title": (self.title or "")[:200],
            "url": (self.url or "")[:400],
            "coverage": round(self.coverage, 4),
            "authority": self.authority,
            "is_local": self.is_local,
        }


def _split_marker(raw: str) -> Tuple[str, str]:
    """Separate the `[source] payload` prefix the tool registry writes."""
    if not raw:
        return "", ""
    text = raw.strip()
    for marker in _SOURCE_MARKERS:
        prefix = f"[{marker}]"
        if text.startswith(prefix):
            return marker, text[len(prefix):].strip()
    if text.startswith("["):
        end = text.find("]")
        if 0 < end < 60:
            return text[1:end], text[end + 1:].strip()
    return "", text


def _coerce_authority(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _records_from_envelope(source: str, payload: str) -> List[EvidenceRecord]:
    """Parse an IntentForge-style JSON envelope into records.

    The envelope is parsed WHOLE. A truncated prefix cannot be parsed, which is
    why the tool registry must stop slicing payloads mid-document.
    """
    try:
        data = json.loads(payload)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    out: List[EvidenceRecord] = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or item.get("snippet") or ""
        title = item.get("title") or ""
        if not (content or title):
            continue
        out.append(EvidenceRecord(
            source=source,
            title=str(title),
            url=str(item.get("url") or ""),
            content=str(content),
            authority=_coerce_authority(item.get("authority")),
            score=_coerce_authority(item.get("score")),
            is_local=bool(item.get("is_local", False)),
        ))
    return out


def _records_from_text(source: str, payload: str) -> List[EvidenceRecord]:
    """A non-JSON payload (a fetched page, script output) is ONE record.

    It is not a result SET, so it gets no relevance filtering — there is no
    set to be a non-answer within. It still becomes structured state rather
    than reply text.
    """
    body = payload.strip()
    if not body:
        return []
    return [EvidenceRecord(source=source, title="", url="", content=body)]


def _weight_terms(query_terms: Iterable[str],
                  records: List[EvidenceRecord]) -> Dict[str, float]:
    """Weight each query term by how much it DISCRIMINATES within this result set.

    A term present in nearly every document separates nothing, so it carries
    almost no weight; a term that appears in one document carries a lot. The
    distribution comes from the result set itself, so this needs no corpus and
    no tuned constant: a rare-in-this-set term is informative by construction.
    """
    terms = list(dict.fromkeys(query_terms))
    if not terms:
        return {}
    total = max(len(records), 1)
    doc_freq: Counter = Counter()
    for record in records:
        present = set(_terms(f"{record.title} {record.content} {record.url}"))
        for term in terms:
            if term in present:
                doc_freq[term] += 1
    weights: Dict[str, float] = {}
    for term in terms:
        df = doc_freq.get(term, 0)
        if df == 0:
            # Appears in no document: carries no evidence for this query.
            weights[term] = 0.0
            continue
        weights[term] = math.log((total + 1.0) / (df + 0.5))
    return weights


def _coverage(weights: Dict[str, float], record: EvidenceRecord) -> Tuple[float, Dict[str, float]]:
    """Fraction of the query's total term weight that this record covers."""
    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        # Every query term is absent from the whole result set, so there is no
        # basis to call anything relevant.
        return 0.0, {}
    present = set(_terms(f"{record.title} {record.content} {record.url}"))
    hit = {t: w for t, w in weights.items() if t in present and w > 0.0}
    return sum(hit.values()) / total_weight, hit


def _required_terms(query_terms: List[str]) -> int:
    """How many distinct query terms a record must cover to be about the query.

    A one- or two-term query ("capital of france") has almost no room to
    discriminate, so requiring full coverage would empty the result set; one
    term is the honest minimum there. As the query grows more specific, the
    bar rises with it, which is what separates a document ABOUT the query from
    one that merely echoes a word the query happens to contain.
    """
    distinct = len(set(query_terms))
    if distinct <= 2:
        return 1
    return (distinct + 1) // 2


def relevant_records(records: List[EvidenceRecord],
                     query: str) -> List[EvidenceRecord]:
    """Keep only records that are actually about `query`.

    A record qualifies when it covers at least half of the query's DISTINCT
    content terms (one, for a short query) AND carries non-trivial share of the
    query's discriminating weight. A document that echoes a single common word
    of a multi-part query is a non-answer and is dropped.
    """
    if not records:
        return []
    query_terms = _terms(query)
    if not query_terms:
        # No query content to judge relevance against: keep everything, since
        # discarding on an empty basis would be an unearned claim of no answer.
        return records

    needed = _required_terms(query_terms)
    weights = _weight_terms(query_terms, records)
    kept: List[EvidenceRecord] = []
    for record in records:
        present = set(_terms(f"{record.title} {record.content} {record.url}"))
        distinct_hits = sum(1 for t in set(query_terms) if t in present)
        coverage, hit = _coverage(weights, record)
        record.coverage = coverage
        record.weights = hit
        if distinct_hits >= needed and coverage > 0.0:
            kept.append(record)
    return kept


def parse_tool_output(raw: str, query: str = "") -> List[EvidenceRecord]:
    """Parse one tool result into RELEVANT, structured evidence records.

    Returns an empty list for an unreachable, unparseable, or wholly
    off-query payload. The caller gets a count and structured records — never a
    string to paste onto a reply.
    """
    source, payload = _split_marker(raw or "")
    if not payload:
        return []
    records = _records_from_envelope(source, payload)
    if not records:
        records = _records_from_text(source, payload)
    if not records:
        return []
    return relevant_records(records, query)


def ingest_into_graph(expander: Any, records: List[EvidenceRecord]) -> Dict[str, int]:
    """Write evidence records into cognitive state as concepts and edges.

    This is what "parse it into cognitive state" means concretely: the
    evidence's own words become concepts, and the expander's own
    nearest-neighbour wiring decides which edges they earn. No relationship is
    asserted here that the embedding has not already earned, and nothing is
    retained that could be mistaken for reply text.

    `expander` is the engine's own concept-expansion entry point (it owns the
    GloVe-backed vector store and the garbage-word filter). It is a callable so
    this module never has to know how the graph is built.

    Returns a small count summary for the spike log.
    """
    summary = {"records": len(records), "concepts_added": 0}
    if not records or expander is None:
        return summary
    for record in records:
        # Title and URL go in too: the topic of a source is part of the source.
        text = " ".join((record.title, record.url, record.content))
        try:
            summary["concepts_added"] += int(expander(text) or 0)
        except Exception:
            # Evidence ingestion is additive: a graph failure must never break
            # the user's turn.
            continue
    return summary


def evidence_log_entry(tool: str, arg: str, query: str,
                       raw: str) -> Dict[str, Any]:
    """A structured record of an agentic action for the action log.

    Replaces logging a 200-character prefix of a raw JSON envelope: this keeps
    the source, the query, and how many records actually survived relevance.
    """
    records = parse_tool_output(raw, query or arg)
    return {
        "tool": tool,
        "arg": arg,
        "records": len(records),
        "records_kept": [r.as_log() for r in records],
    }
