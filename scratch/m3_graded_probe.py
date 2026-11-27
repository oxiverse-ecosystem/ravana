#!/usr/bin/env python3
"""M3 - negation-aware graded-probe reader over SUPPLIED evidence.

SCRATCH / DEFAULT-OFF.  This module is never imported by production RAVANA.
It wires nothing, boots no engine, writes no store and issues no turn.  It is a
pure function of the evidence handed to it by a caller, so a harness can stage
the exact inputs and a boot-free self-test can exercise the whole contract.

Why this exists (failure history)
---------------------------------
The TRA-K read-surface hook (``docs/investigations/TRA_K_READ_SURFACE_NEGATIVE.md``)
was flat in all arms: READ == OFF == SHAM == 0.  It scanned the live graph for a
redundant write-time ``type_label`` stamp; when that stamp was absent at the
first graded read the reader failed closed, emitted a hedge and scored
``no_answer``.  The prior design's pre-registered fork (F3) selected the
"fact absent at that read" branch.  This M3 reader does NOT re-scan the graph:
the caller supplies the evidence.  It accepts only

  * M1 records - identity-keyed, provenance-bearing prior-act records.  These
    must clear the UNMODIFIED ``source_evaluation.evaluate_source`` gate
    (decision == ENDORSE).  Familiarity, reliability and the answer key can
    never substitute for an explicit source attribution.
  * M2 subject-side evidence - typed-slot views binding the SUBJECT to a
    source-native type label.  These must clear the UNMODIFIED
    ``schema_induction._is_trusted_edge`` provenance predicate.

Both record classes carry the label that was actually ENCODED (the graph node's
own source-native label, case preserved); a redundant stamp is accepted only
when it agrees.  A provenance-admitted NEGATED record for (subject, label) is
an explicit veto: that label cannot be picked even with positive support.

Brain basis (verified against publisher/index records; full ledger in
``scratch/m3_citation_ledger.json`` and the design doc)
-------------------------------------------------------
* Tulving & Thomson (1973) Psychological Review 80(5):352-373,
  doi:10.1037/h0020071 - encoding specificity: the encoding operations decide
  which retrieval cues are effective.  -> use the encoded node label, not a
  parallel stamp the test never sees.
* Morris, Bransford & Franks (1977) JVLVB 16(5):519-533,
  doi:10.1016/S0022-5371(77)80016-9 - transfer-appropriate processing:
  retrieval is best when test operations overlap encoding operations.
* Bransford & Johnson (1972) JVLVB 11(6):717-726,
  doi:10.1016/S0022-5371(72)80006-9 - a schema supplied BEFORE the passage
  changes recall; schema operates as a retrieval cue, not just an encoding aid.
* Anderson & Pichert (1978) JVLVB 17(1):1-12 - a schema shift at recall is
  sufficient to change what is retrieved: schema-triggered retrieval exists
  independently of encoding.  -> M2 evidence is a legitimate read cue.
* Tse et al. (2007) Science 316(5821):76-82, doi:10.1126/science.1135935 -
  one-trial assimilation requires a pre-existing schema.  -> M2 is a separate
  necessary condition, not a restatement of M1.
* MacMillan, Peer & Garnham (2001) Memory & Cognition 29(7):960-967,
  doi:10.3758/BF03195758 - explicit negation reduces the accessibility of the
  negated concept.  -> an admitted negation must veto that label.  Honest
  boundary: Giora (2008, doi:10.1515/9783110198843.2.129) shows negated
  concepts are not ALWAYS suppressed, so the veto is scoped to an explicitly
  negated, provenance-admitted (subject, label), never a global word ban.
* Johnson, Hashtroudi & Lindsay (1993) Psychological Bulletin 114(1):3-28,
  doi:10.1037/0033-2909.114.1.3 - source and item memory dissociate.  -> trust
  is checked on provenance, never on confidence/familiarity.
* Levelt, Roelofs & Meyer (1999) Behav Brain Sci 22(1):1-38,
  doi:10.1017/S0140525X99001776 - production commits to a proposition before
  form encoding.  -> structural deferral: an already-admitted production pick
  is returned unchanged; this reader cannot overwrite it.

Honest boundary: these are computational analogies to a symbolic read path,
not a claim that the brain has typed-slot edges or a forced-choice grader.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# Production primitives are imported UNCHANGED.  No gate, threshold, item,
# prompt, answer key or scoring function is re-implemented here.
from ravana.chat.schema_induction import (
    CONGRUENCE_FLOOR,
    STRICT_MARGIN,
    TYPED_SLOT_REL,
    _is_trusted_edge,
)
from ravana.chat.source_evaluation import (
    ClaimContext,
    Decision,
    SourceStatus,
    TrustClass,
    evaluate_source,
)

__all__ = [
    "M3_ENV_FLAG", "m3_enabled", "ISA_PREDICATES", "WRONG_TAG",
    "EvidenceBundle", "ProbeDecision", "wrong_tag_trust", "read_graded_probe",
    "graded_probe_pick", "new_counters",
]

M3_ENV_FLAG = "RAVANA_M3_GRADED_PROBE"
# Mirrors scripts/tra_k_items.ISA_PREDICATES exactly; no widening.
ISA_PREDICATES = frozenset({"is_a", "type_of", "instance_of"})

# Wrong-tag SHAM control: same signature and the same number of predicate calls
# as the production trust predicate, but no earned record carries this tag.
WRONG_TAG: Dict[str, str] = {
    "edge_kind": "unknown", "source": "system", "source_type": "system",
}

_NEGATION_RE = re.compile(
    r"\b(?:is|are|was|were|isn't|aren't|wasn't|weren't)\s+not\b"
    r"|\bnot\s+(?:a|an|the)\b"
    r"|\b(?:is|are|was|were)\s+not\s+(?:a|an|the)\b",
    re.IGNORECASE,
)


def m3_enabled() -> bool:
    """Default-OFF.  Only the literal value ``1`` opts in."""
    try:
        return os.environ.get(M3_ENV_FLAG, "0") == "1"
    except Exception:
        return False


def new_counters() -> Dict[str, int]:
    """Fresh counter dict.  READ and SHAM must match on every 'call' key."""
    return {
        "flag_off": 0,
        "flag_on": 0,
        "production_calls": 0,
        "scorer_calls": 0,
        "trust_calls": 0,
        "records_scanned": 0,
        "m1_scanned": 0,
        "m2_scanned": 0,
        "records_admitted": 0,
        "records_rejected": 0,
        "veto_hits": 0,
        "deferred_to_production": 0,
        "picks": 0,
        "errors": 0,
    }


@dataclass(frozen=True)
class EvidenceBundle:
    """Everything the reader is allowed to see.

    Both collections are plain mappings so a harness can stage them without
    importing this module.  There is deliberately NO ``item``, ``gold``,
    ``world_model`` or ``type_sets`` field: the reader's input signature is
    audited by the self-test for answer-key independence.

    Common record keys
      ``record_id``            stable identity key (audit only)
      ``subject``              subject, source-native case preserved
      ``relation``             is_a / type_of / instance_of (alias: relation_type, predicate)
      ``label``                SOURCE-NATIVE target label (alias: target, object)
      ``provenance``           mapping with edge_kind/source/source_type/source_id
      ``confidence``           stored edge confidence (graded signal)
      ``negated``              explicit negation flag (optional; else read from ``surface``)
      ``surface``              optional teaching surface, scanned for explicit negation
      ``stamp`` / ``type_label`` optional redundant stamp; must agree with ``label``

    M1-only keys
      ``source_status``        explicit credibility token (e.g. 'credible')
      ``prior_decision``       ENDORSE / ABSTAIN / WITHDRAW
      ``invalidated``          bool
      ``cue_present``          bool (default True)
      ``origin_id``            defaults to provenance['source_id']
    """

    m1_records: Tuple[Mapping[str, Any], ...] = ()
    m2_subject_evidence: Tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_lists(cls, m1_records: Optional[Sequence[Mapping[str, Any]]] = None,
                   m2_subject_evidence: Optional[Sequence[Mapping[str, Any]]] = None) -> "EvidenceBundle":
        return cls(tuple(m1_records or ()), tuple(m2_subject_evidence or ()))


@dataclass(frozen=True)
class ProbeDecision:
    """Auditable result.  ``pick`` is the bare source-native displayed option."""

    pick: Optional[str]
    reason: str
    subject: str
    options: Tuple[str, ...]
    scores: Mapping[str, float] = field(default_factory=dict)
    vetoed: Tuple[str, ...] = ()
    admitted: Tuple[str, ...] = ()
    rejected: Tuple[Tuple[str, str], ...] = ()
    congruence_floor: float = CONGRUENCE_FLOOR
    strict_margin: float = STRICT_MARGIN
    margin: Optional[float] = None
    source: str = "m3"


class _ProvenanceEdge:
    """Duck-typed edge so the UNMODIFIED production trust predicate can run."""

    def __init__(self, provenance: Mapping[str, Any]):
        md = provenance if isinstance(provenance, Mapping) else {}
        self.source_metadata = dict(md)
        self.edge_kind = ""
        self.edge_source = ""
        self.confidence = 0.0


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def _provenance_of(record: Mapping[str, Any]) -> Mapping[str, Any]:
    pv = record.get("provenance")
    if isinstance(pv, Mapping):
        return pv
    # Flattened provenance keys at top level (harness-staged shape).
    flat = {k: record.get(k) for k in
            ("edge_kind", "kind", "source", "edge_source", "source_type",
             "source_id") if k in record}
    if "source" in flat and "edge_source" not in flat:
        flat["edge_source"] = flat["source"]
    return flat


def _source_native_label(record: Mapping[str, Any]) -> str:
    """The label that was encoded: node label first, never a stamp."""
    for key in ("label", "target", "object"):
        val = record.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _relation_of(record: Mapping[str, Any]) -> str:
    # Fail closed on an EXPLICIT empty/null relation; only default to is_a when
    # no relation key was supplied at all (duck-typed persisted views).
    for key in ("relation", "relation_type", "predicate"):
        if key in record:
            val = record.get(key)
            if val is None:
                return ""
            return str(val).strip().casefold().replace(" ", "_")
    return TYPED_SLOT_REL


def _is_negated(record: Mapping[str, Any]) -> bool:
    """Explicit negation only: a flag OR a negating teaching surface.

    A bare word 'not' elsewhere is not enough; the surface must contain a
    negated copula.  Absent both, the record is positive.
    """
    if "negated" in record:
        return bool(record.get("negated"))
    surface = record.get("surface") or record.get("claim") or ""
    return bool(surface) and bool(_NEGATION_RE.search(str(surface)))


def _confidence_of(record: Mapping[str, Any]) -> float:
    try:
        conf = float(record.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if conf != conf or conf in (float("inf"), float("-inf")):
        return 0.0
    return conf


def _decision_from(value: Any) -> Optional[Decision]:
    if isinstance(value, Decision):
        return value
    if isinstance(value, str):
        try:
            return Decision(value.strip().upper())
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class _Fact:
    record_id: str
    subject_key: str
    label_key: str
    label: str
    confidence: float
    negated: bool


def _admit_m1(record: Mapping[str, Any], trust: Callable[[Any], bool],
              counters: MutableMapping[str, int]) -> Optional[_Fact]:
    """M1: provenance-trusted AND source-evaluation ENDORSE."""
    prov = _provenance_of(record)
    if not trust(_ProvenanceEdge(prov)):
        return None
    status = str(record.get("source_status") or "").strip().lower()
    trust_class = (TrustClass.CREDIBLE if status in
                   {"credible", "trusted", "verified", "reliable"}
                   else TrustClass.DISCREDITED if status in
                   {"discredited", "unreliable", "invalidated"}
                   else TrustClass.UNKNOWN)
    token = status if trust_class is TrustClass.CREDIBLE else None
    origin = str(record.get("origin_id") or prov.get("source_id") or "").strip()
    src = SourceStatus(
        origin_id=origin,
        source_type=str(prov.get("source_type") or "unknown"),
        trust_class=trust_class,
        active=bool(record.get("active", True)),
        credibility_token=token,
    )
    ctx = ClaimContext(
        source=src,
        prior_decision=_decision_from(record.get("prior_decision")),
        cue_present=bool(record.get("cue_present", True)),
        invalidated=bool(record.get("invalidated", False)),
        trusted_source_ids=None,
        familiarity=float(record.get("familiarity", 0.0) or 0.0),
    )
    claim = str(record.get("claim") or record.get("record_id") or "")
    verdict = evaluate_source(claim, ctx)
    if verdict.decision is not Decision.ENDORSE:
        counters["records_rejected"] += 1
        return None
    return _fact_from(record, counters)


def _fact_from(record: Mapping[str, Any], counters: MutableMapping[str, int]) -> Optional[_Fact]:
    label = _source_native_label(record)
    if not label:
        counters["records_rejected"] += 1
        return None
    # Redundant stamp is a consistency check, never the cue itself.
    stamp = str(record.get("stamp") or record.get("type_label") or "").strip()
    if stamp and _norm(stamp) != _norm(label):
        counters["records_rejected"] += 1
        return None
    return _Fact(
        record_id=str(record.get("record_id") or record.get("id") or ""),
        subject_key=_norm(record.get("subject")),
        label_key=_norm(label),
        label=label,
        confidence=_confidence_of(record),
        negated=_is_negated(record),
    )


def read_graded_probe(
    subject: str,
    options: Sequence[str],
    evidence: EvidenceBundle,
    *,
    trust: Callable[[Any], bool] = _is_trusted_edge,
    congruence_floor: float = CONGRUENCE_FLOOR,
    strict_margin: float = STRICT_MARGIN,
    counters: Optional[MutableMapping[str, int]] = None,
) -> ProbeDecision:
    """Pure graded read over SUPPLIED evidence.  No graph, item, gold or I/O.

    Admission: M1 records clear the unmodified source-evaluation ENDORSE gate;
    M2 subject-side evidence clears the unmodified provenance trust predicate.
    Both then require an is_a-family relation, a source-native label, and a
    label present in the displayed option set.  A contradictory stamp is
    refused.  Admitted negated facts veto their (subject, label).

    Ranking uses the PRODUCTION congruence floor and strict margin; no new
    threshold is introduced.  No pick is returned on a tie or below floor.
    """
    counts = counters if counters is not None else new_counters()
    display = {_norm(o): str(o) for o in (options or ()) if str(o or "").strip()}
    skey = _norm(subject)
    facts: List[_Fact] = []

    for rec in evidence.m1_records:
        counts["records_scanned"] += 1
        counts["m1_scanned"] += 1
        counts["trust_calls"] += 1
        if _relation_of(rec) not in ISA_PREDICATES:
            counts["records_rejected"] += 1
            continue
        fact = _admit_m1(rec, trust, counts)
        if fact is not None:
            facts.append(fact)
            counts["records_admitted"] += 1

    for rec in evidence.m2_subject_evidence:
        counts["records_scanned"] += 1
        counts["m2_scanned"] += 1
        counts["trust_calls"] += 1
        if _relation_of(rec) not in ISA_PREDICATES:
            counts["records_rejected"] += 1
            continue
        prov = _provenance_of(rec)
        if not trust(_ProvenanceEdge(prov)):
            counts["records_rejected"] += 1
            continue
        fact = _fact_from(rec, counts)
        if fact is not None:
            facts.append(fact)
            counts["records_admitted"] += 1

    scores: Dict[str, float] = {}
    vetoed: List[str] = []
    admitted: List[str] = []
    for fact in facts:
        admitted.append(fact.record_id)
        if fact.subject_key != skey or fact.label_key not in display:
            continue
        if fact.negated:
            counts["veto_hits"] += 1
            if fact.label_key not in vetoed:
                vetoed.append(fact.label_key)
            continue
        scores[fact.label_key] = max(scores.get(fact.label_key, 0.0), fact.confidence)

    eligible = {k: v for k, v in scores.items()
                if v >= congruence_floor and k not in set(vetoed)}
    decision = ProbeDecision(
        pick=None, reason="", subject=str(subject or ""),
        options=tuple(str(o) for o in (options or ())),
        scores={k: round(v, 4) for k, v in scores.items()},
        vetoed=tuple(vetoed), admitted=tuple(admitted),
        congruence_floor=congruence_floor, strict_margin=strict_margin,
    )
    if not scores:
        return ProbeDecision(**{**decision.__dict__, "reason": "no_admitted_subject_evidence"})
    if not eligible:
        reason = ("all_candidates_negation_vetoed" if vetoed and scores
                  else "no_trusted_label_above_floor")
        return ProbeDecision(**{**decision.__dict__, "reason": reason})
    ranked = sorted(eligible.items(), key=lambda kv: (-kv[1], kv[0]))
    top_key, top_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = round(top_score - second, 4)
    if margin < strict_margin:
        return ProbeDecision(**{**decision.__dict__, "reason": "strict_margin_not_met",
                                "margin": margin})
    return ProbeDecision(**{**decision.__dict__, "pick": display[top_key],
                            "reason": "admitted_graded_probe_winner", "margin": margin})


def graded_probe_pick(
    subject: str,
    options: Sequence[str],
    evidence: EvidenceBundle,
    *,
    production_pick: Optional[Callable[[str, Sequence[str]], Optional[str]]] = None,
    trust: Callable[[Any], bool] = _is_trusted_edge,
    enabled: Optional[bool] = None,
    counters: Optional[MutableMapping[str, int]] = None,
) -> Optional[ProbeDecision]:
    """Default-off boundary hook with structural deferral to production.

    1. OFF (default): return ``None`` before touching evidence or production.
    2. ON: call the caller-supplied UNMODIFIED production reader FIRST and,
       if it admits a pick, return that pick untouched (``source='production'``).
       The graded path never scores an option production already handled.
    3. Otherwise run :func:`read_graded_probe` and return its decision.

    ``trust`` is the ONLY manipulation knob: READ uses the production
    predicate, SHAM uses :func:`wrong_tag_trust` and admits nothing while
    performing the same scans and scorer calls.
    """
    counts = counters if counters is not None else new_counters()
    on = m3_enabled() if enabled is None else bool(enabled)
    if not on:
        counts["flag_off"] += 1
        return None
    counts["flag_on"] += 1
    try:
        counts["production_calls"] += 1
        existing = production_pick(subject, options) if callable(production_pick) else None
        if existing is not None:
            counts["deferred_to_production"] += 1
            return ProbeDecision(
                pick=str(existing), reason="deferred_to_production",
                subject=str(subject or ""),
                options=tuple(str(o) for o in (options or ())),
                source="production",
            )
        counts["scorer_calls"] += 1
        decision = read_graded_probe(subject, options, evidence, trust=trust,
                                     counters=counts)
        if decision.pick is not None:
            counts["picks"] += 1
        return decision
    except Exception as exc:  # surfaced in counters, never silently swallowed
        counts["errors"] += 1
        counts["last_error"] = repr(exc)  # type: ignore[assignment]
        return None


def wrong_tag_trust(edge: Any) -> bool:
    """Call-matched SHAM: only a deliberately wrong-tagged edge passes."""
    md = getattr(edge, "source_metadata", None)
    if not isinstance(md, dict):
        return False
    kind = str(md.get("edge_kind", "") or "").strip().lower()
    src = str(md.get("source", "") or "").strip().lower()
    return kind == WRONG_TAG["edge_kind"] and src == WRONG_TAG["source"]
