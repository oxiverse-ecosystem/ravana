#!/usr/bin/env python3
"""RSI relation-target evidence channel — SCRATCH, DEFAULT-OFF, PURE.

A new independent evidence channel for RSI relation induction. It turns a
FROZEN CC0 Wikidata snapshot of real relation assertions into records of the
form ``(subject, relation, target)`` with provenance, rank-based credibility,
and an explicit negation veto. It is a pure function of the supplied snapshot:
no engine boot, no weight-store write, no network at import, and NO use of the
item bank's gold target, type sets, world model, answer key, domains or item
object. Only the public object word list was used to FETCH the snapshot
(see rsi_reltarget_fetch.py); this module never imports the item bank.

Production primitives are imported UNCHANGED and no gate, threshold, item,
prompt, scorer or production file is modified:

  * ravana.chat.schema_induction._is_trusted_edge  (provenance trust predicate)
  * ravana.chat.schema_induction.CONGRUENCE_FLOOR  (0.20, unchanged)
  * ravana.chat.schema_induction.STRICT_MARGIN     (0.10, unchanged)

Brain basis (peer-reviewed; full ledger in the design doc):
  * Gentner (1983) Cognitive Science 7(2):155-170, doi:10.1207/s15516709cog0702_3
    — structure mapping: relations, not object attributes, are what transfer.
  * Eichenbaum & Cohen (2001) Relational Memory Theory; Eichenbaum et al. (2007)
    Nat Rev Neurosci 8:663-676 — hippocampus binds arbitrary relations.
  * Tulving & Thomson (1973) Psychological Review 80(5):352-373 — encoding
    specificity: a type-encoded cue cannot retrieve a relation-target trace.
  * Tomasello (1992) First Verbs; Tomasello (2000) TICS 4(6):209-214 — item /
    verb-island based role acquisition precedes abstract schemas.
  * Baddeley, Onishi & Carey (2011) Proc R Soc B 278:1658-1665,
    doi:10.1098/rspb.2010.2455 — fast mapping is initial, fragile, and refined.
  * Johnson, Hashtroudi & Lindsay (1993) Psychological Bulletin 114(1):3-28 —
    source and item memory dissociate; trust is read from provenance, not score.

Honest boundary: these are computational analogies to a symbolic read path,
not a claim that the brain stores Wikidata statements or forced-choice options.
"""
from __future__ import annotations

import ast
import collections
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ravana.chat.schema_induction import (
    CONGRUENCE_FLOOR,
    STRICT_MARGIN,
    _is_trusted_edge,
)

__all__ = [
    "EVIDENCE_FLAG", "enabled", "RELATION_FAMILIES", "PROPERTY_TO_FAMILY",
    "RANK_CREDIBILITY", "EvidenceRecord", "new_counters", "load_snapshot",
    "relation_target_pick", "wrong_tag_trust", "assert_independence",
]

EVIDENCE_FLAG = "RAVANA_RSI_RELTARGET"
RELATION_FAMILIES = frozenset(
    {"CAUSE", "PART_OF", "PRECEDES", "SUPPORTS", "OPPOSITE", "RESEMBLES"})
# Wikidata P-id (statement property local name) -> RSI relation family.
PROPERTY_TO_FAMILY = {
    "P828": "CAUSE", "P1542": "CAUSE",
    "P361": "PART_OF", "P527": "PART_OF",
    "P155": "PRECEDES", "P156": "PRECEDES",
    "P461": "OPPOSITE",
}
# Wikidata rank -> credibility in [0, 1]. Normalised; never tuned to RSI
# outcomes. Only CONGRUENCE_FLOOR / STRICT_MARGIN decide admission/ranking.
RANK_CREDIBILITY = {"preferred": 1.0, "normal": 0.6, "deprecated": 0.2}
_DEFAULT_CREDIBILITY = 0.6

_WRONG_TAG = {"edge_kind": "unknown", "source": "system", "source_type": "system"}
_NEG_RE = re.compile(
    r"\b(?:is|are|was|were)\s+not\b|\bnot\s+(?:a|an|the)\b|\bnever\b",
    re.IGNORECASE)
_BANNED_NAMES = frozenset({
    "gold", "gold_target", "type_sets", "world_model", "answer_key",
    "counterexample_target", "q_gold", "item_gold", "item", "domain",
    "far_domain", "demo_pairs",
})


def enabled() -> bool:
    """Default-OFF. Only the literal value '1' opts in."""
    try:
        return os.environ.get(EVIDENCE_FLAG, "0") == "1"
    except Exception:
        return False


def new_counters() -> Dict[str, int]:
    return {
        "flag_off": 0, "flag_on": 0, "records_scanned": 0,
        "records_admitted": 0, "records_rejected": 0, "trust_calls": 0,
        "records_vetoed": 0, "scorer_calls": 0, "picks": 0, "errors": 0,
    }


@dataclass(frozen=True)
class EvidenceRecord:
    """One real, licensed relation-target assertion with full provenance."""
    subject: str
    relation: str
    target: str
    surface: str
    license: str
    license_url: str
    source_url: str
    statement_url: str
    rank: str
    credibility: float
    negated: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)


class _ProvenanceEdge:
    """Duck-typed edge so the UNMODIFIED production trust predicate can run."""
    def __init__(self, provenance: Mapping[str, Any]):
        md = provenance if isinstance(provenance, Mapping) else {}
        self.source_metadata = dict(md)
        self.edge_kind = ""
        self.edge_source = ""
        self.confidence = 0.0


def _qid(uri: str) -> str:
    text = str(uri or "")
    return text.rstrip("/").rsplit("/", 1)[-1]


def load_snapshot(path: str) -> List[EvidenceRecord]:
    """Read the frozen CC0 snapshot into records. No network, no item bank."""
    with open(path, encoding="utf-8") as fh:
        doc = json_load(fh)
    lic = str(doc.get("license", "") or "")
    lic_url = str(doc.get("license_url", "") or "")
    source = str(doc.get("source", "") or "")
    out: List[EvidenceRecord] = []
    for b in doc.get("bindings", []):
        ps = str(b.get("ps", {}).get("value", "") or "")
        pid = _qid(ps)
        family = PROPERTY_TO_FAMILY.get(pid)
        if family is None:
            continue
        subj = str(b.get("word", {}).get("value", "") or "").strip()
        obj = (b.get("objLabel", {}) or b.get("obj", {})).get("value", "")
        obj = str(obj or "").strip()
        if not subj or not obj:
            continue
        stmt = str(b.get("stmt", {}).get("value", "") or "")
        rank = str(b.get("rank", {}).get("value", "") or "").strip().lower()
        rank = rank.rsplit("/", 1)[-1] if rank else ""
        cred = RANK_CREDIBILITY.get(rank, _DEFAULT_CREDIBILITY)
        item_qid = _qid(str(b.get("item", {}).get("value", "") or ""))
        obj_qid = _qid(str(b.get("obj", {}).get("value", "") or ""))
        src_url = "https://www.wikidata.org/wiki/" + item_qid if item_qid else ""
        prov = {
            "edge_kind": "curated", "source": source or "wikidata",
            "source_type": "external", "source_id": stmt or item_qid,
            "license": lic, "license_url": lic_url,
            "statement_url": stmt, "rank": rank,
            "subject_qid": item_qid, "target_qid": obj_qid,
        }
        surface = "%s %s %s (%s)" % (
            subj, family.lower().replace("_", " "), obj, pid)
        out.append(EvidenceRecord(
            subject=subj, relation=family, target=obj, surface=surface,
            license=lic, license_url=lic_url, source_url=src_url,
            statement_url=stmt, rank=rank, credibility=cred, negated=False,
            provenance=prov))
    return out


def json_load(fh):
    import json
    return json.load(fh)


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def wrong_tag_trust(edge: Any) -> bool:
    """Call-matched SHAM: only a deliberately wrong-tagged edge passes."""
    md = getattr(edge, "source_metadata", None)
    if not isinstance(md, dict):
        return False
    return (str(md.get("edge_kind", "") or "").strip().lower() == _WRONG_TAG["edge_kind"]
            and str(md.get("source", "") or "").strip().lower() == _WRONG_TAG["source"])


def relation_target_pick(
    subject: str,
    relation: str,
    options: Sequence[str],
    records: Sequence[EvidenceRecord],
    *,
    trust: Callable[[Any], bool] = _is_trusted_edge,
    enabled_flag: Optional[bool] = None,
    counters: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Pure default-off read over supplied relation-target evidence.

    Returns ``None`` when the flag is off (before touching any input). When on,
    admits only provenance-trusted, non-negated assertions whose subject matches
    exactly, whose relation family matches exactly, and whose target is one of
    the displayed options. Negated assertions veto that target. Ranking uses
    the production congruence floor and strict margin unchanged.
    """
    counts = counters if counters is not None else new_counters()
    on = enabled() if enabled_flag is None else bool(enabled_flag)
    if not on:
        counts["flag_off"] += 1
        return None
    counts["flag_on"] += 1
    try:
        display = {_norm(o): str(o) for o in (options or ()) if str(o or "").strip()}
        skey, rkey = _norm(subject), _norm(relation)
        vetoed: set = set()
        scores: Dict[str, float] = {}
        admitted_ids: List[str] = []
        for rec in records or ():
            counts["records_scanned"] += 1
            if not isinstance(rec, EvidenceRecord):
                counts["records_rejected"] += 1
                continue
            if _norm(rec.relation) != rkey or _norm(rec.subject) != skey:
                counts["records_rejected"] += 1
                continue
            counts["trust_calls"] += 1
            if not trust(_ProvenanceEdge(rec.provenance)):
                counts["records_rejected"] += 1
                continue
            if rec.license != "CC0-1.0" or not rec.source_url or not rec.statement_url:
                counts["records_rejected"] += 1
                continue
            key = _norm(rec.target)
            if rec.negated or bool(_NEG_RE.search(rec.surface or "")):
                counts["records_vetoed"] += 1
                if key in display:
                    vetoed.add(key)
                continue
            if key not in display:
                counts["records_rejected"] += 1
                continue
            admitted_ids.append(rec.statement_url or rec.surface)
            scores[key] = max(scores.get(key, 0.0), float(rec.credibility))
            counts["records_admitted"] += 1
        result = {
            "subject": str(subject or ""), "relation": str(relation or ""),
            "options": tuple(str(o) for o in (options or ())),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "vetoed": tuple(sorted(vetoed)), "admitted": tuple(admitted_ids),
            "pick": None, "reason": "", "margin": None,
            "congruence_floor": CONGRUENCE_FLOOR, "strict_margin": STRICT_MARGIN,
        }
        if not scores:
            result["reason"] = "no_admitted_subject_evidence"
            return result
        eligible = {k: v for k, v in scores.items()
                    if v >= CONGRUENCE_FLOOR and k not in vetoed}
        if not eligible:
            result["reason"] = ("all_candidates_negation_vetoed" if vetoed
                                else "no_trusted_claim_above_floor")
            return result
        counts["scorer_calls"] += 1
        ranked = sorted(eligible.items(), key=lambda kv: (-kv[1], kv[0]))
        top_key, top_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = round(top_score - second, 4)
        result["margin"] = margin
        if margin < STRICT_MARGIN:
            result["reason"] = "strict_margin_not_met"
            return result
        result["pick"] = display[top_key]
        result["reason"] = "admitted_relation_target_winner"
        counts["picks"] += 1
        return result
    except Exception as exc:  # surfaced, never silently swallowed
        counts["errors"] += 1
        counts["last_error"] = repr(exc)
        return None


def assert_independence(source: Optional[str] = None) -> bool:
    """AST proof that the module body names no answer-key identifier.

    Reads this file's source (or a supplied snippet) and checks every Name and
    Attribute identifier against the banned set. Docstrings are string
    constants and are intentionally not treated as references.
    """
    if source is None:
        with open(__file__, encoding="utf-8") as fh:
            source = fh.read()
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    leaked = sorted(n for n in names if n in _BANNED_NAMES)
    if leaked:
        raise AssertionError("answer-key identifiers present: %s" % leaked)
    return True
