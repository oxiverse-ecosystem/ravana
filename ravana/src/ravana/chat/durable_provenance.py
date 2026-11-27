"""Unified durable-provenance adapter (M1).  DEFAULT-OFF.

This module is the identity-keyed substrate that keeps FOUR things bound
together across a ``process_turn`` and across a cold reload:

    subject/claim identity  -- one short sha1 claim anchor, the same anchor the
                               g05 provenance write and the prior-act trace use
    provenance              -- origin / source_type / edge_kind / trust_class,
                               read from the engine's OWN staged source record
    prior endorsement       -- the engine's OWN emitted decision at the act,
                               plus an ``ever_endorsed`` bit that survives
                               later WITHDRAW decisions
    negation status         -- ``negated``, monotonically True once an
                               invalidation cue has been staged

Nothing here reads an item bank, an expected answer, a gold label, a threshold
or a scoring rule.  ``evaluate_source`` and every gate/threshold are untouched:
the adapter only (a) persists what the engine already decided and stored, and
(b) re-stages a durable prior act into the ClaimContext BEFORE the existing
gate runs, exactly as the existing default-off prior-act trace does -- except
that this record survives a cold reload.

Brain grounding (each independently verified; full ledger in the doc)
---------------------------------------------------------------------
* Teyler & DiScenna (1986), Behav Neurosci 100:147-152; Teyler & Rudy (2007),
  Hippocampus 17:1171-1177 -- hippocampal INDEXING: a short, stable index binds
  a distributed episode; a partial cue re-accesses the whole trace.  -> one
  short claim anchor indexes identity + provenance + act + negation together.
* Johnson, Hashtroudi & Lindsay (1993), Psychol Bull 114(1):3-28,
  doi:10.1037/0033-2909.114.1.3 -- SOURCE MONITORING: internal source judgments
  are evaluated at retrieval from specific remembered characteristics and are
  dissociable from item familiarity; frontal systems support the systematic
  form.  -> the record stores origin/kind/status and the emitted decision,
  never a reliability or familiarity shortcut, and is read at staging time.
* Nader, Schafe & Le Doux (2000), Nature 406:722-726,
  doi:10.1038/35021052 -- RECONSOLIDATION: retrieval returns a consolidated
  trace to a labile state in which new information is written into the OLD
  trace.  -> the act is written at the act and updated in place; a later
  invalidation is merged into the SAME keyed record, never inferred anew.
* McClelland, McNaughton & O'Reilly (1995), Psychol Rev 102(3):419-457 --
  COMPLEMENTARY LEARNING SYSTEMS: a fast index plus a slow durable mirror.
  -> the in-memory ledger is mirrored to an atomically written sidecar.

The symbolic dict/JSON instantiation is an engineering analogue, not a neural
claim.  Support vs extrapolation is spelled out in the design doc.

Public API
----------
``enabled()``               flag read (RAVANA_DURABLE_PROVENANCE == "1")
``claim_key(claim)``        stable identity anchor (text only)
``ProvenanceRecord``        immutable, JSON-round-trippable record
``DurableProvenanceLedger`` in-memory identity -> record map
``DurableProvenanceStore``  ledger + atomic sidecar persistence
``store_for_engine(engine)``lazy, cached store bound to the engine's data_dir
``record_decision(...)``    write the engine's act + provenance + negation

Importing this module has no side effects and changes nothing while the flag
is unset.  It requires RAVANA_SOURCE_EVAL=1 to emit decisions (it is a
substrate for the existing source-monitoring gate, not a new gate).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "FLAG_ENV", "FILE_NAME", "STATE_KEY", "enabled", "claim_key",
    "subject_key", "ProvenanceRecord", "DurableProvenanceLedger",
    "DurableProvenanceStore", "store_for_engine", "record_decision",
]

FLAG_ENV = "RAVANA_DURABLE_PROVENANCE"
FILE_NAME = "durable_provenance.json"
STATE_KEY = "durable_provenance_ledger"
_VERSION = 1


def enabled() -> bool:
    """True only for the explicit opt-in value ``1``; default OFF."""
    return os.environ.get(FLAG_ENV, "0") == "1"


def claim_key(claim: str) -> str:
    """Stable, text-only identity anchor.

    Deliberately byte-identical to ``source_evaluation.claim_act_key`` and the
    WFA ``wfa_claim_token`` provenance node: the prior act, the stored
    provenance and this ledger must resolve to the SAME identity.  No item id,
    answer, condition or credibility label enters this key.
    """
    return "wfa_" + hashlib.sha1((claim or "").encode("utf-8")).hexdigest()[:12]


def subject_key(subject: str) -> str:
    """Stable anchor for a subject identity, kept separate from the claim."""
    return "sub_" + hashlib.sha1((subject or "").encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ProvenanceRecord:
    """One identity-keyed, durable provenance/act record.

    ``reliability`` and ``familiarity`` are deliberately absent: they are
    item-strength signals and may never stand in for a source attribution.
    """

    claim_key: str
    claim: str = ""
    subject: str = ""
    subject_key: str = ""
    origin_id: str = ""
    source_type: str = ""
    edge_kind: str = ""
    trust_class: str = ""
    prior_decision: str = ""
    ever_endorsed: bool = False
    negated: bool = False
    version: int = _VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_key": self.claim_key,
            "claim": self.claim,
            "subject": self.subject,
            "subject_key": self.subject_key,
            "origin_id": self.origin_id,
            "source_type": self.source_type,
            "edge_kind": self.edge_kind,
            "trust_class": self.trust_class,
            "prior_decision": self.prior_decision,
            "ever_endorsed": bool(self.ever_endorsed),
            "negated": bool(self.negated),
            "version": int(self.version),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ProvenanceRecord":
        """Tolerant restore: unknown keys ignored, missing keys defaulted."""
        src = dict(d or {})
        key = str(src.get("claim_key") or "")
        if not key:
            key = claim_key(str(src.get("claim") or ""))
        known = {f: src.get(f) for f in (
            "claim", "subject", "subject_key", "origin_id", "source_type",
            "edge_kind", "trust_class", "prior_decision")}
        out: Dict[str, Any] = {k: str(v or "") for k, v in known.items()}
        out["claim_key"] = key
        out["ever_endorsed"] = bool(src.get("ever_endorsed", False))
        out["negated"] = bool(src.get("negated", False))
        try:
            out["version"] = int(src.get("version", _VERSION))
        except (TypeError, ValueError):
            out["version"] = _VERSION
        return cls(**out)

    def fingerprint(self) -> str:
        """Hash of the four bound aspects; invariant to item-side metadata."""
        payload = {
            "claim_key": self.claim_key,
            "subject_key": self.subject_key,
            "origin_id": self.origin_id,
            "source_type": self.source_type,
            "edge_kind": self.edge_kind,
            "trust_class": self.trust_class,
            "prior_decision": self.prior_decision,
            "ever_endorsed": bool(self.ever_endorsed),
            "negated": bool(self.negated),
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()


class DurableProvenanceLedger:
    """In-memory identity -> record map with deterministic serialisation."""

    def __init__(self, records: Optional[Mapping[str, ProvenanceRecord]] = None):
        self._records: Dict[str, ProvenanceRecord] = dict(records or {})

    def get(self, key: str) -> Optional[ProvenanceRecord]:
        return self._records.get(str(key))

    def upsert(self, record: ProvenanceRecord) -> None:
        self._records[str(record.claim_key)] = record

    def keys(self):
        return sorted(self._records)

    def to_dict(self) -> Dict[str, Any]:
        # Sorted keys -> byte-stable sidecar, independent of insertion order.
        return {
            "version": _VERSION,
            "records": {k: self._records[k].to_dict() for k in sorted(self._records)},
        }

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "DurableProvenanceLedger":
        if not isinstance(d, Mapping):
            return cls()
        raw = d.get("records", d) if "records" in d else d
        if not isinstance(raw, Mapping):
            return cls()
        recs = {}
        for k, v in raw.items():
            try:
                rec = v if isinstance(v, ProvenanceRecord) else ProvenanceRecord.from_dict(v)
            except Exception:
                continue
            recs[str(rec.claim_key or k)] = rec
        return cls(recs)

    def apply_to_context(self, key: str, claim: str, staged: Any, ctx: Any) -> bool:
        """Re-stage a durable prior act + negation BEFORE the existing gate.

        Returns True iff ``ctx.source_context`` was replaced.  Fail-closed for
        every malformed input: a non-dataclass staged context, a missing record
        or a non-dataclass ``replace`` leaves the context completely untouched.
        """
        rec = self.get(key)
        if rec is None:
            return False
        try:
            from .source_evaluation import Decision
            new = staged
            if rec.negated and not getattr(new, "invalidated", False):
                new = replace(new, invalidated=True)
            if (getattr(new, "invalidated", False)
                    and rec.ever_endorsed
                    and getattr(new, "prior_decision", None) is not Decision.ENDORSE):
                new = replace(new, prior_decision=Decision.ENDORSE)
            if new is staged:
                return False
            ctx.source_context = (claim, new)
            return True
        except Exception:
            return False


class DurableProvenanceStore:
    """A ledger plus an atomic JSON sidecar in the engine's data directory.

    The sidecar is the slow/durable mirror.  It is created only while the flag
    is on; with the flag off no file is ever written or read.
    """

    def __init__(self, directory: Optional[str] = None,
                 filename: str = FILE_NAME):
        self.directory = os.path.abspath(directory) if directory else None
        self.path = os.path.join(self.directory, filename) if self.directory else None
        self.ledger = DurableProvenanceLedger()
        self.loaded = False

    def load(self) -> DurableProvenanceLedger:
        """Load once from the sidecar; a missing/corrupt file fails open empty."""
        if self.loaded:
            return self.ledger
        self.loaded = True
        if not self.path or not os.path.exists(self.path):
            return self.ledger
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                self.ledger = DurableProvenanceLedger.from_dict(json.load(fh))
        except Exception:
            self.ledger = DurableProvenanceLedger()
        return self.ledger

    def commit(self) -> bool:
        """Atomic, fsynced, newline-normalised write of the whole ledger."""
        if not self.path or not self.directory:
            return False
        try:
            os.makedirs(self.directory, exist_ok=True)
            payload = json.dumps(
                self.ledger.to_dict(), sort_keys=True,
                separators=(",", ":"), ensure_ascii=True)
            fd, tmp = tempfile.mkstemp(
                prefix=".durable_provenance.", suffix=".tmp", dir=self.directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self.path)
                return True
            except Exception:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
                return False
        except Exception:
            return False


def _engine_data_dir(engine: Any) -> Optional[str]:
    save_path = getattr(engine, "_save_path", None)
    if save_path:
        try:
            return os.path.dirname(os.path.abspath(save_path))
        except Exception:
            pass
    data_dir = getattr(engine, "data_dir", None)
    if data_dir:
        try:
            return os.path.abspath(data_dir)
        except Exception:
            pass
    return None


def store_for_engine(engine: Any) -> DurableProvenanceStore:
    """Lazy, engine-cached store bound to the engine's data directory."""
    store = getattr(engine, "_durable_provenance_store", None)
    if isinstance(store, DurableProvenanceStore):
        return store
    store = DurableProvenanceStore(_engine_data_dir(engine))
    try:
        engine._durable_provenance_store = store
    except Exception:
        pass
    return store


def _enum_value(x: Any) -> str:
    """String value of a str-Enum (or a plain string / duck-typed source)."""
    return str(getattr(x, "value", x) or "")


def _source_fields(source: Any, edge_kind: Optional[str] = None) -> Dict[str, str]:
    """Extract provenance from the engine's staged source record only."""
    if source is None:
        return {}
    if isinstance(source, Mapping):
        origin = str(source.get("origin_id") or source.get("source")
                      or source.get("url") or "")
        stype = str(source.get("source_type") or "")
        kind = str(source.get("edge_kind") or edge_kind or stype or "")
        tc = source.get("trust_class", "")
    else:
        origin = str(getattr(source, "origin_id", "") or "")
        stype = str(getattr(source, "source_type", "") or "")
        kind = str(edge_kind or stype or "")
        tc = getattr(source, "trust_class", "")
    tcval = _enum_value(tc)
    return {
        "origin_id": origin,
        "source_type": stype,
        "edge_kind": kind,
        "trust_class": str(tcval or ""),
    }


def record_decision(store: DurableProvenanceStore, key: str, decision: Any,
                    *, claim: str = "", subject: str = "", source: Any = None,
                    negated: bool = False, edge_kind: Optional[str] = None) -> ProvenanceRecord:
    """Write this engine's emitted act into the SAME identity-keyed record.

    Provenance is first-write-wins (it must stay stable across later turns);
    ``ever_endorsed`` and ``negated`` are monotonic; ``prior_decision`` holds
    the latest emitted decision for audit.  Persists immediately.
    """
    store.load()
    key = str(key)
    rec = store.ledger.get(key) or ProvenanceRecord(claim_key=key)
    fields: Dict[str, Any] = {}
    if claim and not rec.claim:
        fields["claim"] = str(claim)
    if subject and not rec.subject:
        fields["subject"] = str(subject)
        fields["subject_key"] = subject_key(str(subject))
    for name, value in _source_fields(source, edge_kind).items():
        if value and not getattr(rec, name, ""):
            fields[name] = value
    dec_name = getattr(getattr(decision, "value", decision), "upper", lambda: "")()
    dec_name = str(dec_name or "")
    if dec_name:
        # A real emitted Decision enum; never an item/answer string.
        fields["prior_decision"] = dec_name
    if dec_name == "ENDORSE":
        fields["ever_endorsed"] = True
    if negated and not rec.negated:
        fields["negated"] = True
    if fields:
        rec = replace(rec, **fields)
        store.ledger.upsert(rec)
    store.commit()
    return rec
