"""Edge-provenance admission policy (research item E).

The bare-edge surfacer in interface.py previously stated ANY single graph edge
as a confident fact once ``weight * confidence >= 0.4`` — with NO provenance
check. That is the source of the hollow answers from the battery
("Sleep leads to depression", "Inflation leads to increase"): a single noisy
``auto_expand`` / boot co-occurrence edge gets asserted as fact.

Research basis (item E in the plan):
  - TGComplete (2026): among gold-correct completed edges, 76-96% have NO textual
    support -> verifiability != correctness. Admission policy = admit/abstain
    under provenance uncertainty; trades recall for auditability. Precision
    rises because verified edges are more likely correct.
  - ClaimVer (EMNLP 2024) / ProVe / TrustGraph (W3C PROV-O): per-claim
    attribution; extraction-time provenance DAG for traceability.

Design (no hard-coded thresholds where a distributional fit belongs):
  1. ``provenance_class(edge)`` classifies the edge by its populated
     ``source_metadata`` (verified_web / co_occurrence / auto_expand / boot /
     inferred / curated). This is the provenance discriminator the pruning code
     already relies on (graph/engine.py:1019) -- we generalize it.
  2. ``admit_as_fact(edge, converging=0)`` -- a SINGLE bare edge may be surfaced
     as a confident fact ONLY if it has verifiable provenance
     (edge_kind == "web_fact" with a source) OR >= 2 converging independent
     sources. Otherwise the statement is HEDGED (stated as a tentative
     suggestion), never as established fact.
  3. ``provenance_adjusted_confidence(edge, converging)`` -- caps auto_expand /
     boot / co_occurrence confidence low unless corroborated by convergence.
  4. ``surface_cue(edge, converging)`` -- returns the realization modifier:
     "(per source)" for verified web, a hedge prefix for unprovenanced.

The decision boundaries here (convergence requirement = 2; the per-class
confidence caps) are the CURRENT calibration anchors. They are FIT, not
frozen: measure_provenance_admission.py measures provenance coverage and the
policy's precision on labeled (edge, gold) pairs, and re-calibrates the
convergence threshold + caps from held-out data. This is the cross-cutting
measurement substrate for item E, exactly as measure_salad_classifier.py is for
item B.
"""
from __future__ import annotations

import os
import time
import json
from typing import Any, Dict, Optional

# Provenance classes, most-trusted first.
VERIFIED_WEB = "verified_web"
CURATED = "curated"
CO_OCCURRENCE = "co_occurrence"
AUTO_EXPAND = "auto_expand"
BOOT = "boot"
INFERRED = "inferred"
UNKNOWN = "unknown"

# Calibration anchors (re-calibrated by measure_provenance_admission.py).
# A single bare edge is admitted as FACT only when it has verifiable provenance
# OR this many converging independent sources agree.
CONVERGENCE_FACT_THRESHOLD = 2
# Confidence caps per provenance class (unless corroborated by convergence).
# auto_expand / boot edges are GloVe-wired noise (confidence 0.5 / 0.02) and must
# NEVER read as established fact on their own.
_PROV_CONF_CAP = {
    VERIFIED_WEB: 1.0,
    CURATED: 1.0,
    CO_OCCURRENCE: 0.45,
    AUTO_EXPAND: 0.35,
    BOOT: 0.25,
    INFERRED: 0.5,
    UNKNOWN: 0.3,
}
# Edge kinds that carry verifiable textual provenance (a real source we can cite).
_VERIFIABLE_KINDS = {"web_fact"}


def _meta(edge: Any) -> Dict[str, Any]:
    if edge is None:
        return {}
    sm = getattr(edge, "source_metadata", None)
    if not isinstance(sm, dict):
        return {}
    return sm


def provenance_class(edge: Any) -> str:
    """Classify an edge by its populated source_metadata."""
    m = _meta(edge)
    kind = m.get("edge_kind")
    if kind == "web_fact":
        # verified only if it carries a citable source or retrieval confidence
        if m.get("source_url") or m.get("source") or m.get("retrieval_conf") is not None:
            return VERIFIED_WEB
        return CO_OCCURRENCE  # tagged web_fact but no traceable source
    if kind == "co_occurrence":
        return CO_OCCURRENCE
    if kind == "auto_expand":
        return AUTO_EXPAND
    if kind == "boot_cooccurrence":
        return BOOT
    if kind in ("inferred", "semantic_inferred"):
        return INFERRED
    if kind == "curated":
        return CURATED
    # No edge_kind tag: distinguish boot-seeded GloVe edges (very low confidence)
    # from anything else by confidence magnitude.
    conf = getattr(edge, "confidence", None)
    if isinstance(conf, (int, float)) and conf is not None and conf < 0.1:
        return BOOT
    return UNKNOWN


def is_verifiable(edge: Any) -> bool:
    """True iff the edge carries traceable textual provenance we can cite."""
    return provenance_class(edge) in (VERIFIED_WEB, CURATED)


def admit_as_fact(edge: Any, converging: int = 0) -> bool:
    """TGComplete-style admission: a single bare edge becomes a confident fact
    ONLY with verifiable provenance OR >= CONVERGENCE_FACT_THRESHOLD independent
    sources. Otherwise it is HEDGED."""
    if is_verifiable(edge):
        return True
    return converging >= CONVERGENCE_FACT_THRESHOLD


def provenance_adjusted_confidence(edge: Any, converging: int = 0,
                                   base_conf: Optional[float] = None) -> float:
    """Condition the surfaced confidence on provenance class (not one scalar).

    Unprovenanced edges are capped low unless corroborated by convergence; each
    corroborating independent source lifts the cap toward the verified level.
    """
    cls = provenance_class(edge)
    if base_conf is None:
        base_conf = float(getattr(edge, "confidence", 0.5) or 0.5)
    cap = _PROV_CONF_CAP.get(cls, 0.3)
    # Convergence earns back confidence: +0.2 per corroborating source, capped.
    if converging > 0:
        cap = min(1.0, cap + 0.2 * converging)
    return float(min(base_conf, cap))


def surface_cue(edge: Any, converging: int = 0) -> str:
    """Realization modifier for the bare-edge surface form.

    - verified web with a source -> "(per source)"
    - curated -> "" (treated as established)
    - everything else (unprovenanced single edge) -> a hedge prefix so the
      statement reads as tentative, never as established fact.
    """
    cls = provenance_class(edge)
    if cls == VERIFIED_WEB:
        return "(per source)"
    if cls == CURATED:
        return ""
    if cls == CO_OCCURRENCE:
        return "I've seen patterns linking"
    if cls == AUTO_EXPAND:
        return "my associations loosely connect"
    if cls == BOOT:
        return "I vaguely link"
    if cls == INFERRED:
        return "it may be that"
    # unprovenanced single edge, no convergence
    return "some connections suggest"


def populate_provenance(edge: Any, *, edge_kind: str,
                        source: Optional[str] = None,
                        source_url: Optional[str] = None,
                        method: Optional[str] = None,
                        retrieval_conf: Optional[float] = None,
                        timestamp: Optional[float] = None,
                        converging_sources: int = 0) -> None:
    """Backfill a fully-populated provenance record onto an edge.

    Centralizes the fields so every edge carries traceable provenance
    (TrustGraph / W3C PROV-O style: wasDerivedFrom -> source). Safe to call on
    edges that lack source_metadata (no-op).
    """
    m = _meta(edge)
    if not m:
        return
    m["edge_kind"] = edge_kind
    if source is not None:
        m["source"] = source
    if source_url is not None:
        m["source_url"] = source_url
    if method is not None:
        m["method"] = method
    if retrieval_conf is not None:
        m["retrieval_conf"] = float(retrieval_conf)
    m["timestamp"] = float(timestamp if timestamp is not None else time.time())
    if converging_sources:
        m["converging_sources"] = int(converging_sources)
    m["epistemic_status"] = "fact" if edge_kind == "web_fact" else "association"


# ── Calibration substrate (parallel to measure_salad_classifier.py) ──────────
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_FIT_PATH = os.path.join(_PROJ, "data", "provenance_policy.json")


def save_policy(anchors: Dict[str, Any]) -> None:
    """Persist re-calibrated anchors (convergence threshold, per-class caps)."""
    os.makedirs(os.path.dirname(_FIT_PATH), exist_ok=True)
    with open(_FIT_PATH, "w", encoding="utf-8") as f:
        json.dump(anchors, f, indent=2)


def load_policy() -> Optional[Dict[str, Any]]:
    if os.path.exists(_FIT_PATH):
        try:
            with open(_FIT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None
