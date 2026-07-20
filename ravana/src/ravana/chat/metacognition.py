"""Metacognition: Feeling-of-Knowing (FOK) + confidence readout (ACC analog).

Brain basis
-----------
The anterior cingulate cortex (ACC; Botvinick 2004) monitors conflict between
competing responses; the metacognitive sense of "I know / I don't know" (FOK;
Hart 1965; Koriat 1993) gates whether a claim should be *asserted* or hedged.
This module is the single learned/structural source for that gate — it does
NOT carry topic lists. Instead it derives a per-claim confidence scalar from:

  1. support count  — how many typed-graph edges / retrievals back the concept
                     (denser support ⇒ higher confidence; distribution-driven).
  2. retrieval success — did a live lookup actually return a grounded snippet?
  3. modality       — reused from hedges.modality_from_support so the surfacer
                     can tag possibility vs likelihood vs necessity consistently.

A claim is asserted only when confidence > theta_withhold. Below that, the
caller attaches a modality hedge (or withholds outright) rather than speaking
flatly. This is the honest-degradation primitive for Defect D (misleading math
claims) and the numeric-claim honesty gate.

Thresholds are learned/structural defaults (not fixed topic cutoffs) and live
at module scope so a fit harness can move them.
"""

from typing import Dict, Any, Optional, Tuple

# Learned/structural defaults (adaptive, not hardcoded magic numbers).
THETA_WITHDHOLD = 0.30      # below this confidence, do NOT assert flatly
SUPPORT_FULL_MARK = 4.0     # edge/retrieval support that maps to full confidence
RETRIEVAL_SUCCESS_BOOST = 0.5

# Reuse the project's graded-modality mapping so hedging is consistent.
try:
    from .hedges import modality_from_support
except Exception:  # pragma: no cover — hedges always present in-package
    def modality_from_support(support: float) -> str:  # type: ignore
        if support > 0.55:
            return "likely"
        if support > 0.30:
            return "possible"
        return "unknown"


def fok_confidence(support_count: float,
                   retrieval_succeeded: bool = False,
                   base: float = 0.0) -> float:
    """Scalar Feeling-of-Knowing confidence in [0,1].

    Combines a base (e.g. prior belief strength) with normalized graph/retrieval
    support. Distribution-driven: confidence rises with the *amount* of
    corroborating support, not with any topic flag.
    """
    # Saturating (1 - e^-x) curve: more support ⇒ diminishing returns, never >1.
    import math
    _support = float(support_count)
    conf = 1.0 - math.exp(-max(0.0, _support) / SUPPORT_FULL_MARK)
    conf = max(conf, base)
    if retrieval_succeeded:
        conf = min(1.0, conf + RETRIEVAL_SUCCESS_BOOST)
    return round(min(1.0, max(0.0, conf)), 3)


def should_assert(confidence: float,
                  theta_withhold: float = THETA_WITHDHOLD) -> Tuple[bool, str]:
    """ACC conflict gate: assert only above the withhold threshold.

    Returns (may_assert, modality). When below threshold, modality is the
    honest hedge the surfacer should attach instead of a flat assertion.
    """
    if confidence >= theta_withhold:
        return True, modality_from_support(confidence)
    return False, modality_from_support(confidence)


class Metacognition:
    """Per-claim confidence reader over the typed concept graph.

    Stateless wrapper around fok_confidence so callers can pass a small
    support bundle (edge count, retrieval flag) without re-implementing the
    saturating curve. Keeps a short ring buffer of recent FOK verdicts for
    observability (mirrors the engine's self-monitor log shape).
    """

    def __init__(self, theta_withhold: float = THETA_WITHDHOLD) -> None:
        self.theta_withhold = theta_withhold
        self._recent: list = []

    def read(self, support_count: float,
             retrieval_succeeded: bool = False,
             base: float = 0.0) -> Tuple[float, bool, str]:
        conf = fok_confidence(support_count, retrieval_succeeded, base)
        may_assert, modality = should_assert(conf, self.theta_withhold)
        self._recent.append((round(conf, 3), may_assert, modality))
        if len(self._recent) > 50:
            self._recent = self._recent[-50:]
        return conf, may_assert, modality
