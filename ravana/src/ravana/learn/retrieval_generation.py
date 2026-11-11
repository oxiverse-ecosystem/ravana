"""Retrieval-vs-generation switch (the brain's race, translated).

Implements the rule from ``docs/RETRIEVAL_VS_GENERATION.md`` (§10). The brain
has no binary retrieve-vs-generate switch: it runs a *thresholded race* between
a fast stored-representation pathway and a slower compositional one, and the
winner is decided by three quantities:

    R(s)  retrieval strength   -- how strong the best stored candidate is
    PE(s) prediction error     -- how well the stored model accounts for the ask
    D(s)  competitive density  -- how many candidates compete for the subject

    retrieve  iff  R(s) >= T(s)
    T(s) = T_base * (1 + k_pe*PE) * (1 + k_d*D) * (1 - k_f*familiarity)
    T(s) <- T(s) * (1 - k_p*tanh(N_compose(s)/N_practice))     # practice term

Two findings drive the shape, and both invert the intuitive design:

* **High prediction error RAISES the bar** rather than triggering generation.
  The brain's conservative response to uncertainty is to fall back on the most
  practiced stored form, not to compose something novel. So ``k_pe``
  multiplies positively.
* **A practice term lets the decoder EARN reachability** (Logan's instance
  theory of automatization). As the decoder answers more often for a subject,
  the threshold drops and it is reached more easily. This is what a hand-set
  ``use_linggen`` flag cannot express.

Citations are in the doc; the load-bearing ones are Levelt (1989, 1999),
Logan (1988), Anderson's ACT-R, Reder (1987) on cue-familiarity, Koriat (1993)
on the accessibility model of FOK, and Friston/Clark on predictive processing.

This module is a pure decision function plus its config. It does not touch the
dispatch chain -- wiring it in is a separate step, and the rule is a *gate*
(§10.8), not a reordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class RetrievalGenerationConfig:
    """Weights and gains for the switch.

    Defaults are the starting values proposed in the research doc (§10). They
    are judgment, not fitted -- see ``docs/RETRIEVAL_VS_GENERATION.md`` §12 for
    the honest flags. Every one is a knob, and the rule is falsifiable (§11).
    """

    # Retrieval-strength weights. w1 > w2 > w3 ~ w4 ~ w5 ~ w6, sum to 1.
    w_definition: float = 0.30
    w_web: float = 0.20
    w_confidence: float = 0.15
    w_stability: float = 0.10
    w_association: float = 0.15
    w_familiarity: float = 0.10

    # Threshold gains.
    t_base: float = 0.35
    k_pe: float = 0.5       # prediction error RAISES the bar
    k_d: float = 0.3        # competitive density RAISES the bar
    k_f: float = 0.2        # user familiarity LOWERS the bar

    # Practice term (Logan). 0 disables it.
    k_p: float = 0.15
    n_practice: float = 20.0

    def retrieval_weights_sum(self) -> float:
        return (self.w_definition + self.w_web + self.w_confidence
                + self.w_stability + self.w_association + self.w_familiarity)


@dataclass
class RetrievalStrength:
    """The decomposed R(s) terms, kept so the decision can be explained."""

    definition: float = 0.0
    web: float = 0.0
    confidence: float = 0.0
    stability: float = 0.0
    association: float = 0.0
    familiarity: float = 0.0

    def total(self, cfg: RetrievalGenerationConfig) -> float:
        return (
            cfg.w_definition * self.definition
            + cfg.w_web * self.web
            + cfg.w_confidence * self.confidence
            + cfg.w_stability * self.stability
            + cfg.w_association * self.association
            + cfg.w_familiarity * self.familiarity
        )


@dataclass
class SwitchDecision:
    subject: str
    retrieve: bool
    strength: float
    threshold: float
    pe: float
    density: float
    familiarity: float
    practice_discount: float
    reason: str
    detail: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "subject": self.subject,
            "retrieve": bool(self.retrieve),
            "strength": round(self.strength, 4),
            "threshold": round(self.threshold, 4),
            "prediction_error": round(self.pe, 4),
            "density": round(self.density, 4),
            "familiarity": round(self.familiarity, 4),
            "practice_discount": round(self.practice_discount, 4),
            "reason": self.reason,
        }


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


def compute_retrieval_strength(
    subject: str,
    *,
    has_definition: bool = False,
    has_web: bool = False,
    confidence: float = 0.0,
    stability: float = 0.0,
    association: float = 0.0,
    familiarity: float = 0.0,
) -> RetrievalStrength:
    """R(s) from signals RAVANA already computes.

    Each term is clamped to [0, 1] so a malformed upstream value cannot push
    the score out of range.
    """
    return RetrievalStrength(
        definition=1.0 if has_definition else 0.0,
        web=1.0 if has_web else 0.0,
        confidence=_clamp01(confidence),
        stability=_clamp01(stability),
        association=_clamp01(association),
        familiarity=_clamp01(familiarity),
    )


def compute_prediction_error(
    node_pe: Optional[float],
    edge_pes,
    mean_pe: float,
) -> float:
    """PE(s) -- max of the node's own error, its edges' mean, and the global mean.

    All three are already computed and persisted (``engine.py``,
    ``chain_walker.py``, ``graph/engine.py``). Taking the max is deliberate:
    prediction error is a *worst-case* signal, and a single surprising edge
    should raise the bar even when the node itself looks settled.
    """
    candidates = [ _clamp01(mean_pe) ]
    if node_pe is not None:
        candidates.append(_clamp01(node_pe))
    if edge_pes:
        vals = [ _clamp01(v) for v in edge_pes ]
        if vals:
            candidates.append(float(np.mean(vals)))
    return max(candidates) if candidates else 0.0


def compute_density(out_degree: int, in_degree: int, max_degree: int) -> float:
    """D(s) -- normalised degree. High density = more competition.

    Normalising by the graph's max degree keeps the term comparable as the
    graph grows; without it a mature graph would raise every threshold.
    """
    if max_degree <= 0:
        return 0.0
    return _clamp01((float(out_degree) + float(in_degree)) / float(max_degree))


def compute_threshold(
    pe: float,
    density: float,
    familiarity: float,
    *,
    n_compose: int = 0,
    cfg: Optional[RetrievalGenerationConfig] = None,
) -> Tuple[float, float]:
    """T(s), and the practice discount applied to it.

    Returns ``(threshold, practice_discount)`` so the caller can report both.
    The practice term is a *discount*: it multiplies the threshold by
    ``(1 - k_p*tanh(N/N_practice))``, so more decoder practice lowers the bar.
    """
    cfg = cfg or RetrievalGenerationConfig()
    t = (cfg.t_base
         * (1.0 + cfg.k_pe * _clamp01(pe))
         * (1.0 + cfg.k_d * _clamp01(density))
         * (1.0 - cfg.k_f * _clamp01(familiarity)))
    if cfg.k_p > 0 and cfg.n_practice > 0 and n_compose > 0:
        discount = 1.0 - cfg.k_p * float(np.tanh(n_compose / cfg.n_practice))
        discount = max(0.0, min(1.0, discount))
    else:
        discount = 1.0
    return max(0.0, t * discount), discount


def decide(
    subject: str,
    *,
    strength: RetrievalStrength,
    pe: float,
    density: float,
    familiarity: float,
    n_compose: int = 0,
    cfg: Optional[RetrievalGenerationConfig] = None,
) -> SwitchDecision:
    """The switch. Retrieve iff R(s) >= T(s)."""
    cfg = cfg or RetrievalGenerationConfig()
    r = _clamp01(strength.total(cfg))
    t, discount = compute_threshold(pe, density, familiarity,
                                    n_compose=n_compose, cfg=cfg)
    retrieve = r >= t
    if retrieve:
        reason = ("stored candidate clears the bar (R=%.3f >= T=%.3f)"
                  % (r, t))
    else:
        reason = ("no stored candidate clears the bar (R=%.3f < T=%.3f) "
                  "-- compose" % (r, t))
    return SwitchDecision(
        subject=subject,
        retrieve=retrieve,
        strength=r,
        threshold=t,
        pe=_clamp01(pe),
        density=_clamp01(density),
        familiarity=_clamp01(familiarity),
        practice_discount=discount,
        reason=reason,
        detail={
            "r_definition": strength.definition,
            "r_web": strength.web,
            "r_confidence": strength.confidence,
            "r_stability": strength.stability,
            "r_association": strength.association,
            "r_familiarity": strength.familiarity,
        },
    )
