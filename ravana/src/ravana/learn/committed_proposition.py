"""Committed-Proposition Gating (CPG) -- the brain's production constraint.

Implements the mechanism from ``docs/GROUNDING_MECHANISM.md``. The core claim,
from Nozari, Dell & Schwartz (2011) and the semantic-dementia literature
(Snowden et al. 1989; Hodges et al. 1992): the brain does **not** generate
fluent text and then check it. It **commits to a lexical-semantic plan before
form encoding**, and that commitment is gated by a production-side conflict
monitor whose criterion is modulated by prediction error.

The clinical evidence is the syndrome RAVANA's decoder exhibits: fluent,
grammatical, empty speech with preserved syntax -- semantic dementia. The
dissociation with Wernicke's aphasia (fluent nonsense *plus* no error
awareness) shows the constraint lives in the **semantic store**, not the
formulator. RAVANA's formulator is fine; its semantic commitment is missing.

## The five steps

1. **Commit** -- the graph walk yields a committed proposition ``(S, relation,
   T_i)`` with strength from existing edge/node fields.
2. **Set the ACC criterion** -- ``theta = theta_base + alpha*pe +
   beta*free_energy``. High prediction error **raises** the criterion: more
   conservative emission, not less.
3. **Gate at selection time** -- a token is admissible only if it realizes a
   committed target that clears ``theta``. On failure, **relax and retry**
   (conduit d'approche), up to N times. Retry, do not fall through.
4. **Verify** the emitted clause against the commitment (the existing
   perceptual loop, re-anchored).
5. **Update** the commitment -- reinforce on pass, raise free energy on fail.

## Scope of this module

Steps 1-3 and 5. Step 4 already exists as ``_sm_response_grounded``.

This module is a **pure decision function**: given a subject and the graph, it
produces a set of admissible token ids and a retry schedule. It does not touch
``NeuralDecoder`` -- the caller passes the result as ``token_boost``, which the
decoder already accepts (``neural_decoder.py:458``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class CPGConfig:
    """Threshold gains and retry policy.

    Defaults are the starting values proposed in ``docs/GROUNDING_MECHANISM.md``
    section 3. They are judgment, not fitted.
    """

    theta_base: float = 0.30
    alpha_pe: float = 0.5        # prediction error RAISES the criterion
    beta_fe: float = 0.3         # graph free energy RAISES the criterion

    # Commitment strength is c_i = w_edge*weight * w_conf*confidence
    #                        * w_node*node_conf * w_stab*stability
    w_edge: float = 0.35
    w_conf: float = 0.30
    w_node: float = 0.20
    w_stab: float = 0.15

    # Conduit d'approche: relax and retry rather than falling through.
    max_retries: int = 3
    relax_step: float = 0.08

    # Token boost applied to admissible tokens.
    token_boost: float = 2.5

    # NoGo: penalty applied to content words that are NOT admissible. 0
    # disables suppression (boost-only, the pre-NoGo behaviour).
    suppress: float = 0.0
    # A content word this close (in commitment strength) to the weakest
    # admissible proposition is treated as admissible too, so the NoGo
    # pathway does not suppress near-misses.
    suppress_margin: float = 0.05

    # ── Relation-level gating ───────────────────────────────────────────
    # A committed proposition is (subject, relation, target). Boosting the
    # target alone constrains the ARGUMENT but leaves the relation free, and an
    # unsupported assertion is precisely a relation the graph does not support.
    #
    # `relation_words` maps a graph relation_type to the surface words that can
    # realize it. When non-empty, only those words are boosted -- so the
    # decoder is pushed toward the committed RELATION, not just its target.
    # Empty (the default) keeps target-only behaviour.
    relation_words: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def strength_weights_sum(self) -> float:
        return self.w_edge + self.w_conf + self.w_node + self.w_stab


@dataclass
class CommittedProposition:
    """A typed claim the graph supports: (subject, relation, target)."""

    subject: str
    relation: str
    target: str
    strength: float
    prediction_error: float
    edge_weight: float = 0.0
    edge_confidence: float = 0.0
    node_confidence: float = 0.0
    node_stability: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "target": self.target,
            "strength": round(self.strength, 4),
            "prediction_error": round(self.prediction_error, 4),
        }


@dataclass
class CPGDecision:
    subject: str
    admissible_targets: List[CommittedProposition]
    theta: float
    retries_used: int
    final_theta: float
    reason: str
    token_boost: Dict[int, float] = field(default_factory=dict)
    token_suppress: Dict[int, float] = field(default_factory=dict)

    @property
    def committed(self) -> bool:
        return bool(self.admissible_targets)

    def as_dict(self) -> Dict[str, object]:
        return {
            "subject": self.subject,
            "committed": self.committed,
            "theta": round(self.theta, 4),
            "retries_used": self.retries_used,
            "final_theta": round(self.final_theta, 4),
            "n_admissible": len(self.admissible_targets),
            "reason": self.reason,
            "targets": [t.as_dict() for t in self.admissible_targets[:5]],
        }


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


def commitment_strength(
    *,
    edge_weight: float = 0.0,
    edge_confidence: float = 0.0,
    node_confidence: float = 0.0,
    node_stability: float = 0.0,
    cfg: Optional[CPGConfig] = None,
) -> float:
    """c_i -- how strongly the graph commits to one proposition.

    A product of existing fields, so a proposition supported by a weak edge, a
    low-confidence edge, or an unstable node scores low on its own. Using a
    weighted product rather than a sum means no single strong field can carry a
    proposition the rest of the graph does not support.
    """
    cfg = cfg or CPGConfig()
    return _clamp01(
        cfg.w_edge * _clamp01(edge_weight)
        + cfg.w_conf * _clamp01(edge_confidence)
        + cfg.w_node * _clamp01(node_confidence)
        + cfg.w_stab * _clamp01(node_stability)
    )


def acc_criterion(
    prediction_error: float,
    free_energy: float,
    cfg: Optional[CPGConfig] = None,
) -> float:
    """theta -- the ACC conflict criterion.

    **High prediction error raises the bar.** This is the Friston/Clark
    modulation and it is the opposite of RAVANA's current behaviour, which
    falls through to templates when uncertain. The brain's conservative
    response to uncertainty is to demand *more* evidence before speaking, not
    less.
    """
    cfg = cfg or CPGConfig()
    return max(0.0, cfg.theta_base
               + cfg.alpha_pe * _clamp01(prediction_error)
               + cfg.beta_fe * _clamp01(free_energy))


def commit(
    subject: str,
    candidates: Iterable[Tuple[str, str, float, float]],
    *,
    node_confidence: float = 0.0,
    node_stability: float = 0.0,
    prediction_error: float = 0.0,
    cfg: Optional[CPGConfig] = None,
) -> List[CommittedProposition]:
    """Step 1 -- build the committed propositions for a subject.

    ``candidates`` yields ``(relation, target, edge_weight, edge_confidence)``
    for each typed edge leaving the subject. Each becomes a proposition with a
    strength from existing fields.
    """
    cfg = cfg or CPGConfig()
    out: List[CommittedProposition] = []
    for rel, tgt, w, c in candidates:
        if not tgt:
            continue
        s = commitment_strength(
            edge_weight=w, edge_confidence=c,
            node_confidence=node_confidence, node_stability=node_stability,
            cfg=cfg)
        out.append(CommittedProposition(
            subject=subject, relation=rel, target=tgt, strength=s,
            prediction_error=_clamp01(prediction_error),
            edge_weight=_clamp01(w), edge_confidence=_clamp01(c),
            node_confidence=_clamp01(node_confidence),
            node_stability=_clamp01(node_stability)))
    out.sort(key=lambda p: p.strength, reverse=True)
    return out


def decide(
    subject: str,
    propositions: Sequence[CommittedProposition],
    *,
    free_energy: float = 0.0,
    token_for_target: Optional[Dict[str, int]] = None,
    cfg: Optional[CPGConfig] = None,
) -> CPGDecision:
    """Steps 2 and 3 -- set the criterion, then gate.

    A proposition is admissible when its strength clears ``theta``. If none do,
    relax the criterion and retry (conduit d'approche) up to ``max_retries``
    times. Only then does the decision report no commitment.

    ``token_for_target`` maps a target label to a decoder token id; the result
    carries a ``token_boost`` the caller can pass straight to
    ``NeuralDecoder.generate``.
    """
    cfg = cfg or CPGConfig()
    if not propositions:
        return CPGDecision(
            subject=subject, admissible_targets=[], theta=cfg.theta_base,
            retries_used=0, final_theta=cfg.theta_base,
            reason="no candidate propositions from the graph")

    pe = max((p.prediction_error for p in propositions), default=0.0)
    theta = acc_criterion(pe, free_energy, cfg)
    initial_theta = theta

    admissible: List[CommittedProposition] = []
    retries = 0
    # Attempt 0 uses the un-relaxed criterion; each subsequent attempt relaxes
    # once. So max_retries is the number of RELAXATIONS, and the loop runs at
    # most max_retries + 1 times.
    gave_up = True
    for attempt in range(cfg.max_retries + 1):
        admissible = [p for p in propositions if p.strength >= theta]
        if admissible:
            retries = attempt
            gave_up = False
            break
        if attempt < cfg.max_retries:
            theta = max(0.0, theta - cfg.relax_step)
            retries = attempt + 1
    if gave_up:
        admissible = []

    boost: Dict[int, float] = {}
    suppress: Dict[int, float] = {}
    if admissible and token_for_target:
        # Every target the graph proposes, admissible or not.
        for p in propositions:
            tid = token_for_target.get(p.target.lower())
            if tid is None:
                continue
            if p in admissible:
                # Stronger commitment -> stronger boost, so the gate prefers
                # the proposition the graph actually supports.
                boost[tid] = cfg.token_boost * max(0.25, p.strength)
            elif cfg.suppress > 0.0:
                # NoGo. A near-miss (within suppress_margin of the weakest
                # admissible proposition) is left alone -- suppressing it
                # would trade real signal for a marginal gain in purity.
                weakest = min(admissible, key=lambda q: q.strength).strength
                if p.strength < weakest - cfg.suppress_margin:
                    suppress[tid] = cfg.suppress * max(0.25, 1.0 - p.strength)

        # Relation-level gating. The committed proposition is
        # (subject, relation, target); boosting only the target constrains the
        # argument while leaving the relation free, and an unsupported
        # assertion is exactly a relation the graph does not support.
        if cfg.relation_words:
            for p in admissible:
                for word in cfg.relation_words.get(p.relation, ()):
                    rid = token_for_target.get(word)
                    if rid is not None:
                        boost[rid] = max(
                            boost.get(rid, 0.0),
                            cfg.token_boost * max(0.25, p.strength))

    if admissible:
        reason = ("%d proposition(s) clear theta=%.3f (relaxed %d time(s) from "
                  "%.3f)" % (len(admissible), theta, retries, initial_theta))
    else:
        reason = ("no proposition cleared theta even after %d relaxation(s); "
                  "escalate to honest uncertainty" % cfg.max_retries)

    return CPGDecision(
        subject=subject, admissible_targets=admissible,
        theta=initial_theta, retries_used=retries, final_theta=theta,
        reason=reason, token_boost=boost, token_suppress=suppress)


def update_commitment(
    proposition: CommittedProposition,
    *,
    accepted: bool,
    cfg: Optional[CPGConfig] = None,
) -> Tuple[float, float]:
    """Step 5 -- reinforce on acceptance, raise free energy on rejection.

    Returns ``(new_edge_confidence, new_prediction_error)``. Both are bounded
    and both use existing fields, so the caller can write them straight back
    with the graph's own primitives.

    Rejection raises prediction error, which raises ``theta`` on the next turn
    (``acc_criterion``) -- the graph learns to be more conservative about a
    proposition that produced ungrounded speech.
    """
    cfg = cfg or CPGConfig()
    conf = _clamp01(proposition.edge_confidence)
    pe = _clamp01(proposition.prediction_error)
    if accepted:
        return min(1.0, conf + 0.05), max(0.0, pe - 0.05)
    return max(0.0, conf - 0.05), min(1.0, pe + 0.10)
