"""
E — active-inference control spine (capstone of the unified layer).
=================================================================
Curiosity-driven web-reading is epistemic action that minimizes Expected Free
Energy (Friston 2015; Schmidhuber 2010; Oudeyer & Kaplan 2007). This module is
the principled controller that was SKETCHED at C-time (the KnowledgeGap EFE
interface) and is now buildable because:
  - C-lite produces knowledge gaps (graph-neighbourhood sparsity -> EFE proxy)
  - the existing graph nodes carry prediction_free_energy (a real uncertainty
    signal already used by WebLearner._auto_select_curiosity_topics)
  - N3 (DualCodeSpace) gives structure to reason/analogize over

Design: ADDITIVE. It does NOT replace WebLearner's existing curiosity engine.
It computes a unified EFE score per candidate topic (gap + node PFE +
contradiction), selects the max-EFE target, and drives the WebLearner's read.
After reading, EFE must drop (the loop closes). This is the C<->E bridge.

Brain-faithful: the controller is a precision-weighted selection over the
agent's own uncertainty estimates, not a hardcoded topic list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class EFEState:
    """Expected free energy of a candidate topic + its components."""
    topic: str
    gap_efe: float = 0.0       # C-lite graph-neighbourhood sparsity
    node_pfe: float = 0.0      # existing prediction_free_energy on the node
    contradiction: float = 0.0  # unresolved contradiction mismatch
    total: float = 0.0

    def is_curiosity_target(self, threshold: float = 1.0) -> bool:
        return self.total >= threshold


class ActiveInferenceController:
    """Active-inference control spine over uncertainty signals.

    Wires the thin C-time KnowledgeGap interface + existing node PFE into one
    selection loop. Consumes a `gap_fn(topic)->KnowledgeGap` and a
    `pfe_fn(topic)->float`, so it stays decoupled from any specific graph impl.
    """

    def __init__(self, gap_fn, pfe_fn, contradiction_fn=None,
                 w_gap: float = 1.0, w_pfe: float = 1.0, w_contradiction: float = 1.0):
        self.gap_fn = gap_fn
        self.pfe_fn = pfe_fn
        self.contradiction_fn = contradiction_fn
        self.w_gap = w_gap
        self.w_pfe = w_pfe
        self.w_contradiction = w_contradiction
        self._history: List[Tuple[str, float, float]] = []  # (topic, efe_before, efe_after)

    def score(self, topic: str) -> EFEState:
        gap = self.gap_fn(topic)
        gap_efe = gap.efe if gap is not None else 0.0
        node_pfe = self.pfe_fn(topic) or 0.0
        contradiction = self.contradiction_fn(topic) if self.contradiction_fn else 0.0
        total = self.w_gap * gap_efe + self.w_pfe * node_pfe + self.w_contradiction * contradiction
        return EFEState(topic=topic, gap_efe=gap_efe, node_pfe=node_pfe,
                        contradiction=contradiction, total=total)

    def select_target(self, candidates: List[str]) -> Optional[str]:
        """Argmax EFE: the most uncertain topic is the best epistemic action."""
        best, best_e = None, -1.0
        for t in candidates:
            e = self.score(t).total
            if e > best_e:
                best_e, best = e, t
        return best

    def act_and_close_loop(self, topic: str, efe_before: float,
                           efe_after_fn) -> Tuple[float, float, bool]:
        """Record the action + verify EFE dropped (loop closed).
        efe_after_fn: callable returning the post-read EFE for `topic`.
        Returns (efe_before, efe_after, closed)."""
        efe_after = efe_after_fn(topic)
        closed = efe_after < efe_before
        self._history.append((topic, efe_before, efe_after))
        return efe_before, efe_after, closed

    def loop_closed_rate(self) -> float:
        if not self._history:
            return 0.0
        closed = sum(1 for _, b, a in self._history if a < b)
        return closed / len(self._history)
