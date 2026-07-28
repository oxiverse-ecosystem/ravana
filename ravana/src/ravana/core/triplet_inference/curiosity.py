"""Curiosity hook — exposes relational-structure uncertainty to the
existing ActiveInferenceController (core/active_inference.py).

The controller scores topics via gap_fn/pfe_fn callables; this hook
provides an epistemic-value signal: predicates whose profiles have
LITTLE evidence relative to the current distribution of evidence
volumes (distribution-relative, no fixed EXPLORE_THRESHOLD).
"""
from __future__ import annotations

from typing import List

from .memory import TripletMemory


class InferenceCuriosityHook:
    def __init__(self, memory: TripletMemory):
        self.memory = memory

    def epistemic_value(self, predicate: str) -> float:
        """0..1 — how much would more data on this predicate teach us?
        High when the predicate's evidence volume is below the mean
        volume across known predicates."""
        profs = list(self.memory.profiles.values())
        if not profs:
            return 1.0
        volumes = [p.transitivity_n + p.symmetry_n for p in profs]
        mean_v = sum(volumes) / len(volumes)
        prof = self.memory.profiles.get(predicate)
        v = (prof.transitivity_n + prof.symmetry_n) if prof else 0
        if mean_v <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - v / (2.0 * mean_v)))

    def curiosity_targets(self, top_k: int = 5) -> List[str]:
        """Predicates with the highest epistemic value — candidates for
        ActiveInferenceController.select_target / web reading."""
        preds = list(self.memory.profiles.keys())
        preds.sort(key=self.epistemic_value, reverse=True)
        return preds[:top_k]
