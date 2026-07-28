"""AbstentionGate — fail-closed inference filtering.

Distribution-driven (house rule): the confidence floor is the running
distribution of PAST EMITTED inference confidences (mu - sigma), seeded
pessimistically. A candidate must clear the floor of what this system
has previously been willing to assert — no fixed 0.3 constant.
Temporal supersedence always abstains regardless of confidence.
"""
from __future__ import annotations

from typing import Optional

from .core import InferenceResult
from .memory import TripletMemory


class AbstentionGate:
    def __init__(self):
        # Running stats over emitted confidences (EMA, like the engine's
        # _adaptive_baselines {mu, sigma, n} shape).
        self.mu = 0.5
        self.sigma = 0.15
        self.n = 0
        self.eta = 0.05

    def should_abstain(self, result: InferenceResult,
                       memory: TripletMemory) -> bool:
        # 1. Direct lookups never abstain — they're retrieval, not inference.
        if result.operator == "lookup":
            self._update(result.confidence)
            return False
        # 2. Temporal supersedence: a superseded contradicting fact means
        # the world changed — abstain from inferring over stale structure.
        t = result.triple
        for old in memory.triples:
            if (old.superseded and old.subject == t.subject
                    and old.predicate == t.predicate):
                return True
        # 3. Adaptive confidence floor.
        floor = max(0.05, self.mu - self.sigma)
        if result.confidence < floor:
            return True
        self._update(result.confidence)
        return False

    def _update(self, x: float) -> None:
        self.mu = (1.0 - self.eta) * self.mu + self.eta * x
        self.sigma = (1.0 - self.eta) * self.sigma + self.eta * abs(x - self.mu)
        self.n += 1

    def to_dict(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma, "n": self.n}

    def from_dict(self, d: dict) -> None:
        self.mu = float(d.get("mu", 0.5))
        self.sigma = float(d.get("sigma", 0.15))
        self.n = int(d.get("n", 0))
