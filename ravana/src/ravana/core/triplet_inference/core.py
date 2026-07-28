"""Core data structures for the Triplet Inference Operator.

Philosophy (Lippl et al. 2024 PNAS; McClelland et al. 1995 CLS):
inference properties (transitivity, symmetry, inversion, composition)
are LEARNED per-predicate statistics, never hardcoded rules. A
RelationProfile accumulates Beta-style evidence counts; the score is
the posterior mean and gating uses the Wilson lower confidence bound
compared against a cross-predicate adaptive baseline — no fixed
thresholds anywhere (house rule: distribution-driven gates only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 95% two-sided normal quantile for the Wilson score interval. This is a
# statistical constant (confidence level), not a tunable behavior knob.
_WILSON_Z = 1.96


def wilson_lower(pos: int, n: int, z: float = _WILSON_Z) -> float:
    """Wilson score lower bound for a Bernoulli rate.

    Naturally handles the small-n case: with few observations the lower
    bound stays near 0, so no separate MIN_OBSERVATIONS constant is
    needed — evidence volume is priced into the bound itself.
    """
    if n <= 0:
        return 0.0
    phat = pos / n
    denom = 1.0 + z * z / n
    center = phat + z * z / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return max(0.0, (center - margin) / denom)


@dataclass
class Triple:
    """Unified inference triple — bridges FactTriple, Proposition and
    OpenIE Fact (whose object field is named ``obj``)."""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source: str = "conversation"  # conversation | web | inferred | seed
    session_date: Optional[datetime] = None
    absolute_date: Optional[datetime] = None
    superseded: bool = False
    turn_number: int = 0

    def key(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> dict:
        return {
            "s": self.subject, "p": self.predicate, "o": self.object,
            "c": self.confidence, "src": self.source,
            "sup": self.superseded, "turn": self.turn_number,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Triple":
        return cls(subject=d["s"], predicate=d["p"], object=d["o"],
                   confidence=float(d.get("c", 0.8)),
                   source=d.get("src", "conversation"),
                   superseded=bool(d.get("sup", False)),
                   turn_number=int(d.get("turn", 0)))


@dataclass
class RelationProfile:
    """Per-predicate learned inference properties.

    Evidence is kept as raw positive/negative counts (Beta pseudo-
    counts). score properties expose the posterior mean; gate decisions
    use ``wilson_lower`` so weak evidence fails closed automatically.
    Profiles are created LAZILY on first observation of a predicate
    (there is no RelationOntology in this repo — predicates are
    discovered from experience, which is more faithful to the
    'learned not authored' philosophy anyway).
    """
    predicate: str
    family: str = ""  # optional typed family (graph_typing), "" = unknown

    # Transitivity evidence: chains A->B->C observed; pos when A->C held.
    transitivity_pos: int = 0
    transitivity_neg: int = 0
    # Symmetry evidence: pos when (B,r,A) existed for observed (A,r,B).
    symmetry_pos: int = 0
    symmetry_neg: int = 0

    # Learned inverse predicate: counts of (A,r,B) co-occurring with
    # (B,r2,A) per candidate r2.
    inverse_counts: Dict[str, int] = field(default_factory=dict)

    # Composition: (first_pred -> {composed_pred -> count}) observed when
    # (A, self, B) and (B, first, C) chained while (A, composed, C) held.
    composition_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Hierarchy: running mean of observed chain depth for this predicate.
    depth_sum: float = 0.0
    depth_n: int = 0

    # ── derived scores ────────────────────────────────────────────────
    @property
    def transitivity_n(self) -> int:
        return self.transitivity_pos + self.transitivity_neg

    @property
    def transitivity_score(self) -> float:
        n = self.transitivity_n
        return self.transitivity_pos / n if n else 0.0

    def transitivity_lower(self) -> float:
        return wilson_lower(self.transitivity_pos, self.transitivity_n)

    @property
    def symmetry_n(self) -> int:
        return self.symmetry_pos + self.symmetry_neg

    @property
    def symmetry_score(self) -> float:
        n = self.symmetry_n
        return self.symmetry_pos / n if n else 0.0

    def symmetry_lower(self) -> float:
        return wilson_lower(self.symmetry_pos, self.symmetry_n)

    @property
    def conjunctivity_alpha(self) -> float:
        """Lippl et al. 2024 conjunctivity factor: alpha = 1 - transitivity."""
        return 1.0 - self.transitivity_score

    @property
    def hierarchy_depth(self) -> float:
        return self.depth_sum / self.depth_n if self.depth_n else 1.0

    def inverse_predicate(self) -> Tuple[Optional[str], float]:
        """Best inverse candidate and its share of inverse evidence."""
        if not self.inverse_counts:
            return None, 0.0
        total = sum(self.inverse_counts.values())
        pred, cnt = max(self.inverse_counts.items(), key=lambda kv: kv[1])
        return pred, cnt / total if total else 0.0

    # ── persistence ───────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "predicate": self.predicate, "family": self.family,
            "t_pos": self.transitivity_pos, "t_neg": self.transitivity_neg,
            "s_pos": self.symmetry_pos, "s_neg": self.symmetry_neg,
            "inv": dict(self.inverse_counts),
            "comp": {k: dict(v) for k, v in self.composition_counts.items()},
            "d_sum": self.depth_sum, "d_n": self.depth_n,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RelationProfile":
        p = cls(predicate=d["predicate"], family=d.get("family", ""))
        p.transitivity_pos = int(d.get("t_pos", 0))
        p.transitivity_neg = int(d.get("t_neg", 0))
        p.symmetry_pos = int(d.get("s_pos", 0))
        p.symmetry_neg = int(d.get("s_neg", 0))
        p.inverse_counts = {k: int(v) for k, v in d.get("inv", {}).items()}
        p.composition_counts = {
            k: {k2: int(v2) for k2, v2 in v.items()}
            for k, v in d.get("comp", {}).items()}
        p.depth_sum = float(d.get("d_sum", 0.0))
        p.depth_n = int(d.get("d_n", 0))
        return p


@dataclass
class RelationalSchema:
    """A recurring relational pattern extracted during consolidation."""
    pattern_type: str            # transitive-chain | symmetric-pair | composition
    predicate: str
    confidence: float = 0.5
    n_exemplars: int = 0
    first_predicate: Optional[str] = None
    second_predicate: Optional[str] = None
    composed_predicate: Optional[str] = None

    def key(self) -> str:
        if self.pattern_type == "composition":
            return (f"composition:{self.first_predicate}"
                    f"∘{self.second_predicate}→{self.composed_predicate}")
        return f"{self.pattern_type}:{self.predicate}"

    def to_dict(self) -> dict:
        return {
            "pattern_type": self.pattern_type, "predicate": self.predicate,
            "confidence": self.confidence, "n_exemplars": self.n_exemplars,
            "first_predicate": self.first_predicate,
            "second_predicate": self.second_predicate,
            "composed_predicate": self.composed_predicate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RelationalSchema":
        return cls(**d)


@dataclass
class InferenceResult:
    """One inferred (or directly retrieved) answer with provenance."""
    triple: Triple
    confidence: float
    path: str          # human-readable chain, e.g. "cat→mammal→animal"
    operator: str      # lookup | transitive | symmetric | inverse | composition | hrr
