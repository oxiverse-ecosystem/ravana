"""
RAVANA v2 — MEANING ENGINE
Intrinsic motivation via costly coherence gain.

PRINCIPLE: Meaning is not programmed — it emerges from the pursuit of
coherence gains that cost effort to achieve.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math


@dataclass
class MeaningConfig:
    """Configuration for meaning computation."""
    # Weights for meaning formula components
    w_dissonance_reduction: float = 0.4
    w_identity_coherence: float = 0.3
    w_predictive_power: float = 0.3
    
    # Effort scaling
    effort_kappa: float = 0.5  # How much effort amplifies meaning
    
    # Validation
    validation_horizon: int = 10  # Episodes to verify meaning authenticity
    transfer_bonus_weight: float = 0.2
    
    # Accumulation
    decay_rate: float = 0.01  # Meaning decay per episode without gain
    max_history: int = 1000


@dataclass
class MeaningRecord:
    """Record of a meaning computation event."""
    episode: int
    raw_meaning: float
    effort: float
    coherence_gain: float
    identity_coherence_gain: float
    predictive_gain: float
    effective_meaning: float
    authentic: bool
    components: Dict[str, float]


class MeaningEngine:
    """
    Computes meaning as costly coherence gain.
    
    Meaning Formula:
        M = w1 * (-ΔD_future) + w2 * (Δidentity_coherence) + w3 * (Δpredictive_power)
            × (1 + κ * effort_cost)
    
    Meaning-staking: committing to beliefs imposes identity cost if wrong.
    Meaning-driven curiosity: pursue actions with high expected M.
    """
    
    def __init__(self, config: Optional[MeaningConfig] = None):
        self.config = config or MeaningConfig()
        self.accumulated_meaning: float = 0.0
        self.history: List[MeaningRecord] = []
        self._recent_predictive_gains: List[float] = []
        self._commitments: Dict[str, float] = {}  # belief_id -> staked meaning
        
    def compute_meaning(
        self,
        episode: int,
        pre_dissonance: float,
        post_dissonance: float,
        pre_identity: float,
        post_identity: float,
        predictive_gain: float = 0.0,
        effort: float = 0.0,
        returns: Optional[List[float]] = None,
    ) -> MeaningRecord:
        """
        Compute meaning from a cognitive event.
        
        Args:
            pre/post_dissonance: Dissonance before and after event
            pre/post_identity: Identity coherence before and after
            predictive_gain: Improvement in prediction accuracy
            effort: Cognitive effort expended (0-1)
            returns: Future returns for validation (optional)
        
        Returns:
            MeaningRecord with full breakdown
        """
        c = self.config
        
        # Component 1: Dissonance reduction (negative delta = good)
        dissonance_reduction = max(0.0, pre_dissonance - post_dissonance)
        
        # Component 2: Identity coherence gain
        identity_gain = max(0.0, post_identity - pre_identity)
        
        # Component 3: Predictive power gain
        # Smooth predictive gain with EMA
        self._recent_predictive_gains.append(predictive_gain)
        if len(self._recent_predictive_gains) > 10:
            self._recent_predictive_gains = self._recent_predictive_gains[-10:]
        smoothed_predictive = float(np.mean(self._recent_predictive_gains)) if self._recent_predictive_gains else 0.0
        
        # Raw meaning (before effort scaling)
        raw_meaning = (
            c.w_dissonance_reduction * dissonance_reduction
            + c.w_identity_coherence * identity_gain
            + c.w_predictive_power * smoothed_predictive
        )
        
        # Effort scaling: costly gains are more meaningful
        effort_multiplier = 1.0 + c.effort_kappa * effort
        effective_meaning = raw_meaning * effort_multiplier
        
        # Authenticity check: high meaning without real gain is flagged
        authentic = True
        if raw_meaning < 0.05 and effort_multiplier > 1.5:
            # Suspicious: low real gain but high effort multiplier
            authentic = False
            effective_meaning *= 0.5  # Penalty for inauthentic meaning
        
        # Accumulate
        self.accumulated_meaning += effective_meaning
        
        record = MeaningRecord(
            episode=episode,
            raw_meaning=raw_meaning,
            effort=effort,
            coherence_gain=dissonance_reduction,
            identity_coherence_gain=identity_gain,
            predictive_gain=smoothed_predictive,
            effective_meaning=effective_meaning,
            authentic=authentic,
            components={
                "dissonance_reduction": dissonance_reduction,
                "identity_gain": identity_gain,
                "predictive_gain": smoothed_predictive,
                "effort_multiplier": effort_multiplier,
                "raw": raw_meaning,
            }
        )
        
        self.history.append(record)
        if len(self.history) > self.config.max_history:
            self.history = self.history[-self.config.max_history:]
        
        return record
    
    def stake_meaning(self, belief_id: str, stake_amount: float):
        """
        Commit meaning to a belief.
        
        If the belief is later falsified, the staked meaning is lost.
        This creates identity cost for holding false beliefs.
        """
        if belief_id in self._commitments:
            self._commitments[belief_id] += stake_amount
        else:
            self._commitments[belief_id] = stake_amount
    
    def resolve_stake(self, belief_id: str, belief_held: bool) -> float:
        """
        Resolve a staked commitment.
        
        If belief was correct, meaning is preserved.
        If belief was wrong, staked meaning is deducted.
        
        Returns:
            Meaning change (positive for correct, negative for wrong)
        """
        staked = self._commitments.pop(belief_id, 0.0)
        if staked == 0.0:
            return 0.0
        
        if belief_held:
            # Belief was correct — staked meaning is preserved
            return 0.0
        else:
            # Belief was wrong — lose the staked meaning
            loss = -staked * 0.5
            self.accumulated_meaning = max(0.0, self.accumulated_meaning + loss)
            return loss
    
    def get_expected_meaning(
        self,
        predicted_dissonance_gain: float,
        predicted_identity_gain: float,
        predicted_predictive_gain: float,
        estimated_effort: float,
    ) -> float:
        """
        Compute expected meaning for planning/curiosity.
        
        Used by the strategy/intent system to select actions
        that maximize expected meaning gain.
        """
        raw = (
            self.config.w_dissonance_reduction * predicted_dissonance_gain
            + self.config.w_identity_coherence * predicted_identity_gain
            + self.config.w_predictive_power * predicted_predictive_gain
        )
        effort_mult = 1.0 + self.config.effort_kappa * estimated_effort
        return raw * effort_mult
    
    def get_status(self) -> Dict[str, Any]:
        """Full meaning engine status."""
        recent = self.history[-20:] if self.history else []
        return {
            "accumulated_meaning": self.accumulated_meaning,
            "active_commitments": len(self._commitments),
            "total_episodes_tracked": len(self.history),
            "recent_meaning_rate": (
                float(np.mean([r.effective_meaning for r in recent]))
                if recent else 0.0
            ),
            "authenticity_rate": (
                sum(1 for r in recent if r.authentic) / len(recent)
                if recent else 1.0
            ),
        }
