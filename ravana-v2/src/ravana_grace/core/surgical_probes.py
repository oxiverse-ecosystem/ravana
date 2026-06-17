"""
RAVANA v2 — PHASE G.5: Surgical Probing
From "should I probe?" → "which probe maximally separates H1 vs H2?"

PRINCIPLE: KL-divergence driven action selection.
Each probe is an experiment designed to maximally disambiguate competing hypotheses.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from enum import Enum


class ProbeType(Enum):
    """Types of epistemic probes with different information profiles."""
    PERTURB_LOW = "perturb_low"      # Small perturbation, high precision
    PERTURB_MED = "perturb_med"      # Medium perturbation, balanced
    PERTURB_HIGH = "perturb_high"    # Large perturbation, stress test
    EXPLORE_AGGRESSIVE = "explore_aggressive"  # Push boundaries
    STABILIZE_TEST = "stabilize_test"  # Test stabilization response
    NOISE_INJECTION = "noise_injection"  # Test noise sensitivity
    RECOVER_PROBE = "recover_probe"    # Test recovery dynamics


@dataclass
class ProbeOutcome:
    """Predicted outcome of a probe under a hypothesis."""
    probe_type: ProbeType
    hypothesis_id: int
    predicted_dissonance: float
    predicted_identity: float
    predicted_clamp_rate: float
    confidence: float  # How certain is this prediction?


@dataclass
class ProbeExperiment:
    """A designed experiment with expected information gain."""
    probe_type: ProbeType
    expected_kl_divergence: float  # Expected separation between hypotheses
    expected_outcomes: Dict[int, ProbeOutcome]  # Per-hypothesis prediction
    information_gain_estimate: float
    cost: float  # Epistemic/behavioral cost of probe


@dataclass
class SurgicalProbeConfig:
    """Configuration for surgical probing."""
    # KL divergence thresholds
    min_kl_for_probe: float = 0.1    # Minimum KL to justify probing
    target_kl: float = 0.5           # Optimal KL for good separation
    max_kl: float = 1.0             # Avoid overly disruptive probes
    
    # Cost-benefit weights
    information_weight: float = 1.0
    cost_weight: float = 0.3
    risk_weight: float = 0.2
    
    # Probe history
    max_probe_history: int = 100
    min_episodes_between_probes: int = 10
    
    # Adaptation
    learn_probe_effectiveness: bool = True
    probe_success_rate_window: int = 20


class SurgicalProbeSelector:
    """
    Select probes that maximally separate competing hypotheses.
    
    Core insight: Not all probes are equal. Some separate H1 from H2 better.
    """
    
    def __init__(self, config: Optional[SurgicalProbeConfig] = None):
        self.config = config or SurgicalProbeConfig()
        self.probe_history: List[Dict[str, Any]] = []
        self.probe_effectiveness: Dict[ProbeType, List[float]] = {
            probe: [] for probe in ProbeType
        }
        self.last_probe_episode: int = -100  # Allow early probes
        
    def select_surgical_probe(
        self,
        hypotheses: List[Any],  # List of Hypothesis objects
        current_state: Dict[str, float],
        episode: int
    ) -> Tuple[Optional[ProbeType], Dict[str, Any]]:
        """
        Select probe that maximally separates top two hypotheses.
        
        Returns: (probe_type or None, metadata)
        """
        if len(hypotheses) < 2:
            return None, {"reason": "single_hypothesis"}
        
        # Rate limiting
        if episode - self.last_probe_episode < self.config.min_episodes_between_probes:
            return None, {"reason": "rate_limited", "episodes_since_last": episode - self.last_probe_episode}
        
        # Get top two hypotheses by confidence
        sorted_hyps = sorted(enumerate(hypotheses), key=lambda x: x[1].confidence, reverse=True)
        h1_idx, h1 = sorted_hyps[0]
        h2_idx, h2 = sorted_hyps[1] if len(sorted_hyps) > 1 else (None, None)
        
        if h2 is None or h2.confidence < 0.1:
            return None, {"reason": "no_viable_alternative"}
        
        # Design experiments for each probe type
        experiments = self._design_experiments(h1, h2, current_state)
        
        # Score by expected KL divergence (information gain)
        best_experiment = None
        best_score = -np.inf
        
        for exp in experiments:
            # Score = information - cost - risk
            score = (
                self.config.information_weight * exp.information_gain_estimate -
                self.config.cost_weight * exp.cost -
                self.config.risk_weight * self._estimate_risk(exp, current_state)
            )
            
            # Bonus for probes that have worked well historically
            if self.probe_effectiveness[exp.probe_type]:
                avg_effectiveness = np.mean(self.probe_effectiveness[exp.probe_type][-10:])
                score *= (1 + 0.2 * avg_effectiveness)  # Up to 20% bonus
            
            if score > best_score and exp.expected_kl_divergence >= self.config.min_kl_for_probe:
                best_score = score
                best_experiment = exp
        
        if best_experiment is None:
            return None, {"reason": "no_good_probe", "experiments_considered": len(experiments)}
        
        # Record selection
        self.last_probe_episode = episode
        self.probe_history.append({
            "episode": episode,
            "probe": best_experiment.probe_type.value,
            "expected_kl": best_experiment.expected_kl_divergence,
            "h1": h1.boundary_estimate,
            "h2": h2.boundary_estimate if h2 else None
        })
        
        return best_experiment.probe_type, {
            "reason": "kl_maximizing",
            "expected_kl": best_experiment.expected_kl_divergence,
            "target_h1": h1.boundary_estimate,
            "target_h2": h2.boundary_estimate if h2 else None,
            "score": best_score,
            "cost": best_experiment.cost
        }
    
    def _design_experiments(
        self,
        h1: Any,
        h2: Any,
        current_state: Dict[str, float]
    ) -> List[ProbeExperiment]:
        """Design experiments for each probe type, predict outcomes under each hypothesis."""
        experiments = []
        
        current_d = current_state.get('dissonance', 0.5)
        current_i = current_state.get('identity', 0.5)
        
        for probe_type in ProbeType:
            # Predict outcome under H1
            outcome_h1 = self._predict_probe_outcome(probe_type, h1, current_d, current_i)
            # Predict outcome under H2
            outcome_h2 = self._predict_probe_outcome(probe_type, h2, current_d, current_i)
            
            # Calculate expected KL divergence between H1 and H2 outcomes
            kl_div = self._calculate_kl_divergence(outcome_h1, outcome_h2)
            
            # Estimate information gain (proportional to KL)
            info_gain = kl_div * min(h1.confidence, h2.confidence)  # Discount if one is dominant
            
            # Estimate cost
            cost = self._estimate_probe_cost(probe_type, current_state)
            
            exp = ProbeExperiment(
                probe_type=probe_type,
                expected_kl_divergence=kl_div,
                expected_outcomes={h1.id: outcome_h1, h2.id: outcome_h2},
                information_gain_estimate=info_gain,
                cost=cost
            )
            experiments.append(exp)
        
        return experiments
    
    def _predict_probe_outcome(
        self,
        probe_type: ProbeType,
        hypothesis: Any,
        current_d: float,
        current_i: float
    ) -> ProbeOutcome:
        """Predict what would happen if we execute this probe under this hypothesis."""
        # Simplified model: probe affects dissonance based on hypothesis boundary
        h_boundary = hypothesis.boundary_estimate
        
        # Distance to boundary affects response
        distance_to_boundary = abs(h_boundary - current_d)
        
        if probe_type == ProbeType.PERTURB_LOW:
            # Small push, small response
            d_delta = 0.02 * (1 if current_d < h_boundary else -1)
            clamp_pred = 0.02 if distance_to_boundary < 0.1 else 0.0
            
        elif probe_type == ProbeType.PERTURB_MED:
            # Medium push, proportional response
            d_delta = 0.05 * (1 if current_d < h_boundary else -1)
            clamp_pred = 0.05 if distance_to_boundary < 0.15 else 0.02
            
        elif probe_type == ProbeType.PERTURB_HIGH:
            # Large push, may hit boundary
            d_delta = 0.10 * (1 if current_d < h_boundary else -1)
            clamp_pred = 0.15 if distance_to_boundary < 0.20 else 0.08
            
        elif probe_type == ProbeType.EXPLORE_AGGRESSIVE:
            # Push toward boundary aggressively
            d_delta = (h_boundary - current_d) * 0.5
            clamp_pred = 0.20 if abs(d_delta) > 0.1 else 0.10
            
        elif probe_type == ProbeType.STABILIZE_TEST:
            # Test stabilization - should reduce dissonance
            d_delta = -0.03 * (current_d - 0.5)  # Pull toward center
            clamp_pred = 0.01
            
        elif probe_type == ProbeType.NOISE_INJECTION:
            # Add noise - response depends on system stability
            d_delta = np.random.normal(0, 0.05)
            clamp_pred = 0.08 if distance_to_boundary < 0.15 else 0.04
            
        elif probe_type == ProbeType.RECOVER_PROBE:
            # Test recovery from high dissonance
            d_delta = -0.08 if current_d > 0.7 else 0.02
            clamp_pred = 0.10 if current_d > 0.7 else 0.05
            
        else:
            d_delta = 0.0
            clamp_pred = 0.0
        
        predicted_d = np.clip(current_d + d_delta, 0.15, 0.95)
        predicted_i = np.clip(current_i + 0.01, 0.10, 0.95)  # Slight identity gain from learning
        
        return ProbeOutcome(
            probe_type=probe_type,
            hypothesis_id=hypothesis.id,
            predicted_dissonance=predicted_d,
            predicted_identity=predicted_i,
            predicted_clamp_rate=clamp_pred,
            confidence=hypothesis.confidence
        )
    
    def _calculate_kl_divergence(self, outcome1: ProbeOutcome, outcome2: ProbeOutcome) -> float:
        """
        Calculate approximate KL divergence between two predicted outcomes.
        
        Higher KL = probe separates hypotheses better.
        """
        # Simplified KL using dissonance and clamp rate differences
        d_diff = abs(outcome1.predicted_dissonance - outcome2.predicted_dissonance)
        clamp_diff = abs(outcome1.predicted_clamp_rate - outcome2.predicted_clamp_rate)
        
        # Combined divergence (weighted sum)
        kl_approx = d_diff + 0.5 * clamp_diff
        
        return kl_approx
    
    def _estimate_probe_cost(self, probe_type: ProbeType, current_state: Dict[str, float]) -> float:
        """Estimate behavioral/epistemic cost of probe."""
        base_costs = {
            ProbeType.PERTURB_LOW: 0.05,
            ProbeType.PERTURB_MED: 0.10,
            ProbeType.PERTURB_HIGH: 0.20,
            ProbeType.EXPLORE_AGGRESSIVE: 0.25,
            ProbeType.STABILIZE_TEST: 0.08,
            ProbeType.NOISE_INJECTION: 0.15,
            ProbeType.RECOVER_PROBE: 0.18
        }
        
        cost = base_costs.get(probe_type, 0.10)
        
        # Higher cost if near boundary (risky to probe there)
        if current_state.get('dissonance', 0.5) > 0.8:
            cost *= 1.5
        
        return cost
    
    def _estimate_risk(self, experiment: ProbeExperiment, current_state: Dict[str, float]) -> float:
        """Estimate risk of probe (chance of destabilizing system)."""
        # Risk based on predicted clamp rate and current state
        max_clamp = max(
            o.predicted_clamp_rate for o in experiment.expected_outcomes.values()
        )
        
        # Higher risk if already near boundary
        proximity_risk = max(0, current_state.get('dissonance', 0.5) - 0.7) * 2
        
        return max_clamp + proximity_risk
    
    def record_probe_result(
        self,
        probe_type: ProbeType,
        episode: int,
        actual_info_gain: float,
        hypothesis_separation_achieved: float
    ):
        """Record effectiveness of probe for future learning."""
        self.probe_effectiveness[probe_type].append(
            1.0 if actual_info_gain > 0.1 else 0.5 if actual_info_gain > 0.05 else 0.0
        )
        
        # Update history
        for record in self.probe_history:
            if record["episode"] == episode and record["probe"] == probe_type.value:
                record["actual_info_gain"] = actual_info_gain
                record["separation_achieved"] = hypothesis_separation_achieved
                break
    
    def get_surgical_analytics(self) -> Dict[str, Any]:
        """Analytics on surgical probing effectiveness."""
        if not self.probe_history:
            return {"total_probes": 0}
        
        # Calculate probe type effectiveness
        effectiveness = {}
        for probe_type, results in self.probe_effectiveness.items():
            if results:
                effectiveness[probe_type.value] = {
                    "success_rate": np.mean(results),
                    "n_probes": len(results)
                }
        
        # Calculate average expected vs actual KL
        expected_kls = [r.get("expected_kl", 0) for r in self.probe_history]
        actual_kls = [r.get("separation_achieved", 0) for r in self.probe_history if "separation_achieved" in r]
        
        return {
            "total_probes": len(self.probe_history),
            "probe_type_effectiveness": effectiveness,
            "avg_expected_kl": np.mean(expected_kls) if expected_kls else 0,
            "avg_actual_separation": np.mean(actual_kls) if actual_kls else 0,
            "kl_calibration_error": np.mean(expected_kls) - np.mean(actual_kls) if actual_kls else 0
        }


# Convenience alias
SurgicalProbing = SurgicalProbeSelector
