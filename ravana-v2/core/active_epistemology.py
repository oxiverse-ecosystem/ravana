"""
RAVANA v2 — PHASE G: Active Epistemology
From passive reasoner to intentional discoverer.

PRINCIPLE: Don't just maintain hypotheses. Actively resolve them.

Components:
1. Information Gain Calculator (VoI)
2. Hypothesis-Driven Action Selection
3. Intentional Experimentation Planner
4. Uncertainty-Aware Risk Taking

This is where RAVANA becomes curious with direction.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class InformationGainMethod(Enum):
    """Methods for computing information gain."""
    EXPECTED_DIVERGENCE = "expected_divergence"  # How much would beliefs separate?
    ENTROPY_REDUCTION = "entropy_reduction"        # How much uncertainty would decrease?
    HYPOTHESIS_ELIMINATION = "hypothesis_elimination"  # Would this falsify a hypothesis?


@dataclass
class VoIConfig:
    """Value of Information configuration."""
    # Information gain calculation
    method: InformationGainMethod = InformationGainMethod.EXPECTED_DIVERGENCE
    
    # Exploration-exploitation tradeoff
    info_gain_weight: float = 0.3      # How much to value information vs outcome
    uncertainty_threshold: float = 0.15   # When to prioritize info over outcome
    
    # Action space for experimentation
    probe_actions: List[str] = field(default_factory=lambda: [
        "aggressive_explore",   # High dissonance, tests boundary
        "conservative_stabilize",  # Low dissonance, tests floor
        "perturb_test",        # Induced variation, tests response
        "hold_steady",         # Baseline, tests stability
    ])
    
    # Intentional experimentation
    min_probe_interval: int = 50        # Don't probe too often
    max_hypothesis_age: int = 200     # Force resolution if hypothesis old
    confidence_gap_action: float = 0.1  # Act when gap this small


@dataclass
class HypothesisDivergence:
    """Expected divergence between hypotheses under action."""
    action: str
    h1_belief: float
    h2_belief: float
    expected_divergence: float  # How much would they separate?
    info_gain: float
    cost: float  # Risk cost of action
    net_value: float  # info_gain - cost


@dataclass
class ExperimentPlan:
    """Planned experiment to resolve uncertainty."""
    target_hypotheses: Tuple[int, int]  # Which two to distinguish
    action: str
    expected_duration: int
    success_criterion: str
    urgency: float  # How important is this experiment?


class InformationGainCalculator:
    """
    Calculate Value of Information (VoI) for actions.
    
    Asks: "What would I learn by doing X?"
    """
    
    def __init__(self, config: Optional[VoIConfig] = None):
        self.config = config or VoIConfig()
        self.divergence_history: List[HypothesisDivergence] = []
        
    def calculate_voi(
        self,
        hypotheses: Dict[int, Dict[str, Any]],
        current_belief: float,
        current_uncertainty: float,
        available_actions: List[str]
    ) -> Dict[str, float]:
        """
        Calculate Value of Information for each action.
        
        Returns: action -> info_gain mapping
        """
        if len(hypotheses) < 2:
            return {action: 0.0 for action in available_actions}
        
        # Get top two competing hypotheses
        sorted_hyps = sorted(
            hypotheses.items(),
            key=lambda x: x[1]['confidence'],
            reverse=True
        )[:2]
        
        h1_id, h1 = sorted_hyps[0]
        h2_id, h2 = sorted_hyps[1]
        
        info_gains = {}
        
        for action in available_actions:
            # Simulate: what would each hypothesis predict?
            h1_prediction = self._simulate_hypothesis(h1, action)
            h2_prediction = self._simulate_hypothesis(h2, action)
            
            # Information gain = expected divergence
            divergence = abs(h1_prediction - h2_prediction)
            
            # Weight by action cost
            cost = self._action_cost(action, current_belief)
            
            # Net value
            net_value = divergence - cost * 0.1
            
            info_gains[action] = net_value
            
            # Record
            self.divergence_history.append(HypothesisDivergence(
                action=action,
                h1_belief=h1['belief'],
                h2_belief=h2['belief'],
                expected_divergence=divergence,
                info_gain=net_value,
                cost=cost,
                net_value=net_value
            ))
        
        return info_gains
    
    def _simulate_hypothesis(self, hypothesis: Dict, action: str) -> float:
        """
        Simulate what this hypothesis predicts for given action.
        """
        belief = hypothesis['belief']
        
        # Each action probes different aspect
        if action == "aggressive_explore":
            # Tests upper boundary
            return min(0.95, belief + 0.1)
        elif action == "conservative_stabilize":
            # Tests lower region
            return max(0.15, belief - 0.05)
        elif action == "perturb_test":
            # Tests response dynamics
            return belief + np.random.normal(0, 0.05)
        else:  # hold_steady
            return belief
    
    def _action_cost(self, action: str, current_belief: float) -> float:
        """
        Cost of action: risk of hitting constraint.
        """
        if action == "aggressive_explore":
            return 0.3  # Risky
        elif action == "perturb_test":
            return 0.2
        elif action == "conservative_stabilize":
            return 0.1
        else:
            return 0.0  # Safe
    
    def should_probe_for_info(
        self,
        top_two_confidence_gap: float,
        episodes_since_last_probe: int,
        current_uncertainty: float
    ) -> bool:
        """
        Decide: Should we intentionally probe to reduce uncertainty?
        """
        # Not enough time since last probe
        if episodes_since_last_probe < self.config.min_probe_interval:
            return False
        
        # Gap too large - already confident
        if top_two_confidence_gap > self.config.confidence_gap_action * 2:
            return False
        
        # High uncertainty and close competition -> probe!
        if current_uncertainty > self.config.uncertainty_threshold:
            return True
        
        # Close competition needs resolution
        if top_two_confidence_gap < self.config.confidence_gap_action:
            return True
        
        return False


class HypothesisDrivenActionSelector:
    """
    Select actions to maximize hypothesis separation.
    
    Principle: "What action would prove H1 wrong fastest?"
    """
    
    def __init__(self, voi_calc: InformationGainCalculator):
        self.voi = voi_calc
        self.last_probe_episode: int = 0
        self.probe_count: int = 0
        self.experiments_conducted: List[ExperimentPlan] = []
        
    def select_action(
        self,
        hypotheses: Dict[int, Dict[str, Any]],
        current_belief: float,
        current_uncertainty: float,
        episode: int,
        default_action: str = "explore_normal"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select action with hypothesis-driven information gain.
        
        Returns: (action, metadata)
        """
        # Calculate info gain for all actions
        info_gains = self.voi.calculate_voi(
            hypotheses,
            current_belief,
            current_uncertainty,
            self.voi.config.probe_actions + [default_action]
        )
        
        # Check if we should intentionally probe
        sorted_hyps = sorted(
            hypotheses.items(),
            key=lambda x: x[1]['confidence'],
            reverse=True
        )
        
        if len(sorted_hyps) >= 2:
            gap = sorted_hyps[0][1]['confidence'] - sorted_hyps[1][1]['confidence']
        else:
            gap = 1.0
        
        episodes_since_probe = episode - self.last_probe_episode
        
        should_probe = self.voi.should_probe_for_info(
            gap,
            episodes_since_probe,
            current_uncertainty
        )
        
        if should_probe:
            # Select best information-gaining action
            best_action = max(info_gains, key=info_gains.get)
            
            if info_gains[best_action] > 0.05:  # Worthwhile info gain
                self.last_probe_episode = episode
                self.probe_count += 1
                
                # Record experiment
                if len(sorted_hyps) >= 2:
                    experiment = ExperimentPlan(
                        target_hypotheses=(sorted_hyps[0][0], sorted_hyps[1][0]),
                        action=best_action,
                        expected_duration=20,
                        success_criterion=f"confidence_gap > {gap + 0.1:.2f}",
                        urgency=current_uncertainty
                    )
                    self.experiments_conducted.append(experiment)
                
                return best_action, {
                    "reason": "hypothesis_driven_probe",
                    "info_gain": info_gains[best_action],
                    "confidence_gap": gap,
                    "experiment_number": self.probe_count
                }
        
        # Default: normal action with slight info preference
        # Still bias slightly toward informative actions even when not probing
        if current_uncertainty > self.voi.config.uncertainty_threshold * 0.5:
            # Subtle bias toward informative actions
            exploration_bias = {k: v * 0.2 for k, v in info_gains.items()}
            if exploration_bias:
                best_exploratory = max(exploration_bias, key=exploration_bias.get)
                if exploration_bias[best_exploratory] > 0.02:
                    return best_exploratory, {
                        "reason": "info_biased_default",
                        "info_gain": info_gains.get(best_exploratory, 0),
                        "subtle": True
                    }
        
        return default_action, {
            "reason": "exploitation",
            "info_gain": info_gains.get(default_action, 0)
        }
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Summary of active epistemology activity."""
        return {
            "total_probes": self.probe_count,
            "experiments_conducted": len(self.experiments_conducted),
            "last_probe_episode": self.last_probe_episode,
            "avg_divergence": np.mean([d.expected_divergence for d in self.voi.divergence_history[-10:]]) if self.voi.divergence_history else 0.0,
            "recent_info_gains": [d.info_gain for d in self.voi.divergence_history[-5:]] if self.voi.divergence_history else []
        }


class ActiveEpistemologyLayer:
    """
    Full active epistemology integration.
    
    Wraps belief reasoner and adds intentional discovery.
    """
    
    def __init__(
        self,
        belief_reasoner,
        config: Optional[VoIConfig] = None
    ):
        self.belief = belief_reasoner
        self.config = config or VoIConfig()
        self.voi_calc = InformationGainCalculator(self.config)
        self.action_selector = HypothesisDrivenActionSelector(self.voi_calc)
        
        # Tracking
        self.info_gain_history: List[float] = []
        self.uncertainty_resolved: int = 0
        
    def act_and_learn(
        self,
        episode: int,
        pre_state: Dict[str, Any],
        mode: Any
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select action with active epistemology.
        
        Returns: (action, metadata)
        """
        # Get current belief state
        belief_list = self.belief.get_belief_state()
        hypotheses = {i: {'id': h.id, 'belief': h.boundary_estimate, 'confidence': h.confidence, 'uncertainty': h.uncertainty} for i, h in enumerate(belief_list)}
        
        if len(hypotheses) < 2:
            # No competing hypotheses - just act normally
            return "explore_normal", {
                "reason": "single_hypothesis",
                "info_gain": 0.0
            }
        
        # Calculate current uncertainty
        sorted_hyps = sorted(
            hypotheses.items(),
            key=lambda x: x[1]['confidence'],
            reverse=True
        )
        top_confidence = sorted_hyps[0][1]['confidence']
        current_uncertainty = 1.0 - top_confidence
        
        # Select hypothesis-driven action
        action, metadata = self.action_selector.select_action(
            hypotheses,
            self.belief.current_belief,
            current_uncertainty,
            episode
        )
        
        # Track
        self.info_gain_history.append(metadata.get('info_gain', 0.0))
        
        if metadata.get("reason") == "hypothesis_driven_probe":
            self.uncertainty_resolved += 1
        
        return action, metadata
    
    def get_epistemic_status(self) -> Dict[str, Any]:
        """Full epistemic system status."""
        return {
            "belief": self.belief.get_belief_state(),
            "action_selection": self.action_selector.get_experiment_summary(),
            "total_info_gain": sum(self.info_gain_history),
            "uncertainties_resolved": self.uncertainty_resolved,
            "avg_info_gain": np.mean(self.info_gain_history[-50:]) if self.info_gain_history else 0.0
        }


# Convenience alias
ActiveEpistemology = ActiveEpistemologyLayer
