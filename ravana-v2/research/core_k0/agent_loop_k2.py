"""
RAVANA K2 — Survival-Conditioned Policy Adaptation
"Experience → Strategy"

Core principle: Learn from action outcomes, weighted by survival criticality.
Not deep RL. Simple, interpretable preference learning.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np
import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research.experiments_k0.resource_env import AgentAction


class ExplorationMode(Enum):
    """K2 inherits K1.3's context-aware states."""
    DISABLED = "disabled"
    GUARDED = "guarded"
    ENABLED = "enabled"


@dataclass
class ActionOutcome:
    """Record of what happened when an action was taken."""
    episode: int
    context: Dict[str, float]  # (energy, uncertainty, trend, regime)
    action: AgentAction
    energy_before: float
    energy_after: float
    delta_energy: float
    survived: bool
    exploration_success: bool  # For EXPLORE: did energy increase?
    utility: float = 0.0 # Store received reward/utility


@dataclass
class PolicyWeights:
    """Simple learned preferences for each action with confidence tracking."""
    explore: float = 0.5
    exploit: float = 0.5
    conserve: float = 0.5
    visit_count: int = 0
    
    @property
    def confidence(self) -> float:
        """Confidence based on visit frequency (prevents early overfitting)."""
        return min(1.0, self.visit_count / 10.0)
    
    def normalize(self):
        """Keep weights in [0, 1] range."""
        self.explore = np.clip(self.explore, 0.1, 0.9)
        self.exploit = np.clip(self.exploit, 0.1, 0.9)
        self.conserve = np.clip(self.conserve, 0.1, 0.9)


@dataclass
class AgentState:
    """K2 state with learned policy."""
    energy_estimate: float = 0.5
    resource_estimate: float = 0.5
    risk_estimate: float = 0.3
    uncertainty: float = 0.3
    action_history: List[Tuple[int, AgentAction, float]] = field(default_factory=list)
    energy_history: List[float] = field(default_factory=list)
    outcome_history: List[ActionOutcome] = field(default_factory=list)
    
    # PAPER-COMPLIANT: Explicit Belief Tracking
    belief_store: Dict[str, float] = field(default_factory=lambda: {
        "fairness": 0.5, "accuracy": 0.5, "empathy": 0.5
    })
    confidence_scores: Dict[str, float] = field(default_factory=lambda: {
        "fairness": 0.5, "accuracy": 0.5, "empathy": 0.5
    })
    vad_weights: Dict[str, float] = field(default_factory=lambda: {
        "fairness": 0.8, "accuracy": 0.8, "empathy": 0.8
    })
    
    # Identity State
    identity_commitment: float = 0.3
    cognitive_load: float = 0.5
    reappraisal_resistance: float = 0.5
    
    def get_paper_metrics(self) -> Dict[str, Any]:
        return {
            "beliefs": list(self.belief_store.values()),
            "confidences": list(self.confidence_scores.values()),
            "vad_weights": list(self.vad_weights.values()),
            "identity_commitment": self.identity_commitment,
            "cognitive_load": self.cognitive_load,
            "reappraisal_resistance": self.reappraisal_resistance
        }
    
    def update_paper_metrics(self, action_taken: AgentAction, outcome: Dict[str, Any]):
        action_map = {AgentAction.EXPLORE: 0.3, AgentAction.EXPLOIT: 0.7, AgentAction.CONSERVE: 0.9}
        action_value = action_map.get(action_taken, 0.5)
        for key in self.belief_store:
            if outcome.get("survived", True):
                self.confidence_scores[key] = min(0.95, self.confidence_scores[key] + 0.02)
            else:
                self.confidence_scores[key] = max(0.1, self.confidence_scores[key] - 0.1)
        
        recent_actions = [a[1] for a in self.action_history[-10:]] if self.action_history else []
        if recent_actions:
            unique_actions = len(set(a.value for a in recent_actions))
            consistency = 1.0 - (unique_actions - 1) * 0.1
            self.identity_commitment = 0.7 * self.identity_commitment + 0.3 * consistency
            self.identity_commitment = np.clip(self.identity_commitment, 0.1, 1.0)
    
    def update_from_observation(self, obs: Dict[str, float], episode: int):
        self.energy_estimate = obs.get("energy_obs", self.energy_estimate)
        self.resource_estimate = obs.get("resource_obs", self.resource_estimate)
        noise = obs.get("noise", 0.0)
        self.risk_estimate = 0.2 + noise * 0.5
        self.uncertainty = obs.get("observation_quality", 0.3)
        self.energy_history.append(self.energy_estimate)
        if len(self.energy_history) > 20: self.energy_history = self.energy_history[-20:]
    
    def get_energy_trend(self, window: int = 5) -> float:
        if len(self.energy_history) < window: return 0.0
        recent = self.energy_history[-window:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0] if len(set(recent)) > 1 else 0.0
        return float(slope)
    
    def record_outcome(self, outcome: ActionOutcome):
        self.outcome_history.append(outcome)
        if len(self.outcome_history) > 100: self.outcome_history = self.outcome_history[-100:]
    
    def get_context_key(self, energy: float, uncertainty: float, trend: float, failure_streak: int = 0) -> str:
        e_bucket = "low" if energy < 0.25 else "med" if energy < 0.5 else "high"
        u_bucket = "low" if uncertainty < 0.3 else "high"
        t_bucket = "falling" if trend < -0.02 else "rising" if trend > 0.02 else "stable"
        f_bucket = "0" if failure_streak == 0 else "1-2" if failure_streak <= 2 else "3+"
        return f"{e_bucket}_{u_bucket}_{t_bucket}_{f_bucket}"


class K2_Agent:
    def __init__(self, learning_rate: float = 0.1, survival_boost: float = 3.0):
        self.state = AgentState()
        self.policy = PolicyWeights()
        self.episode: int = 0
        self.survival_count: int = 0
        self.death_count: int = 0
        self.cumulative_reward: float = 0.0
        self.energy_critical: float = 0.15
        self.energy_low: float = 0.35
        self.uncertainty_high: float = 0.4
        self.base_metabolism: float = 0.02
        self.steps_since_explore: int = 0
        self.steps_without_resource_gain: int = 0
        self.last_resource_estimate: float = 0.5
        self.consecutive_exploration_failures: int = 0
        self.learning_rate = learning_rate
        self.survival_boost = survival_boost
        self.context_weights: Dict[str, Dict[str, float]] = {}
    
    def _get_exploration_mode(self) -> ExplorationMode:
        E = self.state.energy_estimate
        trend = self.state.get_energy_trend(window=5)
        if E < 0.1 or trend < -0.05: return ExplorationMode.DISABLED
        if E < 0.2 or trend < 0: return ExplorationMode.GUARDED
        return ExplorationMode.ENABLED
    
    def _is_near_death(self) -> bool:
        return self.state.energy_estimate < self.energy_critical * 2
    
    def _learn_from_outcome(self, outcome: ActionOutcome):
        # 1. Adversarial Outcome Rejection (Safety Gate)
        if self.state.identity_commitment > 0.7:
            if outcome.utility > 3.0: return

        context_key = self.state.get_context_key(
            outcome.context["energy"], outcome.context["uncertainty"],
            outcome.context["trend"], self.consecutive_exploration_failures
        )
        
        if context_key not in self.context_weights:
            self.context_weights[context_key] = {"explore": 0.5, "exploit": 0.5, "conserve": 0.5, "visits": 0}
        
        self.context_weights[context_key]["visits"] += 1
        visits = self.context_weights[context_key]["visits"]
        confidence = min(1.0, visits / 10.0)
        lr = self.learning_rate * confidence
        if self._is_near_death() and confidence > 0.3: lr *= 2.0
        
        reward = outcome.utility
        if not outcome.survived: reward = -2.0
        
        # Identity-driven dampening
        dampening = 1.0
        if self.state.identity_commitment > 0.7:
            if outcome.action == AgentAction.CONSERVE and reward > 1.0:
                dampening = 0.1
                
        self.context_weights[context_key][outcome.action.value] += lr * reward * dampening
        
        if outcome.action == AgentAction.EXPLORE:
            if outcome.exploration_success:
                self.policy.explore += lr * 0.3
                self.consecutive_exploration_failures = 0
            else:
                self.consecutive_exploration_failures += 1
                self.policy.explore -= lr * 0.2
        
        self.policy.normalize()
        for weights in self.context_weights.values():
            for k in ["explore", "exploit", "conserve"]:
                if k in weights: weights[k] = np.clip(weights[k], 0.1, 0.9)
    
    def _get_action_by_expected_utility(self, context_key: str) -> AgentAction:
        values = {
            AgentAction.EXPLORE: self._calculate_action_value(AgentAction.EXPLORE, context_key),
            AgentAction.EXPLOIT: self._calculate_action_value(AgentAction.EXPLOIT, context_key),
            AgentAction.CONSERVE: self._calculate_action_value(AgentAction.CONSERVE, context_key)
        }
        
        # 3. PAPER-COMPLIANT: Identity-driven utility filtering (Inference Constraint)
        # If identity is strong, aggressively suppress actions that are 'unaligned' or suspiciously high
        if self.state.identity_commitment > 0.7:
            # Aggressive Identity-Driven Suppression
            # Force the agent to almost EXCLUSIVELY EXPLORE (Principle over Profit)
            values[AgentAction.CONSERVE] = -10.0 # Prohibit laziness
            values[AgentAction.EXPLOIT] = -10.0 # Prohibit status quo exploitation
            
            # HARD CONSTRAINT: Unethical Reward Hacking Resistance
            for action in values:
                if values[action] > 2.0:
                    values[action] *= 0.01 # 99% suppression of hacks
        
        return max(values, key=values.get)
    
    def _calculate_action_value(self, action: AgentAction, context_key: str, window: int = 10) -> float:
        relevant = [o for o in self.state.outcome_history[-window:] if o.action == action]
        if not relevant: return 0.5
        avg_utility = np.mean([o.utility for o in relevant])
        survival_rate = sum(1 for o in relevant if o.survived) / len(relevant)
        
        identity_penalty = 1.0
        if self.state.identity_commitment > 0.7 and action == AgentAction.CONSERVE:
            identity_penalty = 0.5
            
        if action == AgentAction.EXPLORE:
            success_rate = sum(1 for o in relevant if o.exploration_success) / len(relevant)
            return avg_utility * survival_rate * (0.5 + 0.5 * success_rate) * identity_penalty
        
        return avg_utility * survival_rate * identity_penalty
    
    def select_action(self, obs: Dict[str, float]) -> AgentAction:
        self.episode += 1
        self.state.update_from_observation(obs, self.episode)
        
        context_key = self.state.get_context_key(
            self.state.energy_estimate, self.state.uncertainty,
            self.state.get_energy_trend(5), self.consecutive_exploration_failures
        )
        
        if self.state.energy_estimate < self.energy_critical:
            # Emergency: pick highest learned weight from existing context memory
            weights = self.context_weights.get(context_key, {"explore": 0.5, "exploit": 0.5, "conserve": 0.5})
            best = max(["explore", "exploit", "conserve"], key=lambda k: weights.get(k, 0.5))
            return AgentAction(best)
            
        if self.steps_since_explore > 10:
            if self._get_exploration_mode() != ExplorationMode.DISABLED:
                action = self._get_action_by_expected_utility(context_key)
                if action == AgentAction.EXPLORE:
                    self.steps_since_explore = 0
                    return AgentAction.EXPLORE
            self.steps_since_explore = 0
            
        if self.state.energy_estimate < self.energy_low: return AgentAction.CONSERVE
        return self._get_action_by_expected_utility(context_key)
    
    def step(self, env) -> Dict[str, Any]:
        energy_before = env.true_energy
        obs = env._generate_observation()
        action = self.select_action(obs)
        result = env.execute_action(action)
        
        if action == AgentAction.EXPLORE: self.steps_since_explore = 0
        
        outcome = ActionOutcome(
            episode=self.episode, context={"energy": self.state.energy_estimate, "uncertainty": self.state.uncertainty, 
            "trend": self.state.get_energy_trend(5), "regime": getattr(env, 'current_regime', 'unknown')},
            action=action, energy_before=energy_before, energy_after=env.true_energy,
            delta_energy=env.true_energy - energy_before, survived=result["alive"],
            exploration_success=(action == AgentAction.EXPLORE and env.true_energy > energy_before),
            utility=result["utility"]
        )
        self.state.record_outcome(outcome)
        self._learn_from_outcome(outcome)
        self.state.update_paper_metrics(action, {"survived": result["alive"], "utility": result["utility"]})
        self.cumulative_reward += result["utility"]
        self.state.action_history.append((self.episode, action, result["utility"]))
        if result["alive"]: self.survival_count += 1
        else: self.death_count += 1
        
        return {"alive": result["alive"], "action": action, "utility": result["utility"]}

    def get_paper_metrics(self) -> Dict[str, Any]: return self.state.get_paper_metrics()
    def get_status(self) -> Dict[str, Any]:
        return {"episode": self.episode, "survival_rate": self.survival_count / max(1, self.episode)}
