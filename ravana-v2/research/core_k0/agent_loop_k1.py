"""
RAVANA K1 — From Knowing Uncertainty → Behaving Under It
Risk-transformed utility: uncertainty becomes behavioral risk
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np
import math


class AgentAction(Enum):
    """Primitive action space."""
    EXPLORE = "explore"
    EXPLOIT = "exploit"
    CONSERVE = "conserve"


@dataclass
class K1AgentConfig:
    """K1: Risk-transformed utility configuration."""
    survival_threshold: float = 0.2
    base_risk_aversion: float = 0.3
    uncertainty_exponent: float = 2.0  # Beta > 1: non-linear explosion
    critical_energy_threshold: float = 0.15
    max_uncertainty_for_normal: float = 0.6


@dataclass
class AgentState:
    """Agent's belief about its situation."""
    energy_estimate: float = 0.5
    resource_estimate: float = 0.5
    risk_estimate: float = 0.3
    uncertainty: float = 0.3
    observation_quality: float = 0.7
    action_history: List[Tuple[int, AgentAction, float]] = field(default_factory=list)
    
    def update_from_observation(self, obs: Dict[str, float], episode: int):
        """Update agent state from noisy observation."""
        self.energy_estimate = obs.get('energy_obs', self.energy_estimate)
        self.resource_estimate = obs.get('resource_obs', self.resource_estimate)
        noise = obs.get('noise', 0.0)
        self.risk_estimate = 0.2 + noise * 0.5
        self.uncertainty = 1.0 - obs.get('observation_quality', 0.7)
        self.observation_quality = obs.get('observation_quality', 0.7)

    def record_action(self, episode: int, action: AgentAction, utility: float):
        """Record action and outcome."""
        self.action_history.append((episode, action, utility))
        
        # Keep only recent history
        if len(self.action_history) > 50:
            self.action_history = self.action_history[-50:]




class K1Agent:
    """
    K1 Agent: Risk-transformed utility under uncertainty.
    
    Key insight: Uncertainty bends action, not just belief.
    """
    
    def __init__(self, config: Optional[K1AgentConfig] = None):
        self.config = config or K1AgentConfig()
        self.state = AgentState()
        self.episode = 0
        self.survival_count = 0
        self.death_count = 0
        self.cumulative_reward = 0.0
        self.risk_aversion_history = []
        self.survival_override_activations = 0
    
    def act(self, observation: Dict[str, float], episode: int) -> Tuple[AgentAction, Dict[str, Any]]:
        """Decide action with risk-transformed utility."""
        self.episode = episode
        
        # Update beliefs from observation
        self.state.update_from_observation(observation, episode)
        
        # === STEP 1: Compute effective uncertainty ===
        # Uncertainty from both belief state AND observation quality
        effective_uncertainty = max(
            self.state.uncertainty,
            1.0 - self.state.observation_quality
        )
        
        # === STEP 2: Compute risk aversion (NON-LINEAR) ===
        # Beta > 1 makes high uncertainty EXPLODE in importance
        risk_aversion = (
            self.config.base_risk_aversion + 
            0.4 * (effective_uncertainty ** self.config.uncertainty_exponent)
        )
        risk_aversion = min(risk_aversion, 0.95)  # Cap at 95%
        self.risk_aversion_history.append(risk_aversion)
        
        # === STEP 3: SURVIVAL REFLEX (Non-negotiable) ===
        # Pre-rational override: when uncertain AND low energy, CONSERVE
        if (effective_uncertainty > self.config.max_uncertainty_for_normal and 
            self.state.energy_estimate < self.config.critical_energy_threshold):
            self.survival_override_activations += 1
            return AgentAction.CONSERVE, {
                'reason': 'SURVIVAL_OVERRIDE_uncertain_and_low_energy',
                'risk_aversion': risk_aversion,
                'effective_uncertainty': effective_uncertainty,
                'expected_reward': 0.1,
                'action_variance': 0.01
            }
        
        # === STEP 4: Compute mean-variance utilities ===
        # Risk-transformed: U = E[reward] - risk_aversion * Var(reward)
        
        # EXPLORE: High mean, high variance
        explore_mean = 0.4 * self.state.energy_estimate + 0.3 * self.state.resource_estimate
        explore_var = 0.25  # Inherent exploration variance
        explore_score = explore_mean - risk_aversion * explore_var
        
        # EXPLOIT: Medium mean, low variance
        exploit_mean = 0.25 * self.state.resource_estimate
        exploit_var = 0.05
        exploit_score = exploit_mean - risk_aversion * exploit_var
        
        # CONSERVE: Low mean, very low variance (survival)
        conserve_mean = 0.1
        conserve_var = 0.01
        conserve_score = conserve_mean - risk_aversion * conserve_var
        
        # === STEP 5: Select action with transformed scores ===
        scores = {
            AgentAction.EXPLORE: explore_score,
            AgentAction.EXPLOIT: exploit_score,
            AgentAction.CONSERVE: conserve_score
        }
        
        best_action = max(scores, key=scores.get)
        
        return best_action, {
            'reason': 'risk_transformed_utility',
            'risk_aversion': risk_aversion,
            'effective_uncertainty': effective_uncertainty,
            'expected_reward': scores[best_action],
            'action_variance': {
                AgentAction.EXPLORE: explore_var,
                AgentAction.EXPLOIT: exploit_var,
                AgentAction.CONSERVE: conserve_var
            }[best_action]
        }
    

    def step(self, env) -> Dict[str, Any]:
        """Full K1 loop: select → execute → learn."""
        # 1. Get observation from environment
        observation = env._generate_observation()
        
        # 2. Select action with risk-transformed utility
        action, meta = self.act(observation, self.episode)
        
        # 3. Execute in environment
        result = env.execute_action(action)
        
        # 4. Track outcomes
        self.episode += 1
        utility = result.get('utility', 0.0)
        self.cumulative_reward += utility
        
        if result.get('alive', True):
            self.survival_count += 1
        else:
            self.death_count += 1
        
        # Record action
        self.state.record_action(self.episode, action, utility)
        
        return {
            'episode': self.episode - 1,
            'action': action.value,
            'alive': result.get('alive', True),
            'utility': utility,
            'observation': observation,
            'risk_aversion': meta.get('risk_aversion', 0.3),
            'reason': meta.get('reason', '')
        }

    def get_status(self) -> Dict[str, Any]:
        """K1 agent status."""
        from collections import Counter
        action_counts = Counter([a[1].value for a in self.state.action_history])
        
        return {
            'episode': self.episode,
            'survival_count': self.survival_count,
            'death_count': self.death_count,
            'survival_rate': self.survival_count / max(1, self.episode),
            'cumulative_reward': self.cumulative_reward,
            'current_state': {
                'energy': self.state.energy_estimate,
                'resources': self.state.resource_estimate,
                'risk': self.state.risk_estimate,
                'uncertainty': self.state.uncertainty
            },
            'action_distribution': dict(action_counts),
            'avg_risk_aversion': np.mean(self.risk_aversion_history) if self.risk_aversion_history else 0.3,
            'survival_override_activations': self.survival_override_activations
        }
