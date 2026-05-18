"""
RAVANA K0: The Smallest Possible Agent
Not a philosopher. An agent that acts and suffers consequences.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np


class AgentAction(Enum):
    """Primitive action space for K0."""
    EXPLORE = "explore"      # Uncertain reward, risk of loss
    EXPLOIT = "exploit"      # Known reward, lower risk
    CONSERVE = "conserve"    # Low risk, minimal reward, survive


@dataclass
class AgentState:
    """Agent's belief about its situation."""
    energy_estimate: float = 0.5
    resource_estimate: float = 0.5
    risk_estimate: float = 0.3
    uncertainty: float = 0.3
    
    # Track action history
    action_history: List[Tuple[int, AgentAction, float]] = field(default_factory=list)
    
    def update_from_observation(self, obs: Dict[str, float], episode: int):
        """Update agent state from noisy observation."""
        self.energy_estimate = obs.get('energy_obs', self.energy_estimate)
        self.resource_estimate = obs.get('resource_obs', self.resource_estimate)
        
        # Infer risk from noise level
        noise = obs.get('noise', 0.0)
        self.risk_estimate = 0.2 + noise * 0.5  # Noise → risk signal
        
        # Uncertainty from observation quality
        self.uncertainty = obs.get('observation_quality', 0.3)
    
    def record_action(self, episode: int, action: AgentAction, utility: float):
        """Record action and outcome."""
        self.action_history.append((episode, action, utility))
        
        # Keep only recent history
        if len(self.action_history) > 50:
            self.action_history = self.action_history[-50:]


@dataclass 
class K0Config:
    """K0: Hardcoded, simple, no overengineering."""
    # Utility weights (hardcoded, not learned initially)
    energy_weight: float = 1.0
    resource_weight: float = 0.8
    survival_weight: float = 2.0  # Highest priority
    
    # Uncertainty penalty (simple)
    uncertainty_penalty: float = 0.5
    
    # Risk aversion
    risk_aversion: float = 0.3
    
    # Minimum energy to survive
    survival_threshold: float = 0.2


class MinimalAgent:
    """
    K0: The smallest agent that acts on beliefs and suffers consequences.
    
    No complex planning. No fancy decision theory.
    Simple utility: energy + resources - uncertainty_penalty - risk_penalty
    Choose action: argmax(utility - uncertainty_penalty)
    """
    
    def __init__(self, config: K0Config = None):
        self.config = config or K0Config()
        self.state = AgentState()
        self.episode: int = 0
        
        # Track outcomes
        self.survival_count: int = 0
        self.death_count: int = 0
        self.cumulative_reward: float = 0.0
    
    def compute_action_utility(self, action: AgentAction) -> float:
        """
        Hardcoded utility: simple weighted sum.
        Not learned — designed for immediate survival.
        """
        cfg = self.config
        state = self.state
        
        # Base utility components
        energy_utility = state.energy_estimate * cfg.energy_weight
        resource_utility = state.resource_estimate * cfg.resource_weight
        
        # Survival bonus if above threshold
        survival_utility = 0.0
        if state.energy_estimate > cfg.survival_threshold:
            survival_utility = cfg.survival_weight
        
        base_utility = energy_utility + resource_utility + survival_utility
        
        # Action-specific adjustments (hardcoded heuristics)
        if action == AgentAction.EXPLORE:
            # High potential, high risk
            potential = base_utility * 1.5  # Could find more resources
            risk_penalty = state.risk_estimate * 0.4  # But risk of failure
            uncertainty_cost = state.uncertainty * cfg.uncertainty_penalty
            return potential - risk_penalty - uncertainty_cost
        
        elif action == AgentAction.EXPLOIT:
            # Medium reward, lower risk
            potential = base_utility * 1.0
            risk_penalty = state.risk_estimate * 0.2
            uncertainty_cost = state.uncertainty * cfg.uncertainty_penalty * 0.7
            return potential - risk_penalty - uncertainty_cost
        
        elif action == AgentAction.CONSERVE:
            # Survival mode — low risk, minimal reward
            potential = base_utility * 0.5
            risk_penalty = state.risk_estimate * 0.1  # Very safe
            uncertainty_cost = state.uncertainty * cfg.uncertainty_penalty * 0.3
            survival_bonus = 0.5 if state.energy_estimate < 0.3 else 0.0
            return potential - risk_penalty - uncertainty_cost + survival_bonus
        
        return base_utility
    
    def select_action(self) -> AgentAction:
        """
        K0 decision: argmax(utility - uncertainty_penalty)
        No fancy math. Simple comparison.
        """
        utilities = {
            action: self.compute_action_utility(action)
            for action in AgentAction
        }
        
        # Add uncertainty penalty to all
        penalty = self.state.uncertainty * self.config.uncertainty_penalty
        adjusted = {
            action: util - penalty 
            for action, util in utilities.items()
        }
        
        # Select best
        best_action = max(adjusted, key=adjusted.get)
        
        # K0 debug: show reasoning
        if self.episode < 20 or self.episode % 50 == 0:
            print(f"   K0: E={self.state.energy_estimate:.2f} R={self.state.resource_estimate:.2f} "
                  f"Risk={self.state.risk_estimate:.2f} U={self.state.uncertainty:.2f}")
            print(f"   Utilities: EXPLORE={adjusted[AgentAction.EXPLORE]:.2f} "
                  f"EXPLOIT={adjusted[AgentAction.EXPLOIT]:.2f} "
                  f"CONSERVE={adjusted[AgentAction.CONSERVE]:.2f}")
            print(f"   → Action: {best_action.value}")
        
        return best_action
    
    def execute_action(self, env, action: AgentAction) -> Dict[str, Any]:
        """
        Execute action in environment, get feedback.
        This is where reality hits.
        """
        result = env.execute_action(action)
        
        # Update agent state from observation
        self.state.update_from_observation(result['observation'], self.episode)
        
        # Record action and utility
        utility = result.get('utility', 0.0)
        self.state.record_action(self.episode, action, utility)
        self.cumulative_reward += utility
        
        # Track survival
        if result.get('alive', True):
            self.survival_count += 1
        else:
            self.death_count += 1
            print(f"   💀 DEATH at episode {self.episode}")
        
        self.episode += 1
        
        return result
    
    def step(self, env) -> Dict[str, Any]:
        """Full K0 loop: select → execute → learn."""
        # 1. Select action
        action = self.select_action()
        
        # 2. Execute
        result = self.execute_action(env, action)
        
        return {
            'episode': self.episode - 1,
            'action': action.value,
            'alive': result.get('alive', True),
            'utility': result.get('utility', 0.0),
            'observation': result['observation']
        }
    
    def get_status(self) -> Dict[str, Any]:
        """K0 agent status."""
        return {
            'episode': self.episode,
            'survival_count': self.survival_count,
            'survival_rate': self.survival_count / max(1, self.episode),
            'death_count': self.death_count,
            'cumulative_reward': self.cumulative_reward,
            'current_state': {
                'energy': self.state.energy_estimate,
                'resources': self.state.resource_estimate,
                'risk': self.state.risk_estimate,
                'uncertainty': self.state.uncertainty
            }
        }
