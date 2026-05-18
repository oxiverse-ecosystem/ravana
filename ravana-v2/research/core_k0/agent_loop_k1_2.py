"""
RAVANA K1.2 — Starvation Trigger + Exploration Floor
Survival-first agent with anti-conservatism protections.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np

# Import AgentAction from environment (don't define our own)
import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research.experiments_k0.resource_env import AgentAction


@dataclass
class AgentState:
    energy_estimate: float = 0.5
    resource_estimate: float = 0.5
    risk_estimate: float = 0.3
    uncertainty: float = 0.3
    action_history: List[Tuple[int, AgentAction, float]] = field(default_factory=list)
    energy_history: List[float] = field(default_factory=list)
    
    def update_from_observation(self, obs: Dict[str, float], episode: int):
        self.energy_estimate = obs.get("energy_obs", self.energy_estimate)
        self.resource_estimate = obs.get("resource_obs", self.resource_estimate)
        noise = obs.get("noise", 0.0)
        self.risk_estimate = 0.2 + noise * 0.5
        self.uncertainty = obs.get("observation_quality", 0.3)
        self.energy_history.append(self.energy_estimate)
        if len(self.energy_history) > 20:
            self.energy_history = self.energy_history[-20:]
    
    def record_action(self, episode: int, action: AgentAction, utility: float):
        self.action_history.append((episode, action, utility))
        if len(self.action_history) > 50:
            self.action_history = self.action_history[-50:]


class K1_2_Agent:
    def __init__(self):
        self.state = AgentState()
        self.episode: int = 0
        self.survival_count: int = 0
        self.death_count: int = 0
        self.cumulative_reward: float = 0.0
        
        # K1.2: Survival thresholds
        self.energy_critical: float = 0.15
        self.energy_low: float = 0.35
        self.uncertainty_high: float = 0.4
        
        # Track exploration
        self.steps_since_explore: int = 0
        
        # Track starvation
        self.steps_without_resource_gain: int = 0
        self.last_resource_estimate: float = 0.5
    
    def select_action(self, obs: Dict[str, float]) -> AgentAction:
        self.episode += 1
        self.state.update_from_observation(obs, self.episode)
        
        E = self.state.energy_estimate
        U = self.state.uncertainty
        R = self.state.resource_estimate
        
        # Detect resource gain
        if R > self.last_resource_estimate + 0.05:
            self.steps_without_resource_gain = 0
        else:
            self.steps_without_resource_gain += 1
        self.last_resource_estimate = R
        
        # Update exploration tracking
        self.steps_since_explore += 1
        
        # 🔥 K1.2: STARVATION TRIGGERS
        if E < self.energy_critical:
            return AgentAction.EXPLORE
        
        if self.steps_without_resource_gain > 15:
            return AgentAction.EXPLORE
        
        if self.steps_since_explore > 10:
            self.steps_since_explore = 0
            return AgentAction.EXPLORE
        
        # Normal policy
        if U > self.uncertainty_high and E > self.energy_low:
            return AgentAction.EXPLORE
        elif E < self.energy_low:
            return AgentAction.CONSERVE
        else:
            return AgentAction.EXPLOIT
    
    def step(self, env) -> Dict[str, Any]:
        obs = env._generate_observation()  # Use correct method name
        action = self.select_action(obs)
        result = env.execute_action(action)
        
        # Track if this was exploration
        if action == AgentAction.EXPLORE:
            self.steps_since_explore = 0
        
        self.cumulative_reward += result["utility"]  # Use 'utility' key
        self.state.record_action(self.episode, action, result["utility"])
        
        if result["alive"]:
            self.survival_count += 1
        else:
            self.death_count += 1
        
        return {"alive": result["alive"], "observation": obs, "action": action, "episode": self.episode}
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "episode": self.episode,
            "survival_count": self.survival_count,
            "death_count": self.death_count,
            "survival_rate": self.survival_count / max(1, self.episode),
            "cumulative_reward": self.cumulative_reward,
            "current_state": {
                "energy": self.state.energy_estimate,
                "resources": self.state.resource_estimate,
                "risk": self.state.risk_estimate,
                "uncertainty": self.state.uncertainty
            }
        }
