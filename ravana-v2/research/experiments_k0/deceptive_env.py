"""
Deceptive States Environment — K2 Breaker #2

Core idea: Same context → different outcomes depending on hidden history.

Example:
Context (low_energy=0.2, low_uncertainty=0.1)

Sometimes: safe (recovering from minor dip)
Sometimes: trap (about to enter death spiral)

K2 sees same context → same action every time
Optimal policy: history-aware action selection

This requires trajectory awareness to distinguish "safe low" from "dangerous low".
"""

import numpy as np
from typing import Dict, Any, Optional, List
from enum import Enum
import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research.experiments_k0.resource_env import AgentAction, ResourceSurvivalEnv, HiddenRegime


class DeceptiveStateEnv(ResourceSurvivalEnv):
    """
    Environment with deceptive states.
    
    Design:
    - Energy ~0.3 can mean "recovering" (trending up) or "collapsing" (trending down)
    - Same observation (E=0.3, U=0.1) has opposite optimal actions
    - K2 lacks trend information → wrong decisions half the time
    
    The agent must learn:
    - (E=0.3, trend=down) → EXPLORE (aggressive recovery)
    - (E=0.3, trend=up) → EXPLOIT (maintain momentum)
    """
    
    def __init__(self, seed: int = None):
        super().__init__(seed)
        self.energy_trajectory: List[float] = []
        self._setup_deceptive_episodes()
        
    def _setup_deceptive_episodes(self):
        """Pre-plan which episodes will be deceptive."""
        # Episodes 20-30, 50-60, 80-90 will have deceptive states
        self.deceptive_ranges = [(20, 30), (50, 60), (80, 90)]
        
    def _is_deceptive_episode(self) -> bool:
        """Check if current episode should have deceptive state."""
        for start, end in self.deceptive_ranges:
            if start <= self.episode < end:
                return True
        return False
    
    def _get_true_trend(self) -> str:
        """Calculate actual energy trend."""
        if len(self.energy_trajectory) < 3:
            return "stable"
        recent = self.energy_trajectory[-3:]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        if slope > 0.02:
            return "up"
        elif slope < -0.02:
            return "down"
        return "stable"
    
    def execute_action(self, action: AgentAction) -> Dict[str, Any]:
        """Execute with deceptive outcomes."""
        self._update_regime()
        
        # Track true energy trajectory
        self.energy_trajectory.append(self.true_energy)
        
        # Action outcomes depend on hidden trend during deceptive episodes
        is_deceptive = self._is_deceptive_episode()
        true_trend = self._get_true_trend() if is_deceptive else "stable"
        
        if action == AgentAction.EXPLORE:
            if is_deceptive:
                # Deceptive: outcome depends on trend (which K2 can't see)
                if true_trend == "down":
                    # EXPLORE during down trend = good (catches recovery)
                    energy_gain = self.rng.normal(0.20, 0.05)
                    resource_gain = self.rng.normal(0.15, 0.05)
                    utility = 0.8
                else:
                    # EXPLORE during stable/up trend = wasteful
                    energy_gain = self.rng.normal(-0.10, 0.03)
                    resource_gain = self.rng.normal(-0.05, 0.02)
                    utility = -0.3
            else:
                # Normal exploration
                success = self.rng.random() > (self.hidden_risk * 1.5)
                if success:
                    energy_gain = self.rng.normal(0.15, 0.08)
                    resource_gain = self.rng.normal(0.2, 0.1)
                    utility = 1.0
                else:
                    energy_gain = self.rng.normal(-0.2, 0.05)
                    resource_gain = self.rng.normal(-0.05, 0.05)
                    utility = -0.5
                    
        elif action == AgentAction.EXPLOIT:
            # EXPLOIT ignores trend (reliable but not optimal)
            energy_gain = self.rng.normal(0.05, 0.03)
            resource_gain = self.rng.normal(0.1, 0.05)
            utility = 0.5
            
        elif action == AgentAction.CONSERVE:
            # CONSERVE too passive during deceptive down-trend
            if is_deceptive and true_trend == "down":
                # Being conservative during collapse = death
                energy_gain = self.rng.normal(-0.15, 0.03)
                resource_gain = self.rng.normal(0.02, 0.01)
                utility = -0.8
            else:
                energy_gain = self.rng.normal(0.02, 0.01)
                resource_gain = self.rng.normal(0.03, 0.02)
                utility = 0.3
        else:
            energy_gain = 0
            resource_gain = 0
            utility = 0
        
        # Update true state
        self.true_energy = np.clip(self.true_energy + energy_gain - 0.02, 0, 1)
        self.true_resources = np.clip(self.true_resources + resource_gain, 0, 1)
        
        observation = self._generate_observation()
        alive = self.true_energy > 0.1
        
        result = {
            'episode': self.episode,
            'action': action.value,
            'alive': alive,
            'utility': utility,
            'true_energy': self.true_energy,
            'true_resources': self.true_resources,
            'observation': observation,
            'regime': self.current_regime.value,
            'is_deceptive': is_deceptive,  # For analysis
            'true_trend': true_trend  # For analysis
        }
        
        self.history.append(result)
        self.episode += 1
        
        return result
