"""
Delayed Consequences Environment — K2 Breaker #1

Core idea: Actions have delayed rewards. K2's immediate utility learning fails.

Example:
- EXPLORE now → small cost now, BIG payoff 5-10 steps later
- EXPLOIT now → reliable small gain now, but misses delayed opportunity

K2 will learn: "explore = bad" (because immediate cost)
Optimal policy: "explore = good" (because delayed payoff)

This is where trajectory awareness becomes necessary.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research.experiments_k0.resource_env import AgentAction, ResourceSurvivalEnv, HiddenRegime


class DelayedRewardEnv(ResourceSurvivalEnv):
    """
    Environment where exploration has delayed rewards.
    
    Design:
    - When agent explores, it plants a 'seed' that pays off 5-10 steps later
    - Immediate effect: small energy loss (K2 sees negative)
    - Delayed effect: large energy gain (K2 misses this)
    - K2 will under-explore, leading to resource starvation
    """
    
    def __init__(self, seed: int = None, delay_range: Tuple[int, int] = (5, 10)):
        super().__init__(seed)
        self.delay_range = delay_range
        self.pending_rewards: List[Dict[str, Any]] = []  # (episode, energy_gain, resource_gain)
        
    def execute_action(self, action: AgentAction) -> Dict[str, Any]:
        """Execute with delayed rewards."""
        self._update_regime()
        
        # Process pending rewards that mature this episode
        matured_rewards = [r for r in self.pending_rewards if r['episode'] <= self.episode]
        self.pending_rewards = [r for r in self.pending_rewards if r['episode'] > self.episode]
        
        # Apply matured rewards
        matured_energy = sum(r['energy'] for r in matured_rewards)
        matured_resources = sum(r['resources'] for r in matured_rewards)
        
        # Action outcomes (immediate)
        if action == AgentAction.EXPLORE:
            # Immediate: small cost
            immediate_energy = -0.05
            immediate_resource = 0.0
            
            # Delayed: plant a seed that pays off later
            delay = self.rng.randint(self.delay_range[0], self.delay_range[1])
            self.pending_rewards.append({
                'episode': self.episode + delay,
                'energy': self.rng.normal(0.25, 0.05),  # Big delayed payoff
                'resources': self.rng.normal(0.3, 0.1),
                'source': 'exploration_seed'
            })
            
            utility = -0.2  # K2 sees this as bad!
            
        elif action == AgentAction.EXPLOIT:
            # Immediate: reliable small gain
            immediate_energy = self.rng.normal(0.05, 0.02)
            immediate_resource = self.rng.normal(0.1, 0.03)
            utility = 0.4  # K2 sees this as good
            
        elif action == AgentAction.CONSERVE:
            immediate_energy = self.rng.normal(0.03, 0.01)
            immediate_resource = self.rng.normal(0.05, 0.02)
            utility = 0.25
            
        else:
            immediate_energy = 0
            immediate_resource = 0
            utility = 0
        
        # Update true state (immediate + matured rewards)
        total_energy = immediate_energy + matured_energy
        total_resource = immediate_resource + matured_resources
        
        self.true_energy = np.clip(self.true_energy + total_energy - 0.02, 0, 1)
        self.true_resources = np.clip(self.true_resources + total_resource, 0, 1)
        
        # Generate observation (agent doesn't see pending rewards)
        observation = self._generate_observation()
        
        # Check survival
        alive = self.true_energy > 0.1
        
        # Record
        result = {
            'episode': self.episode,
            'action': action.value,
            'alive': alive,
            'utility': utility,
            'true_energy': self.true_energy,
            'true_resources': self.true_resources,
            'observation': observation,
            'regime': self.current_regime.value,
            'matured_rewards': len(matured_rewards),  # For analysis
            'pending_seeds': len(self.pending_rewards)  # For analysis
        }
        
        self.history.append(result)
        self.episode += 1
        
        return result
