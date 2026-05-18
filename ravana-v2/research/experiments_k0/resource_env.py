"""
RAVANA K0: Resource Survival Micro-World
Not trading. Not games. Pure survival with delayed feedback.
"""
import numpy as np
from typing import Dict, Any, Optional
from enum import Enum
import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research.core_k0.agent_loop import AgentAction


class HiddenRegime(Enum):
    """Hidden environment regimes RAVANA must infer."""
    STABLE = "stable"        # Predictable rewards
    VOLATILE = "volatile"    # High variance, changing rules
    SCARCE = "scarce"        # Resources deplete rapidly
    ABUNDANT = "abundant"    # Resources plentiful


class ResourceSurvivalEnv:
    """
    K0 Environment: Simple, brutal, revealing.
    
    Design principles:
    - Partial observability (agent doesn't see true state)
    - Delayed rewards (consequences come later)
    - Shifting rules (regime changes hidden)
    - Resource constraints (scarcity forces decisions)
    """
    
    def __init__(self, seed: int = None):
        self.rng = np.random.RandomState(seed)
        
        # True state (hidden from agent)
        self.true_energy: float = 0.6
        self.true_resources: float = 0.5
        self.hidden_risk: float = 0.2
        
        # Regime (hidden, shifts periodically)
        self.current_regime: HiddenRegime = HiddenRegime.STABLE
        self.regime_start_episode: int = 0
        self.regime_duration: int = 100
        
        # Observation quality (variable noise)
        self.base_noise: float = 0.1
        self.observation_noise_walk: float = 0.0
        
        # Episode tracking
        self.episode: int = 0
        self.history: list = []
        
    def _update_regime(self):
        """Hidden regime shift — agent only sees consequences."""
        if self.episode - self.regime_start_episode > self.regime_duration:
            # Shift regime
            self.current_regime = self.rng.choice(list(HiddenRegime))
            self.regime_start_episode = self.episode
            
            # Adjust hidden parameters based on regime
            if self.current_regime == HiddenRegime.VOLATILE:
                self.hidden_risk = 0.4
                self.base_noise = 0.2
            elif self.current_regime == HiddenRegime.SCARCE:
                self.hidden_risk = 0.25
                self.true_resources = max(0.2, self.true_resources - 0.1)
            elif self.current_regime == HiddenRegime.ABUNDANT:
                self.hidden_risk = 0.15
                self.true_resources = min(0.9, self.true_resources + 0.1)
            else:  # STABLE
                self.hidden_risk = 0.2
                self.base_noise = 0.1
    
    def _generate_observation(self) -> Dict[str, float]:
        """
        Noisy, partial observation.
        Agent never sees true state directly.
        """
        # Variable observation quality
        self.observation_noise_walk += self.rng.normal(0, 0.02)
        self.observation_noise_walk = np.clip(self.observation_noise_walk, -0.15, 0.15)
        
        noise_level = self.base_noise + abs(self.observation_noise_walk)
        
        # Noisy observations
        energy_obs = np.clip(
            self.true_energy + self.rng.normal(0, noise_level), 0, 1
        )
        resource_obs = np.clip(
            self.true_resources + self.rng.normal(0, noise_level * 0.8), 0, 1
        )
        
        return {
            'energy_obs': energy_obs,
            'resource_obs': resource_obs,
            'noise': noise_level,
            'observation_quality': 1.0 - noise_level  # Higher noise = lower quality
        }
    
    def execute_action(self, action: AgentAction) -> Dict[str, Any]:
        """
        Execute action, update true state, return noisy observation.
        This is where the agent's choices have consequences.
        """
        self._update_regime()
        
        # Action outcomes (deterministic base + noise + risk)
        if action == AgentAction.EXPLORE:
            # High variance outcome
            success = self.rng.random() > (self.hidden_risk * 1.5)
            if success:
                energy_gain = self.rng.normal(0.15, 0.08)
                resource_gain = self.rng.normal(0.2, 0.1)
            else:
                energy_gain = self.rng.normal(-0.2, 0.05)  # Could lose energy
                resource_gain = self.rng.normal(-0.05, 0.05)
            
            utility = 1.0 if success else -0.5
            
        elif action == AgentAction.EXPLOIT:
            # Medium, reliable outcome
            energy_gain = self.rng.normal(0.05, 0.03)
            resource_gain = self.rng.normal(0.1, 0.05)
            utility = 0.5
            
        elif action == AgentAction.CONSERVE:
            # Low gain, very safe
            energy_gain = self.rng.normal(0.02, 0.01)
            resource_gain = self.rng.normal(0.03, 0.02)
            utility = 0.3  # Survival bonus
            
        else:
            energy_gain = 0
            resource_gain = 0
            utility = 0
        
        # Update true state
        self.true_energy = np.clip(self.true_energy + energy_gain - 0.02, 0, 1)  # -0.02 = metabolism
        self.true_resources = np.clip(self.true_resources + resource_gain, 0, 1)
        
        # Generate observation
        observation = self._generate_observation()
        
        # Check survival
        alive = self.true_energy > 0.1  # Death threshold
        
        # Record
        result = {
            'episode': self.episode,
            'action': action.value,
            'alive': alive,
            'utility': utility,
            'true_energy': self.true_energy,
            'true_resources': self.true_resources,
            'observation': observation,
            'regime': self.current_regime.value  # For analysis only, agent never sees this
        }
        
        self.history.append(result)
        self.episode += 1
        
        return result
    
    def get_hidden_truth(self) -> Dict[str, Any]:
        """For post-analysis only. Agent never sees this during operation."""
        return {
            'true_energy': self.true_energy,
            'true_resources': self.true_resources,
            'hidden_risk': self.hidden_risk,
            'current_regime': self.current_regime.value,
            'episode': self.episode
        }
