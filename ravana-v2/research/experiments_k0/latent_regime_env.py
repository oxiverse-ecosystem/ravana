"""
Latent Regime World — K2 Breaker (Surgical)

Core idea: Same observation, opposite optimal action.
Hidden regime ∈ {GOOD, BAD} determines dynamics.
Agent must infer regime from history of exploration signals.

Design: Minimal, brutal, clean.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum, auto
import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research.experiments_k0.resource_env import ResourceSurvivalEnv, AgentAction, HiddenRegime


class LatentRegime(Enum):
    """Hidden regime for the latent environment."""
    GOOD = "GOOD"
    BAD = "BAD"


class LatentRegimeEnv(ResourceSurvivalEnv):
    """
    Latent regime environment where observations are ambiguous.
    
    Hidden state: regime ∈ {GOOD, BAD}
    Observation: (energy, uncertainty) — same in both regimes!
    
    Dynamics:
        GOOD regime:
            - exploit: +energy (safe, reliable)
            - explore: small cost, reveals signal
            - wait: slight decay
            
        BAD regime:
            - exploit: -energy (DANGEROUS — hurts you!)
            - explore: small cost, reveals signal  
            - wait: heavy decay
            
    Signal mechanism (from exploration):
        GOOD regime: signal is 80% "positive" (accurate)
        BAD regime: signal is 80% "negative" (accurate)
        
    Regime switching: occasional, unobserved
    
    Why this breaks K2:
        K2 sees: (low_energy, low_uncertainty)
        In GOOD → exploit is correct
        In BAD → exploit is DEATH
        
        K2 learns average reward per action per context.
        Average is a weighted mix of GOOD and BAD outcomes.
        Result: hesitation, suboptimal, eventual failure.
        
    What wins:
        Infer regime from history of signals.
        Then pick action conditioned on belief(regime).
    """
    
    def __init__(self, seed: int = None, signal_accuracy: float = 0.8):
        super().__init__(seed)
        
        # Hidden state (agent never sees this directly)
        self.regime_history = []  # Track regime over time for analysis
        self.signal_accuracy = signal_accuracy
        
        # More frequent regime switches (every 20-40 episodes)
        self.regime_duration_min = 20
        self.regime_duration_max = 40
        
        # Force initial regime
        self._force_next_regime()
        
    def _force_next_regime(self):
        """Pick new regime and duration."""
        self.current_regime = np.random.choice([LatentRegime.GOOD, LatentRegime.BAD])
        self.regime_start_episode = self.episode
        self.regime_duration = np.random.randint(
            self.regime_duration_min, 
            self.regime_duration_max
        )
        self.regime_history.append((self.episode, self.current_regime.value))
        
    def _update_regime(self):
        """Override: more frequent, sharper regime changes."""
        if self.episode - self.regime_start_episode >= self.regime_duration:
            # Regime flip
            old_regime = self.current_regime
            self._force_next_regime()
            
    def execute_action(self, action: AgentAction) -> Dict[str, Any]:
        """
        Execute with REGIME-DEPENDENT dynamics.
        Same observation, opposite outcomes!
        """
        self._update_regime()
        
        # Capture pre-action state
        energy_before = self.true_energy
        regime_now = self.current_regime
        
        # === REGIME-SPECIFIC DYNAMICS ===
        
        if regime_now == LatentRegime.GOOD:
            # GOOD: Exploitation is safe, waiting is cheap
            if action == AgentAction.EXPLORE:
                # Small cost, gives signal
                energy_gain = np.random.normal(-0.05, 0.02)
                resource_gain = np.random.normal(0.05, 0.03)
                utility = 0.2
                signal = self._generate_signal(is_good_regime=True)
                
            elif action == AgentAction.EXPLOIT:
                # Safe, reliable gain
                energy_gain = np.random.normal(0.08, 0.03)
                resource_gain = np.random.normal(0.05, 0.02)
                utility = 0.6
                signal = None
                
            else:  # CONSERVE/WAIT
                # Cheap to wait
                energy_gain = np.random.normal(-0.01, 0.01)
                resource_gain = np.random.normal(0.02, 0.01)
                utility = 0.1
                signal = None
                
        else:  # BAD regime
            # BAD: Exploitation is DANGEROUS, waiting is deadly
            if action == AgentAction.EXPLORE:
                # Small cost, gives signal (same as good, but more important!)
                energy_gain = np.random.normal(-0.05, 0.02)
                resource_gain = np.random.normal(0.05, 0.03)
                utility = 0.3  # Higher utility because info is valuable
                signal = self._generate_signal(is_good_regime=False)
                
            elif action == AgentAction.EXPLOIT:
                # DANGEROUS: hurts you!
                energy_gain = np.random.normal(-0.12, 0.04)
                resource_gain = np.random.normal(-0.05, 0.02)
                utility = -0.8  # Strongly negative
                signal = None
                
            else:  # CONSERVE/WAIT
                # DEADLY to wait
                energy_gain = np.random.normal(-0.06, 0.02)
                resource_gain = np.random.normal(-0.02, 0.01)
                utility = -0.3
                signal = None
        
        # Apply metabolism
        self.true_energy = np.clip(self.true_energy + energy_gain - 0.02, 0, 1)
        self.true_resources = np.clip(self.true_resources + resource_gain, 0, 1)
        
        # Generate observation (SAME regardless of regime!)
        observation = self._generate_observation()
        
        # Check survival
        alive = self.true_energy > 0.1
        
        result = {
            'episode': self.episode,
            'action': action.value,
            'alive': alive,
            'utility': utility,
            'true_energy': self.true_energy,
            'true_resources': self.true_resources,
            'observation': observation,
            'regime': regime_now.value,  # For analysis only!
            'signal': signal,  # Only non-None for exploration
            'energy_delta': self.true_energy - energy_before,
        }
        
        self.history.append(result)
        self.episode += 1
        
        return result
    
    def _generate_signal(self, is_good_regime: bool) -> Optional[str]:
        """
        Generate noisy signal about current regime.
        
        Returns: "positive" or "negative" with 80% accuracy
        """
        if np.random.random() < self.signal_accuracy:
            # Accurate signal
            return "positive" if is_good_regime else "negative"
        else:
            # Noise
            return "negative" if is_good_regime else "positive"
    
    def get_regime_history(self) -> list:
        """For analysis: when did regimes change?"""
        return self.regime_history


class K2_Baseline:
    """Simple K2-style agent for comparison (no belief state)."""
    
    def __init__(self):
        self.state = {"energy": 0.5, "uncertainty": 0.3}
        self.context_rewards = {}  # (e_bucket, u_bucket) -> action -> avg_reward
        
    def select_action(self, obs: Dict[str, float]) -> AgentAction:
        # Simple bucketing
        e = obs.get('energy_obs', 0.5)
        u = obs.get('uncertainty', 0.3)
        
        e_bucket = int(e * 3)  # 0, 1, 2
        u_bucket = int(u * 2)  # 0, 1
        context = (e_bucket, u_bucket)
        
        # Get learned preferences (or default)
        prefs = self.context_rewards.get(context, {})
        
        if not prefs:
            # Default: explore to gather data
            return AgentAction.EXPLORE
        
        # Pick action with highest average reward
        best_action = max(prefs, key=prefs.get)
        return AgentAction(best_action)
    
    def learn(self, context: Tuple, action: str, reward: float):
        """Simple averaging update."""
        if context not in self.context_rewards:
            self.context_rewards[context] = {"explore": 0.5, "exploit": 0.5, "conserve": 0.5}
        
        # Exponential moving average
        old = self.context_rewards[context][action]
        self.context_rewards[context][action] = 0.9 * old + 0.1 * reward
