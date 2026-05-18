"""
K3 Experiment 3: Meta-Stability Layer

Question: Can we dynamically throttle K3 based on K2 health?

Design: Monitor K2's recent performance and dampen K3 when K2 destabilizes.

Mechanism:
- Track K2's action entropy (should be stable)
- Track survival rate in recent window
- If K2 shows stress → reduce alpha temporarily
- If K2 stable → allow full K3 coupling
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop_k2 import K2_Agent
from research.core_k0.agent_loop_k3_belief import K3_Belief_Agent, ActionOutcome
from research.experiments_k0.latent_regime_env import LatentRegimeEnv
import numpy as np
from typing import Dict, List
from collections import deque


# Re-define K3_Coupled_Agent here for standalone use
class K3_Coupled_Agent(K3_Belief_Agent):
    """K3 with tunable coupling strength (from Exp 2)."""
    
    def __init__(self, alpha: float = 0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.coupling_history = []


class MetaStabilityMonitor:
    """
    Monitors K2 health and dampens K3 coupling dynamically.
    
    Health indicators:
    - Action entropy (high = erratic, low = stable)
    - Survival rate in window
    - Energy trend volatility
    """
    
    def __init__(self, window: int = 10):
        self.window = window
        self.action_history: deque = deque(maxlen=window)
        self.outcome_history: deque = deque(maxlen=window)
        self.energy_history: deque = deque(maxlen=window)
    
    def update(self, action: str, survived: bool, energy: float):
        self.action_history.append(action)
        self.outcome_history.append(1 if survived else 0)
        self.energy_history.append(energy)
    
    def compute_health_score(self) -> float:
        """Returns health ∈ [0, 1]. 1 = healthy, 0 = critical."""
        if len(self.outcome_history) < self.window // 2:
            return 1.0  # Not enough data, assume healthy
        
        # Survival rate component
        survival_rate = sum(self.outcome_history) / len(self.outcome_history)
        
        # Action stability component (entropy of action distribution)
        if len(self.action_history) > 5:
            actions = list(self.action_history)
            action_counts = {a: actions.count(a) for a in set(actions)}
            # Lower entropy = more stable
            total = sum(action_counts.values())
            probs = [c/total for c in action_counts.values()]
            entropy = -sum(p * np.log(p) for p in probs if p > 0)
            # Normalize: max entropy for 3 actions is ln(3) ≈ 1.1
            stability = 1.0 - (entropy / 1.1)
        else:
            stability = 1.0
        
        # Energy trend volatility
        if len(self.energy_history) > 5:
            energy_volatility = np.std(list(self.energy_history))
            # Lower volatility = more stable
            energy_stability = 1.0 - min(1.0, energy_volatility / 0.2)
        else:
            energy_stability = 1.0
        
        # Combined health score
        health = 0.4 * survival_rate + 0.3 * stability + 0.3 * energy_stability
        return np.clip(health, 0.0, 1.0)
    
    def get_damping_factor(self) -> float:
        """
        Returns damping factor ∈ [0, 1].
        1 = full K3 coupling allowed
        0 = K3 fully dampened (K2 pure)
        """
        health = self.compute_health_score()
        
        # Damping curve:
        # health > 0.7 → damping = 1.0 (full K3)
        # health < 0.4 → damping = 0.0 (K3 off)
        # in between → linear interpolation
        
        if health > 0.7:
            return 1.0
        elif health < 0.4:
            return 0.0
        else:
            return (health - 0.4) / 0.3


class K3_MetaStable_Agent(K3_Belief_Agent):
    """
    K3 with dynamic coupling controlled by meta-stability layer.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = MetaStabilityMonitor(window=10)
        self.coupling_history = []
    
    def step(self, env) -> Dict:
        """Execute one step with dynamic coupling."""
        # Get K2's decision
        obs = env._generate_observation()
        
        # Update K3's belief state
        if hasattr(env, 'last_explore_signal'):
            self._update_belief()
        
        # Check K2 health and get damping
        health = self.monitor.compute_health_score()
        damping = self.monitor.get_damping_factor()
        effective_alpha = self.alpha * damping
        
        # Get base K2 action (what would K2 do?)
        base_action = super().select_action(obs)  # This includes K2's logic
        
        # If damping is high, potentially adjust
        if effective_alpha > 0.1 and abs(self.belief_good - 0.5) > 0.2:
            # K3 wants to influence
            E = self.state.energy_estimate
            context_key = self.state.get_context_key(E, self.state.uncertainty, self.state.get_energy_trend(5))
            
            # Get K2's preferences
            prefs = self._get_prefs_as_dict(context_key)
            
            # Apply belief adjustment with effective alpha
            adjusted = prefs.copy()
            
            if self.belief_good > 0.7:
                boost = 0.1 * effective_alpha
                adjusted["exploit"] = min(0.9, adjusted["exploit"] + boost)
            elif self.belief_good < 0.3:
                penalty = 0.1 * effective_alpha
                adjusted["exploit"] = max(0.1, adjusted["exploit"] - penalty)
            
            # Pick action from adjusted preferences
            from research.experiments_k0.resource_env import AgentAction
            action = AgentAction(max(adjusted, key=adjusted.get))
        else:
            # K2 pure (damping active)
            action = base_action
        
        # Execute
        result = env.execute_action(action)
        
        # Track
        self.episode += 1
        if action == AgentAction.EXPLORE:
            self.steps_since_explore = 0
        
        # Record for belief update
        energy_before = env.true_energy
        outcome = ActionOutcome(
            episode=self.episode,
            context={},
            action=action,
            energy_before=energy_before,
            energy_after=result['true_energy'],
            delta_energy=result['true_energy'] - energy_before,
            survived=result['alive'],
            exploration_success=(result['true_energy'] > energy_before + 0.05)
        )
        self._record_outcome(env, action, result)
        
        # Update meta-stability monitor
        self.monitor.update(action.value, result['alive'], result['true_energy'])
        
        # Log coupling state
        self.coupling_history.append({
            "episode": self.episode,
            "health": health,
            "damping": damping,
            "effective_alpha": effective_alpha,
            "base_action": base_action.value,
            "final_action": action.value,
        })
        
        return {
            "alive": result['alive'],
            "observation": obs,
            "action": action,
            "episode": self.episode,
            "health": health,
            "damping": damping,
        }


def test_metastability_vs_fixed(n_runs: int = 5, episodes: int = 50):
    """Compare meta-stable K3 vs fixed-alpha K3."""
    
    print("=" * 70)
    print("K3 EXPERIMENT 3: Meta-Stability Layer")
    print("=" * 70)
    print("\nComparing:")
    print("  - Fixed α=0.5 (no meta-stability)")
    print("  - Meta-stable (dynamic damping based on K2 health)")
    print()
    
    results_fixed = []
    results_meta = []
    
    for seed in range(n_runs):
        # Fixed alpha
        env_fixed = LatentRegimeEnv(seed=seed * 100)
        agent_fixed = K3_Coupled_Agent(alpha=0.5)
        
        survived_fixed = 0
        for ep in range(episodes):
            result = agent_fixed.step(env_fixed)
            if result["alive"]:
                survived_fixed += 1
        
        results_fixed.append(survived_fixed / episodes)
        
        # Meta-stable
        env_meta = LatentRegimeEnv(seed=seed * 100)
        agent_meta = K3_MetaStable_Agent(alpha=0.5)
        
        survived_meta = 0
        for ep in range(episodes):
            result = agent_meta.step(env_meta)
            if result["alive"]:
                survived_meta += 1
        
        results_meta.append(survived_meta / episodes)
    
    print("RESULTS:")
    print(f"  Fixed α=0.5: {np.mean(results_fixed):.1%} ± {np.std(results_fixed):.1%}")
    print(f"  Meta-stable: {np.mean(results_meta):.1%} ± {np.std(results_meta):.1%}")
    print()
    
    if np.mean(results_meta) > np.mean(results_fixed):
        print("✅ Meta-stability helps")
    elif np.mean(results_meta) < np.mean(results_fixed):
        print("🔴 Meta-stability hurts (unexpected)")
    else:
        print("🟡 No significant difference")
    
    return {
        "fixed": results_fixed,
        "meta": results_meta,
    }


if __name__ == "__main__":
    results = test_metastability_vs_fixed(n_runs=5, episodes=50)
