"""
K3 Regime Shift Test: Adaptation Speed

K2's value is proven on static environments.
K3's value should be: faster adaptation when environments CHANGE.

Test: Mid-run regime shift (stable → scarce → stable)
Measure: How quickly does each agent recover performance?
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop_k2 import K2_Agent
from research.core_k0.agent_loop_k3 import K3_Agent
from research.experiments_k0.resource_env import ResourceSurvivalEnv, HiddenRegime
from typing import List, Dict, Any, Tuple
import numpy as np


class RegimeShiftEnv(ResourceSurvivalEnv):
    """Environment that shifts regimes mid-run."""
    
    def __init__(self, seed: int = None, shift_points: List[int] = None):
        super().__init__(seed)
        self.shift_points = shift_points or [30, 60]  # EP30: scarce, EP60: stable
        self.regime_sequence = [HiddenRegime.STABLE, HiddenRegime.SCARCE, HiddenRegime.STABLE]
        self.current_regime_idx = 0
    
    def _update_regime(self):
        """Override to force regime shifts at specific points."""
        # Check if we should force a shift
        if self.current_regime_idx < len(self.shift_points):
            if self.episode >= self.shift_points[self.current_regime_idx]:
                # Force regime shift
                next_idx = self.current_regime_idx + 1
                if next_idx < len(self.regime_sequence):
                    self.current_regime = self.regime_sequence[next_idx]
                    self.regime_start_episode = self.episode
                    self.current_regime_idx += 1
                    
                    # Apply regime effects
                    if self.current_regime == HiddenRegime.SCARCE:
                        self.hidden_risk = 0.35  # Higher risk
                        self.true_resources = max(0.2, self.true_resources - 0.2)
                    elif self.current_regime == HiddenRegime.STABLE:
                        self.hidden_risk = 0.2
                        self.true_resources = min(0.9, self.true_resources + 0.1)
        else:
            # Normal regime updates after sequence ends
            super()._update_regime()


def run_regime_shift_test(agent_class, n_episodes: int = 100, 
                          env_seed: int = 42) -> Dict[str, Any]:
    """Run agent through regime shifts and track adaptation."""
    
    env = RegimeShiftEnv(seed=env_seed, shift_points=[25, 50, 75])
    agent = agent_class()
    
    # Tracking by phase
    phase_names = ["STABLE_1", "SCARCE", "STABLE_2"]
    phase_boundaries = [0, 25, 50, 100]
    
    phase_metrics = {name: {"survival": [], "energy": [], "near_deaths": 0} 
                     for name in phase_names}
    
    current_phase_idx = 0
    energy_trajectory = []
    
    for ep in range(n_episodes):
        # Track phase
        if ep >= phase_boundaries[current_phase_idx + 1]:
            current_phase_idx += 1
        current_phase = phase_names[min(current_phase_idx, len(phase_names) - 1)]
        
        # Step
        result = agent.step(env)
        
        energy = env.true_energy
        energy_trajectory.append(energy)
        
        # Record metrics for current phase
        phase_metrics[current_phase]["survival"].append(1 if result["alive"] else 0)
        phase_metrics[current_phase]["energy"].append(energy)
        if energy < 0.2:
            phase_metrics[current_phase]["near_deaths"] += 1
        
        if not result["alive"]:
            break
    
    # Calculate phase statistics
    for name in phase_names:
        data = phase_metrics[name]
        if data["survival"]:
            data["survival_rate"] = np.mean(data["survival"])
            data["avg_energy"] = np.mean(data["energy"])
        else:
            data["survival_rate"] = 0
            data["avg_energy"] = 0
    
    # Calculate adaptation metric: how well did it handle SCARCE phase?
    scarce_recovery = phase_metrics["SCARCE"]["survival_rate"]
    stable2_recovery = phase_metrics["STABLE_2"]["survival_rate"]
    
    # Adaptation score: survival in SCARCE + recovery in STABLE_2
    adaptation_score = (scarce_recovery + stable2_recovery) / 2
    
    return {
        "total_episodes": ep + 1,
        "survived": result["alive"],
        "phase_metrics": phase_metrics,
        "adaptation_score": adaptation_score,
        "scarce_survival": scarce_recovery,
        "final_recovery": stable2_recovery,
        "min_energy": min(energy_trajectory),
        "patterns_learned": result.get("pattern_count", 0) if hasattr(agent, 'patterns') else 0,
    }


def test_regime_shift_adaptation(n_runs: int = 5, verbose: bool = True) -> Dict[str, Any]:
    """Compare K2 vs K3 on regime shift adaptation."""
    
    if verbose:
        print("\n" + "="*70)
        print("K3 REGIME SHIFT TEST: Adaptation Speed")
        print("="*70)
        print("\nEnvironment: STABLE → SCARCE → STABLE (forced shifts at EP25, EP50)")
        print("Metric: How well does each agent handle the SCARCE phase?")
        print()
    
    k2_results = []
    k3_results = []
    
    for i in range(n_runs):
        k2_result = run_regime_shift_test(K2_Agent, env_seed=i*42)
        k3_result = run_regime_shift_test(K3_Agent, env_seed=i*42)
        
        k2_results.append(k2_result)
        k3_results.append(k3_result)
        
        if verbose:
            print(f"Run {i+1}:")
            print(f"  K2: adaptation={k2_result['adaptation_score']:.2f} | "
                  f"scarce={k2_result['scarce_survival']:.2f} | "
                  f"recovery={k2_result['final_recovery']:.2f}")
            print(f"  K3: adaptation={k3_result['adaptation_score']:.2f} | "
                  f"scarce={k3_result['scarce_survival']:.2f} | "
                  f"recovery={k3_result['final_recovery']:.2f} | "
                  f"patterns={k3_result['patterns_learned']:.0f}")
            print()
    
    # Aggregate
    k2_adaptation = np.mean([r["adaptation_score"] for r in k2_results])
    k3_adaptation = np.mean([r["adaptation_score"] for r in k3_results])
    
    k2_scarce = np.mean([r["scarce_survival"] for r in k2_results])
    k3_scarce = np.mean([r["scarce_survival"] for r in k3_results])
    
    k3_patterns = np.mean([r["patterns_learned"] for r in k3_results])
    
    if verbose:
        print("="*70)
        print("RESULTS")
        print("="*70)
        print(f"\n🎯 ADAPTATION SCORE (higher = better):")
        print(f"  K2: {k2_adaptation:.3f}")
        print(f"  K3: {k3_adaptation:.3f}")
        print(f"  Δ: {k3_adaptation - k2_adaptation:+.3f}")
        
        print(f"\n🎯 SCARCE PHASE SURVIVAL:")
        print(f"  K2: {k2_scarce:.3f}")
        print(f"  K3: {k3_scarce:.3f}")
        
        print(f"\n🎯 PATTERNS LEARNED (K3):")
        print(f"  {k3_patterns:.1f} patterns")
        
        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)
        
        if k3_adaptation > k2_adaptation:
            print("🟢 K3 ADAPTATION ADVANTAGE CONFIRMED")
            print(f"   +{(k3_adaptation - k2_adaptation) * 100:.1f}% better adaptation to regime shifts")
            print("   Trajectory awareness enables faster recovery")
        elif k3_adaptation >= k2_adaptation - 0.05:
            print("🟡 K3 MATCHES K2 (within margin)")
            print("   No degradation from added complexity")
        else:
            print("🔴 K3 UNDERPERFORMS")
            print("   Mode switching may be destabilizing")
    
    return {
        "k2_adaptation": k2_adaptation,
        "k3_adaptation": k3_adaptation,
        "k2_scarce": k2_scarce,
        "k3_scarce": k3_scarce,
        "k3_patterns": k3_patterns,
        "advantage": k3_adaptation > k2_adaptation,
    }


if __name__ == "__main__":
    results = test_regime_shift_adaptation(n_runs=5, verbose=True)
    
    exit_code = 0 if results["advantage"] else 1
    print(f"\nExit code: {exit_code}")
    sys.exit(exit_code)
