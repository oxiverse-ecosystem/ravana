"""
K3 Experiment 2: Coupling Threshold Test

Question: At what amplitude does K3's "nudge" start destabilizing K2?

Method: Gradually increase K3's influence weight (alpha) and measure:
- K2 action distribution shift
- Survival rate degradation
- Belief accuracy vs. action correctness

Hypothesis: There's a threshold below which K3 helps, above which it hurts.
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop_k2 import K2_Agent
from research.core_k0.agent_loop_k3_belief import K3_Belief_Agent
from research.experiments_k0.latent_regime_env import LatentRegimeEnv
import numpy as np


class K3_Coupled_Agent(K3_Belief_Agent):
    """
    K3 with tunable coupling strength to K2.
    
    alpha ∈ [0, 1]: How much K3 influences K2's decisions
        0 = K2 pure (baseline)
        1 = K3 full override (dangerous)
    """
    
    def __init__(self, alpha: float = 0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha  # Coupling strength
        self.coupling_history = []
    
    def _adjust_for_belief(self, prefs: Dict[str, float], context_key: str) -> Dict[str, float]:
        """
        SOFT NUDGE: Adjust K2's preferences by belief, scaled by alpha.
        
        Instead of overriding, we reweight K2's own learned preferences.
        """
        adjusted = prefs.copy()
        
        if abs(self.belief_good - 0.5) < 0.2:
            # Uncertain: no adjustment
            self.coupling_history.append({"episode": self.episode, "type": "no_adjust", "alpha": self.alpha})
            return adjusted
        
        # Belief-based nudge (scaled by alpha)
        if self.belief_good > 0.7:
            # Confident GOOD: boost exploit slightly
            boost = 0.1 * self.alpha
            adjusted["exploit"] = min(0.9, adjusted["exploit"] + boost)
            adjusted["explore"] = max(0.1, adjusted["explore"] - boost * 0.5)
            self.coupling_history.append({"episode": self.episode, "type": "boost_exploit", "alpha": self.alpha, "boost": boost})
            
        elif self.belief_good < 0.3:
            # Confident BAD: suppress exploit
            penalty = 0.1 * self.alpha
            adjusted["exploit"] = max(0.1, adjusted["exploit"] - penalty)
            adjusted["explore"] = min(0.9, adjusted["explore"] + penalty * 0.5)
            self.coupling_history.append({"episode": self.episode, "type": "suppress_exploit", "alpha": self.alpha, "penalty": penalty})
        
        return adjusted


def test_coupling_threshold(alpha: float, n_runs: int = 5, episodes: int = 50) -> Dict:
    """Test one coupling strength."""
    results = []
    
    for seed in range(n_runs):
        env = LatentRegimeEnv(seed=seed * 42)
        agent = K3_Coupled_Agent(alpha=alpha)
        
        survived = 0
        actions = []
        beliefs = []
        
        for ep in range(episodes):
            result = agent.step(env)
            if result["alive"]:
                survived += 1
            actions.append(result["action"])
            beliefs.append(agent.belief_good)
        
        # Analyze coupling effects
        nudges = [h for h in agent.coupling_history if h["type"] != "no_adjust"]
        
        results.append({
            "survival_rate": survived / episodes,
            "nudges_per_run": len(nudges),
            "avg_belief": np.mean(beliefs),
            "belief_variance": np.var(beliefs),
        })
    
    return {
        "alpha": alpha,
        "survival_mean": np.mean([r["survival_rate"] for r in results]),
        "survival_std": np.std([r["survival_rate"] for r in results]),
        "nudges_mean": np.mean([r["nudges_per_run"] for r in results]),
        "results": results,
    }


def run_coupling_sweep():
    """Sweep alpha from 0 to 1 and find the threshold."""
    
    print("=" * 70)
    print("K3 EXPERIMENT 2: Coupling Threshold Test")
    print("=" * 70)
    print("\nSweeping alpha (coupling strength) from 0.0 to 1.0...")
    print("Looking for threshold where K3 helps → hurts K2")
    print()
    
    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    all_results = []
    
    for alpha in alphas:
        print(f"Testing α = {alpha:.1f}...")
        result = test_coupling_threshold(alpha, n_runs=5, episodes=50)
        all_results.append(result)
        
        print(f"  Survival: {result['survival_mean']:.1%} ± {result['survival_std']:.1%}")
        print(f"  Nudges/run: {result['nudges_mean']:.1f}")
        print()
    
    # Find optimal alpha
    best = max(all_results, key=lambda x: x['survival_mean'])
    worst = min(all_results, key=lambda x: x['survival_mean'])
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nBest performance: α = {best['alpha']:.1f} → {best['survival_mean']:.1%} survival")
    print(f"Worst performance: α = {worst['alpha']:.1f} → {worst['survival_mean']:.1%} survival")
    
    # Detect threshold
    baseline = all_results[0]['survival_mean']  # alpha=0 (K2 pure)
    improved = [r for r in all_results if r['survival_mean'] > baseline + 0.05]
    degraded = [r for r in all_results if r['survival_mean'] < baseline - 0.05]
    
    if improved and not degraded:
        print(f"\n✅ K3 helps at all tested alphas")
        print(f"   Optimal: α = {best['alpha']:.1f}")
    elif degraded:
        print(f"\n⚠️  Threshold detected: K3 helps below α ≈ {degraded[0]['alpha']:.1f}")
        print(f"   Degradation starts at: α = {degraded[0]['alpha']:.1f}")
    else:
        print(f"\n🟡 No clear threshold — needs finer sweep")
    
    return {
        "alphas_tested": alphas,
        "results": all_results,
        "optimal_alpha": best['alpha'],
        "baseline": baseline,
    }


if __name__ == "__main__":
    results = run_coupling_sweep()
