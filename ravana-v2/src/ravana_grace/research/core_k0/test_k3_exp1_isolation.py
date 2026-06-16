"""
K3 Experiment 1: Isolation Test

Question: Does K3's belief mechanism work correctly when K2 is frozen?

Design: Freeze K2's learning (no weight updates) and test whether K3's
belief layer alone can adapt its regime inference from exploration signals.

Method:
- Run K3_Belief_Agent on LatentRegimeEnv (regime-switching environment)
- Freeze K2's context weights (disable learning)
- Measure: belief accuracy, survival rate, action distribution shifts
- Compare: K3 with frozen K2 vs K3 with learning K2 (exp2 baseline)

Hypothesis: K3's belief layer should track regime changes even without
K2 learning, because it operates on exploration signals independently.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ..research.core_k0.agent_loop_k3_belief import K3_Belief_Agent
from ..research.experiments_k0.latent_regime_env import LatentRegimeEnv
import numpy as np
from typing import Dict, List, Any


class K3_Isolated_Agent(K3_Belief_Agent):
    """
    K3 with K2 learning frozen.

    Belief layer still works (updates from signals),
    but K2's context weights are locked after initialization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._learning_disabled = True

    def _learn_from_outcome(self, outcome):
        """Override: no-op when isolated (K2 frozen)."""
        if self._learning_disabled:
            return  # Skip K2 learning
        super()._learn_from_outcome(outcome)


def test_belief_isolation(n_runs: int = 5, episodes: int = 100) -> Dict[str, Any]:
    """
    Test K3's belief mechanism in isolation (K2 frozen).

    Measures whether belief_good tracks regime changes without K2 learning.
    """
    print("=" * 70)
    print("K3 EXPERIMENT 1: Belief Isolation Test")
    print("=" * 70)
    print("\nQuestion: Does K3's belief mechanism work when K2 is frozen?")
    print("Design: Disable K2 learning, run belief layer alone")
    print()

    results = []

    for seed in range(n_runs):
        env = LatentRegimeEnv(seed=seed * 42)
        agent = K3_Isolated_Agent(signal_accuracy=0.8)

        beliefs = []
        actions = []
        signals = []
        survived = 0

        for ep in range(episodes):
            result = agent.step(env)
            beliefs.append(result['belief_good'])
            actions.append(result['action'])
            if result.get('signal'):
                signals.append(result['signal'])
            if result['alive']:
                survived += 1

        # Analyze belief tracking
        belief_mean = np.mean(beliefs)
        belief_std = np.std(beliefs)
        belief_range = max(beliefs) - min(beliefs)

        # Action distribution
        from collections import Counter
        action_counts = Counter(a.value for a in actions)
        total_actions = len(actions)
        explore_frac = action_counts.get('explore', 0) / total_actions
        exploit_frac = action_counts.get('exploit', 0) / total_actions
        conserve_frac = action_counts.get('conserve', 0) / total_actions

        results.append({
            "seed": seed,
            "survival_rate": survived / episodes,
            "belief_mean": belief_mean,
            "belief_std": belief_std,
            "belief_range": belief_range,
            "explore_frac": explore_frac,
            "exploit_frac": exploit_frac,
            "conserve_frac": conserve_frac,
            "n_signals": len(signals),
        })

    # Aggregate
    survival_rates = [r["survival_rate"] for r in results]
    belief_means = [r["belief_mean"] for r in results]
    belief_ranges = [r["belief_range"] for r in results]
    explore_fracs = [r["explore_frac"] for r in results]

    print("RESULTS:")
    print(f"  Survival rate:     {np.mean(survival_rates):.1%} +/- {np.std(survival_rates):.1%}")
    print(f"  Belief mean:       {np.mean(belief_means):.3f} +/- {np.std(belief_means):.3f}")
    print(f"  Belief range:      {np.mean(belief_ranges):.3f} +/- {np.std(belief_ranges):.3f}")
    print(f"  Explore fraction:  {np.mean(explore_fracs):.1%} +/- {np.std(explore_fracs):.1%}")
    print(f"  Signals received:  {np.mean([r['n_signals'] for r in results]):.1f}")
    print()

    # Verdict
    belief_is_dynamic = np.mean(belief_ranges) > 0.15
    survival_ok = np.mean(survival_rates) > 0.3

    if belief_is_dynamic and survival_ok:
        print("[PASS] Belief mechanism works in isolation")
        print("   K3 tracks regime changes even with K2 frozen")
        print(f"   Belief varies by {np.mean(belief_ranges):.3f} (threshold: 0.15)")
    elif not belief_is_dynamic:
        print("[FAIL] Belief mechanism is static (range < 0.15)")
        print("   K3 needs K2 learning to generate useful signals")
    else:
        print("[WARN] Belief works but survival is poor")
        print("   Belief tracks regime but K2 frozen = can't adapt actions")

    print()
    return {
        "per_run": results,
        "survival_mean": float(np.mean(survival_rates)),
        "belief_range_mean": float(np.mean(belief_ranges)),
        "isolation_works": belief_is_dynamic and survival_ok,
    }


if __name__ == "__main__":
    results = test_belief_isolation(n_runs=5, episodes=100)

    exit_code = 0 if results["isolation_works"] else 1
    print(f"Exit code: {exit_code}")
    sys.exit(exit_code)
