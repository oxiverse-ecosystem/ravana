"""
Latent Regime Test: Proves belief-based reasoning is required

Environment: Same observations, opposite optimal actions.
K2: Fails (averages rewards, converges to hesitation)
K3-Belief: Succeeds (infers regime, acts optimally)
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop_k2 import K2_Agent
from research.core_k0.agent_loop_k3_belief import K3_Belief_Agent
from research.experiments_k0.latent_regime_env import LatentRegimeEnv
import numpy as np


def run_single_run(agent_class, env_seed: int, episodes: int = 100, verbose: bool = False):
    """Run one agent on LatentRegimeEnv."""
    env = LatentRegimeEnv(seed=env_seed)
    agent = agent_class()
    
    alive_count = 0
    exploit_in_good = 0  # Correct: exploit when GOOD
    exploit_in_bad = 0   # WRONG: exploit when BAD (should explore)
    explore_count = 0
    signals_received = 0
    belief_accuracy = []  # How well does K3 track true regime?
    
    for ep in range(episodes):
        result = agent.step(env)
        
        if result['alive']:
            alive_count += 1
        
        action = result['action']
        true_regime = env.history[-1]['regime'] if env.history else 'unknown'
        
        # Track action appropriateness
        if action.value == 'exploit':
            if true_regime == 'GOOD':
                exploit_in_good += 1
            elif true_regime == 'BAD':
                exploit_in_bad += 1
        elif action.value == 'explore':
            explore_count += 1
            if result.get('signal'):
                signals_received += 1
        
        # For K3: track belief accuracy
        if hasattr(agent, 'belief_good') and true_regime != 'unknown':
            true_is_good = (true_regime == 'GOOD')
            belief_correct = (agent.belief_good > 0.5) == true_is_good
            belief_accuracy.append(1 if belief_correct else 0)
    
    # Calculate regime-aware correctness
    total_exploits = exploit_in_good + exploit_in_bad
    exploit_correct_rate = exploit_in_good / max(1, total_exploits)
    
    return {
        'survival_rate': alive_count / episodes,
        'exploit_correct_rate': exploit_correct_rate,
        'exploit_in_bad': exploit_in_bad,
        'explore_count': explore_count,
        'signals_received': signals_received,
        'belief_accuracy': np.mean(belief_accuracy) if belief_accuracy else 0,
        'final_energy': env.true_energy,
    }


def test_latent_regime(n_runs: int = 10, episodes_per_run: int = 100, verbose: bool = True):
    """Compare K2 vs K3-Belief on LatentRegimeEnv."""
    
    if verbose:
        print("\n" + "="*70)
        print("LATENT REGIME TEST: Does K2 fail where K3-Belief succeeds?")
        print("="*70)
        print("\nEnvironment: Same observation, opposite optimal actions")
        print("  GOOD regime: exploit → +energy (safe)")
        print("  BAD regime:  exploit → -energy (DEATH)")
        print("  K2 sees: average reward of exploit in context")
        print("  K3 sees: inferred regime from signal history")
        print("\nHypothesis:")
        print("  K2: Hesitates, makes mistakes, lower survival")
        print("  K3: Correctly infers, acts optimally, higher survival")
    
    k2_results = []
    k3_results = []
    
    for i in range(n_runs):
        k2_result = run_single_run(K2_Agent, env_seed=i*42, episodes=episodes_per_run)
        k3_result = run_single_run(K3_Belief_Agent, env_seed=i*42, episodes=episodes_per_run)
        
        k2_results.append(k2_result)
        k3_results.append(k3_result)
        
        if verbose:
            print(f"\nRun {i+1}:")
            print(f"  K2: {k2_result['survival_rate']:.1%} | exploits_correct={k2_result['exploit_correct_rate']:.1%} | bad_exploits={k2_result['exploit_in_bad']}")
            print(f"  K3: {k3_result['survival_rate']:.1%} | exploits_correct={k3_result['exploit_correct_rate']:.1%} | belief_acc={k3_result['belief_accuracy']:.1%}")
    
    # Aggregate results
    k2_survival = np.mean([r['survival_rate'] for r in k2_results])
    k3_survival = np.mean([r['survival_rate'] for r in k3_results])
    
    k2_correct = np.mean([r['exploit_correct_rate'] for r in k2_results])
    k3_correct = np.mean([r['exploit_correct_rate'] for r in k3_results])
    
    k2_bad_exploits = np.mean([r['exploit_in_bad'] for r in k2_results])
    k3_bad_exploits = np.mean([r['exploit_in_bad'] for r in k3_results])
    
    k3_belief_acc = np.mean([r['belief_accuracy'] for r in k3_results])
    
    if verbose:
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"\n📊 SURVIVAL:")
        print(f"  K2: {k2_survival:.1%}")
        print(f"  K3: {k3_survival:.1%}")
        print(f"  Advantage: {k3_survival - k2_survival:+.1%}")
        
        print(f"\n📊 EXPLOIT CORRECTNESS (critical metric):")
        print(f"  K2: {k2_correct:.1%} (random due to averaging)")
        print(f"  K3: {k3_correct:.1%} (belief-based selection)")
        
        print(f"\n📊 DEADLY MISTAKES (exploit in BAD regime):")
        print(f"  K2: {k2_bad_exploits:.1f} per run")
        print(f"  K3: {k3_bad_exploits:.1f} per run")
        
        print(f"\n📊 K3 BELIEF ACCURACY:")
        print(f"  Avg: {k3_belief_acc:.1%}")
        
        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)
        
        k3_advantage = k3_survival > k2_survival and k3_correct > k2_correct
        
        if k3_advantage:
            print("🟢 K3-BELIEF ADVANTAGE CONFIRMED")
            print("   K2 fails on latent-state environments")
            print("   K3 succeeds with belief-based reasoning")
        elif k3_survival >= k2_survival:
            print("🟡 K3 matches K2 (may need harder environment)")
        else:
            print("🔴 K3 underperforms (belief mechanism needs tuning)")
        print("="*70)
    
    return {
        'k2': {
            'survival': k2_survival,
            'correct_rate': k2_correct,
            'bad_exploits': k2_bad_exploits,
        },
        'k3': {
            'survival': k3_survival,
            'correct_rate': k3_correct,
            'bad_exploits': k3_bad_exploits,
            'belief_accuracy': k3_belief_acc,
        },
        'k3_advantage': k3_survival > k2_survival and k3_correct > k2_correct,
    }


if __name__ == "__main__":
    results = test_latent_regime(n_runs=10, episodes_per_run=100, verbose=True)
    
    exit_code = 0 if results['k3_advantage'] else 1
    print(f"\nExit code: {exit_code}")
    sys.exit(exit_code)
