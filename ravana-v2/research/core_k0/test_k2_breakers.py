"""
K2 Breaker Test: Prove K2 fails on harder environments

This validates that:
1. Delayed consequences break K2's immediate utility learning
2. Deceptive states break K2's context-only decision making
3. These environments genuinely need trajectory awareness (K3 concept)
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop_k2 import K2_Agent
from research.experiments_k0.resource_env import ResourceSurvivalEnv
from research.experiments_k0.delayed_env import DelayedRewardEnv
from research.experiments_k0.deceptive_env import DeceptiveStateEnv
from typing import Dict, Any, List
import numpy as np


def test_environment(env_class, env_name: str, n_runs: int = 10, 
                     n_episodes: int = 100) -> Dict[str, Any]:
    """Test K2 on a specific environment."""
    
    results = []
    
    for i in range(n_runs):
        env = env_class(seed=i*42)
        agent = K2_Agent()
        
        survived = True
        death_episode = n_episodes
        
        for ep in range(n_episodes):
            result = agent.step(env)
            if not result["alive"]:
                survived = False
                death_episode = ep
                break
        
        results.append({
            "survived": survived,
            "death_episode": death_episode,
            "final_energy": env.true_energy if survived else 0,
        })
    
    # Aggregate
    survival_rate = sum(1 for r in results if r["survived"]) / n_runs
    avg_death_episode = np.mean([r["death_episode"] for r in results if not r["survived"]]) if any(not r["survived"] for r in results) else n_episodes
    
    return {
        "env_name": env_name,
        "survival_rate": survival_rate,
        "avg_death_episode": avg_death_episode,
        "raw_results": results
    }


def run_k2_breaker_test(verbose: bool = True) -> Dict[str, Any]:
    """Run K2 on all environments."""
    
    if verbose:
        print("\n" + "="*70)
        print("K2 BREAKER TEST: Does K2 fail on harder environments?")
        print("="*70)
        print("\nBaseline: K2 achieves 100% on standard ResourceSurvivalEnv")
        print("Question: Does K2 fail on DelayedReward and DeceptiveState?")
        print()
    
    # Test 1: Standard (should pass)
    standard_result = test_environment(ResourceSurvivalEnv, "Standard", n_runs=10)
    
    # Test 2: Delayed consequences (expect failure)
    delayed_result = test_environment(DelayedRewardEnv, "Delayed", n_runs=10)
    
    # Test 3: Deceptive states (expect failure)
    deceptive_result = test_environment(DeceptiveStateEnv, "Deceptive", n_runs=10)
    
    if verbose:
        print("RESULTS:")
        print(f"  Standard:    {standard_result['survival_rate']*100:.0f}% survival")
        print(f"  Delayed:     {delayed_result['survival_rate']*100:.0f}% survival  {'✗ BROKEN' if delayed_result['survival_rate'] < 0.8 else '✓ OK'}")
        print(f"  Deceptive:   {deceptive_result['survival_rate']*100:.0f}% survival  {'✗ BROKEN' if deceptive_result['survival_rate'] < 0.8 else '✓ OK'}")
        
        if delayed_result['survival_rate'] < 1.0:
            print(f"\n  Delayed avg death: EP{delayed_result['avg_death_episode']:.0f}")
        if deceptive_result['survival_rate'] < 1.0:
            print(f"  Deceptive avg death: EP{deceptive_result['avg_death_episode']:.0f}")
        
        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)
        
        standard_intact = standard_result['survival_rate'] >= 0.95
        delayed_broken = delayed_result['survival_rate'] < 0.8
        deceptive_broken = deceptive_result['survival_rate'] < 0.8
        
        if standard_intact and (delayed_broken or deceptive_broken):
            print("🟢 K2 BREAKERS CONFIRMED")
            print("   Harder environments expose K2's limitations:")
            if delayed_broken:
                print("   - Delayed consequences: K2 under-explores (sees only immediate cost)")
            if deceptive_broken:
                print("   - Deceptive states: K2 can't distinguish safe/dangerous same-context")
            print("   → These environments genuinely need trajectory awareness")
        elif not standard_intact:
            print("🔴 K2 REGRESSED on standard env (unexpected)")
        else:
            print("🟡 K2 handles all environments (breakers not strong enough)")
    
    return {
        "standard": standard_result,
        "delayed": delayed_result,
        "deceptive": deceptive_result,
        "breakers_confirmed": standard_result['survival_rate'] >= 0.95 and 
                             (delayed_result['survival_rate'] < 0.8 or 
                              deceptive_result['survival_rate'] < .8)
    }


if __name__ == "__main__":
    results = run_k2_breaker_test(verbose=True)
    
    exit_code = 0 if results["breakers_confirmed"] else 1
    print(f"\nExit code: {exit_code}")
    sys.exit(exit_code)
