"""
K1.3 Validation Test: 20-episode multi-run comparison
Tests if K1.3 fixes the EP24 deterministic death from K1.2.
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop_k1_2 import K1_2_Agent
from research.core_k0.agent_loop_k1_3 import K1_3_Agent
from research.experiments_k0.resource_env import ResourceSurvivalEnv, AgentAction
from typing import List, Dict, Any
import numpy as np


def run_single_agent(agent_class, env_seed: int, verbose: bool = False, debug: bool = False) -> Dict[str, Any]:
    """Run one agent through 50 episodes or until death."""
    env = ResourceSurvivalEnv(seed=env_seed)
    agent = agent_class()
    
    episode_logs = []
    death_episode = None
    explore_results = []  # Track success/failure
    
    for ep in range(50):  # Max 50 episodes
        prev_energy = env.true_energy
        result = agent.step(env)
        
        # Track exploration outcomes
        if result["action"] == AgentAction.EXPLORE:
            # Check if energy increased (success) or decreased (failure)
            energy_delta = env.true_energy - prev_energy + 0.02  # Adjust for metabolism
            explore_results.append({
                "episode": result["episode"],
                "energy_delta": energy_delta,
                "success": energy_delta > 0
            })
        
        log_entry = {
            "episode": result["episode"],
            "action": result["action"].value if hasattr(result["action"], 'value') else str(result["action"]),
            "alive": result["alive"],
            "energy": result["observation"]["energy_obs"],
            "true_energy": env.true_energy,
            "mode": result.get("mode", "N/A"),
            "energy_trend": result.get("energy_trend", 0.0),
        }
        episode_logs.append(log_entry)
        
        if debug and ep < 30:
            mode_str = f"[{result.get('mode', 'N/A')}]" if hasattr(agent, '_get_exploration_mode') else ""
            print(f"    EP{ep:2d}: E={env.true_energy:.3f} | Action={result['action'].value:8s} {mode_str}")
        
        if not result["alive"]:
            death_episode = result["episode"]
            if verbose:
                print(f"  💀 DIED at EP{death_episode}")
            break
    
    status = agent.get_status()
    
    # Calculate exploration success rate
    explore_success_rate = 0.0
    if explore_results:
        explore_success_rate = sum(1 for r in explore_results if r["success"]) / len(explore_results)
    
    return {
        "agent_type": agent_class.__name__,
        "seed": env_seed,
        "survived": death_episode is None,
        "death_episode": death_episode,
        "final_episode": status["episode"],
        "survival_rate": status["survival_rate"],
        "cumulative_reward": status["cumulative_reward"],
        "logs": episode_logs,
        "exploration_count": sum(1 for log in episode_logs if log["action"] == "explore"),
        "conserve_count": sum(1 for log in episode_logs if log["action"] == "conserve"),
        "exploit_count": sum(1 for log in episode_logs if log["action"] == "exploit"),
        "explore_success_rate": explore_success_rate,
        "explore_details": explore_results,
    }


def run_multi_test(n_runs: int = 20, verbose: bool = True) -> Dict[str, Any]:
    """Run n_runs of both K1.2 and K1.3, compare results."""
    
    if verbose:
        print("\n" + "="*70)
        print("K1.3 VALIDATION TEST: 20-Run Multi-Episode Analysis")
        print("="*70)
        print(f"Testing K1.2 (baseline) vs K1.3 (fix)")
        print(f"Runs per agent: {n_runs}")
        print(f"Max episodes per run: 50")
        print("="*70 + "\n")
    
    k1_2_results = []
    k1_3_results = []
    
    # Run K1.2
    if verbose:
        print("\n🔴 RUNNING K1.2 (baseline with EP24 bug)...")
        print("-" * 50)
    
    for i in range(min(3, n_runs)):  # Debug first 3 runs
        print(f"\n  🔍 DEBUG Run {i+1} (K1.2):")
        result = run_single_agent(K1_2_Agent, env_seed=i*42, verbose=False, debug=True)
        print(f"    Exploration success rate: {result['explore_success_rate']:.1%}")
        k1_2_results.append(result)
    
    # Continue remaining without debug
    for i in range(3, n_runs):
        result = run_single_agent(K1_2_Agent, env_seed=i*42, verbose=verbose)
        k1_2_results.append(result)
        if verbose:
            status = "✓ SURVIVED" if result["survived"] else f"💀 DIED EP{result['death_episode']}"
            print(f"  Run {i+1:2d}: {status} | Reward: {result['cumulative_reward']:+.2f}")
    
    # Run K1.3
    if verbose:
        print("\n🟢 RUNNING K1.3 (context-aware fix)...")
        print("-" * 50)
    
    for i in range(n_runs):
        result = run_single_agent(K1_3_Agent, env_seed=i*42, verbose=verbose)
        k1_3_results.append(result)
        if verbose:
            status = "✓ SURVIVED" if result["survived"] else f"💀 DIED EP{result['death_episode']}"
            print(f"  Run {i+1:2d}: {status} | Reward: {result['cumulative_reward']:+.2f}")
    
    # Analyze results
    k1_2_survivals = sum(1 for r in k1_2_results if r["survived"])
    k1_3_survivals = sum(1 for r in k1_3_results if r["survived"])
    
    k1_2_deaths = [r["death_episode"] for r in k1_2_results if r["death_episode"] is not None]
    k1_3_deaths = [r["death_episode"] for r in k1_3_results if r["death_episode"] is not None]
    
    k1_2_exploration = np.mean([r["exploration_count"] for r in k1_2_results])
    k1_3_exploration = np.mean([r["exploration_count"] for r in k1_3_results])
    
    k1_2_reward = np.mean([r["cumulative_reward"] for r in k1_2_results])
    k1_3_reward = np.mean([r["cumulative_reward"] for r in k1_3_results])
    
    # Check for deterministic EP24 death
    k1_2_ep24_deaths = sum(1 for d in k1_2_deaths if d == 24)
    k1_3_ep24_deaths = sum(1 for d in k1_3_deaths if d == 24)
    
    if verbose:
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        print(f"\n📊 SURVIVAL RATES:")
        print(f"  K1.2: {k1_2_survivals}/{n_runs} ({k1_2_survivals/n_runs*100:.1f}%)")
        print(f"  K1.3: {k1_3_survivals}/{n_runs} ({k1_3_survivals/n_runs*100:.1f}%)")
        
        print(f"\n📊 DEATH EPISODE ANALYSIS:")
        if k1_2_deaths:
            print(f"  K1.2 deaths: {k1_2_deaths}")
            print(f"  K1.2 EP24 deaths: {k1_2_ep24_deaths} ({'DETERMINISTIC!' if k1_2_ep24_deaths > n_runs*0.5 else 'scattered'})")
        if k1_3_deaths:
            print(f"  K1.3 deaths: {k1_3_deaths}")
            print(f"  K1.3 EP24 deaths: {k1_3_ep24_deaths} ({'DETERMINISTIC!' if k1_3_ep24_deaths > n_runs*0.5 else 'scattered if any'})")
        
        print(f"\n📊 EXPLORATION FREQUENCY:")
        print(f"  K1.2: {k1_2_exploration:.1f} explores/run")
        print(f"  K1.3: {k1_3_exploration:.1f} explores/run")
        print(f"  Reduction: {(1 - k1_3_exploration/k1_2_exploration)*100:.1f}%")
        
        print(f"\n📊 REWARD:")
        print(f"  K1.2: {k1_2_reward:+.2f} avg")
        print(f"  K1.3: {k1_3_reward:+.2f} avg")
        
        # Verdict
        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)
        
        if k1_2_ep24_deaths > n_runs * 0.5:
            print("🔴 K1.2: CONFIRMED EP24 DETERMINISTIC DEATH BUG")
        
        if k1_3_survivals > k1_2_survivals:
            print("🟢 K1.3: EP24 DEATH ELIMINATED")
            print(f"   Survival improved: {k1_2_survivals} → {k1_3_survivals}")
        
        if k1_3_exploration < k1_2_exploration * 0.8:
            print("🟢 K1.3: Exploration frequency reduced (quality over quantity)")
        
        print("="*70 + "\n")
    
    return {
        "k1_2": {
            "survival_rate": k1_2_survivals / n_runs,
            "survivals": k1_2_survivals,
            "deaths": n_runs - k1_2_survivals,
            "death_episodes": k1_2_deaths,
            "ep24_deaths": k1_2_ep24_deaths,
            "avg_exploration": k1_2_exploration,
            "avg_reward": k1_2_reward,
            "raw_results": k1_2_results,
        },
        "k1_3": {
            "survival_rate": k1_3_survivals / n_runs,
            "survivals": k1_3_survivals,
            "deaths": n_runs - k1_3_survivals,
            "death_episodes": k1_3_deaths,
            "ep24_deaths": k1_3_ep24_deaths,
            "avg_exploration": k1_3_exploration,
            "avg_reward": k1_3_reward,
            "raw_results": k1_3_results,
        },
        "fix_verified": k1_3_survivals > k1_2_survivals and k1_3_exploration < k1_2_exploration
    }


if __name__ == "__main__":
    results = run_multi_test(n_runs=20, verbose=True)
    
    # Exit code: 0 if fix verified (survival improved + exploration reduced), 1 otherwise
    exit_code = 0 if results["fix_verified"] else 1
    print(f"\nExit code: {exit_code} (0 = success, 1 = failure)")
    sys.exit(exit_code)
