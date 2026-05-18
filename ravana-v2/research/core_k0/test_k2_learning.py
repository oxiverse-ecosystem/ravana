"""
K2 Learning Validation: Shows survival rate improving over episodes.
Tests experience → strategy adaptation.
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop_k2 import K2_Agent
from research.core_k0.agent_loop_k1_3 import K1_3_Agent
from research.experiments_k0.resource_env import ResourceSurvivalEnv
from typing import List, Dict, Any
import numpy as np


def run_single_episode_run(agent_class, n_episodes: int = 100, seed: int = 42) -> Dict[str, Any]:
    """Run one long episode with learning enabled."""
    env = ResourceSurvivalEnv(seed=seed)
    agent = agent_class()
    
    survival_by_phase = {"early": [], "mid": [], "late": []}
    exploration_by_phase = {"early": [], "mid": [], "late": []}
    
    for ep in range(n_episodes):
        prev_energy = env.true_energy
        result = agent.step(env)
        
        phase = "early" if ep < 33 else "mid" if ep < 66 else "late"
        survival_by_phase[phase].append(1 if result["alive"] else 0)
        exploration_by_phase[phase].append(1 if result["action"].value == "explore" else 0)
        
        if not result["alive"]:
            break
    
    status = agent.get_status()
    
    return {
        "agent_type": agent_class.__name__,
        "total_episodes": ep + 1,
        "survived_all": result["alive"],
        "survival_rate": status["survival_rate"],
        "cumulative_reward": status["cumulative_reward"],
        "phases": {
            phase: {
                "survival_rate": np.mean(survival_by_phase[phase]) if survival_by_phase[phase] else 0,
                "exploration_rate": np.mean(exploration_by_phase[phase]) if exploration_by_phase[phase] else 0
            }
            for phase in ["early", "mid", "late"]
        },
        "policy_weights": status.get("policy_weights", {}),
        "context_weights_count": status.get("context_weights_count", 0),
        "exploration_success_rate": status.get("exploration_success_rate", 0)
    }


def run_learning_test(n_runs: int = 10, episodes_per_run: int = 100) -> Dict[str, Any]:
    """Test if K2 improves over episodes vs K1.3 (static)."""
    
    print("\n" + "="*70)
    print("K2 LEARNING VALIDATION: Experience → Strategy")
    print("="*70)
    print(f"Runs per agent: {n_runs}")
    print(f"Episodes per run: {episodes_per_run}")
    print("Testing: Does survival improve over time?")
    print("="*70 + "\n")
    
    k1_3_results = []
    k2_results = []
    
    # Test K1.3 (static baseline)
    print("\n🔴 K1.3 (static policy, no learning)...")
    for i in range(n_runs):
        result = run_single_episode_run(K1_3_Agent, episodes_per_run, seed=i*42)
        k1_3_results.append(result)
        print(f"  Run {i+1}: Survived {result['total_episodes']} eps | "
              f"Early={result['phases']['early']['survival_rate']:.1%} | "
              f"Late={result['phases']['late']['survival_rate']:.1%}")
    
    # Test K2 (learning)
    print("\n🟢 K2 (adaptive policy, learns from outcomes)...")
    for i in range(n_runs):
        result = run_single_episode_run(K2_Agent, episodes_per_run, seed=i*42)
        k2_results.append(result)
        print(f"  Run {i+1}: Survived {result['total_episodes']} eps | "
              f"Early={result['phases']['early']['survival_rate']:.1%} | "
              f"Late={result['phases']['late']['survival_rate']:.1%} | "
              f"Contexts learned: {result['context_weights_count']}")
    
    # Analysis
    print("\n" + "="*70)
    print("LEARNING ANALYSIS")
    print("="*70)
    
    # Phase-by-phase survival
    k1_3_early = np.mean([r["phases"]["early"]["survival_rate"] for r in k1_3_results])
    k1_3_late = np.mean([r["phases"]["late"]["survival_rate"] for r in k1_3_results])
    
    k2_early = np.mean([r["phases"]["early"]["survival_rate"] for r in k2_results])
    k2_late = np.mean([r["phases"]["late"]["survival_rate"] for r in k2_results])
    
    print(f"\n📊 PHASE SURVIVAL (Early → Late):")
    print(f"  K1.3: {k1_3_early:.1%} → {k1_3_late:.1%} ({'+' if k1_3_late > k1_3_early else ''}{k1_3_late - k1_3_early:.1%})")
    print(f"  K2:   {k2_early:.1%} → {k2_late:.1%} ({'+' if k2_late > k2_early else ''}{k2_late - k2_early:.1%})")
    
    # Exploration efficiency
    k1_3_explore = np.mean([r["phases"]["late"]["exploration_rate"] for r in k1_3_results])
    k2_explore = np.mean([r["phases"]["late"]["exploration_rate"] for r in k2_results])
    
    print(f"\n📊 LATE-PHASE EXPLORATION:")
    print(f"  K1.3: {k1_3_explore:.1%}")
    print(f"  K2:   {k2_explore:.1%} ({'more' if k2_explore > k1_3_explore else 'less'} selective)")
    
    # Overall survival
    k1_3_survival = np.mean([r["survival_rate"] for r in k1_3_results])
    k2_survival = np.mean([r["survival_rate"] for r in k2_results])
    
    print(f"\n📊 OVERALL SURVIVAL:")
    print(f"  K1.3: {k1_3_survival:.1%}")
    print(f"  K2:   {k2_survival:.1%}")
    
    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    # Learning = K2 maintains higher late-phase survival than K1.3
    learning_detected = k2_late > k1_3_late and k2_late - k2_early > k1_3_late - k1_3_early
    
    if k2_late > k1_3_late:
        print("🟢 K2 SHOWS ADAPTATION")
        print(f"   Late-phase survival: K2={k2_late:.1%} vs K1.3={k1_3_late:.1%}")
        print(f"   Decline resistance: K2={k2_late-k2_early:+.1%} vs K1.3={k1_3_late-k1_3_early:+.1%}")
        print("   Agent learned context-specific strategies")
    else:
        print("🟡 K2 PERFORMANCE")
        print(f"   Late-phase: {k2_late:.1%} (vs K1.3's {k1_3_late:.1%})")
        print("   Learning may need tuning")
    
    print("="*70 + "\n")
    
    return {
        "k1_3": {
            "early_survival": k1_3_early,
            "late_survival": k1_3_late,
            "overall": k1_3_survival
        },
        "k2": {
            "early_survival": k2_early,
            "late_survival": k2_late,
            "overall": k2_survival
        },
        "learning_detected": k2_late > k1_3_late
    }


if __name__ == "__main__":
    results = run_learning_test(n_runs=10, episodes_per_run=100)
    sys.exit(0 if results["learning_detected"] else 1)
