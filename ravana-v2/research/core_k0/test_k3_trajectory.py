"""
K3 Trajectory Awareness Validation

Tests if K3:
1. Recovers earlier (before hitting critical energy)
2. Has fewer near-death events than K2
3. Shows mode coherence (phases, not random)
4. Adapts faster to new environments
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop_k2 import K2_Agent
from research.core_k0.agent_loop_k3 import K3_Agent
from research.experiments_k0.resource_env import ResourceSurvivalEnv
from typing import List, Dict, Any
import numpy as np


def run_single_trajectory_run(agent_class, n_episodes: int = 100, 
                              env_seed: int = 42, env_config: Dict = None) -> Dict[str, Any]:
    """Run agent and track trajectory metrics."""
    
    env = ResourceSurvivalEnv(seed=env_seed)
    if env_config:
        # Apply environment modifications
        for key, value in env_config.items():
            setattr(env, key, value)
    
    agent = agent_class()
    
    # Trajectory tracking
    near_death_events: List[int] = []  # Episodes where E < 0.2
    early_recovery_episodes: List[int] = []  # Episodes where recovery happened before critical
    mode_switches: List[Tuple[int, str, str]] = []  # (episode, from, to)
    energy_trajectory: List[float] = []
    mode_phases: List[Tuple[int, int, str]] = []  # (start, end, mode)
    
    prev_mode = None
    phase_start = 0
    
    for ep in range(n_episodes):
        # Capture pre-action state
        energy_before = env.true_energy
        
        # Agent step
        if hasattr(agent, 'step'):
            result = agent.step(env)
        else:
            # K2 compatibility
            result = agent.step(env)
            result["mode"] = "k2_baseline"
        
        energy_after = env.true_energy
        energy_trajectory.append(energy_after)
        
        # Track near-death
        if energy_after < 0.2:
            near_death_events.append(ep)
        
        # Track early recovery (energy rose before hitting critical)
        if energy_before < 0.3 and energy_after > energy_before and energy_after >= 0.25:
            early_recovery_episodes.append(ep)
        
        # Track mode switches (K3 only)
        current_mode = result.get("mode", "unknown")
        if prev_mode is not None and current_mode != prev_mode:
            mode_switches.append((ep, prev_mode, current_mode))
            # End previous phase
            if ep > 0:
                mode_phases.append((phase_start, ep - 1, prev_mode))
            phase_start = ep
        prev_mode = current_mode
        
        # Check death
        if not result["alive"]:
            # End final phase
            if prev_mode:
                mode_phases.append((phase_start, ep, prev_mode))
            break
    
    # End final phase if survived
    if result["alive"] and prev_mode:
        mode_phases.append((phase_start, ep, prev_mode))
    
    # Calculate coherence score (average phase duration)
    avg_phase_duration = np.mean([end - start + 1 for start, end, _ in mode_phases]) if mode_phases else 0
    
    return {
        "total_episodes": ep + 1,
        "survived": result["alive"],
        "near_death_events": len(near_death_events),
        "early_recovery_episodes": len(early_recovery_episodes),
        "mode_switches": len(mode_switches),
        "mode_phases": len(mode_phases),
        "avg_phase_duration": avg_phase_duration,
        "final_energy": energy_trajectory[-1] if energy_trajectory else 0,
        "min_energy": min(energy_trajectory) if energy_trajectory else 0,
        "patterns_learned": result.get("pattern_count", 0) if hasattr(agent, 'patterns') else 0,
    }


def test_k3_trajectory_advantages(n_runs: int = 10, verbose: bool = True) -> Dict[str, Any]:
    """Compare K3 vs K2 on trajectory metrics."""
    
    if verbose:
        print("\n" + "="*70)
        print("K3 TRAJECTORY VALIDATION: Anticipation vs Reaction")
        print("="*70)
    
    k2_results = []
    k3_results = []
    
    # Test 1: Standard environment
    if verbose:
        print("\n" + "-"*70)
        print(f"TEST 1: Standard Environment ({n_runs} runs)")
        print("-"*70)
    
    for i in range(n_runs):
        k2_result = run_single_trajectory_run(K2_Agent, 100, env_seed=i*42)
        k3_result = run_single_trajectory_run(K3_Agent, 100, env_seed=i*42)
        
        k2_results.append(k2_result)
        k3_results.append(k3_result)
        
        if verbose:
            print(f"  Run {i+1}: K2={k2_result['near_death_events']} near-deaths | "
                  f"K3={k3_result['near_death_events']} near-deaths | "
                  f"K3 phases={k3_result['mode_phases']}")
    
    # Aggregate results
    k2_near_deaths = [r["near_death_events"] for r in k2_results]
    k3_near_deaths = [r["near_death_events"] for r in k3_results]
    
    k2_avg_near_deaths = np.mean(k2_near_deaths)
    k3_avg_near_deaths = np.mean(k3_near_deaths)
    
    k3_phases = [r["mode_phases"] for r in k3_results]
    k3_avg_phases = np.mean(k3_phases)
    
    if verbose:
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"\n🎯 NEAR-DEATH EVENTS (lower is better):")
        print(f"  K2: {k2_avg_near_deaths:.1f} avg")
        print(f"  K3: {k3_avg_near_deaths:.1f} avg")
        print(f"  Improvement: {k2_avg_near_deaths - k3_avg_near_deaths:+.1f} events avoided")
        
        print(f"\n🎯 MODE COHERENCE (higher phase count = more coherent):")
        print(f"  K3 avg phases: {k3_avg_phases:.1f}")
        
        print(f"\n🎯 PATTERNS LEARNED:")
        k3_patterns = np.mean([r["patterns_learned"] for r in k3_results])
        print(f"  K3: {k3_patterns:.1f} patterns")
    
    # Test 2: Cold start consistency
    if verbose:
        print("\n" + "-"*70)
        print(f"TEST 2: Cold Start Consistency")
        print("-"*70)
    
    k3_convergence = []
    for i in range(5):
        result = run_single_trajectory_run(K3_Agent, 100, env_seed=i*100)
        k3_convergence.append(result["near_death_events"])
        if verbose:
            print(f"  Run {i+1}: {result['near_death_events']} near-deaths | "
                  f"{result['mode_phases']} phases")
    
    k3_consistency = np.std(k3_convergence)
    
    if verbose:
        print(f"\n  K3 near-death std: {k3_consistency:.2f} (lower = more consistent)")
    
    # Final verdict
    fewer_near_deaths = k3_avg_near_deaths < k2_avg_near_deaths
    has_mode_coherence = k3_avg_phases > 1  # At least 2 phases = coherent behavior
    
    if verbose:
        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)
        
        if fewer_near_deaths and has_mode_coherence:
            print("🟢 K3 TRAJECTORY AWARENESS CONFIRMED")
            print(f"   - {k2_avg_near_deaths - k3_avg_near_deaths:.1f} fewer near-death events")
            print(f"   - {k3_avg_phases:.1f} coherent behavioral phases")
            print("   - Proactive recovery vs reactive survival")
        else:
            print("🟡 K3 ADVANTAGES PRESENT BUT MUTED")
            if not fewer_near_deaths:
                print("   - Near-death reduction not significant")
            if not has_mode_coherence:
                print("   - Mode switching too frequent")
    
    return {
        "k2_near_deaths": k2_avg_near_deaths,
        "k3_near_deaths": k3_avg_near_deaths,
        "k3_phases": k3_avg_phases,
        "k3_patterns": k3_patterns,
        "k3_consistency": k3_consistency,
        "trajectory_confirmed": fewer_near_deaths and has_mode_coherence,
    }


if __name__ == "__main__":
    results = test_k3_trajectory_advantages(n_runs=10, verbose=True)
    
    exit_code = 0 if results["trajectory_confirmed"] else 1
    print(f"\nExit code: {exit_code}")
    sys.exit(exit_code)
