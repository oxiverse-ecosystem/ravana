#!/usr/bin/env python3
"""
RAVANA K1.2 TEST — Starvation Trigger + Exploration Floor
Critical hypothesis: Anti-conservatism enables survival.
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop_k1_2 import K1_2_Agent, AgentAction
from research.experiments_k0.resource_env import ResourceSurvivalEnv, HiddenRegime


def run_k1_2_test():
    print("=" * 70)
    print("🧪 K1.2 TEST: Starvation Trigger + Exploration Floor")
    print("=" * 70)
    print("\n🔧 CONFIGURATION")
    print("   Energy critical threshold: < 0.15")
    print("   Resource starvation trigger: > 15 steps without gain")
    print("   Exploration floor: Every 10 steps must explore")
    print("   Uncertainty high threshold: > 0.4")
    
    # Create agent and environment
    agent = K1_2_Agent()
    env = ResourceSurvivalEnv(seed=42)
    
    print(f"\n🌍 ENVIRONMENT")
    print(f"   Initial regime: {env.current_regime.value}")
    print(f"   Observation noise: {env.base_noise:.2f}")
    
    # Track key metrics
    energy_trajectory = []
    action_distribution = {action: 0 for action in AgentAction}
    explore_triggers = []
    starvation_attempts = []
    
    total_episodes = 500
    print(f"\n🚀 RUNNING K1.2 LOOP: {total_episodes} episodes")
    print("-" * 70)
    
    for ep in range(total_episodes):
        # Pre-step state
        pre_energy = agent.state.energy_estimate
        
        result = agent.step(env)
        
        # Track energy trajectory
        energy_trajectory.append(agent.state.energy_estimate)
        
        # Track action distribution
        action_distribution[result['action']] += 1
        
        # Track exploration triggers
        if result['action'] == AgentAction.EXPLORE:
            trigger_reason = "normal"
            if pre_energy < agent.energy_critical:
                trigger_reason = "🔥 CRITICAL_ENERGY"
                explore_triggers.append((ep, "critical_energy"))
            elif agent.steps_without_resource_gain > 15:
                trigger_reason = "🔥 RESOURCE_STARVATION"
                explore_triggers.append((ep, "resource_starvation"))
            elif agent.steps_since_explore == 0:  # Just reset
                trigger_reason = "🔥 EXPLORATION_FLOOR"
                explore_triggers.append((ep, "exploration_floor"))
        
        # Track starvation recovery attempts
        if agent.steps_without_resource_gain > 10 and result['action'] == AgentAction.EXPLORE:
            starvation_attempts.append((ep, agent.state.energy_estimate))
        
        # Show death
        if not result['alive']:
            print(f"\n💀 EP{ep:04d}: AGENT DIED")
            print(f"   Final energy: {agent.state.energy_estimate:.3f}")
            print(f"   Steps without resource gain: {agent.steps_without_resource_gain}")
            print(f"   Last action: {result['action'].value}")
            break
        
        # Periodic status
        if ep % 100 == 0 and ep > 0:
            status = agent.get_status()
            recent_explores = sum(1 for a in agent.state.action_history[-20:] if a[1] == AgentAction.EXPLORE)
            print(f"\n📊 EP{ep:04d} Status")
            print(f"   Survival rate: {status['survival_rate']:.1%}")
            print(f"   Energy: {status['current_state']['energy']:.2f} | Resources: {status['current_state']['resources']:.2f}")
            print(f"   Recent explores (last 20): {recent_explores}")
            print(f"   Hidden regime: {env.current_regime.value}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("K1.2 TEST COMPLETE")
    print("=" * 70)
    
    status = agent.get_status()
    print(f"\n📈 SURVIVAL METRICS")
    print(f"   Total episodes: {status['episode']}")
    print(f"   Survival count: {status['survival_count']}")
    print(f"   Death count: {status['death_count']}")
    print(f"   Final survival rate: {status['survival_rate']:.1%}")
    print(f"   Cumulative reward: {status['cumulative_reward']:.1f}")
    
    print(f"\n🧠 FINAL STATE")
    print(f"   Energy estimate: {status['current_state']['energy']:.3f}")
    print(f"   Resource estimate: {status['current_state']['resources']:.3f}")
    print(f"   Risk estimate: {status['current_state']['risk']:.3f}")
    print(f"   Uncertainty: {status['current_state']['uncertainty']:.3f}")
    
    print(f"\n🎯 ACTION DISTRIBUTION")
    for action, count in sorted(action_distribution.items(), key=lambda x: -x[1]):
        pct = count / max(1, status['episode']) * 100
        print(f"   {action.value}: {count} ({pct:.1f}%)")
    
    print(f"\n🔥 EXPLORATION TRIGGERS")
    print(f"   Total starvation-triggered explores: {len(explore_triggers)}")
    trigger_breakdown = {}
    for ep, reason in explore_triggers:
        trigger_breakdown[reason] = trigger_breakdown.get(reason, 0) + 1
    for reason, count in sorted(trigger_breakdown.items(), key=lambda x: -x[1]):
        print(f"   - {reason}: {count}")
    
    print(f"\n🔄 STARVATION RECOVERY ATTEMPTS")
    print(f"   Total recovery explores (>10 steps w/o gain): {len(starvation_attempts)}")
    if starvation_attempts:
        successful_recoveries = sum(1 for ep, energy in starvation_attempts if energy > 0.3)
        print(f"   Successful recoveries (energy > 0.3): {successful_recoveries}")
    
    # Energy trajectory analysis
    if len(energy_trajectory) >= 50:
        recent = energy_trajectory[-50:]
        min_recent = min(recent)
        max_recent = max(recent)
        print(f"\n📊 ENERGY TRAJECTORY (last 50 steps)")
        print(f"   Range: {min_recent:.3f} → {max_recent:.3f} (spread: {max_recent - min_recent:.3f})")
        if max_recent - min_recent > 0.3:
            print("   Pattern: 🔥 SURVIVAL LOOPS (decline → spike → decline)")
        else:
            print("   Pattern: ⚠️ STAGNATION (low variance)")
    
    # K1.2 Verdict
    print(f"\n🏆 K1.2 VERDICT")
    if status['survival_rate'] > 0.9:
        verdict = "🟢 EXCELLENT: Starvation triggers enable survival"
    elif status['survival_rate'] > 0.7:
        verdict = "🟢 GOOD: Anti-conservatism works, policy can refine"
    elif status['survival_rate'] > 0.5:
        verdict = "🟡 PARTIAL: Direction correct, needs tuning"
    else:
        verdict = "🔴 NEEDS WORK: Triggers insufficient or wrong thresholds"
    print(f"   {verdict}")
    
    # Compare to K1
    print(f"\n📊 COMPARISON TO K1")
    print(f"   K1 survival rate: ~5% (died at EP25)")
    print(f"   K1.2 survival rate: {status['survival_rate']:.1%}")
    improvement = (status['survival_rate'] - 0.05) / 0.05 * 100
    print(f"   Improvement: {improvement:+.0f}%")
    
    return {
        'survival_rate': status['survival_rate'],
        'episodes': status['episode'],
        'exploration_triggers': len(explore_triggers),
        'trigger_breakdown': trigger_breakdown,
        'verdict': verdict
    }


if __name__ == "__main__":
    results = run_k1_2_test()
    print("\n" + "=" * 70)
    print("Test complete. Results captured.")
    print("=" * 70)
