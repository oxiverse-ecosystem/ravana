#!/usr/bin/env python3
"""
RAVANA K1.2 MULTI-EPISODE ANALYTICS
Validate: Is K1.2 robust... or just lucky?

Captures:
- Survival distribution
- Trigger effectiveness curve
- Timing sensitivity (EP24 insight)
- Over-exploration risk
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop_k1_2 import K1_2_Agent, AgentAction
from research.experiments_k0.resource_env import ResourceSurvivalEnv
import numpy as np
from typing import List, Dict, Any


def run_single_life(seed: int) -> Dict[str, Any]:
    """Run one complete life. Return full trace."""
    env = ResourceSurvivalEnv(seed=seed)
    agent = K1_2_Agent()
    
    trace = {
        'episodes': [],
        'triggers_fired': [],
        'recovery_events': [],
        'exploration_payoff': [],
        'energy_trajectory': [],
        'death_episode': None,
        'death_cause': None,
    }
    
    prev_energy = 0.6
    
    for episode in range(500):
        # Run agent step
        result = agent.step(env)
        
        # Extract true state from environment
        true_energy = env.true_energy
        true_resources = env.true_resources
        
        # Track energy trajectory
        trace['energy_trajectory'].append({
            'episode': episode,
            'true_energy': true_energy,
            'estimated_energy': agent.state.energy_estimate,
            'action': result['action'].value,
        })
        
        # Detect triggers fired this step
        if result['action'] == AgentAction.EXPLORE:
            # Determine which trigger fired
            E = agent.state.energy_estimate
            
            if E < 0.15:
                trigger = 'critical_energy'
            elif agent.steps_without_resource_gain > 15:
                trigger = 'starvation_recovery'
            elif agent.steps_since_explore <= 1:  # Just reset
                trigger = 'exploration_floor'
            else:
                trigger = 'uncertainty'
            
            trace['triggers_fired'].append({
                'episode': episode,
                'trigger': trigger,
                'energy_before': E,
                'energy_after': true_energy,
            })
            
            # Track exploration payoff
            energy_delta = true_energy - prev_energy
            trace['exploration_payoff'].append({
                'episode': episode,
                'energy_delta': energy_delta,
                'success': energy_delta > 0,
            })
        
        # Detect recovery events
        if true_energy > prev_energy + 0.05:
            trace['recovery_events'].append({
                'episode': episode,
                'energy_gain': true_energy - prev_energy,
                'triggered_by': trace['triggers_fired'][-1] if trace['triggers_fired'] else None,
            })
        
        prev_energy = true_energy
        
        # Check death
        if not result['alive']:
            trace['death_episode'] = episode
            
            # Determine death cause
            if true_energy < 0.15:
                trace['death_cause'] = 'critical_depletion'
            elif agent.steps_without_resource_gain > 20:
                trace['death_cause'] = 'starvation'
            else:
                trace['death_cause'] = 'unknown'
            
            break
    
    return trace


def analyze_multi_episode(traces: List[Dict], num_runs: int) -> Dict[str, Any]:
    """Full analytics across all runs."""
    
    # 1. Survival Distribution
    survival_lengths = [t['death_episode'] if t['death_episode'] else 500 for t in traces]
    deaths = sum(1 for t in traces if t['death_episode'] is not None)
    
    # 2. Trigger Effectiveness
    all_triggers = []
    for t in traces:
        all_triggers.extend(t['triggers_fired'])
    
    trigger_effectiveness = {}
    for trigger_type in ['exploration_floor', 'starvation_recovery', 'critical_energy', 'uncertainty']:
        trigger_events = [e for e in all_triggers if e['trigger'] == trigger_type]
        if trigger_events:
            recovery_rate = sum(1 for e in trigger_events if e['energy_after'] > e['energy_before']) / len(trigger_events)
            avg_gain = np.mean([e['energy_after'] - e['energy_before'] for e in trigger_events])
            trigger_effectiveness[trigger_type] = {
                'count': len(trigger_events),
                'recovery_rate': recovery_rate,
                'avg_energy_delta': avg_gain,
            }
    
    # 3. Timing Sensitivity (EP24 insight)
    # For deaths, find: last trigger fired → death
    death_timing = []
    for t in traces:
        if t['death_episode']:
            # Find last trigger before death
            last_trigger = None
            for trig in reversed(t['triggers_fired']):
                if trig['episode'] < t['death_episode']:
                    last_trigger = trig
                    break
            
            if last_trigger:
                time_to_death = t['death_episode'] - last_trigger['episode']
                death_timing.append({
                    'death_ep': t['death_episode'],
                    'last_trigger_ep': last_trigger['episode'],
                    'time_to_death': time_to_death,
                    'trigger_type': last_trigger['trigger'],
                })
    
    # 4. Over-Exploration Risk
    exploration_payoffs = []
    for t in traces:
        exploration_payoffs.extend(t['exploration_payoff'])
    
    successful_explores = sum(1 for p in exploration_payoffs if p['success'])
    failed_explores = len(exploration_payoffs) - successful_explores
    
    # Energy crashes after exploration
    energy_crashes = 0
    for t in traces:
        for i, traj in enumerate(t['energy_trajectory']):
            if traj['action'] == 'explore':
                # Check if energy crashed in next 3 steps
                if i + 3 < len(t['energy_trajectory']):
                    future_energy = t['energy_trajectory'][i+3]['true_energy']
                    if future_energy < traj['true_energy'] - 0.15:
                        energy_crashes += 1
    
    return {
        'survival': {
            'runs': num_runs,
            'deaths': deaths,
            'survival_rate': 1 - (deaths / num_runs),
            'avg_lifespan': np.mean(survival_lengths),
            'min_lifespan': min(survival_lengths),
            'max_lifespan': max(survival_lengths),
            'std_lifespan': np.std(survival_lengths),
        },
        'trigger_effectiveness': trigger_effectiveness,
        'death_timing': death_timing,
        'over_exploration': {
            'total_explores': len(exploration_payoffs),
            'successful': successful_explores,
            'failed': failed_explores,
            'success_rate': successful_explores / len(exploration_payoffs) if exploration_payoffs else 0,
            'energy_crashes_after_explore': energy_crashes,
        },
    }


def main():
    print("=" * 70)
    print("🧬 K1.2 MULTI-EPISODE VALIDATION")
    print("   Testing: Is K1.2 robust... or just lucky?")
    print("   Running: 20 independent lives")
    print("=" * 70)
    
    NUM_RUNS = 20
    traces = []
    
    for i in range(NUM_RUNS):
        print(f"\n💀 Life {i+1}/{NUM_RUNS}")
        trace = run_single_life(seed=i*123 + 42)
        traces.append(trace)
        
        if trace['death_episode']:
            print(f"   DIED at EP{trace['death_episode']:03d} — {trace['death_cause']}")
        else:
            print(f"   SURVIVED 500 episodes 🎉")
    
    # Full analytics
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE ANALYTICS")
    print("=" * 70)
    
    stats = analyze_multi_episode(traces, NUM_RUNS)
    
    # 1. Survival Distribution
    print("\n🎯 SURVIVAL DISTRIBUTION")
    s = stats['survival']
    print(f"   Runs: {s['runs']}")
    print(f"   Deaths: {s['deaths']} | Survival Rate: {s['survival_rate']:.1%}")
    print(f"   Lifespan: {s['avg_lifespan']:.1f} ± {s['std_lifespan']:.1f}")
    print(f"   Range: [{s['min_lifespan']:.0f}, {s['max_lifespan']:.0f}]")
    
    # Bimodal check
    survival_lengths = [t['death_episode'] if t['death_episode'] else 500 for t in traces]
    early_deaths = sum(1 for l in survival_lengths if l < 100)
    late_deaths = sum(1 for l in survival_lengths if 100 <= l < 500)
    survivors = sum(1 for l in survival_lengths if l == 500)
    
    print(f"\n   Distribution Shape:")
    print(f"   - Early deaths (<100): {early_deaths}")
    print(f"   - Late deaths (100-499): {late_deaths}")
    print(f"   - Full survivors (500): {survivors}")
    
    if early_deaths > 0 and survivors > 0:
        print(f"   ⚠️  BIMODAL: Some die early, some survive — UNSTABLE")
    elif survivors >= 15:
        print(f"   🟢 STABLE: High survival consistency")
    elif survivors >= 10:
        print(f"   🟡 MODERATE: Variable but functional")
    else:
        print(f"   🔴 UNSTABLE: High death rate — needs tuning")
    
    # 2. Trigger Effectiveness
    print("\n🔥 TRIGGER EFFECTIVENESS CURVE")
    for trigger, data in stats['trigger_effectiveness'].items():
        print(f"   {trigger}:")
        print(f"      Count: {data['count']}")
        print(f"      Recovery rate: {data['recovery_rate']:.1%}")
        print(f"      Avg energy delta: {data['avg_energy_delta']:+.3f}")
    
    # 3. Timing Sensitivity
    print("\n⏱️  TIMING SENSITIVITY (EP24 Insight)")
    if stats['death_timing']:
        avg_time_to_death = np.mean([d['time_to_death'] for d in stats['death_timing']])
        print(f"   Avg time from last trigger → death: {avg_time_to_death:.1f} episodes")
        print(f"   Deaths analyzed: {len(stats['death_timing'])}")
        
        for d in stats['death_timing'][:3]:  # Show first 3
            print(f"   - EP{d['death_ep']:03d}: last trigger at EP{d['last_trigger_ep']:03d} ({d['trigger_type']})")
    else:
        print("   No deaths to analyze timing")
    
    # 4. Over-Exploration Risk
    print("\n⚠️  OVER-EXPLORATION RISK")
    oe = stats['over_exploration']
    print(f"   Total explores: {oe['total_explores']}")
    print(f"   Success rate: {oe['success_rate']:.1%}")
    print(f"   Failed explores: {oe['failed']} ({oe['failed']/oe['total_explores']:.1%})")
    print(f"   Energy crashes after explore: {oe['energy_crashes_after_explore']}")
    
    if oe['success_rate'] < 0.5:
        print(f"   🔴 HIGH RISK: Most explorations fail — K2 needs damping")
    elif oe['success_rate'] < 0.7:
        print(f"   🟡 MODERATE RISK: Some failed explorations")
    else:
        print(f"   🟢 LOW RISK: Explorations generally productive")
    
    # Final Verdict
    print("\n" + "=" * 70)
    print("🏆 K1.2 MULTI-EPISODE VERDICT")
    print("=" * 70)
    
    s = stats['survival']
    if s['survival_rate'] > 0.9 and s['std_lifespan'] < 50:
        verdict = "🟢 EXCELLENT: Stable, robust, ready for K2"
        recommendation = "Proceed to K2: Learn WHEN and HOW MUCH"
    elif s['survival_rate'] > 0.7:
        verdict = "🟡 GOOD: Functional but variable — K1.3 refinement recommended"
        recommendation = "Add: anticipatory triggers, trend-based detection"
    else:
        verdict = "🔴 NEEDS WORK: Unstable — refine before K2"
        recommendation = "Focus: earlier detection, better trigger tuning"
    
    print(f"\n   {verdict}")
    print(f"\n   Recommendation: {recommendation}")
    
    return stats


if __name__ == "__main__":
    main()
