#!/usr/bin/env python3
"""
RAVANA K1.1: Calibrated Survival Intelligence
Tuned parameters: beta=1.2, base_risk=0.15, critical=0.10
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop_k1 import K1Agent, AgentAction, K1AgentConfig
from research.experiments_k0.resource_env import ResourceSurvivalEnv

def main():
    print("="*70)
    print("RAVANA K1.1: CALIBRATED SURVIVAL INTELLIGENCE")
    print("="*70)
    
    env = ResourceSurvivalEnv(seed=42)  # Same environment
    
    # CALIBRATED parameters (tuned from K1 failure)
    config = K1AgentConfig(
        survival_threshold=0.2,
        base_risk_aversion=0.15,      # Lower from 0.3 → 0.15
        uncertainty_exponent=1.2,       # Lower from 2.0 → 1.2 (less explosive)
        critical_energy_threshold=0.10, # Lower from 0.15 → 0.10
        max_uncertainty_for_normal=0.7  # Higher from 0.6 → 0.7 (more tolerance)
    )
    agent = K1Agent(config)
    
    print(f"\n🧪 K1.1 CALIBRATED CONFIGURATION")
    print(f"   Base risk aversion: {config.base_risk_aversion} (was 0.3)")
    print(f"   Uncertainty exponent (beta): {config.uncertainty_exponent} (was 2.0)")
    print(f"   Critical energy threshold: {config.critical_energy_threshold} (was 0.15)")
    print(f"   Max uncertainty for normal: {config.max_uncertainty_for_normal} (was 0.6)")
    print(f"\n🌍 ENVIRONMENT (identical to K0/K1)")
    print(f"   Seed: 42")
    print(f"   Hidden risk: {env.hidden_risk}")
    print(f"   Observation noise: {env.base_noise}")
    print(f"\n🚀 RUNNING K1.1: 500 episodes")
    print("-"*70)
    
    for ep in range(500):
        result = agent.step(env)
        
        # Show death events
        if not result['alive']:
            print(f"\n💀 EP{ep:04d}: AGENT DIED")
            print(f"   Final observation: {result['observation']}")
            print(f"   Hidden truth: {env.get_hidden_truth()}")
            break
        
        # Periodic status
        if ep % 100 == 0 and ep > 0:
            status = agent.get_status()
            print(f"\n📊 EP{ep:04d} Status")
            print(f"   Survival rate: {status['survival_rate']:.1%}")
            print(f"   Cumulative reward: {status['cumulative_reward']:.1f}")
            print(f"   Current: E={status['current_state']['energy']:.2f} "
                  f"R={status['current_state']['resources']:.2f}")
            print(f"   Hidden regime: {env.current_regime.value}")
            print(f"   Avg risk aversion: {status.get('avg_risk_aversion', 0.3):.2f}")
            print(f"   Survival overrides: {status.get('survival_override_activations', 0)}")
    
    # Final summary
    status = agent.get_status()
    print(f"\n{'='*70}")
    print(f"K1.1 TEST COMPLETE")
    print(f"{'='*70}")
    print(f"\n📈 SURVIVAL METRICS")
    print(f"   Total episodes: {status['episode']}")
    print(f"   Death count: {status['death_count']}")
    print(f"   Survival rate: {status['survival_rate']:.1%}")
    print(f"   Cumulative reward: {status['cumulative_reward']:.1f}")
    print(f"\n🎯 COMPARISON:")
    print(f"   K0:  EP220 death, 44% survival")
    print(f"   K1:  EP25 death, 96% survival (too conservative)")
    print(f"   K1.1: EP{status['episode']} death, {status['survival_rate']:.1%} survival")
    if status['episode'] > 220:
        print(f"   ✅ K1.1 OUTLIVES K0!")
    elif status['episode'] > 25:
        print(f"   ✅ K1.1 improves on K1 by {status['episode']-25} episodes")
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
