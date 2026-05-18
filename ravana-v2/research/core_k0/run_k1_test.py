#!/usr/bin/env python3
"""
RAVANA K1 Test: Risk-Transformed Utility Under Uncertainty
Compare K0 (died at EP220) vs K1 (should survive longer)
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop_k1 import K1Agent, AgentAction, K1AgentConfig
from research.experiments_k0.resource_env import ResourceSurvivalEnv

def main():
    print("="*70)
    print("RAVANA K1: RISK-TRANSFORMED UTILITY TEST")
    print("="*70)
    
    # Same environment that killed K0
    env = ResourceSurvivalEnv(seed=42)
    
    # K1 Agent with risk transformation
    config = K1AgentConfig(
        survival_threshold=0.2,
        base_risk_aversion=0.3,
        uncertainty_exponent=2.0,
        critical_energy_threshold=0.15,
        max_uncertainty_for_normal=0.6
    )
    agent = K1Agent(config)
    
    print(f"\n🧪 K1 CONFIGURATION")
    print(f"   Base risk aversion: {config.base_risk_aversion}")
    print(f"   Uncertainty exponent (beta): {config.uncertainty_exponent}")
    print(f"   Critical energy threshold: {config.critical_energy_threshold}")
    print(f"\n🌍 ENVIRONMENT (same as K0 death scenario)")
    print(f"   Seed: 42 (same as K0)")
    print(f"   Hidden risk: {env.hidden_risk}")
    print(f"   Observation noise: {env.base_noise}")
    print(f"   Regime: shifts periodically (hidden from agent)")
    print(f"\n🚀 RUNNING K1 LOOP: 500 episodes")
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
    print(f"K1 TEST COMPLETE")
    print(f"{'='*70}")
    print(f"\n📈 SURVIVAL METRICS")
    print(f"   Total episodes: {status['episode']}")
    print(f"   Death count: {status['death_count']}")
    print(f"   Survival rate: {status['survival_rate']:.1%}")
    print(f"   Cumulative reward: {status['cumulative_reward']:.1f}")
    print(f"\n🧠 K1 BEHAVIORAL ANALYSIS")
    print(f"   Actions taken: {len(status['action_distribution'])}")
    for action, count in status['action_distribution'].items():
        print(f"      {action}: {count}")
    print(f"\n🔍 COMPARE TO K0:")
    print(f"   K0 died at: EP220 (survival rate: 44%)")
    print(f"   K1 survived to: EP{status['episode']} (survival rate: {status['survival_rate']:.1%})")
    if status['episode'] > 220:
        print(f"   ✅ K1 OUTLIVED K0 by {status['episode']-220} episodes!")
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
