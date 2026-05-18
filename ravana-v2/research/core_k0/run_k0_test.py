#!/usr/bin/env python3
"""
RAVANA K0 Test: The First Agent Loop
Does it survive? Does it learn? Does it die intelligently?
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop import MinimalAgent, K0Config, AgentAction
from research.experiments_k0.resource_env import ResourceSurvivalEnv, HiddenRegime


def main():
    print("=" * 70)
    print("RAVANA K0: THE SMALLEST POSSIBLE AGENT")
    print("Testing: Belief → Decision → Action → Consequence")
    print("=" * 70)
    
    # Create agent and environment
    agent = MinimalAgent(K0Config())
    env = ResourceSurvivalEnv(seed=42)
    
    print("\n🧪 K0 CONFIGURATION")
    print(f"   Survival threshold: {agent.config.survival_threshold}")
    print(f"   Risk aversion: {agent.config.risk_aversion}")
    print(f"   Uncertainty penalty: {agent.config.uncertainty_penalty}")
    print("\n🌍 ENVIRONMENT")
    print(f"   Starting regime: {env.current_regime.value}")
    print(f"   Hidden risk: {env.hidden_risk:.2f}")
    print(f"   Observation noise: {env.base_noise:.2f}")
    
    # Run K0 loop
    total_episodes = 500
    print(f"\n🚀 RUNNING K0 LOOP: {total_episodes} episodes")
    print("-" * 70)
    
    for ep in range(total_episodes):
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
    
    # Final summary
    print("\n" + "=" * 70)
    print("K0 TEST COMPLETE")
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
    
    # Action distribution
    action_counts = {}
    for _, action, _ in agent.state.action_history:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print(f"\n🎯 ACTION DISTRIBUTION")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / len(agent.state.action_history) * 100
        print(f"   {action.value}: {count} ({pct:.1f}%)")
    
    # K0 Verdict
    print(f"\n🏆 K0 VERDICT")
    if status['survival_rate'] > 0.9:
        verdict = "🟢 EXCELLENT: Agent survives intelligently"
    elif status['survival_rate'] > 0.7:
        verdict = "🟡 GOOD: Agent survives most of the time"
    elif status['survival_rate'] > 0.5:
        verdict = "🟠 STRUGGLING: Agent barely survives"
    else:
        verdict = "🔴 FAILING: Agent dies too often"
    
    print(f"   {verdict}")
    print(f"\n🧭 K0 TEST QUESTION")
    print(f"   Does belief → action alignment hold?")
    print(f"   Does uncertainty → cautious behavior?")
    print(f"   Does survival improve over time?")
    print(f"\n{'='*70}\n")
    
    return status


if __name__ == "__main__":
    main()
