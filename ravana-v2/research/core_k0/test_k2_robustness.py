"""
K2 Robustness Validation: 4 Tests Before K3

Tests if 100% survival is robust intelligence or overfitted perfection.
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop_k2 import K2_Agent
from research.core_k0.agent_loop_k1_3 import K1_3_Agent
from research.experiments_k0.resource_env import ResourceSurvivalEnv, AgentAction
from typing import List, Dict, Any
import numpy as np
import copy


class PerturbedResourceEnv(ResourceSurvivalEnv):
    """Environment with controllable perturbations for robustness testing."""
    
    def __init__(self, seed=None, resource_mult=1.0, noise_mult=1.0, 
                 metabolism=0.02, risk_mult=1.0):
        super().__init__(seed=seed)
        self.resource_mult = resource_mult
        self.noise_mult = noise_mult
        self.metabolism = metabolism
        self.risk_mult = risk_mult
    
    def execute_action(self, action: AgentAction) -> Dict[str, Any]:
        """Execute with perturbations applied."""
        self._update_regime()
        
        # Modified action outcomes
        if action == AgentAction.EXPLORE:
            success = self.rng.random() > (self.hidden_risk * 1.5 * self.risk_mult)
            if success:
                energy_gain = self.rng.normal(0.15 * self.resource_mult, 0.08 * self.noise_mult)
                resource_gain = self.rng.normal(0.2 * self.resource_mult, 0.1 * self.noise_mult)
            else:
                energy_gain = self.rng.normal(-0.2 * self.noise_mult, 0.05 * self.noise_mult)
                resource_gain = self.rng.normal(-0.05, 0.05)
            utility = 1.0 if success else -0.5
            
        elif action == AgentAction.EXPLOIT:
            energy_gain = self.rng.normal(0.05 * self.resource_mult, 0.03 * self.noise_mult)
            resource_gain = self.rng.normal(0.1 * self.resource_mult, 0.05 * self.noise_mult)
            utility = 0.5
            
        elif action == AgentAction.CONSERVE:
            energy_gain = self.rng.normal(0.02, 0.01)
            resource_gain = self.rng.normal(0.03, 0.02)
            utility = 0.3
        else:
            energy_gain = 0
            resource_gain = 0
            utility = 0
        
        # Apply metabolism
        self.true_energy = np.clip(self.true_energy + energy_gain - self.metabolism, 0, 1)
        self.true_resources = np.clip(self.true_resources + resource_gain, 0, 1)
        
        observation = self._generate_observation()
        alive = self.true_energy > 0.1
        
        return {
            'episode': self.episode,
            'action': action.value,
            'alive': alive,
            'utility': utility,
            'true_energy': self.true_energy,
            'true_resources': self.true_resources,
            'observation': observation,
            'regime': self.current_regime.value
        }


class AdversarialResourceEnv(ResourceSurvivalEnv):
    """Environment that occasionally gives misleading signals."""
    
    def __init__(self, seed=None, misleading_rate=0.2):
        super().__init__(seed=seed)
        self.misleading_rate = misleading_rate
        self.misleading_episodes = set()
        
        # Pre-generate misleading episodes
        rng = np.random.RandomState(seed)
        for ep in range(100):
            if rng.random() < self.misleading_rate:
                self.misleading_episodes.add(ep)
    
    def execute_action(self, action: AgentAction) -> Dict[str, Any]:
        """Sometimes invert exploration outcomes (looks good, actually bad)."""
        result = super().execute_action(action)
        
        # On misleading episodes, make exploration look successful but actually damage
        if self.episode - 1 in self.misleading_episodes and action == AgentAction.EXPLORE:
            # Override to look like success
            result['utility'] = 1.0
            # But secretly damage more
            self.true_energy = np.clip(self.true_energy - 0.1, 0, 1)
            result['true_energy'] = self.true_energy
            result['alive'] = self.true_energy > 0.1
        
        return result


def run_single_episode_run(agent_class, env_class, n_episodes: int = 100, 
                          env_kwargs: Dict = None, seed: int = 42) -> Dict[str, Any]:
    """Run one agent through episodes with given environment config."""
    env_kwargs = env_kwargs or {}
    env = env_class(seed=seed, **env_kwargs)
    agent = agent_class()
    
    early_deaths = 0
    late_deaths = 0
    total_reward = 0.0
    contexts_learned = 0
    
    for ep in range(n_episodes):
        result = agent.step(env)
        total_reward += env.history[-1]['utility'] if env.history else 0
        
        if not result['alive']:
            if ep < n_episodes // 2:
                early_deaths += 1
            else:
                late_deaths += 1
            break
    
    if hasattr(agent, 'context_weights'):
        contexts_learned = len(agent.context_weights)
    
    return {
        'survived': result['alive'] and len(env.history) >= n_episodes - 1,
        'total_episodes': len(env.history),
        'early_deaths': early_deaths,
        'late_deaths': late_deaths,
        'cumulative_reward': total_reward,
        'contexts_learned': contexts_learned
    }


def test_1_environment_perturbation():
    """Test 1: Change resource frequency, reward magnitudes, noise."""
    print("\n" + "="*70)
    print("TEST 1: Environment Perturbation")
    print("="*70)
    
    perturbations = [
        ("Baseline", {}),
        ("Scarce resources", {"resource_mult": 0.6}),
        ("High noise", {"noise_mult": 2.0}),
        ("Fast metabolism", {"metabolism": 0.03}),
        ("High risk", {"risk_mult": 1.5}),
        ("Combined harsh", {"resource_mult": 0.7, "noise_mult": 1.5, "metabolism": 0.025}),
    ]
    
    results = []
    for name, kwargs in perturbations:
        k2_results = []
        k13_results = []
        
        for i in range(5):  # 5 runs per condition
            k2_result = run_single_episode_run(K2_Agent, PerturbedResourceEnv, 
                                               n_episodes=100, env_kwargs=kwargs, seed=i*42)
            k13_result = run_single_episode_run(K1_3_Agent, PerturbedResourceEnv,
                                                n_episodes=100, env_kwargs=kwargs, seed=i*42)
            k2_results.append(k2_result)
            k13_results.append(k13_result)
        
        k2_survival = sum(1 for r in k2_results if r['survived']) / len(k2_results)
        k13_survival = sum(1 for r in k13_results if r['survived']) / len(k13_results)
        
        results.append({
            'condition': name,
            'k2_survival': k2_survival,
            'k13_survival': k13_survival,
            'improvement': k2_survival - k13_survival
        })
        
        status = "✓" if k2_survival >= k13_survival else "⚠"
        print(f"  {status} {name:20s}: K2={k2_survival:.1%} vs K1.3={k13_survival:.1%} (Δ{results[-1]['improvement']:+.1%})")
    
    return results


def test_2_adversarial_runs():
    """Test 2: Misleading signals and fake good outcomes."""
    print("\n" + "="*70)
    print("TEST 2: Adversarial Runs (Misleading Signals)")
    print("="*70)
    
    misleading_rates = [0.0, 0.1, 0.2, 0.3]
    
    results = []
    for rate in misleading_rates:
        k2_results = []
        k13_results = []
        
        for i in range(5):
            k2_result = run_single_episode_run(K2_Agent, AdversarialResourceEnv,
                                               n_episodes=100, env_kwargs={"misleading_rate": rate}, seed=i*42)
            k13_result = run_single_episode_run(K1_3_Agent, AdversarialResourceEnv,
                                                n_episodes=100, env_kwargs={"misleading_rate": rate}, seed=i*42)
            k2_results.append(k2_result)
            k13_results.append(k13_result)
        
        k2_survival = sum(1 for r in k2_results if r['survived']) / len(k2_results)
        k13_survival = sum(1 for r in k13_results if r['survived']) / len(k13_results)
        
        results.append({
            'misleading_rate': rate,
            'k2_survival': k2_survival,
            'k13_survival': k13_survival,
        })
        
        status = "✓" if k2_survival >= k13_survival * 0.9 else "⚠"  # Allow 10% degradation
        print(f"  {status} Misleading {rate:.0%}: K2={k2_survival:.1%} vs K1.3={k13_survival:.1%}")
    
    return results


def test_3_cold_start_robustness():
    """Test 3: Reset learning, test relearning speed and consistency."""
    print("\n" + "="*70)
    print("TEST 3: Cold Start Robustness")
    print("="*70)
    
    # Multiple cold starts
    convergence_episodes = []
    
    for i in range(10):
        env = ResourceSurvivalEnv(seed=i*42)
        agent = K2_Agent()
        
        survived = True
        for ep in range(100):
            result = agent.step(env)
            if not result['alive']:
                survived = False
                convergence_episodes.append(ep)
                break
        
        if survived:
            convergence_episodes.append(100)
    
    consistency = sum(1 for ep in convergence_episodes if ep >= 90) / len(convergence_episodes)
    avg_convergence = np.mean(convergence_episodes)
    
    print(f"  Convergence rate (≥90 eps): {consistency:.1%}")
    print(f"  Avg episodes to converge: {avg_convergence:.1f}")
    print(f"  Min episodes: {min(convergence_episodes)}")
    print(f"  Max episodes: {max(convergence_episodes)}")
    
    status = "✓" if consistency >= 0.8 else "⚠"
    print(f"  {status} {'PASS' if consistency >= 0.8 else 'NEEDS WORK'}")
    
    return {
        'consistency': consistency,
        'avg_convergence': avg_convergence,
        'convergence_episodes': convergence_episodes
    }


def test_4_long_horizon_stress():
    """Test 4: Very long episodes (5x-10x normal duration)."""
    print("\n" + "="*70)
    print("TEST 4: Long Horizon Stress (500 episodes)")
    print("="*70)
    
    durations = [100, 250, 500]
    
    results = []
    for duration in durations:
        k2_survivals = []
        
        for i in range(5):
            env = ResourceSurvivalEnv(seed=i*42)
            agent = K2_Agent()
            
            survived = True
            for ep in range(duration):
                result = agent.step(env)
                if not result['alive']:
                    survived = False
                    break
            
            k2_survivals.append(1 if survived else 0)
        
        survival_rate = sum(k2_survivals) / len(k2_survivals)
        results.append({
            'duration': duration,
            'survival': survival_rate
        })
        
        status = "✓" if survival_rate >= 0.8 else "⚠"
        print(f"  {status} {duration:3d} episodes: {survival_rate:.1%} survival")
    
    return results


def run_all_robustness_tests():
    """Run all 4 robustness tests."""
    print("\n" + "="*70)
    print("K2 ROBUSTNESS VALIDATION SUITE")
    print("Testing if 100% survival is robust or overfitted")
    print("="*70)
    
    test1_results = test_1_environment_perturbation()
    test2_results = test_2_adversarial_runs()
    test3_results = test_3_cold_start_robustness()
    test4_results = test_4_long_horizon_stress()
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    # Calculate pass rates
    test1_pass = sum(1 for r in test1_results if r['improvement'] >= -0.1) / len(test1_results)
    test2_pass = sum(1 for r in test2_results if r['k2_survival'] >= r['k13_survival'] * 0.9) / len(test2_results)
    test3_pass = test3_results['consistency']
    test4_pass = sum(1 for r in test4_results if r['survival'] >= 0.8) / len(test4_results)
    
    overall = (test1_pass + test2_pass + test3_pass + test4_pass) / 4
    
    print(f"\n  Test 1 (Perturbation):     {test1_pass:.1%} pass")
    print(f"  Test 2 (Adversarial):      {test2_pass:.1%} pass")
    print(f"  Test 3 (Cold Start):     {test3_pass:.1%} pass")
    print(f"  Test 4 (Long Horizon):     {test4_pass:.1%} pass")
    print(f"\n  OVERALL ROBUSTNESS:        {overall:.1%}")
    
    if overall >= 0.8:
        print("\n  🟢 ROBUST INTELLIGENCE CONFIRMED")
        print("  → K2 is genuinely adaptive, not overfitted")
        print("  → Ready for K3 (meta-learning)")
    elif overall >= 0.6:
        print("\n  🟡 PARTIALLY ROBUST")
        print("  → Some adaptation, but fragile in certain conditions")
        print("  → Needs more tuning before K3")
    else:
        print("\n  🔴 OVERFITTING DETECTED")
        print("  → High performance on narrow conditions only")
        print("  → Requires structural fixes")
    
    print("="*70 + "\n")
    
    return {
        'test1': test1_results,
        'test2': test2_results,
        'test3': test3_results,
        'test4': test4_results,
        'overall_robustness': overall,
        'ready_for_k3': overall >= 0.8
    }


if __name__ == "__main__":
    results = run_all_robustness_tests()
    sys.exit(0 if results['ready_for_k3'] else 1)
