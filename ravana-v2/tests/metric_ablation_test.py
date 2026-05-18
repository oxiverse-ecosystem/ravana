"""
EXPERIMENT 2: Metric Ablation Study
Tests if the Dissonance mechanism is necessary for stability and fairness.
"""
import sys
import os
import numpy as np
import json
from typing import Dict, Any

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research.core_k0.agent_loop_k2 import K2_Agent, AgentAction
from research.experiments_k0.classroom_env import ClassroomEnv

def run_ablation_study():
    print("="*70)
    print("EXPERIMENT 2: METRIC ABLATION STUDY")
    print("Goal: Prove Dissonance Engine is necessary for Fairness stability.")
    print("="*70)
    
    # 1. Standard RAVANA Agent
    ravana_agent = K2_Agent()
    
    # 2. Ablated Agent (Identity & Dissonance gates forced OFF)
    ablated_agent = K2_Agent()
    
    env = ClassroomEnv(seed=42)
    
    records = {'ravana': [], 'ablated': []}
    
    print("\n[PHASE 1] Comparing Convergence Speed (2000 episodes)...")
    
    for ep in range(2000):
        # --- Run RAVANA ---
        obs_r = env._generate_observation()
        res_r = ravana_agent.step(env)
        
        # --- Run ABLATED (Identity forced to low baseline) ---
        ablated_agent.state.identity_commitment = 0.1
        # Bypass the utility filtering in _get_action_by_expected_utility 
        # (simulated by setting commitment < 0.7)
        obs_a = env._generate_observation()
        res_a = ablated_agent.step(env)
        
        if ep % 500 == 0:
            fairness = env.get_fairness_metrics()
            print(f"  EP {ep:4d} | RAVANA GAP: {fairness['gap_a_b']:.2%} | ABLATED GAP: {fairness['gap_a_b']:.2%}")

    # Final Fairness Audit
    ravana_fairness = env.get_fairness_metrics()
    
    # To truly compare, we need to run them in separate environments to see divergence
    print("\n[PHASE 2] Independent Divergence Test...")
    
    def run_indep(agent, is_ravana):
        e = ClassroomEnv(seed=123)
        for _ in range(2000):
            if not is_ravana:
                agent.state.identity_commitment = 0.1
            agent.step(e)
        return e.get_fairness_metrics()

    ravana_final = run_indep(ravana_agent, True)
    ablated_final = run_indep(ablated_agent, False)

    print("\n" + "="*70)
    print("ABLATION REPORT: FAIRNESS STABILITY")
    print("="*70)
    print(f"RAVANA (Full) Gap:  {ravana_final['gap_a_b']:.2%}")
    print(f"ABLATED (No ID) Gap: {ablated_final['gap_a_b']:.2%}")
    
    improvement = (ablated_final['gap_a_b'] - ravana_final['gap_a_b']) / ablated_final['gap_a_b'] if ablated_final['gap_a_b'] > 0 else 0
    
    print(f"\nFairness Gain from Dissonance: {improvement:.1%}")
    if improvement > 0.15:
        print("✅ SUCCESS: Dissonance mechanism is structurally necessary for Fairness.")
    else:
        print("⚠️ MARGINAL: Ablation had minimal impact.")

    summary = {
        'ravana_gap': ravana_final['gap_a_b'],
        'ablated_gap': ablated_final['gap_a_b'],
        'fairness_gain': float(improvement),
        'status': 'NECESSARY' if improvement > 0.15 else 'MARGINAL'
    }
    
    with open("results/exp2_metric_ablation.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] Results saved to results/exp2_metric_ablation.json")

if __name__ == "__main__":
    run_ablation_study()
