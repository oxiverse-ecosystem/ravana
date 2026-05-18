"""
EXPERIMENT 3: Cross-Environment Generalization
Tests if an identity trained in Survival carries over to the Classroom.
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
from research.experiments_k0.resource_env import ResourceSurvivalEnv
from research.experiments_k0.classroom_env import ClassroomEnv

def run_cross_domain_test():
    print("="*70)
    print("EXPERIMENT 3: CROSS-ENVIRONMENT GENERALIZATION")
    print("Goal: Prove Identity persists across disparate domains.")
    print("="*70)
    
    # 1. Pre-train Agent in Survival Environment
    print("\n[PHASE 1] Pre-training Agent in Survival (2000 episodes)...")
    transfer_agent = K2_Agent()
    survival_env = ResourceSurvivalEnv(seed=42)
    for ep in range(2000):
        transfer_agent.step(survival_env)
    
    print(f"  Survival Training Complete. Identity={transfer_agent.state.identity_commitment:.3f}")
    
    # 2. Initialize Naive Agent (No pre-training)
    naive_agent = K2_Agent()
    
    # 3. Test both in Classroom
    print("\n[PHASE 2] Comparative Deployment in Classroom (Short Horizon: 100 episodes)...")
    classroom_env_t = ClassroomEnv(seed=123)
    classroom_env_n = ClassroomEnv(seed=123)
    
    for ep in range(100):
        transfer_agent.step(classroom_env_t)
        naive_agent.step(classroom_env_n)
        
    # Analysis
    t_metrics = classroom_env_t.get_fairness_metrics()
    n_metrics = classroom_env_n.get_fairness_metrics()
    
    print("\n" + "="*70)
    print("GENERALIZATION REPORT: DOMAIN TRANSFER")
    print("="*70)
    print(f"Pre-trained Agent Gap: {t_metrics['gap_a_b']:.2%}")
    print(f"Naive Agent Gap:       {n_metrics['gap_a_b']:.2%}")
    
    improvement = (n_metrics['gap_a_b'] - t_metrics['gap_a_b']) / n_metrics['gap_a_b'] if n_metrics['gap_a_b'] > 0 else 0
    
    print(f"\nFairness Transfer Efficiency: {improvement:.1%}")
    if improvement > 0.10:
        print("✅ SUCCESS: Survival identity accelerated fairness in the classroom.")
    else:
        print("⚠️ MARGINAL: Cross-domain transfer was minimal.")

    summary = {
        'transfer_gap': t_metrics['gap_a_b'],
        'naive_gap': n_metrics['gap_a_b'],
        'transfer_efficiency': float(improvement),
        'status': 'GENERALIZED' if improvement > 0.10 else 'NARROW'
    }
    
    with open("results/exp3_cross_domain.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] Results saved to results/exp3_cross_domain.json")

if __name__ == "__main__":
    run_cross_domain_test()
