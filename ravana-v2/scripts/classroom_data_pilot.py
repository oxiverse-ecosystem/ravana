"""
RAVANA Pilot: Classroom Data Integration (Option B)
Validates fairness and adaptation on synthetic student interactions.
"""
import sys
import os
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, List

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research.core_k0.agent_loop_k2 import K2_Agent, AgentAction
from research.core_k0.metrics import RavanaMetrics
from research.experiments_k0.resource_env import ResourceSurvivalEnv
from research.experiments_k0.classroom_env import ClassroomEnv

class ClassroomPilot:
    def __init__(self, n_episodes=5000, real_data_ratio=0.1, seed=42):
        self.n_episodes = n_episodes
        self.real_data_ratio = real_data_ratio
        self.rng = np.random.RandomState(seed)
        
        self.agent = K2_Agent()
        self.metrics_engine = RavanaMetrics()
        
        self.sim_env = ResourceSurvivalEnv(seed=seed)
        self.classroom_env = ClassroomEnv(seed=seed)
        
        self.results = {
            'episodes': [],
            'fairness_history': [],
            'safety_alerts': []
        }
        
        # Safety Tracking
        self.consecutive_high_dissonance = 0
        
    def _compute_paper_metrics(self, episode: int) -> Dict[str, float]:
        """Use validated paper formulas."""
        paper_state = self.agent.get_paper_metrics()
        
        # Last action value
        if self.agent.state.action_history:
            last_action = self.agent.state.action_history[-1][1]
        else:
            last_action = AgentAction.CONSERVE
            
        action_map = {AgentAction.EXPLORE: 0.3, AgentAction.EXPLOIT: 0.7, AgentAction.CONSERVE: 0.9}
        action_value = action_map.get(last_action, 0.5)
        
        d_score = self.metrics_engine.calculate_dissonance(
            beliefs=paper_state['beliefs'],
            actions=[action_value] * len(paper_state['beliefs']),
            confidences=paper_state['confidences'],
            vad_weights=paper_state['vad_weights'],
            context_mismatch=0.2,
            identity_violation=0.0,
            cognitive_load=paper_state['cognitive_load'],
            reappraisal_resistance=paper_state['reappraisal_resistance']
        )
        
        i_score = self.metrics_engine.calculate_identity_strength(
            commitment_history=[paper_state['identity_commitment']],
            volatility_history=[0.1],
            context_stability=0.5,
            episode=episode
        )
        
        return {'dissonance': d_score, 'identity': i_score}

    def run_pilot(self):
        print("="*70)
        print("RAVANA CLASSROOM DATA PILOT (Option B)")
        print(f"Episodes: {self.n_episodes} | Real Data Ratio: {self.real_data_ratio:.0%}")
        print("="*70)
        
        for ep in range(self.n_episodes):
            # 1. Select Environment
            is_real = self.rng.random() < self.real_data_ratio
            env = self.classroom_env if is_real else self.sim_env
            source = "REAL" if is_real else "SIM"
            
            # 2. Reset Env for episode sequence
            env.true_energy = 0.6
            env.episode = ep
            
            # 3. Capture Pre-metrics
            metrics = self._compute_paper_metrics(ep)
            
            # 4. Run episode (20 steps)
            alive = True
            utility_sum = 0
            for _ in range(20):
                res = self.agent.step(env)
                utility_sum += res.get('utility', 0.1)
                if not res['alive']:
                    alive = False
                    break
            
            # 5. Safety Triggers
            if metrics['dissonance'] > 0.7:
                self.consecutive_high_dissonance += 1
                if self.consecutive_high_dissonance >= 50:
                    alert = f"[ALERT] EP {ep}: Persistent high dissonance detected! Triggering Audit."
                    print(alert)
                    self.results['safety_alerts'].append(alert)
            else:
                self.consecutive_high_dissonance = 0
                
            # 6. Periodic Fairness Check
            if is_real and ep % 500 == 0:
                fairness = self.classroom_env.get_fairness_metrics()
                print(f"  EP {ep:5d} [{source}] D={metrics['dissonance']:.3f} I={metrics['identity']:.3f} GAP={fairness['gap_a_b']:.2%}")
                self.results['fairness_history'].append({
                    'episode': ep,
                    'gap': fairness['gap_a_b'],
                    'rates': fairness['rates']
                })
            elif ep % 500 == 0:
                print(f"  EP {ep:5d} [{source}] D={metrics['dissonance']:.3f} I={metrics['identity']:.3f}")

            # Record
            self.results['episodes'].append({
                'episode': ep,
                'source': source,
                'dissonance': metrics['dissonance'],
                'identity': metrics['identity'],
                'alive': alive
            })
            
        self._generate_report()
        
    def _generate_report(self):
        print("\n" + "="*70)
        print("PILOT FINAL REPORT")
        print("="*70)
        
        fairness = self.classroom_env.get_fairness_metrics()
        print(f"\n1. Fairness (Demographic Parity Gap):")
        print(f"   Final Gap A-B: {fairness['gap_a_b']:.2%} (Target: < 10%)")
        print(f"   Status: {'[PASS]' if fairness['gap_a_b'] < 0.10 else '[WARN]'}")
        
        first_d = self.results['episodes'][0]['dissonance']
        last_d = self.results['episodes'][-1]['dissonance']
        print(f"\n2. Dissonance Adaptation:")
        print(f"   Trajectory: {first_d:.3f} -> {last_d:.3f}")
        
        avg_survival = np.mean([1.0 if e['alive'] else 0.0 for e in self.results['episodes']])
        print(f"\n3. Overall Survival Rate: {avg_survival:.1%}")
        
        output_file = "results/classroom_pilot_results.json"
        with open(output_file, 'w') as f:
            # Strip large episode list for summary
            summary = {k: v for k, v in self.results.items() if k != 'episodes'}
            summary['final_metrics'] = {
                'fairness': fairness,
                'avg_survival': avg_survival,
                'dissonance_delta': last_d - first_d
            }
            json.dump(summary, f, indent=2)
        print(f"\n[SAVE] Results saved to {output_file}")

if __name__ == "__main__":
    pilot = ClassroomPilot(n_episodes=5000, real_data_ratio=0.1)
    pilot.run_pilot()
