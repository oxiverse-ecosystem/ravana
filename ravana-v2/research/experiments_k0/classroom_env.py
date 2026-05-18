"""
RAVANA K0: Classroom Data Environment
Validates fairness and adaptation using synthetic student interaction patterns.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import os
import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from research.core_k0.agent_loop_k2 import AgentAction

class ClassroomEnv:
    """
    Environment that replays synthetic student interactions.
    
    Tests:
    - Demographic Parity Gap (Bias mitigation)
    - Dissonance Spikes (Uncertainty adaptation)
    - Adversarial Robustness
    """
    
    def __init__(self, data_path: str = "results/synthetic_student_interactions.csv", seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Synthetic data not found at {data_path}. Run generate_student_data.py first.")
            
        self.df = pd.read_csv(data_path)
        self.total_records = len(self.df)
        self.current_idx = 0
        
        # State tracking
        self.true_energy: float = 0.6
        self.episode: int = 0
        self.history: list = []
        
        # Fairness tracking
        self.group_outcomes = {
            'Group A': {'success': 0, 'total': 0},
            'Group B': {'success': 0, 'total': 0},
            'Group C': {'success': 0, 'total': 0}
        }
        
    def _generate_observation(self) -> Dict[str, Any]:
        """Expose student features as observations."""
        record = self.df.iloc[self.current_idx]
        
        return {
            'energy_obs': self.true_energy,
            'student_quality': record['response_quality'],
            'demographic_group': record['demographic_group'],
            'interaction_type': record['interaction_type'],
            'adversarial': bool(record['adversarial_flag']),
            'noise': record['noise_level'],
            'observation_quality': 1.0 - record['noise_level']
        }
        
    def execute_action(self, action: AgentAction) -> Dict[str, Any]:
        """
        Execute pedagogical action and compute outcome.
        
        Actions:
        - EXPLORE: High-effort tailored feedback (High risk/reward)
        - EXPLOIT: Standard instruction (Medium risk/reward)
        - CONSERVE: Minimal intervention (Low risk/reward)
        """
        record = self.df.iloc[self.current_idx]
        group = record['demographic_group']
        quality = record['response_quality']
        
        # Logic: Action success depends on student quality + agent effort
        if action == AgentAction.EXPLORE:
            # Targeted intervention helps low quality, but uses more energy
            success_prob = max(quality, 0.7) # Boost low quality
            energy_cost = 0.05
            utility = 1.0 if success_prob > self.rng.random() else -0.5
        elif action == AgentAction.EXPLOIT:
            # Standard instruction relies on student quality
            success_prob = quality
            energy_cost = 0.02
            utility = 0.5 if success_prob > self.rng.random() else 0.0
        else: # CONSERVE
            # Minimal effort, relies on student autonomy
            success_prob = quality * 0.8
            energy_cost = 0.01
            utility = 0.3 if success_prob > self.rng.random() else 0.1
            
        success = utility > 0
        
        # Update energy
        self.true_energy = np.clip(self.true_energy + (utility * 0.1) - energy_cost, 0, 1)
        
        # Update fairness tracking
        self.group_outcomes[group]['total'] += 1
        if success:
            self.group_outcomes[group]['success'] += 1
            
        # Record results
        result = {
            'episode': self.episode,
            'student_id': record['student_id'],
            'group': group,
            'action': action.value,
            'success': success,
            'utility': utility,
            'alive': self.true_energy > 0.1,
            'true_energy': self.true_energy,
            'adversarial_hit': record['adversarial_flag'] and not success,
            'dissonance_trigger': record['noise_level'] > 0.15
        }
        
        self.history.append(result)
        self.episode += 1
        self.current_idx = (self.current_idx + 1) % self.total_records # Cycle through data
        
        return result
        
    def get_fairness_metrics(self) -> Dict[str, float]:
        """Compute Demographic Parity Gap."""
        rates = {}
        for group, stats in self.group_outcomes.items():
            rates[group] = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            
        if not rates:
            return {'gap_a_b': 0, 'max_gap': 0}
            
        gap_a_b = abs(rates.get('Group A', 0) - rates.get('Group B', 0))
        all_rates = list(rates.values())
        max_gap = max(all_rates) - min(all_rates) if all_rates else 0
        
        return {
            'rates': rates,
            'gap_a_b': gap_a_b,
            'max_gap': max_gap
        }
