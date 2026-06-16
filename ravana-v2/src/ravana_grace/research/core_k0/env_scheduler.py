"""
RAVANA Environment Scheduler

Dynamic reward/risk structure shifts for Transfer Efficiency testing.
"""

import numpy as np


class EnvironmentScheduler:
    """Manages environment phases with dynamic shifts."""
    
    def __init__(self, total_episodes):
        self.total_episodes = total_episodes
        self.current_phase = "stable"
        self.phase_duration = total_episodes // 6
        
    def get_phase(self, episode):
        """Determine phase from episode count."""
        phase_idx = episode // self.phase_duration
        phases = ["stable", "scarce", "stable", "volatile", "latent", "stable"]
        return phases[min(phase_idx, len(phases)-1)]
    
    def apply_shifts(self, env, episode):
        """Modify environment dynamics dynamically."""
        phase = self.get_phase(episode)
        
        if phase == "stable":
            env.reward_weights = {"accuracy": 0.5, "fairness": 0.5}
            env.risk_model = "standard"
            env.cognitive_load_pressure = 0.3
            
        elif phase == "scarce":
            # Resource scarcity increases cognitive load
            env.reward_weights = {"accuracy": 0.8, "fairness": 0.2}
            env.cognitive_load_pressure = 0.8 
            env.risk_model = "high_stakes"
            
        elif phase == "volatile":
            # Rapid changes test transfer efficiency
            env.reward_weights = {
                "accuracy": np.random.uniform(0.3, 0.7), 
                "fairness": np.random.uniform(0.3, 0.7)
            }
            env.risk_model = "unpredictable"
            env.cognitive_load_pressure = 0.6
            
        elif phase == "latent":
            # Hidden biases emerge (demographic parity test)
            env.demographic_bias_factor = 0.5
            env.risk_model = "biased_history"
            env.cognitive_load_pressure = 0.7
            
        return phase
    
    def get_state(self):
        """Get current scheduler state."""
        return {
            'current_regime': self.current_phase,
            'phase_duration': self.phase_duration,
            'noise_level': 0.1 if self.current_phase == 'stable' else 0.3
        }
    
    def step(self, episode):
        """Update scheduler state for episode."""
        self.current_phase = self.get_phase(episode)
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_phase = 'stable'
