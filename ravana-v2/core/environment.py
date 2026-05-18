"""
RAVANA v2 — PHASE E: Non-Stationary Environment
Open world with hidden dynamics. RAVANA must discover, not be told.

PRINCIPLE: Intelligence grows from dealing with the unexpected.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class HiddenDynamics(Enum):
    """Types of hidden environmental changes."""
    BOUNDARY_SHIFT = "boundary_shift"      # Constraints move
    NOISE_DRIFT = "noise_drift"            # Noise pattern changes
    GOAL_FLIP = "goal_flip"                # What "good" means changes
    DIFFICULTY_CYCLE = "difficulty_cycle"  # Hidden periodic pattern


@dataclass
class WorldState:
    """Current state of the external environment."""
    # Observable (RAVANA can infer these)
    effective_boundary: float = 0.95       # Current dissonance ceiling
    noise_level: float = 0.0               # Current environmental noise
    difficulty_level: float = 0.5          # Current task difficulty
    
    # Hidden (RAVANA cannot directly see these)
    goal_flip_bias: float = 0.0            # Shift in success criteria
    underlying_regime: str = "stable"      # Current environmental regime
    cycle_phase: float = 0.0               # Position in hidden cycle
    
    def to_observable(self) -> Dict[str, float]:
        """Return only what RAVANA can observe."""
        return {
            'dissonance': self.effective_boundary,  # Through consequences
            'noise': self.noise_level,              # Through variance
            'difficulty': self.difficulty_level     # Through outcomes
        }


@dataclass
class EnvironmentConfig:
    """Configuration for non-stationary environment."""
    # Boundary dynamics
    boundary_shift_frequency: int = 500      # Episodes between boundary shifts
    boundary_shift_magnitude: float = 0.15  # How much boundaries move
    
    # Noise dynamics  
    noise_drift_rate: float = 0.02         # Rate of noise pattern drift
    noise_random_walk_scale: float = 0.01  # Step size for noise walk
    
    # Goal dynamics
    goal_flip_period: int = 400            # Episodes between goal flips
    goal_flip_duration: int = 100            # How long flip lasts
    
    # Hidden cycles
    difficulty_cycle_period: int = 300     # Hidden periodic pattern
    cycle_amplitude: float = 0.2           # Cycle strength


class NonStationaryEnvironment:
    """
    🌍 NON-STATIONARY ENVIRONMENT
    
    A world that changes in ways RAVANA must discover.
    
    Hidden dynamics:
    1. Boundaries shift periodically (every N episodes)
    2. Noise follows random walk (non-stationary)
    3. Goals flip periodically (what "good" means changes)
    4. Hidden difficulty cycle (discoverable pattern)
    
    RAVANA is NOT told when these happen.
    It must detect them through consequences.
    """
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()
        self.current_state = WorldState()
        
        # Hidden state tracking
        self.episode_count: int = 0
        self.boundary_base: float = 0.95
        self.noise_walk: float = 0.0
        self.goal_flip_active: bool = False
        self.goal_flip_end: int = 0
        
        # History for analysis (RAVANA cannot see this)
        self.dynamics_history: List[Dict] = []
        
    def step(self, episode: int) -> WorldState:
        """
        Advance environment one step.
        Returns: WorldState (RAVANA observes consequences, not this directly)
        """
        self.episode_count = episode
        
        # 1. BOUNDARY SHIFT: Periodic constraint movement
        if episode > 0 and episode % self.config.boundary_shift_frequency == 0:
            shift = np.random.choice([-1, 1]) * self.config.boundary_shift_magnitude
            self.boundary_base = np.clip(self.boundary_base + shift, 0.7, 0.99)
            self._record_dynamic(HiddenDynamics.BOUNDARY_SHIFT, {
                'new_boundary': self.boundary_base,
                'shift': shift
            })
        
        # 2. NOISE DRIFT: Random walk pattern
        self.noise_walk += np.random.normal(0, self.config.noise_random_walk_scale)
        self.noise_walk = np.clip(self.noise_walk, -0.1, 0.1)
        noise_level = abs(self.noise_walk) + 0.02  # Base noise + drift
        
        # 3. GOAL FLIP: Periodic reversal of what's "good"
        if episode > 0 and episode % self.config.goal_flip_period == 0:
            self.goal_flip_active = True
            self.goal_flip_end = episode + self.config.goal_flip_duration
            self._record_dynamic(HiddenDynamics.GOAL_FLIP, {
                'flip_start': episode,
                'flip_end': self.goal_flip_end,
                'bias': -0.3 if np.random.random() < 0.5 else 0.3
            })
        
        if self.goal_flip_active and episode >= self.goal_flip_end:
            self.goal_flip_active = False
        
        goal_bias = self.dynamics_history[-1].get('bias', 0) if self.dynamics_history and self.dynamics_history[-1]['type'] == HiddenDynamics.GOAL_FLIP else 0
        
        # 4. HIDDEN DIFFICULTY CYCLE: Sinusoidal pattern
        cycle_phase = (episode % self.config.difficulty_cycle_period) / self.config.difficulty_cycle_period
        cycle_value = np.sin(2 * np.pi * cycle_phase) * self.config.cycle_amplitude
        difficulty = 0.5 + cycle_value + np.random.normal(0, 0.05)
        difficulty = np.clip(difficulty, 0.2, 0.9)
        
        # Update current state
        self.current_state = WorldState(
            effective_boundary=self.boundary_base,
            noise_level=noise_level,
            difficulty_level=difficulty,
            goal_flip_bias=goal_bias if self.goal_flip_active else 0.0,
            underlying_regime=self._detect_regime(),
            cycle_phase=cycle_phase
        )
        
        return self.current_state
    
    def _record_dynamic(self, dynamic_type: HiddenDynamics, data: Dict):
        """Record hidden dynamic for post-hoc analysis."""
        self.dynamics_history.append({
            'episode': self.episode_count,
            'type': dynamic_type,
            **data
        })
    
    def _detect_regime(self) -> str:
        """Internal regime detection (not visible to RAVANA)."""
        if self.goal_flip_active:
            return "goal_flipped"
        elif self.noise_walk > 0.05:
            return "high_noise"
        elif self.boundary_base < 0.85:
            return "tight_constraints"
        else:
            return "normal"
    
    def get_hidden_truth(self) -> Dict[str, Any]:
        """
        Reveal hidden dynamics for analysis.
        (Call this AFTER RAVANA runs, not during)
        """
        return {
            'total_episodes': self.episode_count,
            'dynamics_history': self.dynamics_history,
            'final_boundary': self.boundary_base,
            'final_noise_walk': self.noise_walk,
            'regime_transitions': len(self.dynamics_history)
        }


class WorldModelEvaluator:
    """
    📊 WORLD MODEL EVALUATOR
    
    Compares RAVANA's inferred world model against ground truth.
    """
    
    def __init__(self, environment: NonStationaryEnvironment):
        self.env = environment
        self.ravana_beliefs: List[Dict] = []
        
    def record_ravana_belief(self, episode: int, belief: Dict):
        """Record what RAVANA believed at this episode."""
        self.ravana_beliefs.append({
            'episode': episode,
            'believed_boundary': belief.get('believed_boundary'),
            'inferred_noise': belief.get('inferred_noise_pattern'),
            'detected_regime': belief.get('detected_regime')
        })
    
    def evaluate_model_accuracy(self) -> Dict[str, Any]:
        """
        Evaluate how well RAVANA's world model matched reality.
        """
        if not self.ravana_beliefs or not self.env.dynamics_history:
            return {"status": "insufficient_data"}
        
        # Boundary detection accuracy
        boundary_errors = []
        for belief in self.ravana_beliefs:
            ep = belief['episode']
            # Find actual boundary at this episode
            actual = self._get_actual_boundary(ep)
            if actual and belief['believed_boundary']:
                error = abs(actual - belief['believed_boundary'])
                boundary_errors.append(error)
        
        mean_error = np.mean(boundary_errors) if boundary_errors else 1.0
        
        # Anomaly detection: how many true anomalies did RAVANA catch?
        true_anomalies = {d['episode'] for d in self.env.dynamics_history}
        detected_anomalies = {b['episode'] for b in self.ravana_beliefs 
                             if b.get('detected_regime') not in ['unknown', 'stable']}
        
        true_positives = len(true_anomalies & detected_anomalies)
        false_positives = len(detected_anomalies - true_anomalies)
        
        return {
            "boundary_model_error": round(mean_error, 3),
            "true_anomalies": len(true_anomalies),
            "detected_anomalies": len(detected_anomalies),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "detection_recall": round(true_positives / max(1, len(true_anomalies)), 3),
            "detection_precision": round(true_positives / max(1, len(detected_anomalies)), 3),
            "assessment": "🟢 GOOD" if mean_error < 0.05 and true_positives > 0 else \
                         "🟡 PARTIAL" if mean_error < 0.1 else \
                         "🔴 POOR"
        }
    
    def _get_actual_boundary(self, episode: int) -> Optional[float]:
        """Get actual boundary at episode."""
        # Start from base and apply shifts
        boundary = 0.95
        for dynamic in self.env.dynamics_history:
            if dynamic['episode'] <= episode and dynamic['type'] == HiddenDynamics.BOUNDARY_SHIFT:
                boundary = dynamic.get('new_boundary', boundary)
        return boundary


if __name__ == "__main__":
    # Quick test
    env = NonStationaryEnvironment()
    
    print("🧪 Environment Test: 1000 episodes")
    for ep in range(0, 1001, 100):
        state = env.step(ep)
        if ep % 500 == 0 and ep > 0:
            print(f"  EP{ep}: boundary={state.effective_boundary:.2f}, "
                  f"noise={state.noise_level:.3f}, regime={state.underlying_regime}")
    
    truth = env.get_hidden_truth()
    print(f"\n📊 Hidden Dynamics: {truth['regime_transitions']} transitions")
    for d in truth['dynamics_history']:
        print(f"  EP{d['episode']}: {d['type'].value}")
