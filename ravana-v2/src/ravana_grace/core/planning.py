"""
RAVANA v2 — PHASE D.5: Micro-Planning Layer
Future simulation: From "I feel stable" → "If I explore, I'll reach better."

PRINCIPLE: Intent based on predicted outcomes, not just current satisfaction.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

from .strategy import ExplorationMode, BehavioralContext


@dataclass
class SimulatedFuture:
    """Result of forward simulation."""
    dissonance_trajectory: list
    identity_trajectory: list
    clamp_risk: float  # Probability of clamp in horizon
    terminal_score: float  # Value at horizon
    steps: int


@dataclass  
class PlanningConfig:
    """Configuration for micro-planning."""
    horizon: int = 5  # Steps to simulate
    dissonance_drift_per_step: float = 0.02  # Natural drift
    clamp_threshold: float = 0.15  # Clamps per 20 steps = crisis
    
    # Mode-specific deltas (physics-based policy simulation)
    mode_deltas: Dict[ExplorationMode, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.mode_deltas is None:
            self.mode_deltas = {
                ExplorationMode.EXPLORE_AGGRESSIVE: {
                    'dissonance': 0.08,   # High change
                    'identity': 0.01,
                    'noise': 0.04,
                },
                ExplorationMode.EXPLORE_SAFE: {
                    'dissonance': 0.03,
                    'identity': 0.02,
                    'noise': 0.02,
                },
                ExplorationMode.STABILIZE: {
                    'dissonance': 0.005,
                    'identity': 0.03,
                    'noise': 0.005,
                },
                ExplorationMode.RECOVER: {
                    'dissonance': -0.1,   # Force reduction
                    'identity': 0.05,
                    'noise': 0.0,
                },
            }


class MicroPlanner:
    """
    🎯 Micro-Planning: Simulate before choosing.
    
    Instead of reacting to current state, simulates forward
    and chooses based on predicted outcomes.
    
    This is the jump from reactive → anticipatory goals.
    """
    
    def __init__(self, config: Optional[PlanningConfig] = None):
        self.config = config or PlanningConfig()
    
    def simulate_forward(
        self,
        context: BehavioralContext,
        mode: ExplorationMode,
        steps: Optional[int] = None
    ) -> SimulatedFuture:
        """
        Simulate running in mode for N steps.
        
        Returns predicted trajectory and terminal state.
        """
        steps = steps or self.config.horizon
        
        # Get mode parameters
        mode_params = self.config.mode_deltas.get(mode, {})
        d_delta = mode_params.get('dissonance', 0.0)
        i_delta = mode_params.get('identity', 0.02)
        noise = mode_params.get('noise', 0.02)
        
        # Initial state
        d = context.dissonance
        i = context.identity
        
        # Trajectory tracking
        d_traj = [d]
        i_traj = [i]
        clamp_count = 0
        
        # Simulate steps
        for _ in range(steps):
            # Apply mode deltas
            d += d_delta + np.random.normal(0, noise)
            i += i_delta + np.random.normal(0, noise * 0.5)
            
            # Hard bounds
            d = np.clip(d, 0.15, 0.95)
            i = np.clip(i, 0.10, 0.95)
            
            # Track clamp risk (near bounds = risk)
            if d > 0.90 or d < 0.20 or i < 0.15:
                clamp_count += 1
            
            d_traj.append(d)
            i_traj.append(i)
        
        # Compute metrics
        clamp_risk = clamp_count / steps
        terminal_d = d_traj[-1]
        terminal_i = i_traj[-1]
        
        # Terminal score: sweet spot for dissonance, high for identity
        d_score = 1.0 - abs(terminal_d - 0.50) / 0.50  # Peak at 0.5
        i_score = terminal_i  # Higher is better
        terminal_score = 0.6 * d_score + 0.4 * i_score
        
        return SimulatedFuture(
            dissonance_trajectory=d_traj,
            identity_trajectory=i_traj,
            clamp_risk=clamp_risk,
            terminal_score=terminal_score,
            steps=steps
        )
    
    def score_future(
        self,
        context: BehavioralContext,
        future: SimulatedFuture
    ) -> float:
        """
        Score a simulated future.
        
        Combines:
        - Terminal value (where do we end)
        - Clamp risk (probability of violation)
        - Stability (low variance = predictable)
        """
        # Terminal value (0-1)
        value_score = future.terminal_score
        
        # Clamp penalty (exponential: small risk OK, high risk terrible)
        clamp_penalty = 1.0 - np.exp(-5 * future.clamp_risk)
        
        # Stability bonus (low trajectory variance)
        d_variance = np.var(future.dissonance_trajectory)
        stability_bonus = np.exp(-10 * d_variance)  # 1.0 if stable, decay if volatile
        
        # Combined score
        score = (
            0.5 * value_score +
            0.3 * (1.0 - clamp_penalty) +
            0.2 * stability_bonus
        )
        
        return score
    
    def plan_and_select(
        self,
        context: BehavioralContext,
        available_modes: list = None
    ) -> tuple:
        """
        Full planning: simulate all modes, select best.
        
        Returns: (best_mode, predictions_dict, scores_dict)
        """
        if available_modes is None:
            available_modes = list(ExplorationMode)
        
        predictions = {}
        scores = {}
        
        for mode in available_modes:
            # Simulate
            future = self.simulate_forward(context, mode)
            predictions[mode] = future
            
            # Score
            score = self.score_future(context, future)
            scores[mode] = score
        
        # Select best
        best_mode = max(scores.keys(), key=lambda m: scores[m])
        
        return best_mode, predictions, scores
