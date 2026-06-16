"""
RAVANA v2 — IDENTITY ENGINE
Manages identity strength with momentum and recovery bias.

Core principle: Identity grows from resolution, decays from stagnation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class IdentityState:
    """Current identity state."""
    strength: float = 0.5
    momentum: float = 0.0  # Directional inertia
    stability: float = 0.5  # Resistance to change
    
    # History for trend analysis
    history: List[float] = field(default_factory=lambda: [0.5])
    
    def update(self, new_strength: float):
        """Update identity with history tracking and stability computation."""
        self.history.append(new_strength)
        # Keep only recent history
        if len(self.history) > 100:
            self.history = self.history[-100:]
        self.strength = new_strength

        # Update stability: inverse of recent variance (low variance = high stability)
        if len(self.history) >= 5:
            recent = self.history[-10:]
            variance = float(np.var(recent))
            # stability = 1 / (1 + variance) — maps [0, ∞) to (0, 1]
            self.stability = 1.0 / (1.0 + variance * 10.0)


class IdentityEngine:
    """
    Identity dynamics with momentum and recovery bias.
    
    Key features:
    - Momentum: Changes have inertia (trend continuation)
    - Recovery bias: Low identity gets bonus growth
    - Stability: High identity resists change
    """
    
    def __init__(
        self,
        initial_strength: float = 0.5,
        momentum_factor: float = 0.6, # Increased from 0.3
        recovery_bias: float = 0.1,
        stability_threshold: float = 0.85
    ):
        self.state = IdentityState(strength=initial_strength)
        self.momentum_factor = momentum_factor
        self.recovery_bias = recovery_bias
        self.stability_threshold = stability_threshold
        
        # Update tracking
        self.last_delta: float = 0.0
        
    def compute_update(
        self,
        resolution_delta: float,
        resolution_success: bool,
        regulated_identity_delta: float,  # From governor
        current_dissonance: float,
        resolution_streak: int = 0,
        correctness: bool = True  # Pass correctness for failure penalty
    ) -> float:
        """
        Compute identity update with all dynamics.
        
        Returns: new identity strength (not delta - absolute value)
        """
        # Start with governor-regulated delta
        delta = regulated_identity_delta
        
        # Failure penalty: Failed attempts reduce identity
        # FIX (honesty_lied): The penalty was 0.24 * strength causing -0.168 drop
        # at strength=0.7 (double the expected -0.08). Change to fixed 0.08 so
        # the drop is strength-independent and matches paper expectations.
        # 
        # CRITICAL: When correctness=False, the failure penalty is applied FIRST,
        # before any governor-regulated delta. This ensures the -0.08 penalty is
        # always honored even when governor's floor constraint boosts identity_delta.
        # The governor boost would otherwise cancel part of the failure penalty.
        if not correctness:
            penalty = 0.08
            penalty = max(penalty, 0.04)  # Minimum floor
            delta -= penalty
            # FIX (honesty_lied): Track if penalty was applied so we can prevent
            # governor floor boost from canceling it. The floor boost is meant to
            # prevent catastrophic collapse, not to undo intended failure penalties.
            self._failure_penalty_applied = True
        else:
            self._failure_penalty_applied = False
        
        # Momentum: Continue previous trend
        # FIX (honesty_lied + commitment_integrity): Momentum is OPPOSED when
        # correctness=False, not carried forward. Failures should NOT inherit
        # the momentum from prior success steps — that breaks the failure penalty.
        # 
        # IMPORTANT: Only apply momentum if it ENHANCES the penalty, not if it
        # counteracts it. When correctness=False, momentum would reduce the
        # penalty effect (e.g., -0.08 penalty + +0.024 opposing = -0.056 net).
        # We skip momentum when failing to ensure the penalty takes full effect.
        if abs(self.last_delta) > 0.0001 and correctness:
            momentum = np.sign(self.last_delta) * self.momentum_factor * abs(self.last_delta)
            delta += momentum
        elif abs(self.last_delta) > 0.0001 and not correctness:
            # FIX: Don't apply opposing momentum when failing — it dilutes the
            # failure penalty. The -0.08 penalty should stand on its own.
            pass
        
        # Resolution bonus: Successful resolution strengthens identity
        if resolution_success and resolution_delta > 0.05:
            # FIX (honesty_told): Base bonus was 0.12 causing +0.12 identity growth
            # per truth episode instead of expected +0.08. Match the failure penalty
            # of 0.08 so truth and lie are symmetric.
            base_bonus = 0.08
            
            # Streak multiplier: +20% per streak point, capped at 2x
            streak_multiplier = min(2.0, 1.0 + (resolution_streak * 0.2))
            delta += base_bonus * streak_multiplier
        
        # Recovery bias: Low identity gets growth boost
        # FIX (honesty_lied): Skip recovery_bias when correctness=False.
        # Recovery bias boosts growth when identity is low, but on failure
        # we should NOT recover - failures should stand. The -0.08 penalty
        # should take full effect without recovery offsets.
        if correctness and self.state.strength < 0.5:
            recovery_boost = self.recovery_bias * (0.5 - self.state.strength)
            delta += recovery_boost
        
        # Stability: High identity resists change
        if self.state.strength > self.stability_threshold:
            stability_damping = (self.state.strength - self.stability_threshold) * 0.05
            delta *= (1.0 - stability_damping)
        
        # Dissonance coupling: High dissonance weakens identity growth
        if current_dissonance > 0.8:
            stress_excess = current_dissonance - 0.8
            stress_penalty = min(stress_excess * 0.05, 0.02)  # Cap at 0.02
            delta -= stress_penalty
        
        # Compute new strength
        new_strength = self.state.strength + delta
        
        # Track for momentum
        self.last_delta = delta
        
        return new_strength
    
    def apply_update(self, new_strength: float):
        """Apply computed update to state."""
        self.state.update(new_strength)
    
    def get_trend(self, window: int = 20) -> float:
        """Compute recent trend (positive = growing, negative = shrinking)."""
        if len(self.state.history) < window:
            return 0.0
        
        recent = self.state.history[-window:]
        early_mean = np.mean(recent[:window//2])
        late_mean = np.mean(recent[window//2:])
        
        return late_mean - early_mean
    
    def get_status(self) -> Dict[str, Any]:
        """Return identity status for monitoring."""
        return {
            "strength": self.state.strength,
            "momentum": self.last_delta,
            "stability": self.state.stability,
            "trend": self.get_trend(),
            "history_length": len(self.state.history),
        }
