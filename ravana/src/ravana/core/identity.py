"""
Identity Engine for RAVANA.

Tracks the agent's sense of self-coherence and continuity.
Based on: Friston's Free Energy Principle, identity as a generative model
that minimizes surprise about self-generated predictions.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class IdentityState:
    """Current identity state."""
    strength: float = 0.25  # 0 (fragmented) to 1 (coherent)
    momentum: float = 0.0   # rate of change
    history: list = None    # recent strength values

    def __post_init__(self):
        if self.history is None:
            self.history = []

    def to_dict(self) -> dict:
        return {
            'strength': self.strength,
            'momentum': self.momentum,
            'history': self.history[-10:] if len(self.history) > 10 else self.history
        }


@dataclass
class IdentityConfig:
    """Configuration for identity dynamics."""
    initial_strength: float = 0.25
    momentum_factor: float = 0.3
    recovery_bias: float = 0.15
    min_strength: float = 0.0
    max_strength: float = 1.0


class IdentityEngine:
    """Identity generator: maintains self-coherence through prediction error minimization."""

    def __init__(self, config: Optional[IdentityConfig] = None):
        self.config = config or IdentityConfig()
        self.state = IdentityState(strength=self.config.initial_strength)
        self.last_delta = 0.0

    def compute_update(self, resolution_delta: float, resolution_success: bool,
                       regulated_identity_delta: float, current_dissonance: float,
                       resolution_streak: int, correctness: bool) -> float:
        """Compute identity update from cognitive signals."""
        cfg = self.config

        # Resolution success: positive delta boosts identity
        resolution_signal = 1.0 if resolution_success else -0.5

        # Consistency bonus: streak of successful resolutions
        streak_bonus = min(0.05 * resolution_streak, 0.15)

        # Dissonance penalty: high dissonance erodes identity
        dissonance_penalty = current_dissonance * 0.3

        # Regulated delta from meaning engine
        regulated_signal = regulated_identity_delta

        # Momentum: persistence of previous direction
        momentum_signal = self.state.momentum * cfg.momentum_factor

        # Recovery bias: pulls toward baseline when fragmented
        recovery = cfg.recovery_bias * (cfg.initial_strength - self.state.strength)

        delta = (0.3 * resolution_signal +
                 0.2 * streak_bonus +
                 0.2 * regulated_signal +
                 0.15 * momentum_signal +
                 0.1 * recovery +
                 0.05 * (-dissonance_penalty))

        # Clamp delta
        delta = np.clip(delta, -0.2, 0.2)
        return delta

    def apply_update(self, delta: float):
        """Apply computed delta to identity state."""
        self.last_delta = delta
        self.state.momentum = delta
        self.state.strength = np.clip(
            self.state.strength + delta,
            self.config.min_strength,
            self.config.max_strength)
        self.state.history.append(self.state.strength)
        if len(self.state.history) > 50:
            self.state.history = self.state.history[-50:]

    def get_trend(self) -> float:
        """Return recent trend in identity strength."""
        if len(self.state.history) < 2:
            return 0.0
        recent = self.state.history[-5:]
        if len(recent) < 2:
            return 0.0
        return float(np.mean(np.diff(recent)))