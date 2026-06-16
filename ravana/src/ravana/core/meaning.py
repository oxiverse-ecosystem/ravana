"""
Meaning Engine for RAVANA.

Computes accumulated meaning from resolution of dissonance.
Based on: Free Energy Principle - meaning as reduction of uncertainty.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MeaningConfig:
    """Configuration for meaning dynamics."""
    w_dissonance_reduction: float = 0.3
    w_identity_coherence: float = 0.3
    w_predictive_power: float = 0.4
    effort_kappa: float = 0.5


@dataclass
class MeaningState:
    """Current meaning state."""
    accumulated_meaning: float = 0.0
    history: list = None

    def __post_init__(self):
        if self.history is None:
            self.history = []

    def to_dict(self) -> dict:
        return {'accumulated_meaning': self.accumulated_meaning, 'history': self.history[-10:]}


class MeaningEngine:
    """Meaning accumulator: meaning emerges from successful dissonance resolution."""

    def __init__(self, config: Optional[MeaningConfig] = None):
        self.config = config or MeaningConfig()
        self.state = MeaningState()

    def compute_meaning(self, episode: int, pre_dissonance: float, post_dissonance: float,
                        pre_identity: float, post_identity: float,
                        predictive_gain: float, effort: float):
        """Compute meaning increment from a cognitive episode."""
        cfg = self.config

        # Dissonance reduction (primary source of meaning)
        dissonance_reduction = max(0.0, pre_dissonance - post_dissonance)

        # Identity coherence gain
        identity_gain = max(0.0, post_identity - pre_identity)

        # Predictive power (successful predictions)
        predictive_power = predictive_gain

        # Effort discount (meaning requires effort but too much effort reduces value)
        effort_discount = np.exp(-cfg.effort_kappa * effort)

        # Weighted combination
        meaning_increment = (cfg.w_dissonance_reduction * dissonance_reduction +
                            cfg.w_identity_coherence * identity_gain +
                            cfg.w_predictive_power * predictive_power) * effort_discount

        self.state.accumulated_meaning += meaning_increment
        self.state.history.append(meaning_increment)
        if len(self.state.history) > 50:
            self.state.history = self.state.history[-50:]