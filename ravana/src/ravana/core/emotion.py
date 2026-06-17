"""
VAD (Valence-Arousal-Dominance) Emotion Engine for RAVANA.

Neuroscience basis:
- Russell's circumplex model (valence-arousal) + Mehrabian's dominance
- Valence: pleasantness (-1 to 1)
- Arousal: activation/alertness (0 to 1)
- Dominance: control/power (-1 to 1)

Update dynamics based on Bogacz (2017) active inference tutorial.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VADConfig:
    """Configuration for VAD emotion dynamics."""
    eta_valence: float = 0.3      # valence update rate
    eta_arousal: float = 0.4      # arousal update rate
    eta_dominance: float = 0.25   # dominance update rate
    decay_rate: float = 0.95      # return-to-baseline per turn
    baseline_arousal: float = 0.3 # engagement floor


@dataclass
class VADState:
    """Current VAD emotion state."""
    valence: float = 0.0      # -1 (unpleasant) to 1 (pleasant)
    arousal: float = 0.3      # 0 (calm) to 1 (high alert)
    dominance: float = 0.5    # -1 (submissive) to 1 (dominant)

    def to_dict(self) -> dict:
        return {'valence': self.valence, 'arousal': self.arousal, 'dominance': self.dominance}


class VADEmotionEngine:
    """Valence-Arousal-Dominance emotion engine with active inference dynamics."""

    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self.state = VADState()

    def update(self, stimulus_valence: float = 0.0, stimulus_arousal: float = 0.0,
               stimulus_dominance: float = 0.0, uncertainty: float = 0.0, dt: float = 1.0):
        """Update VAD state with stimulus and decay toward baseline."""
        cfg = self.config

        # Apply decay toward baseline
        self.state.valence *= cfg.decay_rate ** dt
        self.state.arousal = cfg.baseline_arousal + (self.state.arousal - cfg.baseline_arousal) * (cfg.decay_rate ** dt)
        self.state.dominance = 0.5 + (self.state.dominance - 0.5) * (cfg.decay_rate ** dt)

        # Integrate stimulus
        self.state.valence = np.clip(
            self.state.valence + cfg.eta_valence * stimulus_valence, -1.0, 1.0)
        self.state.arousal = np.clip(
            self.state.arousal + cfg.eta_arousal * stimulus_arousal, 0.0, 1.0)
        self.state.dominance = np.clip(
            self.state.dominance + cfg.eta_dominance * stimulus_dominance, -1.0, 1.0)

        # Uncertainty boosts arousal (surprise)
        if uncertainty > 0:
            self.state.arousal = np.clip(self.state.arousal + uncertainty * 0.3, 0.0, 1.0)

    def get_emotional_label(self) -> str:
        """Map VAD to discrete emotion label for trace/debug."""
        v, a, d = self.state.valence, self.state.arousal, self.state.dominance

        # High arousal emotions
        if a > 0.6:
            if v > 0.3:
                return "excitement" if d > 0 else "joy"
            elif v < -0.3:
                return "anger" if d > 0 else "fear"
            else:
                return "surprise"
        # Low arousal emotions
        elif a < 0.3:
            if v > 0.3:
                return "contentment"
            elif v < -0.3:
                return "sadness"
            else:
                return "calm"
        # Medium arousal
        else:
            if v > 0.3:
                return "interest" if d > 0 else "hope"
            elif v < -0.3:
                return "frustration" if d > 0 else "anxiety"
            else:
                return "neutral"