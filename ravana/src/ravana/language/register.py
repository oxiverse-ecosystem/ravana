"""Register controller — prefrontal register modulation for controlled generation.

Roadmap #12: Prefrontal cortex sets 'registers' that modulate language production.
Three knobs: formality, verbosity, certainty.
Learned from user feedback via REINFORCE-style policy gradient.
"""

from typing import Dict, Optional, Tuple
import numpy as np


class RegisterController:
    """Controls language generation register through three modulation knobs.

    Neuroscience: Prefrontal cortex maintains task-set 'registers' that bias
    downstream language production (Hymes 1974, register theory).

    Knobs:
    - formality (0-1):  0=slang/casual, 1=formal/academic
    - verbosity (0-1):  0=terse, 1=elaborate
    - certainty (0-1):  0=hedged/uncertain, 1=confident/assertive
    """

    REGISTERS: Dict[str, Dict[str, float]] = {
        "casual":   {"formality": 0.2, "verbosity": 0.3, "certainty": 0.5},
        "didactic": {"formality": 0.7, "verbosity": 0.8, "certainty": 0.9},
        "terse":    {"formality": 0.5, "verbosity": 0.1, "certainty": 0.6},
        "formal":   {"formality": 0.9, "verbosity": 0.6, "certainty": 0.8},
        "empathetic": {"formality": 0.3, "verbosity": 0.7, "certainty": 0.3},
    }

    def __init__(self, default_register: str = "casual"):
        if default_register not in self.REGISTERS:
            default_register = "casual"
        self.current_register_name = default_register
        self.knobs: Dict[str, float] = dict(self.REGISTERS[default_register])
        self.history: list = []

    def set_register(self, name: str) -> bool:
        if name in self.REGISTERS:
            self.current_register_name = name
            self.knobs = dict(self.REGISTERS[name])
            return True
        return False

    def get_register(self) -> str:
        return self.current_register_name

    def get_knobs(self) -> Dict[str, float]:
        return dict(self.knobs)

    def modulate_formality(self, base: float) -> float:
        return self._modulate(base, self.knobs["formality"])

    def modulate_verbosity(self, base: float) -> float:
        return self._modulate(base, self.knobs["verbosity"])

    def modulate_certainty(self, base: float) -> float:
        return self._modulate(base, self.knobs["certainty"])

    def apply_certainty_hedge(self, text: str, confidence: float) -> str:
        """Modulate text certainty based on register and confidence.

        Instead of hardcoded hedge prefixes, this now works through the
        free-energy-driven SurfaceRealizer's epistemic frame system.
        The surface realizer handles hedging dynamically based on the
        actual confidence of the discourse context, not a separate
        post-processing step.

        Args:
            text: The generated response text
            confidence: Confidence score (0-1) from cognitive state

        Returns:
            Unmodified text — hedging is handled by the SurfaceRealizer
            during generation, not as a post-processing step.
        """
        # Hedging is now handled entirely by the SurfaceRealizer's
        # free-energy-driven epistemic frame system during generation.
        # This method is kept as a no-op for backward compatibility.
        return text

    def _modulate(self, base: float, knob: float) -> float:
        return base * (0.5 + 0.5 * knob)

    def adapt_from_feedback(self, reward: float, lr: float = 0.05):
        """REINFORCE-style policy gradient adaptation from user feedback.

        Positive reward pushes knobs toward current register values.
        Negative reward pushes knobs away (user doesn't like this style).
        """
        for k in self.knobs:
            target = self.REGISTERS[self.current_register_name][k]
            delta = lr * reward * (target - self.knobs[k])
            self.knobs[k] = np.clip(self.knobs[k] + delta, 0.0, 1.0)

    def get_state(self) -> dict:
        return {
            'current_register': self.current_register_name,
            'knobs': dict(self.knobs),
            'history': list(self.history),
        }

    def set_state(self, state: dict):
        self.current_register_name = state.get('current_register', 'casual')
        self.knobs = dict(state.get('knobs', self.REGISTERS.get(self.current_register_name, {})))
        self.history = list(state.get('history', []))
