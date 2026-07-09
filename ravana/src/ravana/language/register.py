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

        Kept for backward compatibility. The live path now routes through
        ``compose`` (which folds the register's certainty knob into the text
        only when it deviates from neutral, so it never double-hedges the
        SurfaceRealizer's own free-energy-driven epistemic frame).
        """
        return text

    # Epistemic markers the SurfaceRealizer may already have prepended. Used so
    # the register certainty knob never stacks a second hedge on top of one the
    # realizer already produced.
    _EPISTEMIC_MARKERS = (
        "i think", "maybe", "perhaps", "it seems", "i suspect", "i believe",
        "from what i understand", "it appears",
    )

    def apply_affective_state(self, vad_state, relationship_depth: float = 0.0,
                              conversation_depth: float = 0.0,
                              uncertainty: float = 0.0) -> None:
        """Couple VAD + relationship state into the register knobs.

        This is the missing link the brief calls out: nothing previously read
        ``VADEmotionEngine.state`` and set the register. Wiring is heuristic
        (no LLM, no hard thresholds — monotonic gains around neutral 0.5):

        - valence  > 0.2  -> more cooperative / direct (nudge certainty up,
                             formality up slightly)
        - valence  < -0.2  -> soften (hedge: certainty down)
        - arousal  > 0.6   -> compress to the point (verbosity down, urgency)
        - arousal  < 0.3   -> allow elaboration (verbosity up slightly)
        - relationship_depth high -> ellipsis/terse (verbosity down): close
          rapport is shorter and more elliptical
        - conversation_depth / uncertainty high -> explain more (verbosity up)

        Knobs are moved toward the named register base, not replaced, so the
        REINFORCE adaptation in :meth:`adapt_from_feedback` stays meaningful.
        """
        base = dict(self.REGISTERS.get(self.current_register_name,
                                       {"formality": 0.5, "verbosity": 0.3,
                                        "certainty": 0.5}))
        k = self.knobs
        v = getattr(vad_state, "valence", 0.0)
        a = getattr(vad_state, "arousal", 0.3)

        # Valence -> directness / softening
        vshift = 0.15 * np.tanh(v * 2.0)          # +/-~0.15 around 0
        # Arousal -> brevity under urgency, slack when calm
        ashift = -0.20 * np.tanh((a - 0.6) * 3.0)  # <0.6 gives positive slack
        # Relationship -> closer = terser
        rshift = -0.25 * np.clip(relationship_depth, 0.0, 1.0)
        # Depth/uncertainty -> explain more
        dshift = 0.20 * np.clip(conversation_depth + uncertainty, 0.0, 1.0)

        k["formality"] = float(np.clip(base["formality"] + vshift * 0.5, 0.0, 1.0))
        k["verbosity"] = float(np.clip(
            base["verbosity"] + ashift + rshift + dshift, 0.0, 1.0))
        k["certainty"] = float(np.clip(base["certainty"] + vshift, 0.0, 1.0))
        self.history.append(("affective", dict(k)))

    def compose(self, text: str, base_confidence: float = 0.5) -> str:
        """Apply the register knobs to finished text (deviation-only).

        Only acts when a knob deviates enough from its neutral value to matter,
        and never hedges a response the SurfaceRealizer already hedged.
        """
        if not text:
            return text
        k = self.knobs
        verbosity = k["verbosity"]
        certainty = k["certainty"]
        formality = k["formality"]

        # 1) Verbosity -> truncate to first sentence when terse.
        if verbosity < 0.30:
            first_stop = min(
                (i for i, ch in enumerate(text) if ch in ".!?"),
                default=-1)
            if first_stop > 0:
                text = text[:first_stop + 1].strip()

        # 2) Certainty -> hedge only if low AND not already hedged by the
        #    SurfaceRealizer's epistemic frame (avoids double-hedge).
        lowered = text.lower()
        already_hedged = any(lowered.startswith(m) for m in self._EPISTEMIC_MARKERS)
        if certainty < 0.40 and not already_hedged:
            text = f"i'm not certain, but {text[0].lower()}{text[1:]}"

        # 3) Formality -> casual flattening when high; no-op when low.
        if formality > 0.75:
            text = text.replace("gonna", "going to").replace("wanna", "want to")
            if text.endswith("!"):
                text = text[:-1] + "."

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
