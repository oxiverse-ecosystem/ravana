"""
Unified Cognitive Currency System for RAVANA.

All pressure signals that drive learning and behavior are managed here.
Previously scattered across RLM as bare attributes (identity_strength,
dissonance_ema, valence, etc.), now consolidated into one coherent module.

Currencies:
  - Identity  (strength, momentum, history)
  - Emotion   (valence, arousal, dominance)  -- VAD differential equations
  - Meaning   (accumulated, history)
  - Sleep     (pressure, threshold)
  - Dissonance (EMA of prediction error)
  - Regulation mode

Usage:
    currencies = CognitiveCurrencies()
    currencies.update(prediction_error=0.8, is_correct=False)
    state = currencies.get_state()    # for checkpointing
    currencies.load_state(state)      # for restore
"""

import numpy as np
from typing import List, Tuple, Dict, Any


class CognitiveCurrencies:
    """Unified cognitive currency system for RAVANA.

    All pressure signals that drive learning and behavior are managed here.
    Provides a single update() method, state snapshot/restore, and
    regulation logic.
    """

    def __init__(self):
        # ── Identity ──
        self.identity_strength: float = 0.5       # [0,1] — self-concept coherence
        self.identity_momentum: float = 0.0       # directional inertia
        self.identity_history: List[float] = []   # last 100 values

        # ── Emotion (VAD) ──
        self.valence: float = 0.0                 # [-1,1] — positive/negative affect
        self.arousal: float = 0.3                 # [0,1] — activation level (baseline=0.3)
        self.dominance: float = 0.5               # [0,1] — sense of control

        # ── Meaning ──
        self.accumulated_meaning: float = 0.0     # running meaning total
        self.meaning_history: List[float] = []    # last 100 values

        # ── Sleep ──
        self.sleep_pressure: float = 0.0          # [0,1] — accumulates from errors
        self.sleep_pressure_threshold: float = 0.7  # when to trigger auto-sleep

        # ── Dissonance ──
        self.dissonance_ema: float = 0.5          # exponential moving average of prediction error

        # ── Regulation ──
        self.regulation_mode: str = "NORMAL"      # NORMAL, EXPLORATION, RESOLUTION, RECOVERY, PLATEAU

    # ──────────────────────────────────────────────────────────────
    # Core Update
    # ──────────────────────────────────────────────────────────────

    def update(self, conceptual_error: float, is_correct: bool,
               meaning_gain: float = None):
        """Update all currencies based on a learning signal.

        Args:
            conceptual_error: 0.0 if correct, 1.0 if incorrect (binary).
            is_correct: whether the prediction was correct.
            meaning_gain: pre-computed meaning gain. If None, computed internally.
        """
        # 1. Dissonance EMA
        self.dissonance_ema = 0.9 * self.dissonance_ema + 0.1 * conceptual_error

        # 2. Identity update
        identity_delta = self._compute_identity_update(conceptual_error, is_correct)
        self.identity_strength = float(np.clip(self.identity_strength + identity_delta, 0.0, 1.0))
        self.identity_momentum = 0.6 * self.identity_momentum + 0.4 * identity_delta
        self.identity_history.append(self.identity_strength)
        if len(self.identity_history) > 100:
            self.identity_history.pop(0)

        # 3. Emotion update (VAD differential equations)
        valence_stimulus = -conceptual_error if is_correct else conceptual_error
        self._update_emotion(valence_stimulus, arousal_stimulus=conceptual_error)

        # 4. Meaning computation
        if meaning_gain is None:
            meaning_gain = self._compute_meaning(conceptual_error)
        self.accumulated_meaning += meaning_gain
        self.meaning_history.append(meaning_gain)
        if len(self.meaning_history) > 100:
            self.meaning_history.pop(0)

        # 5. Sleep pressure accumulation
        self.sleep_pressure = min(1.0, self.sleep_pressure
                                  + conceptual_error * 0.01
                                  + (0.005 if not is_correct else 0.0))

    # ──────────────────────────────────────────────────────────────
    # Identity Engine
    # ──────────────────────────────────────────────────────────────

    def _compute_identity_update(self, error: float, is_correct: bool) -> float:
        """Compute identity delta from prediction outcome."""
        if is_correct:
            base = 0.02 * (1.0 - error)
            if self.identity_strength < 0.5:
                base *= 1.2  # recovery bias
            if len(self.identity_history) >= 3 and all(h > 0.5 for h in self.identity_history[-3:]):
                base *= 1.3  # streak bonus
        else:
            base = -0.05  # fixed failure penalty
            if self.dissonance_ema > 0.8:
                base *= 1.3  # dissonance coupling
        # Momentum
        base += 0.3 * self.identity_momentum
        # Damping when identity is high
        if self.identity_strength > 0.85:
            base *= 0.5
        return base

    # ──────────────────────────────────────────────────────────────
    # Emotion Engine (VAD differential equations)
    # ──────────────────────────────────────────────────────────────

    def _update_emotion(self, valence_stimulus: float, arousal_stimulus: float,
                        dominance_stimulus: float = 0.0):
        """VAD differential equations.

        dV/dt = eta_v * (stimulus - V) - lambda_v * V
        dA/dt = eta_a * (stimulus + uncertainty - A) - lambda_a * (A - baseline)
        dD/dt = eta_d * (stimulus - D) - lambda_d * D
        """
        dv = 0.3 * (valence_stimulus - self.valence) - 0.1 * self.valence
        da = 0.4 * (arousal_stimulus + 0.3 * self.dissonance_ema - self.arousal) - 0.1 * (self.arousal - 0.3)
        dd = 0.25 * (dominance_stimulus - self.dominance) - 0.1 * self.dominance
        self.valence = float(np.clip(self.valence + dv, -1.0, 1.0))
        self.arousal = float(np.clip(self.arousal + da, 0.0, 1.0))
        self.dominance = float(np.clip(self.dominance + dd, 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────
    # Meaning Engine
    # ──────────────────────────────────────────────────────────────

    def _compute_meaning(self, error: float) -> float:
        """Meaning from dissonance reduction + identity gain + predictive power."""
        dissonance_reduction = max(0, self.dissonance_ema - error)
        prev_identity = self.identity_history[-2] if len(self.identity_history) >= 2 else 0.5
        identity_gain = self.identity_strength - prev_identity
        predictive_power = max(0, 1.0 - error)
        return 0.4 * dissonance_reduction + 0.3 * max(0, identity_gain) + 0.3 * predictive_power

    # ──────────────────────────────────────────────────────────────
    # Regulation (lightweight Governor)
    # ──────────────────────────────────────────────────────────────

    def regulate(self):
        """Lightweight self-regulation — prevents runaway state, detects mode."""
        # Hard constraints
        self.identity_strength = float(np.clip(self.identity_strength, 0.1, 0.95))
        self.sleep_pressure = float(np.clip(self.sleep_pressure, 0.0, 1.0))

        # Mode detection
        if self.dissonance_ema > 0.8:
            self.regulation_mode = "RECOVERY"
        elif self.dissonance_ema > 0.5:
            self.regulation_mode = "RESOLUTION"
        elif self.dissonance_ema < 0.15:
            self.regulation_mode = "EXPLORATION"
        else:
            self.regulation_mode = "NORMAL"

        # Boundary pressure (sigmoid near limits)
        if self.identity_strength > 0.85:
            overshoot = (self.identity_strength - 0.85) / 0.15
            self.identity_strength -= 0.01 * overshoot
        if self.identity_strength < 0.2:
            recovery = (0.2 - self.identity_strength) / 0.2
            self.identity_strength += 0.02 * recovery

        # Dissonance dampening
        if self.dissonance_ema > 0.9:
            self.dissonance_ema *= 0.95

    # ──────────────────────────────────────────────────────────────
    # Dissonance (normalized for reporting)
    # ──────────────────────────────────────────────────────────────

    @property
    def dissonance_normalized(self) -> float:
        """Paper-comparable dissonance in [0.1, 0.9] range."""
        return 0.1 + 0.8 * min(1.0, self.dissonance_ema / 1.5)

    # ──────────────────────────────────────────────────────────────
    # Sleep cycle consolidation
    # ──────────────────────────────────────────────────────────────

    def consolidate_on_sleep(self):
        """Called during sleep_cycle to normalize cognitive state."""
        # Arousal toward baseline
        self.arousal = 0.3 + (self.arousal - 0.3) * 0.5
        # Valence dampening
        self.valence *= 0.8
        # Dominance toward baseline
        self.dominance = 0.5 + (self.dominance - 0.5) * 0.7
        # Meaning slight decay
        self.accumulated_meaning *= 0.99
        # Identity consolidation: smooth toward recent history
        if len(self.identity_history) >= 10:
            self.identity_strength = 0.9 * self.identity_strength + 0.1 * float(np.mean(self.identity_history[-10:]))
        # Reset sleep pressure
        self.sleep_pressure = 0.0

    # ──────────────────────────────────────────────────────────────
    # State Snapshot / Restore
    # ──────────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        """Snapshot all currencies for checkpointing."""
        return {
            "identity_strength": self.identity_strength,
            "identity_momentum": self.identity_momentum,
            "identity_history": self.identity_history,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "accumulated_meaning": self.accumulated_meaning,
            "meaning_history": self.meaning_history,
            "sleep_pressure": self.sleep_pressure,
            "sleep_pressure_threshold": self.sleep_pressure_threshold,
            "regulation_mode": self.regulation_mode,
            "dissonance_ema": self.dissonance_ema,
        }

    def load_state(self, state: Dict[str, Any]):
        """Restore currencies from checkpoint."""
        self.identity_strength = state.get("identity_strength", 0.5)
        self.identity_momentum = state.get("identity_momentum", 0.0)
        self.identity_history = state.get("identity_history", [])
        self.valence = state.get("valence", 0.0)
        self.arousal = state.get("arousal", 0.3)
        self.dominance = state.get("dominance", 0.5)
        self.accumulated_meaning = state.get("accumulated_meaning", 0.0)
        self.meaning_history = state.get("meaning_history", [])
        self.sleep_pressure = state.get("sleep_pressure", 0.0)
        self.sleep_pressure_threshold = state.get("sleep_pressure_threshold", 0.7)
        self.regulation_mode = state.get("regulation_mode", "NORMAL")
        self.dissonance_ema = state.get("dissonance_ema", 0.5)
