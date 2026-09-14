"""
Neuromodulator System — Dopamine, Acetylcholine, Noradrenaline, Serotonin.

Neuroscience basis:
- VTA dopamine encodes reward prediction error (Schultz 1997): phasic bursts
  for better-than-expected outcomes, dips for worse. This is the "teaching
  signal" that tells the system what to consolidate and at what strength.
- Basal forebrain acetylcholine gates cortical plasticity (Hasselmo 2006):
  high ACh = encoding mode (learn new things); low ACh = recall mode
  (use existing knowledge). Also sharpens signal-to-noise via lateral
  inhibition.
- Locus coeruleus noradrenaline modulates gain (Aston-Jones & Cohen 2005):
  unexpected uncertainty raises NE → broadens attention, speeds unlearning
  of outdated predictions.
- Raphe serotonin stabilizes updating (Cools et al. 2011): high 5-HT =
  conservative, resistant to new (potentially noisy) evidence; low 5-HT =
  volatile but adaptable.

In RAVANA:
  - Dopamine -> multiplies the Hebbian learning rate and memory encoding weight
  - Acetylcholine -> binary(ish) gate: turns learning ON/OFF per-turn AND
    controls how many edges get updated (signal-to-noise sharpening)
  - Noradrenaline -> gain multiplier on the dopamine signal (amplifies the
    teaching signal when surprise is high)
  - Serotonin -> stability anchor: damps learning rate volatility over time,
    preventing runaway reinforcement

All four levels are SEEDABLE (legitimate prior structure) and adapted ONLINE
from real engine signals: prediction error, novelty, valence, arousal, and
resolution success. No authored reply strings. No if/elif reply pools.
"""

import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ModulatorState:
    """Snapshot of all four neuromodulator levels at a point in time."""
    dopamine: float       # 0..1  (VTA teaching signal)
    acetylcholine: float  # 0..1  (plasticity gate)
    noradrenaline: float  # 0..1  (gain / arousal)
    serotonin: float      # 0..1  (stability anchor)

    # Derived control signals
    effective_lr_mult: float = 1.0   # multiplier applied to base_lr
    plasticity_open: bool = True     # whether learning is allowed this turn
    learning_sharpening: float = 0.5 # 0..1, higher = fewer edges updated


@dataclass
class NeuromodulatorConfig:
    # Baseline tonic levels (seeded, adapted online)
    dopamine_baseline: float = 0.5
    acetylcholine_baseline: float = 0.5
    noradrenaline_baseline: float = 0.3
    serotonin_baseline: float = 0.5

    # ACh gate threshold: above this -> plasticity open
    ach_gate_threshold: float = 0.25

    # Serotonin damping factor: higher = more stability (slower adaptation)
    serotonin_damping: float = 0.7

    # Dopamine ceiling (prevent runaway positive feedback)
    dopamine_max: float = 0.95

    # Learning rate multiplier bounds
    lr_mult_min: float = 0.1   # floor: never fully silent
    lr_mult_max: float = 3.0   # ceiling: cap learning speed

    # History window for adaptation (exponential moving average)
    ema_alpha: float = 0.15


class NeuromodulatorSystem:
    """Four-pathway neuromodulator controller for RAVANA's learning system.

    Computes tonic neuromodulator levels per turn from engine signals and
    exposes derived control signals (effective learning rate multiplier,
    plasticity gate, learning sharpening).
    """

    def __init__(self, config: Optional[NeuromodulatorConfig] = None):
        self.config = config or NeuromodulatorConfig()

        # Current tonic levels (seeded from config, adapted online)
        self.dopamine = self.config.dopamine_baseline
        self.acetylcholine = self.config.acetylcholine_baseline
        self.noradrenaline = self.config.noradrenaline_baseline
        self.serotonin = self.config.serotonin_baseline

        # History for adaptation EMA
        self._dopamine_history: list = []
        self._ach_history: list = []
        self._ne_history: list = []
        self._ser_history: list = []

        # Running statistics (for adaptive baselines)
        self._pe_mean = 0.0
        self._pe_var = 0.1
        self._novelty_mean = 0.0

        # Turn counter
        self.turns_processed = 0

    def update(self,
               prediction_error: float = 0.0,
               novelty: float = 0.0,
               valence: float = 0.0,
               arousal: float = 0.3,
               resolution_success: bool = True,
               surprise: float = 0.0,
               attention_focus: float = 0.5) -> ModulatorState:
        """Compute neuromodulator levels for this turn from real signals.

        All inputs are 0..1 normalized floats from the engine state:
          - prediction_error: current PE from curiosity engine
          - novelty: graph novelty signal (new edges / dormant edges reactivated)
          - valence: VAD valence (-1..1, normalized to 0..1 internally)
          - arousal: VAD arousal (0..1)
          - resolution_success: whether the turn produced a coherent response
          - surprise: unexpected reward signal (from user surprise / novelty)
          - attention_focus: working memory load / PFC focus (0..1)

        Returns:
            ModulatorState with all four levels + derived control signals.
        """
        cfg = self.config
        alpha = cfg.ema_alpha

        # --- 1. Dopamine (VTA): reward prediction error ---
        # Phasic burst for positive PE (better than expected), dip for negative.
        # Surprise amplifies the signal (unexpected outcomes matter more).
        # Valence modulates: positive outcomes -> more dopamine.
        pe_clipped = np.clip(prediction_error, 0.0, 1.0)
        valence_boost = 0.5 + 0.5 * np.clip(valence, -1.0, 1.0)  # 0..1
        surprise_boost = 1.0 + surprise  # >= 1.0

        # Target dopamine: high when PE is high AND valence is positive AND surprise is high
        dopamine_target = pe_clipped * valence_boost * surprise_boost
        dopamine_target = min(dopamine_target, cfg.dopamine_max)

        # Negative PE -> dopamine dip (but never below 0)
        if prediction_error < 0.1 and surprise < 0.2:
            dopamine_target *= 0.5  # low signal when nothing surprising happened

        # EMA adaptation (smoothed)
        self.dopamine = (1 - alpha) * self.dopamine + alpha * dopamine_target
        self.dopamine = np.clip(self.dopamine, 0.0, cfg.dopamine_max)

        # --- 2. Acetylcholine (basal forebrain): plasticity gate ---
        # High when: attention is focused, resolution succeeded, novelty is present.
        # Low when: distracted, resolution failed, nothing new to learn.
        # Models the "encoding vs recall" switch.
        ach_target = (
            attention_focus * 0.4 +
            float(resolution_success) * 0.3 +
            novelty * 0.2 +
            (1.0 - abs(valence)) * 0.1  # neutral valence = more encoding-ready
        )
        self.acetylcholine = (1 - alpha) * self.acetylcholine + alpha * ach_target
        self.acetylcholine = np.clip(self.acetylcholine, 0.0, 1.0)

        # --- 3. Noradrenaline (locus coeruleus): gain ---
        # High when: surprise is high, arousal is high, PE is high.
        # Low when: calm, predictable.
        ne_target = (
            surprise * 0.4 +
            arousal * 0.35 +
            pe_clipped * 0.25
        )
        self.noradrenaline = (1 - alpha) * self.noradrenaline + alpha * ne_target
        self.noradrenaline = np.clip(self.noradrenaline, 0.0, 1.0)

        # --- 4. Serotonin (raphe): stability anchor ---
        # High when: resolution success is consistent, valence is positive.
        # Low when: recent volatility / failure.
        # Higher serotonin = more conservative updating.
        ser_target = (
            float(resolution_success) * 0.5 +
            valence_boost * 0.3 +
            (1.0 - pe_clipped) * 0.2  # low prediction error = stable
        )
        # Serotonin adapts SLOWER (it's the stability anchor)
        ser_alpha = alpha * cfg.serotonin_damping
        self.serotonin = (1 - ser_alpha) * self.serotonin + ser_alpha * ser_target
        self.serotonin = np.clip(self.serotonin, 0.0, 1.0)

        # --- Derived control signals ---

        # Effective learning rate multiplier:
        # dopamine * noradrenaline amplifies, serotonin damps
        raw_lr_mult = (
            self.dopamine * (1.0 + self.noradrenaline) /
            (1.0 + self.serotonin * cfg.serotonin_damping)
        )
        effective_lr_mult = np.clip(
            raw_lr_mult,
            cfg.lr_mult_min,
            cfg.lr_mult_max
        )

        # Plasticity gate: open when acetylcholine exceeds threshold
        plasticity_open = self.acetylcholine >= cfg.ach_gate_threshold

        # Learning sharpening: high ACh + high 5-HT = sparse, precise updates
        # Low ACh + low 5-HT = broad, noisy updates (fallback / confusion)
        sharpening = (
            self.acetylcholine * 0.6 +
            self.serotonin * 0.4
        )
        sharpening = np.clip(sharpening, 0.0, 1.0)

        self.turns_processed += 1

        # Track history
        self._dopamine_history.append(self.dopamine)
        self._ach_history.append(self.acetylcholine)
        self._ne_history.append(self.noradrenaline)
        self._ser_history.append(self.serotonin)

        # Trim history (keep last 100 turns)
        if len(self._dopamine_history) > 100:
            self._dopamine_history.pop(0)
            self._ach_history.pop(0)
            self._ne_history.pop(0)
            self._ser_history.pop(0)

        return ModulatorState(
            dopamine=self.dopamine,
            acetylcholine=self.acetylcholine,
            noradrenaline=self.noradrenaline,
            serotonin=self.serotonin,
            effective_lr_mult=effective_lr_mult,
            plasticity_open=plasticity_open,
            learning_sharpening=sharpening,
        )

    def get_dopamine_teaching_signal(self) -> float:
        """Return the current dopamine teaching signal (0..1).

        Use this to multiply edge weight deltas during Hebbian updates.
        High dopamine = "this was important, strengthen it more".
        """
        return self.dopamine

    def is_plasticity_open(self) -> bool:
        """Return whether plasticity is currently enabled.

        Use this as a gate BEFORE calling learn_from_turn or any
        Hebbian update. When closed, learning is skipped entirely.
        """
        return self.acetylcholine >= self.config.ach_gate_threshold

    def get_learning_rate_multiplier(self) -> float:
        """Return the effective learning rate multiplier for this turn.

        Multiply the base learning rate by this to get the actual LR.
        Combines dopamine (amplify) and serotonin (damp) signals.
        """
        raw = (
            self.dopamine * (1.0 + self.noradrenaline) /
            (1.0 + self.serotonin * self.config.serotonin_damping)
        )
        return np.clip(raw, self.config.lr_mult_min, self.config.lr_mult_max)

    def get_sharpening(self) -> float:
        """Return the learning sharpening factor (0..1).

        High sharpening = update only the strongest edges (sparse).
        Low sharpening = update many edges broadly (exploratory).
        """
        return self.acetylcholine * 0.6 + self.serotonin * 0.4

    def get_state(self) -> Dict[str, Any]:
        """Serialize to dict (for engine save)."""
        return {
            "dopamine": self.dopamine,
            "acetylcholine": self.acetylcholine,
            "noradrenaline": self.noradrenaline,
            "serotonin": self.serotonin,
            "turns_processed": self.turns_processed,
            "dopamine_history": list(self._dopamine_history),
            "ach_history": list(self._ach_history),
            "ne_history": list(self._ne_history),
            "ser_history": list(self._ser_history),
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore from dict (engine load)."""
        self.dopamine = state.get("dopamine", self.config.dopamine_baseline)
        self.acetylcholine = state.get("acetylcholine", self.config.acetylcholine_baseline)
        self.noradrenaline = state.get("noradrenaline", self.config.noradrenaline_baseline)
        self.serotonin = state.get("serotonin", self.config.serotonin_baseline)
        self.turns_processed = state.get("turns_processed", 0)
        self._dopamine_history = list(state.get("dopamine_history", []))
        self._ach_history = list(state.get("ach_history", []))
        self._ne_history = list(state.get("ne_history", []))
        self._ser_history = list(state.get("ser_history", []))

    def get_diagnostics(self) -> Dict[str, Any]:
        """Human-readable diagnostic snapshot."""
        recent_da = self._dopamine_history[-10:] if self._dopamine_history else [self.dopamine]
        recent_ach = self._ach_history[-10:] if self._ach_history else [self.acetylcholine]
        recent_ne = self._ne_history[-10:] if self._ne_history else [self.noradrenaline]
        recent_ser = self._ser_history[-10:] if self._ser_history else [self.serotonin]

        return {
            "current": {
                "dopamine": round(self.dopamine, 3),
                "acetylcholine": round(self.acetylcholine, 3),
                "noradrenaline": round(self.noradrenaline, 3),
                "serotonin": round(self.serotonin, 3),
            },
            "avg_recent": {
                "dopamine": round(float(np.mean(recent_da)), 3),
                "acetylcholine": round(float(np.mean(recent_ach)), 3),
                "noradrenaline": round(float(np.mean(recent_ne)), 3),
                "serotonin": round(float(np.mean(recent_ser)), 3),
            },
            "plasticity_open": self.is_plasticity_open(),
            "lr_multiplier": round(self.get_learning_rate_multiplier(), 3),
            "sharpening": round(self.get_sharpening(), 3),
            "turns_processed": self.turns_processed,
        }
