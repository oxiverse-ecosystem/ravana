"""
RAVANA NeuromodulatorEngine — 4-System Ascending Neuromodulation
=================================================================
Models the four major ascending neuromodulatory systems and their
effects on cortical processing, learning, and behavior.

Architecture (Yu & Dayan 2002/2005, Hasselmo 1999, Aston-Jones & Cohen 2005):
  Prediction Error Signals → Neuromodulator Levels → Cortical Modulation

  ACh (Nucleus Basalis of Meynert):
    Signals expected uncertainty — known unreliability of predictions within a context.
    High ACh → suppress top-down predictions, favor bottom-up encoding,
                sharpen representations, increase encoding rate.
    Low ACh → allow consolidation, replay, top-down inference.
    Driven by: sustained prediction error, cue unreliability.

  NE (Locus Coeruleus):
    Signals unexpected uncertainty — gross environmental changes, surprising events.
    High NE (phasic) → reset current beliefs, trigger exploration,
                        increase learning rate globally.
    Low NE (tonic) → exploit known strategies, maintain focus.
    Driven by: large prediction error spikes, context switches.

  DA (Ventral Tegmental Area / Substantia Nigra):
    Signals reward prediction error (RPE) — better/worse than expected.
    Phasic DA → positive RPE: strengthen recently active patterns, increase vigor.
    Negative DA → negative RPE: reduce confidence in current predictions, explore.
    Tonic DA → average reward rate, baseline motivation.
    Driven by: outcome valence relative to expectation.

  5-HT (Raphe Nuclei):
    Signals punishment prediction error, behavioral inhibition, patience.
    High 5-HT → promote waiting, harm avoidance, increase repetition penalty.
    Low 5-HT → impulsivity, disinhibition, faster switching.
    Drives: opponent to DA, signals long-term average punishment rate.

  Interactions:
    ACh × NE: ACh sets gain on NE responses (high ACh → more NE reactivity).
    DA × 5-HT: Opponent encoding of valence (DA = approach, 5-HT = avoidance).
    NE × DA: Both drive exploration but via different mechanisms
             (NE = random exploration, DA = directed exploration).
    ACh × 5-HT: Both modulate temporal discounting.
"""

import numpy as np
from typing import Dict, Optional, Tuple


class NeuromodulatorEngine:
    """Four-system ascending neuromodulation with biologically-plausible dynamics.

    Each modulator has a tonic (baseline) and phasic (event-driven) component.
    Dynamics are governed by differential equations with time constants matching
    known neuromodulator kinetics.

    The engine outputs a ModulationBundle that the decoder and BG gate use to
    adjust parameters.
    """

    # Time constants (in arbitrary units, proportional to biological decay)
    TAU_ACH = 0.15   # ACh decays over ~seconds (fast)
    TAU_NE = 0.10    # NE phasic bursts decay over ~100-300ms (very fast)
    TAU_DA_PHASIC = 0.08  # DA phasic: ~100-200ms
    TAU_DA_TONIC = 0.95   # DA tonic: very slow (minutes)
    TAU_5HT = 0.30   # 5-HT: slow (seconds to minutes)

    def __init__(self,
                 dim: int = 64,
                 seed: int = 42,
                 lr: float = 0.001,
                 temperature_base: float = 0.5,
                 learning_rate_base: float = 0.001):
        self.dim = dim
        self._rng = np.random.RandomState(seed)

        # ─── Tonic levels (baseline) ───
        self.ach_tonic: float = 0.3
        self.ne_tonic: float = 0.3
        self.da_tonic: float = 0.5
        self.serotonin_tonic: float = 0.4

        # ─── Phasic levels (event-driven, decay rapidly) ───
        self.ach_phasic: float = 0.0
        self.ne_phasic: float = 0.0
        self.da_phasic: float = 0.0
        self.serotonin_phasic: float = 0.0

        # ─── State variables ───
        self.expected_uncertainty: float = 0.0
        self.unexpected_uncertainty: float = 0.0
        self.reward_prediction_error: float = 0.0
        self.punishment_prediction_error: float = 0.0
        self.average_prediction_error: float = 0.0
        self.prediction_error_ema: float = 0.0

        # ─── Base learning parameters (override by decoder) ───
        self.temperature_base = temperature_base
        self.learning_rate_base = lr

        # ─── Learning rate multiplier per component ───
        self._lr_mult_proj: float = 1.0
        self._lr_mult_gru: float = 1.0
        self._lr_mult_emb: float = 1.0
        self._lr_mult_attn: float = 1.0

        # ─── Decoder generation modulation ───
        self._temperature_mod: float = 1.0
        self._repetition_penalty_mod: float = 1.0
        self._exploration_bonus: float = 0.0
        self._confidence_threshold_mod: float = 1.0

        # ─── Internal dynamics ───
        self.step_count: int = 0
        self._last_pe: float = 0.0
        self._pe_derivative: float = 0.0
        self._pe_ema_alpha: float = 0.01

        # ─── BG gate modulation ───
        self._go_threshold_mod: float = 1.0
        self._dopamine_tone_mod: float = 0.0

    def reset(self):
        self.ach_tonic = 0.3
        self.ne_tonic = 0.3
        self.da_tonic = 0.5
        self.serotonin_tonic = 0.4
        self.ach_phasic = 0.0
        self.ne_phasic = 0.0
        self.da_phasic = 0.0
        self.serotonin_phasic = 0.0
        self.step_count = 0
        self._recompute_mods()

    def update_from_prediction_error(self, prediction_error: float,
                                     prev_prediction_error: Optional[float] = None):
        """Core update: driven by prediction error signals.

        This is the primary entry point — called after each sentence or
        generation step. Computes expected vs unexpected uncertainty
        from the PE signal.
        """
        self.step_count += 1
        pe = float(prediction_error)

        # Update running averages
        self.prediction_error_ema = (
            (1 - self._pe_ema_alpha) * self.prediction_error_ema
            + self._pe_ema_alpha * pe
        )

        # Derivative: how fast is PE changing?
        if prev_prediction_error is not None:
            self._pe_derivative = pe - prev_prediction_error
        else:
            self._pe_derivative = pe - self._last_pe
        self._last_pe = pe

        # ── ACh: Expected uncertainty ──
        # ACh tracks sustained, predictable unreliability.
        # High when prediction error is persistently above average.
        pe_deviation = max(0.0, pe - self.prediction_error_ema)
        self.expected_uncertainty = (1 - self.TAU_ACH) * self.expected_uncertainty \
            + self.TAU_ACH * pe_deviation
        target_ach = np.clip(self.expected_uncertainty * 2.0, 0.0, 1.0)
        self.ach_phasic = (1 - self.TAU_ACH) * self.ach_phasic + self.TAU_ACH * target_ach

        # ── NE: Unexpected uncertainty ──
        # NE fires on large PE jumps (derivative spikes), not sustained PE.
        pe_jump = abs(self._pe_derivative)
        self.unexpected_uncertainty = (1 - self.TAU_NE) * self.unexpected_uncertainty \
            + self.TAU_NE * pe_jump
        # ACh amplifies NE reactivity (ACh ↑ → LC more reactive to surprises)
        ach_gain = 1.0 + self.ach_level * 2.0
        target_ne = np.clip(self.unexpected_uncertainty * 3.0 * ach_gain, 0.0, 1.0)
        self.ne_phasic = (1 - self.TAU_NE) * self.ne_phasic + self.TAU_NE * target_ne

        # ── DA: Reward prediction error ──
        # Negative PE (worse than expected) → negative DA dip.
        # For generation, prediction error = how surprising each word choice was.
        # We reinterpret: if the model confidently predicts a word and is wrong,
        # that's a negative RPE. If it's surprised by a word it should have predicted,
        # that could be positive or negative depending on valence.
        pe_sign = -self._pe_derivative  # PE decrease → negative DA
        pe_magnitude = min(1.0, abs(self._pe_derivative) * 5.0)
        self.reward_prediction_error = (
            (1 - self.TAU_DA_PHASIC) * self.reward_prediction_error
            + self.TAU_DA_PHASIC * pe_sign * pe_magnitude
        )
        self.da_phasic = np.clip(self.reward_prediction_error, -0.8, 0.8)

        # ── 5-HT: Punishment prediction error ──
        # Serotonin is opponent to DA. High negative PE → high 5-HT.
        negative_pe = max(0.0, -self.reward_prediction_error)
        self.punishment_prediction_error = (
            (1 - self.TAU_5HT) * self.punishment_prediction_error
            + self.TAU_5HT * negative_pe
        )
        target_5ht = np.clip(self.punishment_prediction_error * 1.5, 0.0, 1.0)
        self.serotonin_phasic = (1 - self.TAU_5HT) * self.serotonin_phasic + self.TAU_5HT * target_5ht

        # Recompute modulation outputs
        self._recompute_mods()

    def update_from_context_switch(self, magnitude: float = 0.5):
        """Explicitly trigger NE burst on known context switches."""
        self.ne_phasic = np.clip(self.ne_phasic + magnitude * 0.5, 0.0, 1.0)
        self.unexpected_uncertainty = np.clip(
            self.unexpected_uncertainty + magnitude, 0.0, 1.0
        )
        self._recompute_mods()

    def update_from_reward(self, reward: float, expected: float):
        """Direct RPE update from an explicit reward signal (value 0-1)."""
        rpe = reward - expected
        self.reward_prediction_error = (
            (1 - self.TAU_DA_PHASIC) * self.reward_prediction_error
            + self.TAU_DA_PHASIC * rpe
        )
        self.da_phasic = np.clip(self.reward_prediction_error, -0.8, 0.8)
        self._recompute_mods()

    def tick_decay(self):
        """Passive decay of all phasic levels (called each generation step)."""
        self.ach_phasic *= (1 - self.TAU_ACH * 0.5)
        self.ne_phasic *= (1 - self.TAU_NE * 0.5)
        self.da_phasic *= (1 - self.TAU_DA_PHASIC * 0.3)
        self.serotonin_phasic *= (1 - self.TAU_5HT * 0.3)
        self._recompute_mods()

    @property
    def ach_level(self) -> float:
        """Total ACh = tonic + phasic."""
        return np.clip(self.ach_tonic + self.ach_phasic, 0.0, 1.0)

    @property
    def ne_level(self) -> float:
        """Total NE = tonic + phasic."""
        return np.clip(self.ne_tonic + self.ne_phasic, 0.0, 1.0)

    @property
    def da_level(self) -> float:
        """Total DA = tonic + phasic. Can be negative (dip) or positive (burst)."""
        return np.clip(self.da_tonic + self.da_phasic, -0.5, 1.0)

    @property
    def serotonin_level(self) -> float:
        """Total 5-HT = tonic + phasic."""
        return np.clip(self.serotonin_tonic + self.serotonin_phasic, 0.0, 1.0)

    def _recompute_mods(self):
        """Recompute all modulation outputs from current neuromodulator levels.

        Neuroscience mappings:
          ACh ↑ → sharpen representations, increase encoding LR, suppress top-down
          NE  ↑ → explore more, increase temperature, raise global LR
          DA  ↑ → increase vigor, raise confidence, strengthen recent patterns
          5-HT ↑ → increase caution, raise repetition penalty, lower temperature
        """
        ach = self.ach_level
        ne = self.ne_level
        da = self.da_level
        ht = self.serotonin_level

        # ── Temperature modulation ──
        # NE drives exploration (↑ temperature). ACh sharpens (↓ temperature).
        # 5-HT inhibits (↓ temperature). DA increases vigor (slight ↑).
        temp_mod = 1.0 + ne * 1.5 - ach * 0.5 - ht * 0.5 + max(0, da) * 0.3
        self._temperature_mod = np.clip(temp_mod, 0.2, 2.5)

        # ── Repetition penalty ──
        # 5-HT ↑ → more repetition avoidance (don't repeat mistakes).
        # NE ↑ → less repetition penalty (explore new patterns).
        # ACh ↑ → sharper, more distinct outputs (slight increase).
        rep_mod = 1.0 + ht * 1.5 - ne * 0.5 + ach * 0.3
        self._repetition_penalty_mod = np.clip(rep_mod, 0.2, 3.0)

        # ── Learning rate multipliers per component ──
        # Output projection (content word learning): ACh ↑ → learn content faster
        lr_proj = 1.0 + ach * 1.5 + ne * 1.0 - ht * 0.5 + max(0, da) * 0.5
        self._lr_mult_proj = np.clip(lr_proj, 0.2, 4.0)

        # GRU (sequence dynamics): NE ↑ → adapt faster to new sequences
        lr_gru = 1.0 + ne * 1.2 + ach * 0.5 - ht * 0.5
        self._lr_mult_gru = np.clip(lr_gru, 0.2, 3.0)

        # Embedding (word representations): ACh ↑ → sharpen word boundaries
        lr_emb = 1.0 + ach * 1.0 - ht * 0.3
        self._lr_mult_emb = np.clip(lr_emb, 0.2, 3.0)

        # Attention: NE ↑ → shift attention, ACh ↑ → focus attention
        lr_attn = 1.0 + ne * 0.8 + ach * 0.6 - ht * 0.5
        self._lr_mult_attn = np.clip(lr_attn, 0.2, 3.0)

        # ── Exploration bonus (added to logits for rare/content words) ──
        # NE → explore novel words, DA → exploit known rewarding words
        self._exploration_bonus = ne * 8.0 - max(0, da) * 1.0
        self._exploration_bonus = np.clip(self._exploration_bonus, -1.0, 8.0)

        # ── Confidence threshold ──
        # DA ↑ → more confident (higher threshold for rejecting top prediction).
        # NE ↑ → less confident (more willing to try alternatives).
        conf_mod = 1.0 + max(0, da) * 0.3 - ne * 0.3
        self._confidence_threshold_mod = np.clip(conf_mod, 0.3, 1.5)

        # ── BG gate modulation ──
        # Go threshold: lower when NE ↑ (more exploratory), higher when 5-HT ↑ (cautious)
        go_mod = 1.0 - ne * 0.3 + ht * 0.3
        self._go_threshold_mod = np.clip(go_mod, 0.4, 1.5)

        # Dopamine tone for BG: higher when DA is positive
        self._dopamine_tone_mod = np.clip(da, -0.3, 0.8)

    # ─── Readout methods (called by decoder generate/train) ───

    def generation_mods(self) -> Dict[str, float]:
        """Return modulation dict for NeuralDecoder.generate()."""
        return {
            'temperature_mod': self._temperature_mod,
            'repetition_penalty_mod': self._repetition_penalty_mod,
            'exploration_bonus': self._exploration_bonus,
            'confidence_threshold_mod': self._confidence_threshold_mod,
            'ne_level': self.ne_level,
            'da_level': self.da_level,
            'ach_level': self.ach_level,
            'serotonin_level': self.serotonin_level,
        }

    def training_mods(self) -> Dict[str, float]:
        """Return modulation dict for NeuralDecoder.train_on_sentence()."""
        return {
            'lr_mult_proj': self._lr_mult_proj,
            'lr_mult_gru': self._lr_mult_gru,
            'lr_mult_emb': self._lr_mult_emb,
            'lr_mult_attn': self._lr_mult_attn,
            'temperature_mod': self._temperature_mod,
            'repetition_penalty_mod': self._repetition_penalty_mod,
            'exploration_bonus': self._exploration_bonus,
            'ne_level': self.ne_level,
            'da_level': self.da_level,
            'ach_level': self.ach_level,
            'serotonin_level': self.serotonin_level,
        }

    def bg_gate_mods(self) -> Dict[str, float]:
        """Return modulation dict for BG gate (Go threshold, dopamine tone)."""
        return {
            'go_threshold_mod': self._go_threshold_mod,
            'dopamine_tone_mod': self._dopamine_tone_mod,
        }

    def get_state(self) -> Dict:
        return {
            'ach_tonic': self.ach_tonic,
            'ne_tonic': self.ne_tonic,
            'da_tonic': self.da_tonic,
            'serotonin_tonic': self.serotonin_tonic,
            'step_count': self.step_count,
        }

    def load_state(self, state: Dict):
        for k, v in state.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self._recompute_mods()

    def __repr__(self) -> str:
        return (
            f"NeuromodulatorEngine(ACh={self.ach_level:.2f}, NE={self.ne_level:.2f}, "
            f"DA={self.da_level:.2f}, 5-HT={self.serotonin_level:.2f})"
        )
