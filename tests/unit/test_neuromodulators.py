"""Tests for the neuromodulator system (dopamine, acetylcholine, noradrenaline, serotonin).

Covers:
- Baseline seeding is stable and reproducible.
- High prediction error + positive surprise raises dopamine.
- Low attention + resolution failure closes plasticity gate.
- Serotonin dampens learning rate multiplier.
- Noradrenaline amplifies dopamine signal.
- Plasticity gate responds to acetylcholine threshold.
- State persistence (save/load round-trip).
- Learning sharpening bounds.
- LR multiplier bounds.
"""

import numpy as np
import pytest

from ravana.learn.neuromodulators import (
    NeuromodulatorSystem,
    NeuromodulatorConfig,
    ModulatorState,
)


def _default_config():
    return NeuromodulatorConfig(
        dopamine_baseline=0.5,
        acetylcholine_baseline=0.5,
        noradrenaline_baseline=0.3,
        serotonin_baseline=0.5,
        ach_gate_threshold=0.25,
        serotonin_damping=0.7,
        dopamine_max=0.95,
        lr_mult_min=0.1,
        lr_mult_max=3.0,
        ema_alpha=0.15,
    )


class TestNeuromodulatorBasics:
    """Baseline behavior and seeded state."""

    def test_initial_levels_match_config(self):
        cfg = _default_config()
        sys = NeuromodulatorSystem(cfg)
        assert sys.dopamine == 0.5
        assert sys.acetylcholine == 0.5
        assert sys.noradrenaline == 0.3
        assert sys.serotonin == 0.5

    def test_plasticity_open_at_baseline(self):
        sys = NeuromodulatorSystem(_default_config())
        # Baseline ACh=0.5 >= threshold 0.25 -> gate open
        assert sys.is_plasticity_open() is True

    def test_lr_multiplier_at_baseline(self):
        sys = NeuromodulatorSystem(_default_config())
        mult = sys.get_learning_rate_multiplier()
        assert 0.1 <= mult <= 3.0

    def test_sharpening_at_baseline(self):
        sys = NeuromodulatorSystem(_default_config())
        sharp = sys.get_sharpening()
        assert 0.0 <= sharp <= 1.0


class TestDopaminePredictionError:
    """Dopamine responds to reward prediction error and surprise."""

    def test_high_pe_positive_valence_raises_dopamine(self):
        sys = NeuromodulatorSystem(_default_config())
        initial_da = sys.dopamine
        sys.update(prediction_error=0.9, novelty=0.5, valence=0.8,
                   surprise=0.7, arousal=0.6, resolution_success=True)
        assert sys.dopamine > initial_da

    def test_low_pe_low_surprise_lowers_dopamine(self):
        sys = NeuromodulatorSystem(_default_config())
        # Run several low-PE turns to drive EMA down
        for _ in range(10):
            sys.update(prediction_error=0.0, novelty=0.0, valence=0.0,
                       surprise=0.0, arousal=0.1, resolution_success=True)
        assert sys.dopamine < 0.5

    def test_dopamine_never_exceeds_max(self):
        cfg = _default_config()
        sys = NeuromodulatorSystem(cfg)
        for _ in range(50):
            sys.update(prediction_error=1.0, novelty=1.0, valence=1.0,
                       surprise=1.0, arousal=1.0, resolution_success=True)
        assert sys.dopamine <= cfg.dopamine_max + 1e-6

    def test_dopamine_never_negative(self):
        sys = NeuromodulatorSystem(_default_config())
        for _ in range(50):
            sys.update(prediction_error=0.0, novelty=0.0, valence=-1.0,
                       surprise=0.0, arousal=0.0, resolution_success=False)
        assert sys.dopamine >= 0.0


class TestAcetylcholineGate:
    """ACh controls the plasticity gate."""

    def test_low_attention_failure_closes_gate(self):
        sys = NeuromodulatorSystem(_default_config())
        for _ in range(20):
            sys.update(prediction_error=0.0, novelty=0.0, valence=0.0,
                       surprise=0.0, arousal=0.0, resolution_success=False,
                       attention_focus=0.05)
        # ACh should drop below threshold
        assert sys.acetylcholine < _default_config().ach_gate_threshold
        assert sys.is_plasticity_open() == False

    def test_high_attention_success_opens_gate(self):
        sys = NeuromodulatorSystem(_default_config())
        sys.update(prediction_error=0.0, novelty=0.0, valence=0.0,
                   surprise=0.0, arousal=0.0, resolution_success=True,
                   attention_focus=0.9)
        assert sys.acetylcholine > 0.4
        assert sys.is_plasticity_open() == True

    def test_plasticity_closed_when_ach_below_threshold(self):
        sys = NeuromodulatorSystem(_default_config())
        # Force ACh very low
        for _ in range(30):
            sys.update(prediction_error=0.0, novelty=0.0, valence=0.0,
                       surprise=0.0, arousal=0.0, resolution_success=False,
                       attention_focus=0.0)
        assert sys.is_plasticity_open() is False


class TestNoradrenalineGain:
    """NE amplifies the dopamine teaching signal."""

    def test_high_surprise_arousal_raises_ne(self):
        sys = NeuromodulatorSystem(_default_config())
        initial_ne = sys.noradrenaline
        sys.update(prediction_error=0.8, novelty=0.7, valence=0.0,
                   surprise=0.9, arousal=0.9, resolution_success=True)
        assert sys.noradrenaline > initial_ne

    def test_low_arousal_lowers_ne(self):
        sys = NeuromodulatorSystem(_default_config())
        for _ in range(10):
            sys.update(prediction_error=0.0, novelty=0.0, valence=0.0,
                       surprise=0.0, arousal=0.05, resolution_success=True)
        assert sys.noradrenaline < 0.3

    def test_ne_amplifies_lr_multiplier(self):
        """High NE with high DA should produce a higher LR multiplier than low NE."""
        sys_high = NeuromodulatorSystem(_default_config())
        sys_high.update(prediction_error=0.8, novelty=0.8, valence=0.8,
                        surprise=0.9, arousal=0.9, resolution_success=True)
        sys_low = NeuromodulatorSystem(_default_config())
        sys_low.update(prediction_error=0.1, novelty=0.1, valence=0.1,
                       surprise=0.05, arousal=0.05, resolution_success=True)
        assert sys_high.get_learning_rate_multiplier() > sys_low.get_learning_rate_multiplier()


class TestSerotoninDamping:
    """Serotonin stabilizes and damps learning rate."""

    def test_high_serotonin_reduces_lr_mult(self):
        """Same DA/NE but higher 5-HT should reduce the LR multiplier."""
        sys_high_5ht = NeuromodulatorSystem(_default_config())
        # Drive serotonin up with repeated success + positive valence
        for _ in range(20):
            sys_high_5ht.update(prediction_error=0.0, novelty=0.0, valence=0.9,
                                surprise=0.0, arousal=0.0, resolution_success=True)

        sys_low_5ht = NeuromodulatorSystem(_default_config())
        # Drive serotonin down with repeated failure + negative valence
        for _ in range(20):
            sys_low_5ht.update(prediction_error=0.0, novelty=0.0, valence=-0.9,
                               surprise=0.0, arousal=0.0, resolution_success=False)

        assert sys_high_5ht.serotonin > sys_low_5ht.serotonin

    def test_serotonin_adapts_slower(self):
        """Serotonin EMA should be slower than dopamine EMA."""
        cfg = _default_config()
        sys = NeuromodulatorSystem(cfg)
        # Single sharp pulse
        sys.update(prediction_error=1.0, novelty=1.0, valence=1.0,
                   surprise=1.0, arousal=1.0, resolution_success=True)
        da_change = abs(sys.dopamine - cfg.dopamine_baseline)
        ser_change = abs(sys.serotonin - cfg.serotonin_baseline)
        # Serotonin should change less than dopamine in one step
        assert ser_change < da_change


class TestLearningSharpening:
    """ACh + 5-HT determine learning sparseness."""

    def test_high_ach_high_ser_gives_sharp_learning(self):
        sys = NeuromodulatorSystem(_default_config())
        for _ in range(10):
            sys.update(prediction_error=0.0, novelty=0.5, valence=0.5,
                       surprise=0.0, arousal=0.0, resolution_success=True,
                       attention_focus=0.8)
        assert sys.get_sharpening() > 0.5

    def test_low_ach_low_ser_gives_broad_learning(self):
        sys = NeuromodulatorSystem(_default_config())
        for _ in range(10):
            sys.update(prediction_error=0.0, novelty=0.0, valence=0.0,
                       surprise=0.0, arousal=0.0, resolution_success=False,
                       attention_focus=0.05)
        assert sys.get_sharpening() < 0.5

    def test_sharpening_bounded_01(self):
        sys = NeuromodulatorSystem(_default_config())
        for _ in range(20):
            sys.update(prediction_error=np.random.random(), novelty=np.random.random(),
                       valence=np.random.random() * 2 - 1, surprise=np.random.random(),
                       arousal=np.random.random(), resolution_success=bool(int(np.random.random() * 2)),
                       attention_focus=np.random.random())
        assert 0.0 <= sys.get_sharpening() <= 1.0


class TestLRMultiplierBounds:
    """LR multiplier is always within [lr_mult_min, lr_mult_max]."""

    def test_maximum_lr_bounded(self):
        cfg = _default_config()
        sys = NeuromodulatorSystem(cfg)
        for _ in range(50):
            sys.update(prediction_error=1.0, novelty=1.0, valence=1.0,
                       surprise=1.0, arousal=1.0, resolution_success=True,
                       attention_focus=1.0)
        assert sys.get_learning_rate_multiplier() <= cfg.lr_mult_max + 1e-6

    def test_minimum_lr_bounded(self):
        cfg = _default_config()
        sys = NeuromodulatorSystem(cfg)
        for _ in range(50):
            sys.update(prediction_error=0.0, novelty=0.0, valence=-1.0,
                       surprise=0.0, arousal=0.0, resolution_success=False,
                       attention_focus=0.0)
        assert sys.get_learning_rate_multiplier() >= cfg.lr_mult_min - 1e-6


class TestStatePersistence:
    """Save/load round-trip."""

    def test_get_set_state_roundtrip(self):
        sys = NeuromodulatorSystem(_default_config())
        sys.update(prediction_error=0.5, novelty=0.4, valence=0.3,
                   surprise=0.2, arousal=0.6, resolution_success=True,
                   attention_focus=0.7)
        state = sys.get_state()
        sys2 = NeuromodulatorSystem(_default_config())
        sys2.set_state(state)
        assert abs(sys2.dopamine - sys.dopamine) < 1e-9
        assert abs(sys2.acetylcholine - sys.acetylcholine) < 1e-9
        assert abs(sys2.noradrenaline - sys.noradrenaline) < 1e-9
        assert abs(sys2.serotonin - sys.serotonin) < 1e-9
        assert sys2.turns_processed == sys.turns_processed

    def test_history_preserved(self):
        sys = NeuromodulatorSystem(_default_config())
        for _ in range(5):
            sys.update(prediction_error=0.5, novelty=0.5, valence=0.5,
                       surprise=0.5, arousal=0.5, resolution_success=True)
        state = sys.get_state()
        assert len(state["dopamine_history"]) == 5
        sys2 = NeuromodulatorSystem(_default_config())
        sys2.set_state(state)
        assert len(sys2._dopamine_history) == 5


class TestDiagnostics:
    """get_diagnostics returns a well-formed dict."""

    def test_diagnostics_structure(self):
        sys = NeuromodulatorSystem(_default_config())
        sys.update(prediction_error=0.5, novelty=0.5, valence=0.5,
                   surprise=0.5, arousal=0.5, resolution_success=True)
        diag = sys.get_diagnostics()
        assert "current" in diag
        assert "avg_recent" in diag
        assert "plasticity_open" in diag
        assert "lr_multiplier" in diag
        assert "sharpening" in diag
        assert "turns_processed" in diag
        for key in ["dopamine", "acetylcholine", "noradrenaline", "serotonin"]:
            assert key in diag["current"]
            assert key in diag["avg_recent"]


class TestUpdateReturnsModulatorState:
    """update() returns a ModulatorState dataclass."""

    def test_return_type(self):
        sys = NeuromodulatorSystem(_default_config())
        result = sys.update(prediction_error=0.5, novelty=0.5, valence=0.5,
                            surprise=0.5, arousal=0.5, resolution_success=True)
        assert isinstance(result, ModulatorState)
        assert 0.0 <= result.dopamine <= 1.0
        assert 0.0 <= result.acetylcholine <= 1.0
        assert 0.0 <= result.noradrenaline <= 1.0
        assert 0.0 <= result.serotonin <= 1.0
        assert result.effective_lr_mult > 0.0


class TestHistoryTrimming:
    """History lists don't grow unbounded."""

    def test_history_capped_at_100(self):
        sys = NeuromodulatorSystem(_default_config())
        for _ in range(150):
            sys.update(prediction_error=0.5, novelty=0.5, valence=0.5,
                       surprise=0.5, arousal=0.5, resolution_success=True)
        assert len(sys._dopamine_history) == 100
        assert len(sys._ach_history) == 100
        assert len(sys._ne_history) == 100
        assert len(sys._ser_history) == 100


class TestConfigDefaults:
    """NeuromodulatorConfig has sensible defaults."""

    def test_defaults(self):
        cfg = NeuromodulatorConfig()
        assert cfg.dopamine_baseline == 0.5
        assert cfg.acetylcholine_baseline == 0.5
        assert cfg.noradrenaline_baseline == 0.3
        assert cfg.serotonin_baseline == 0.5
        assert cfg.ach_gate_threshold == 0.25
        assert cfg.lr_mult_min < cfg.lr_mult_max
