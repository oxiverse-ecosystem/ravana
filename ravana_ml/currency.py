"""
Cognitive Currency System for RAVANA

Unified framework for all cognitive scalar signals in the RLM. Instead of
scattered `self.sleep_pressure`, `self.dissonance_ema`, etc., all signals
are registered in one place with:
- Range constraints (min/max)
- Decay rates
- Composition rules (derived signals)
- Threshold-based alerts (for regulation mode switching)
- Serialization (for checkpoint save/load)

This module is ADDITIVE — it wraps around existing RLM scalars without
replacing them. The RLM continues to use `self.sleep_pressure` directly;
the currency provides a unified view and composition layer.

Design principles:
1. Zero breaking changes — existing code keeps working
2. Signals are registered once, updated in-place
3. Derived signals auto-compute from dependencies
4. New signals (Bayesian posteriors, episodic confidence) plug in by name
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class Signal:
    """A named scalar signal with range constraints and decay."""
    name: str
    value: float
    min_val: float = 0.0
    max_val: float = 1.0
    decay_rate: float = 0.0        # per-step exponential decay (0 = no decay)
    description: str = ""

    def clamp(self):
        self.value = max(self.min_val, min(self.max_val, self.value))
        return self

    def decay(self):
        if self.decay_rate > 0:
            self.value *= (1.0 - self.decay_rate)
        return self

    def update(self, new_value: float):
        self.value = new_value
        return self.clamp()

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'value': self.value,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'decay_rate': self.decay_rate,
        }


@dataclass
class ThresholdAlert:
    """Fires when a signal crosses a threshold boundary."""
    signal_name: str
    threshold: float
    direction: str  # "above" or "below"
    mode: str       # regulation mode to activate
    fired: bool = False

    def check(self, signals: Dict[str, Signal]) -> Optional[str]:
        """Check if threshold is crossed. Returns mode name if fired."""
        sig = signals.get(self.signal_name)
        if sig is None:
            return None
        if self.direction == "above" and sig.value > self.threshold:
            if not self.fired:
                self.fired = True
                return self.mode
        elif self.direction == "below" and sig.value < self.threshold:
            if not self.fired:
                self.fired = True
                return self.mode
        else:
            self.fired = False  # reset when condition no longer holds
        return None


class CognitiveCurrency:
    """Unified cognitive signal registry.

    Usage:
        currency = CognitiveCurrency()
        currency.register('sleep_pressure', 0.0, min_val=0.0, max_val=1.0, decay_rate=0.001)
        currency.register('dissonance_ema', 0.5, min_val=0.0, max_val=2.0, decay_rate=0.0)
        currency.register_derived('cognitive_load', lambda s: s['sleep_pressure'].value * s['dissonance_ema'].value)

        currency.update('sleep_pressure', 0.7)
        currency.step_decay()  # apply per-step decay to all signals

        # Threshold alerts
        currency.add_alert('dissonance_ema', 0.8, 'above', 'RECOVERY')
        mode = currency.check_alerts()  # returns 'RECOVERY' if dissonance > 0.8
    """

    def __init__(self):
        self._signals: Dict[str, Signal] = {}
        self._derived: Dict[str, Callable[[Dict[str, Signal]], float]] = {}
        self._alerts: List[ThresholdAlert] = []
        self._history: Dict[str, List[float]] = {}
        self._history_max = 100

    def register(self, name: str, value: float,
                 min_val: float = 0.0, max_val: float = 1.0,
                 decay_rate: float = 0.0, description: str = "") -> Signal:
        """Register a new signal. Returns the Signal object."""
        sig = Signal(name=name, value=value, min_val=min_val,
                     max_val=max_val, decay_rate=decay_rate,
                     description=description)
        self._signals[name] = sig
        self._history[name] = []
        return sig

    def register_derived(self, name: str,
                         formula: Callable[[Dict[str, Signal]], float],
                         min_val: float = 0.0, max_val: float = 1.0,
                         description: str = ""):
        """Register a derived signal computed from other signals."""
        self._derived[name] = formula
        # Also create a signal entry so it can be read like any other
        self._signals[name] = Signal(name=name, value=0.0, min_val=min_val,
                                     max_val=max_val, description=description)
        self._history[name] = []

    def add_alert(self, signal_name: str, threshold: float,
                  direction: str, mode: str):
        """Add a threshold alert. direction: 'above' or 'below'."""
        self._alerts.append(ThresholdAlert(
            signal_name=signal_name, threshold=threshold,
            direction=direction, mode=mode
        ))

    def update(self, name: str, value: float):
        """Update a signal's value. Clamps to [min_val, max_val]."""
        if name in self._signals:
            self._signals[name].update(value)

    def get(self, name: str) -> Optional[float]:
        """Get a signal's current value."""
        sig = self._signals.get(name)
        return sig.value if sig else None

    def get_signal(self, name: str) -> Optional[Signal]:
        """Get the Signal object."""
        return self._signals.get(name)

    def step_decay(self):
        """Apply per-step decay to all signals."""
        for sig in self._signals.values():
            if sig.name not in self._derived:
                sig.decay()

    def compute_derived(self):
        """Recompute all derived signals from current values."""
        for name, formula in self._derived.items():
            val = formula(self._signals)
            self._signals[name].value = max(
                self._signals[name].min_val,
                min(self._signals[name].max_val, val)
            )

    def record_history(self):
        """Snapshot current values into history."""
        for name, sig in self._signals.items():
            self._history[name].append(sig.value)
            if len(self._history[name]) > self._history_max:
                self._history[name].pop(0)

    def check_alerts(self) -> Optional[str]:
        """Check all threshold alerts. Returns first triggered mode, or None."""
        for alert in self._alerts:
            mode = alert.check(self._signals)
            if mode is not None:
                return mode
        return None

    def report(self) -> Dict[str, float]:
        """Return all signal values as a flat dict."""
        return {name: sig.value for name, sig in self._signals.items()}

    def report_full(self) -> Dict[str, Any]:
        """Return full signal info including ranges and history length."""
        result = {}
        for name, sig in self._signals.items():
            result[name] = {
                'value': sig.value,
                'range': (sig.min_val, sig.max_val),
                'decay_rate': sig.decay_rate,
                'history_len': len(self._history.get(name, [])),
                'is_derived': name in self._derived,
            }
        return result

    def to_dict(self) -> dict:
        """Serialize to dict for checkpoint save."""
        return {
            'signals': {name: sig.to_dict() for name, sig in self._signals.items()},
            'history': {name: list(vals) for name, vals in self._history.items()},
        }

    def load_dict(self, data: dict):
        """Restore from serialized dict."""
        for name, sig_data in data.get('signals', {}).items():
            if name in self._signals:
                self._signals[name].value = sig_data.get('value', 0.0)
        for name, vals in data.get('history', {}).items():
            if name in self._history:
                self._history[name] = list(vals)

    def __repr__(self):
        vals = ", ".join(f"{n}={s.value:.3f}" for n, s in self._signals.items()
                         if n not in self._derived)
        derived = ", ".join(f"{n}={s.value:.3f}" for n, s in self._signals.items()
                            if n in self._derived)
        return f"CognitiveCurrency({vals} | derived: {derived})"


def create_rlm_currency() -> CognitiveCurrency:
    """Create a CognitiveCurrency pre-configured with RLM's standard signals.

    This factory sets up the signals that RLM already tracks, with appropriate
    ranges and decay rates matching the current scalar implementations.
    """
    c = CognitiveCurrency()

    # Core cognitive signals (matching RLM defaults)
    c.register('identity_strength', 0.5, min_val=0.1, max_val=0.95,
               description="Self-concept coherence [0,1]")
    c.register('dissonance_ema', 0.5, min_val=0.0, max_val=2.0,
               description="Exponential moving average of prediction error")
    c.register('sleep_pressure', 0.0, min_val=0.0, max_val=1.0,
               description="Accumulates from free energy + contradictions")
    c.register('conceptual_accuracy', 0.0, min_val=0.0, max_val=1.0,
               decay_rate=0.0,
               description="Running prediction accuracy")

    # Emotion signals (VAD model)
    c.register('valence', 0.0, min_val=-1.0, max_val=1.0,
               description="Positive/negative affect")
    c.register('arousal', 0.3, min_val=0.0, max_val=1.0,
               description="Activation level")
    c.register('dominance', 0.5, min_val=0.0, max_val=1.0,
               description="Sense of control")

    # Meaning and free energy
    c.register('accumulated_meaning', 0.0, min_val=0.0, max_val=1e6,
               description="Running meaning total")
    c.register('total_free_energy', 0.0, min_val=0.0, max_val=1e6,
               description="Free energy accumulator")

    # Convergence tracking
    c.register('edge_weight_ema', 0.0, min_val=0.0, max_val=1.0,
               description="EMA of mean edge weight")
    c.register('token_hit_ema', 0.5, min_val=0.0, max_val=1.0,
               description="EMA of token-level prediction hit rate")

    # Derived signals — computed from others
    c.register_derived(
        'cognitive_load',
        lambda s: min(1.0, (s['sleep_pressure'].value * 0.5 +
                            s['dissonance_ema'].value * 0.3 +
                            (1.0 - s['identity_strength'].value) * 0.2)),
        min_val=0.0, max_val=1.0,
        description="Composite cognitive load metric"
    )

    c.register_derived(
        'stability_index',
        lambda s: 0.5 + 0.5 * s['identity_strength'].value - 0.3 * s['dissonance_ema'].value,
        min_val=0.0, max_val=1.0,
        description="How stable the cognitive state is"
    )

    # Threshold alerts for regulation mode switching
    c.add_alert('dissonance_ema', 0.8, 'above', 'RECOVERY')
    c.add_alert('dissonance_ema', 0.5, 'above', 'RESOLUTION')
    c.add_alert('dissonance_ema', 0.15, 'below', 'EXPLORATION')
    c.add_alert('identity_strength', 0.85, 'above', 'PLATEAU')
    c.add_alert('sleep_pressure', 0.7, 'above', 'SLEEP')

    return c
