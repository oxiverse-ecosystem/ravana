"""Tests for CognitiveCurrency system."""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
from ravana_ml.currency import CognitiveCurrency, create_rlm_currency


def test_register_and_get():
    c = CognitiveCurrency()
    c.register('test_signal', 0.5, min_val=0.0, max_val=1.0)
    assert c.get('test_signal') == 0.5
    c.update('test_signal', 0.8)
    assert c.get('test_signal') == 0.8


def test_clamping():
    c = CognitiveCurrency()
    c.register('bounded', 0.5, min_val=0.0, max_val=1.0)
    c.update('bounded', 1.5)
    assert c.get('bounded') == 1.0
    c.update('bounded', -0.5)
    assert c.get('bounded') == 0.0


def test_decay():
    c = CognitiveCurrency()
    c.register('decaying', 1.0, decay_rate=0.1)
    c.step_decay()
    assert abs(c.get('decaying') - 0.9) < 1e-10


def test_derived():
    c = CognitiveCurrency()
    c.register('a', 0.5)
    c.register('b', 0.3)
    c.register_derived('sum', lambda s: s['a'].value + s['b'].value,
                       min_val=0.0, max_val=1.0)
    c.compute_derived()
    assert abs(c.get('sum') - 0.8) < 1e-10


def test_threshold_alert():
    c = CognitiveCurrency()
    c.register('dissonance', 0.5)
    c.add_alert('dissonance', 0.8, 'above', 'RECOVERY')
    c.add_alert('dissonance', 0.15, 'below', 'EXPLORATION')

    # Normal range — no alert
    assert c.check_alerts() is None

    # Trigger high dissonance
    c.update('dissonance', 0.9)
    assert c.check_alerts() == 'RECOVERY'

    # After alert fires, subsequent checks return None until reset
    # Reset: value drops below threshold then rises again
    c.update('dissonance', 0.7)
    c.check_alerts()  # resets fired flag
    c.update('dissonance', 0.9)
    assert c.check_alerts() == 'RECOVERY'


def test_history():
    c = CognitiveCurrency()
    c.register('x', 1.0, min_val=0.0, max_val=1000.0)
    for i in range(105):
        c.update('x', float(i))
        c.record_history()
    history = c.to_dict()['history']['x']
    assert len(history) == 100  # max history
    assert history[-1] == 104.0


def test_serialization():
    c = CognitiveCurrency()
    c.register('val', 0.42)
    c.update('val', 0.77)
    c.record_history()

    data = c.to_dict()
    c2 = CognitiveCurrency()
    c2.register('val', 0.0)
    c2.load_dict(data)
    assert abs(c2.get('val') - 0.77) < 1e-10


def test_report():
    c = CognitiveCurrency()
    c.register('a', 0.5)
    c.register('b', 0.3)
    r = c.report()
    assert r == {'a': 0.5, 'b': 0.3}


def test_create_rlm_currency():
    c = create_rlm_currency()
    # Should have all standard signals
    assert c.get('identity_strength') == 0.5
    assert c.get('dissonance_ema') == 0.5
    assert c.get('sleep_pressure') == 0.0
    assert c.get('valence') == 0.0
    assert c.get('arousal') == 0.3

    # Derived signals should be computed
    c.compute_derived()
    load = c.get('cognitive_load')
    assert load is not None
    assert 0.0 <= load <= 1.0

    stability = c.get('stability_index')
    assert stability is not None
    assert 0.0 <= stability <= 1.0


def test_rlm_currency_alerts():
    c = create_rlm_currency()
    # Normal state — no alert
    assert c.check_alerts() is None

    # High dissonance triggers RECOVERY
    c.update('dissonance_ema', 0.85)
    assert c.check_alerts() == 'RECOVERY'


def test_full_rlm_currency_lifecycle():
    c = create_rlm_currency()
    # Simulate a few steps
    for i in range(10):
        c.update('sleep_pressure', 0.0 + i * 0.05)
        c.update('dissonance_ema', 0.5 + i * 0.03)
        c.update('identity_strength', 0.5 + i * 0.01)
        c.compute_derived()
        c.record_history()

    data = c.to_dict()
    assert 'signals' in data
    assert 'history' in data
    assert len(data['history']['sleep_pressure']) == 10
