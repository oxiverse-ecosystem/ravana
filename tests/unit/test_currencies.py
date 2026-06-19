"""Tests for ravana_ml.currencies and ravana_ml.currency."""

import pytest
import numpy as np
from ravana_ml.currencies import CognitiveCurrencies
from ravana_ml.currency import CognitiveCurrency, Signal, create_rlm_currency


class TestCognitiveCurrencies:
    def test_init(self):
        c = CognitiveCurrencies()
        assert c.identity_strength == 0.5
        assert c.valence == 0.0
        assert c.sleep_pressure == 0.0

    def test_update_correct(self):
        c = CognitiveCurrencies()
        c.update(conceptual_error=0.2, is_correct=True)
        assert c.dissonance_ema > 0
        assert c.identity_strength >= 0.5

    def test_update_incorrect(self):
        c = CognitiveCurrencies()
        c.update(conceptual_error=0.8, is_correct=False)
        assert c.dissonance_ema > 0.5
        assert c.identity_strength < 0.5

    def test_identity_update_correct(self):
        c = CognitiveCurrencies()
        initial = c.identity_strength
        c._compute_identity_update(0.1, True)
        # identity should increase slightly
        assert c.identity_strength >= initial

    def test_regulate(self):
        c = CognitiveCurrencies()
        c.dissonance_ema = 0.9
        c.regulate()
        assert c.regulation_mode in ("RECOVERY",)

    def test_regulate_exploration(self):
        c = CognitiveCurrencies()
        c.dissonance_ema = 0.1
        c.regulate()
        assert c.regulation_mode == "EXPLORATION"

    def test_consolidate_on_sleep(self):
        c = CognitiveCurrencies()
        c.sleep_pressure = 0.8
        c.valence = 0.5
        c.consolidate_on_sleep()
        assert c.sleep_pressure == 0.0
        assert c.valence < 0.5

    def test_get_state(self):
        c = CognitiveCurrencies()
        state = c.get_state()
        assert "identity_strength" in state
        assert "dissonance_ema" in state

    def test_load_state(self):
        c = CognitiveCurrencies()
        c.load_state({"identity_strength": 0.8, "dissonance_ema": 0.2})
        assert c.identity_strength == 0.8
        assert c.dissonance_ema == 0.2

    def test_dissonance_normalized(self):
        c = CognitiveCurrencies()
        c.dissonance_ema = 0.5
        dn = c.dissonance_normalized
        assert 0.1 <= dn <= 0.9


class TestSignal:
    def test_init(self):
        s = Signal(name="test", value=0.5, min_val=0.0, max_val=1.0)
        assert s.value == 0.5

    def test_clamp(self):
        s = Signal(name="test", value=1.5, min_val=0.0, max_val=1.0)
        s.clamp()
        assert s.value == 1.0

    def test_decay(self):
        s = Signal(name="test", value=1.0, decay_rate=0.1)
        s.decay()
        assert s.value == 0.9

    def test_update(self):
        s = Signal(name="test", value=0.0, max_val=1.0)
        s.update(0.7)
        assert s.value == 0.7


class TestCognitiveCurrency:
    def test_init(self):
        cc = CognitiveCurrency()
        assert cc._signals == {}

    def test_register(self):
        cc = CognitiveCurrency()
        s = cc.register("pressure", 0.0, min_val=0.0, max_val=1.0)
        assert s.name == "pressure"

    def test_register_derived(self):
        cc = CognitiveCurrency()
        cc.register("a", 0.3)
        cc.register("b", 0.2)
        cc.register_derived("c", lambda sigs: sigs["a"].value + sigs["b"].value,
                           min_val=0.0, max_val=1.0)
        cc.compute_derived()
        assert cc.get("c") == pytest.approx(0.5)

    def test_update(self):
        cc = CognitiveCurrency()
        cc.register("test", 0.5)
        cc.update("test", 0.8)
        assert cc.get("test") == 0.8

    def test_update_clamp(self):
        cc = CognitiveCurrency()
        cc.register("test", 0.5, max_val=1.0)
        cc.update("test", 5.0)
        assert cc.get("test") == 1.0

    def test_get_nonexistent(self):
        cc = CognitiveCurrency()
        assert cc.get("nope") is None

    def test_step_decay(self):
        cc = CognitiveCurrency()
        cc.register("test", 1.0, decay_rate=0.1)
        cc.step_decay()
        assert cc.get("test") == 0.9

    def test_add_alert(self):
        cc = CognitiveCurrency()
        cc.register("pressure", 0.5)
        cc.add_alert("pressure", 0.8, "above", "RECOVERY")
        assert len(cc._alerts) == 1

    def test_check_alert(self):
        cc = CognitiveCurrency()
        cc.register("pressure", 0.9)
        cc.add_alert("pressure", 0.8, "above", "RECOVERY")
        mode = cc.check_alerts()
        assert mode == "RECOVERY"

    def test_report(self):
        cc = CognitiveCurrency()
        cc.register("a", 0.5)
        cc.register("b", 0.7)
        r = cc.report()
        assert r["a"] == 0.5

    def test_report_full(self):
        cc = CognitiveCurrency()
        cc.register("a", 0.5)
        r = cc.report_full()
        assert "a" in r

    def test_to_dict(self):
        cc = CognitiveCurrency()
        cc.register("a", 0.5)
        d = cc.to_dict()
        assert "signals" in d
        assert "history" in d

    def test_load_dict(self):
        cc = CognitiveCurrency()
        cc.register("a", 0.5)
        cc.load_dict({"signals": {"a": {"value": 0.9}}, "history": {}})
        assert cc.get("a") == 0.9

    def test_repr(self):
        cc = CognitiveCurrency()
        cc.register("a", 0.5)
        r = repr(cc)
        assert "a=0.500" in r


class TestCreateRlmCurrency:
    def test_create(self):
        cc = create_rlm_currency()
        assert cc.get("identity_strength") == 0.5
        assert cc.get("dissonance_ema") == 0.5
        assert cc.get("sleep_pressure") == 0.0

    def test_derived_signals(self):
        cc = create_rlm_currency()
        cc.compute_derived()
        assert cc.get("cognitive_load") is not None
        assert cc.get("stability_index") is not None

    def test_alerts(self):
        cc = create_rlm_currency()
        assert len(cc._alerts) == 5  # 5 alerts defined
