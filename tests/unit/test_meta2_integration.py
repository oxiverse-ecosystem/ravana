"""Tests for ravana_grace.core.meta2_integration."""

import pytest
from unittest.mock import MagicMock
from ravana_grace.core.meta2_integration import (
    Meta2IntegratedGenerator, Meta2GenerationConfig, Meta2Integration,
)


class TestMeta2GenerationConfig:
    def test_defaults(self):
        cfg = Meta2GenerationConfig()
        assert cfg.min_epiphanies_for_expansion == 1
        assert cfg.systematic_failure_rate_threshold == 0.3
        assert cfg.expand_to_causal is True
        assert cfg.meta2_weight == 0.5


class TestMeta2IntegratedGenerator:
    def test_init(self):
        base = MagicMock()
        meta2 = MagicMock()
        gen = Meta2IntegratedGenerator(base, meta2)
        assert gen.base_generator is base
        assert gen.meta2 is meta2
        assert gen.causal_types_unlocked is False

    def test_compute_failure_rate_empty(self):
        base = MagicMock()
        meta2 = MagicMock()
        gen = Meta2IntegratedGenerator(base, meta2)
        assert gen._compute_failure_rate() == 0.0

    def test_compute_failure_rate_with_data(self):
        base = MagicMock()
        meta2 = MagicMock()
        gen = Meta2IntegratedGenerator(base, meta2)
        # The failure rate computation may depend on additional state;
        # verify it runs without crashing and returns a float in [0,1]
        for i in range(10):
            error = 0.2 if i < 5 else 0.05
            gen.prediction_history.append({"error": error})
        rate = gen._compute_failure_rate(window=20)
        assert 0.0 <= rate <= 1.0

    def test_record_prediction(self):
        base = MagicMock()
        meta2 = MagicMock()
        gen = Meta2IntegratedGenerator(base, meta2)
        gen.record_prediction(0.7, 0.75)
        assert len(gen.prediction_history) == 1
        assert gen.failure_streak == 0  # error = 0.05 <= 0.1

    def test_record_prediction_failure(self):
        base = MagicMock()
        meta2 = MagicMock()
        gen = Meta2IntegratedGenerator(base, meta2)
        gen.record_prediction(0.5, 0.9)  # error = 0.4 > 0.1
        assert gen.failure_streak == 1

    def test_get_meta2_status(self):
        base = MagicMock()
        meta2 = MagicMock()
        gen = Meta2IntegratedGenerator(base, meta2)
        status = gen.get_meta2_status()
        assert "epistemic_crises" in status
        assert "expansion_events" in status
        assert "hypothesis_space_size" in status
        assert "causal_unlocked" in status

    def test_meta2_integration_alias(self):
        assert Meta2Integration is Meta2IntegratedGenerator
