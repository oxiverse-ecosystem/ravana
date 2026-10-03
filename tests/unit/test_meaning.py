"""Tests for ravana_grace.core.meaning.

Covers MeaningConfig, MeaningEngine.compute_meaning, stake_meaning /
resolve_stake, get_expected_meaning and get_status.

Every test uses a bare `assert` — a test that RETURNS a bool instead of
asserting is reported green by pytest regardless of the value, which has
hidden real defects in this repo before.
"""

import pytest

from ravana_grace.core.meaning import MeaningConfig, MeaningEngine, MeaningRecord


def _engine(**overrides) -> MeaningEngine:
    cfg = MeaningConfig(**overrides)
    return MeaningEngine(cfg)


class TestMeaningConfig:
    def test_defaults(self):
        cfg = MeaningConfig()
        assert cfg.w_dissonance_reduction == 0.4
        assert cfg.w_identity_coherence == 0.3
        assert cfg.w_predictive_power == 0.3
        assert cfg.effort_kappa == 0.5
        assert cfg.max_history == 1000

    def test_weights_are_independently_configurable(self):
        cfg = MeaningConfig(w_dissonance_reduction=1.0)
        assert cfg.w_dissonance_reduction == 1.0
        # the other weights are untouched
        assert cfg.w_identity_coherence == 0.3
        assert cfg.w_predictive_power == 0.3


class TestComputeMeaning:
    def test_returns_meaning_record_with_full_breakdown(self):
        eng = _engine()
        rec = eng.compute_meaning(
            episode=0,
            pre_dissonance=0.8,
            post_dissonance=0.2,
            pre_identity=0.3,
            post_identity=0.5,
            predictive_gain=0.4,
            effort=0.0,
        )
        assert isinstance(rec, MeaningRecord)
        assert rec.episode == 0
        assert rec.coherence_gain == pytest.approx(0.6)
        assert rec.identity_coherence_gain == pytest.approx(0.2)
        assert rec.predictive_gain == pytest.approx(0.4)
        assert rec.effort == 0.0
        assert rec.authentic is True
        for key in (
            "dissonance_reduction",
            "identity_gain",
            "predictive_gain",
            "effort_multiplier",
            "raw",
        ):
            assert key in rec.components, f"missing component {key}"

    def test_dissonance_reduction_is_clamped_at_zero(self):
        """Increased dissonance contributes nothing (max(0, pre - post))."""
        eng = _engine()
        rec = eng.compute_meaning(0, pre_dissonance=0.1, post_dissonance=0.9,
                                  pre_identity=0.5, post_identity=0.5)
        assert rec.coherence_gain == 0.0
        assert rec.raw_meaning == pytest.approx(0.0)

    def test_identity_gain_is_clamped_at_zero(self):
        """Lost identity coherence contributes nothing."""
        eng = _engine()
        rec = eng.compute_meaning(0, pre_dissonance=0.5, post_dissonance=0.5,
                                  pre_identity=0.9, post_identity=0.1)
        assert rec.identity_coherence_gain == 0.0

    def test_dissonance_weight_responds_to_config(self):
        """Raising w_dissonance_reduction raises raw meaning for the same event."""
        low = _engine(w_dissonance_reduction=0.1)
        high = _engine(w_dissonance_reduction=0.9)
        kwargs = dict(episode=0, pre_dissonance=1.0, post_dissonance=0.0,
                      pre_identity=0.0, post_identity=0.0, effort=0.0)
        rec_low = low.compute_meaning(**kwargs)
        rec_high = high.compute_meaning(**kwargs)
        assert rec_low.raw_meaning == pytest.approx(0.1 * 1.0)
        assert rec_high.raw_meaning == pytest.approx(0.9 * 1.0)
        assert rec_high.raw_meaning > rec_low.raw_meaning

    def test_identity_weight_responds_to_config(self):
        low = _engine(w_identity_coherence=0.0)
        high = _engine(w_identity_coherence=0.5)
        kwargs = dict(episode=0, pre_dissonance=0.0, post_dissonance=0.0,
                      pre_identity=0.2, post_identity=0.8, effort=0.0)
        assert low.compute_meaning(**kwargs).raw_meaning == pytest.approx(0.0)
        assert high.compute_meaning(**kwargs).raw_meaning == pytest.approx(0.3)

    def test_predictive_weight_responds_to_config(self):
        low = _engine(w_predictive_power=0.0)
        high = _engine(w_predictive_power=0.5)
        kwargs = dict(episode=0, pre_dissonance=0.0, post_dissonance=0.0,
                      pre_identity=0.0, post_identity=0.0, effort=0.0)
        assert low.compute_meaning(predictive_gain=1.0, **kwargs).raw_meaning == 0.0
        assert high.compute_meaning(predictive_gain=1.0, **kwargs).raw_meaning == pytest.approx(0.5)

    def test_effort_amplifies_meaning(self):
        """Costly gains are worth more: effort_kappa scales effective_meaning."""
        cheap = _engine()
        costly = _engine()
        kwargs = dict(episode=0, pre_dissonance=1.0, post_dissonance=0.0,
                      pre_identity=0.0, post_identity=0.0)
        rec_cheap = cheap.compute_meaning(effort=0.0, **kwargs)
        rec_costly = costly.compute_meaning(effort=1.0, **kwargs)
        assert rec_cheap.components["effort_multiplier"] == pytest.approx(1.0)
        assert rec_costly.components["effort_multiplier"] == pytest.approx(1.5)
        assert rec_costly.effective_meaning > rec_cheap.effective_meaning

    def test_effort_kappa_responds_to_config(self):
        low = _engine(effort_kappa=0.0)
        high = _engine(effort_kappa=2.0)
        kwargs = dict(episode=0, pre_dissonance=1.0, post_dissonance=0.0,
                      pre_identity=0.0, post_identity=0.0, effort=0.5)
        assert low.compute_meaning(**kwargs).effective_meaning == pytest.approx(
            low.compute_meaning(**kwargs).raw_meaning)
        assert high.compute_meaning(**kwargs).effective_meaning > low.compute_meaning(**kwargs).effective_meaning

    def test_inauthentic_high_effort_is_penalised(self):
        """High effort with no real gain is flagged inauthentic and halved."""
        eng = _engine()
        rec = eng.compute_meaning(0, pre_dissonance=0.0, post_dissonance=0.0,
                                  pre_identity=0.0, post_identity=0.0,
                                  predictive_gain=0.0, effort=1.0)
        assert rec.raw_meaning == pytest.approx(0.0)
        assert rec.authentic is False
        assert rec.effective_meaning == pytest.approx(0.0)

    def test_authentic_effortful_gain_is_not_penalised(self):
        eng = _engine()
        rec = eng.compute_meaning(0, pre_dissonance=1.0, post_dissonance=0.0,
                                  pre_identity=0.0, post_identity=0.0,
                                  predictive_gain=0.0, effort=1.0)
        assert rec.raw_meaning >= 0.05
        assert rec.authentic is True

    def test_predictive_gain_is_ema_smoothed(self):
        """A lone high gain is smoothed against the trailing window of 10."""
        eng = _engine()
        for i in range(9):
            eng.compute_meaning(i, 0.0, 0.0, 0.0, 0.0, predictive_gain=0.0)
        rec = eng.compute_meaning(9, 0.0, 0.0, 0.0, 0.0, predictive_gain=1.0)
        # mean of [0]*9 + [1.0]
        assert rec.predictive_gain == pytest.approx(0.1)

    def test_predictive_window_is_bounded_at_ten(self):
        eng = _engine()
        for i in range(25):
            eng.compute_meaning(i, 0.0, 0.0, 0.0, 0.0, predictive_gain=0.0)
        assert len(eng._recent_predictive_gains) == 10

    def test_accumulated_meaning_grows_and_never_negative(self):
        eng = _engine()
        eng.compute_meaning(0, 1.0, 0.0, 0.0, 0.0, effort=0.0)
        first = eng.accumulated_meaning
        assert first > 0.0
        eng.compute_meaning(1, 1.0, 0.0, 0.0, 0.0, effort=0.0)
        assert eng.accumulated_meaning > first

    def test_history_is_capped_at_max_history(self):
        eng = _engine(max_history=5)
        for i in range(12):
            eng.compute_meaning(i, 1.0, 0.0, 0.0, 0.0)
        assert len(eng.history) == 5
        # the cap keeps the NEWEST records
        assert [r.episode for r in eng.history] == [7, 8, 9, 10, 11]


class TestMeaningStakes:
    def test_stake_then_held_costs_nothing(self):
        eng = _engine()
        eng.stake_meaning("b1", 2.0)
        assert eng._commitments["b1"] == pytest.approx(2.0)
        assert eng.resolve_stake("b1", belief_held=True) == pytest.approx(0.0)
        # the stake is consumed either way
        assert "b1" not in eng._commitments

    def test_stake_then_falsified_deducts_half(self):
        eng = _engine()
        eng.accumulated_meaning = 5.0
        eng.stake_meaning("b2", 2.0)
        loss = eng.resolve_stake("b2", belief_held=False)
        assert loss == pytest.approx(-1.0)
        assert eng.accumulated_meaning == pytest.approx(4.0)
        assert "b2" not in eng._commitments

    def test_falsified_stake_cannot_drive_meaning_negative(self):
        eng = _engine()
        eng.accumulated_meaning = 0.2
        eng.stake_meaning("b3", 10.0)
        eng.resolve_stake("b3", belief_held=False)
        assert eng.accumulated_meaning == 0.0

    def test_repeated_staking_accumulates(self):
        eng = _engine()
        eng.stake_meaning("b4", 1.0)
        eng.stake_meaning("b4", 0.5)
        assert eng._commitments["b4"] == pytest.approx(1.5)
        assert eng.resolve_stake("b4", belief_held=False) == pytest.approx(-0.75)

    def test_resolving_unknown_belief_is_a_noop(self):
        eng = _engine()
        eng.accumulated_meaning = 1.0
        assert eng.resolve_stake("never-staked", belief_held=False) == 0.0
        assert eng.accumulated_meaning == 1.0

    def test_second_resolve_of_same_belief_does_not_double_charge(self):
        eng = _engine()
        eng.accumulated_meaning = 4.0
        eng.stake_meaning("b5", 2.0)
        first = eng.resolve_stake("b5", belief_held=False)
        second = eng.resolve_stake("b5", belief_held=False)
        assert first == pytest.approx(-1.0)
        assert second == 0.0
        assert eng.accumulated_meaning == pytest.approx(3.0)


class TestExpectedMeaning:
    def test_aggregates_the_three_gain_terms(self):
        eng = _engine(w_dissonance_reduction=0.4, w_identity_coherence=0.3,
                      w_predictive_power=0.3, effort_kappa=0.5)
        val = eng.get_expected_meaning(
            predicted_dissonance_gain=1.0,
            predicted_identity_gain=1.0,
            predicted_predictive_gain=1.0,
            estimated_effort=0.0,
        )
        assert val == pytest.approx(1.0)

    def test_held_beats_contradicted_on_expected_meaning(self):
        """The curiosity signal must prefer the held hypothesis over the contradicted one."""
        eng = _engine()
        held = eng.get_expected_meaning(1.0, 0.5, 0.5, 0.0)
        contradicted = eng.get_expected_meaning(0.0, 0.0, 0.0, 0.0)
        assert held > contradicted

    def test_effort_amplifies_expected_meaning(self):
        eng = _engine()
        cheap = eng.get_expected_meaning(0.5, 0.5, 0.5, 0.0)
        costly = eng.get_expected_meaning(0.5, 0.5, 0.5, 1.0)
        assert costly > cheap
        assert costly == pytest.approx(cheap * 1.5)

    def test_no_gain_means_no_expected_meaning(self):
        eng = _engine()
        assert eng.get_expected_meaning(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_prediction_does_not_mutate_history(self):
        """Expectation is a read-only look-ahead, not a commitment."""
        eng = _engine()
        eng.get_expected_meaning(1.0, 1.0, 1.0, 1.0)
        assert eng.history == []
        assert eng.accumulated_meaning == 0.0


class TestGetStatus:
    def test_empty_engine_status_has_real_keys(self):
        eng = _engine()
        st = eng.get_status()
        assert isinstance(st, dict)
        assert st["accumulated_meaning"] == 0.0
        assert st["active_commitments"] == 0
        assert st["total_episodes_tracked"] == 0
        assert st["recent_meaning_rate"] == 0.0
        assert st["authenticity_rate"] == 1.0

    def test_status_tracks_episodes_and_meaning_rate(self):
        eng = _engine()
        eng.compute_meaning(0, 1.0, 0.0, 0.0, 0.0)
        eng.compute_meaning(1, 1.0, 0.0, 0.0, 0.0)
        st = eng.get_status()
        assert st["total_episodes_tracked"] == 2
        assert st["recent_meaning_rate"] > 0.0
        assert st["accumulated_meaning"] == pytest.approx(
            sum(r.effective_meaning for r in eng.history))

    def test_status_counts_active_commitments(self):
        eng = _engine()
        eng.stake_meaning("x", 1.0)
        eng.stake_meaning("y", 1.0)
        eng.resolve_stake("x", belief_held=True)
        st = eng.get_status()
        assert st["active_commitments"] == 1

    def test_authenticity_rate_reflects_penalised_episodes(self):
        eng = _engine()
        eng.compute_meaning(0, 1.0, 0.0, 0.0, 0.0)          # authentic
        eng.compute_meaning(1, 0.0, 0.0, 0.0, 0.0, effort=1.0)  # inauthentic
        st = eng.get_status()
        assert st["authenticity_rate"] == pytest.approx(0.5)

    def test_status_window_is_limited_to_last_twenty(self):
        eng = _engine()
        for i in range(30):
            eng.compute_meaning(i, 1.0, 0.0, 0.0, 0.0)
        st = eng.get_status()
        assert st["total_episodes_tracked"] == 30
        expected = sum(r.effective_meaning for r in eng.history[-20:]) / 20
        assert st["recent_meaning_rate"] == pytest.approx(expected)
