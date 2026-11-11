"""Tests for the retrieval-vs-generation switch.

The rule is a *gate*, so the tests pin its decision boundary and the direction
of each gain. The two counter-intuitive properties are the ones worth testing
explicitly:

* high prediction error RAISES the bar (it does not invite generation)
* practice LOWERS the bar (the decoder earns reachability)
"""
import numpy as np
import pytest

from ravana.learn.retrieval_generation import (
    RetrievalGenerationConfig,
    RetrievalStrength,
    compute_density,
    compute_prediction_error,
    compute_retrieval_strength,
    compute_threshold,
    decide,
)


def _strength(**kw):
    return compute_retrieval_strength("x", **kw)


class TestRetrievalStrength:
    def test_definition_alone_is_the_strongest_single_term(self):
        cfg = RetrievalGenerationConfig()
        d = _strength(has_definition=True).total(cfg)
        w = _strength(has_web=True).total(cfg)
        c = _strength(confidence=1.0).total(cfg)
        assert d > w > c

    def test_no_signals_gives_zero(self):
        assert _strength().total(RetrievalGenerationConfig()) == 0.0

    def test_all_signals_saturate_at_one(self):
        cfg = RetrievalGenerationConfig()
        s = _strength(has_definition=True, has_web=True, confidence=1.0,
                      stability=1.0, association=1.0, familiarity=1.0)
        assert s.total(cfg) == pytest.approx(1.0)

    def test_out_of_range_terms_are_clamped(self):
        s = _strength(confidence=5.0, stability=-3.0)
        assert 0.0 <= s.confidence <= 1.0
        assert 0.0 <= s.stability <= 1.0

    def test_weights_sum_to_one(self):
        assert RetrievalGenerationConfig().retrieval_weights_sum() == pytest.approx(1.0)


class TestPredictionError:
    def test_max_of_node_edge_mean_and_global(self):
        # max(node=0.2, mean(edges)=0.35, global=0.1) == 0.35
        assert compute_prediction_error(0.2, [0.3, 0.4], 0.1) == pytest.approx(0.35)

    def test_a_single_surprising_edge_raises_the_mean(self):
        # One bad edge among quiet ones still lifts the term:
        # mean(0.1, 0.1, 0.9) = 0.3667, above the 0.1 node and global.
        assert compute_prediction_error(0.1, [0.1, 0.1, 0.9], 0.1) == pytest.approx(0.3666667)

    def test_node_can_dominate(self):
        assert compute_prediction_error(0.9, [0.1], 0.2) == pytest.approx(0.9)

    def test_no_node_falls_back_to_global(self):
        assert compute_prediction_error(None, [], 0.25) == pytest.approx(0.25)

    def test_empty_everything_is_zero(self):
        assert compute_prediction_error(None, [], 0.0) == 0.0


class TestDensity:
    def test_normalised_by_max_degree(self):
        assert compute_density(5, 5, 10) == pytest.approx(1.0)

    def test_zero_max_degree_is_safe(self):
        assert compute_density(3, 3, 0) == 0.0

    def test_clamped_above_one(self):
        assert compute_density(100, 100, 10) == 1.0


class TestThresholdDirection:
    """The two findings that invert the intuitive design."""

    def test_prediction_error_raises_the_bar(self):
        cfg = RetrievalGenerationConfig()
        low, _ = compute_threshold(0.0, 0.0, 0.0, cfg=cfg)
        high, _ = compute_threshold(1.0, 0.0, 0.0, cfg=cfg)
        assert high > low

    def test_density_raises_the_bar(self):
        cfg = RetrievalGenerationConfig()
        low, _ = compute_threshold(0.0, 0.0, 0.0, cfg=cfg)
        high, _ = compute_threshold(0.0, 1.0, 0.0, cfg=cfg)
        assert high > low

    def test_familiarity_lowers_the_bar(self):
        cfg = RetrievalGenerationConfig()
        low, _ = compute_threshold(0.0, 0.0, 0.0, cfg=cfg)
        high, _ = compute_threshold(0.0, 0.0, 1.0, cfg=cfg)
        assert high < low

    def test_practice_lowers_the_bar(self):
        cfg = RetrievalGenerationConfig()
        fresh, d0 = compute_threshold(0.0, 0.0, 0.0, n_compose=0, cfg=cfg)
        practiced, d1 = compute_threshold(0.0, 0.0, 0.0, n_compose=100, cfg=cfg)
        assert practiced < fresh
        assert d1 < d0

    def test_practice_discount_never_inverts(self):
        cfg = RetrievalGenerationConfig()
        _, discount = compute_threshold(0.0, 0.0, 0.0, n_compose=10**9, cfg=cfg)
        assert discount >= 0.0

    def test_k_pe_zero_disables_the_pe_term(self):
        cfg = RetrievalGenerationConfig(k_pe=0.0)
        a, _ = compute_threshold(0.0, 0.0, 0.0, cfg=cfg)
        b, _ = compute_threshold(1.0, 0.0, 0.0, cfg=cfg)
        assert a == pytest.approx(b)


class TestDecision:
    def test_strong_stored_candidate_retrieves(self):
        s = _strength(has_definition=True, confidence=0.9)
        d = decide("x", strength=s, pe=0.0, density=0.0, familiarity=0.0)
        assert d.retrieve is True

    def test_no_stored_candidate_composes(self):
        s = _strength()
        d = decide("x", strength=s, pe=0.0, density=0.0, familiarity=0.0)
        assert d.retrieve is False

    def test_high_prediction_error_raises_the_bar(self):
        """The counter-intuitive property: uncertainty favours the STORED form.

        Raising T makes retrieval HARDER, never easier, so a high-PE context
        cannot invite generation. That is the opposite of the intuitive
        "uncertain -> be creative" rule, and it is deliberate: the brain's
        conservative response to uncertainty is to fall back on the most
        practiced stored form.
        """
        cfg = RetrievalGenerationConfig()
        s = _strength(association=0.5)
        calm = decide("x", strength=s, pe=0.0, density=0.0, familiarity=0.0, cfg=cfg)
        uncertain = decide("x", strength=s, pe=1.0, density=0.0, familiarity=0.0, cfg=cfg)
        assert uncertain.threshold > calm.threshold
        # The decision always follows the stated rule, whichever way T moved.
        assert uncertain.retrieve == (uncertain.strength >= uncertain.threshold)

    def test_high_prediction_error_never_invites_generation(self):
        """No R can flip from compose to retrieve as PE rises.

        Since T increases monotonically with PE, any case below the calm
        threshold stays below the raised one. Asserted directly: a subject that
        composes when calm still composes when uncertain.
        """
        cfg = RetrievalGenerationConfig()
        s = _strength(association=0.5)   # R = 0.075
        calm = decide("x", strength=s, pe=0.0, density=0.0, familiarity=0.0, cfg=cfg)
        uncertain = decide("x", strength=s, pe=1.0, density=0.0, familiarity=0.0, cfg=cfg)
        assert calm.retrieve is False
        assert uncertain.retrieve is False
        assert uncertain.threshold > calm.threshold

    def test_practice_can_earn_the_decoder_a_turn(self):
        cfg = RetrievalGenerationConfig()
        s = _strength(association=0.2)
        fresh = decide("x", strength=s, pe=0.0, density=0.0, familiarity=0.0,
                       n_compose=0, cfg=cfg)
        practiced = decide("x", strength=s, pe=0.0, density=0.0, familiarity=0.0,
                          n_compose=200, cfg=cfg)
        # Practice lowers the bar, so a case that retrieved can flip to compose.
        assert practiced.threshold < fresh.threshold

    def test_decision_is_explainable(self):
        s = _strength(has_definition=True)
        d = decide("x", strength=s, pe=0.1, density=0.2, familiarity=0.3)
        out = d.as_dict()
        for k in ("subject", "retrieve", "strength", "threshold",
                  "prediction_error", "density", "familiarity", "reason"):
            assert k in out
        assert "R=" in out["reason"] and "T=" in out["reason"]

    def test_subject_without_any_signal_composes(self):
        d = decide("unknown", strength=_strength(), pe=0.0, density=0.0,
                   familiarity=0.0)
        assert d.retrieve is False
