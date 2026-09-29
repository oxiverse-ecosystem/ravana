"""Dedicated test suite for `MeaningEngine` (ravana-v2/src/ravana_grace/core/meaning.py).

Backlog item #3: `core/meaning.py` was the one YELLOW row in
docs/ACCEPTANCE_LEDGER.md whose stated reason was true — no test file
matching "meaning" existed anywhere under tests/ (the only trace was a
stale tests/unit/__pycache__/test_meaning.cpython-311-pytest-9.1.1.pyc
left behind by a run that lost its source).

Import path matches ravana/src/ravana/chat/engine.py so the test exercises
the real module the engine boots, not a copy. Every test uses `assert`:
a test that RETURNS a bool is always green under pytest, a defect class
this repo has hit before.
"""
import os
import sys

import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana_grace.core.meaning import (  # noqa: E402
    MeaningConfig,
    MeaningEngine,
    MeaningRecord,
)


def _engine(**cfg):
    return MeaningEngine(MeaningConfig(**cfg))


# ── compute_meaning: the formula responds to its own config weight ─────────
def test_compute_meaning_returns_a_meaning_record():
    eng = _engine()
    rec = eng.compute_meaning(
        episode=0, pre_dissonance=0.8, post_dissonance=0.2,
        pre_identity=0.4, post_identity=0.7, predictive_gain=0.3,
        effort=0.2)
    assert isinstance(rec, MeaningRecord)
    assert rec.episode == 0
    assert rec.effort == 0.2


def test_dissonance_reduction_component_tracks_the_config_weight():
    """The w_dissonance_reduction term must actually drive raw_meaning."""
    lo = _engine(w_dissonance_reduction=0.0)
    hi = _engine(w_dissonance_reduction=1.0)
    kwargs = dict(episode=0, pre_dissonance=0.9, post_dissonance=0.1,
                  pre_identity=0.5, post_identity=0.5, predictive_gain=0.0,
                  effort=0.0)
    r_lo = lo.compute_meaning(**kwargs)
    r_hi = hi.compute_meaning(**kwargs)
    # 0.9 -> 0.1 is a reduction of 0.8; with weight 1.0 that is the whole
    # raw meaning, with weight 0.0 it contributes nothing.
    assert r_lo.raw_meaning == pytest.approx(0.0, abs=1e-9)
    assert r_hi.raw_meaning == pytest.approx(0.8, abs=1e-9)
    assert r_hi.raw_meaning > r_lo.raw_meaning


def test_increased_dissonance_yields_no_meaning():
    """Meaning is COSTLY COHERENCE GAIN: a regression is clamped to zero."""
    eng = _engine()
    rec = eng.compute_meaning(
        episode=0, pre_dissonance=0.1, post_dissonance=0.9,
        pre_identity=0.5, post_identity=0.5, predictive_gain=0.0, effort=0.0)
    assert rec.coherence_gain == 0.0
    assert rec.raw_meaning == pytest.approx(0.0, abs=1e-9)


def test_effort_amplifies_meaning():
    """Costly gains are worth more: effort scales the same raw meaning."""
    eng = _engine()
    base = dict(episode=0, pre_dissonance=1.0, post_dissonance=0.0,
                pre_identity=0.5, post_identity=0.5, predictive_gain=0.0)
    cheap = eng.compute_meaning(effort=0.0, **base)
    costly = eng.compute_meaning(effort=1.0, **base)
    assert costly.effective_meaning > cheap.effective_meaning
    assert costly.components["effort_multiplier"] == pytest.approx(1.5)
    assert cheap.components["effort_multiplier"] == pytest.approx(1.0)


def test_high_effort_with_no_real_gain_is_flagged_inauthentic():
    """Authenticity gate: effort alone must not manufacture meaning.

    The gate is `raw_meaning < 0.05 and effort_multiplier > 1.5`; with the
    default effort_kappa=0.5 that needs effort > 1.0. The magnitude of the
    penalty is asserted in test_inauthentic_penalty_halves_a_small_gain.
    """
    eng = _engine()
    rec = eng.compute_meaning(
        episode=0, pre_dissonance=0.0, post_dissonance=0.0,
        pre_identity=0.5, post_identity=0.5, predictive_gain=0.0, effort=2.0)
    assert rec.raw_meaning < 0.05
    assert rec.components["effort_multiplier"] > 1.5
    assert rec.authentic is False
    # Effort bought nothing: no meaning was produced.
    assert rec.effective_meaning == 0.0


def test_inauthentic_penalty_halves_a_small_gain():
    eng = _engine()
    # reduction 0.1 * default weight 0.4 == raw 0.04, just under the 0.05 gate.
    rec = eng.compute_meaning(
        episode=0, pre_dissonance=0.1, post_dissonance=0.0,
        pre_identity=0.5, post_identity=0.5, predictive_gain=0.0, effort=2.0)
    unpenalised = rec.raw_meaning * rec.components["effort_multiplier"]
    assert rec.raw_meaning == pytest.approx(0.04, abs=1e-9)
    assert rec.authentic is False
    assert rec.effective_meaning == pytest.approx(unpenalised * 0.5, abs=1e-9)


def test_moderate_effort_with_no_gain_is_still_authentic():
    """The authenticity gate must not fire on ordinary effort."""
    eng = _engine()
    rec = eng.compute_meaning(
        episode=0, pre_dissonance=0.0, post_dissonance=0.0,
        pre_identity=0.5, post_identity=0.5, predictive_gain=0.0, effort=1.0)
    assert rec.authentic is True


def test_predictive_gain_is_smoothed_over_recent_episodes():
    """The EMA window means a single spike is not the whole signal."""
    eng = _engine()
    for i in range(10):
        eng.compute_meaning(
            episode=i, pre_dissonance=0.5, post_dissonance=0.5,
            pre_identity=0.5, post_identity=0.5, predictive_gain=0.0,
            effort=0.0)
    spike = eng.compute_meaning(
        episode=10, pre_dissonance=0.5, post_dissonance=0.5,
        pre_identity=0.5, post_identity=0.5, predictive_gain=1.0, effort=0.0)
    # 1.0 spread over the 10-deep window == 0.1, not 1.0.
    assert spike.predictive_gain == pytest.approx(0.1, abs=1e-6)


def test_accumulated_meaning_is_the_running_sum_of_effective_values():
    eng = _engine()
    total = 0.0
    for i in range(5):
        total += eng.compute_meaning(
            episode=i, pre_dissonance=1.0, post_dissonance=0.0,
            pre_identity=0.5, post_identity=0.5, predictive_gain=0.0,
            effort=0.0).effective_meaning
    assert eng.accumulated_meaning == pytest.approx(total, abs=1e-9)


# ── stake_meaning / resolve_stake round-trip ──────────────────────────────
def test_held_belief_preserves_staked_meaning():
    eng = _engine()
    eng.stake_meaning("b1", 4.0)
    assert eng.resolve_stake("b1", belief_held=True) == 0.0
    # Resolved: the commitment is no longer outstanding.
    assert eng.get_status()["active_commitments"] == 0


def test_falsified_belief_costs_half_the_stake():
    eng = _engine()
    eng.compute_meaning(
        episode=0, pre_dissonance=1.0, post_dissonance=0.0,
        pre_identity=0.5, post_identity=0.5, predictive_gain=0.0, effort=0.0)
    before = eng.accumulated_meaning
    assert before > 0.0
    # Stake less than the accumulated meaning: resolve_stake floors
    # accumulated_meaning at 0, so an oversized loss would clamp and hide
    # the arithmetic. (The clamp itself is covered by
    # test_accumulated_meaning_never_goes_negative.)
    eng.stake_meaning("b1", 0.4)
    loss = eng.resolve_stake("b1", belief_held=False)
    assert loss == pytest.approx(-0.2)
    assert eng.accumulated_meaning == pytest.approx(before - 0.2)


def test_stake_accumulates_and_resolves_once():
    eng = _engine()
    eng.stake_meaning("b1", 1.0)
    eng.stake_meaning("b1", 3.0)
    assert eng.get_status()["active_commitments"] == 1
    assert eng.resolve_stake("b1", belief_held=False) == pytest.approx(-2.0)
    # A second resolve pops nothing and must not double-charge.
    assert eng.resolve_stake("b1", belief_held=False) == 0.0


def test_resolving_an_unstaked_belief_is_a_no_op():
    eng = _engine()
    assert eng.resolve_stake("never-staked", belief_held=False) == 0.0
    assert eng.accumulated_meaning == 0.0


def test_accumulated_meaning_never_goes_negative():
    eng = _engine()
    eng.stake_meaning("b1", 100.0)
    eng.resolve_stake("b1", belief_held=False)
    assert eng.accumulated_meaning >= 0.0


# ── get_expected_meaning aggregates the same three terms ──────────────────
def test_expected_meaning_matches_the_compute_formula():
    """A prediction of a real gain must predict the meaning actually earned."""
    cfg = MeaningConfig()
    eng = MeaningEngine(cfg)
    predicted = eng.get_expected_meaning(
        predicted_dissonance_gain=0.8, predicted_identity_gain=0.0,
        predicted_predictive_gain=0.0, estimated_effort=0.0)
    actual = eng.compute_meaning(
        episode=0, pre_dissonance=0.8, post_dissonance=0.0,
        pre_identity=0.5, post_identity=0.5, predictive_gain=0.0,
        effort=0.0).raw_meaning
    assert predicted == pytest.approx(actual, abs=1e-9)


def test_expected_meaning_rises_with_effort():
    eng = _engine()
    cheap = eng.get_expected_meaning(0.5, 0.5, 0.5, estimated_effort=0.0)
    costly = eng.get_expected_meaning(0.5, 0.5, 0.5, estimated_effort=1.0)
    assert costly > cheap


def test_expected_meaning_is_zero_for_no_predicted_gain():
    eng = _engine()
    assert eng.get_expected_meaning(0.0, 0.0, 0.0, estimated_effort=0.5) == 0.0


# ── get_status reports real keys derived from real state ──────────────────
def test_get_status_on_a_fresh_engine():
    eng = _engine()
    status = eng.get_status()
    assert status["accumulated_meaning"] == 0.0
    assert status["active_commitments"] == 0
    assert status["total_episodes_tracked"] == 0
    assert status["recent_meaning_rate"] == 0.0
    assert status["authenticity_rate"] == 1.0


def test_get_status_tracks_episodes_and_commitments():
    eng = _engine()
    for i in range(3):
        eng.compute_meaning(
            episode=i, pre_dissonance=1.0, post_dissonance=0.0,
            pre_identity=0.5, post_identity=0.5, predictive_gain=0.0,
            effort=0.0)
    eng.stake_meaning("b1", 2.0)
    eng.stake_meaning("b2", 1.0)
    status = eng.get_status()
    assert status["total_episodes_tracked"] == 3
    assert status["active_commitments"] == 2
    assert status["recent_meaning_rate"] > 0.0
    assert 0.0 <= status["authenticity_rate"] <= 1.0


def test_history_is_bounded_by_max_history():
    eng = _engine(max_history=5)
    for i in range(12):
        eng.compute_meaning(
            episode=i, pre_dissonance=0.5, post_dissonance=0.0,
            pre_identity=0.5, post_identity=0.5, predictive_gain=0.0,
            effort=0.0)
    assert len(eng.history) == 5
    # The window keeps the NEWEST records.
    assert eng.history[-1].episode == 11
