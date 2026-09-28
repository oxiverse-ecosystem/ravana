"""Contract tests for online epistemic calibration (chat/calibration.py).

Every test uses a bare ``assert``. A test that RETURNS a bool is reported
green by pytest regardless of its value — a defect class this repo has shipped
before (``test_temporal_grounding.py`` had 3 genuinely failing tests invisible
under pytest).

The tests exercise the module's CONTRACT, not its current behaviour, so they
encode what the capability is for:

* pairing a prediction with its outcome (the pair the engine never made),
* learning a signed over/under-confidence bias online,
* turning that bias into a shifted assert-gate, and
* staying the identity at cold start, so wiring it in changes nothing until
  real experience arrives.
"""

import math

import pytest

from ravana.chat.calibration import (
    BAND_EDGES,
    THETA_MAX,
    THETA_MIN,
    EpistemicCalibrator,
    band_index,
)


# ── cold start: the identity property ─────────────────────────────────

def test_cold_start_bias_is_zero():
    c = EpistemicCalibrator()
    assert c.n == 0
    assert c.mean_bias() == 0.0
    assert c.bias() == 0.0
    assert c.brier() == 0.0


def test_cold_start_theta_is_exactly_base():
    """The safety property that makes this feature safe to wire in.

    With no observations the learned bias is 0, so the assert-gate must be
    byte-identical to the uncalibrated value. A gate that moves on a cold
    engine would change every reply before RAVANA has learned anything.
    """
    c = EpistemicCalibrator(theta_base=0.30)
    assert c.theta_withhold() == 0.30
    # and for an arbitrary caller-supplied base
    assert c.theta_withhold(0.47) == 0.47


def test_cold_start_reliability_curve_is_all_absent():
    c = EpistemicCalibrator()
    curve = c.reliability()
    assert len(curve) == len(BAND_EDGES) - 1
    assert all(entry is None for entry in curve)


def test_cold_start_is_not_claimed_calibrated():
    """One observation must never be enough to declare calibration."""
    c = EpistemicCalibrator()
    c.observe(0.5, 0.5)
    assert c.n == 1
    assert c.is_calibrated() is False


# ── the core capability: overconfidence is learned ────────────────────

def test_perfectly_calibrated_stream_leaves_theta_unchanged():
    c = EpistemicCalibrator(theta_base=0.30)
    for _ in range(20):
        c.observe(0.60, 0.60)
    assert c.mean_bias() == pytest.approx(0.0, abs=1e-9)
    assert c.theta_withhold() == pytest.approx(0.30)


def test_overconfident_stream_raises_the_assert_gate():
    """The payoff: predicting 0.8 and delivering 0.4 must make RAVANA
    hedge more, i.e. the bar to assert flatly rises."""
    c = EpistemicCalibrator(theta_base=0.30)
    for _ in range(20):
        c.observe(0.80, 0.40)
    assert c.mean_bias() == pytest.approx(0.40)
    assert c.bias() > 0.0
    assert c.theta_withhold() > 0.30


def test_underconfident_stream_lowers_the_assert_gate():
    c = EpistemicCalibrator(theta_base=0.30)
    for _ in range(20):
        c.observe(0.20, 0.60)
    assert c.mean_bias() == pytest.approx(-0.40)
    assert c.theta_withhold() < 0.30


def test_bias_magnitude_is_the_mean_signed_gap():
    """A mixed stream must average, not max out."""
    c = EpistemicCalibrator()
    c.observe(0.9, 0.1)   # +0.8
    c.observe(0.3, 0.3)   # +0.0
    c.observe(0.1, 0.5)   # -0.4
    assert c.mean_bias() == pytest.approx((0.8 + 0.0 - 0.4) / 3)


def test_hard_easy_effect_biases_the_gate_by_difficulty():
    """Kornell (2009): the same stated confidence can hide very different
    reliability. The gate tracks the *signed* bias, so it is the Brier score
    that exposes what a symmetric but erratic reliability curve looks like.

    Three engines all state 0.7 every turn. One is well calibrated, one is
    erratic-but-unbiased (outcomes symmetric about 0.7), one is steadily
    overconfident. Only the third moves the gate; the second is caught only
    by the Brier score. Asserting all three keeps the diagnostic honest
    instead of letting the signed bias look like a complete picture.
    """
    easy = EpistemicCalibrator(theta_base=0.30)
    for _ in range(20):
        easy.observe(0.7, 0.7)                     # reliable

    erratic = EpistemicCalibrator(theta_base=0.30)
    for i in range(20):
        erratic.observe(0.7, 1.0 if i % 2 else 0.4)  # symmetric about 0.7

    over = EpistemicCalibrator(theta_base=0.30)
    for _ in range(20):
        over.observe(0.7, 0.2)                      # same stated conf, poor

    assert easy.theta_withhold() == pytest.approx(0.30)
    assert erratic.theta_withhold() == pytest.approx(0.30, abs=1e-9)
    assert over.theta_withhold() > 0.30
    # the gate is blind to the erratic case, by design; Brier is not
    assert erratic.brier() > easy.brier()
    assert over.brier() > erratic.brier()


# ── shrinkage: stability, not jitter ──────────────────────────────────

def test_single_observation_barely_moves_the_gate():
    """One bad turn must not swing the assert-gate.

    Without shrinkage a single overconfident reply would immediately shift
    the gate by its full magnitude; the shrinkage weight makes the response
    gradual and proportional to evidence.
    """
    c = EpistemicCalibrator(theta_base=0.30, prior_n=10.0)
    c.observe(1.0, 0.0)
    assert c.mean_bias() == pytest.approx(1.0)
    # shrunk: 1.0 * 1/(1+10)
    assert c.bias() == pytest.approx(1.0 / 11.0)
    assert c.theta_withhold() < 0.30 + 0.10


def test_shrinkage_weight_is_n_over_n_plus_prior():
    c = EpistemicCalibrator(prior_n=4.0)
    for _ in range(8):
        c.observe(0.9, 0.1)
    assert c.bias() == pytest.approx(c.mean_bias() * (8 / 12))


def test_gate_converges_as_evidence_accumulates():
    """The gate should approach base + raw_bias as evidence accumulates,
    never snapping. Uses a modest gap (0.9/0.5 → bias 0.4) so the
    converged value stays inside the structural bounds and the convergence
    target is the real asymptote rather than a clamped one."""
    c = EpistemicCalibrator(theta_base=0.30, prior_n=10.0)
    thetas = []
    for _ in range(1, 41):
        c.observe(0.9, 0.5)
        thetas.append(c.theta_withhold())
    assert thetas[0] < thetas[-1]
    # asymptote: 0.30 + 0.4 * (n/(n+10)) as n → ∞ is 0.30 + 0.4
    assert thetas[-1] == pytest.approx(0.30 + 0.4 * (40 / 50))
    assert thetas[-1] < 0.30 + 0.4
    # monotone non-decreasing while the same-signed evidence keeps arriving
    assert all(b >= a - 1e-12 for a, b in zip(thetas, thetas[1:]))


def test_gate_converges_to_the_clamped_bound_when_evidence_is_extreme():
    """A wildly overconfident stream must still produce a legal gate, not an
    unbounded one — the clamp is a real behaviour, so it gets a real test."""
    c = EpistemicCalibrator(theta_base=0.30, prior_n=10.0)
    for _ in range(40):
        c.observe(0.9, 0.1)
    assert c.theta_withhold() == pytest.approx(THETA_MAX)


# ── structural bounds ─────────────────────────────────────────────────

def test_theta_is_clamped_to_structural_bounds():
    over = EpistemicCalibrator(theta_base=0.30)
    for _ in range(200):
        over.observe(1.0, 0.0)
    assert over.theta_withhold() <= THETA_MAX

    under = EpistemicCalibrator(theta_base=0.30)
    for _ in range(200):
        under.observe(0.0, 1.0)
    assert under.theta_withhold() >= THETA_MIN


# ── the reliability curve is a store that grows online ────────────────

def test_bands_fill_in_as_experience_arrives():
    c = EpistemicCalibrator()
    assert all(e is None for e in c.reliability())

    c.observe(0.9, 0.2)   # band 4 (0.8-1.0)
    curve = c.reliability()
    assert curve[4] is not None
    assert curve[4]["observed"] == pytest.approx(0.2)
    assert curve[4]["n"] == 1
    # every other band is still absent — unvisited is not zero
    assert all(curve[i] is None for i in (0, 1, 2, 3))


def test_reliability_curve_reports_the_reliability_diagram():
    """The point of the curve: stated confidence vs actual hit-rate.

    Feed 10 predictions at 0.9 that land 0.3, and the ledger must report a
    band observed-rate of 0.3 — a visible gap between what RAVANA claimed and
    what it delivered. That gap is the finding, and it is derived purely from
    the accumulated store.
    """
    c = EpistemicCalibrator()
    for _ in range(10):
        c.observe(0.9, 0.3)
    band = c.reliability()[4]
    assert band["observed"] == pytest.approx(0.3)
    assert band["observed"] < band["lo"]  # the miscalibration is visible


def test_band_counts_sum_to_observation_count():
    c = EpistemicCalibrator()
    for conf in (0.05, 0.15, 0.25, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0):
        c.observe(conf, 0.5)
    total = sum(e["n"] for e in c.reliability() if e is not None)
    assert total == c.n == 10


def test_band_index_edges_are_unambiguous():
    assert band_index(0.0) == 0
    assert band_index(0.19) == 0
    assert band_index(0.2) == 1     # interior edge goes to the upper band
    assert band_index(0.99) == 4
    assert band_index(1.0) == 4     # closed upper edge, not an IndexError
    # out-of-range input clamps instead of raising
    assert band_index(-5.0) == 0
    assert band_index(99.0) == 4


def test_bands_visited_grows_with_diversity_of_experience():
    c = EpistemicCalibrator()
    c.observe(0.1, 0.1)
    assert c.get_status()["bands_visited"] == 1
    c.observe(0.5, 0.5)
    assert c.get_status()["bands_visited"] == 2
    c.observe(0.9, 0.9)
    assert c.get_status()["bands_visited"] == 3
    assert c.get_status()["bands_total"] == len(BAND_EDGES) - 1


# ── robustness: a bad metric must not poison the ledger ───────────────

def test_out_of_range_values_are_clamped_at_the_edges():
    """An upstream metric drifting out of range must be clamped, not stored
    raw — otherwise one bad metric permanently skews the running bias."""
    c = EpistemicCalibrator()
    c.observe(5.0, 0.0)      # predicted clamps to 1.0
    assert c.mean_bias() == pytest.approx(1.0)

    d = EpistemicCalibrator()
    d.observe(0.0, -2.0)     # outcome clamps to 0.0
    assert d.mean_bias() == pytest.approx(0.0)

    e = EpistemicCalibrator()
    e.observe(5.0, -2.0)     # both clamp -> the widest possible pair
    assert e.mean_bias() == pytest.approx(1.0)
    assert e.reliability()[4]["observed"] == pytest.approx(0.0)


def test_nan_observation_is_dropped_and_does_not_poison_sums():
    c = EpistemicCalibrator()
    c.observe(0.9, 0.3)
    c.observe(float("nan"), 0.3)
    c.observe(0.9, 0.3)
    assert c.n == 2
    assert c.reliability()[4]["observed"] == pytest.approx(0.3)
    assert not math.isnan(c.bias())


def test_window_rolls_off_old_observations():
    c = EpistemicCalibrator(window=10)
    for _ in range(50):
        c.observe(0.9, 0.9)     # well calibrated
    assert c.n == 10
    assert c.get_status()["total_observed"] == 50
    for _ in range(10):
        c.observe(0.9, 0.1)    # recent evidence: overconfident
    assert c.n == 10           # window respected
    assert c.get_status()["total_observed"] == 60
    assert c.mean_bias() == pytest.approx(0.8)   # only the new evidence remains


# ── brier: the diagnostic the signed bias hides ───────────────────────

def test_brier_is_zero_for_perfect_predictions():
    c = EpistemicCalibrator()
    for _ in range(10):
        c.observe(0.7, 0.7)
    assert c.brier() == pytest.approx(0.0)


def test_brier_detects_symmetric_uncertainty_that_bias_hides():
    c = EpistemicCalibrator()
    for i in range(10):
        c.observe(0.5, 1.0 if i % 2 else 0.0)
    assert c.mean_bias() == pytest.approx(0.0)   # no net bias
    assert c.brier() == pytest.approx(0.25)      # but badly wrong


# ── observability ─────────────────────────────────────────────────────

def test_get_status_shape():
    c = EpistemicCalibrator(theta_base=0.30)
    for _ in range(20):
        c.observe(0.8, 0.4)
    st = c.get_status()
    for key in ("n", "total_observed", "mean_bias", "bias", "brier",
                "theta_withhold", "bands_visited", "bands_total", "calibrated"):
        assert key in st, key
    assert st["n"] == 20
    assert st["total_observed"] == 20
    assert st["bands_visited"] >= 1
    assert st["calibrated"] is False   # bias 0.4 is not calibrated


# ── persistence: the ledger must survive a save/load ─────────────────

def test_state_round_trip():
    c = EpistemicCalibrator(theta_base=0.42, window=25, prior_n=7.0)
    for _ in range(20):
        c.observe(0.85, 0.35)
    state = c.get_state()

    r = EpistemicCalibrator()
    assert r.load_state(state) is True
    assert r.n == c.n
    assert r.theta_base == pytest.approx(0.42)
    assert r.window == 25
    assert r.prior_n == pytest.approx(7.0)
    assert r.bias() == pytest.approx(c.bias())
    assert r.theta_withhold() == pytest.approx(c.theta_withhold())
    assert r.reliability() == c.reliability()


def test_load_state_fails_closed_on_garbage():
    c = EpistemicCalibrator()
    assert c.load_state(None) is False
    assert c.load_state("not a dict") is False
    assert c.load_state({}) is False
    assert c.load_state({"pairs": [(0.1, 0.2)]}) is False   # band arrays missing
    assert c.load_state({"pairs": "junk", "band_sum": [0.0] * 5,
                         "band_n": [0] * 5}) is False
    # a failed load must leave the ledger cold, not half-restored
    assert c.n == 0
    assert c.theta_withhold() == pytest.approx(0.30)


def test_learned_state_survives_into_a_fresh_ledger_and_still_shifts_the_gate():
    """End-to-end: experience accrued in one ledger changes the behaviour of
    a *new* ledger restored from it — i.e. the learning is durable, not a
    per-turn transient."""
    first = EpistemicCalibrator(theta_base=0.30)
    for _ in range(30):
        first.observe(0.95, 0.15)
    restored = EpistemicCalibrator()
    restored.load_state(first.get_state())
    assert restored.theta_withhold() > 0.30
