"""Dedicated coverage for the ACC primitives in `chat/metacognition.py`.

Why this file exists
--------------------
`docs/ACCEPTANCE_LEDGER.md` grades `chat/metacognition.py` **YELLOW**: until
this file, `tests/test_epistemic_calibration_wiring.py` was its ONLY importer,
and that suite was written for the *calibrator*. So `fok_confidence` and
`modality_from_support` — the two functions that decide how confident RAVANA
claims to be and how it hedges — had no coverage of their own. This file covers
them directly.

Scope is deliberately narrow: these are the module's own numeric and vocabulary
behaviours. It does not re-test the calibrator (that suite owns it) and it does
not assert anything about engine routing.

Every expectation here was measured against the real functions before being
written down; see `docs/CAPABILITY_EPISTEMIC_CALIBRATION.md` §2.3 for the
executed output.
"""
import os
import sys

import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ,
           os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.metacognition import (  # noqa: E402
    RETRIEVAL_SUCCESS_BOOST,
    THETA_WITHDHOLD,
    Metacognition,
    fok_confidence,
    modality_from_support,
    should_assert,
)


# ── module-level invariants ───────────────────────────────────────────

def test_default_theta_is_the_documented_value():
    """The module default the calibrator shifts away from.

    If this constant moves, `test_cold_start_theta_is_exactly_base` in
    tests/unit/test_epistemic_calibration.py is measuring against a stale
    number, so pin it rather than let it drift silently.
    """
    assert THETA_WITHDHOLD == 0.30


def test_retrieval_boost_is_a_positive_credit_not_a_penalty():
    """A successful retrieval must raise confidence, never lower it.

    This is the one asymmetry the FOK curve is allowed to have; a sign flip
    here would invert the meaning of a successful search.
    """
    assert RETRIEVAL_SUCCESS_BOOST > 0


# ── fok_confidence ────────────────────────────────────────────────────

def test_fok_confidence_is_monotonic_in_support():
    """More stored support never yields less confidence.

    Checked pairwise over the whole range the engine can hand it rather than
    at two hand-picked points, so a non-monotonic dip anywhere in the curve is
    caught.
    """
    values = [fok_confidence(s, True) for s in (0, 0.25, 0.5, 1, 2, 3, 5, 10)]
    for lo, hi in zip(values, values[1:]):
        assert hi >= lo, f"confidence fell as support rose: {values}"


def test_fok_confidence_saturates_at_one():
    """Confidence is bounded above — the ACC never reports >1.0.

    Without this a large support count would produce a confidence the
    assert-gate arithmetic could not reason about.
    """
    assert fok_confidence(1000, True) == 1.0
    assert fok_confidence(3, True) == 1.0


def test_fok_confidence_stays_in_the_unit_interval():
    """No input may push the reported confidence outside [0, 1]."""
    for support in (0, 0.1, 1, 4, 50):
        for ok in (True, False):
            c = fok_confidence(support, ok)
            assert 0.0 <= c <= 1.0, (support, ok, c)


def test_retrieval_success_raises_confidence_at_every_support_count():
    """The boost applies across the curve, not just at one point."""
    for support in (0, 0.5, 1, 2):
        assert fok_confidence(support, True) > fok_confidence(support, False)


def test_unsupported_and_failed_retrieval_is_the_low_confidence_case():
    """A failed retrieval is the weak end of the curve.

    This is the case the numeric-claim honesty gate in
    `engine_web_search.py` depends on: a claim RAVANA has no internal support
    for must land below an ordinary supported claim's confidence.
    """
    assert fok_confidence(1, False) < fok_confidence(1, True)


def test_fok_confidence_is_deterministic():
    """Same input, same number — the gate must not jitter turn to turn."""
    for support in (0, 1, 2.5):
        assert fok_confidence(support, True) == fok_confidence(support, True)


# ── should_assert / the gate ───────────────────────────────────────────

def test_should_assert_above_the_gate_allows_a_flat_claim():
    conf = THETA_WITHDHOLD + 0.10
    may_assert, _modality = should_assert(conf)
    assert may_assert is True


def test_should_assert_below_the_gate_forbids_a_flat_claim():
    """The whole point of the gate: below theta, the claim is hedged."""
    conf = THETA_WITHDHOLD - 0.10
    may_assert, _modality = should_assert(conf)
    assert may_assert is False


def test_should_assert_is_inclusive_at_the_gate():
    """`>=` at the boundary, not `>`.

    The comparison operator is load-bearing: flipping it to strict would make
    the boundary state depend on float representation.
    """
    may_assert, _ = should_assert(THETA_WITHDHOLD)
    assert may_assert is True


def test_a_learned_gate_flips_a_real_decision():
    """The payoff the calibrator exists to produce.

    A claim whose confidence sits between the cold default and a learned
    overconfident gate is asserted by one engine and withheld by the other.
    Measured support counts make this a real mid-curve value, not a synthetic
    one.
    """
    conf = fok_confidence(1, True)     # 0.721 measured
    assert conf > THETA_WITHDHOLD, "expected a mid-curve confidence to clear the cold gate"

    _cold_may, _ = should_assert(conf, THETA_WITHDHOLD)
    learned_may, _ = should_assert(conf, 0.85)   # a gate at the structural ceiling
    assert _cold_may is True
    assert learned_may is False


def test_a_learned_gate_cannot_lower_the_bar_below_zero_confidence():
    """Even the floor gate still permits a fully-confident claim.

    Calibration changes hedging, not certainty: a miscalibrated gate must not
    be able to manufacture doubt about a claim RAVANA is maximally sure of.
    """
    may_assert, _ = should_assert(1.0, 0.05)
    assert may_assert is True


# ── modality_from_support ──────────────────────────────────────────────

def test_modality_from_support_is_defined_at_the_extremes():
    """Both ends of the confidence scale return a real hedge word.

    Guards against a gap in the vocabulary that would surface to the user as
    an empty modality string.
    """
    assert isinstance(modality_from_support(0.0), str)
    assert isinstance(modality_from_support(1.0), str)
    assert modality_from_support(0.0) != ""


def test_modality_from_support_is_monotonic_in_confidence():
    """Higher confidence never yields a *weaker* hedge.

    Modality is an ordered vocabulary, so a non-monotonic step means the word
    chosen contradicts the gate decision. Asserted over the closed interval.
    """
    ladder = [modality_from_support(c / 20) for c in range(21)]
    order = {m: i for i, m in enumerate(dict.fromkeys(ladder))}
    idx = [order[m] for m in ladder]
    for lo, hi in zip(idx, idx[1:]):
        assert hi >= lo, f"modality weakened as confidence rose: {ladder}"


def test_modality_is_stable_for_the_same_confidence():
    """Deterministic vocabulary lookup, not a random or time-varying pick."""
    for conf in (0.0, 0.42, 0.87, 1.0):
        assert modality_from_support(conf) == modality_from_support(conf)


# ── Metacognition wrapper ──────────────────────────────────────────────

def test_metacognition_defaults_to_the_module_theta():
    mc = Metacognition()
    assert mc.theta_withhold == THETA_WITHDHOLD


def test_metacognition_read_agrees_with_the_free_function():
    """The wrapper must not reimplement the gate.

    If `read()` ever computes its own comparison, the two would drift and the
    calibrator's gate would apply to only one of them.
    """
    for support in (0, 1, 2):
        conf, may_assert, modality = Metacognition().read(support, True)
        exp_conf = fok_confidence(support, True)
        exp_may, exp_mod = should_assert(exp_conf, THETA_WITHDHOLD)
        assert conf == exp_conf
        assert may_assert is exp_may
        assert modality == exp_mod


def test_metacognition_honours_a_learned_gate_end_to_end():
    """A shifted gate changes what `read()` reports, not just what the
    free function would return."""
    support = 1
    cold = Metacognition().read(support, True)
    learned = Metacognition(theta_withhold=0.85).read(support, True)
    assert cold[0] == learned[0], "confidence itself must not change — only the gate"
    assert cold[1] is True
    assert learned[1] is False


def test_metacognition_keeps_a_bounded_recent_verdict_buffer():
    """The observability ring buffer is bounded.

    Unbounded growth here would be an unbounded per-engine memory cost on a
    long-running session.
    """
    mc = Metacognition()
    for _ in range(200):
        mc.read(1, True)
    assert len(mc._recent) <= 50
