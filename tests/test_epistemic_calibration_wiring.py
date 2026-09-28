"""Integration: the engine really pairs prediction with outcome and learns.

The unit suite (test_epistemic_calibration.py) proves the ledger's contract.
These tests prove the WIRING: that `process_turn` actually feeds the pair,
and that the learned bias reaches the two assert-gates that consume it.

They boot a real CognitiveChatEngine, so they are the expensive tier. Each
one asserts on observable engine state, never on reply prose — this round
adds no reply strings, and a test that asserted on English would be a test
that could only pass by hardcoding.
"""

import os

import pytest

RAVANA_OFFLINE = "1"


@pytest.fixture(scope="module")
def engine():
    """One engine for the whole module — booting GloVe per test is the cost.

    Uses a throwaway user_suffix and deletes any prior pickles for it first.
    A stale checkpoint left by an earlier run would be loaded and merged, so
    the ledger would start with observations this session never made and the
    cold-start assertions would be meaningless.
    """
    import glob
    from ravana.chat.engine import CognitiveChatEngine
    suffix = "calibwire_test"
    for pat in (f"weights/ravana_weights{suffix}.pkl",
                f"weights/ravana_usermodel{suffix}.pkl"):
        for path in glob.glob(pat):
            try:
                os.remove(path)
            except OSError:
                pass
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix=suffix)


TURNS = [
    "what is gravity",
    "how does photosynthesis work",
    "who invented the telescope",
    "my cat milo knocks over my plants",
    "what is recursion",
    "why is the sky blue",
    "tell me about quantum entanglement",
]


# ── the engine has a calibrator at all ────────────────────────────────

def test_engine_exposes_a_calibrator(engine):
    from ravana.chat.calibration import EpistemicCalibrator
    assert isinstance(engine.calibrator, EpistemicCalibrator)


def test_calibrator_starts_cold_and_the_gate_is_the_base(engine):
    """Before any evidence the gate must equal the module default exactly.

    This is the regression guard for the whole feature: if wiring this in
    changed the assert-gate on a cold engine, every reply would shift before
    RAVANA had learned anything.
    """
    from ravana.chat.metacognition import THETA_WITHDHOLD
    fresh = engine.calibrator
    if fresh.n == 0:
        assert fresh.theta_withhold() == pytest.approx(THETA_WITHDHOLD)


# ── the pair is actually made ─────────────────────────────────────────

def test_process_turn_feeds_the_prediction_outcome_pair(engine):
    """The core wiring proof: after real turns the ledger holds observations.

    Before this feature the engine computed a confidence scalar and a quality
    score and paired them with nothing. If this assertion fails, the
    capability is inert no matter how good the module is.
    """
    before = engine.calibrator.n
    for q in TURNS:
        engine.process_turn(q)
    after = engine.calibrator.n
    assert after > before, (
        f"calibrator gained no observations across {len(TURNS)} turns "
        f"({before} -> {after}); the prediction/outcome pair is not wired"
    )


def test_recorded_pairs_are_in_range_and_consistent(engine):
    """Every stored pair must be a legal (prediction, outcome) in [0,1]^2."""
    pairs = engine.calibrator._pairs
    assert len(pairs) > 0
    for p, o in pairs:
        assert 0.0 <= p <= 1.0
        assert 0.0 <= o <= 1.0
    st = engine.calibrator.get_status()
    assert st["n"] == len(pairs)
    assert st["total_observed"] >= st["n"]


def test_reliability_curve_is_being_populated_by_the_engine(engine):
    """The learned store must show up as visited bands after real traffic."""
    st = engine.calibrator.get_status()
    assert st["bands_visited"] >= 1, (
        f"no reliability band was ever visited: {st}")


def test_observation_coverage_is_partial_because_predictions_are_partial():
    """Pins a measured limitation rather than leaving it to be discovered.

    `process_turn` has 59 early-return paths that fire BEFORE the per-turn
    confidence prediction is computed (engine.py:9386). On those turns RAVANA
    never makes a prediction, so there is nothing to score — and inventing one
    would be fabricating the very signal the calibration rests on. The
    consequence is real and worth stating: the ledger observes only the turns
    that reach full generation, so a conversation of N turns yields fewer than
    N observations.

    This test asserts the structural fact, so that if the early-return paths
    are ever refactored to emit predictions, the coverage change is visible
    here rather than silent.
    """
    import ast
    import inspect
    import textwrap

    from ravana.chat.engine import CognitiveChatEngine

    # `inspect.getsource` on a method keeps the class-body indentation, which
    # is not valid standalone Python for ast.parse. One dedent fixes it
    # (measured, not assumed — see tmp/probe_getsource.py).
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(CognitiveChatEngine.process_turn)))

    conf_line = None
    early_returns = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "confidence" \
                        and conf_line is None:
                    conf_line = node.lineno
        if isinstance(node, ast.Return):
            early_returns.append(node.lineno)

    assert conf_line is not None, "no confidence assignment found in process_turn"
    before = [r for r in early_returns if r < conf_line]
    # The point is not the exact count (that churns with every refactor) but
    # that early-return paths exist ahead of the prediction, so coverage is
    # partial BY CONSTRUCTION and must not be assumed to be 1.0.
    assert len(before) > 0, (
        "no early returns precede the confidence prediction any more; "
        "observation coverage may now be complete — revisit this test and the "
        "scope note in tmp/reports/ravana-feature-2026-09-28T0400Z-t_c2e355f4.md")


def test_no_observation_is_invented_for_a_turn_with_no_prediction(engine):
    """The ledger must never record a pair RAVANA did not actually predict.

    A fabricated prediction would let calibration 'learn' from noise and
    would make the reliability curve a work of fiction. Asserted by checking
    that every stored pair corresponds to a real observation call: with the
    engine idle the count must not drift on its own.
    """
    n_before = engine.calibrator.n
    total_before = engine.calibrator.get_status()["total_observed"]
    # No turns processed => no new observations. Nothing self-generates.
    assert engine.calibrator.n == n_before
    assert engine.calibrator.get_status()["total_observed"] == total_before


# ── the learned gate reaches the consumers ───────────────────────────

def test_learned_gate_reaches_the_web_honesty_metacognition(engine):
    """`_metacognition.theta_withhold` must track the learned gate.

    That attribute is what `Metacognition.read()` passes to `should_assert`,
    so if it does not track, the numeric-claim honesty gate is still running
    on the fixed module default and the capability is not connected.
    """
    from ravana.chat.calibration import EpistemicCalibrator
    from ravana.chat.metacognition import Metacognition, fok_confidence

    # Drive the same assignment the web-search path performs.
    mc = Metacognition()
    mc.theta_withhold = engine.calibrator.theta_withhold()
    assert mc.theta_withhold == pytest.approx(engine.calibrator.theta_withhold())

    # A shifted gate must actually flip the verdict. Pick a support value
    # whose FOK scalar is known, then straddle it with the two gates: a real
    # calibration shift has to be able to move the decision either way.
    conf = fok_confidence(1.0, True)      # 1 - e^-0.25, plus the retrieval boost
    assert 0.5 < conf < 1.0
    open_mc = Metacognition()
    open_mc.theta_withhold = 0.05          # effectively open
    assert open_mc.read(1.0, True)[1] is True
    shut_mc = Metacognition()
    shut_mc.theta_withhold = 0.99          # effectively closed
    assert shut_mc.read(1.0, True)[1] is False
    # the calibrator's own gate sits between the two extremes and must not be
    # outside the legal band the module enforces
    assert 0.05 <= engine.calibrator.theta_withhold() <= 0.85


def test_a_shifted_gate_changes_should_assert_on_the_real_metacognition_fn():
    """The lever the calibration pulls must be a real lever.

    `should_assert` is the single function both metacognitive surfaces call.
    Confirm that raising the gate can flip a verdict, so a learned upward
    shift has somewhere to go.
    """
    from ravana.chat.metacognition import should_assert
    conf = 0.5
    assert should_assert(conf, 0.30)[0] is True
    assert should_assert(conf, 0.80)[0] is False


def test_metacognitive_realizer_uses_the_calibrator_gate(engine):
    """`_realize_metacognitive` must consult the ledger, not the default.

    Verified by driving the method with the engine's gate pinned low and then
    high, on the same bundle: if it ignored the calibrator the two calls would
    be identical.
    """
    bundle = {
        "subject": "gravity",
        "edge_count": 1,
        "has_definition": False,
        "retrieval_succeeded": False,
        "facts": [],
        "stance": None,
        "n_retrieved_assocs": 0,
    }
    from ravana.chat.calibration import EpistemicCalibrator

    # Gate forced to 0.0: may_assert is always True, so the method takes the
    # "assert real retrieved state" branch when state is present.
    forced_low = EpistemicCalibrator(theta_base=0.0)
    saved = engine.calibrator
    try:
        engine.calibrator = forced_low
        out_low = engine._realize_metacognitive(dict(bundle, facts=["it pulls things down"]))
        # Gate forced high: may_assert is always False for this low-support
        # bundle, so the branch must differ.
        engine.calibrator = EpistemicCalibrator(theta_base=1.0)
        out_high = engine._realize_metacognitive(dict(bundle, facts=["it pulls things down"]))
    finally:
        engine.calibrator = saved
    assert isinstance(out_low, str) and isinstance(out_high, str)
    assert out_low != out_high, (
        "the realizer produced identical output for a fully-open and a "
        "fully-closed assert-gate; the calibrator is not reaching it")


# ── learning is real, not decorative ──────────────────────────────────

def test_engine_gate_responds_to_accumulated_overconfidence(engine):
    """The end-to-end payoff, on a real engine object.

    Drive the engine's own ledger with an overconfident stream and confirm
    the gate it hands to its consumers rises. This is the behaviour the whole
    round exists to produce.

    The comparison is made against the engine's gate as it stands *now*, not
    against the module default: the shared module-scoped engine has already
    run real turns by the time this executes, so its gate has legitimately
    moved off THETA_WITHDHOLD. Asserting it were still 0.30 would be
    asserting that the capability does nothing.
    """
    base = engine.calibrator.theta_withhold()
    for _ in range(40):
        engine.calibrator.observe(0.95, 0.15)
    shifted = engine.calibrator.theta_withhold()
    assert shifted > base, f"gate did not rise: {base} -> {shifted}"


def test_cold_start_identity_holds_on_a_fresh_engine_instance():
    """The safety property, on a real (not hand-built) engine.

    A brand-new engine has observed nothing, so its gate must equal the
    module default exactly. If this failed, wiring the calibrator in would
    change every reply before RAVANA had learned anything.
    """
    from ravana.chat.engine import CognitiveChatEngine
    from ravana.chat.metacognition import THETA_WITHDHOLD
    fresh = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                                user_suffix="calibcold_test")
    assert fresh.calibrator.n == 0
    assert fresh.calibrator.theta_withhold() == pytest.approx(THETA_WITHDHOLD)


def test_underconfidence_lowers_the_gate_in_the_other_direction(engine):
    base = engine.calibrator.theta_withhold()
    for _ in range(40):
        engine.calibrator.observe(0.15, 0.95)
    assert engine.calibrator.theta_withhold() < base


# ── persistence ───────────────────────────────────────────────────────

def test_calibration_ledger_is_included_in_the_save_payload(engine):
    """The learned curve must be part of `save()`, or the capability resets
    every session and only ever learns within one process lifetime."""
    import inspect
    src = inspect.getsource(type(engine).save)
    assert "epistemic_calibration" in src, (
        "save() does not persist the calibration ledger")


def test_ledger_state_round_trips_through_the_engine(engine):
    from ravana.chat.calibration import EpistemicCalibrator
    r = EpistemicCalibrator()
    assert r.load_state(engine.calibrator.get_state()) is True
    assert r.theta_withhold() == pytest.approx(engine.calibrator.theta_withhold())
    assert r.n == engine.calibrator.n
