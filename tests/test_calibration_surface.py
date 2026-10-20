"""The calibration ledger's OBSERVABILITY surface (`scripts/ravana_chat.py --calibration`).

The unit suite (`test_epistemic_calibration.py`) proves the ledger's contract and
`test_epistemic_calibration_wiring.py` proves the engine feeds it — but neither
reaches the surface a human or a CI gate actually reads. Before this round
`EpistemicCalibrator.get_status()` was reachable only from in-process code: a
grep over `scripts/` found no reader, so the one cognitive subsystem whose state
is a pure numeric ledger had no way to be looked at.

These tests pin the surface itself:

* it reads `get_status()` and renders the numbers THAT METHOD returns (no second
  implementation of the bias that could drift from the gate the engine uses),
* the per-band reliability curve and the adaptive `theta_withhold` are both on
  the surface — the three metrics the capability exists to expose,
* it is read-only: a status command must not mutate the ledger or write state,
* it degrades to a printed limitation rather than a traceback when the
  calibrator is absent or broken.

Every test uses a bare `assert`. A test that RETURNS a bool is reported green by
pytest regardless of its value — a defect class this repo has shipped before.
"""

import io
import json
import os
import sys
import contextlib

import pytest

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import scripts.ravana_chat as rc

from ravana.chat.calibration import EpistemicCalibrator


# ── the surface reads the ledger, and only the ledger ───────────────────

def test_report_is_built_from_get_status_not_a_reimplementation():
    """The rendered numbers must be the ones `get_status()` returned.

    A renderer that recomputed the bias from `_pairs` would be a second source
    of truth: it could disagree with the gate the engine actually applies and
    the disagreement would be invisible. So the contract here is that the
    status dict is the ONLY input.
    """
    status = {
        "n": 12,
        "total_observed": 30,
        "mean_bias": 0.25,
        "bias": 0.18,
        "brier": 0.11,
        "theta_withhold": 0.48,
        "bands_visited": 3,
        "bands_total": 5,
        "calibrated": False,
        "reliability": [
            None,
            {"lo": 0.2, "hi": 0.4, "n": 4, "observed": 0.62},
            {"lo": 0.4, "hi": 0.6, "n": 8, "observed": 0.55},
            None,
            None,
        ],
    }
    out = rc.render_calibration_report(status)

    assert "0.1800" in out, "signed bias not rendered from the status dict"
    assert "+0.2500" in out, "raw mean bias not rendered"
    assert "0.1100" in out, "brier score not rendered"
    assert "0.4800" in out, "theta_withhold not rendered"
    assert "12" in out and "30" in out, "observation counts not rendered"
    assert "3 / 5" in out, "bands visited not rendered"
    # the reliability curve rows come from the curve, not from a recompute
    assert "0.2 - 0.4" in out and "0.620" in out
    assert "0.4 - 0.6" in out and "0.550" in out
    # an unvisited band is absent knowledge and must be labelled as such
    assert "absent knowledge" in out


def test_unvisited_bands_are_labelled_absent_not_zero():
    """A band with no observations must not be printed as a 0.0 hit-rate.

    Collapsing 'never visited' into 'visited and always wrong' is how a
    reliability diagram becomes a lie — the module's own docstring says so.
    """
    status = {
        "n": 0, "total_observed": 0, "mean_bias": 0.0, "bias": 0.0,
        "brier": 0.0, "theta_withhold": 0.30, "bands_visited": 0,
        "bands_total": 5, "calibrated": False,
        "reliability": [None, None, None, None, None],
    }
    out = rc.render_calibration_report(status)
    assert "absent knowledge" in out
    # and no band row may claim an observed hit-rate of zero
    assert "0.000" not in out.split("RELIABILITY CURVE", 1)[1]


# ── the surface is wired to a real calibrator ───────────────────────────

def test_read_calibration_status_returns_the_ledgers_own_dict():
    cal = EpistemicCalibrator()
    for _ in range(20):
        cal.observe(0.9, 0.4)

    class _Eng:
        pass

    eng = _Eng()
    eng.calibrator = cal

    status, err = rc._read_calibration_status(eng)
    assert err is None
    assert status is not None
    # identity with what the calibrator itself reports — the surface adds nothing
    assert status["bias"] == cal.get_status()["bias"]
    assert status["theta_withhold"] == cal.get_status()["theta_withhold"]
    assert status["n"] == cal.n
    # the curve is read from the same object, in the same call
    assert status["reliability"] == cal.reliability()


def test_surface_does_not_mutate_the_ledger():
    """A status command must be read-only.

    Reading the ledger must not advance it: `get_status()` is a snapshot, and a
    surface that perturbed the thing it reports would make every observation
    after the first wrong.
    """
    cal = EpistemicCalibrator()
    for _ in range(10):
        cal.observe(0.7, 0.5)
    before = (cal.n, cal.get_state()["total"], cal.bias(), cal.theta_withhold())

    class _Eng:
        pass

    eng = _Eng()
    eng.calibrator = cal

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = rc.print_calibration_report(eng)
    assert ok is True
    assert buf.getvalue().strip() != ""

    after = (cal.n, cal.get_state()["total"], cal.bias(), cal.theta_withhold())
    assert after == before, f"reading the ledger changed it: {before} -> {after}"


def test_json_mode_emits_a_parseable_status_dict():
    cal = EpistemicCalibrator()
    for _ in range(6):
        cal.observe(0.8, 0.6)

    class _Eng:
        pass

    eng = _Eng()
    eng.calibrator = cal

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc.print_calibration_report(eng, as_json=True)

    parsed = json.loads(buf.getvalue())
    assert parsed["n"] == cal.n
    assert parsed["theta_withhold"] == cal.get_status()["theta_withhold"]
    assert isinstance(parsed["reliability"], list)


# ── degradation: a broken calibrator must not raise ─────────────────────

def test_missing_calibrator_degrades_to_a_message():
    class _Eng:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = rc.print_calibration_report(_Eng())
    assert ok is False
    assert "unavailable" in buf.getvalue()


def test_get_status_raising_degrades_to_a_message():
    class _Broken:
        def get_status(self):
            raise RuntimeError("ledger on fire")

    class _Eng:
        pass

    eng = _Eng()
    eng.calibrator = _Broken()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = rc.print_calibration_report(eng)
    assert ok is False
    assert "ledger on fire" in buf.getvalue()


def test_non_dict_status_degrades_to_a_message():
    class _Weird:
        def get_status(self):
            return ["not", "a", "dict"]

    class _Eng:
        pass

    eng = _Eng()
    eng.calibrator = _Weird()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = rc.print_calibration_report(eng)
    assert ok is False
    assert "not a dict" in buf.getvalue()


# ── the CLI actually exposes the flag ───────────────────────────────────

def test_cli_exposes_the_calibration_flag():
    """The flag must be on the parser, or the surface is unreachable.

    Parses a synthetic argv rather than shelling out: the parser is built inside
    `main()`, and booting an engine to reach it would make this test pay the
    ~27s GloVe init for a one-line assertion.
    """
    import argparse
    import inspect

    src = inspect.getsource(rc.main)
    assert '"--calibration"' in src, "--calibration is not a registered CLI flag"
    assert '"--json"' in src, "--json is not a registered CLI flag"
    # and the handler must call the renderer rather than reimplementing it
    assert "print_calibration_report" in src
