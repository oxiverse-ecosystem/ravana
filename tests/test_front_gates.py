"""Fast, engine-free tests for the Tier-0 instrument gates."""
import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, "..", "scripts"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fg = _load("front_gates", os.path.join(SCRIPTS, "front_gates.py"))
mf = _load("measure_fronts", os.path.join(SCRIPTS, "measure_fronts.py"))

JUDGES = mf.JUDGES
RECORDS = [
    {"front": front, "probe_index": i, "subject": f"s{i}",
     "passed": i % 2 == 0, "control": False}
    for front in fg.QUERY_FRONTS for i in range(1, 11)
]


def test_paired_rubric_passes_with_real_judges():
    r = fg.gate_paired(JUDGES)
    assert r.status == "PASS"
    assert r.metrics["bad_lt_good"] == len(fg.PAIRED_CASES)


def test_paired_rubric_catches_inverted_judge():
    r = fg.gate_paired({k: (lambda *a, **kw: (True, "always pass"))
                        for k in JUDGES})
    assert r.status == "FAIL" and r.blocking


def test_symmetric_guard_passes_and_detects_decline_everything():
    assert fg.gate_symmetric(JUDGES).status == "PASS"
    bad = dict(JUDGES)
    bad["thinking / reasoning"] = lambda *a, **kw: (False, "always decline")
    r = fg.gate_symmetric(bad)
    assert r.status == "FAIL"
    assert r.metrics["thinking_good_rate"] == 0.0


def test_selectivity_rejects_a_rubric_that_accepts_controls():
    accept_all = {k: (lambda *a, **kw: (True, "noise")) for k in JUDGES}
    assert fg.gate_selectivity(accept_all, RECORDS, mf.N_MAP).status == "FAIL"
    assert fg.gate_selectivity(JUDGES, RECORDS, mf.N_MAP).status == "PASS"


def test_recompute_scores_uses_earned_points_only():
    recs = [{"front": "language learning (online)", "passed": True,
             "sub_scores": {"taught": 40, "recall": 40}} for _ in range(4)]
    assert fg.recompute_scores(recs, mf.N_MAP)["language learning (online)"] == 80


def test_parity_flags_phantom_credit():
    summary = {"scores": {"language learning (online)": 100}}
    r = fg.gate_parity(mf.__file__, records=[
        {"front": "language learning (online)", "passed": True,
         "sub_scores": {"taught": 40, "recall": 40}}],
        summary=summary, compare_modes=False, n_map=mf.N_MAP)
    assert r.status == "FAIL"
    assert r.metrics["earned_vs_reported"]["deltas"]["language learning (online)"] == 20


def test_source_scan_ignores_documented_bonus_and_honest_subscore():
    region = fg._score_region(mf.__file__)
    assert region is not None
    assert "RAVANA_OFFLINE" not in region
    # 'bonus' appears only in a comment; '+ 20 * honest' is a real sub-score.
    assert "bonus" not in region


def test_saturation_and_blocking_status():
    r = fg.gate_saturation({"code generation": 100, "other": 42})
    assert r.status == "FLAG" and not r.blocking
    assert "code generation" in r.flagged_fronts
    assert fg.gate_saturation({"other": 42}).status == "PASS"
    fail = fg.GateResult("X", "x", "FAIL", "boom")
    assert not fg.gate_valid([r, fail])


def test_census_rejects_missing_probes():
    short = [r for r in RECORDS if r["probe_index"] > 3]
    r = fg.gate_census(short, {"only_front": 10}, mf.__file__,
                       cold_info={"passed": True})
    assert r.status == "FAIL"
