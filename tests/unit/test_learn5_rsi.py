"""Unit tests for the RAVANA-LEARN-5 RSI measurement harness.

Pin the load-bearing claims of scripts/measure_learn5_rsi.py:
  * the 100-family bank validates; primary >= 40 and sealed >= 40
  * the sha256 split is stable across processes and pools are disjoint
  * inverse/permuted surface phrases canonicalise to the same relation
  * the structural arm: exact match checks BOTH relation field and target;
    it is permutation-invariant, rejects the foil, and transfers far
  * the two confound arms behave as a control requires: the naive surface/
    order matcher fails permutation and accepts the foil, and the shuffled-
    label chance arm stays near 1/|relations|
  * all six instrument controls pass, and each hard fail flips decide()
  * no engine boot, no eng.save, no network, no global patch at import
"""
import os
import sys

import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(PROJ, "scripts")
for _p in (PROJ, SCRIPTS, os.path.join(PROJ, "ravana", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import measure_learn5_rsi as rsi  # noqa: E402


@pytest.fixture(scope="module")
def items():
    return rsi.build_items()


@pytest.fixture(scope="module")
def rows(items):
    return rsi.run_sweep(items)


@pytest.fixture(scope="module")
def summary(items, rows):
    return rsi.summarize(items, rows)


# ── bank validity / sealed split ────────────────────────────────────────────
def test_all_families_validate(items):
    assert rsi.validate_items(items) == []


def test_pool_sizes_meet_design_minimum(items):
    from collections import Counter
    c = Counter(i["pool"] for i in items)
    assert c["primary"] >= 40 and c["sealed"] >= 40
    assert len(items) == 100


def test_ids_unique_and_pools_disjoint(items):
    ids = [i["item_id"] for i in items]
    assert len(ids) == len(set(ids))
    p = {i["item_id"] for i in items if i["pool"] == "primary"}
    s = {i["item_id"] for i in items if i["pool"] == "sealed"}
    assert p.isdisjoint(s)


def test_hash_split_stable_across_processes(items):
    h = rsi.item_manifest_hash(items)
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
    assert h == rsi.item_manifest_hash(rsi.build_items())


def test_domain_vocabularies_disjoint():
    names = list(rsi.DOMAINS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert set(rsi.OBJECTS[a]).isdisjoint(rsi.OBJECTS[b])


def test_far_domain_differs_and_object_sets_disjoint(items):
    for it in items:
        assert it["domain"] != it["far_domain"]
        demo = {o for pair in it["demo_pairs"] for o in pair}
        assert it["exact"]["subject"] not in demo
        assert it["exact"]["gold_target"] not in demo
        # no exact candidate is a demonstrated object (naive-control hygiene)
        assert all(c not in demo for c in it["exact"]["candidates"])


# ── parsing / canonicalisation ──────────────────────────────────────────────
def test_inverse_phrase_canonicalises_to_forward_relation():
    assert rsi.canonicalise("a", "causes", "b") == ("a", "CAUSE", "b")
    assert rsi.canonicalise("b", "is caused by", "a") == ("a", "CAUSE", "b")
    assert rsi.canonicalise("x", "contains", "y") == ("y", "PART_OF", "x")
    assert rsi.canonicalise("y", "is part of", "x") == ("y", "PART_OF", "x")


def test_unknown_phrase_returns_none():
    assert rsi.canonicalise("a", "is unrelated to", "b") is None


# ── structural arm ──────────────────────────────────────────────────────────
def test_exact_checks_relation_field_and_target(items, rows):
    # an answer with the right target but the wrong relation is NOT a match
    rec = {"condition": "exact", "verdict": "match",
           "output": {"relation": "PART_OF", "target": "fires", "verdict": "match"},
           "gold": {"relation": "CAUSE", "target": "fires"}}
    assert rsi.grade(rec)["flags"]["correct"] is False
    rec2 = dict(rec, output={"relation": "CAUSE", "target": "sparks", "verdict": "match"})
    assert rsi.grade(rec2)["flags"]["correct"] is False
    rec3 = dict(rec, output={"relation": "CAUSE", "target": "fires", "verdict": "match"})
    assert rsi.grade(rec3)["flags"]["correct"] is True


def test_structural_exact_permuted_foil_far(summary):
    for pool in ("primary", "sealed"):
        p = summary[pool]
        assert p["exact_exact"] >= 0.90
        assert p["permuted_exact"] >= p["exact_exact"] - 0.10
        assert p["foil_foil_reject"] >= 0.90
        assert p["far_exact"] >= 0.90
        assert p["relation_coverage"] >= 0.95
        assert p["exact_minus_permuted"] >= -0.10
        for k in map(str, rsi.K_LEVELS):
            assert p["by_k"][k]["exact"] >= 0.90
            assert p["by_k"][k]["foil"] >= 0.90


def test_relation_field_present_on_every_structural_answer(rows):
    srows = [r for r in rows if r["arm"] == "structural"]
    assert all(r["flags"]["relation_present"] for r in srows)


# ── confound arms (the controls that give the test power) ───────────────────
def test_naive_fails_permutation(rows):
    for pool in ("primary", "sealed"):
        ex = [r for r in rows if r["arm"] == "naive" and r["pool"] == pool
              and r["condition"] == "exact"]
        pm = [r for r in rows if r["arm"] == "naive" and r["pool"] == pool
              and r["condition"] == "permuted"]
        assert rsi._rate(ex, "exact_structural_match") - \
            rsi._rate(pm, "exact_structural_match") >= rsi.TH["permuted_hard_margin"]


def test_naive_accepts_foil(rows):
    for pool in ("primary", "sealed"):
        foil = [r for r in rows if r["arm"] == "naive" and r["pool"] == pool
                and r["condition"] == "foil"]
        assert rsi._rate(foil, "correct") <= 0.10


def test_chance_arm_at_chance_and_far_beats_it(summary):
    for pool in ("primary", "sealed"):
        p = summary[pool]
        assert p["chance"] <= 0.35
        assert p["far_minus_chance"] >= rsi.TH["far_over_chance"]


# ── six instrument controls ─────────────────────────────────────────────────
def test_six_instrument_controls_pass(items, rows, summary):
    checks = rsi.instrument_controls(items, rows, summary)
    assert len(checks) == 6
    assert all(ok for _, ok in checks), [n for n, ok in checks if not ok]


# ── decision rule: hard fails flip FAIL ─────────────────────────────────────
def _base_summary():
    per = {"exact_exact": 1.0, "permuted_exact": 1.0,
           "foil_foil_reject": 1.0, "far_exact": 1.0, "chance": 0.20,
           "far_minus_chance": 0.80, "relation_coverage": 1.0}
    return {"primary": dict(per), "sealed": dict(per)}


def test_decision_passes_clean_summary():
    rules, passed = rsi.decide(_base_summary(), True, "FLAG")
    assert passed, [k for k, v in rules.items() if not v]


def test_decision_fails_on_permuted_drop():
    s = _base_summary()
    s["primary"]["permuted_exact"] = 0.80   # exact - 0.20 -> hard fail
    rules, passed = rsi.decide(s, True, "FLAG")
    assert not passed and not rules["primary_permuted_not_hard_fail"]


def test_decision_fails_on_foil_below_hard_floor():
    s = _base_summary()
    s["primary"]["foil_foil_reject"] = 0.65  # < 0.70 hard floor
    rules, passed = rsi.decide(s, True, "FLAG")
    assert not passed and not rules["primary_foil_not_hard_fail"]


def test_decision_fails_on_far_at_chance_plus_005():
    s = _base_summary()
    s["primary"]["far_minus_chance"] = 0.05  # <= 0.10 hard fail
    rules, passed = rsi.decide(s, True, "FLAG")
    assert not passed and not rules["primary_far_not_hard_fail"]


def test_decision_fails_on_relation_coverage():
    s = _base_summary()
    s["primary"]["relation_coverage"] = 0.90  # < 0.95
    rules, passed = rsi.decide(s, True, "FLAG")
    assert not passed and not rules["primary_relation_coverage>=0.95"]


def test_decision_fails_when_controls_fail():
    s = _base_summary()
    rules, passed = rsi.decide(s, False, "FLAG")
    assert not passed and not rules["instrument_controls_ok"]


def test_decision_fails_on_sealed_exact_drop():
    s = _base_summary()
    s["sealed"]["exact_exact"] = 0.80        # primary 1.00 - 0.20
    rules, passed = rsi.decide(s, True, "FLAG")
    assert not passed and not rules["sealed_exact>=primary_exact-0.10"]


# ── pre-registered thresholds fixed from design 7.6 ────────────────────────
def test_thresholds_match_design():
    assert rsi.TH["exact_min"] == 0.75
    assert rsi.TH["permuted_margin"] == 0.10
    assert rsi.TH["permuted_hard_margin"] == 0.15
    assert rsi.TH["foil_reject_min"] == 0.85
    assert rsi.TH["foil_reject_hard_min"] == 0.70
    assert rsi.TH["far_over_chance"] == 0.20
    assert rsi.TH["far_over_chance_hard"] == 0.10
    assert rsi.TH["relation_coverage_min"] == 0.95


# ── no engine / no save / no network / no import-time patch ────────────────
def test_no_engine_boot_or_save_in_source():
    path = os.path.join(SCRIPTS, "measure_learn5_rsi.py")
    src = open(path, encoding="utf-8").read()
    # strip triple-quoted docstrings/comments before the static check
    import re
    code = re.sub(r'""" .*? """', "", src, flags=re.S | re.X)
    code = re.sub(r"'''.*?'''", "", code, flags=re.S)
    code = "\n".join(line.split("#", 1)[0] for line in code.splitlines())
    assert "CognitiveChatEngine" not in code
    assert "eng.save(" not in code
    assert "urlopen" not in code


def test_module_import_installs_no_global_patches():
    # The harness records counters but does not monkeypatch process globals at
    # import; importing it in a shared pytest process is side-effect free.
    assert rsi.NETWORK_ATTEMPTS == []
    assert rsi.SAVE_ATTEMPTS == []
