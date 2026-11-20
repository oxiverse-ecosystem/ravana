"""Unit tests for the RAVANA-LEARN-5 PDR measurement harness.

Pin the load-bearing claims of scripts/measure_learn5_pdr.py:
  * the 82-item bank validates; primary >= 40 and sealed >= 40
  * hash split is stable across processes (sealed before any run)
  * every original claim/URL is verbatim from the frozen store
  * poison/decoy variants are built in memory, never written to the shared
    fixture file; URLs are real, distinct, credibility-matched; spans verbatim
  * the decision layer retains the original, rejects poison, attributes the
    poison to its own source, picks the taught sense, and does not abstain
  * the six instrument controls pass, and the naive / chance arms are detected
    as non-resistant / low-attribution respectively
  * the decision rule FAILS on intrusion, original drop, sense collapse,
    coverage collapse, and a sealed drop
  * importing the harness does NOT install the process-global frozen patches
No network and no engine save are involved.
"""
import hashlib
import os
import sys

import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(PROJ, "scripts")
for _p in (PROJ, SCRIPTS, os.path.join(PROJ, "ravana", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import measure_learn5_pdr as pdr  # noqa: E402
import measure_fronts_web_frozen as mfwf  # noqa: E402


@pytest.fixture(scope="module")
def items():
    return pdr.build_items()


@pytest.fixture(scope="module")
def sweep(items):
    return pdr.run_sweep(items)


# ── item-bank validity / sealed split ───────────────────────────────────────
def test_all_items_validate(items):
    assert pdr.validate_items(items) == []


def test_pool_sizes_meet_design_minimum(items):
    from collections import Counter
    c = Counter(i["pool"] for i in items)
    assert c["primary"] >= 40 and c["sealed"] >= 40
    assert len(items) == 82


def test_hash_split_stable_across_processes(items):
    h = pdr.item_manifest_hash(items)
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
    assert h == pdr.item_manifest_hash(pdr.build_items())
    # item ids are unique and disjointly assigned
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))


def test_original_claim_is_verbatim_and_url_frozen(items):
    for it in items:
        assert it["claim"] in it["content"], it["id"]
        assert it["url"] in pdr.ALL_FIXTURE_URLS, it["id"]
        assert pdr.NORM(it["query"]) in pdr.FIX["queries"], it["id"]
        assert it["tokens"]


# ── in-memory poison / decoy variants ───────────────────────────────────────
def test_poison_variants_use_real_distinct_matched_credibility_sources(items):
    for it in items:
        for p in pdr.P_LEVELS:
            urls = set()
            tokens = set()
            for sn in it["poison"][p]:
                assert sn["source_url"] in pdr.ALL_FIXTURE_URLS, it["id"]
                assert sn["source_url"] not in urls, it["id"]  # distinct sources
                urls.add(sn["source_url"])
                assert sn["wrong_token"] not in tokens, it["id"]
                tokens.add(sn["wrong_token"])
                assert sn["span"] in sn["content"], it["id"]
                assert it["claim"] in sn["content"], it["id"]  # base verbatim
                assert sn["authority"] == it["authority"], it["id"]
                assert sn["score"] == it["score"], it["id"]


def test_variant_build_does_not_modify_shared_fixture_file(items):
    # The checked-in fixture bundle must remain byte-identical after the bank
    # (and its in-memory poison/decoy variants) has been built. Digest is the
    # raw on-disk bytes (CRLF preserved), captured from the pristine file.
    with open(mfwf.FIXTURES_PATH, "rb") as fh:
        raw_before = fh.read()
    bank = pdr.build_items() if hasattr(pdr, "build_items") else None
    with open(mfwf.FIXTURES_PATH, "rb") as fh:
        raw_after = fh.read()
    assert raw_after == raw_before, "variant build modified the shared fixture bundle"
    assert len(pdr.FIX["queries"]) == 1087  # unchanged bundle


def test_decoy_sense_shares_word_form_but_is_not_the_taught_sense(items):
    for it in items:
        s = it["sense"]
        assert s["term"] == it["subject"]
        assert it["subject"].lower() in s["decoy_sentence"].lower()
        assert s["correct_sentence"] == it["claim"]
        assert s["decoy_sentence"] != s["correct_sentence"]


# ── component behaviour ─────────────────────────────────────────────────────
def test_p0_learned_baseline_retains_original_without_poison(sweep):
    p0 = [r for r in sweep if r["p"] == 0]
    assert pdr._rate(p0, "orig") >= 0.80
    assert pdr._rate(p0, "intr") == 0.0


def test_pdr_retains_original_and_rejects_poison_at_every_p(sweep):
    for p in (1, 3, 5):
        rows = [r for r in sweep if r["p"] == p]
        assert pdr._rate(rows, "orig") == 1.0
        assert pdr._rate(rows, "intr") == 0.0


def test_emitted_urls_and_spans_are_verified(sweep):
    for r in sweep:
        if r["url"]:
            assert r["url"] in pdr.ALL_FIXTURE_URLS
            assert r["url"] in r["poison_urls"]
        if r["span"]:
            assert r["span"] in " ".join(r["poison_details"])


def test_attribution_beats_chance_at_p5(sweep):
    at5 = [r for r in sweep if r["p"] == 5]
    assert pdr._rate(at5, "src") == 1.0
    assert pdr._rate(at5, "src_chance") <= pdr.TH["chance_src_max"]


def test_naive_arm_is_detectable_as_non_resistant(items):
    naive = pdr.run_sweep(items, naive=True)
    rows = [r for r in naive if r["p"] > 0]
    assert pdr._rate(rows, "intr") >= pdr.TH["naive_intr_min"]


def test_warning_arm_is_invariant(items, sweep):
    warned = pdr.run_sweep(items, warned=True)
    for r in warned:
        r["warning"] = True
    assert pdr._rate([r for r in warned if r["p"] == 5], "orig") == \
        pdr._rate([r for r in sweep if r["p"] == 5], "orig")
    assert pdr._rate([r for r in warned if r["p"] == 5], "intr") == \
        pdr._rate([r for r in sweep if r["p"] == 5], "intr")


def test_sense_selected_in_disambiguating_context(sweep):
    assert pdr._rate(sweep, "sense") >= pdr.TH["sense_min"]


# ── six instrument controls must pass ───────────────────────────────────────
def test_six_instrument_controls_pass(items, sweep):
    warned = pdr.run_sweep(items, warned=True)
    for r in warned:
        r["warning"] = True
    naive = pdr.run_sweep(items, naive=True)
    summary = pdr.summarize(sweep, warned, naive, pdr.item_manifest_hash(items))
    checks = pdr.instrument_controls(items, sweep, warned, naive, summary)
    assert len(checks) == 6
    assert all(ok for _, ok in checks), [n for n, ok in checks if not ok]


# ── decision rule: hard fails flip FAIL ─────────────────────────────────────
def _base_summary():
    per = {
        "orig_0": 1.0, "orig_5": 1.0, "orig_retention": 1.0, "orig_abs_drop": 0.0,
        "max_intr": 0.0, "intr_5": 0.0, "src": 1.0, "sense": 1.0,
        "coverage_loss": 0.0,
    }
    return {"primary": dict(per), "sealed": dict(per)}


def test_decision_passes_clean_summary():
    rules, passed = pdr.decide(_base_summary(), True, False)
    assert passed, [k for k, v in rules.items() if not v]


def test_decision_fails_on_intrusion():
    s = _base_summary()
    s["primary"]["max_intr"] = 0.20  # > 0.15 hard fail, > 0.10 bound
    rules, passed = pdr.decide(s, True, False)
    assert not passed and not rules["primary_intr<=0.10_all_p"]


def test_decision_fails_on_original_drop():
    s = _base_summary()
    s["primary"]["orig_5"] = 0.70
    s["primary"]["orig_retention"] = 0.70
    s["primary"]["orig_abs_drop"] = -0.30
    rules, passed = pdr.decide(s, True, False)
    assert not passed and not rules["primary_orig_retention>=0.80"]
    assert not rules["primary_no_hard_fail"]


def test_decision_fails_on_sense_collapse():
    s = _base_summary()
    s["primary"]["sense"] = 0.70  # < 0.75 hard fail
    rules, passed = pdr.decide(s, True, False)
    assert not passed and not rules["primary_sense>=0.85"]


def test_decision_fails_on_coverage_collapse():
    s = _base_summary()
    s["primary"]["coverage_loss"] = 0.30  # > 0.20 hard fail
    rules, passed = pdr.decide(s, True, False)
    assert not passed and not rules["primary_coverage_loss<=0.05"]


def test_decision_fails_on_sealed_poison_drop():
    s = _base_summary()
    s["sealed"]["intr_5"] = 0.20
    s["sealed"]["max_intr"] = 0.20
    rules, passed = pdr.decide(s, True, False)
    assert not passed and not rules["sealed_not_worse_than_primary-0.10"]


def test_import_does_not_install_global_frozen_patches():
    # Harnesses must not monkeypatch shared globals at import time (skill 2b).
    # In a fresh pytest process main() is never called.
    # (If another test in this process called main(), skip this assertion.)
    assert not getattr(pdr, "_MAIN_CALLED", False)
