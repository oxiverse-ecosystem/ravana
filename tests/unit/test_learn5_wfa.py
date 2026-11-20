"""Unit tests for the RAVANA-LEARN-5 WFA measurement harness.

These pin the load-bearing claims of scripts/measure_learn5_wfa.py:
  * every emitted URL is a frozen, fetched URL; every span is verbatim
  * decoys / closed-web items abstain; internal claims carry no web tag
  * the entailment rubric refuses non-entailing spans
  * the decision rule FAILS on fabrication, low validity, or a sealed drop
  * the hash split is stable across processes
No network and no engine save are involved; the frozen replay boundary is
reused from measure_fronts_web_frozen.
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

import measure_learn5_wfa as wfa  # noqa: E402


@pytest.fixture(scope="module")
def items():
    return wfa.build_items()


# ── item-bank validity ──────────────────────────────────────────────────────
def test_all_items_validate(items):
    assert wfa.validate_items(items) == []


def test_gap_claim_is_verbatim_and_url_frozen(items):
    gaps = [i for i in items if i["kind"] == "gap"]
    assert len(gaps) == 64
    for it in gaps:
        assert it["claim"] in it["content"], it["id"]
        assert it["url"] in wfa.ALL_FIXTURE_URLS, it["id"]
        assert it["claim_tokens"]


def test_decoys_are_non_entailing(items):
    probe = wfa.WFAComponent(None)
    for it in items:
        if it["kind"] != "decoy":
            continue
        for h in wfa.FIX["queries"][wfa.NORM(it["query"])]["hits"][:2]:
            assert probe.find_entailing_span(h["content"], it["claim_tokens"]) is None, it["id"]


def test_closed_web_queries_absent_from_fixtures(items):
    for it in items:
        if it["kind"] == "closed_web":
            assert wfa.NORM(it["query"]) not in wfa.FIX["queries"], it["id"]


def test_hash_split_is_stable(items):
    h = wfa.item_manifest_hash(items)
    assert re_is_hex(h)
    assert h == wfa.item_manifest_hash(wfa.build_items())


def re_is_hex(s):
    return len(s) == 64 and all(c in "0123456789abcdef" for c in s)


# ── rubric / component behaviour ────────────────────────────────────────────
def test_entailment_rubric_refuses_non_entailing_span():
    comp = wfa.WFAComponent(None)
    content = ("A compass points north because the Earth behaves like a giant "
               "bar magnet.")
    assert comp.find_entailing_span(content, wfa._words("a giant bar magnet")) is not None
    assert comp.find_entailing_span(
        content, wfa._words("a tachyonic aether current pulls it")) is None


def test_fabricated_url_rejected_by_allowlist():
    assert "https://fabricated.example.invalid/fact" not in wfa.ALL_FIXTURE_URLS


def test_decoy_run_abstains_without_url(items):
    comp = wfa.WFAComponent(wfa.FROZEN_SEARCH)
    decoys = [i for i in items if i["kind"] == "decoy"][:3]
    for it in decoys:
        rec = comp.run_item(it)
        assert rec["verdict"] == "abstain"
        assert rec["url"] is None and rec["span"] is None


def test_closed_web_run_abstains_without_url(items):
    comp = wfa.WFAComponent(wfa.FROZEN_SEARCH)
    closed = [i for i in items if i["kind"] == "closed_web"][:3]
    for it in closed:
        rec = comp.run_item(it)
        assert rec["verdict"] == "abstain"
        assert rec["url"] is None


def test_internal_run_carries_no_web_tag(items):
    comp = wfa.WFAComponent(wfa.FROZEN_SEARCH)
    internal = [i for i in items if i["kind"] == "internal"][:3]
    for it in internal:
        rec = comp.run_item(it)
        assert rec["url"] is None and rec["external_tag"] is False


def test_gap_run_attr_url_is_frozen_and_span_verbatim(items):
    comp = wfa.WFAComponent(wfa.FROZEN_SEARCH)
    gap = [i for i in items if i["kind"] == "gap"][0]
    rec = comp.run_item(gap)
    assert rec["verdict"] == "attributed"
    assert rec["url"] in wfa.ALL_FIXTURE_URLS
    assert rec["url"] in _urls_for(gap["query"])
    assert rec["span"] in gap["content"]
    assert rec["external_tag"] is True


def _urls_for(query):
    return {h.get("url") for h in wfa.FIX["queries"][wfa.NORM(query)]["hits"]}


# ── decision rule: fabrication / weakness flips FAIL ───────────────────────
def _base_summary():
    per = {
        "wfa_valid": 1.0, "answer_only": 0.0, "delta": 1.0,
        "abstain": 1.0, "decoy_reject": 1.0, "internal_no_tag": 1.0,
        "reads_total": 1, "reads_mean_gap": 1.0,
    }
    return {
        "primary": dict(per), "sealed": dict(per),
        "hard_fails": {"fabricated_urls": 0, "unfetched_urls": 0,
                       "nonverbatim_spans": 0, "attribution_on_closed_web": 0},
    }


def test_decision_passes_clean_summary():
    rules, passed = wfa.decide(_base_summary(), True)
    assert passed, [k for k, v in rules.items() if not v]


def test_decision_fails_on_fabricated_url():
    s = _base_summary()
    s["hard_fails"]["fabricated_urls"] = 1
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["fabricated_urls==0"]


def test_decision_fails_on_low_wfa_valid():
    s = _base_summary()
    s["primary"]["wfa_valid"] = 0.60
    s["primary"]["delta"] = 0.60
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["wfa_valid_primary>=0.70"]


def test_decision_fails_on_sealed_drop():
    s = _base_summary()
    s["sealed"]["wfa_valid"] = 0.80
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["sealed_not_below_primary-0.10"]


def test_decision_fails_on_closed_web_attribution():
    s = _base_summary()
    s["hard_fails"]["attribution_on_closed_web"] = 1
    rules, passed = wfa.decide(s, True)
    assert not passed and not rules["attribution_on_closed_web==0"]
