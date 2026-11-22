"""Unit tests for the UNSATURATED RSI redesign (scripts/measure_learn5_rsi.py).

Pin the load-bearing claims WITHOUT booting an engine:

  * 120-family bank validates; primary/sealed >= 40; hash split stable;
    k levels are exactly {1,2,4,8} and conditions include negative rules
  * multi-word relation phrases canonicalise (regression for the re-split bug)
  * structural arm: near/far transfer, foil rejection, negative-rule rejection;
    counterexample-ignoring confound arm wrongly induces and is detectable
  * graded difficulty near >= far >= foil at each k; monotonic curve across k;
    growth headroom is REPORTED not gated (so 1.00 is flagged, not hidden)
  * naive confound fails permutation and accepts the foil; chance stays low
  * eight instrument controls pass; decide() flips FAIL on each hard criterion
  * fake path never boots CognitiveChatEngine; no network/save.
"""
import os
import re
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
    return rsi.run_sweep(items, backend=rsi.FakeBackend())


@pytest.fixture(scope="module")
def summary(items, rows):
    return rsi.summarize(items, rows)


# ── bank validity / sealed split ───────────────────────────────────────────
def test_all_families_validate(items):
    assert rsi.validate_items(items) == []


def test_pool_sizes_and_k_levels(items):
    from collections import Counter
    c = Counter(i["pool"] for i in items)
    assert c["primary"] >= 40 and c["sealed"] >= 40
    assert len(items) == 120
    assert rsi.K_LEVELS == (1, 2, 4, 8)
    assert "negative" in rsi.CONDITIONS
    assert "counterexample" in rsi.ARMS


def test_hash_split_stable_across_processes(items):
    h = rsi.item_manifest_hash(items)
    assert len(h) == 64 and all(ch in "0123456789abcdef" for ch in h)
    assert h == rsi.item_manifest_hash(rsi.build_items())


def test_ids_unique_and_pools_disjoint(items):
    ids = [i["item_id"] for i in items]
    assert len(ids) == len(set(ids))
    p = {i["item_id"] for i in items if i["pool"] == "primary"}
    s = {i["item_id"] for i in items if i["pool"] == "sealed"}
    assert p.isdisjoint(s)


def test_domain_vocabularies_disjoint():
    names = list(rsi.DOMAINS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert set(rsi.OBJECTS[a]).isdisjoint(rsi.OBJECTS[b])


def test_held_out_queries_do_not_reuse_demo_objects(items):
    for it in items:
        assert it["domain"] != it["far_domain"]
        demo = {o for pair in it["demo_pairs"] for o in pair}
        assert len(it["demo_pairs"]) == max(rsi.K_LEVELS)
        assert it["near"]["subject"] not in demo
        assert it["near"]["gold_target"] not in demo
        assert it["negative"]["subject"] not in demo
        assert it["negative"]["counterexample_target"] in it["negative"]["candidates"]


# ── parsing / canonicalisation (regression for multi-word phrases) ─────────
def test_canonicalise_forward_inverse_and_multiword():
    assert rsi.canonicalise("a", "causes", "b") == ("a", "CAUSE", "b")
    assert rsi.canonicalise("b", "is caused by", "a") == ("a", "CAUSE", "b")
    assert rsi.canonicalise("x", "contains", "y") == ("y", "PART_OF", "x")
    assert rsi.canonicalise("x", "is part of", "y") == ("x", "PART_OF", "y")
    assert rsi.canonicalise("a", "is unrelated to", "b") is None


def test_multiword_demos_induce_correct_relation(items):
    # A demo set expressed entirely with multi-word phrases must still induce
    # the rule (guards against re-splitting emitted strings on whitespace).
    for it in items[:12]:
        comp = rsi.RSIComponent("structural", rsi.FakeBackend())
        triples = [("a", "is part of", "b"), ("b", "is a component of", "c"),
                   ("c", "belongs to", "d")]
        assert comp.induce(triples) == "PART_OF"


def test_negation_licenses_no_rule():
    comp = rsi.RSIComponent("structural", rsi.FakeBackend())
    assert comp.induce([("a", "not causes", "b"),
                        ("b", "not causes", "c")]) is None
    confound = rsi.RSIComponent("counterexample", rsi.FakeBackend())
    assert confound.induce([("a", "not causes", "b"),
                            ("b", "not causes", "c")]) == "CAUSE"


# ── structural arm: graded difficulty + negative rules ────────────────────
def test_structural_near_far_foil_negative(summary):
    for pool in ("primary", "sealed"):
        d = summary[pool]
        assert d["near"] >= 0.90
        assert d["far"] >= 0.90
        assert d["foil"] >= 0.90
        assert d["negative_reject"] >= 0.85
        assert d["relation_coverage"] >= 0.95


def test_monotonic_curve_and_graded_difficulty(summary):
    for pool in ("primary", "sealed"):
        d = summary[pool]
        assert d["monotonic_max_drop"] <= rsi.TH["monotonic_tol"]
        assert d["graded_max_drop"] <= rsi.TH["graded_drop_max"]
        for k in map(str, rsi.K_LEVELS):
            b = d["by_k"][k]
            assert b["near"] >= b["far"] - rsi.TH["graded_drop_max"]
            assert b["far"] >= b["foil"] - rsi.TH["graded_drop_max"]
            assert b["negative"] >= rsi.TH["negative_reject_min"]


def test_growth_headroom_reported_not_gated(summary):
    for pool in ("primary", "sealed"):
        # At a saturated 1.00 the headroom is 0 and is REPORTED, not hidden;
        # decide() must still PASS. G6 flags the saturation separately.
        assert isinstance(summary[pool]["growth_headroom"], float)


def test_additive_json_fields_present(rows):
    required = {"k", "condition", "arm", "backend", "gold"}
    for r in rows[:8]:
        assert required.issubset(r.keys())
        assert r["k"] in rsi.K_LEVELS


# ── confound arms (the controls that give the test power) ──────────────────
def test_naive_fails_permutation_and_accepts_foil(rows):
    for pool in ("primary", "sealed"):
        ex = [r for r in rows if r["arm"] == "naive" and r["pool"] == pool
              and r["condition"] == "near"]
        pm = [r for r in rows if r["arm"] == "naive" and r["pool"] == pool
              and r["condition"] == "permuted"]
        assert rsi._rate(ex, "exact_structural_match") - \
            rsi._rate(pm, "exact_structural_match") >= rsi.TH["permuted_hard_margin"]
        foil = [r for r in rows if r["arm"] == "naive" and r["pool"] == pool
                and r["condition"] == "foil"]
        assert rsi._rate(foil, "correct") <= 0.10


def test_counterexample_arm_detected_as_non_resistant(summary):
    for pool in ("primary", "sealed"):
        assert summary[pool]["counterexample_reject"] <= \
            rsi.TH["counterexample_reject_max"]


def test_chance_arm_at_chance_and_far_beats_it(summary):
    for pool in ("primary", "sealed"):
        assert summary[pool]["chance"] <= rsi.TH["chance_max"]
        assert summary[pool]["far_minus_chance"] >= rsi.TH["far_over_chance"]


# ── instrument controls ────────────────────────────────────────────────────
def test_eight_instrument_controls_pass(items, rows, summary):
    checks = rsi.instrument_controls(items, rows, summary)
    assert len(checks) == 8
    assert all(ok for _, ok in checks), [n for n, ok in checks if not ok]


# ── decision rule hard fails ───────────────────────────────────────────────
def _base_summary():
    per = {"near": 1.0, "far": 1.0, "foil": 1.0, "negative_reject": 1.0,
           "counterexample_reject": 0.0, "permuted": 1.0, "chance": 0.20,
           "far_minus_chance": 0.80, "relation_coverage": 1.0,
           "monotonic_max_drop": 0.0, "graded_max_drop": 0.0,
           "growth_headroom": 0.0, "by_k": {}, "monotonic_by_condition": {}}
    return {"primary": dict(per), "sealed": dict(per)}


def test_decision_passes_clean_summary():
    rules, passed = rsi.decide(_base_summary(), True, "FLAG")
    assert passed, [k for k, v in rules.items() if not v]


def test_decision_fails_on_monotonicity_violation():
    s = _base_summary()
    s["primary"]["monotonic_max_drop"] = 0.20
    rules, passed = rsi.decide(s, True, "FLAG")
    assert not passed and not rules["primary_monotonic<=0.05"]


def test_decision_fails_on_negative_rule_leak():
    s = _base_summary()
    s["primary"]["negative_reject"] = 0.50
    rules, passed = rsi.decide(s, True, "FLAG")
    assert not passed and not rules["primary_negative_reject>=0.80"]


def test_decision_fails_on_graded_inversion():
    s = _base_summary()
    s["primary"]["graded_max_drop"] = 0.30
    rules, passed = rsi.decide(s, True, "FLAG")
    assert not passed and not rules["primary_graded<=0.15"]


def test_decision_fails_on_sealed_gap_and_controls():
    s = _base_summary()
    s["sealed"]["near"] = 0.70
    rules, passed = rsi.decide(s, False, "FLAG")
    assert not passed
    assert not rules["sealed_near>=primary_near-0.10"]
    assert not rules["instrument_controls_ok"]


def test_decision_fails_on_backend_error():
    rules, passed = rsi.decide(_base_summary(), True, "FLAG", "live", "boom")
    assert not passed and not rules["backend_error_free"]


# ── no engine / no network / no save on the fake path ──────────────────────
def test_fake_path_does_not_boot_engine():
    src = open(os.path.join(SCRIPTS, "measure_learn5_rsi.py"),
               encoding="utf-8").read()
    code = re.sub(r'""".*?"""', "", src, flags=re.S)
    code = "\n".join(line.split("#", 1)[0] for line in code.splitlines())
    assert "CognitiveChatEngine" in code             # real path exists
    assert code.index("class RealBackend") < code.index("CognitiveChatEngine")
    assert rsi.NETWORK_ATTEMPTS == [] and rsi.SAVE_ATTEMPTS == []


# ── LIVE OBSERVATION WIRING (fake engine emission; NO engine boot) ──────────
# The stub duck-types process_turn; no CognitiveChatEngine import/boot occurs.
class _ResponseEngine:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.graph = None
        self._concept_keywords = {}

    def process_turn(self, s):
        out = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return out


def _live(responses):
    b = rsi.RealBackend()          # no data_dir creation, no boot
    b._engine = _ResponseEngine(responses)   # bypasses _boot()
    return b


def test_real_present_grades_engine_not_demo_triples(items):
    it = items[0]
    q = it["near"]
    cand = q["candidates"][0]
    b = _live(["relation=%s target=%s verdict=match" % (it["relation"], cand)])
    demos = [("a", "causes", "b")]

    emitted = b.present(demos, it, 1, "near")
    assert isinstance(emitted, list) and all(isinstance(x, str) for x in emitted)
    assert emitted[0].startswith("relation=")         # engine emission, not demos
    assert b.api_calls["process_turn"] == 1
    assert b.engine_response_unparsed == 0

    ans = b.last_engine_answer
    assert ans["relation"] == it["relation"]
    assert ans["target"] == cand and ans["verdict"] == "match"
    # The fake backend still returns the demo triples (oracle path intact).
    assert rsi.FakeBackend().present(demos, it, 1, "near") == demos
    assert rsi.FakeBackend().engine_graded is False


def test_unparsed_response_scores_no_answer(items):
    it = items[0]
    b = _live([""])                       # no structured answer recoverable
    before = b.engine_response_unparsed
    b.present([("a", "causes", "b")], it, 1, "near")
    assert b.last_engine_answer["verdict"] == "no_answer"
    assert b.last_unparsed is True
    assert b.engine_response_unparsed == before + 1

    rec = rsi.RSIComponent("structural", b).run_item(it, 1, "near")
    assert rec["output"]["verdict"] == "no_answer"
    assert rec["engine_unparsed"] is True
    assert rsi.grade(rec)["flags"]["correct"] is False
    # NO ANSWER is a failure on reject conditions too (not a free pass).
    foil = rsi.RSIComponent("structural", b).run_item(it, 1, "foil")
    assert rsi.grade(foil)["flags"]["correct"] is False


def test_engine_emission_mutation_moves_grade(items):
    it = items[0]
    q = it["near"]
    good = q["gold_target"]
    wrong = next(c for c in q["candidates"] if c != good)
    bg = _live(["relation=%s target=%s verdict=match" % (it["relation"], good)])
    bb = _live(["relation=%s target=%s verdict=match" % (it["relation"], wrong)])
    rg = rsi.grade(rsi.RSIComponent("structural", bg).run_item(it, 1, "near"))
    rb = rsi.grade(rsi.RSIComponent("structural", bb).run_item(it, 1, "near"))
    assert rg["flags"]["correct"] is True
    assert rb["flags"]["correct"] is False       # one wrong emission moved the score


def test_confound_arms_keep_deterministic_controls_under_live_backend(items):
    it = items[0]
    b = _live(["verdict=reject"])          # engine declines the foil
    rows = [rsi.grade(rsi.RSIComponent(arm, b).run_item(it, 1, "foil"))
            for arm in ("structural", "naive", "chance", "counterexample")]
    # Only the structural arm's grade came from the engine; the confounds are
    # unchanged deterministic manipulations, so the controls still discriminate.
    assert rows[0]["flags"]["correct"] is True
    assert rows[1]["flags"]["correct"] is False   # naive accepts foil
    assert "engine_graded" in rows[0]


def test_threshold_neutrality_all_old_criteria_identical():
    assert rsi.TH == {
        "near_min": 0.75, "far_min": 0.70, "foil_reject_min": 0.85,
        "negative_reject_min": 0.80, "monotonic_tol": 0.05,
        "graded_drop_max": 0.15, "sealed_drop_max": 0.10,
        "relation_coverage_min": 0.95, "chance_max": 0.35,
        "far_over_chance": 0.20, "foil_reject_hard_min": 0.70,
        "counterexample_reject_max": 0.20, "permuted_hard_margin": 0.15,
        "permuted_margin": 0.10,
    }
    expected = set()
    for pool in ("primary", "sealed"):
        expected |= {
            pool + "_near>=0.75", pool + "_far>=0.70",
            pool + "_foil_reject>=0.85", pool + "_negative_reject>=0.80",
            pool + "_monotonic<=0.05", pool + "_graded<=0.15",
            pool + "_far>=chance+0.20", pool + "_relation_coverage>=0.95"}
    expected |= {"sealed_near>=primary_near-0.10", "instrument_controls_ok",
                 "no_network", "no_save", "backend_error_free",
                 "saturation_flag_recorded"}
    rules, passed = rsi.decide(_base_summary(), True, "FLAG")
    assert set(rules) == expected and passed
    # Fake rows carry no live-observation fields: byte-identical old criteria.
    fake_rows = rsi.run_sweep(rsi.build_items()[:4], backend=rsi.FakeBackend())
    assert all("engine_graded" not in r and "engine_answer" not in r
               for r in fake_rows)


def test_summary_carries_additive_provenance(items, rows):
    out = rsi.summarize(items, rows)
    assert out["engine_observation_missing"] == 0
    assert out["engine_response_unparsed"] == 0
