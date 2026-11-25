"""Fake-based unit tests for the pre-registered TRA harness.

NO ENGINE BOOT. Every test here uses ``FakeBackend`` and pure helpers; no
``CognitiveChatEngine`` is imported, constructed or saved, and no benchmark is
run. These tests pin the frozen thresholds/manifests, the phase structure, the
parser, C1-C13 behaviour on the oracle, fake-backend invalidation (C11),
saturation mutation controls, additive JSON and flag restoration.
"""
import collections
import json
import os
import subprocess
import sys

import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKTREE_SCRIPTS = os.path.join(PROJ, "scripts")
for _p in (PROJ, WORKTREE_SCRIPTS, os.path.join(PROJ, "ravana", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import measure_tra as mt                    # noqa: E402
import tra_items                            # noqa: E402


# Frozen pre-registration snapshots (spec section 4 / TH in the harness).
MANIFEST = {
    "primary": "711e50a489fe5fa5f375b2267a3944d1f5b06741713f34cacafc9a4abde43d76",
    "sealed": "405fdcbd99d7904304fc176acd35ee5452b051d8791ecbf862a6588d4bf6acd5",
}
BANK_SHA = "03537712fec70e7f3bd6226d7e97ee636b2dd99cac47d197eba3f0620500c512"
TH_SNAPSHOT = {
    "transfer_min": 0.75, "near_min": 0.75, "far_min": 0.70,
    "far_chance_min": 0.20, "chance_max": 0.35,
    "foil_reject_min": 0.85, "foil_hard_min": 0.70,
    "negative_reject_min": 0.80, "negative_hard_min": 0.70,
    "permuted_margin": 0.10, "permuted_hard_margin": 0.15,
    "monotonic_tol": 0.05, "graded_drop_max": 0.15,
    "coverage_min": 0.95, "coverage_hard_min": 0.90,
    "retention_min": 0.80, "control_delta_max": 0.05,
    "sealed_drop_max": 0.10, "growth_min": 0.15,
    "c1_near_min": 0.90, "c2_foil_min": 0.90, "c2_naive_max": 0.10,
    "c5_struct_min": 0.85, "c5_counter_max": 0.20,
    "c8_coverage_min": 0.95,
}


@pytest.fixture(scope="module")
def items():
    return tra_items.build_items()


@pytest.fixture(scope="module")
def backend():
    return mt.FakeBackend()


@pytest.fixture(scope="module")
def rows(items, backend):
    return mt.run_sweep(items, backend=backend)


@pytest.fixture(scope="module")
def summary(items, rows):
    return mt.summarize(items, rows)


@pytest.fixture(scope="module")
def checks(items, rows, summary, backend):
    return mt.instrument_controls(items, rows, summary, backend, False, None)


# ── bank counts, split, hashes ──────────────────────────────────────────────
def test_module_resolves_to_worktree():
    assert os.path.realpath(mt.__file__).startswith(os.path.realpath(PROJ))
    assert os.path.realpath(tra_items.__file__).startswith(os.path.realpath(PROJ))


def test_400_families_validate(items):
    assert tra_items.validate_items(items) == []
    assert len(items) == 400
    pools = collections.Counter(it["pool"] for it in items)
    assert pools == {"primary": 200, "sealed": 200}


def test_exactly_40_families_per_k_per_pool(items):
    counts = tra_items.pool_k_counts(items)
    for pool in ("primary", "sealed"):
        for k in mt.K_LEVELS:
            assert counts[(pool, k)] == 40
    # one k per item: no item appears at two levels
    for it in items:
        assert it["k"] in mt.K_LEVELS
    assert mt.K_LEVELS == (0, 1, 2, 4, 8)


def test_manifest_hashes_frozen(items):
    assert tra_items.manifest_hashes(items) == MANIFEST
    assert tra_items.item_manifest_hash(items) == BANK_SHA
    assert tra_items.manifest_hashes(tra_items.build_items()) == MANIFEST
    assert len(MANIFEST["primary"]) == len(MANIFEST["sealed"]) == 64


def test_hash_split_deterministic(items):
    again = tra_items.build_items()
    assert [(x["item_id"], x["pool"], x["k"]) for x in items] == [
        (x["item_id"], x["pool"], x["k"]) for x in again]


def test_th_frozen_snapshot():
    assert mt.TH == TH_SNAPSHOT


# ── teaching text is provenance-safe ────────────────────────────────────────
def test_teaching_surface_names_slot_only_with_exemplar_tokens(items):
    it = items[0]
    ex = it["taught_exemplars"][0]
    surface = ex["surface"]
    assert ex["r1"] in surface and ex["r2"] in surface
    assert ex["type_label"] in surface and it["type_label"] in surface
    # answer key / held-out metadata never reach the teaching channel
    gold = it["unseen_test_words"]["near"]["gold"]
    assert gold not in surface
    for target in it["world_model"].values():
        for word in target:
            assert word not in surface
    assert it["domain"] not in surface and it["far_domain"] not in surface


def test_k0_uses_eight_matched_neutral_turns(items):
    it = next(x for x in items if x["k"] == 0)
    surfaces = mt.teaching_surfaces(it, 0, "near", "structural")
    assert len(surfaces) == max(mt.K_LEVELS) == 8
    assert all(" is a " not in s and " is an " not in s for s in surfaces)


def test_k_teaching_turn_count(items):
    it = next(x for x in items if x["k"] == 8)
    assert len(mt.teaching_surfaces(it, 8, "near", "structural")) == 8
    assert len(mt.teaching_surfaces(it, 2, "near", "structural")) == 2
    neg = mt.teaching_surfaces(it, 4, "negative", "structural")
    assert len(neg) == 4 and all(" does not " in s for s in neg)


def test_query_surface_has_no_key_tokens_or_gold(items):
    it = items[0]
    for condition in mt.CONDITIONS:
        surface = mt.query_surface(it, condition)
        low = surface.lower()
        for token in ("role=", "target=", "verdict=", "world_model", "type_sets"):
            assert token not in low
        q = mt.query_for(it, condition)
        if q["gold"]:
            # Gold is legitimately ONE unmarked option in the forced-choice
            # display; what must not leak is an answer-key annotation.
            assert surface.count(q["gold"]) == 1
            assert "gold" not in low and "answer:" not in low
        para = mt.query_surface(it, condition, paraphrase=True)
        assert para != surface
        assert q["subject"] in para


def test_phase_structure_single_graded_query(items, rows):
    first = next(it for it in items if it["pool"] == "primary")
    rr = [r for r in rows if r["id"] == first["item_id"]]
    phases = collections.Counter(r["phase"] for r in rr)
    assert phases["baseline"] == 1
    assert phases["query"] == len(mt.CONDITIONS) * len(mt.ARMS)
    assert phases["permuted_probe"] == 1
    assert phases["immediate"] == len(mt.CONDITIONS)
    assert phases["delayed"] == 2
    # exactly ONE graded phase per item/condition/arm
    graded = [r for r in rr if r["phase"] == "query"]
    assert len(graded) == len(mt.CONDITIONS) * len(mt.ARMS)
    assert all(r["gold"]["should_reject"] == (r["condition"] in ("foil", "negative"))
               for r in graded)


def test_baseline_is_cold_even_for_high_k(items, rows):
    it = next(x for x in items if x["k"] == 8)
    base = [r for r in rows if r["id"] == it["item_id"]
            and r["phase"] == "baseline"][0]
    assert base["input"]["teaching_turns"] == 0
    assert not base["flags"]["correct"]


# ── parser ──────────────────────────────────────────────────────────────────
def test_parser_explicit_form(items):
    it = items[0]
    q = mt.query_for(it, "near")
    ans = mt._parse_engine_answer(
        ["role=%s target=%s verdict=match" % (it["role"], q["gold"])],
        it, "near", q["subject"], q["candidates"])
    assert ans["role"] == it["role"]
    assert ans["target"] == q["gold"]
    assert ans["verdict"] == "match"


def test_parser_bare_option_is_a_forced_choice(items):
    it = items[0]
    q = mt.query_for(it, "near")
    ans = mt._parse_engine_answer([q["gold"]], it, "near", q["subject"],
                                  q["candidates"])
    assert ans["verdict"] == "match" and ans["target"] == q["gold"]
    assert ans["role"] == it["role"]


def test_parser_hedge_does_not_become_evidence(items):
    it = items[0]
    q = mt.query_for(it, "near")
    ans = mt._parse_engine_answer(
        ["I'm not totally sure about %s yet." % it["type_label"],
         "role=EFFECT target=%s verdict=match" % q["gold"]],
        it, "near", q["subject"], q["candidates"])
    assert ans["target"] == q["gold"] and ans["verdict"] == "match"


def test_parser_unparseable_is_no_answer(items):
    it = items[0]
    q = mt.query_for(it, "near")
    assert mt._parse_engine_answer([""],
                                   it, "near", q["subject"], q["candidates"]) is None
    ans = mt._parse_engine_answer(["perhaps"], it, "near", q["subject"],
                                  q["candidates"])
    assert ans is None


def test_parser_reject_verdict(items):
    it = items[0]
    q = mt.query_for(it, "foil")
    ans = mt._parse_engine_answer(["role=%s target=none verdict=reject" % it["role"]],
                                  it, "foil", q["subject"], q["candidates"])
    assert ans["verdict"] == "reject" and ans["target"] is None


# ── instrument controls / fake invalidation ────────────────────────────────
def test_all_instrument_controls_except_c11_pass_on_fake(checks):
    by_id = {c["id"]: c for c in checks}
    assert set(by_id) == {"C%d" % i for i in range(1, 14)}
    for cid in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
                "C9", "C10", "C12", "C13"):
        assert by_id[cid]["ok"] is True, (cid, by_id[cid]["detail"])
    assert by_id["C11"]["ok"] is False      # fake arm MUST be invalid as capability


def test_fake_backend_cannot_emit_capability(items, rows, summary, checks, backend):
    rules, passed = mt.decide(summary, checks, backend, False, None, "FLAG", True)
    assert passed is False
    assert rules["c11_anti_fake_ok"] is False
    assert rules["backend_live"] is False
    assert rules["instrument_controls_ok"] is True
    assert backend.engine_graded is False
    assert backend.api_calls["boot"] == 0 and backend.api_calls["process_turn"] == 0


def test_summary_endpoints_and_retention(summary):
    for pool in ("primary", "sealed"):
        d = summary[pool]
        assert d["transfer"] == 1.0
        assert d["near"] == 0.8 and d["far"] == 0.8
        assert d["foil"] == 1.0 and d["negative_reject"] == 1.0
        assert d["chance"] == 0.0
        assert d["monotonic_max_drop"] == 0.0
        assert d["graded_max_drop"] == 0.0
        assert d["delayed_retention"] == 1.0
        assert d["growth"] == 1.0
        assert d["control_max_delta"] == 0.0
        assert d["role_coverage"] == 1.0


# ── saturation + mutation controls ─────────────────────────────────────────
def test_saturation_gate_flags_boundary():
    import front_gates
    sat = front_gates.gate_saturation(
        {"tra_near": 100.0, "tra_negative_reject": 100.0}, None, "tra")
    assert sat.status == "FLAG"
    assert "tra_near" in sat.as_dict("tra")["metrics"]["saturated"]


def test_mutation_control_moves_verdict(items):
    mut = mt.mutation_control(items)
    assert mut["M-A"]["moved"] is True
    assert mut["M-B"]["moved"] is True
    assert mut["moved"] is True
    assert mut["M-A"]["before"] == "match" and mut["M-A"]["after"] == "reject"


def test_mutation_control_uses_supplied_hook(items):
    def hook(item, kind):
        item = dict(item)
        item["type_label"] = "not-a-taught-type"
        return item
    mut = mt.mutation_control(items, bank_mutator=hook)
    assert mut["moved"] is True


def test_c10_type_contrast_fail_closed():
    probe = mt.type_contrast_probe()
    assert probe["typed_pick"] == "effect"
    assert probe["flipped_pick"] == "weather-event"
    assert probe["disjoint_pick"] is None
    assert probe["ok"] is True


# ── no answer-key leakage into graded inputs ────────────────────────────────
def test_rows_never_carry_answer_key_into_engine_input(items, rows):
    for rec in rows[::97]:
        inp = rec["input"]
        assert "world_model" not in inp and "type_sets" not in inp
        assert "gold" not in inp and "gold_target" not in inp
        assert rec["observation_source"] in ("fake_oracle", "query_response")
        assert rec["trust_gate_disabled"] is False
    # query surfaces (not just stored strings) are also clean
    for it in items[::37]:
        surface = mt.query_surface(it, "near")
        gold = it["unseen_test_words"]["near"]["gold"]
        assert surface.count(gold) == 1
        assert "gold" not in surface.lower()
        assert it["domain"] not in surface and it["far_domain"] not in surface


def test_import_does_not_boot_engine():
    code = ("import sys, os; "
            "sys.path.insert(0, %r); "
            "import measure_tra; "
            "print('OK' if 'ravana.chat.engine' not in sys.modules else 'BOOTED')"
            % WORKTREE_SCRIPTS)
    env = dict(os.environ)
    env["PYTHONPATH"] = WORKTREE_SCRIPTS + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", code], cwd=PROJ, env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout, proc.stdout


# ── flag lifecycle: real arm only, always restored ─────────────────────────
def test_flag_context_sets_and_restores_prior_value():
    key = mt.SLOT_ENV_FLAG
    old = os.environ.get(key)
    try:
        os.environ[key] = "0"
        with mt.tra_slots_flag():
            assert os.environ[key] == "1"
        assert os.environ[key] == "0"
        os.environ.pop(key, None)
        with mt.tra_slots_flag():
            assert os.environ[key] == "1"
        assert key not in os.environ
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def test_fake_sweep_never_touches_slot_flag(items):
    key = mt.SLOT_ENV_FLAG
    old = os.environ.get(key)
    try:
        os.environ.pop(key, None)
        mt.run_sweep(items[:2], backend=mt.FakeBackend())
        assert key not in os.environ
    finally:
        if old is not None:
            os.environ[key] = old


# ── additive JSON output / validate-only / exit codes ───────────────────────
def test_validate_only_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "MANIFEST_PATH", str(tmp_path / "manifest.json"))
    assert mt.main(["--validate-only"]) == 0
    data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert data["manifest_sha256"] == MANIFEST
    assert data["bank_sha256"] == BANK_SHA
    assert data["thresholds"] == TH_SNAPSHOT


def test_fake_main_is_additive_and_never_capability(tmp_path, monkeypatch):
    data_path = tmp_path / "rows.jsonl"
    monkeypatch.setattr(mt, "DATA_PATH", str(data_path))
    monkeypatch.setattr(mt, "SUMMARY_PATH", str(tmp_path / "summary.json"))
    monkeypatch.setattr(mt, "GATE_PATH", str(tmp_path / "gates.jsonl"))
    key = mt.SLOT_ENV_FLAG
    old = os.environ.get(key)
    try:
        os.environ.pop(key, None)
        assert mt.main([]) == 1              # fake arm: no capability claim
        first = len(data_path.read_text(encoding="utf-8").splitlines())
        assert mt.main([]) == 1
        second = len(data_path.read_text(encoding="utf-8").splitlines())
        assert second == 2 * first and first > 0
        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert summary["interpretation"] == "mechanism_probe_only_fake_arm"
        assert summary["pass"] is False
        assert summary["backend"] == "fake" and summary["live"] is False
        assert key not in os.environ
    finally:
        if old is not None:
            os.environ[key] = old


def test_real_backend_is_lazy_and_flag_gated():
    # Constructing the object must not import or boot the engine.
    old_flag = os.environ.get(mt.SLOT_ENV_FLAG)
    obj = mt.RealBackend(data_dir=os.path.join(str(os.devnull), "unused"))
    assert obj.engine_graded is True and obj.label == "live"
    assert obj._engine is None
    assert obj.api_calls["boot"] == 0
    assert mt.SLOT_ENV_FLAG == "RAVANA_TRA_SLOTS"
    assert os.environ.get(mt.SLOT_ENV_FLAG) != "1" or old_flag == "1"
