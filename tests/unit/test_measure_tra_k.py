"""Fake-only unit tests for the pre-registered TRA-K harness.

NO ENGINE BOOT. Every test uses ``FakeBackend`` and pure helpers; no
``CognitiveChatEngine`` is imported, constructed or saved. These tests pin the
frozen thresholds/manifests, the item schema with ``channel_evidence``, the
P1-P6+SI phase structure, the parser, C1-C15 (C11 invalid on fake), the two new
HARD-FAIL controls, G6 saturation + M-A/M-B, pluggable channel JSON, focused
subsets, additive JSON and flag restoration.
"""
import collections
import copy
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

import measure_tra_k as mt                    # noqa: E402
import tra_k_items as tki                     # noqa: E402


# Frozen pre-registration snapshots (spec / TH in the harness).
MANIFEST = {
    "primary": "b3545ec4fd1553421b78a925d56d891bf70c1f275f4cb74b61dc138109b88e60",
    "sealed": "e2b586eeaab8b65a4e6a23f2c512ce8161430737c165401b010c7f66132c887b",
}
BANK_SHA = "0ed8bdc2ee1dd555adc4595868bce6794a14085c0d02761b8a957fc152e147cd"
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
ALL_CONTROL_IDS = {"C%d" % i for i in range(1, 16)}


@pytest.fixture(scope="module")
def items():
    return tki.build_items()


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


# ── module resolution, bank counts, split, hashes ───────────────────────────
def test_module_resolves_to_worktree():
    """The imported modules must be THIS worktree's copies, not main's."""
    for mod in (mt, tki):
        real = os.path.realpath(mod.__file__)
        assert real.startswith(os.path.realpath(PROJ)), (mod.__name__, real)
        assert os.path.dirname(real) == os.path.realpath(WORKTREE_SCRIPTS)
    assert mt.HASH_PREFIX == "RAVANA-TRA-K-v1"
    assert mt.COMPONENT == "TRA-K"


def test_400_families_validate(items):
    assert tki.validate_items(items) == []
    assert len(items) == 400
    pools = collections.Counter(it["pool"] for it in items)
    assert pools == {"primary": 200, "sealed": 200}


def test_exactly_40_families_per_k_per_pool(items):
    counts = tki.pool_k_counts(items)
    for pool in ("primary", "sealed"):
        for k in mt.K_LEVELS:
            assert counts[(pool, k)] == 40
    for it in items:
        assert it["k"] in mt.K_LEVELS
    assert mt.K_LEVELS == (0, 1, 2, 4, 8)


def test_manifest_hashes_frozen(items):
    assert tki.manifest_hashes(items) == MANIFEST
    assert tki.item_manifest_hash(items) == BANK_SHA
    assert tki.manifest_hashes(tki.build_items()) == MANIFEST
    assert len(MANIFEST["primary"]) == len(MANIFEST["sealed"]) == 64


def test_hash_split_deterministic(items):
    again = tki.build_items()
    assert [(x["item_id"], x["pool"], x["k"]) for x in items] == [
        (x["item_id"], x["pool"], x["k"]) for x in again]


def test_th_frozen_snapshot():
    assert mt.TH == TH_SNAPSHOT
    frozen = json.dumps(TH_SNAPSHOT, sort_keys=True, separators=(",", ":"))
    actual = json.dumps(mt.TH, sort_keys=True, separators=(",", ":"))
    assert actual.encode("utf-8") == frozen.encode("utf-8")


# ── channel evidence schema (the ONLY new item field) ───────────────────────
def test_every_item_has_provenance_stamped_channel_evidence(items):
    for it in items:
        ev = it["channel_evidence"]
        assert isinstance(ev, list) and ev
        for fact in ev:
            for key in tki.REQUIRED_FACT_KEYS:
                assert fact.get(key) not in (None, "")
            assert fact["predicate"].lower() in tki.ISA_PREDICATES
            assert fact["source"]
            assert fact["source_id"] and fact["snapshot"]
        # Oracle fields remain separate, never merged into the channel field.
        assert "type_sets" not in fact and "world_model" not in fact


def test_channel_evidence_never_names_answer_key(items):
    for it in items:
        key = tki.answer_key_tokens(it)
        subjects = {f["subject"] for f in it["channel_evidence"]}
        assert not (key & subjects)
        objects = {f["object"] for f in it["channel_evidence"]}
        assert objects <= {it["type_label"]}


def test_channel_overlap_reports_benign_overlap(items):
    rep = tki.channel_overlap(items)
    assert rep["ok"] is True and not rep["offenders"]
    tot = rep["totals"]
    assert tot["overlap_type_sets"] == 3200
    assert tot["overlap_world_model"] == 0
    assert tot["answer_key_leak"] == 0
    assert tot["oracle_domain_tag_leak"] == 0
    assert tot["provenance_missing"] == 0


def test_channel_assertion_uses_only_fact_fields():
    fact = {"subject": "drizzle", "predicate": "is_a", "object": "effect",
            "source": "unit-test", "source_id": "x", "snapshot": "s"}
    assert mt_chan(fact) == "drizzle is an effect."
    assert tki.channel_assertion(dict(fact, predicate="related_to")) is None
    assert tki.channel_assertion(dict(fact, subject="")) is None


def mt_chan(fact):
    """Thin wrapper so the test names the harness seam explicitly."""
    return tki.channel_assertion(fact)


def test_channel_teaching_surfaces_are_fact_only(items):
    it = next(x for x in items if x["k"] == 8)
    surfaces = mt.channel_teaching_surfaces(it)
    expected = [tki.channel_assertion(f) for f in it["channel_evidence"]]
    assert surfaces == [s for s in expected if s][:max(mt.K_LEVELS)]
    gold = it["unseen_test_words"]["near"]["gold"]
    assert all(gold not in s for s in surfaces)


# ── teaching / query surfaces are provenance-safe ───────────────────────────
def test_teaching_surface_names_slot_only_with_exemplar_tokens(items):
    it = items[0]
    ex = it["taught_exemplars"][0]
    surface = ex["surface"]
    assert ex["r1"] in surface and ex["r2"] in surface
    assert ex["type_label"] in surface and it["type_label"] in surface
    gold = it["unseen_test_words"]["near"]["gold"]
    assert gold not in surface
    for target in it["world_model"].values():
        for word in target:
            assert word not in surface
    assert it["domain"] not in surface and it["far_domain"] not in surface


def test_k0_channel_arm_still_gets_channel_evidence(items):
    it = next(x for x in items if x["k"] == 0)
    surfaces = mt.teaching_surfaces(it, 0, "near", mt.CHANNEL_ARM)
    assert len(surfaces) == max(mt.K_LEVELS) + len(mt.channel_teaching_surfaces(it))
    neutral = surfaces[:max(mt.K_LEVELS)]
    assert all(" is a " not in s and " is an " not in s for s in neutral)
    assert all(s.endswith(".") for s in surfaces[max(mt.K_LEVELS):])


def test_k_teaching_turn_count(items):
    it = next(x for x in items if x["k"] == 8)
    assert len(mt.teaching_surfaces(it, 8, "near", "ostensive")) == 8
    assert len(mt.teaching_surfaces(it, 2, "near", "ostensive")) == 2
    neg = mt.teaching_surfaces(it, 4, "negative", "ostensive")
    assert len(neg) == 4 and all(" does not " in s for s in neg)


def test_query_surface_has_no_key_tokens_or_gold(items):
    it = items[0]
    for condition in ("near", "far", "foil", "negative", "unrelated"):
        surface = mt.query_surface(it, condition)
        low = surface.lower()
        for token in ("role=", "target=", "verdict=", "world_model", "type_sets",
                      "channel_evidence", "gold", "answer:"):
            assert token not in low
        q = mt.query_for(it, condition)
        if q["gold"]:
            assert surface.count(q["gold"]) == 1
        para = mt.query_surface(it, condition, paraphrase=True)
        assert para != surface and q["subject"] in para
    assert mt.query_for(it, "unrelated")["subject"] == tki.UNRELATED_SUBJECT
    assert mt.query_for(it, "unrelated")["gold"] is None


def test_phase_structure_single_graded_query(items, rows):
    first = next(it for it in items if it["pool"] == "primary")
    rr = [r for r in rows if r["id"] == first["item_id"]]
    phases = collections.Counter(r["phase"] for r in rr)
    assert phases["baseline"] == 1
    assert phases["query"] == len(mt.CONDITIONS) * len(mt.ARMS)
    assert phases["permuted_probe"] == 1
    assert phases["subject_independence"] == 1
    assert phases["immediate"] == len(mt.CONDITIONS)
    assert phases["delayed"] == 2
    graded = [r for r in rr if r["phase"] == "query"]
    assert all(r["gold"]["should_reject"] ==
               (r["condition"] in ("foil", "negative", "unrelated"))
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
    assert ans["role"] == it["role"] and ans["target"] == q["gold"]
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
    assert mt._parse_engine_answer([""], it, "near", q["subject"],
                                   q["candidates"]) is None
    assert mt._parse_engine_answer(["perhaps"], it, "near", q["subject"],
                                   q["candidates"]) is None


def test_parser_reject_verdict(items):
    it = items[0]
    q = mt.query_for(it, "foil")
    ans = mt._parse_engine_answer(
        ["role=%s target=none verdict=reject" % it["role"]],
        it, "foil", q["subject"], q["candidates"])
    assert ans["verdict"] == "reject" and ans["target"] is None


# ── instrument controls / fake invalidation ────────────────────────────────
def test_all_instrument_controls_except_c11_pass_on_fake(checks):
    by_id = {c["id"]: c for c in checks}
    assert set(by_id) == ALL_CONTROL_IDS
    for cid in sorted(ALL_CONTROL_IDS - {"C11"}):
        assert by_id[cid]["ok"] is True, (cid, by_id[cid]["detail"])
    assert by_id["C11"]["ok"] is False       # fake arm MUST fail the live gate


def test_fake_backend_cannot_emit_capability(items, rows, summary, checks, backend):
    rules, passed = mt.decide(summary, checks, backend, False, None, "FLAG", True)
    assert passed is False
    assert rules["c11_anti_fake_ok"] is False
    assert rules["backend_live"] is False
    assert rules["instrument_controls_ok"] is True
    assert rules["subject_independence_ok"] is True
    assert rules["channel_independence_ok"] is True
    assert backend.engine_graded is False
    assert backend.api_calls["boot"] == 0
    assert backend.api_calls["process_turn"] == 0


def test_summary_endpoints_and_retention(summary):
    for pool in ("primary", "sealed"):
        d = summary[pool]
        assert d["transfer"] == 1.0
        assert d["near"] == 1.0 and d["far"] == 1.0   # channel present at k=0
        assert d["foil"] == 1.0 and d["negative_reject"] == 1.0
        assert d["chance"] == 0.0
        assert d["monotonic_max_drop"] == 0.0
        assert d["graded_max_drop"] == 0.0
        assert d["delayed_retention"] == 1.0
        assert d["growth"] == 1.0
        assert d["control_max_delta"] == 0.0
        assert d["role_coverage"] == 1.0
        assert d["ostensive_transfer"] == 1.0


# ── new HARD-FAIL controls ──────────────────────────────────────────────────
def test_subject_independence_control_discriminates(items):
    it = items[0]
    q = mt.query_for(it, "near")
    copied = ["role=EFFECT target=%s verdict=match" % q["gold"]]
    bad = mt.subject_independence_probe(tki.UNRELATED_SUBJECT, q["candidates"],
                                        copied, it)
    good = mt.subject_independence_probe(
        tki.UNRELATED_SUBJECT, q["candidates"],
        ["I'm not totally sure about that."], it)
    assert bad["confident_match"] is True and bad["ok"] is False
    assert good["confident_match"] is False and good["ok"] is True


def test_channel_independence_audit_hard_fails_on_contamination(items, rows):
    clean = mt.channel_independence_audit(items, rows)
    assert clean["ok"] is True
    bad_item = copy.deepcopy(items[0])
    gold = bad_item["unseen_test_words"]["near"]["gold"]
    bad_item["channel_evidence"].append({
        "subject": gold, "predicate": "is_a",
        "object": bad_item["type_label"], "source": "contaminated",
        "source_id": "x", "snapshot": "x"})
    dirty = mt.channel_independence_audit([bad_item], rows)
    assert dirty["ok"] is False
    assert dirty["overlap"]["ok"] is False
    assert dirty["overlap"]["totals"]["answer_key_leak"] >= 1

    bad_row = copy.deepcopy(rows[0])
    bad_row["input"]["type_sets"] = items[0]["type_sets"]
    leaky = mt.channel_independence_audit(items, [bad_row])
    assert leaky["ok"] is False
    assert any(x["field"] == "type_sets" for x in leaky["input_leaks"])


# ── saturation + mutation controls ──────────────────────────────────────────
def test_saturation_gate_flags_boundary():
    import front_gates
    sat = front_gates.gate_saturation(
        {"tra_k_near": 100.0, "tra_k_negative_reject": 100.0}, None, "tra_k")
    assert sat.status == "FLAG"
    assert "tra_k_near" in sat.as_dict("tra_k")["metrics"]["saturated"]


def test_mutation_control_moves_verdict(items):
    mut = mt.mutation_control(items)
    assert mut["M-A"]["moved"] is True
    assert mut["M-B"]["moved"] is True
    assert mut["moved"] is True
    assert mut["M-A"]["before"] == "match" and mut["M-A"]["after"] == "reject"
    assert mut["M-B"]["after"] == "reject"


def test_mutation_control_uses_supplied_hook(items):
    def hook(item, kind):
        item = dict(item)
        item["channel_evidence"] = []
        return item
    mut = mt.mutation_control(items, bank_mutator=hook)
    assert mut["moved"] is True


def test_c10_type_contrast_fail_closed():
    probe = mt.type_contrast_probe()
    assert probe["typed_pick"] == "effect"
    assert probe["flipped_pick"] == "weather-event"
    assert probe["disjoint_pick"] is None and probe["ok"] is True


# ── no answer-key leakage / no engine boot ──────────────────────────────────
def test_rows_never_carry_oracle_fields_into_engine_input(items, rows):
    for rec in rows[::97]:
        inp = rec["input"]
        for field in ("world_model", "type_sets", "gold", "gold_target"):
            assert field not in inp
        assert rec["observation_source"] in ("fake_oracle", "query_response")
        assert rec["trust_gate_disabled"] is False
    for it in items[::37]:
        surface = mt.query_surface(it, "near")
        gold = it["unseen_test_words"]["near"]["gold"]
        assert surface.count(gold) == 1
        assert "gold" not in surface.lower()


def test_import_does_not_boot_engine():
    code = ("import sys; "
            "sys.path.insert(0, %r); "
            "import measure_tra_k; "
            "print('OK' if 'ravana.chat.engine' not in sys.modules else 'BOOTED')"
            % WORKTREE_SCRIPTS)
    env = dict(os.environ)
    env["PYTHONPATH"] = WORKTREE_SCRIPTS + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", code], cwd=PROJ, env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout, proc.stdout


# ── flag lifecycle: live arm only, always restored ──────────────────────────
def test_flag_context_sets_and_restores_prior_value():
    keys = (mt.SLOT_ENV_FLAG, mt.TRAK_ENV_FLAG)
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = "0"
        with mt.trak_slots_flag():
            assert all(os.environ[k] == "1" for k in keys)
        assert all(os.environ[k] == "0" for k in keys)
        for k in keys:
            os.environ.pop(k, None)
        with mt.trak_slots_flag():
            assert all(os.environ[k] == "1" for k in keys)
        assert all(k not in os.environ for k in keys)
    finally:
        for k in keys:
            if old[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old[k]


def test_fake_sweep_never_touches_flags(items):
    keys = (mt.SLOT_ENV_FLAG, mt.TRAK_ENV_FLAG)
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        mt.run_sweep(items[:2], backend=mt.FakeBackend())
        assert all(k not in os.environ for k in keys)
    finally:
        for k in keys:
            if old[k] is not None:
                os.environ[k] = old[k]


# ── pluggable channel JSON / focused subset / additive JSON ─────────────────
def test_external_channel_json_is_loaded(tmp_path, monkeypatch, items):
    it = items[0]
    demo = it["taught_exemplars"][0]["r2"]
    fact = {"subject": demo, "predicate": "is_a", "object": it["type_label"],
            "source": "ontology-B-test-snapshot", "source_id": "snap-1",
            "snapshot": "test-v1"}
    path = tmp_path / "channel.json"
    path.write_text(json.dumps({"items": {it["item_id"]: [fact]}}),
                    encoding="utf-8")
    monkeypatch.setenv(tki.CHANNEL_ENV, str(path))
    loaded = tki.load_channel_evidence(items)
    assert loaded["error"] is None and loaded["invalid"] == 0
    assert loaded["external_items"] == 1
    built = tki.build_items(channel_override=loaded["facts"])
    got = next(x for x in built if x["item_id"] == it["item_id"])
    assert got["channel_evidence"][0]["source"] == "ontology-B-test-snapshot"
    assert tki.validate_items(built) == []
    assert tki.channel_overlap(built)["ok"] is True


def test_invalid_channel_facts_are_rejected(tmp_path, items):
    it = items[0]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"items": {
        it["item_id"]: [{"subject": "x", "predicate": "is_a"}]}}),
        encoding="utf-8")
    loaded = tki.load_channel_evidence(items, path=str(path), env="NEVER_SET_X")
    assert loaded["invalid"] == 1 and loaded["external_items"] == 0
    built = tki.build_items(channel_override=loaded["facts"])
    assert tki.validate_items(built) == []


def test_missing_channel_file_degrades_not_raises(tmp_path, items):
    missing = str(tmp_path / "does_not_exist.json")
    loaded = tki.load_channel_evidence(items, path=missing, env="NEVER_SET_X")
    assert loaded["facts"] == {} and loaded["error"] is not None


def test_validate_only_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "MANIFEST_PATH", str(tmp_path / "manifest.json"))
    monkeypatch.delenv(tki.CHANNEL_ENV, raising=False)
    monkeypatch.delenv("TRAK_MAX_PER_POOL", raising=False)
    assert mt.main(["--validate-only"]) == 0
    data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert data["manifest_sha256"] == MANIFEST
    assert data["bank_sha256"] == BANK_SHA
    assert data["thresholds"] == TH_SNAPSHOT
    assert data["channel"]["fallback_items"] == 400


def test_fake_main_is_additive_and_never_capability(tmp_path, monkeypatch):
    data_path = tmp_path / "rows.jsonl"
    monkeypatch.setattr(mt, "DATA_PATH", str(data_path))
    monkeypatch.setattr(mt, "SUMMARY_PATH", str(tmp_path / "summary.json"))
    monkeypatch.setattr(mt, "GATE_PATH", str(tmp_path / "gates.jsonl"))
    monkeypatch.delenv(tki.CHANNEL_ENV, raising=False)
    monkeypatch.delenv("TRAK_MAX_PER_POOL", raising=False)
    keys = (mt.SLOT_ENV_FLAG, mt.TRAK_ENV_FLAG)
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        assert mt.main([]) == 1               # fake arm: no capability claim
        first = len(data_path.read_text(encoding="utf-8").splitlines())
        assert mt.main([]) == 1
        second = len(data_path.read_text(encoding="utf-8").splitlines())
        assert second == 2 * first and first > 0
        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert summary["interpretation"] == "mechanism_probe_only_fake_arm"
        assert summary["pass"] is False
        assert summary["backend"] == "fake" and summary["live"] is False
        assert summary["invalid_controls"] == []
        assert summary["channel_independence"]["ok"] is True
        assert all(k not in os.environ for k in keys)
    finally:
        for k in keys:
            if old[k] is not None:
                os.environ[k] = old[k]


def test_focused_subset_env_is_k_stratified_and_non_final(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "DATA_PATH", str(tmp_path / "rows.jsonl"))
    monkeypatch.setattr(mt, "SUMMARY_PATH", str(tmp_path / "summary.json"))
    monkeypatch.setattr(mt, "GATE_PATH", str(tmp_path / "gates.jsonl"))
    monkeypatch.setenv("TRAK_MAX_PER_POOL", "20")
    monkeypatch.delenv(tki.CHANNEL_ENV, raising=False)
    assert mt.main([]) == 1
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["families"] == 40
    assert summary["summary"]["focused_subset"]["full_bank_verdict"] is False
    assert summary["invalid_controls"] == []
    assert summary["summary"]["primary"]["growth"] == 1.0


def test_real_backend_is_lazy_and_flag_gated():
    old_flag = os.environ.get(mt.SLOT_ENV_FLAG)
    obj = mt.RealBackend(data_dir=os.path.join(os.devnull, "unused"))
    assert obj.engine_graded is True and obj.label == "live"
    assert obj._engine is None and obj.api_calls["boot"] == 0
    assert mt.SLOT_ENV_FLAG == "RAVANA_TRA_SLOTS"
    assert mt.TRAK_ENV_FLAG == "RAVANA_TRA_K_SLOTS"
    assert os.environ.get(mt.SLOT_ENV_FLAG) != "1" or old_flag == "1"


def test_simulated_live_low_endpoints_is_scored_fail(tmp_path, monkeypatch):
    class _SimulatedLiveBackend(mt.FakeBackend):
        label = "live"
        engine_graded = True

        def __init__(self, data_dir=None):
            super().__init__()
            self.api_calls = {"boot": 1, "intake": 1, "save": 1,
                              "cold_reload": 1, "process_turn": 1}

        def _response(self, item, condition, arm, k, permuted=False,
                      phase="query"):
            return "Perhaps we could discuss something else."

    monkeypatch.setattr(mt, "RealBackend", _SimulatedLiveBackend)
    monkeypatch.setattr(mt, "DATA_PATH", str(tmp_path / "rows.jsonl"))
    monkeypatch.setattr(mt, "SUMMARY_PATH", str(tmp_path / "summary.json"))
    monkeypatch.setattr(mt, "GATE_PATH", str(tmp_path / "gates.jsonl"))
    monkeypatch.delenv(tki.CHANNEL_ENV, raising=False)
    monkeypatch.delenv("TRAK_MAX_PER_POOL", raising=False)
    keys = (mt.SLOT_ENV_FLAG, mt.TRAK_ENV_FLAG)
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        assert mt.main(["--live"]) == 1            # scored FAIL, never exit 2
        assert all(k not in os.environ for k in keys)
        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert summary["backend"] == "live" and summary["live"] is True
        assert summary["invalid_controls"] == []
        assert summary["pass"] is False
        assert summary["interpretation"] == "genuine_fail"
        by_id = {c["id"]: c for c in summary["instrument_controls"]}
        assert set(by_id) == ALL_CONTROL_IDS
        assert all(c["ok"] for c in by_id.values())
        assert summary["rules"]["c11_anti_fake_ok"] is True
        assert summary["rules"]["primary_near>=0.75"] is False
    finally:
        for k in keys:
            if old[k] is not None:
                os.environ[k] = old[k]


# ── provenance seam (fake-only, no engine boot) ─────────────────────────────
def test_taught_slot_pairs_never_name_query_subject_or_gold(items):
    it = next(x for x in items if x["k"] == 8)
    surfaces = mt.teaching_surfaces(it, 8, "near", mt.CHANNEL_ARM)
    pairs = mt.taught_slot_pairs(surfaces, it["type_label"])
    assert pairs and all(tgt == it["type_label"] for _, tgt in pairs)
    q = mt.query_for(it, "near")
    assert (q["subject"], q["gold"]) not in pairs
    assert mt.taught_slot_pairs([], it["type_label"]) == set()


def test_scoped_views_drop_boot_seed_web_and_other_item_edges(items):
    from types import SimpleNamespace
    it = next(x for x in items if x["k"] == 8)
    pairs = mt.taught_slot_pairs(
        mt.teaching_surfaces(it, 8, "near", mt.CHANNEL_ARM), it["type_label"])
    src, tgt = sorted(pairs)[0]

    def view(**kw):
        base = dict(source=src, target=tgt, type_label=it["type_label"],
                    relation_type="is_a", edge_kind="curated",
                    source_type="user", edge_source="conversation",
                    confidence=0.9, url="", fabricated_span=False)
        base.update(kw)
        return SimpleNamespace(**base)

    census = [view(),
              view(source="boot_seed", source_type="seed", edge_source="pmi"),
              view(source="other_demo"),
              view(edge_source="web", edge_kind="web_fact",
                   source_type="web", url="https://example.test/x")]
    scoped = mt.scope_taught_views(census, pairs)
    assert len(scoped) == 1
    assert scoped[0]["subject"] == src and scoped[0]["target"] == tgt
    assert mt.scope_taught_views([], pairs) == []
    assert mt.scope_taught_views(census, set()) == []


def test_real_backend_refreshes_scoped_views_at_teach_and_query(tmp_path, items):
    from types import SimpleNamespace
    it = next(x for x in items if x["k"] == 8)
    surfaces = mt.teaching_surfaces(it, 8, "near", mt.CHANNEL_ARM)
    pairs = mt.taught_slot_pairs(surfaces, it["type_label"])
    src, tgt = sorted(pairs)[0]
    taught = SimpleNamespace(
        relation_type="is_a", confidence=0.9,
        source_metadata={"source": "conversation", "source_type": "user",
                         "edge_kind": "curated", "type_label": it["type_label"]})
    seeded = SimpleNamespace(
        relation_type="is_a", confidence=0.9,
        source_metadata={"source": "pmi", "source_type": "seed",
                         "edge_kind": "curated", "type_label": it["type_label"]})
    node = lambda label: SimpleNamespace(label=label)

    class StubEngine:
        def __init__(self, graph):
            self.graph = graph

        def process_turn(self, text):
            return "acknowledged"

    def backend_with(edges, nodes, suffix):
        obj = mt.RealBackend(data_dir=str(tmp_path / suffix))
        obj.reset_for_item(it)
        obj._engine = StubEngine(SimpleNamespace(nodes=nodes, edges=edges))
        return obj

    obj = backend_with({(1, 2): taught, (3, 2): seeded},
                       {1: node(src), 2: node(tgt), 3: node("boot_seed")}, "live")
    obj.teach(it, 8, "near", mt.CHANNEL_ARM, surfaces)
    views = obj.provenance_views()
    assert len(views) == 1 and views[0]["subject"] == src
    assert mt._provenance_ok(views) is True
    obj.query(it, "near", 8)
    assert obj.provenance_views() == views
    assert obj.api_calls["boot"] == 0

    empty = backend_with({(3, 2): seeded},
                         {2: node(tgt), 3: node("boot_seed")}, "no_taught_edge")
    empty.teach(it, 8, "near", mt.CHANNEL_ARM, surfaces)
    assert empty.provenance_views() == []
    assert mt._provenance_ok(empty.provenance_views()) is False


def test_provenance_views_accumulate_across_items_without_boot():
    b = mt.RealBackend(data_dir=os.path.join(mt.SCRATCH, "acc_test"))
    b.reset_for_item({"item_id": "a"})
    b.provenance_views_cache = [{"subject": "s1", "target": "effect"}]
    b._all_provenance_views.extend(b.provenance_views_cache)
    b.reset_for_item({"item_id": "b"})
    assert [v["subject"] for v in b.provenance_views()] == ["s1"]
