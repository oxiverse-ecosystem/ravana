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
import hashlib
import json
import os
import subprocess
import sys

import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKTREE_SCRIPTS = os.path.join(PROJ, "scripts")
WORKTREE_SCRATCH = os.path.join(PROJ, "scratch")
for _p in (PROJ, WORKTREE_SCRATCH, WORKTREE_SCRIPTS, os.path.join(PROJ, "ravana", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import measure_tra_k as mt                    # noqa: E402
import tra_k_items as tki                     # noqa: E402
import trak_channel_from_substrate as adapter   # noqa: E402


# Frozen pre-registration snapshots (spec / TH in the harness).
MANIFEST = {
    # Fallback-channel bank. The option-space fields are part of the canonical
    # sealed records; the pre-registered BANK_SHA (item_id/difficulty/pool/role/
    # k/hash) is deliberately UNCHANGED by the option-space fix.
    # These are the hashes of the committed (LF-blob) substrate snapshot.
    # substrate_sha256() normalises CRLF worktree bytes to LF before hashing,
    # so Windows and Linux checkouts now freeze to these same values.
    "primary": "05439e9fb9208b78af934575864af16656273a363c13b5c787685b952132775e",
    "sealed": "32facbe95e7f558967f679f89851b3797119d82438daf1132d1c8fd1390558be",
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


def test_substrate_digest_and_manifest_ignore_checkout_line_endings(tmp_path):
    """CRLF worktrees and LF CI checkouts must freeze to the same pin."""
    with open(tki.SUBSTRATE_PATH, "rb") as fh:
        raw = fh.read()
    lf = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    lf_path = tmp_path / "substrate_lf.json"
    crlf_path = tmp_path / "substrate_crlf.json"
    lf_path.write_bytes(lf)
    crlf_path.write_bytes(lf.replace(b"\n", b"\r\n"))
    sub_lf = tki.load_substrate(str(lf_path))
    sub_crlf = tki.load_substrate(str(crlf_path))
    assert sub_lf["digest"] == sub_crlf["digest"] == tki.load_substrate()["digest"]
    bank_lf = tki.build_items(substrate=sub_lf)
    bank_crlf = tki.build_items(substrate=sub_crlf)
    assert tki.manifest_hashes(bank_lf) == MANIFEST
    assert tki.manifest_hashes(bank_crlf) == MANIFEST


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
    """Gold/target are forbidden; the QUERY SUBJECT's own fact is admitted."""
    for it in items:
        forbidden = tki.forbidden_channel_subjects(it)
        subjects = {str(f["subject"]).strip().lower()
                    for f in it["channel_evidence"]}
        assert not (forbidden & subjects)
        for subj in tki.query_subject_words(it):
            fact = tki.admitted_subject_fact(it, subj)
            assert fact is not None and tki.substrate_provenance_ok(fact)
        assert any(f.get("evidence_role") == tki.MISLEADING_ROLE
                   for f in it["channel_evidence"])


def test_channel_overlap_reports_benign_overlap(items):
    rep = tki.channel_overlap(items)
    assert rep["ok"] is True and not rep["offenders"]
    tot = rep["totals"]
    # 3200 demo facts + 1200 subject-type + 1200 misleading (subjects also
    # appear in the authored type_sets, so this overlap is benign and reported).
    assert tot["facts"] == 5600 and tot["overlap_type_sets"] == 4412
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
    expected = [tki.channel_assertion(f)
                for f in tki.ordered_channel_facts(it)]
    assert surfaces == [s for s in expected if s][:max(mt.K_LEVELS)]
    assert it["unseen_test_words"]["near"]["subject"] in " ".join(surfaces)
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


def test_k0_channel_arm_is_neutral_no_channel_teaching(items):
    """k=0 is the matched neutral null for the deployable arm (documented dev)."""
    it = next(x for x in items if x["k"] == 0)
    surfaces = mt.teaching_surfaces(it, 0, "near", mt.CHANNEL_ARM)
    assert len(surfaces) == max(mt.K_LEVELS)
    assert all(" is a " not in s and " is an " not in s for s in surfaces)
    assert mt.teaching_surfaces(it, 8, "near", mt.CHANNEL_ARM)[max(mt.K_LEVELS):] == (
        mt.channel_teaching_surfaces(it))


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
            assert surface.count(q["gold"]) >= 1
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
        # k=0 is neutral (documented P2c deviation): 40/200 families per pool
        # are absent, so the pooled rate is 0.8 while k=8 stays 1.0.
        assert d["near"] == 0.8 and d["far"] == 0.8
        assert d["by_k"]["0"]["near"] == 0.0 and d["by_k"]["8"]["near"] == 1.0
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
        gold = mt.query_for(it, "near")["gold"]   # typed channel label
        assert surface.count(gold) >= 1
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
    assert any(f["source"] == "ontology-B-test-snapshot"
               for f in got["channel_evidence"])
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


# ── Wikidata substrate integration (fake-only, static) ───────────────────────
def _adapter_substrate():
    sub = adapter.load_substrate(adapter.DEFAULT_SUBSTRATE)
    sub["digest"] = tki.substrate_sha256(adapter.DEFAULT_SUBSTRATE)
    return sub


def test_adapter_emits_trak_channel_format_with_source_native_provenance():
    """Every emitted fact is a verbatim projection of a substrate record."""
    sub = _adapter_substrate()
    items = tki.build_items()
    projection = adapter.build_projection(sub, items)
    assert set(projection) == {it["item_id"] for it in items}
    role_labels = set(tki.ROLE_TO_TYPE.values())
    raw_index = {
        (f["subject"], f["type_label"], f["source_url"], f["type_qid"],
         f["wikidata_property"], f["sense"]): f for f in sub["facts"]}
    seen = 0
    for facts in projection.values():
        assert facts
        for fact in facts:
            seen += 1
            assert not (set(tki.REQUIRED_FACT_KEYS) - set(fact))
            assert fact["channel"] == "B-wikidata"
            assert fact["predicate"] == "is_a"
            assert fact["source"] == fact["source_url"]
            assert fact["license"] == "CC0-1.0"
            assert fact["object"] not in role_labels  # source-native, not TRA labels
            key = (fact["subject"], fact["object"], fact["source_url"],
                   fact["type_qid"], fact["wikidata_property"], fact["sense"])
            raw = raw_index[key]                     # no fabricated provenance
            for field in adapter.PROVENANCE_FIELDS:
                assert fact[field] == raw[field]
    assert seen == 3200


def test_adapter_output_ignores_oracle_fields():
    """Poisoning type_sets/world_model/gold cannot change the projection."""
    sub = _adapter_substrate()
    items = tki.build_items()
    base = adapter.build_projection(sub, items)
    poisoned = copy.deepcopy(items[:4])
    for it in poisoned:
        it["type_sets"] = {"POISON": ["effect", "structured-whole"]}
        it["world_model"] = {"POISON": ["gold-poison"]}
        it["gold"] = "gold-poison"
        it["unseen_test_words"] = {
            "near": {"subject": "POISON", "gold": "gold-poison"}}
    got = adapter.build_projection(sub, poisoned)
    assert got == {it["item_id"]: base[it["item_id"]] for it in poisoned}
    for facts in got.values():
        for fact in facts:
            assert "type_sets" not in fact and "world_model" not in fact
            assert "gold" not in fact and "unseen_test_words" not in fact


def test_adapter_no_answer_key_leak_and_schema_valid():
    sub = _adapter_substrate()
    items = tki.build_items()
    projection = adapter.build_projection(sub, items)
    for it in items:
        key = tki.answer_key_tokens(it)
        assert not ({f["subject"] for f in projection[it["item_id"]]} & key)
    report = adapter.validate_projection(sub, projection, items)
    assert report["ok"] is True
    assert report["validate_errors"] == 0
    assert report["overlap"]["answer_key_leak"] == 0
    assert report["overlap"]["oracle_domain_tag_leak"] == 0
    assert report["overlap"]["provenance_missing"] == 0


def test_adapter_projection_is_byte_stable():
    sub = _adapter_substrate()
    items = tki.build_items()
    a = json.dumps(adapter.build_projection(sub, items), sort_keys=True)
    b = json.dumps(adapter.build_projection(sub, items), sort_keys=True)
    assert a == b
    for facts in adapter.build_projection(sub, items).values():
        keys = [(f["subject"], f["object"], f["source_url"], f["type_qid"],
                 f["sense"]) for f in facts]
        assert keys == sorted(keys)


def test_mutation_hook_stays_connected_for_source_native_channel(items):
    """The scratch regrade moves on substrate facts whose type != role label."""
    projection = adapter.build_projection(_adapter_substrate(), items)
    it = next(x for x in items if x["pool"] == "primary" and x["k"] == 8)
    sourced = dict(it)
    sourced["channel_evidence"] = copy.deepcopy(projection[it["item_id"]])
    assert sourced["channel_evidence"][0]["object"] not in tki.ROLE_TO_TYPE.values()
    assert mt._channel_verdict(sourced) == "match"
    ma = copy.deepcopy(sourced)
    for fact in ma["channel_evidence"]:
        fact["object"] = mt.MUTATION_SENTINEL
    mb = copy.deepcopy(sourced)
    mb["channel_evidence"] = []
    assert mt._channel_verdict(ma) == "reject"
    assert mt._channel_verdict(mb) == "reject"


def test_substrate_channel_fake_main_restores_flags(tmp_path, monkeypatch):
    """Fake arm with the substrate channel: mechanism probe only, flags intact."""
    projection = adapter.build_projection(_adapter_substrate(), tki.build_items())
    channel = tmp_path / "substrate_channel.json"
    channel.write_text(json.dumps({"items": projection}), encoding="utf-8")
    monkeypatch.setattr(mt, "DATA_PATH", str(tmp_path / "rows.jsonl"))
    monkeypatch.setattr(mt, "SUMMARY_PATH", str(tmp_path / "summary.json"))
    monkeypatch.setattr(mt, "GATE_PATH", str(tmp_path / "gates.jsonl"))
    monkeypatch.setenv(tki.CHANNEL_ENV, str(channel))
    monkeypatch.delenv("TRAK_MAX_PER_POOL", raising=False)
    keys = (mt.SLOT_ENV_FLAG, mt.TRAK_ENV_FLAG)
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = "0"
        assert mt.main([]) == 1
        assert all(os.environ[k] == "0" for k in keys)
        for k in keys:
            os.environ.pop(k, None)
        assert mt.main([]) == 1
        assert all(k not in os.environ for k in keys)
        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    finally:
        for k in keys:
            if old[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old[k]
    assert summary["interpretation"] == "mechanism_probe_only_fake_arm"
    assert summary["backend"] == "fake" and summary["live"] is False
    assert summary["invalid_controls"] == []
    assert summary["channel"]["external_items"] == 400
    assert summary["channel"]["fallback_items"] == 0
    assert summary["channel"]["overlap"]["answer_key_leak"] == 0
    by_id = {c["id"]: c for c in summary["instrument_controls"]}
    assert by_id["C11"]["ok"] is False
    assert by_id["C14"]["ok"] is True and by_id["C15"]["ok"] is True
    assert summary["saturation_gate"]["status"] == "FLAG"
    assert summary["mutation_control"]["moved"] is True
    assert summary["rules"]["backend_live"] is False
    # The graded rows themselves carry source-native type labels as options.
    dom, glob = tki.collect_channel_label_pools(
        tki.build_items(channel_override=projection))
    all_labels = set().union(*dom.values()) | glob
    rows_on_disk = [json.loads(line) for line in
                    (tmp_path / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    graded = [r for r in rows_on_disk if r["phase"] == "query"
              and r["arm"] == "channel" and r["condition"] in ("near", "far")]
    assert graded
    for rec in graded[::97]:
        assert len(rec["input"]["candidates"]) == tki.N_TYPED_OPTIONS
        assert set(rec["input"]["candidates"]) <= all_labels
        assert rec["gold"]["target"] in all_labels
        assert rec["gold"]["target"] in rec["input"]["candidates"]


# ── typed answer-option space: channel type labels, not TRA domain words ─────
def test_typed_options_are_four_distinct_channel_labels(items):
    dom, glob = tki.collect_channel_label_pools(items)
    for it in items:
        pool = dom.get(it["domain"], set()) | glob
        ev_labels = {str(f["object"]).strip() for f in it["channel_evidence"]}
        domain_words = set(it["unseen_test_words"]["near"]["candidates"])
        for cond in tki.TYPED_CONDITIONS:
            q = it["typed_queries"][cond]
            cands = q["candidates"]
            assert len(cands) == tki.N_TYPED_OPTIONS, (it["item_id"], cond)
            assert len(set(cands)) == len(cands), (it["item_id"], cond)
            assert set(cands) <= pool, (it["item_id"], cond)
            # A source-native Wikidata label can coincide with an oracle
            # domain word (e.g. "contract" is both a type and a candidate).
            # That is benign: the label is selected from the SUBSTRATE, not
            # from type_sets/world_model/gold. Only OTHER domain words are
            # forbidden as foils.
            allowed = {q.get("gold"), q.get("counterexample_target")}
            assert not ((set(cands) & domain_words) - allowed), (it["item_id"], cond)
        assert it["typed_queries"]["near"]["gold"] in ev_labels
        assert it["typed_queries"]["far"]["gold"] in ev_labels
        for cond in ("foil", "negative", "unrelated"):
            assert it["typed_queries"][cond]["gold"] is None
        assert it["typed_queries"]["negative"]["counterexample_target"] in pool


def test_typed_options_with_source_native_channel_are_substrate_labels():
    sub = _adapter_substrate()
    projection = adapter.build_projection(sub, tki.build_items())
    built = tki.build_items(channel_override=projection)
    assert tki.validate_items(built) == []
    dom, glob = tki.collect_channel_label_pools(built)
    role_labels = set(tki.ROLE_TO_TYPE.values())
    native = 0
    for it in built:
        pool = dom.get(it["domain"], set()) | glob
        ev_labels = {str(f["object"]).strip() for f in it["channel_evidence"]}
        for cond in tki.TYPED_CONDITIONS:
            q = it["typed_queries"][cond]
            assert set(q["candidates"]) <= pool, (it["item_id"], cond)
            assert len(set(q["candidates"])) == tki.N_TYPED_OPTIONS
        gold = it["typed_queries"]["near"]["gold"]
        assert gold in ev_labels
        assert gold != it["unseen_test_words"]["near"]["gold"]  # no answer-key reuse
        if gold not in role_labels:
            native += 1
    assert native == 400                      # every near gold is source-native
    it0 = next(x for x in built if x["item_id"] == "tra_k_0000")
    near_subject = it0["unseen_test_words"]["near"]["subject"]
    assert it0["typed_queries"]["near"]["gold"] == str(
        tki.admitted_subject_fact(it0, near_subject)["object"]).strip()
    assert "hydrometeor" in glob


def test_typed_options_ignore_oracle_fields(items):
    dom, glob = tki.collect_channel_label_pools(items)
    it = next(x for x in items if x["pool"] == "primary" and x["k"] == 8)
    base = tki.typed_option_space(it, dom.get(it["domain"], set()), glob)
    poisoned = copy.deepcopy(it)
    poisoned["type_sets"] = {"POISON": ["poison-type"]}
    poisoned["world_model"] = {"POISON": ["poison-target"]}
    for cond in ("near", "far"):
        poisoned["unseen_test_words"][cond]["gold"] = "poison-gold"
        poisoned["unseen_test_words"][cond]["candidates"] = ["poison"] * 4
    poisoned["negative"]["counterexample_target"] = "poison-neg"
    poisoned["gold"] = "poison"
    got = tki.typed_option_space(poisoned, dom.get(it["domain"], set()), glob)
    assert got == base


def test_channel_swap_changes_typed_answer():
    sub = _adapter_substrate()
    projection = adapter.build_projection(sub, tki.build_items())
    built = tki.build_items(channel_override=projection)
    dom, glob = tki.collect_channel_label_pools(built)
    a = next(x for x in built if x["item_id"] == "tra_k_0000")
    b = next(x for x in built if x["item_id"] == "tra_k_0114")
    original = a["typed_queries"]["near"]["gold"]
    transplanted = tki.channel_answer_label(
        b["channel_evidence"], tki.channel_demo_objects(b),
        tki._digest(b["item_id"], "typed-options"))
    swapped = copy.deepcopy(a)
    # Keep A's deployable demo subjects, transplant B's channel type label.
    for fact in swapped["channel_evidence"]:
        fact["object"] = transplanted
    got = tki.typed_option_space(swapped, dom.get(a["domain"], set()), glob)
    assert got["near"]["gold"] == transplanted
    assert got["near"]["gold"] != original


def test_negative_channel_surfaces_negate_channel_labels():
    sub = _adapter_substrate()
    projection = adapter.build_projection(sub, tki.build_items())
    built = tki.build_items(channel_override=projection)
    it = next(x for x in built if x["k"] == 8)
    surfaces = mt.channel_negation_surfaces(it)
    assert len(surfaces) == max(mt.K_LEVELS)
    assert all(" is not " in s for s in surfaces)
    for fact in it["channel_evidence"][:max(mt.K_LEVELS)]:
        expected = tki.channel_negation_assertion(fact)
        if expected:
            assert expected in surfaces
    sweep = mt.teaching_surfaces(it, 8, "negative", mt.CHANNEL_ARM)
    assert sweep[-max(mt.K_LEVELS):] == surfaces


def test_external_manifest_hash_stable_and_option_aware():
    sub = _adapter_substrate()
    projection = adapter.build_projection(sub, tki.build_items())
    a = tki.build_items(channel_override=projection)
    b = tki.build_items(channel_override=projection)
    assert tki.manifest_hashes(a) == tki.manifest_hashes(b)
    assert tki.item_manifest_hash(a) == BANK_SHA   # split/identity unchanged
    assert tki.manifest_hashes(a) != MANIFEST      # option space is hashed


# ── subject-channel fix: coverage, admitted subject facts, anti-saturation ────
def test_query_subjects_are_all_substrate_covered(items):
    """No item may query a word the substrate cannot type (root cause)."""
    covered = tki.covered_subjects()
    uncovered = {"sleet", "chill", "marrow", "thorn"}
    assert not (uncovered & covered)
    for it in items:
        for subj in tki.query_subject_words(it):
            assert subj in covered, (it["item_id"], subj)
    # Uncovered words may still occur as distractors/demo words: coverage of
    # the 400-family bank and its 40/k/pool split must be unchanged.
    assert len(items) == 400
    assert all(tki.pool_k_counts(items)[(pool, k)] == 40
               for pool in ("primary", "sealed") for k in mt.K_LEVELS)


def test_each_item_carries_its_subject_type_assertion(items):
    for it in items:
        for cond, subj in (("near", it["unseen_test_words"]["near"]["subject"]),
                           ("far", it["unseen_test_words"]["far"]["subject"]),
                           ("negative", it["negative"]["subject"])):
            fact = tki.admitted_subject_fact(it, subj)
            assert fact is not None, (it["item_id"], cond)
            assert fact["subject"] == subj
            assert fact["predicate"] in tki.ISA_PREDICATES
            assert tki.substrate_provenance_ok(fact)
            assert fact["license"] == "CC0-1.0"
            assert fact["source_url"] and fact["source_id"] and fact["span"]
            if cond in ("near", "far"):
                assert it["typed_queries"][cond]["gold"] == str(
                    fact["object"]).strip()


def test_subject_fact_ignores_oracle_fields(items):
    """Poisoning type_sets/world_model/gold cannot change the subject fact."""
    base = next(it for it in items if it["k"] == 8)
    subj = base["unseen_test_words"]["near"]["subject"]
    before = tki.admitted_subject_fact(base, subj)
    poisoned = copy.deepcopy(base)
    poisoned["type_sets"] = {"POISON": ["poison-type"]}
    poisoned["world_model"] = {"POISON": ["poison-word"]}
    poisoned["unseen_test_words"]["near"]["gold"] = "poison-gold"
    poisoned["negative"]["counterexample_target"] = "poison-neg"
    assert tki.admitted_subject_fact(poisoned, subj) == before
    assert poisoned["channel_evidence"] == base["channel_evidence"]


def test_anti_saturation_conflicts_are_misleading_not_gold(items):
    """The right label is not the only type in the channel."""
    for it in items[::17]:
        labels = {str(f["object"]).strip()
                  for f in tki.ordered_channel_facts(it)}
        assert len(labels) >= 2
        misleading = [f for f in it["channel_evidence"]
                      if f.get("evidence_role") == tki.MISLEADING_ROLE]
        assert misleading
        gold = it["typed_queries"]["near"]["gold"]
        assert any(str(f["object"]).strip() != gold for f in misleading)
        assert gold in labels


def test_admission_rule_rejects_unprovenanced_subject_fact(items):
    it = copy.deepcopy(items[0])
    subj = it["unseen_test_words"]["near"]["subject"]
    bad = dict(tki.admitted_subject_fact(it, subj))
    bad.pop("license")
    it["channel_evidence"].append(bad)
    assert not tki.substrate_provenance_ok(bad)
    assert not tki.admitted_channel_fact(bad, it)


def test_multiword_typed_option_parses_explicit_target(items):
    """Source-native labels contain spaces; the parser must not truncate."""
    it = next(x for x in items if " " in x["typed_queries"]["near"]["gold"])
    q = mt.query_for(it, "near")
    ans = mt._parse_engine_answer(
        ["role=%s target=%s verdict=match" % (it["role"], q["gold"])],
        it, "near", q["subject"], q["candidates"])
    assert ans["target"] == q["gold"] and ans["verdict"] == "match"


def test_subject_facts_reach_fake_rows_and_teaching(items, rows):
    it = next(x for x in items if x["k"] == 8)
    q = mt.query_for(it, "near")
    surfaces = mt.teaching_surfaces(it, 8, "near", mt.CHANNEL_ARM)
    assertion = "%s is a %s." % (
        q["subject"], q["gold"]) if q["gold"][:1].lower() not in "aeiou" else (
        "%s is an %s." % (q["subject"], q["gold"]))
    assert assertion in surfaces
    graded = [r for r in rows if r["id"] == it["item_id"]
              and r["phase"] == "query" and r["arm"] == mt.CHANNEL_ARM
              and r["condition"] == "near"][0]
    assert any(f["type_label"] == q["gold"]
               for f in graded["input"]["subject_type_facts"])


def test_subject_channel_bank_hash_stable(tmp_path):
    sub = _adapter_substrate()
    projection = adapter.build_projection(sub, tki.build_items())
    a = tki.build_items(channel_override=projection)
    b = tki.build_items(channel_override=projection)
    assert tki.manifest_hashes(a) == tki.manifest_hashes(b)
    dump = lambda bank: json.dumps(bank, sort_keys=True, ensure_ascii=False)
    assert dump(a) == dump(b)
    assert tki.item_manifest_hash(a) == BANK_SHA
    assert tki.validate_items(a) == []
