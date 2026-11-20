"""Unit tests for the RAVANA-LEARN-5 CLI instrument.

These validate the *instrument's* deterministic logic (hash split, corpus
audit, replay-ablation control, matched replay budget, lexical-copy detector,
fixed thresholds) without booting the chat engine. The engine-backed
measurement lives in scripts/measure_learn5_cli.py; its output is
scratch/learn5_cli_measured.json and docs/LEARN5_CLI_MEASURED.md.
"""
from __future__ import annotations

import hashlib
import os
import sys

import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (PROJ, os.path.join(PROJ, "scripts"), os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana_ml", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import measure_learn5_cli as cli


def test_bank_sizes_unique_subjects_and_disjoint_vocabularies():
    a = cli._build(cli._A_CANDIDATES, "A")
    b = cli._build(cli._B_CANDIDATES, "B")
    t = cli._build(cli._T_CANDIDATES, "T")
    assert (len(a), len(b), len(t)) == (40, 36, 10)
    assert len({i["x"] for i in a}) == 40
    assert len({i["x"] for i in b}) == 36
    assert len({i["x"] for i in t}) == 10
    audit = cli.audit_corpus(a, b, t)
    assert audit["a_b_token_overlap"] == []
    assert audit["a_t_token_overlap"] == []
    assert audit["b_t_token_overlap"] == []
    assert audit["a_subjects_in_b_corpus"] == []
    assert audit["pass"] is True


def test_hash_rule_is_explicit_seed_independent():
    item = {"role": "A", "difficulty": "easy", "item_id": "A_000"}
    expect = hashlib.sha256(b"RAVANA-LEARN-5|CLI|A|easy|A_000").hexdigest()
    assert cli.item_hash(item) == expect
    assert cli.item_hash(dict(item)) == expect


def test_hash_split_disjoint_exhaustive_and_deterministic():
    a = cli.assign_difficulty(cli._build(cli._A_CANDIDATES, "A"),
                              [("easy", 20), ("medium", 10), ("hard", 10)])
    p1, s1 = cli.assign_pools(a)
    p2, s2 = cli.assign_pools(a)
    assert [i["item_id"] for i in p1] == [i["item_id"] for i in p2]
    assert [i["item_id"] for i in s1] == [i["item_id"] for i in s2]
    assert not ({i["item_id"] for i in p1} & {i["item_id"] for i in s1})
    assert len(p1) == 20 and len(s1) == 20
    assert {i["item_id"] for i in p1 + s1} == {i["item_id"] for i in a}


def test_canonical_hash_is_order_invariant_on_keys():
    assert cli.canonical_hash([{"b": 2, "a": 1}]) == cli.canonical_hash([{"a": 1, "b": 2}])
    assert cli.canonical_hash([{"a": 1, "b": 2}]) != cli.canonical_hash([{"a": 1, "b": 9}])


def test_replay_budget_is_matched_and_pad_is_noop():
    real = [(1, 0, 2, 1.0), (3, 0, 4, 1.0)]
    on = cli.pad_replay(real, cli.REPLAY_STEPS)
    off = cli.pad_replay([], cli.REPLAY_STEPS)
    assert len(on) == len(off) == cli.REPLAY_STEPS
    # Padded entries use out-of-range ids so run_cycle iterates but strengthens nothing.
    assert all(t[0] >= 10 ** 9 for t in off)
    assert all(t[0] >= 10 ** 9 for t in on[2:])


def test_replay_strengthening_control_real_vs_noop():
    ctl = cli.replay_strengthening_control()
    assert ctl["real_strengthened"] is True
    assert ctl["noop_unchanged"] is True
    assert ctl["pass"] is True


def test_lexical_copy_detector_fires_on_taught_token_only():
    taught = {"fog", "chill", "condensation", "steel", "forging"}
    assert cli.transfer_lex_copy("fog occurs because chill grows.", taught) == ["chill", "fog"]
    assert cli.transfer_lex_copy("violin occurs because bowing grows.", taught) == []


def test_compute_parity_bounds_are_fixed():
    assert 0 < cli.COMPUTE_RATIO_LO < 1 < cli.COMPUTE_RATIO_HI
    # Pre-registered +/-10% band (restored from the reverted +/-30% band).
    assert cli.COMPUTE_RATIO_LO == 0.90
    assert cli.COMPUTE_RATIO_HI == 1.10
    assert abs(cli.COMPUTE_RATIO_LO * cli.COMPUTE_RATIO_HI - 0.99) < 1e-9


def test_thresholds_are_not_tuned_down():
    assert cli.PASS_RET_FRACTION == 0.85
    assert cli.PASS_ON_MINUS_OFF == 0.15
    assert cli.PASS_B_DEGRADE == 0.10
    assert cli.PASS_TRANSFER_STRUCT == 0.60
    assert cli.PASS_LEX_COPY == 0.10
    assert cli.FAIL_RET_FLOOR == 0.70
    assert cli.FAIL_ON_MINUS_OFF == 0.10
    assert cli.FAIL_LEX_COPY == 0.15
    assert cli.FAIL_B_DEGRADE == 0.15
    src = open(os.path.join(PROJ, "scripts", "measure_learn5_cli.py"),
               encoding="utf-8").read()
    assert "PASS_ON_MINUS_OFF = 0.15" in src
    assert "FAIL_ON_MINUS_OFF = 0.10" in src


def test_decide_flags_failed_on_minus_off_and_high_copy():
    ctrl = {"pass": True, "audit": {"pass": True}}
    base = {"pool": "PRIMARY", "criterion_rate": 1.0, "ret_on": 1.0,
            "on_minus_off": 0.0, "b_degrade": 0.0, "transfer_struct": 1.0,
            "lexical_copy": 0.0,
            "compute": {"on_secs": 0.1, "off_secs": 0.1,
                        "on_steps": 10, "off_steps": 10,
                        "on_cycles": 3, "off_cycles": 3}}
    ok, crit = cli.decide([base], ctrl, ["PRIMARY"])
    assert ok is False
    hits = [c for c in crit if "on-minus-off" in c["criterion"]]
    assert hits and not all(c["pass"] for c in hits)
    good = dict(base, on_minus_off=0.20, ret_off=0.80)
    good["compute"] = dict(base["compute"])
    ok2, _ = cli.decide([good], ctrl, ["PRIMARY"])
    assert ok2 is True


def test_transfer_fresh_baseline_saturation_is_flagged():
    # A transfer front where the fresh engine already scores 1.0 is an
    # invalid progress signal (G6) even though the 0.60 floor is met.
    assert cli.FAIL_LEX_COPY > cli.PASS_LEX_COPY
    src = open(os.path.join(PROJ, "scripts", "measure_learn5_cli.py"),
               encoding="utf-8").read()
    assert "transfer_saturated_invalid_g6" in src


def test_no_eng_save_and_isolated_data_dir_in_script():
    src = open(os.path.join(PROJ, "scripts", "measure_learn5_cli.py"),
               encoding="utf-8").read()
    assert "eng.save()" not in src
    assert "eng_save_called\": False" in src
    assert "os.path.join(SCRATCH, \"learn5_cli\")" in src


# ── P1: snapshot-maintenance + metadata-overlay commit ──────────────────────
# These validate the restoration mechanism on a synthetic graph WITHOUT booting
# the chat engine: earned write-time provenance restores an assertable chain;
# the wrong-tag control restores nothing the gate accepts; no unearned kind is
# ever forged; and the overlay touches an arm-independent number of keys.

def _taught_graph():
    import numpy as np
    from ravana_ml.graph import ConceptGraph
    g = ConceptGraph(dim=8)
    ids = {}
    for lab in ("chill", "condensation", "fog"):
        v = np.random.RandomState(abs(hash(lab)) % 1000).randn(8).astype("float32")
        v /= (np.linalg.norm(v) + 1e-9)
        n = g.add_node(vector=v, label=lab)
        ids[lab] = n.id
    for s, t in (("chill", "condensation"), ("condensation", "fog")):
        e = g.add_edge(ids[s], ids[t], weight=0.8, relation_type="causal",
                       confidence=0.9)
        e.source_metadata = {"edge_kind": "curated", "source": "conversation:1",
                             "acquired_by": "user_assertion"}
    return g, ids


def _assertable_from_x(g, x_id):
    from ravana.learn.explanation import enumerate_chains, assertable_chain
    chains = enumerate_chains([x_id], g, direction="backward",
                              min_steps=2, max_steps=4, limit=64)
    return any(assertable_chain(c) for c in chains)


def test_capture_replay_memory_records_write_time_provenance():
    import types
    g, ids = _taught_graph()
    eng = types.SimpleNamespace(graph=g)
    items = [{"a": "chill", "b": "condensation", "x": "fog"}]
    mem = cli.capture_replay_memory(items, eng)
    assert (ids["chill"], ids["condensation"]) in mem
    assert (ids["condensation"], ids["fog"]) in mem
    rec = mem[(ids["chill"], ids["condensation"])]
    assert rec["source_metadata"]["edge_kind"] == "curated"
    assert rec["source_metadata"]["acquired_by"] == "user_assertion"
    assert rec["relation_type"] == "causal"


def test_overlay_earned_restores_chain_wrong_tag_does_not():
    import copy
    import types
    g, ids = _taught_graph()
    eng = types.SimpleNamespace(graph=g)
    items = [{"a": "chill", "b": "condensation", "x": "fog"}]
    mem = cli.capture_replay_memory(items, eng)
    assert _assertable_from_x(g, ids["fog"])  # intact before damage

    # REM-style damage: flip one hop non-causal, delete the other.
    g.get_edge(ids["chill"], ids["condensation"]).relation_type = "semantic"
    g.remove_edge(ids["condensation"], ids["fog"])
    assert not _assertable_from_x(g, ids["fog"])

    # ON: earned overlay restores both hops -> assertable again.
    g_on = copy.deepcopy(g)
    n_on = cli.apply_replay_overlay(g_on, mem, replay_on=True)
    assert n_on == 2
    assert _assertable_from_x(g_on, ids["fog"])

    # OFF: identical operations, wrong tag -> gate still refuses.
    g_off = copy.deepcopy(g)
    n_off = cli.apply_replay_overlay(g_off, mem, replay_on=False)
    assert n_off == n_on                       # arm-independent work
    assert not _assertable_from_x(g_off, ids["fog"])
    assert g_off.get_edge(ids["chill"], ids["condensation"]).source_metadata["edge_kind"] == "inferred"


def test_overlay_never_forges_an_unearned_kind():
    import types
    g, ids = _taught_graph()
    # A record whose kind the gate does NOT accept must not be stamped trusted.
    mem = {(ids["chill"], ids["condensation"]): {
        "source_metadata": {"edge_kind": "co_occurrence", "source": "web"},
        "relation_type": "causal", "weight": 0.8, "confidence": 0.9}}
    g.get_edge(ids["chill"], ids["condensation"]).source_metadata = {"edge_kind": "unknown"}
    cli.apply_replay_overlay(g, mem, replay_on=True)
    # co_occurrence is not earned -> the live edge keeps its (untrusted) kind.
    assert g.get_edge(ids["chill"], ids["condensation"]).source_metadata["edge_kind"] == "unknown"


def test_overlay_recreates_deleted_edge_only_from_its_record():
    import copy
    import types
    g, ids = _taught_graph()
    eng = types.SimpleNamespace(graph=g)
    mem = cli.capture_replay_memory([{"a": "chill", "b": "condensation", "x": "fog"}], eng)
    g.remove_edge(ids["chill"], ids["condensation"])
    g.remove_edge(ids["condensation"], ids["fog"])
    assert g.get_edge(ids["chill"], ids["condensation"]) is None
    cli.apply_replay_overlay(g, mem, replay_on=True)
    e = g.get_edge(ids["chill"], ids["condensation"])
    assert e is not None and e.relation_type == "causal"
    assert e.source_metadata["edge_kind"] == "curated"
    assert _assertable_from_x(g, ids["fog"])
