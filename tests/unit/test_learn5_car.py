"""Unit tests for the RAVANA-LEARN-5 CAR instrument.

These validate the *instrument's* deterministic logic (hash split, stats,
provenance-gate census, paraphrase hygiene) without booting the engine. The
engine-backed measurement lives in scripts/measure_learn5_car.py and its
measured output in scratch/learn5_car_measured.json + docs/LEARN5_CAR_MEASURED.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import measure_learn5_car as car


def test_bank_sizes_and_unique_subjects():
    bank = car._build_bank(car._CANDIDATES, "subject", 100)
    defbank = car._build_bank(car._DEF_CANDIDATES, "defonly", 20)
    assert len(bank) == 100 and len(defbank) == 20
    assert len({i["x"] for i in bank}) == 100
    assert len({i["x"] for i in defbank}) == 20


def test_hash_rule_is_explicit_and_seed_independent():
    item = {"role": "subject", "difficulty": "easy", "item_id": "subject_000"}
    expect = hashlib.sha256(
        b"RAVANA-LEARN-5|CAR|subject|easy|subject_000").hexdigest()
    assert car.item_hash(item) == expect
    # Same key -> same hash regardless of process hash seed (explicit string).
    assert car.item_hash(dict(item)) == expect


def test_hash_split_disjoint_exhaustive_and_deterministic():
    bank = car.assign_difficulty(car._build_bank(car._CANDIDATES, "subject", 100),
                                 [("easy", 34), ("medium", 34), ("hard", 32)])
    p1, s1 = car.assign_pools(bank)
    p2, s2 = car.assign_pools(bank)
    assert [i["item_id"] for i in p1] == [i["item_id"] for i in p2]
    assert [i["item_id"] for i in s1] == [i["item_id"] for i in s2]
    assert not ({i["item_id"] for i in p1} & {i["item_id"] for i in s1})
    assert len(p1) == 50 and len(s1) == 50
    assert {i["item_id"] for i in p1 + s1} == {i["item_id"] for i in bank}


def test_manifest_hash_is_canonical():
    # sort_keys makes dict key order irrelevant; list order is meaningful
    # (a manifest is an ordered record set), so that must change the hash.
    h = car.canonical_hash([{"b": 2, "a": 1}])
    assert h == car.canonical_hash([{"a": 1, "b": 2}])
    assert h != car.canonical_hash([{"a": 1, "b": 9}])
    assert h != car.canonical_hash([{"a": 1, "b": 2}, {"a": 3}])


def test_latin_square_gives_each_item_exactly_one_balanced_k():
    bank = car.assign_difficulty(car._build_bank(car._CANDIDATES, "subject", 100),
                                 [("easy", 34), ("medium", 34), ("hard", 32)])
    pool, _ = car.assign_pools(bank)
    car.assign_k(pool)
    counts = {k: 0 for k in car.K_LEVELS}
    seen = set()
    for it in pool:
        assert it["item_id"] not in seen
        seen.add(it["item_id"])
        assert it["k"] in car.K_LEVELS
        counts[it["k"]] += 1
    assert set(counts.values()) == {10}


def test_paraphrases_share_no_taught_mechanism_words():
    # The held-out probe must not contain the taught a/b tokens: that is the
    # parroting control. Only the subject X may be shared (it is the topic).
    for q in car.PARAPHRASES:
        assert "{x}" in q
        assert "{a}" not in q and "{b}" not in q


def test_wilson_bounds_and_degeneracy():
    lo, hi = car.wilson(8, 10)
    assert 0 <= lo <= 0.8 <= hi <= 1
    assert car.wilson(0, 10) == (0.0, car.wilson(0, 10)[1])
    assert car.wilson(5, 0) == (0.0, 0.0)


def test_fisher_p_sanity():
    p_same = car.fisher_p(8, 2, 2, 8)   # strong association
    p_null = car.fisher_p(5, 5, 5, 5)   # no association
    assert 0 <= p_same < 0.05
    assert p_null > 0.9


def test_gate_census_rejects_untrusted_accepts_trusted():
    from ravana_ml.graph import ConceptGraph
    g = ConceptGraph()
    a = g.add_node(label="zza")
    b = g.add_node(label="zzb")
    x = g.add_node(label="zzx")
    for s, t in ((a.id, b.id), (b.id, x.id)):
        e = g.add_edge(source=s, target=t, weight=0.8,
                       relation_type="causal", confidence=0.9)
        e.source_metadata.update({"edge_kind": "web_fact"})
    trusted = car.gate_census([x.id], g)
    assert trusted["raw_chains"] >= 1
    assert trusted["trusted_all_hops"] >= 1
    assert trusted["rejected_by_trust"] == 0
    # Remove provenance: every hop becomes untrusted, chain must be rejected.
    g2 = ConceptGraph()
    a2 = g2.add_node(label="zza")
    b2 = g2.add_node(label="zzb")
    x2 = g2.add_node(label="zzx")
    for s, t in ((a2.id, b2.id), (b2.id, x2.id)):
        g2.add_edge(source=s, target=t, weight=0.8,
                    relation_type="causal", confidence=0.9)
    untrusted = car.gate_census([x2.id], g2)
    assert untrusted["trusted_all_hops"] == 0
    assert untrusted["rejected_by_trust"] >= 1


def test_defonly_subjects_disjoint_from_teach_subjects():
    # A shared subject is a shared graph node: the "control" would inherit the
    # taught chain. This was the confound found in the invalid pilot run.
    teach = {x for x, _, _ in car._CANDIDATES}
    defs = {x for x, _, _ in car._DEF_CANDIDATES}
    assert not (teach & defs), teach & defs


def test_traversal_limit_documented_and_gate_unchanged():
    assert car.TRAVERSAL_LIMIT == 64
    src = open(os.path.join(PROJ, "scripts", "measure_learn5_car.py"),
               encoding="utf-8").read()
    assert "grade_chain(c)" in src and "assertable_chain(c)" in src


def test_trusted_chain_detected_finds_chain_below_decoys():
    from ravana_ml.graph import ConceptGraph
    g = ConceptGraph()
    a = g.add_node(label="zza"); b = g.add_node(label="zzb"); x = g.add_node(label="zzx")
    for s, t in ((a.id, b.id), (b.id, x.id)):
        e = g.add_edge(source=s, target=t, weight=0.4,
                       relation_type="causal", confidence=0.5)
        e.source_metadata.update({"edge_kind": "curated"})
    # Add many high-weight UNTRUSTED chains that outrank the trusted one.
    for i in range(10):
        p = g.add_node(label="dec%d" % i); q = g.add_node(label="dc%d" % i)
        for s, t in ((x.id, p.id), (p.id, q.id)):
            g.add_edge(source=s, target=t, weight=0.99,
                       relation_type="causal", confidence=1.0)
    c, gr = car.trusted_chain_detected([x.id], g)
    assert c is not None and gr["verdict"] == "assert"
    # top-8 would miss it; the larger budget must find it
    from ravana.learn.explanation import enumerate_chains
    top8 = enumerate_chains([x.id], g, direction="backward",
                            min_steps=2, max_steps=4, limit=8)
    assert all(not (len(cc.hops) == 2 and
                    {h.from_label for h in cc.hops} >= {"zza", "zzb"})
               for cc in top8)


def test_criteria_are_not_tuned_down(capsys=None):
    # The fixed pass thresholds from BRAIN_LEARN5_DESIGN.md section 3.6.
    assert car.K_LEVELS == (0, 1, 2, 4, 8)
    assert 0.25 > 0 and 0.10 > 0 and 0.05 > 0
    src = open(os.path.join(PROJ, "scripts", "measure_learn5_car.py"),
               encoding="utf-8").read()
    assert "d8 >= 0.25" in src
    assert 'max_drop >= -0.10' in src
    assert 'd0 <= 0.05' in src
