"""Unit tests for the default-off subject-side type evidence channel.

Static only: no engine, no process_turn, no store, no network, no benchmark
score. Run with PYTHONPATH pointing at this worktree's ravana/src.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ravana.learn import subject_type_evidence as ste

FORBIDDEN = {
    "gold", "type_sets", "world_model", "typed_queries", "channel_evidence",
    "item_id", "role", "type_label", "unseen_test_words", "domain",
    "far_domain", "process_turn", "CognitiveChatEngine",
}
FORBIDDEN_MODULES = {
    "tra_k_items", "tra_items", "front_gates", "measure_tra", "measure_tra_k",
    "ravana",
}


@pytest.fixture()
def snap():
    return ste.load_snapshot()


@pytest.fixture()
def on(monkeypatch):
    monkeypatch.setenv(ste.ENV_FLAG, "1")
    return True


def test_default_off(monkeypatch, snap):
    monkeypatch.delenv(ste.ENV_FLAG, raising=False)
    a, _ = ste.evidence_for_subject("arc", snap)
    b, _ = ste.evidence_for_subject("arc", snap)
    assert a == () and b == ()
    assert ste.enabled() is False


def test_evidence_is_for_the_actual_subject_with_provenance(on, snap):
    ev, rej = ste.evidence_for_subject("arc", snap)
    assert ev and not rej
    assert all(e.subject == "arc" for e in ev)
    for e in ev:
        assert e.type_name and e.assertion_text
        assert e.license in ste.LICENSE_CREDIBILITY
        assert e.license_url and e.source_url and e.snapshot_sha256
        assert e.credibility >= ste.CREDIBILITY_FLOOR
    other, _ = ste.evidence_for_subject("cone", snap)
    assert ste.signature(other) != ste.signature(ev)


@pytest.mark.parametrize("surface", [
    "arc is not a curve.",
    "arc is never a curve.",
    "arc isn't a curve.",
    "arc is no curve.",
    "arc cannot be a curve.",
    "arc is not classified as a curve.",
    "it is not the case that arc is a curve.",
])
def test_negated_assertions_rejected(on, surface):
    a = ste.Assertion("arc", "is_a", "curve", surface, "wikidata",
                      "CC0-1.0", credibility=0.7)
    ev, rej = ste.admit("arc", [a])
    assert ev == ()
    assert any(r.reason == "negated" for r in rej)


def test_contradiction_rejected_by_higher_credibility_negation(on):
    positive = ste.Assertion("arc", "is_a", "curve", "arc is a curve.",
                             "wikidata", "CC0-1.0", credibility=0.7)
    negative = ste.Assertion("arc", "is_a", "curve", "arc is never a curve.",
                             "wordnet", "WordNet-3.0", credibility=0.9)
    ev, rej = ste.admit("arc", [positive, negative])
    assert ev == ()
    assert {r.reason for r in rej} == {"contradicted", "negated"}


@pytest.mark.parametrize("assertion,reason", [
    (ste.Assertion("arc", "is_a", "curve", "arc is a curve.", "wikidata",
                   "CC0-1.0", credibility=0.1), "low_credibility"),
    (ste.Assertion("arc", "is_a", "curve", "arc is a curve.", "unknown",
                   "", credibility=0.9), "unlicensed"),
    (ste.Assertion("arc", "is_a", "curve", "", "wikidata", "CC0-1.0"),
     "empty_assertion"),
    (ste.Assertion("elsewhere", "is_a", "curve", "elsewhere is a curve.",
                   "wikidata", "CC0-1.0"), "wrong_subject"),
    (ste.Assertion("arc", "causes", "curve", "arc causes curve.", "wikidata",
                   "CC0-1.0"), "bad_relation"),
])
def test_bad_assertions_rejected(on, assertion, reason):
    ev, rej = ste.admit("arc", [assertion])
    assert ev == ()
    assert any(r.reason == reason for r in rej)


def test_positive_sources_admitted(on):
    for source, license_name, cred in (("wikidata", "CC0-1.0", 0.7),
                                       ("wordnet", "WordNet-3.0", 0.9)):
        a = ste.Assertion("arc", "is_a", "curve", "arc is a curve.",
                          source, license_name, credibility=cred)
        ev, rej = ste.admit("arc", [a])
        assert len(ev) == 1 and not rej
        assert ev[0].credibility == cred


def test_declared_credibility_cannot_inflate(on):
    a = ste.Assertion("arc", "is_a", "curve", "arc is a curve.", "wikidata",
                      "CC0-1.0", credibility=99.0)
    ev, _ = ste.admit("arc", [a])
    assert ev[0].credibility == 0.7


def test_candidate_permutation_is_a_noop(on, snap):
    a, _ = ste.evidence_for_subject("arc", snap)
    b, _ = ste.evidence_for_subject("arc", snap)
    assert ("a", "b") != ("b", "a")
    assert ste.signature(a) == ste.signature(b)
    assert "candidates" not in __import__("inspect").signature(
        ste.evidence_for_subject).parameters


def test_ast_anti_leak():
    tree = ast.parse(Path(ste.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] not in FORBIDDEN_MODULES
                       for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in FORBIDDEN_MODULES
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                ident = None
                if isinstance(sub, ast.Name):
                    ident = sub.id
                elif isinstance(sub, ast.arg):
                    ident = sub.arg
                elif isinstance(sub, ast.Attribute):
                    ident = sub.attr
                elif isinstance(sub, ast.keyword):
                    ident = sub.arg
                assert ident not in FORBIDDEN


def test_snapshot_license_metadata(snap):
    assert snap["schema"] == "ravana-m2-subject-type-evidence-v1"
    prov = snap["provenance"]
    assert prov["sources"][0]["license"] == "CC0-1.0"
    assert len(prov["facts_sha256"]) == 64
    assert prov["upstream_schema"] == "ravana-trak-semantic-substrate-v1"
    assert snap["facts"]
