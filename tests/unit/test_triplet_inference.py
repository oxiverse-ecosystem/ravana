"""Tests for the Triplet Inference Operator (core/triplet_inference/).

Verifies the CLS claim end-to-end: inference properties are LEARNED
per-predicate statistics — nothing fires until evidence accumulates,
and evidence volume is priced in via Wilson lower bounds (fail-closed).
"""
import pytest

from ravana.core.triplet_inference import (
    InferenceCuriosityHook, SleepSchemaExtractor, Triple,
    TripletInferenceOperator, TripletMemory, canonical_predicate,
    wilson_lower,
)


def _op():
    return TripletInferenceOperator(seed=False)


def _teach_transitive(op, pred="is", n=10, prefix=""):
    """Feed n complete transitive chains (ground truth includes A->C)."""
    for i in range(n):
        a, b, c = f"{prefix}x{i}", f"{prefix}y{i}", f"{prefix}z{i}"
        op.ingest_triple(Triple(a, pred, b))
        op.ingest_triple(Triple(b, pred, c))
        op.ingest_triple(Triple(a, pred, c))  # the closing ground truth


# ── canonicalization (the fragmentation fix) ─────────────────────────

def test_canonical_predicate_collapses_copula_variants():
    assert canonical_predicate("is") == "is"
    assert canonical_predicate("is_a") == "is"
    assert canonical_predicate("are") == "is"
    assert canonical_predicate("was an") == "is"
    assert canonical_predicate("has_a") == "has"
    assert canonical_predicate("Married_To") == "married to"


def test_evidence_does_not_fragment_across_surface_forms():
    op = _op()
    op.ingest_triple(Triple("cat", "is_a", "mammal"))
    op.ingest_triple(Triple("dogs", "are", "mammal"))
    # Both land on the SAME canonical profile.
    assert "is" in op.memory.profiles
    assert "is_a" not in op.memory.profiles
    assert "are" not in op.memory.profiles


# ── wilson bound sanity ──────────────────────────────────────────────

def test_wilson_lower_fails_closed_on_small_n():
    assert wilson_lower(1, 1) < 0.5      # one perfect example: not enough
    assert wilson_lower(2, 2) < 0.5      # two: still not enough
    assert wilson_lower(10, 10) > 0.5    # ten consistent: fires
    assert wilson_lower(0, 0) == 0.0


# ── transitivity learning ────────────────────────────────────────────

def test_no_transitive_inference_before_evidence():
    """Plan test #1: 2 facts alone must NOT license transitivity."""
    op = _op()
    op.ingest_triple(Triple("cat", "is", "mammal"))
    op.ingest_triple(Triple("mammal", "is", "animal"))
    objs = {r.triple.object for r in op.infer("cat", "is")}
    assert objs == {"mammal"}  # direct only — no leap to "animal"


def test_transitivity_learned_after_consistent_evidence():
    op = _op()
    _teach_transitive(op, "is", n=10)
    op.ingest_triple(Triple("cat", "is", "mammal"))
    op.ingest_triple(Triple("mammal", "is", "animal"))
    results = op.infer("cat", "is", max_results=5)
    objs = {r.triple.object for r in results}
    assert "mammal" in objs                      # direct
    assert "animal" in objs                      # inferred transitively
    prof = op.memory.profiles["is"]
    assert prof.transitivity_score > 0.7
    ops = {r.operator for r in results if r.triple.object == "animal"}
    assert "transitive" in ops


def test_transitivity_not_learned_from_open_chains():
    """Chains WITHOUT the closing edge are negative evidence."""
    op = _op()
    for i in range(10):
        op.ingest_triple(Triple(f"a{i}", "next to", f"b{i}"))
        op.ingest_triple(Triple(f"b{i}", "next to", f"c{i}"))
        # never a->c: adjacency is not transitive
    prof = op.memory.profiles["next to"]
    assert prof.transitivity_score < 0.2
    op.ingest_triple(Triple("p", "next to", "q"))
    op.ingest_triple(Triple("q", "next to", "r"))
    objs = {r.triple.object for r in op.infer("p", "next to")}
    assert "r" not in objs


def test_conjunctivity_alpha_is_inverse_of_transitivity():
    op = _op()
    _teach_transitive(op, "is", n=10)
    prof = op.memory.profiles["is"]
    assert prof.conjunctivity_alpha == pytest.approx(
        1.0 - prof.transitivity_score)


# ── symmetry learning ────────────────────────────────────────────────

def test_symmetry_learned_from_reciprocal_pairs():
    op = _op()
    # 10 reciprocal pairs: with fewer (e.g. 5) the Wilson lower bound of
    # 5 pos / 1 neg is 0.44 < 0.5 and the gate correctly stays closed —
    # fail-closed at low evidence volume is the designed behavior.
    names = ["alice", "bob", "carol", "dave", "erin", "frank",
             "gina", "hank", "iris", "jack", "kate", "liam",
             "mona", "nick", "olga", "pete", "quin", "rosa",
             "sam", "tina"]
    for i in range(0, len(names), 2):
        op.ingest_triple(Triple(names[i], "married to", names[i + 1]))
        op.ingest_triple(Triple(names[i + 1], "married to", names[i]))
    prof = op.memory.profiles["married to"]
    assert prof.symmetry_score > 0.8
    # One-directional new fact → symmetric closure fires.
    op.ingest_triple(Triple("zed", "married to", "yara"))
    objs = {r.triple.object for r in op.infer("yara", "married to")}
    assert "zed" in objs


def test_asymmetry_learned_from_one_directional_facts():
    op = _op()
    pairs = [("rain", "wet"), ("fire", "smoke"), ("virus", "illness"),
             ("stress", "insomnia"), ("heat", "expansion"),
             ("friction", "warmth"), ("wind", "erosion"),
             ("sun", "daylight"), ("gravity", "tides"), ("ice", "slip")]
    for a, b in pairs:
        op.ingest_triple(Triple(a, "causes", b))
    prof = op.memory.profiles["causes"]
    assert prof.symmetry_score < 0.2
    objs = {r.triple.object for r in op.infer("wet", "causes")}
    assert "rain" not in objs  # no reverse inference


# ── inverse predicate learning ───────────────────────────────────────

def test_inverse_predicate_detected_and_used():
    op = _op()
    pairs = [("ann", "ben"), ("cal", "dee"), ("eli", "fay"),
             ("gus", "hal"), ("ivy", "jon")]
    for a, b in pairs:
        op.ingest_triple(Triple(a, "parent of", b))
        op.ingest_triple(Triple(b, "child of", a))
    prof = op.memory.profiles["parent of"]
    inv, share = prof.inverse_predicate()
    assert inv == "child of"
    assert share > 0.5
    # New one-directional fact: (kim, parent of, lou) known, ask child of.
    op.ingest_triple(Triple("kim", "parent of", "lou"))
    results = op.infer("lou", "child of", max_results=5)
    objs = {r.triple.object for r in results}
    assert "kim" in objs
    assert any(r.operator == "inverse" for r in results
               if r.triple.object == "kim")


# ── abstention / fail-closed ─────────────────────────────────────────

def test_direct_lookup_never_abstains():
    op = _op()
    op.ingest_triple(Triple("sky", "has", "stars"))
    results = op.infer("sky", "has")
    assert [r.triple.object for r in results] == ["stars"]
    assert results[0].operator == "lookup"


def test_seed_triples_are_stored_but_not_evidence():
    op = TripletInferenceOperator(seed=True)
    assert len(op.memory.triples) > 0
    # No profile accumulated evidence from seeds.
    for prof in op.memory.profiles.values():
        assert prof.transitivity_n == 0
        assert prof.symmetry_n == 0


# ── persistence round-trip (write AND reload — the dead-gate lesson) ─

def test_persistence_round_trip_preserves_learned_gates():
    op = _op()
    _teach_transitive(op, "is", n=10)
    op.ingest_triple(Triple("cat", "is", "mammal"))
    op.ingest_triple(Triple("mammal", "is", "animal"))
    state = op.to_dict()

    op2 = _op()
    op2.from_dict(state)
    prof = op2.memory.profiles["is"]
    assert prof.transitivity_pos == op.memory.profiles["is"].transitivity_pos
    objs = {r.triple.object for r in op2.infer("cat", "is", max_results=5)}
    assert "animal" in objs  # gate still open after reload


# ── sleep consolidation ──────────────────────────────────────────────

def test_sleep_extracts_transitive_schema():
    op = _op()
    _teach_transitive(op, "is", n=10)
    extractor = SleepSchemaExtractor()
    n = extractor.extract_schemas(op.memory)
    assert n >= 1
    assert "transitive-chain:is" in op.memory.schemas
    sc = op.memory.schemas["transitive-chain:is"]
    assert sc.confidence > 0.5


def test_rem_sabotage_is_bounded():
    op = _op()
    _teach_transitive(op, "is", n=5)
    prof = op.memory.profiles["is"]
    before = prof.transitivity_pos
    extractor = SleepSchemaExtractor()
    perturbed = extractor.rem_sabotage(op.memory, rate=1.0)
    assert perturbed >= 1
    assert prof.transitivity_pos == before + 1  # one count, not a rewrite


# ── curiosity hook ───────────────────────────────────────────────────

def test_curiosity_targets_low_evidence_predicates():
    op = _op()
    _teach_transitive(op, "is", n=10)          # rich evidence
    op.ingest_triple(Triple("a", "orbits", "b"))  # scarce evidence
    hook = InferenceCuriosityHook(op.memory)
    assert hook.epistemic_value("orbits") > hook.epistemic_value("is")
    assert "orbits" in hook.curiosity_targets(top_k=3)


# ── adapters ─────────────────────────────────────────────────────────

def test_openie_fact_adapter_maps_obj_field():
    from ravana.web.openie import Fact
    op = _op()
    op.ingest_openie_fact(Fact(subject="exercise", relation="improves",
                               obj="mood", confidence=0.7))
    assert op.memory.has_fact("exercise", "improves", "mood")


def test_proposition_adapter():
    from ravana.core.proposition_parser import Proposition
    op = _op()
    op.ingest_proposition(Proposition(subject="socrates", predicate="is",
                                      object="man", confidence=0.8))
    assert op.memory.has_fact("socrates", "is", "man")


# ── supersedence ─────────────────────────────────────────────────────

def test_superseded_facts_are_excluded_from_inference():
    op = _op()
    _teach_transitive(op, "is", n=10)
    op.ingest_triple(Triple("pluto", "is", "planet"))
    op.memory.supersede("pluto", "is", "planet")
    objs = {r.triple.object for r in op.infer("pluto", "is")}
    assert "planet" not in objs
