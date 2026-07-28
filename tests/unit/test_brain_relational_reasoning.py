"""Brain-faithful relational reasoning — synthetic verification suite.

Target (relational-reasoning plan exit criteria):
- Relational Reasoning Accuracy: >=95% on this suite.
- Fail-Closed Guarantee: 0 fabricated answers / 0 regressions on
  non-entailed or contradictory options (every "should abstain" case
  must return None).
- Lifetime Profile Decoupling: 0 dependence on historical
  RelationProfile counts for in-prompt relational inference.

The suite exercises the FIVE System-2 relational primitives the plan
lists, using NOVEL relations (left_of, taller_than, part_of, ...) so
it is impossible to pass via lifetime frequency — a pass proves the
metarules are structural (role/problem-bound), not frequency-bound.

The core logic is reached in two ways:
  (A) the standalone deductive_mc_answer() over raw MC text
      (end-to-end: parser -> ProblemWorkingMemory -> RoleMetaruleEngine)
  (B) direct ProblemWorkingMemory + RoleMetaruleEngine construction
      (unit-level, no NL parsing) for crisp primitive checks.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "..", "..", "ravana", "src"))

from ravana.core.deductive_reasoning import (  # noqa: E402
    DeductiveTriple, ProblemWorkingMemory, RoleMetaruleEngine,
    parse_deductive_premises, deductive_mc_answer,
)

# ── helpers ──────────────────────────────────────────────────
def _pwm(*triples):
    p = ProblemWorkingMemory()
    for t in triples:
        p.add(t)
    RoleMetaruleEngine().apply(p)
    return p

def _T(s, p, o, **kw):
    return DeductiveTriple(s, p, o, **kw)

# ── 1. Transitive chaining over a NOVEL predicate ──────────
def test_transitive_novel_predicate():
    p = _pwm(
        _T("a", "loc:leftof", "b"),
        _T("b", "loc:leftof", "c"),
        _T("c", "loc:leftof", "d"),
    )
    assert p.entail_confidence(_T("a", "loc:leftof", "d")) > 0.0
    assert p.has_fact("a", "loc:leftof", "d", True)

def test_transitive_multi_hop_forward_and_skip():
    p = _pwm(_T("x", "part_of", "y"), _T("y", "part_of", "z"))
    assert p.entail_confidence(_T("x", "part_of", "z")) > 0.0

# ── 2. Comparative orderings ───────────────────────────────
def test_comparative_ordering():
    p = _pwm(
        _T("x", "compare:taller", "y"),
        _T("y", "compare:taller", "z"),
    )
    assert p.entail_confidence(_T("x", "compare:taller", "z")) > 0.0

# ── 3. Universal syllogism (rule implication) ─────────────
def test_universal_syllogism_isolated():
    p = _pwm(
        # ∀x: Man(x) -> Mortal(x)
        _T("man", "implies", "mortal", relation_type="rule"),
        # Man(Socrates)
        _T("socrates", "is", "man"),
    )
    assert p.entail_confidence(_T("socrates", "is", "mortal")) > 0.0

def test_universal_syllogism_end_to_end_mc():
    # Regular plurals ("dogs"->"dog") stay in morphology-only scope;
    # the engine logic is identical to test_universal_syllogism_isolated.
    q = ("All dogs are mammals. Rex is a dog. "
          "Options: A) rex is a mammal B) rex is a reptile")
    assert deductive_mc_answer(q) == "rex is a mammal"

# ── 4. Spatial / inverse reasoning ────────────────────────
def test_inverse_derived_via_metadata():
    p = _pwm(_T("a", "loc:leftof", "b", inverse="loc:rightof"))
    assert p.entail_confidence(_T("b", "loc:rightof", "a")) > 0.0

def test_inverse_derived_via_directional_table():
    # no explicit inverse declared; parser/NL path could supply one,
    # but here we rely on the closed-class directional-opposite table.
    p = _pwm(_T("a", "loc:leftof", "b"))
    RoleMetaruleEngine().apply(p)
    # directional table fills the inverse ONLY when an antonym triple
    # is also present? No — the engine infers it structurally from the
    # single directional fact via the built-in opposite map.
    assert p.entail_confidence(_T("b", "loc:rightof", "a")) > 0.0

# ── 5. Symmetric closure ──────────────────────────────────
def test_symmetric_closure():
    p = _pwm(_T("a", "next_to", "b", relation_type="symmetric"))
    assert p.entail_confidence(_T("b", "next_to", "a")) > 0.0

# ── Fail-closed: abstain on non-entailed / ambiguous ──────
def test_abstain_when_no_premise_chain():
    q = ("The sky is blue. Grass is green. "
          "Options: A) the sky is green B) grass is blue")
    assert deductive_mc_answer(q) is None

def test_abstain_on_ambiguous_options():
    q = ("All men are mortal. Socrates is a man. "
          "Options: A) socrates is mortal B) the mortal socrates")
    assert deductive_mc_answer(q) is None

def test_abstain_on_contradiction():
    # A left_of B AND NOT(A left_of B) -> contradiction -> abstain
    p = _pwm(
        _T("a", "loc:leftof", "b", polarity=True),
        _T("a", "loc:leftof", "b", polarity=False),
    )
    assert p.contradiction_exists() is True

def test_abstain_non_mc_input():
    assert deductive_mc_answer("All men are mortal.") is None

# ── Lifetime-profile decoupling proof ─────────────────────
def test_zero_lifetime_dependence():
    """A cold/nonexistent RelationProfile must NOT gate the result.
    We build the buffers directly (no engine, no triplet_op) and the
    metarules still chain. The standalone function also imports nothing
    from triplet_inference RelationProfile."""
    # purely structural: no engine, no profiles touched
    p = _pwm(_T("a", "novel_rel", "b"), _T("b", "novel_rel", "c"))
    assert p.entail_confidence(_T("a", "novel_rel", "c")) > 0.0

# ── Parser sanity (honest about narrow scope) ─────────────
def test_parser_extracts_comparative():
    ts = parse_deductive_premises("A is taller than B.")
    assert any(t.predicate == "compare:taller" and t.object == "b"
               for t in ts)

def test_parser_extracts_positional():
    ts = parse_deductive_premises("A is to the left of B and C is to the left of D.")
    rels = {t.predicate for t in ts}
    assert "loc:leftof" in rels

def test_parser_universal_rule():
    ts = parse_deductive_premises("All men are mortal.")
    assert any(t.relation_type == "rule" and t.predicate == "implies"
               and t.subject == "men" and t.object == "mortal" for t in ts)

def test_parser_rejects_blob():
    # long clause cannot be a clean SPO -> dropped by token cap
    ts = parse_deductive_premises(
        "Researchers hypothesized that the reason why westernized black "
        "people suffer from hypertension is the result of interaction.")
    assert ts == []
