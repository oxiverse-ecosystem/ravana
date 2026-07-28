"""Lever-2 tests: neuro-symbolic premise extractor.

Verifies the STG/Broca analogue de-blobs clause fragments into
compact entities AND binds the existing relation / quantifier /
negation frames, with the 6.5 channel's fail-closed guarantee
preserved (noise -> []).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ravana", "src"))

from ravana.core.deductive_extractor import DeductivePremiseExtractor
from ravana.core.deductive_reasoning import (
    DeductiveTriple, parse_deductive_premises, deductive_mc_answer,
)


def _ext(text, spacy=True):
    return DeductivePremiseExtractor(use_spacy=spacy).extract(text)


# --- 1. De-blobbing: long preamble must NOT become an entity --------
def test_deblob_long_preamble():
    # The classic P0 blob case.
    clause = ("researchers hypothesized that the reason why westernized "
              "black people suffer from hypertension is the result of the "
              "interaction of two environmental and genetic factors")
    tris = _ext(clause)
    # No clean relational frame should be extracted from a hypothesis blob.
    assert tris == [], f"expected [], got {tris}"


def test_deblob_positional_compact_entities():
    text = ("the administrative service area is southwest of the cultural "
            "area, and the cultural area is southeast of the leisure area")
    tris = _ext(text)
    preds = {t.predicate for t in tris}
    assert "loc:southwestof" in preds or "loc:southwest" in preds
    # entities must be compact (<=3 content words), not the whole clause
    for t in tris:
        assert len(t.subject.split()) <= 3, t.subject
        assert len(t.object.split()) <= 3, t.object


# --- 2. Comparative -------------------------------------------------------
def test_comparative_taller():
    tris = _ext("Alice is taller than Bob, and Bob is taller than Carol")
    assert any(t.predicate == "compare:taller" for t in tris)
    # transitive chainable: Alice taller Bob, Bob taller Carol
    chain = [t for t in tris if t.predicate == "compare:taller"]
    subs = {t.subject for t in chain}
    assert "alice" in subs and "bob" in subs


# --- 3. Positional ---------------------------------------------------------
def test_positional_left_of():
    tris = _ext("A is to the left of B, and B is to the left of C")
    assert any(t.predicate == "loc:leftof" for t in tris)
    lefts = [t for t in tris if t.predicate == "loc:leftof"]
    assert len(lefts) >= 2


# --- 4. Universal quantifier ---------------------------------------------
def test_universal_forall():
    tris = _ext("All men are mortal")
    assert any(t.quantifier == "forall" and t.relation_type == "rule"
               for t in tris), tris
    # regular plural normalization: "men" -> "man"
    assert any(t.object == "mortal" for t in tris)


# --- 5. Negation / polarity ---------------------------------------------
def test_negated_polarity():
    tris = _ext("Socrates is not mortal")
    neg = [t for t in tris if t.predicate == "is"]
    assert neg, tris
    assert neg[0].polarity is False, neg[0]


# --- 6. Fail-closed on pure noise ---------------------------------------
def test_fail_closed_noise():
    assert _ext("Meanwhile, the committee deliberated at length.") == []
    assert _ext("") == []


# --- 7. Regex fallback still works (spaCy off) -------------------------
def test_regex_fallback():
    tris_spacy = _ext("A is taller than B", spacy=True)
    tris_regex = _ext("A is taller than B", spacy=False)
    # both should find the comparative; fallback must not crash
    assert any(t.predicate == "compare:taller" for t in tris_regex)
    assert len(tris_regex) >= 1


# --- 8. Integration: 6.5 channel via extractor, fail-closed --------
def test_integration_end_to_end():
    q = ("All dogs are mammals. Rex is a dog. "
          "Options: A) rex is a mammal B) rex is a reptile")
    # The channel must pick exactly one entailed option.
    ans = deductive_mc_answer(q)
    assert ans is not None
    assert "mammal" in ans.lower()


def test_integration_zero_fp_on_noise():
    # A non-chainable MC must abstain (None), never fabricate.
    q = ("The committee met on Tuesday. "
          "Options: A) they ate lunch B) they solved the case")
    assert deductive_mc_answer(q) is None
