"""Relation-vector templates are cached but callers get independent copies."""
import numpy as np

from ravana_ml.graph import ConceptEdge


def test_cached_relation_vector_returns_independent_copy():
    first = ConceptEdge._init_relation_vector("causal", 16)
    second = ConceptEdge._init_relation_vector("causal", 16)
    assert np.array_equal(first, second)
    assert first is not second
    first[:] = 0.0
    third = ConceptEdge._init_relation_vector("causal", 16)
    assert np.array_equal(second, third)
    assert not np.array_equal(first, third)


def test_cached_relation_vector_values_are_unchanged():
    expected = ConceptEdge._relation_vector_template("temporal", 16)
    got = ConceptEdge._init_relation_vector("temporal", 16)
    assert np.array_equal(got, expected)
    assert got is not expected
