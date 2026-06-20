"""Tests for ravana_ml.relation_ontology."""

import pytest
from ravana_ml.relation_ontology import (
    SUB_FAMILIES, SUPER_FAMILIES, Candidate, TraversalConfig,
    get_sub_family, get_family, get_confidence, matches_config,
)


class TestOntologyData:
    def test_sub_families_defined(self):
        assert len(SUB_FAMILIES) >= 7

    def test_sub_family_has_keys(self):
        for name, info in SUB_FAMILIES.items():
            assert "family" in info
            assert "predicates" in info
            assert "description" in info
            assert "confidence_range" in info
            assert len(info["predicates"]) > 0
            lo, hi = info["confidence_range"]
            assert 0 <= lo <= hi <= 1.0

    def test_causal_family(self):
        assert SUB_FAMILIES["causal-strong"]["family"] == "causal"
        assert SUB_FAMILIES["causal-moderate"]["family"] == "causal"
        assert SUB_FAMILIES["causal-weak"]["family"] == "causal"

    def test_directional_family(self):
        assert SUB_FAMILIES["directional-positive"]["family"] == "directional"
        assert SUB_FAMILIES["directional-negative"]["family"] == "directional"

    def test_super_families(self):
        assert "causal_all" in SUPER_FAMILIES
        assert "directional_all" in SUPER_FAMILIES
        assert "causal_directional" in SUPER_FAMILIES

    def test_causal_super_contains_all(self):
        for sub in ["causal-strong", "causal-moderate", "causal-weak"]:
            assert sub in SUPER_FAMILIES["causal_all"]


class TestOntologyFunctions:
    def test_get_sub_family_found(self):
        assert get_sub_family("causes") == "causal-strong"

    def test_get_sub_family_contributory(self):
        assert get_sub_family("contributes_to") == "causal-weak"

    def test_get_sub_family_not_found(self):
        assert get_sub_family("nonexistent_verb") is None

    def test_get_family_causal(self):
        assert get_family("causes") == "causal"

    def test_get_family_directional(self):
        assert get_family("increases") == "directional"

    def test_get_family_not_found(self):
        assert get_family("xyzzy") is None

    def test_get_confidence_causal_strong(self):
        conf = get_confidence("causes")
        assert 0.85 <= conf <= 0.95

    def test_get_confidence_causal_weak(self):
        conf = get_confidence("contributes_to")
        assert 0.35 <= conf <= 0.60

    def test_get_confidence_default(self):
        assert get_confidence("unknown_predicate") == 0.5


class TestCandidate:
    def test_candidate_default_path(self):
        c = Candidate(
            word="heat",
            predicate="causes",
            family="causal",
            sub_family="causal-strong",
            depth=1,
            confidence=0.9,
        )
        assert c.word == "heat"
        assert c.predicate == "causes"
        assert c.path == []

    def test_candidate_with_path(self):
        c = Candidate(
            word="heat", predicate="causes", family="causal",
            sub_family="causal-strong", depth=2, confidence=0.85,
            path=[("fire", "causes")],
        )
        assert len(c.path) == 1


class TestTraversalConfig:
    def test_default_config(self):
        tc = TraversalConfig()
        assert tc.mode == "family"
        assert tc.family is None
        assert tc.sub_family is None
        assert tc.super_family is None

    def test_custom_config(self):
        tc = TraversalConfig(mode="predicate", sub_family="causes")
        assert tc.mode == "predicate"


class TestMatchesConfig:
    def test_relaxed_matches_all(self):
        tc = TraversalConfig(mode="relaxed")
        assert matches_config("causes", tc) is True

    def test_predicate_mode_match(self):
        tc = TraversalConfig(mode="predicate", sub_family="causes")
        assert matches_config("causes", tc) is True

    def test_predicate_mode_no_match(self):
        tc = TraversalConfig(mode="predicate", sub_family="causes")
        assert matches_config("increases", tc) is False

    def test_sub_family_mode_match(self):
        tc = TraversalConfig(mode="sub_family", sub_family="causal-strong")
        assert matches_config("causes", tc) is True

    def test_sub_family_mode_no_match(self):
        tc = TraversalConfig(mode="sub_family", sub_family="causal-weak")
        assert matches_config("causes", tc) is False

    def test_family_mode_match(self):
        tc = TraversalConfig(mode="family", family="causal")
        assert matches_config("causes", tc) is True
        assert matches_config("contributes_to", tc) is True

    def test_family_mode_no_match(self):
        tc = TraversalConfig(mode="family", family="temporal")
        assert matches_config("causes", tc) is False

    def test_super_family_mode_match(self):
        tc = TraversalConfig(mode="super_family", super_family="causal_all")
        assert matches_config("causes", tc) is True
        assert matches_config("contributes_to", tc) is True

    def test_super_family_mode_no_match(self):
        tc = TraversalConfig(mode="super_family", super_family="directional_all")
        assert matches_config("causes", tc) is False

    def test_matches_config_no_predicate(self):
        tc = TraversalConfig(mode="predicate", sub_family="nonexistent")
        assert matches_config("causes", tc) is False
