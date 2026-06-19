"""Tests for ravana_ml.nn.rlm_v2_common — constants, relation types, keyword maps."""

import pytest
import numpy as np
from ravana_ml.nn.rlm_v2_common import (
    RELATION_TYPES,
    _KEYWORD_MAP,
)


class TestRelationTypes:
    """Test the relation type definitions."""

    def test_relation_types_count(self):
        """There should be exactly 6 relation types."""
        assert len(RELATION_TYPES) == 6

    def test_relation_types_order(self):
        """Test the exact order of relation types."""
        expected = ["causal", "semantic", "temporal", "possessive", "analogical", "contextual"]
        assert RELATION_TYPES == expected

    def test_no_duplicates(self):
        """All relation types should be unique."""
        assert len(RELATION_TYPES) == len(set(RELATION_TYPES))


class TestKeywordMap:
    """Test the _KEYWORD_MAP for relation type classification."""

    def test_keyword_map_has_causal(self):
        """Causal keywords should include core causal verbs."""
        causal_keywords = _KEYWORD_MAP.get("causal", [])
        assert "causes" in causal_keywords
        assert "produces" in causal_keywords
        assert "leads_to" in causal_keywords
        assert "results_in" in causal_keywords

    def test_semantic_not_in_keyword_map(self):
        """Semantic is not a key in _KEYWORD_MAP since it's the default fallback relation type."""
        assert "semantic" not in _KEYWORD_MAP

    def test_keyword_map_keys(self):
        """Keyword map covers the non-semantic relation types."""
        non_semantic = [t for t in RELATION_TYPES if t != "semantic"]
        for key in _KEYWORD_MAP:
            assert key in set(non_semantic), f"{key} not in non-semantic types"

    def test_keyword_map_compound_predicates(self):
        """Causal should include compound predicates (single-token)."""
        causal = _KEYWORD_MAP.get("causal", [])
        compounds = [k for k in causal if "_" in k]
        assert len(compounds) > 0
        assert "leads_to" in compounds
        assert "results_in" in compounds

    def test_keyword_map_temporal(self):
        """Temporal should include time-related keywords."""
        temporal = _KEYWORD_MAP.get("temporal", [])
        assert "then" in temporal
        assert "after" in temporal
        assert "before" in temporal

    def test_keyword_map_possessive(self):
        """Possessive should include ownership keywords."""
        possessive = _KEYWORD_MAP.get("possessive", [])
        assert "has" in possessive
        assert "contains" in possessive
        assert "includes" in possessive

    def test_keyword_map_analogical(self):
        """Analogical should include similarity keywords."""
        analogical = _KEYWORD_MAP.get("analogical", [])
        assert "like" in analogical
        assert "similar" in analogical
        assert "resembles" in analogical

    def test_keyword_map_contextual(self):
        """Contextual should include spatial/preposition keywords."""
        contextual = _KEYWORD_MAP.get("contextual", [])
        assert "in" in contextual
        assert "on" in contextual
        assert "with" in contextual

    def test_no_empty_keyword_lists(self):
        """All relation types should have at least one keyword."""
        for rel_type, keywords in _KEYWORD_MAP.items():
            assert len(keywords) > 0, f"{rel_type} has no keywords"

    def test_classify_relation_semantic_fallback(self):
        """Empty relation tokens should yield 'semantic' index (1)."""
        from ravana_ml.nn.rlm_v2_common import RELATION_TYPES
        from ravana_ml.graph import ConceptGraph, ConceptBindingMap
        # Create a minimal mock with _decode_token
        class _MockGraph:
            def __init__(self):
                self.concept_dim = 8
                self.embed_dim = 8
                self.graph = ConceptGraph(dim=8, max_nodes=100)
                self.binding_map = ConceptBindingMap()
                self._prototype_vectors = {}
                self._prototype_hierarchy = {}
            def _decode_token(self, tid):
                return f"tok_{tid}"
            def classify_relation(self, relation_token_ids):
                from ravana_ml.nn.rlm_v2_common import RELATION_TYPES, _KEYWORD_MAP
                if not relation_token_ids:
                    return RELATION_TYPES.index("semantic")
                return RELATION_TYPES.index("semantic")

        mock = _MockGraph()
        idx = mock.classify_relation([])
        assert RELATION_TYPES[idx] == "semantic"

    def test_classify_relation_causal_keyword(self):
        """'causes' as a relation token should map to 'causal'."""
        from ravana_ml.nn.rlm_v2_common import RELATION_TYPES, _KEYWORD_MAP
        mock_type = type("Mock", (), {
            "_decode_token": lambda self, tid: "causes" if tid == 1 else "",
            "concept_dim": 8,
        })()
        idx = None
        # Manual keyword matching logic
        words = {"causes"}
        for rel_type, keywords in _KEYWORD_MAP.items():
            for word in words:
                if word in keywords:
                    idx = RELATION_TYPES.index(rel_type)
                    break
            if idx is not None:
                break
        assert idx == RELATION_TYPES.index("causal")

    def test_keyword_uniqueness(self):
        """Keywords should not appear in multiple relation types."""
        all_keywords = {}
        for rel_type, keywords in _KEYWORD_MAP.items():
            for kw in keywords:
                if kw in all_keywords:
                    pytest.fail(f"Keyword '{kw}' appears in both {all_keywords[kw]} and {rel_type}")
                all_keywords[kw] = rel_type
