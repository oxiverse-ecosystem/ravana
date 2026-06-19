"""Tests for rlm_v2_graph methods — concept lookup, prototypes, graph traversal, edge management."""

import pytest
import numpy as np
from ravana_ml.nn.rlm_v2_graph import GraphMixin
from ravana_ml.graph import ConceptGraph, ConceptBindingMap


class _MockGraphModel(GraphMixin):
    """Minimal mock with graph-related methods."""
    def __init__(self, concept_dim=8, embed_dim=8):
        self.concept_dim = concept_dim
        self.embed_dim = embed_dim
        self._max_concepts = 100
        self.use_prototype_inheritance = True
        self._prototype_similarity_threshold = 0.5

        self.graph = ConceptGraph(dim=concept_dim, max_nodes=100)
        self.binding_map = ConceptBindingMap()

        self._prototype_hierarchy = {}
        self._prototype_vectors = {}
        self._prototype_levels = {}
        self._prototype_children = {}
        self._novel_entity_concepts = {}
        self._entity_adapters = {}

        # Projection (simple linear from embed_dim to concept_dim)
        self._proj_W = np.random.randn(concept_dim, embed_dim).astype(np.float32) * 0.1

        self._node_matrix_version = -1
        self._node_matrix_cache = None
        self._norm_cache = {}
        self._rel_vector_version = -1
        self._rel_vector_cache = {}
        self._tokenizer = None
        self._cross_domain_edges_injected = set()
        self._verb_offsets = {}
        self._verb_offset_count = {}

    def _project_to_concept(self, embed_vec):
        if len(embed_vec) == self.concept_dim:
            return embed_vec
        result = self._proj_W @ embed_vec
        return result

    def _project_to_embed(self, concept_vec):
        if len(concept_vec) == self.embed_dim:
            return concept_vec
        return concept_vec  # simplified

    def _decode_token(self, token_id):
        return f"tok_{token_id}"

    def _invalidate_caches(self):
        self._node_matrix_version = -1

    def _get_or_create_concept(self, token_id, embed_vec):
        concept_vec = self._project_to_concept(embed_vec)
        bindings = self.binding_map.get_concepts(token_id, min_confidence=0.1)
        if bindings:
            cid = bindings[0].concept_id
            if self.graph.get_node(cid) is not None:
                return cid
        if len(self.graph.nodes) < self._max_concepts:
            node = self.graph.add_node(vector=concept_vec, label=f"tok_{token_id}")
            nid = node.id
        else:
            results = self.graph.find_similar(concept_vec, k=1)
            nid = results[0][0] if results else -1
        self.binding_map.bind(token_id, nid, confidence=0.9)
        return nid


class TestGetOrCreateConcept:
    def test_creates_new(self):
        model = _MockGraphModel()
        cid = model._get_or_create_concept(0, np.random.randn(model.embed_dim).astype(np.float32))
        assert cid >= 0
        assert model.graph.get_node(cid) is not None

    def test_reuses_existing(self):
        model = _MockGraphModel()
        vec = np.random.randn(model.embed_dim).astype(np.float32)
        cid1 = model._get_or_create_concept(0, vec)
        cid2 = model._get_or_create_concept(0, vec)
        assert cid1 == cid2

    def test_node_has_label(self):
        model = _MockGraphModel()
        cid = model._get_or_create_concept(5, np.random.randn(model.embed_dim).astype(np.float32))
        assert model.graph.get_node(cid).label == "tok_5"


class TestPrototypeSystem:
    def test_init_default_prototypes_no_crash(self):
        model = _MockGraphModel()
        for i in range(10):
            vec = np.random.randn(model.concept_dim).astype(np.float32)
            model.graph.add_node(vector=vec / np.linalg.norm(vec), label=f"c_{i}")
        model._init_default_prototypes()  # Should not crash

    def test_register_prototype(self):
        model = _MockGraphModel()
        n1 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32), label="a")
        n2 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32), label="b")
        model._register_prototype("test_proto", [n1.id, n2.id])
        assert "test_proto" in model._prototype_hierarchy
        assert model._prototype_hierarchy["test_proto"] == [n1.id, n2.id]
        assert "test_proto" in model._prototype_vectors

    def test_inherit_from_prototype(self):
        model = _MockGraphModel()
        p1 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32), label="proto")
        p2 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32), label="target")
        model.graph.add_edge(p1.id, p2.id, weight=0.8, relation_type="causal")
        model._prototype_hierarchy["semantic"] = [p1.id]
        model._prototype_vectors["semantic"] = p1.vector
        new_node = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32), label="new")
        model._inherit_from_prototype(new_node.id, "semantic", similarity=0.7)
        edge = model.graph.get_edge(new_node.id, p2.id)
        assert edge is not None
        assert edge.relation_type == "causal"

    def test_find_nearest_prototype(self):
        model = _MockGraphModel()
        model._prototype_vectors["a"] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        model._prototype_vectors["b"] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        embed = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        label, sim = model._find_nearest_prototype(embed)
        assert label == "a"
        assert sim > 0.5


class TestGraphTraversal:
    def test_normalize_outgoing_weights(self):
        model = _MockGraphModel()
        n1 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32), label="src")
        n2 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32), label="t1")
        n3 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32), label="t2")
        model.graph.add_edge(n1.id, n2.id, weight=5.0, relation_type="causal")
        model.graph.add_edge(n1.id, n3.id, weight=3.0, relation_type="semantic")
        model._normalize_outgoing_weights(budget=4.0)
        e1 = model.graph.get_edge(n1.id, n2.id)
        e2 = model.graph.get_edge(n1.id, n3.id)
        total = (e1.weight if e1 else 0) + (e2.weight if e2 else 0)
        assert total <= 4.0

    def test_prune_weak_edges(self):
        model = _MockGraphModel()
        n1 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        n2 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        model.graph.add_edge(n1.id, n2.id, weight=0.05, relation_type="semantic")
        model._prune_weak_edges(threshold=0.1)
        assert len(model.graph.edges) == 0

    def test_decompose_triple_three(self):
        model = _MockGraphModel()
        s, r, o = model.decompose_triple([10, 20, 30])
        assert s == [10] and r == [20] and o == [30]

    def test_decompose_triple_four(self):
        model = _MockGraphModel()
        s, r, o = model.decompose_triple([10, 20, 25, 30])
        assert s == [10] and r == [20, 25] and o == [30]

    def test_decompose_triple_two(self):
        model = _MockGraphModel()
        s, r, o = model.decompose_triple([10, 30])
        assert s == [10] and r == [] and o == [30]

    def test_decompose_triple_one(self):
        model = _MockGraphModel()
        s, r, o = model.decompose_triple([10])
        assert s == [10] and r == [] and o == []

    def test_decompose_triple_empty(self):
        model = _MockGraphModel()
        s, r, o = model.decompose_triple([])
        assert s == [] and r == [] and o == []

    def test_classify_relation_empty(self):
        from ravana_ml.nn.rlm_v2_common import RELATION_TYPES
        model = _MockGraphModel()
        idx = model.classify_relation([])
        assert RELATION_TYPES[idx] == "semantic"

    def test_inject_cross_domain_edge(self):
        model = _MockGraphModel()
        n1 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        n2 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        model._inject_cross_domain_edge(n1.id, n2.id, "causal", subject_tid=None)
        assert model.graph.get_edge(n1.id, n2.id) is not None

    def test_inject_cross_domain_idempotent(self):
        model = _MockGraphModel()
        n1 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        n2 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        model._inject_cross_domain_edge(n1.id, n2.id, "causal", subject_tid=None)
        model._inject_cross_domain_edge(n1.id, n2.id, "causal", subject_tid=None)
        n_injected = len(model._cross_domain_edges_injected)
        assert n_injected == 1

    def test_get_query_confidence(self):
        model = _MockGraphModel()
        n1 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        n2 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        e = model.graph.add_edge(n1.id, n2.id, weight=0.5, relation_type="causal")
        e.confidence = 0.8
        conf = model.get_query_confidence(n1.id, "causal")
        assert conf == 0.5 * 0.8

    def test_traverse_no_tokenizer(self):
        model = _MockGraphModel()
        result = model.traverse("hello", steps=3)
        assert result == []


class TestEdgeManagement:
    def test_anti_hebbian_prune_empty(self):
        model = _MockGraphModel()
        assert model._anti_hebbian_prune_polluted_edges() == 0

    def test_inject_direct_edges(self):
        model = _MockGraphModel()
        n1 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        n2 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        n_inj = model._inject_direct_edges_if_needed(n1.id, n2.id, "causal")
        assert n_inj == 1
        assert model.graph.get_edge(n1.id, n2.id) is not None

    def test_inject_direct_edges_strengthen(self):
        model = _MockGraphModel()
        n1 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        n2 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        model.graph.add_edge(n1.id, n2.id, weight=0.1, relation_type="causal")
        n_inj = model._inject_direct_edges_if_needed(n1.id, n2.id, "causal", threshold=0.5)
        assert n_inj == 1
        e = model.graph.get_edge(n1.id, n2.id)
        assert e.weight >= 0.5
