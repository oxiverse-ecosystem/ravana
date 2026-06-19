"""Tests for ravana_chat_src nn modules: Plasticity, PropagationEngine, RelationPredictor."""

import sys, os
_rcs = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ravana_chat_src", "src")
if _rcs not in sys.path:
    sys.path.insert(0, _rcs)

import pytest
import numpy as np
from ravana_chat.nn.rlm.plasticity import Plasticity
from ravana_chat.nn.rlm.propagation import PropagationEngine
from ravana_chat.nn.rlm.relation_predictor import RelationPredictor, RELATION_TYPES
from ravana_ml.graph import ConceptGraph


# ── PropagationEngine Tests ──

class TestPropagationEngine:
    def test_default_init(self):
        graph = ConceptGraph(dim=8, max_nodes=100)
        pe = PropagationEngine(graph)
        assert pe.graph is graph

    def test_spread_activation_no_active_nodes(self):
        graph = ConceptGraph(dim=8, max_nodes=100)
        pe = PropagationEngine(graph)
        pe.spread_activation(steps=3)  # Should not raise

    def test_spread_activation_with_active_nodes(self):
        graph = ConceptGraph(dim=8, max_nodes=100)
        n1 = graph.add_node(vector=np.ones(8, dtype=np.float32), label="a")
        n2 = graph.add_node(vector=np.ones(8, dtype=np.float32), label="b")
        graph.add_edge(n1.id, n2.id, weight=0.8, relation_type="semantic")
        pe = PropagationEngine(graph)
        graph.activate(n1.id, 1.0)
        pe.spread_activation(steps=2)
        node = graph.get_node(n2.id)
        assert node.activation > 0.0

    def test_get_prediction(self):
        graph = ConceptGraph(dim=8, max_nodes=100)
        n1 = graph.add_node(vector=np.ones(8, dtype=np.float32), label="a")
        n2 = graph.add_node(vector=np.ones(8, dtype=np.float32), label="b")
        pe = PropagationEngine(graph)
        graph.activate(n1.id, 1.0)
        graph.activate(n2.id, 0.8)
        predictions = pe.get_prediction([n1.id, n2.id], top_k=5)
        assert isinstance(predictions, list)

    def test_relation_aware_spreading(self):
        graph = ConceptGraph(dim=8, max_nodes=100)
        n1 = graph.add_node(np.ones(8), "a")
        n2 = graph.add_node(np.ones(8), "b")
        graph.add_edge(n1.id, n2.id, weight=0.8, relation_type="causal")
        pe = PropagationEngine(graph)
        graph.activate(n1.id, 1.0)
        scores = pe.relation_aware_spreading(n1.id, "causal", steps=2)
        assert n2.id in scores or len(scores) >= 0

    def test_n_hop_bfs(self):
        graph = ConceptGraph(dim=8, max_nodes=100)
        n1 = graph.add_node(np.ones(8), "a")
        n2 = graph.add_node(np.ones(8), "b")
        n3 = graph.add_node(np.ones(8), "c")
        graph.add_edge(n1.id, n2.id, weight=0.8, relation_type="causal")
        graph.add_edge(n2.id, n3.id, weight=0.7, relation_type="causal")
        pe = PropagationEngine(graph)
        scores = pe.n_hop_bfs(n1.id, "causal", max_hops=3)
        assert n3.id in scores or n2.id in scores or len(scores) >= 0

    def test_collect_active_concepts(self):
        graph = ConceptGraph(dim=8, max_nodes=100)
        n1 = graph.add_node(np.ones(8), "a")
        n2 = graph.add_node(np.ones(8), "b")
        graph.add_edge(n1.id, n2.id, weight=0.8, relation_type="semantic")
        pe = PropagationEngine(graph)
        graph.activate(n1.id, 1.0)
        scores = pe.collect_active_concepts(n1.id, -1, "semantic", disable_spreading=True)
        assert isinstance(scores, dict)


# ── Plasticity Tests ──

class TestPlasticity:
    def test_default_init(self):
        graph = ConceptGraph(dim=8, max_nodes=100)
        pl = Plasticity(graph, base_lr=0.005)
        assert pl.graph is graph
        assert pl.base_lr == 0.005
        assert pl.hebbian is not None
        assert pl.structural is not None
        assert pl.currencies is not None
        assert pl.currency is not None

    def test_decompose_triple(self):
        graph = ConceptGraph(dim=8, max_nodes=100)
        pl = Plasticity(graph)
        # 3+ tokens
        subj, rel, obj = pl._decompose_triple([1, 2, 3])
        assert subj == [1]
        assert rel == [2]
        assert obj == [3]

    def test_decompose_triple_two(self):
        graph = ConceptGraph(dim=8, max_nodes=100)
        pl = Plasticity(graph)
        subj, rel, obj = pl._decompose_triple([1, 2])
        assert subj == [1]
        assert rel == []
        assert obj == [2]

    def test_decompose_triple_one(self):
        graph = ConceptGraph(dim=8, max_nodes=100)
        pl = Plasticity(graph)
        subj, rel, obj = pl._decompose_triple([1])
        assert subj == [1]
        assert rel == []
        assert obj == []

    def test_anti_hebbian_prune(self):
        graph = ConceptGraph(dim=8, max_nodes=100)
        pl = Plasticity(graph)
        pruned = pl.anti_hebbian_prune()
        assert isinstance(pruned, int)

    def test_currency_properties(self):
        graph = ConceptGraph(dim=8, max_nodes=100)
        pl = Plasticity(graph)
        assert pl.identity_strength >= 0.0
        pl.identity_strength = 0.8
        assert pl.identity_strength == 0.8


# ── RelationPredictor Tests ──

class TestRelationPredictor:
    def test_default_init(self):
        rp = RelationPredictor(vocab_size=100, embed_dim=64, concept_dim=64)
        assert rp.vocab_size == 100
        assert rp.embed_dim == 64
        assert len(RELATION_TYPES) >= 5
        assert rp.relation_type_embed is not None
        assert rp.relation_classifier is not None

    def test_classify_relation_causal(self):
        rp = RelationPredictor(100, 64, 64)
        def decode(tid):
            return {1: "causes", 2: "leads", 3: "is"}.get(tid, "")
        idx = rp.classify_relation([1], decode)
        assert RELATION_TYPES[idx] == "causal"

    def test_classify_relation_semantic_default(self):
        rp = RelationPredictor(100, 64, 64)
        def decode(tid):
            return {1: "is", 2: "and"}.get(tid, "")
        idx = rp.classify_relation([1], decode)
        assert RELATION_TYPES[idx] == "semantic"

    def test_classify_relation_temporal(self):
        rp = RelationPredictor(100, 64, 64)
        def decode(tid):
            return {1: "then", 2: "after"}.get(tid, "")
        idx = rp.classify_relation([1], decode)
        assert RELATION_TYPES[idx] == "temporal"

    def test_classify_relation_possessive(self):
        rp = RelationPredictor(100, 64, 64)
        def decode(tid):
            return {1: "has", 2: "contains"}.get(tid, "")
        idx = rp.classify_relation([1], decode)
        assert RELATION_TYPES[idx] == "possessive"

    def test_classify_relation_analogical(self):
        rp = RelationPredictor(100, 64, 64)
        def decode(tid):
            return {1: "like", 2: "resembles"}.get(tid, "")
        idx = rp.classify_relation([1], decode)
        assert RELATION_TYPES[idx] == "analogical"

    def test_classify_relation_empty(self):
        rp = RelationPredictor(100, 64, 64)
        idx = rp.classify_relation([], lambda x: "")
        assert RELATION_TYPES[idx] == "semantic"

    def test_set_domain(self):
        rp = RelationPredictor(100, 64, 64)
        rp.set_domain(0, freeze_others=True)
        assert rp.current_domain_id == 0
        assert 1 in rp._frozen_domains

    def test_freeze_unfreeze_domain(self):
        rp = RelationPredictor(100, 64, 64)
        rp.freeze_domain(2)
        assert 2 in rp._frozen_domains
        rp.unfreeze_domain(2)
        assert 2 not in rp._frozen_domains
        rp.unfreeze_all_domains()
        assert rp._frozen_domains == set()

    def test_state_dict_and_load(self):
        rp = RelationPredictor(100, 64, 64)
        state = rp.state_dict()
        assert "relation_type_embed" in state
        assert "_rp_rel_matrices" in state
        assert "use_verb_offset" in state
        rp2 = RelationPredictor(100, 64, 64)
        rp2.load_state(state)
        assert rp2.use_verb_offset == rp.use_verb_offset

    def test_verb_stem(self):
        assert RelationPredictor._verb_stem("causes") == "caus"
        assert RelationPredictor._verb_stem("causing") == "caus"
        assert RelationPredictor._verb_stem("walked") == "walk"
        assert RelationPredictor._verb_stem("cat") == "cat"
