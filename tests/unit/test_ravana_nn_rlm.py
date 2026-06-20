"""Tests for ravana/src/ravana/nn/rlm modules: RelationPredictor, PropagationEngine, Plasticity."""

import pytest
import numpy as np
from ravana.nn.rlm.relation_predictor import RelationPredictor, RELATION_TYPES, _KEYWORD_MAP
from ravana.nn.rlm.propagation import PropagationEngine
from ravana.nn.rlm.plasticity import Plasticity
from ravana_ml.graph import ConceptGraph


# ── RelationPredictor Tests ──

class TestRELATION_TYPES:
    def test_relation_types_defined(self):
        assert "causal" in RELATION_TYPES
        assert "semantic" in RELATION_TYPES
        assert "temporal" in RELATION_TYPES
        assert len(RELATION_TYPES) >= 4


class TestRelationPredictor:
    def test_default_init(self):
        rp = RelationPredictor(vocab_size=100, embed_dim=8, concept_dim=8)
        assert rp.vocab_size == 100
        assert rp.embed_dim == 8
        assert rp.concept_dim == 8
        assert rp._verb_offsets == {}
        assert rp.use_verb_offset is False
        assert rp.use_cross_domain_alignment is False

    def test_verb_stem(self):
        assert RelationPredictor._verb_stem("causes") == "caus"
        assert RelationPredictor._verb_stem("liked") == "lik"
        assert RelationPredictor._verb_stem("playing") == "play"
        assert RelationPredictor._verb_stem("makes") == "mak"
        assert RelationPredictor._verb_stem("run") == "run"  # too short to strip

    def test_classify_relation_causal(self):
        rp = RelationPredictor(100, 8, 8)
        mock_decode = lambda tid: "causes"
        idx = rp.classify_relation([1], mock_decode)
        assert idx == RELATION_TYPES.index("causal")

    def test_classify_relation_semantic_default(self):
        rp = RelationPredictor(100, 8, 8)
        mock_decode = lambda tid: "unknownwordxyz"
        idx = rp.classify_relation([1], mock_decode)
        assert idx == RELATION_TYPES.index("semantic")

    def test_classify_relation_temporal(self):
        rp = RelationPredictor(100, 8, 8)
        mock_decode = lambda tid: "before"
        idx = rp.classify_relation([1], mock_decode)
        assert idx == RELATION_TYPES.index("temporal")

    def test_classify_relation_possessive(self):
        rp = RelationPredictor(100, 8, 8)
        mock_decode = lambda tid: "has"
        idx = rp.classify_relation([1], mock_decode)
        assert idx == RELATION_TYPES.index("possessive")

    def test_classify_relation_empty(self):
        rp = RelationPredictor(100, 8, 8)
        mock_decode = lambda tid: ""
        idx = rp.classify_relation([], mock_decode)
        assert idx == RELATION_TYPES.index("semantic")

    def test_accumulate_verb_offset(self):
        rp = RelationPredictor(100, 8, 8)
        rp.use_verb_offset = True
        mock_embed = type('MockEmbed', (), {'weight': type('MockWeight', (), {'data': np.zeros((100, 8), dtype=np.float32)})()})()
        rp.accumulate_verb_offset(0, 1, "causes", mock_embed)
        assert len(rp._verb_accum_buffer) == 1
        assert rp._verb_accum_buffer[0][0] == "caus"

    def test_accumulate_verb_offset_disabled(self):
        rp = RelationPredictor(100, 8, 8)
        rp.use_verb_offset = False
        mock_embed = type('MockEmbed', (), {'weight': type('MockWeight', (), {'data': np.zeros((100, 8))})()})()
        rp.accumulate_verb_offset(0, 1, "causes", mock_embed)
        assert len(rp._verb_accum_buffer) == 0

    def test_set_domain(self):
        rp = RelationPredictor(100, 8, 8)
        assert rp.current_domain_id is None
        rp.set_domain(0)
        assert rp.current_domain_id == 0

    def test_set_domain_none(self):
        rp = RelationPredictor(100, 8, 8)
        rp.set_domain(None)
        assert rp.current_domain_id is None

    def test_freeze_and_unfreeze(self):
        rp = RelationPredictor(100, 8, 8)
        rp.freeze_domain(0)
        assert 0 in rp._frozen_domains
        rp.unfreeze_domain(0)
        assert 0 not in rp._frozen_domains
        rp.unfreeze_all_domains()
        assert rp._frozen_domains == set()

    def test_encoder_forward_full_flat(self):
        rp = RelationPredictor(100, 8, 8, latent_dim=8)
        x = np.random.randn(8).astype(np.float32)
        latent, z1, h1, z2 = rp._encoder_forward_full(x)
        assert latent.shape[0] == 8

    def test_encoder_forward_full_batch(self):
        rp = RelationPredictor(100, 8, 8, latent_dim=8)
        X = np.random.randn(5, 8).astype(np.float32)
        latent, z1, h1, z2 = rp._encoder_forward_full(X)
        assert latent.shape == (5, 8)

    def test_state_dict_and_load(self):
        rp = RelationPredictor(100, 8, 8)
        state = rp.state_dict()
        assert '_rp_rel_matrices' in state
        assert 'relation_type_embed' in state

        rp2 = RelationPredictor(100, 8, 8)
        rp2.load_state(state)
        assert np.allclose(rp2._rp_rel_matrices, rp._rp_rel_matrices)


# ── PropagationEngine Tests ──

class TestPropagationEngine:
    @pytest.fixture
    def small_graph(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.ones(8, dtype=np.float32), label="a")
        n2 = g.add_node(vector=np.ones(8, dtype=np.float32) * 0.5, label="b")
        n3 = g.add_node(vector=np.ones(8, dtype=np.float32) * 0.2, label="c")
        g.add_edge(n1.id, n2.id, weight=0.7, relation_type="semantic")
        g.add_edge(n2.id, n3.id, weight=0.6, relation_type="causal")
        return g

    def test_default_init(self, small_graph):
        pe = PropagationEngine(small_graph)
        assert pe.graph is small_graph

    def test_spread_activation(self, small_graph):
        pe = PropagationEngine(small_graph)
        n1_id = list(small_graph.nodes.keys())[0]
        small_graph.activate(n1_id, amount=1.0)
        pe.spread_activation(steps=2, k_active=5, decay=0.5)
        # After spreading, some nodes should have activation
        active_count = sum(1 for n in small_graph.nodes.values() if n.activation > 0.01)
        assert active_count >= 1

    def test_get_prediction(self, small_graph):
        pe = PropagationEngine(small_graph)
        nids = list(small_graph.nodes.keys())
        small_graph.activate(nids[0], 1.0)
        predicted = pe.get_prediction(nids, top_k=2)
        assert len(predicted) <= 2
        assert all(nid in small_graph.nodes for nid in predicted)

    def test_n_hop_bfs(self, small_graph):
        pe = PropagationEngine(small_graph)
        nids = list(small_graph.nodes.keys())
        mock_decode = lambda tid: ""
        scores = pe.n_hop_bfs(nids[0], "semantic", max_hops=2,
                              query_verb_word="", decode_token_fn=mock_decode)
        assert isinstance(scores, dict)

    def test_direct_edge_boost(self, small_graph):
        pe = PropagationEngine(small_graph)
        nids = list(small_graph.nodes.keys())
        mock_decode = lambda tid: ""
        small_graph.activate(nids[0], 1.0)
        scores = pe.direct_edge_boost(nids[0], "semantic",
                                      query_verb_word="", decode_token_fn=mock_decode)
        assert isinstance(scores, dict)


# ── Plasticity Tests ──

class TestPlasticity:
    @pytest.fixture
    def plasticity(self):
        g = ConceptGraph(dim=8, max_nodes=100)
        n1 = g.add_node(vector=np.ones(8, dtype=np.float32), label="tok_0")
        n2 = g.add_node(vector=np.ones(8, dtype=np.float32), label="tok_1")
        n3 = g.add_node(vector=np.ones(8, dtype=np.float32), label="tok_2")
        g.add_edge(n1.id, n2.id, weight=0.5, relation_type="semantic")
        g.add_edge(n2.id, n3.id, weight=0.5, relation_type="causal")
        p = Plasticity(graph=g, base_lr=0.005)
        return p

    def test_default_init(self, plasticity):
        assert plasticity.base_lr == 0.005
        assert plasticity.sleep_cycles_completed == 0
        assert plasticity._step_counter == 0
        assert plasticity._episodic_triples == []

    def test_decompose_triple_three_ids(self, plasticity):
        subj, rel, obj = plasticity._decompose_triple([0, 1, 2])
        assert subj == [0]
        assert rel == [1]
        assert obj == [2]

    def test_decompose_triple_two_ids(self, plasticity):
        subj, rel, obj = plasticity._decompose_triple([0, 1])
        assert subj == [0]
        assert rel == []
        assert obj == [1]

    def test_decompose_triple_one_id(self, plasticity):
        subj, rel, obj = plasticity._decompose_triple([0])
        assert subj == [0]
        assert rel == []
        assert obj == []

    def test_decompose_triple_empty(self, plasticity):
        subj, rel, obj = plasticity._decompose_triple([])
        assert subj == []
        assert rel == []
        assert obj == []

    def test_identity_strength_property(self, plasticity):
        plasticity.identity_strength = 0.7
        assert plasticity.identity_strength == 0.7
        assert plasticity.currencies.identity_strength == 0.7

    def test_sleep_pressure_property(self, plasticity):
        plasticity.sleep_pressure = 0.5
        assert plasticity.sleep_pressure == 0.5
        assert plasticity.currencies.sleep_pressure == 0.5

    def test_state_dict(self, plasticity):
        state = plasticity.state_dict()
        assert "graph_nodes" in state
        assert "graph_edges" in state
        assert "_episodic_triples" in state
        assert "_step_counter" in state
        assert "_train_correct" in state

    def test_cognitive_currencies_initialized(self, plasticity):
        assert plasticity.currencies is not None
        assert plasticity.currency is not None
