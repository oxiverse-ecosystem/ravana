"""Tests for rlm_v2_sleep methods — sleep cycle, phantom pruning, memory consolidation."""

import pytest
import numpy as np
import time
from ravana_ml.nn.rlm_v2_sleep import SleepMixin
from ravana_ml.graph import ConceptGraph, ConceptBindingMap


class _MockSleepModel(SleepMixin):
    def __init__(self):
        self.concept_dim = 8
        self.embed_dim = 8
        self.latent_dim = 4
        self.hidden_dim = 16

        self.graph = ConceptGraph(dim=self.concept_dim, max_nodes=100)
        self.binding_map = ConceptBindingMap()

        self._episodic_triples = []
        self._semantic_memories = {}
        self._semantic_memory_max = 100
        self._episodic_buffer = []
        self._episodic_buffer_max = 100
        self._replay_buffer = []
        self._replay_buffer_max = 1000
        self._replay_n_samples = 20
        self._step_counter = 0
        self._last_sleep_step = 0
        self._in_sleep_cycle = False
        self._sleep_pressure = 0.0
        self.alignment_needed = False
        self.wake_epochs_since_sleep = 0
        self.sleep_every_n_wake_epochs = 10
        self.sleep_cycles_completed = 0
        self._last_aligned_version = -1

        self._enc_W1 = np.random.randn(self.hidden_dim, self.embed_dim).astype(np.float32) * 0.1
        self._enc_b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self._enc_W2 = np.random.randn(self.latent_dim, self.hidden_dim).astype(np.float32) * 0.1
        self._enc_b2 = np.zeros(self.latent_dim, dtype=np.float32)
        self._enc_mW1 = np.zeros_like(self._enc_W1)
        self._enc_mb1 = np.zeros_like(self._enc_b1)
        self._enc_mW2 = np.zeros_like(self._enc_W2)
        self._enc_mb2 = np.zeros_like(self._enc_b2)
        self._rp_momentum = 0.9
        self._tokenizer = None
        self._token_embed_norms = None
        self.alignment_edge_threshold = 0.3
        self.alignment_lr = 0.005
        self.alignment_margin = 0.15
        self.lambda_anchor = 0.05
        self.lambda_recon = 0.0
        self.max_alignment_epochs = 10

        self.vocab_size = 10
        self.token_embed = type('MockEmbed', (), {
            'weight': type('MockWeight', (), {
                'data': np.random.randn(self.vocab_size, self.embed_dim).astype(np.float32) * 0.1
            })()
        })()

        self.currencies = type('MockCurrencies', (), {
            'consolidate_on_sleep': lambda self: None,
            'regulate': lambda self: None,
        })()

        self._node_matrix_version = -1
        self._node_matrix_cache = None
        self._rel_vector_version = -1
        self._rel_vector_cache = {}
        self._norm_cache = {}
        self._last_hidden_state = None
        self._last_predicted_concepts = set()
        self.valence = 0.5
        self.arousal = 0.3
        self._domain_memories = {}

    def _normalize_outgoing_weights(self, budget=5.0):
        pass

    def _prune_weak_edges(self, threshold=0.1):
        pass

    def _anti_hebbian_prune_polluted_edges(self):
        return 0

    def _invalidate_caches(self):
        self._node_matrix_version = -1

    def _bridge_memories_to_graph(self):
        pass

    def _replay_old_memories(self, n_samples=20):
        pass

    def _regulate_cognitive_state(self):
        pass


class TestSleepCycle:
    def test_sleep_cycle_basic(self):
        model = _MockSleepModel()
        model.sleep_cycle()
        assert model.sleep_cycles_completed == 1
        assert model._in_sleep_cycle is False

    def test_sleep_cycle_idempotent(self):
        model = _MockSleepModel()
        model._in_sleep_cycle = True
        model.sleep_cycle()
        assert model.sleep_cycles_completed == 0

    def test_sleep_cycle_resets_pressure(self):
        model = _MockSleepModel()
        model._sleep_pressure = 0.8
        model.sleep_cycle()
        assert model._sleep_pressure == 0.0

    def test_sleep_cycle_increments_counter(self):
        model = _MockSleepModel()
        assert model.sleep_cycles_completed == 0
        model.sleep_cycle()
        assert model.sleep_cycles_completed == 1


class TestPhantomNodePruning:
    def test_prune_no_phantoms(self):
        model = _MockSleepModel()
        n = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        model.binding_map.bind(0, n.id, confidence=0.9)
        model._prune_phantom_nodes(min_degree=2)
        assert model.graph.get_node(n.id) is not None

    def test_phantom_removal(self):
        model = _MockSleepModel()
        n = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        model._prune_phantom_nodes(min_degree=2)
        assert model.graph.get_node(n.id) is None


class TestEpisodicBuffer:
    def test_store_episode(self):
        model = _MockSleepModel()
        model._store_episode(error=0.2, is_correct=True)
        assert len(model._episodic_buffer) == 1

    def test_error_boosted_importance(self):
        """Error-boosted importance: incorrect predictions get higher importance."""
        model = _MockSleepModel()
        model._store_episode(error=0.8, is_correct=False)
        # importance = 1.0 - min(1.0, 0.8) = 0.2
        # not correct -> min(1.0, 0.2 + 0.3) = 0.5
        actual = model._episodic_buffer[0]['importance']
        assert actual == pytest.approx(0.5, abs=1e-10)

    def test_buffer_experience(self):
        model = _MockSleepModel()
        inp = np.array([1, 2, 3])
        tgt = np.array([4, 5, 6])
        model.buffer_experience(inp, tgt, domain="test")
        assert len(model._replay_buffer) == 1


class TestAlignment:
    def test_mark_alignment_needed(self):
        model = _MockSleepModel()
        model.alignment_needed = False
        model.mark_alignment_needed()
        assert model.alignment_needed is True

    def test_compute_neighbor_recall_empty(self):
        model = _MockSleepModel()
        assert model.compute_neighbor_recall_at_5() == 0.0

    def test_end_wake_epoch(self):
        model = _MockSleepModel()
        model.sleep_every_n_wake_epochs = 3
        assert model.wake_epochs_since_sleep == 0
        model.end_wake_epoch()
        assert model.wake_epochs_since_sleep == 1


class TestMemoryConsolidation:
    def test_consolidate_adds_semantic(self):
        """_consolidate_episodic_to_semantic should add concepts to semantic_memories."""
        model = _MockSleepModel()
        cid1 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        cid2 = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        model._episodic_buffer.append({
            'correct': True,
            'error': 0.1,
            'importance': 0.8,
            'concepts': [cid1.id, cid2.id],
            'consolidation_state': 'fresh',
            'timestamp': time.time(),
        })
        # Call the real implementation (not mock override)
        SleepMixin._consolidate_episodic_to_semantic(model)
        assert cid1.id in model._semantic_memories
        assert cid2.id in model._semantic_memories

    def test_decay_semantic_memories(self):
        """_decay_semantic_memories should reduce strength over time."""
        model = _MockSleepModel()
        cid = model.graph.add_node(vector=np.random.randn(model.concept_dim).astype(np.float32))
        model._semantic_memories[cid.id] = {
            'strength': 0.8,
            'access_count': 10,
            'last_access': time.time() - 100,
        }
        SleepMixin._decay_semantic_memories(model)
        assert cid.id in model._semantic_memories
        assert model._semantic_memories[cid.id]['strength'] < 0.8
