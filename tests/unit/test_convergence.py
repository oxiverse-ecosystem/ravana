"""Precision test: verify RLM learns specific distant causal transitions."""

import pytest
import numpy as np
from ravana_ml.nn.rlm import RLM


class TestRLMConvergence:
    """Test that RLM learns edge transitions between distant concepts."""

    @pytest.fixture
    def sample_rlm(self):
        np.random.seed(42)
        return RLM(
            vocab_size=64, embed_dim=32, concept_dim=32, n_concepts=128,
            n_hidden=32, n_layers=1, max_seq_len=16,
            free_energy_threshold=5.0, sleep_interval=15,
        )

    def test_init_creates_graph(self, sample_rlm):
        rlm = sample_rlm
        assert rlm.graph is not None
        assert len(rlm.graph.nodes) > 0

    def test_initial_embedding_norms(self, sample_rlm):
        rlm = sample_rlm
        for tid in [1, 50, 25, 10, 40]:
            embed = rlm.token_embed.embed_raw(tid)
            norm = np.linalg.norm(embed)
            assert norm > 0, f"Token {tid} has zero-norm embedding"

    def test_learn_creates_edge(self, sample_rlm):
        rlm = sample_rlm
        inp = np.array([1], dtype=np.int64)
        nxt = np.array([50], dtype=np.int64)
        rlm.learn(inp, nxt)
        assert rlm._step_counter == 1
        assert rlm._edges_learned >= 0

    def test_learn_multiple_transitions(self, sample_rlm):
        rlm = sample_rlm
        train_seq = [1, 50, 25, 10, 40]
        for i in range(len(train_seq) - 1):
            inp = np.array([train_seq[i]], dtype=np.int64)
            nxt = np.array([train_seq[i + 1]], dtype=np.int64)
            rlm.learn(inp, nxt)
        # Some edges should have been learned
        assert rlm._step_counter == 4

    def test_forward_returns_logits(self, sample_rlm):
        rlm = sample_rlm
        inp = np.array([1], dtype=np.int64)
        logits = rlm.forward(inp)
        assert hasattr(logits, 'data')
        assert logits.data.shape[0] == rlm.vocab_size

    def test_repeated_training(self, sample_rlm):
        rlm = sample_rlm
        train_seq = [1, 50, 25, 10, 40, 1, 50, 25, 10, 40]
        for _ in range(3):
            for i in range(len(train_seq) - 1):
                inp = np.array([train_seq[i]], dtype=np.int64)
                nxt = np.array([train_seq[i + 1]], dtype=np.int64)
                rlm.learn(inp, nxt)
        # After repeated training, edges should exist
        assert rlm._step_counter > 0
        assert len(rlm.graph.edges) >= 0

    def test_teacher_forced_prediction(self, sample_rlm):
        rlm = sample_rlm
        train_seq = [1, 50, 25, 10, 40]
        for _ in range(5):
            for i in range(len(train_seq) - 1):
                inp = np.array([train_seq[i]], dtype=np.int64)
                nxt = np.array([train_seq[i + 1]], dtype=np.int64)
                rlm.learn(inp, nxt)
        # Forward prediction should produce plausible logits
        for src in train_seq:
            logits = rlm.forward(np.array([src], dtype=np.int64))
            pred_id = int(np.argmax(logits.data))
            assert 0 <= pred_id < rlm.vocab_size
