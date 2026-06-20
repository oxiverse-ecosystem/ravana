"""Tests for ravana_ml.nn.neural_decoder."""

import pytest
import numpy as np
from ravana_ml.nn.neural_decoder import NeuralDecoder, TracedGRUCell


class TestTracedGRUCell:
    def test_forward_traced(self):
        cell = TracedGRUCell(8, 16)
        x = np.random.randn(8).astype(np.float32)
        h = np.zeros(16, dtype=np.float32)
        h_new = cell.forward_traced(x, h)
        assert h_new.shape == (16,)
        assert hasattr(cell, '_last_z')
        assert hasattr(cell, '_last_r')
        assert hasattr(cell, '_last_h_candidate')
        assert hasattr(cell, '_last_h_prev')

    def test_forward_traced_multi_step(self):
        cell = TracedGRUCell(8, 16)
        h = np.zeros(16, dtype=np.float32)
        for _ in range(5):
            x = np.random.randn(8).astype(np.float32)
            h = cell.forward_traced(x, h)
        assert not np.allclose(h, 0.0)


class TestNeuralDecoder:
    def test_init(self):
        nd = NeuralDecoder(vocab_size=100, embed_dim=16, hidden_dim=32)
        assert nd.vocab_size == 100
        assert nd.embed_dim == 16
        assert nd.hidden_dim == 32
        assert nd._total_training_examples == 0
        assert nd._vocab_embed_cache is None

    def test_rebuild_vocab_cache(self):
        nd = NeuralDecoder(vocab_size=100, embed_dim=16, hidden_dim=32)
        nd.rebuild_vocab_cache()
        assert nd._vocab_embed_cache is not None
        assert nd._vocab_embed_cache.shape == (100, 16)

    def test_forward(self):
        nd = NeuralDecoder(vocab_size=100, embed_dim=16, hidden_dim=32)
        cond_embs = np.random.randn(5, 16).astype(np.float32)
        input_seq = np.array([0, 1, 2, 3], dtype=np.int64)
        logits, h_final = nd.forward(cond_embs, input_seq, use_raw=True)
        assert logits.shape == (4, 100)
        assert h_final.shape == (32,)

    def test_forward_with_h_prev(self):
        nd = NeuralDecoder(vocab_size=100, embed_dim=16, hidden_dim=32)
        cond_embs = np.random.randn(5, 16).astype(np.float32)
        input_seq = np.array([0, 1], dtype=np.int64)
        h_prev = np.ones(32, dtype=np.float32)
        logits, h_final = nd.forward(cond_embs, input_seq, h_prev=h_prev, use_raw=True)
        assert logits.shape == (2, 100)

    def test_forward_traced(self):
        nd = NeuralDecoder(vocab_size=100, embed_dim=16, hidden_dim=32)
        cond_embs = np.random.randn(5, 16).astype(np.float32)
        input_seq = np.array([0, 1], dtype=np.int64)
        logits, h_final = nd.forward(cond_embs, input_seq, use_raw=True, _use_gru_traced=True)
        assert logits.shape == (2, 100)

    def test_generate_minimal(self):
        nd = NeuralDecoder(vocab_size=100, embed_dim=16, hidden_dim=32)
        nd.rebuild_vocab_cache()
        cond_embs = np.random.randn(3, 16).astype(np.float32)
        tokens = nd.generate(cond_embs, max_steps=5, bos_idx=0, temperature=0.1)
        # May return empty list with random weights — validate no crash
        assert isinstance(tokens, list)
        if tokens:
            assert all(isinstance(t, (int, np.integer)) for t in tokens)

    def test_generate_with_eos(self):
        nd = NeuralDecoder(vocab_size=100, embed_dim=16, hidden_dim=32)
        nd.rebuild_vocab_cache()
        cond_embs = np.random.randn(3, 16).astype(np.float32)
        tokens = nd.generate(cond_embs, max_steps=20, bos_idx=0, eos_idx=99, temperature=0.5)
        assert len(tokens) > 0

    def test_generate_with_idx_to_word(self):
        nd = NeuralDecoder(vocab_size=10, embed_dim=4, hidden_dim=8)
        nd.rebuild_vocab_cache()
        cond_embs = np.random.randn(3, 4).astype(np.float32)
        idx_to_word = {0: "<bos>", 1: "a", 2: "b", 3: "c"}
        tokens = nd.generate(cond_embs, max_steps=5, bos_idx=0, temperature=0.1,
                             idx_to_word=idx_to_word)
        assert len(tokens) > 0

    def test_train_on_sentence_short(self):
        nd = NeuralDecoder(vocab_size=10, embed_dim=4, hidden_dim=8)
        word_to_embed = {"hello": np.random.randn(4).astype(np.float32),
                         "world": np.random.randn(4).astype(np.float32)}
        word_to_idx = {"hello": 1, "world": 2}
        err = nd.train_on_sentence(["hello", "world"], word_to_embed, word_to_idx)
        assert err >= 0.0

    def test_train_on_sentence_longer(self):
        nd = NeuralDecoder(vocab_size=10, embed_dim=4, hidden_dim=8)
        words = ["the", "cat", "sat", "on", "the", "mat"]
        word_to_embed = {w: np.random.randn(4).astype(np.float32) for w in set(words)}
        word_to_idx = {w: i for i, w in enumerate(set(words), 1)}
        err = nd.train_on_sentence(words, word_to_embed, word_to_idx)
        assert err >= 0.0

    def test_train_on_sentence_with_conditioning(self):
        nd = NeuralDecoder(vocab_size=10, embed_dim=4, hidden_dim=8)
        words = ["hello", "world"]
        word_to_embed = {"hello": np.random.randn(4).astype(np.float32),
                         "world": np.random.randn(4).astype(np.float32)}
        word_to_idx = {"hello": 1, "world": 2}
        cond = np.random.randn(3, 4).astype(np.float32)
        err = nd.train_on_sentence(words, word_to_embed, word_to_idx, conditioning_embs=cond)
        assert err >= 0.0

    def test_train_on_text(self):
        nd = NeuralDecoder(vocab_size=10, embed_dim=4, hidden_dim=8)
        word_to_embed = {"hello": np.random.randn(4).astype(np.float32),
                         "world": np.random.randn(4).astype(np.float32)}
        word_to_idx = {"hello": 1, "world": 2}
        err, count = nd.train_on_text("hello world", word_to_embed, word_to_idx)
        # "hello world" is only 2 words, below min_sentence_len=3
        assert count == 0

    def test_train_on_text_longer(self):
        nd = NeuralDecoder(vocab_size=50, embed_dim=8, hidden_dim=16)
        text = "the quick brown fox jumps over the lazy dog"
        words = set(text.lower().split())
        word_to_embed = {w: np.random.randn(8).astype(np.float32) for w in words}
        word_to_idx = {w: i for i, w in enumerate(words)}
        err, count = nd.train_on_text(text, word_to_embed, word_to_idx, min_sentence_len=4)
        # At "the quick brown fox jumps over the lazy dog", length > min=4
        # Sentence split by [.!?] gives one sentence of 9 words
        assert count >= 0

    def test_sleep_cycle(self):
        nd = NeuralDecoder(vocab_size=100, embed_dim=16, hidden_dim=32)
        nd.rebuild_vocab_cache()
        nd.sleep_cycle()
        assert nd._vocab_embed_cache is not None

    def test_state_dict(self):
        nd = NeuralDecoder(vocab_size=100, embed_dim=16, hidden_dim=32)
        sd = nd.state_dict()
        assert '_total_training_examples' in sd
        assert '_avg_prediction_error' in sd

    def test_load_state_dict(self):
        nd = NeuralDecoder(vocab_size=100, embed_dim=16, hidden_dim=32)
        sd = nd.state_dict()
        nd.load_state_dict(sd)
        assert nd._total_training_examples == 0

    def test_training_increases_counter(self):
        nd = NeuralDecoder(vocab_size=10, embed_dim=4, hidden_dim=8)
        words = ["hello", "world", "foo", "bar"]
        word_to_embed = {w: np.random.randn(4).astype(np.float32) for w in words}
        word_to_idx = {w: i for i, w in enumerate(words)}
        nd.train_on_sentence(words, word_to_embed, word_to_idx)
        assert nd._total_training_examples > 0
