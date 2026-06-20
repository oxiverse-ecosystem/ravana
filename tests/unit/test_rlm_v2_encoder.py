"""Tests for rlm_v2_encoder methods — encoder forward, backward, contrastive gradients."""

import pytest
import numpy as np
from ravana_ml.nn.rlm_v2_encoder import EncoderMixin


class _MockEncoderModel(EncoderMixin):
    """Minimal mock with just encoder parameters."""
    def __init__(self):
        self.embed_dim = 8
        self.vocab_size = 10
        self.latent_dim = 4
        self.hidden_dim = 16

        # Encoder weights
        self._enc_W1 = np.random.randn(self.hidden_dim, self.embed_dim).astype(np.float32) * 0.1
        self._enc_b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self._enc_W2 = np.random.randn(self.latent_dim, self.hidden_dim).astype(np.float32) * 0.1
        self._enc_b2 = np.zeros(self.latent_dim, dtype=np.float32)

        # Decoder weights (for autoencoder path)
        self._dec_W1 = np.random.randn(self.hidden_dim, self.latent_dim).astype(np.float32) * 0.1
        self._dec_b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self._dec_W2 = np.random.randn(self.embed_dim, self.hidden_dim).astype(np.float32) * 0.1
        self._dec_b2 = np.zeros(self.embed_dim, dtype=np.float32)

        # Momentum buffers
        self._enc_mW1 = np.zeros_like(self._enc_W1)
        self._enc_mb1 = np.zeros_like(self._enc_b1)
        self._enc_mW2 = np.zeros_like(self._enc_W2)
        self._enc_mb2 = np.zeros_like(self._enc_b2)
        self._rp_momentum = 0.9

        # Token embeddings
        self.token_embed = type('MockEmbed', (), {
            'weight': type('MockWeight', (), {
                'data': np.random.randn(self.vocab_size, self.embed_dim).astype(np.float32) * 0.1
            })()
        })()

        # Contrastive learning
        self.neg_sample_size = 5
        self.semantic_pairs = []
        self._tokenizer = None
        self._rp_use_encoder_latent = True
        self._token_embed_norms = None
        self._tokenizer_val = None

    def mark_alignment_needed(self):
        pass


class TestEncoderMixin:
    """Test encoder forward/backward methods."""

    def test_encoder_forward_full_shape(self):
        model = _MockEncoderModel()
        X = np.random.randn(5, model.embed_dim).astype(np.float32)
        latent, z1, h1, z2 = model._encoder_forward_full(X)
        assert latent.shape == (5, model.latent_dim)
        assert z1.shape == (5, model.hidden_dim)
        assert h1.shape == (5, model.hidden_dim)
        assert z2.shape == (5, model.latent_dim)

    def test_encoder_forward_flat(self):
        model = _MockEncoderModel()
        X = np.random.randn(model.embed_dim).astype(np.float32)
        latent, z1, h1, z2 = model._encoder_forward_full(X)
        assert latent.shape == (model.latent_dim,)
        assert z1.shape == (model.hidden_dim,)

    def test_encoder_forward_tanh_activation(self):
        """h1 (tanh(z1)) and latent (tanh(z2)) should be in [-1.0, 1.0]."""
        model = _MockEncoderModel()
        X = np.random.randn(10, model.embed_dim).astype(np.float32) * 5.0
        latent, z1, h1, z2 = model._encoder_forward_full(X)
        # h1 is tanh(z1) -> bounded [-1, 1]
        assert np.all(h1 >= -1.0 - 1e-7) and np.all(h1 <= 1.0 + 1e-7), \
            f"h1 out of range: min={h1.min()}, max={h1.max()}"
        # latent is tanh(z2) -> bounded [-1, 1]
        assert np.all(latent >= -1.0 - 1e-7) and np.all(latent <= 1.0 + 1e-7), \
            f"latent out of range: min={latent.min()}, max={latent.max()}"
        # z2 (pre-activation) can be outside [-1, 1] — this is expected since it's linear

    def test_encoder_backward_shape(self):
        model = _MockEncoderModel()
        X = np.random.randn(1, model.embed_dim).astype(np.float32)
        latent, z1, h1, z2 = model._encoder_forward_full(X)
        d_h2 = np.random.randn(1, model.latent_dim).astype(np.float32)
        dW1, db1, dW2, db2 = model._encoder_backward(X, z1, h1, z2, latent, d_h2)
        assert dW1.shape == model._enc_W1.shape
        assert db1.shape == model._enc_b1.shape
        assert dW2.shape == model._enc_W2.shape
        assert db2.shape == model._enc_b2.shape

    def test_encoder_backward_finite(self):
        model = _MockEncoderModel()
        X = np.random.randn(1, model.embed_dim).astype(np.float32)
        latent, z1, h1, z2 = model._encoder_forward_full(X)
        d_h2 = np.random.randn(1, model.latent_dim).astype(np.float32)
        dW1, db1, dW2, db2 = model._encoder_backward(X, z1, h1, z2, latent, d_h2)
        for g in [dW1, db1, dW2, db2]:
            assert np.all(np.isfinite(g)), f"Non-finite gradient: {g}"

    def test_contrastive_gradients_empty_pairs(self):
        model = _MockEncoderModel()
        dW1, db1, dW2, db2, loss = model._compute_contrastive_gradients()
        assert np.all(dW1 == 0)
        assert np.all(db1 == 0)
        assert np.all(dW2 == 0)
        assert np.all(db2 == 0)
        assert loss == 0.0

    def test_encoder_backward_flat_input(self):
        """Backward should work with 1D input (reshaped to (1, dim))."""
        model = _MockEncoderModel()
        X = np.random.randn(model.embed_dim).astype(np.float32)
        # Simulate what would happen: flatten 1D to (1, dim)
        X_batch = X[np.newaxis, :]
        latent, z1, h1, z2 = model._encoder_forward_full(X_batch)
        assert latent.shape == (1, model.latent_dim)
        d_h2 = np.random.randn(1, model.latent_dim).astype(np.float32)
        dW1, db1, dW2, db2 = model._encoder_backward(X_batch, z1, h1, z2, latent, d_h2)
        assert dW1.shape == model._enc_W1.shape
