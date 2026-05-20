"""
Baselines for RLM vs LLM experiments.
- SimpleMLP: NumPy 2-layer MLP with backprop (same param count as RLM)
- FrozenLLM: Simulates a frozen pre-trained LLM (uniform random logits — cannot learn)
"""

import numpy as np
import time
from typing import List, Tuple, Optional


class SimpleMLP:
    """2-layer MLP with ReLU + cross-entropy loss + SGD backprop.
    Same parameter budget as RLM (~100k params for embed_dim=32, n_hidden=32, vocab=256).
    """

    def __init__(self, vocab_size: int, embed_dim: int, n_hidden: int, lr: float = 0.01):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_hidden = n_hidden
        self.lr = lr

        # Kaiming init
        self.embed = np.random.randn(vocab_size, embed_dim).astype(np.float32) * np.sqrt(2.0 / embed_dim)
        self.W1 = np.random.randn(embed_dim, n_hidden).astype(np.float32) * np.sqrt(2.0 / embed_dim)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = np.random.randn(n_hidden, vocab_size).astype(np.float32) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(vocab_size, dtype=np.float32)

        # Cache for backprop
        self._cache = {}

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / (e.sum(axis=-1, keepdims=True) + 1e-10)

    def _cross_entropy(self, probs: np.ndarray, targets: np.ndarray) -> float:
        eps = 1e-10
        log_probs = np.log(probs + eps)
        if targets.ndim == 1:
            # targets is class indices
            return -np.mean(log_probs[np.arange(len(targets)), targets])
        else:
            # targets is one-hot
            return -np.mean(np.sum(targets * log_probs, axis=-1))

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Forward pass. token_ids: (batch,) or (batch, seq_len). Returns (batch, vocab_size) logits."""
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]

        # Embed and mean-pool over sequence
        embedded = self.embed[token_ids]  # (batch, seq_len, embed_dim)
        x = embedded.mean(axis=1)  # (batch, embed_dim)

        # Hidden layer
        h_pre = x @ self.W1 + self.b1
        h = np.maximum(0, h_pre)  # ReLU

        # Output layer
        logits = h @ self.W2 + self.b2

        self._cache = {'x': x, 'h_pre': h_pre, 'h': h, 'logits': logits, 'token_ids': token_ids}
        return logits

    def train_step(self, token_ids: np.ndarray, targets: np.ndarray) -> float:
        """One training step: forward + loss + backprop + update. Returns loss."""
        logits = self.forward(token_ids)
        probs = self._softmax(logits)
        loss = self._cross_entropy(probs, targets)

        batch_size = logits.shape[0]

        # Backprop: d_logits
        d_logits = probs.copy()
        if targets.ndim == 1:
            d_logits[np.arange(batch_size), targets] -= 1.0
        else:
            d_logits -= targets
        d_logits /= batch_size

        # d_W2, d_b2
        h = self._cache['h']
        d_W2 = h.T @ d_logits
        d_b2 = d_logits.sum(axis=0)

        # d_h
        d_h = d_logits @ self.W2.T

        # ReLU gradient
        d_h_pre = d_h * (self._cache['h_pre'] > 0).astype(np.float32)

        # d_W1, d_b1
        x = self._cache['x']
        d_W1 = x.T @ d_h_pre
        d_b1 = d_h_pre.sum(axis=0)

        # SGD update
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1

        return loss

    def predict(self, token_ids: np.ndarray) -> np.ndarray:
        """Predict logits for token_ids."""
        return self.forward(token_ids)

    def param_count(self) -> int:
        return (self.embed.size + self.W1.size + self.b1.size +
                self.W2.size + self.b2.size)

    def save(self, path: str):
        np.savez(path, embed=self.embed, W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2)

    def load(self, path: str):
        data = np.load(path)
        self.embed = data['embed']
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']


class FrozenLLM:
    """Simulates a frozen pre-trained LLM.
    Returns uniform random logits — cannot learn from new examples.
    This represents the fundamental limitation: pre-trained weights are frozen.
    """

    def __init__(self, vocab_size: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.rng = np.random.RandomState(seed)
        # Fixed "frozen" weights — random projection
        self.fixed_embed = self.rng.randn(vocab_size, 32).astype(np.float32) * 0.1
        self.fixed_W = self.rng.randn(32, vocab_size).astype(np.float32) * 0.1

    def predict(self, token_ids: np.ndarray) -> np.ndarray:
        """Returns fixed logits — cannot change regardless of input."""
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]
        embedded = self.fixed_embed[token_ids].mean(axis=1)
        return embedded @ self.fixed_W

    def train_step(self, token_ids: np.ndarray, targets: np.ndarray) -> float:
        """No-op — frozen models don't learn. Returns 0."""
        return 0.0

    def param_count(self) -> int:
        return self.fixed_embed.size + self.fixed_W.size


def measure_time_and_memory(fn, *args, **kwargs):
    """Measure wall-clock time of a function call. Returns (result, time_ms)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms
