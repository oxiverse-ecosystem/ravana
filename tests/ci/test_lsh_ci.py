"""CI-critical tests for the LSH token-scoring invariants (P3 Sprint 7).

These are the regression guards that MUST pass on every push — they encode the
four properties that the hardened LSH implementation guarantees:

  1. Training bypass: LSH never masks logits during training (dense loss needs
     the full vocabulary distribution).
  2. Recall floor: LSH never silently over-prunes to a tiny candidate set.
  3. Clean fallback: when LSH is unsuitable (tiny vocab, sparse buckets), it
     defers to full scoring instead of producing degenerate output.
  4. Determinism + int32 safety: hashing is reproducible and configs that would
     overflow the bucket id space are rejected.

Marked ``@pytest.mark.ci`` so the ``--ci`` filter in tests/ci/conftest.py picks
them up in the critical-path job of the GitHub Actions workflow.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.ci

from ravana_ml.nn.rlm_v2_rp import RPMixin
from ravana_ml.nn.rlm_v2_verb import VerbMixin


class _LSHCIMock(RPMixin):
    """Bare mock: just enough attributes for the LSH mixin + _rp_forward."""
    def __init__(self, vocab_size=500, latent_dim=96):
        self.vocab_size = vocab_size
        self.embed_dim = latent_dim
        self.latent_dim = latent_dim
        self._rp_use_encoder_latent = False
        self.training = True
        self.current_domain_id = 0
        self.num_domains = 3
        self.use_verb_offset = False
        self._rp_lr = 0.01
        self._rp_momentum = 0.9
        self.rp_scale = 1.0
        self._entity_adapter_lr = 0.01
        self._entity_adapter_momentum = 0.9
        self._entity_adapter_rank = 4
        self._entity_adapters = {}
        self._entity_adapter_momentums = {}
        self._verb_offsets = {0: {}}
        self._verb_offset_count = {0: {}}
        self._verb_offset_variance = {0: {}}
        self._rp_cache = None
        self.freeze_token_embeds_in_rp = True
        self._test_time_adapt_mode = False
        rng = np.random.RandomState(0)
        data = rng.randn(vocab_size, latent_dim).astype(np.float32)
        self.token_embed = type('E', (), {'weight': type('W', (), {'data': data})()})()
        self._token_embed_norms = None
        n_rel, rank = 6, 8
        self._rp_low_rank_rank = rank
        self._rp_W_base = np.ones((n_rel, latent_dim), dtype=np.float32)
        self._rp_U_d = np.zeros((self.num_domains, n_rel, rank, latent_dim), dtype=np.float32)
        self._rp_V_d = np.zeros((self.num_domains, n_rel, rank, latent_dim), dtype=np.float32)
        self._rp_mW_base = np.zeros_like(self._rp_W_base)
        self._rp_mU_d = np.zeros_like(self._rp_U_d)
        self._rp_mV_d = np.zeros_like(self._rp_V_d)

    def _encoder_forward_full(self, X):
        return X, None, None, None

    def get_robust_embedding(self, tid):
        return self.token_embed.weight.data[tid]

    def _verb_stem(self, word):
        return VerbMixin._verb_stem(word)

    def _rp_forward_verb_offset(self, subject_tid, verb_word, return_count=False):
        return (None, 0, 0.0) if return_count else None

    def _get_or_adapt_entity_adapter(self, subject_tid, verb_word=None):
        if subject_tid not in self._entity_adapters:
            r = self._entity_adapter_rank
            U = np.zeros((r, self.latent_dim), dtype=np.float32)
            V = np.zeros((r, self.latent_dim), dtype=np.float32)
            self._entity_adapters[subject_tid] = (U, V)
            self._entity_adapter_momentums[subject_tid] = (np.zeros_like(U), np.zeros_like(V))
        return self._entity_adapters[subject_tid]


class TestLSHTrainingBypass:
    def test_lsh_defers_to_full_scoring_in_training(self):
        """The softmax loss needs the full-vocab distribution — LSH must not mask it."""
        model = _LSHCIMock()
        model._lsh_init()
        model.training = True
        src = model.token_embed.weight.data[0]
        logits, info = model._lsh_scoring(
            src, model.token_embed.weight.data,
            np.eye(model.latent_dim, dtype=np.float32), training=True,
        )
        assert logits is None and info is None

    def test_forward_produces_all_finite_logits_in_training(self):
        """End-to-end: a training forward pass must have zero -inf values."""
        model = _LSHCIMock(vocab_size=200, latent_dim=96)
        model._lsh_init()
        model.training = True
        logits = model._rp_forward(subject_tid=0, rel_type_idx=0)
        assert np.all(np.isfinite(logits)), (
            "Training forward leaked LSH masking (-inf present); the softmax "
            "loss would be corrupted."
        )


class TestLSHRecallFloor:
    def test_never_over_prunes_below_floor(self):
        """Regression guard: finite-logit count must meet min_candidates or fall back.

        The pre-hardening bug kept only 1-2 tokens finite (96%+ pruned).
        """
        model = _LSHCIMock(vocab_size=5000, latent_dim=96)
        model._lsh_init(n_buckets=8, n_hashes=4, min_candidates=32)
        model.training = False
        src = model.token_embed.weight.data[0]
        logits, info = model._lsh_scoring(
            src, model.token_embed.weight.data,
            np.eye(model.latent_dim, dtype=np.float32), training=False,
        )
        if logits is None:
            return  # clean fallback is acceptable
        assert int(np.sum(np.isfinite(logits))) >= 32

    def test_tiny_vocab_falls_back(self):
        """A vocab too small for LSH must defer to full scoring, not over-prune."""
        model = _LSHCIMock(vocab_size=50, latent_dim=8)
        model._lsh_init(n_buckets=8, n_hashes=4, min_candidates=32)
        model.training = False
        src = model.token_embed.weight.data[0]
        logits, info = model._lsh_scoring(
            src, model.token_embed.weight.data,
            np.eye(model.latent_dim, dtype=np.float32), training=False,
        )
        assert logits is None, "Tiny vocab must fall back to full scoring"


class TestLSHSafety:
    def test_oversized_config_rejected(self):
        """Configs whose bucket space overflows int32 must raise at init."""
        model = _LSHCIMock()
        with pytest.raises(ValueError, match="exceeds int32"):
            model._lsh_init(n_buckets=64, n_hashes=8)

    def test_masked_logits_are_neg_inf(self):
        """Non-candidate logits must be exactly -inf so exp() -> 0 probability."""
        model = _LSHCIMock(vocab_size=2000, latent_dim=96)
        model._lsh_init(n_buckets=8, n_hashes=4, min_candidates=32)
        model.training = False
        src = model.token_embed.weight.data[0]
        logits, info = model._lsh_scoring(
            src, model.token_embed.weight.data,
            np.eye(model.latent_dim, dtype=np.float32), training=False,
        )
        if logits is None:
            pytest.skip("Fell back (acceptable)")
        masked = logits[~np.isfinite(logits)]
        assert np.all(masked == -np.inf)

    def test_hash_is_deterministic(self):
        """Same embeddings must always hash to the same buckets (reproducibility)."""
        model = _LSHCIMock()
        model._lsh_init()
        embeds = model.token_embed.weight.data
        np.testing.assert_array_equal(model._lsh_hash(embeds), model._lsh_hash(embeds))
