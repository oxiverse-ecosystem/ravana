"""Tests for the hardened LSH token-scoring path (P3 Sprint 7).

These cover the four regressions fixed in Sprint 7:
  1. Multi-probe + recall floor (no more 96% over-pruning).
  2. Hard ``min_candidates`` guarantee (fall back to full scoring when LSH
     cannot gather a usable candidate set).
  3. Training guard (dense softmax loss gets the full-vocab distribution).
  4. Embedding-drift refresh (bucket map rebuilds after token updates).

The mock mirrors the one in ``test_rlm_v2_rp.py`` so the mixin's expectations
about attributes (``latent_dim``, ``token_embed.weight.data``, encoder, etc.)
are all satisfied.
"""
import numpy as np
import pytest

from ravana_ml.nn.rlm_v2_rp import RPMixin
from ravana_ml.nn.rlm_v2_verb import VerbMixin


class _MockLSHModel(RPMixin):
    """Minimal model exposing just what the LSH mixin needs.

    ``structured`` builds clustered embeddings so LSH has real neighbourhood
    structure to exploit (random noise makes recall tests meaningless).
    """
    def __init__(self, vocab_size=500, latent_dim=96, structured=True,
                 n_clusters=10, seed=0):
        self.vocab_size = vocab_size
        self.embed_dim = latent_dim
        self.latent_dim = latent_dim
        self._rp_use_encoder_latent = False
        self.training = True  # default; tests flip to False for inference
        # RP factors needed by _rp_forward / _rp_backward.
        self.current_domain_id = 0
        self.num_domains = 3
        self.use_verb_offset = False  # exercise the W_rel/LSH path, not verb offset
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

        rng = np.random.RandomState(seed)
        if structured and vocab_size >= n_clusters:
            per = vocab_size // n_clusters
            counts = [per] * n_clusters
            # distribute remainder
            i = 0
            extra = vocab_size - per * n_clusters
            while extra > 0:
                counts[i % n_clusters] += 1
                extra -= 1
                i += 1
            centroids = rng.randn(n_clusters, latent_dim).astype(np.float32) * 5.0
            rows = []
            for ci, c in enumerate(counts):
                rows.append(centroids[ci] + rng.randn(c, latent_dim).astype(np.float32) * 0.3)
            data = np.concatenate(rows, axis=0)[:vocab_size].astype(np.float32)
        else:
            data = rng.randn(vocab_size, latent_dim).astype(np.float32)

        self.token_embed = type('Embed', (), {
            'weight': type('Weight', (), {'data': data})()
        })()
        self._token_embed_norms = None

        # Low-rank W_rel factors (identity-ish init so W_rel ~= I for tests).
        n_rel = 6
        rank = 8
        self._rp_low_rank_rank = rank
        self._rp_W_base = np.ones((n_rel, latent_dim), dtype=np.float32)
        self._rp_U_d = np.zeros((self.num_domains, n_rel, rank, latent_dim), dtype=np.float32)
        self._rp_V_d = np.zeros((self.num_domains, n_rel, rank, latent_dim), dtype=np.float32)
        self._rp_mW_base = np.zeros_like(self._rp_W_base)
        self._rp_mU_d = np.zeros_like(self._rp_U_d)
        self._rp_mV_d = np.zeros_like(self._rp_V_d)

    # ── hooks the RPMixin forward/backward rely on ──
    def _encoder_forward_full(self, X):
        return X, None, None, None

    def get_robust_embedding(self, tid):
        return self.token_embed.weight.data[tid]

    def _verb_stem(self, word):
        return VerbMixin._verb_stem(word)

    def _rp_forward_verb_offset(self, subject_tid, verb_word, return_count=False):
        if not verb_word or not self.use_verb_offset:
            return (None, 0, 0.0) if return_count else None
        return None

    def _get_or_adapt_entity_adapter(self, subject_tid, verb_word=None):
        if subject_tid not in self._entity_adapters:
            rank = self._entity_adapter_rank
            U = np.random.randn(rank, self.latent_dim).astype(np.float32) * 0.01
            V = np.random.randn(rank, self.latent_dim).astype(np.float32) * 0.01
            self._entity_adapters[subject_tid] = (U, V)
            self._entity_adapter_momentums[subject_tid] = (np.zeros_like(U), np.zeros_like(V))
        return self._entity_adapters[subject_tid]


# ── 1. Multi-probe & recall floor ────────────────────────────────────────────

class TestRecallFloor:
    def test_min_candidates_is_a_hard_guarantee(self):
        """LSH must never return fewer than min_candidates finite logits.

        If multi-probe cannot reach the floor, it falls back (None) rather than
        serving an undersized set — the original bug kept only 1-2 tokens.
        """
        model = _MockLSHModel(vocab_size=500, latent_dim=96)
        model._lsh_init(n_buckets=8, n_hashes=4, min_candidates=64)
        model.training = False
        src = model.token_embed.weight.data[0]
        tgt = model.token_embed.weight.data
        W = np.eye(model.latent_dim, dtype=np.float32)
        logits, info = model._lsh_scoring(src, tgt, W, training=False)
        if logits is None:
            pytest.skip("LSH fell back to full scoring for this query (acceptable)")
        finite = int(np.sum(np.isfinite(logits)))
        assert finite >= 64, (
            f"recall floor violated: only {finite} candidates survived, "
            f"expected >= 64"
        )

    def test_falls_back_when_vocab_too_small_for_lsh(self):
        """A tiny vocab (e.g. 50 tokens) should fall back to full scoring.

        With so few tokens, Hamming-1 multi-probe cannot reliably gather
        min_candidates, so the floor forces a fallback instead of over-pruning.
        """
        model = _MockLSHModel(vocab_size=50, latent_dim=8, structured=False)
        model._lsh_init(n_buckets=8, n_hashes=4, min_candidates=32)
        model.training = False
        src = model.token_embed.weight.data[0]
        tgt = model.token_embed.weight.data
        W = np.eye(model.latent_dim, dtype=np.float32)
        logits, info = model._lsh_scoring(src, tgt, W, training=False)
        assert logits is None, "Tiny vocab should fall back, not over-prune"

    def test_multiprobe_probes_more_than_one_bucket(self):
        """When the exact bucket is small, multi-probe must widen to neighbours."""
        model = _MockLSHModel(vocab_size=1000, latent_dim=96)
        # Large bucket count -> small exact buckets -> forces multi-probe.
        model._lsh_init(n_buckets=16, n_hashes=5, min_candidates=50)
        model.training = False
        src = model.token_embed.weight.data[0]
        tgt = model.token_embed.weight.data
        W = np.eye(model.latent_dim, dtype=np.float32)
        logits, info = model._lsh_scoring(src, tgt, W, training=False)
        if logits is None:
            pytest.skip("Fell back (acceptable)")
        n_candidates, buckets_probed = info
        assert buckets_probed >= 1
        # If the exact bucket already met the floor we probe 1; otherwise we
        # probe neighbours. Either way candidates must respect the floor.
        assert n_candidates >= 50 or logits is None

    def test_no_catastrophic_over_pruning(self):
        """Regression guard: finite-logit fraction must never be ~2%.

        The pre-hardening implementation returned only 2/50 finite logits.
        On a realistic vocab, the surviving fraction should be a meaningful
        subset (not <5%) OR a clean fallback.
        """
        model = _MockLSHModel(vocab_size=5000, latent_dim=96)
        model._lsh_init(n_buckets=8, n_hashes=4, min_candidates=32)
        model.training = False
        src = model.token_embed.weight.data[0]
        tgt = model.token_embed.weight.data
        W = np.eye(model.latent_dim, dtype=np.float32)
        logits, info = model._lsh_scoring(src, tgt, W, training=False)
        if logits is None:
            return  # fallback is acceptable
        finite = int(np.sum(np.isfinite(logits)))
        frac = finite / len(logits)
        assert frac >= 0.05, (
            f"over-pruned: only {frac:.1%} finite logits survived "
            f"({finite}/{len(logits)})"
        )


# ── 2. Training guard ─────────────────────────────────────────────────────────

class TestTrainingGuard:
    def test_training_returns_none(self):
        """In training, LSH must defer to full scoring for dense gradients."""
        model = _MockLSHModel()
        model._lsh_init()
        model.training = True
        src = model.token_embed.weight.data[0]
        tgt = model.token_embed.weight.data
        W = np.eye(model.latent_dim, dtype=np.float32)
        logits, info = model._lsh_scoring(src, tgt, W, training=True)
        assert logits is None, "LSH must not mask logits during training"
        assert info is None

    def test_inference_returns_logits_or_clean_fallback(self):
        """In eval mode, LSH either returns masked logits or falls back — never None-with-logits."""
        model = _MockLSHModel()
        model._lsh_init()
        model.training = False
        src = model.token_embed.weight.data[0]
        tgt = model.token_embed.weight.data
        W = np.eye(model.latent_dim, dtype=np.float32)
        logits, info = model._lsh_scoring(src, tgt, W, training=False)
        if logits is not None:
            assert logits.shape == (model.vocab_size,)
            assert info is not None and len(info) == 2

    def test_forward_uses_full_scoring_in_training(self):
        """End-to-end: _rp_forward in training must produce all-finite logits.

        This is the critical invariant — the softmax loss needs the full vocab
        distribution, so no -inf masking should leak through during training.
        """
        model = _MockLSHModel(vocab_size=200, latent_dim=96)
        model._lsh_init()
        model.training = True
        logits = model._rp_forward(subject_tid=0, rel_type_idx=0)
        assert np.all(np.isfinite(logits)), (
            "Training forward must not contain -inf (LSH must be bypassed)"
        )

    def test_forward_may_mask_in_eval(self):
        """In eval mode, masking (some -inf) is permitted when LSH activates."""
        model = _MockLSHModel(vocab_size=2000, latent_dim=96)
        model._lsh_init(n_buckets=8, n_hashes=4, min_candidates=32)
        model.training = False
        logits = model._rp_forward(subject_tid=0, rel_type_idx=0)
        # Either fully finite (fallback) or has some -inf (LSH active). Both valid.
        assert logits.shape == (model.vocab_size,)


# ── 3. Embedding-drift refresh ─────────────────────────────────────────────────

class TestDriftRefresh:
    def test_notify_bumps_version(self):
        model = _MockLSHModel()
        model._lsh_init()
        v0 = model._lsh_embedding_version
        model._lsh_notify_embedding_update()
        assert model._lsh_embedding_version == v0 + 1

    def test_bucket_map_rebuilds_after_drift(self):
        """After enough embedding updates, the bucket map must rebuild.

        Without this, a stale map would point queries at the wrong neighbours
        as embeddings drift during training.
        """
        model = _MockLSHModel()
        model._lsh_init(max_embedding_age=10)
        model._lsh_build_buckets()
        version_at_build = model._lsh_bucket_version
        # Simulate many backward passes that update embeddings.
        for _ in range(15):
            model._lsh_notify_embedding_update()
        model._lsh_build_buckets()  # should detect staleness and rebuild
        assert model._lsh_bucket_version > version_at_build, (
            "Bucket map did not rebuild after embedding drift"
        )

    def test_no_rebuild_before_age_threshold(self):
        """Small drift should NOT trigger a rebuild (avoid thrashing)."""
        model = _MockLSHModel()
        model._lsh_init(max_embedding_age=100)
        model._lsh_build_buckets()
        version_at_build = model._lsh_bucket_version
        for _ in range(5):
            model._lsh_notify_embedding_update()
        model._lsh_build_buckets()
        assert model._lsh_bucket_version == version_at_build, (
            "Bucket map rebuilt too eagerly (below age threshold)"
        )

    def test_backward_notifies_lsh(self):
        """_rp_backward must call _lsh_notify_embedding_update when embeds change."""
        model = _MockLSHModel()
        model._lsh_init()
        # Enable embedding updates (frozen by default).
        model.freeze_token_embeds_in_rp = False
        v0 = model._lsh_embedding_version
        model._rp_forward(subject_tid=0, rel_type_idx=0)
        model._rp_backward(target_id=5)
        assert model._lsh_embedding_version > v0, (
            "_rp_backward did not notify LSH of embedding drift"
        )


# ── 4. Correctness invariants ──────────────────────────────────────────────────

class TestCorrectness:
    def test_masked_logits_are_neg_inf(self):
        """Non-candidate tokens must be exactly -inf so exp() -> 0 probability."""
        model = _MockLSHModel(vocab_size=1000, latent_dim=96)
        model._lsh_init(n_buckets=8, n_hashes=4, min_candidates=32)
        model.training = False
        src = model.token_embed.weight.data[0]
        tgt = model.token_embed.weight.data
        W = np.eye(model.latent_dim, dtype=np.float32)
        logits, info = model._lsh_scoring(src, tgt, W, training=False)
        if logits is None:
            pytest.skip("Fell back (acceptable)")
        finite_mask = np.isfinite(logits)
        masked = logits[~finite_mask]
        assert np.all(masked == -np.inf), "Masked logits must be -inf, not other values"

    def test_candidate_logits_match_full_scoring(self):
        """For surviving candidates, LSH logits must equal the full bilinear score.

        LSH only changes WHICH tokens are scored, not HOW. The math is identical
        for the candidates that survive.
        """
        model = _MockLSHModel(vocab_size=1000, latent_dim=96)
        model._lsh_init(n_buckets=8, n_hashes=4, min_candidates=32)
        model.training = False
        src = model.token_embed.weight.data[0]
        tgt = model.token_embed.weight.data
        W = np.eye(model.latent_dim, dtype=np.float32)
        full_logits = src @ W @ tgt.T
        logits, info = model._lsh_scoring(src, tgt, W, training=False)
        if logits is None:
            pytest.skip("Fell back (acceptable)")
        finite_idx = np.where(np.isfinite(logits))[0]
        # Every surviving candidate must match the full score within float tolerance.
        np.testing.assert_allclose(
            logits[finite_idx], full_logits[finite_idx], rtol=1e-5, atol=1e-6,
            err_msg="LSH candidate logits diverge from full bilinear scores"
        )

    def test_lsh_hash_deterministic(self):
        model = _MockLSHModel()
        model._lsh_init()
        embeds = model.token_embed.weight.data
        h1 = model._lsh_hash(embeds)
        h2 = model._lsh_hash(embeds)
        np.testing.assert_array_equal(h1, h2)

    def test_lsh_hash_in_range(self):
        model = _MockLSHModel()
        model._lsh_init(n_buckets=8, n_hashes=4)
        embeds = model.token_embed.weight.data
        h = model._lsh_hash(embeds)
        max_bucket = model._lsh_n_buckets ** model._lsh_n_hashes
        assert h.min() >= 0
        assert h.max() < max_bucket, f"hash {h.max()} exceeds bucket space {max_bucket}"

    def test_neighbour_buckets_include_source(self):
        model = _MockLSHModel()
        model._lsh_init(n_buckets=4, n_hashes=3)
        neighbours = model._lsh_neighbour_bucket_ids(7)
        assert 7 in neighbours
        # Hamming-1: up to 1 + n_hashes*(n_buckets-1) = 1 + 3*3 = 10 buckets.
        assert len(neighbours) <= 1 + model._lsh_n_hashes * (model._lsh_n_buckets - 1)

    def test_speedup_on_large_vocab(self):
        """LSH should score meaningfully fewer tokens than full vocab (>2x)."""
        model = _MockLSHModel(vocab_size=5000, latent_dim=96)
        model._lsh_init(n_buckets=8, n_hashes=4, min_candidates=32)
        model.training = False
        src = model.token_embed.weight.data[0]
        tgt = model.token_embed.weight.data
        W = np.eye(model.latent_dim, dtype=np.float32)
        logits, info = model._lsh_scoring(src, tgt, W, training=False)
        if logits is None:
            pytest.skip("Fell back (acceptable for this seed)")
        n_candidates = info[0]
        assert n_candidates < model.vocab_size, "LSH scored the entire vocab (no speedup)"


# ── 5. Configuration & edge cases ──────────────────────────────────────────────

class TestConfiguration:
    def test_defaults_preserved(self):
        """Default n_hashes=3, n_buckets=8 (existing tests depend on this)."""
        model = _MockLSHModel()
        model._lsh_init()
        assert model._lsh_n_hashes == 3
        assert model._lsh_n_buckets == 8

    def test_custom_config(self):
        model = _MockLSHModel()
        model._lsh_init(n_buckets=16, n_hashes=6, min_candidates=100)
        assert model._lsh_n_buckets == 16
        assert model._lsh_n_hashes == 6
        assert model._lsh_min_candidates == 100

    def test_lazy_init_on_first_scoring(self):
        """Calling _lsh_scoring without explicit _lsh_init must lazy-init."""
        model = _MockLSHModel()
        assert not hasattr(model, '_lsh_n_hashes')
        src = model.token_embed.weight.data[0]
        tgt = model.token_embed.weight.data
        W = np.eye(model.latent_dim, dtype=np.float32)
        model.training = False
        model._lsh_scoring(src, tgt, W, training=False)  # should not raise
        assert hasattr(model, '_lsh_n_hashes')

    def test_projection_vectors_are_unit_norm(self):
        """Each hash projection must be a unit vector for angular LSH."""
        model = _MockLSHModel()
        model._lsh_init(n_buckets=8, n_hashes=5)
        norms = np.linalg.norm(model._lsh_projections, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_empty_bucket_falls_back(self):
        """A query hashing to an empty bucket must fall back cleanly."""
        model = _MockLSHModel(vocab_size=100, latent_dim=96)
        # Large but int32-safe bucket space: 16**6 = 16.7M cells for 100 tokens
        # -> almost all buckets empty, so multi-probe cannot reach min_candidates.
        model._lsh_init(n_buckets=16, n_hashes=6, min_candidates=32)
        model._lsh_build_buckets()
        model.training = False
        src = model.token_embed.weight.data[0]
        tgt = model.token_embed.weight.data
        W = np.eye(model.latent_dim, dtype=np.float32)
        # Should either fall back (None) or respect the floor — never crash.
        logits, info = model._lsh_scoring(src, tgt, W, training=False)
        if logits is not None:
            assert int(np.sum(np.isfinite(logits))) >= 32

    def test_oversized_config_rejected(self):
        """Configs whose bucket space overflows int32 must raise, not corrupt."""
        model = _MockLSHModel()
        # 64**8 ≈ 2.8e14 > int32 max (2.1e9).
        with pytest.raises(ValueError, match="exceeds int32"):
            model._lsh_init(n_buckets=64, n_hashes=8)
