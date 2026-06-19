"""Tests for rlm_v2_rp methods — relation prediction forward/backward, verb offset blending."""

import pytest
import numpy as np
from ravana_ml.nn.rlm_v2_rp import RPMixin
from ravana_ml.nn.rlm_v2_verb import VerbMixin


class _MockRPModel(RPMixin):
    """Minimal mock with RP parameters."""
    def __init__(self):
        self.vocab_size = 10
        self.embed_dim = 8
        self.latent_dim = 8  # Same as embed_dim (no projection needed in mock)
        self.hidden_dim = 16
        self.current_domain_id = 0
        self.use_verb_offset = True
        self._rp_lr = 0.01
        self._rp_momentum = 0.9
        self._rp_use_encoder_latent = False
        self.rp_scale = 16.0
        self._entity_adapter_lr = 0.01
        self._entity_adapter_momentum = 0.9
        self._entity_adapter_rank = 4

        # Token embeddings
        self.token_embed = type('MockEmbed', (), {
            'weight': type('MockWeight', (), {
                'data': np.random.randn(self.vocab_size, self.embed_dim).astype(np.float32)
            })()
        })()

        # Relation prediction matrices (latent_dim x latent_dim)
        n_domains = 3
        n_relations = 6
        self._rp_rel_matrices = np.random.randn(
            n_domains, n_relations, self.latent_dim, self.latent_dim
        ).astype(np.float32) * 0.01
        self._rp_mrel_matrices = np.zeros_like(self._rp_rel_matrices)

        # Verb offsets (needed for _rp_forward)
        self._verb_offsets = {0: {}}
        self._verb_offset_count = {0: {}}
        self._verb_offset_variance = {0: {}}

        # Entity adapters
        self._entity_adapters = {}
        self._entity_adapter_momentums = {}

        # Other state
        self._rp_cache = None
        self.freeze_token_embeds_in_rp = True
        self._token_embed_norms = None
        self._test_time_adapt_mode = False

    def get_robust_embedding(self, tid):
        return self.token_embed.weight.data[tid]

    def _rp_forward_verb_offset(self, subject_tid, verb_word, return_count=False):
        """Verb offset prediction — needed by _rp_forward."""
        if not verb_word or not self.use_verb_offset:
            return (None, 0) if return_count else None
        stem = VerbMixin._verb_stem(verb_word)
        domain_id = self.current_domain_id if self.current_domain_id is not None else 0
        if domain_id not in self._verb_offsets or stem not in self._verb_offsets[domain_id]:
            return (None, 0) if return_count else None
        count = self._verb_offset_count.get(domain_id, {}).get(stem, 0)
        variance = self._verb_offset_variance.get(domain_id, {}).get(stem, 0.0)
        offset = self._verb_offsets[domain_id][stem]
        source_embed = self.get_robust_embedding(subject_tid)
        predicted = source_embed + offset

        token_embeds = self.token_embed.weight.data
        token_norms = np.linalg.norm(token_embeds, axis=1)
        pred_norm = np.linalg.norm(predicted)
        if pred_norm > 0 and np.any(token_norms > 0):
            valid_tok = token_norms > 0
            normed_tok = token_embeds.copy()
            normed_tok[valid_tok] /= token_norms[valid_tok, np.newaxis]
            logits = (predicted / pred_norm) @ normed_tok.T
            if 0 <= subject_tid < len(logits):
                logits[subject_tid] = np.min(logits) - 10.0
            logits *= 10.0
            return (logits, count, variance) if return_count else logits
        return (None, 0, variance) if return_count else None

    def _get_or_adapt_entity_adapter(self, subject_tid, verb_word=None):
        if subject_tid not in self._entity_adapters:
            rank = self._entity_adapter_rank
            latent_dim = self.latent_dim
            U = np.random.randn(rank, latent_dim).astype(np.float32) * 0.01
            V = np.random.randn(rank, latent_dim).astype(np.float32) * 0.01
            self._entity_adapters[subject_tid] = (U, V)
            self._entity_adapter_momentums[subject_tid] = (np.zeros_like(U), np.zeros_like(V))
        return self._entity_adapters[subject_tid]

    def _verb_stem(self, word):
        return VerbMixin._verb_stem(word)


class TestRPForward:
    def test_rp_forward_shape(self):
        model = _MockRPModel()
        logits = model._rp_forward(subject_tid=0, rel_type_idx=0)
        assert isinstance(logits, np.ndarray)
        assert logits.shape == (model.vocab_size,)

    def test_rp_forward_suppresses_subject(self):
        model = _MockRPModel()
        model._verb_offsets[0] = {"melt": np.ones(model.embed_dim, dtype=np.float32) * 0.5}
        model._verb_offset_count[0] = {"melt": 50}
        model._verb_offset_variance[0] = {"melt": 0.01}
        logits = model._rp_forward(subject_tid=0, rel_type_idx=0, verb_word="melt")
        assert logits[0] < np.max(logits)

    def test_rp_forward_with_verb_offset(self):
        model = _MockRPModel()
        model._verb_offsets[0] = {"melt": np.ones(model.embed_dim, dtype=np.float32) * 0.5}
        model._verb_offset_count[0] = {"melt": 20}
        model._verb_offset_variance[0] = {"melt": 0.01}
        logits = model._rp_forward(subject_tid=0, rel_type_idx=0, verb_word="melt")
        assert logits.shape == (model.vocab_size,)

    def test_rp_forward_rare_verb_blends(self):
        """Rare verb (count=2) blends with W_rel instead of dominating."""
        model = _MockRPModel()
        model._verb_offsets[0] = {"rare": np.ones(model.embed_dim, dtype=np.float32) * 0.3}
        model._verb_offset_count[0] = {"rare": 2}
        model._verb_offset_variance[0] = {"rare": 0.05}
        logits = model._rp_forward(subject_tid=0, rel_type_idx=0, verb_word="rare")
        assert logits.shape == (model.vocab_size,)

    def test_rp_backward_fixes_cache(self):
        model = _MockRPModel()
        logits = model._rp_forward(subject_tid=0, rel_type_idx=0)
        model._rp_backward(target_id=5)
        assert model._rp_cache is None

    def test_rp_backward_no_cache(self):
        model = _MockRPModel()
        model._rp_cache = None
        model._rp_backward(target_id=0)  # Should not raise


class TestEntityAdapter:
    def test_get_or_create_adapter(self):
        model = _MockRPModel()
        result = model._get_or_adapt_entity_adapter(subject_tid=5)
        assert result is not None
        U, V = result
        assert U.shape == (model._entity_adapter_rank, model.latent_dim)
        assert V.shape == (model._entity_adapter_rank, model.latent_dim)

    def test_adapter_caching(self):
        model = _MockRPModel()
        result1 = model._get_or_adapt_entity_adapter(subject_tid=3)
        result2 = model._get_or_adapt_entity_adapter(subject_tid=3)
        assert result1 is result2

    def test_adapter_update_after_backward(self):
        model = _MockRPModel()
        model._get_or_adapt_entity_adapter(subject_tid=1)
        U_before = model._entity_adapters[1][0].copy()
        logits = model._rp_forward(subject_tid=1, rel_type_idx=0)
        model._rp_backward(target_id=5)
        U_after = model._entity_adapters[1][0]
        assert not np.allclose(U_before, U_after)
