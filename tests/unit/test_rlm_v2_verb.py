"""Tests for rlm_v2_verb methods — verb stem extraction, offset accumulation, clustering."""

import pytest
import numpy as np
from ravana_ml.nn.rlm_v2_verb import VerbMixin


class _MockVerbModel(VerbMixin):
    """Minimal mock RLMv2 model with just the verb methods."""
    def __init__(self):
        self.use_verb_offset = True
        self.current_domain_id = 0
        self.token_embed = type('MockEmbed', (), {
            'weight': type('MockWeight', (), {
                'data': np.eye(10).astype(np.float32)
            })()
        })()
        self._verb_offsets = {}
        self._verb_offset_count = {}
        self._verb_offset_variance = {}
        self._verb_accum_buffer = []

    def get_robust_embedding(self, tid):
        return self.token_embed.weight.data[tid]


class TestVerbStem:
    """Test the _verb_stem static method."""

    def test_stem_short_words(self):
        assert VerbMixin._verb_stem("is") == "is"
        assert VerbMixin._verb_stem("do") == "do"
        assert VerbMixin._verb_stem("go") == "go"

    def test_stem_ing_suffix(self):
        assert VerbMixin._verb_stem("making") == "mak"
        assert VerbMixin._verb_stem("causing") == "caus"
        assert VerbMixin._verb_stem("running") == "runn"

    def test_stem_ed_suffix(self):
        """'ed' suffix stripped: caused -> caus, produced -> produc."""
        assert VerbMixin._verb_stem("caused") == "caus"
        assert VerbMixin._verb_stem("produced") == "produc"
        assert VerbMixin._verb_stem("liked") == "lik"  # 'ed' matches before 'd'

    def test_stem_es_suffix(self):
        """'es' suffix stripped: causes -> caus, produces -> produc, makes -> mak."""
        assert VerbMixin._verb_stem("causes") == "caus"
        assert VerbMixin._verb_stem("produces") == "produc"
        assert VerbMixin._verb_stem("freezes") == "freez"
        # 'makes' ends with 'es', so 'es' strips first -> 'mak'
        assert VerbMixin._verb_stem("makes") == "mak"

    def test_stem_s_suffix(self):
        """'s' suffix: melts -> melt, breaks -> break."""
        assert VerbMixin._verb_stem("melts") == "melt"
        assert VerbMixin._verb_stem("breaks") == "break"

    def test_stem_case_insensitive(self):
        assert VerbMixin._verb_stem("CAUSES") == "caus"
        # 'MAKES' -> lowercase -> 'makes' -> 'es' match -> 'mak'
        assert VerbMixin._verb_stem("Makes") == "mak"

    def test_stem_d_suffix(self):
        """'d' suffix only matched when 'ed' doesn't apply."""
        assert VerbMixin._verb_stem("liked") == "lik"  # 'ed' matched first
        assert VerbMixin._verb_stem("used") == "us"    # 'ed' matched first

    def test_stem_preserves_core_contours(self):
        """Stem 'melts' should match 'melt' stem."""
        s1 = VerbMixin._verb_stem("melts")
        s2 = VerbMixin._verb_stem("melt")
        assert s1 == s2

    def test_stem_compound_underscore_suffix(self):
        """Compound with '_': verb part gets stemmed, particle preserved."""
        assert VerbMixin._verb_stem("contributes_to") == "contribut_to"
        assert VerbMixin._verb_stem("leads_to") == "lead_to"
        assert VerbMixin._verb_stem("results_in") == "result_in"
        assert VerbMixin._verb_stem("associated_with") == "associat_with"
        assert VerbMixin._verb_stem("correlated_with") == "correlat_with"

    def test_stem_compound_underscore_no_suffix(self):
        """Compound where verb part has no suffix to strip."""
        assert VerbMixin._verb_stem("is_a") == "is_a"
        assert VerbMixin._verb_stem("type_of") == "type_of"
        assert VerbMixin._verb_stem("same_as") == "same_as"
        assert VerbMixin._verb_stem("able_to") == "able_to"

    def test_stem_compound_hyphen(self):
        """Compound with '-' (hyphen): converted to '_' after stemming."""
        assert VerbMixin._verb_stem("leads-to") == "lead_to"
        assert VerbMixin._verb_stem("made-of") == "made_of"

    def test_stem_compound_multi_part(self):
        """Multi-part compounds: only first part stemmed."""
        assert VerbMixin._verb_stem("is_type_of") == "is_type_of"
        assert VerbMixin._verb_stem("can_cause") == "can_cause"
        assert VerbMixin._verb_stem("may_cause") == "may_cause"

    def test_stem_compound_ing(self):
        """Compound with 'ing' on verb part."""
        assert VerbMixin._verb_stem("leading_to") == "lead_to"
        assert VerbMixin._verb_stem("consisting_of") == "consist_of"


class TestVerbOffsetAccumulation:
    """Test verb offset accumulation logic."""

    def test_accumulate_disabled(self):
        model = _MockVerbModel()
        model.use_verb_offset = False
        model._accumulate_verb_offset(0, 1, "causes", domain_id=0)
        assert len(model._verb_accum_buffer) == 0

    def test_accumulate_without_verb_word(self):
        model = _MockVerbModel()
        model._accumulate_verb_offset(0, 1, "", domain_id=0)
        assert len(model._verb_accum_buffer) == 0

    def test_accumulate_adds_to_buffer(self):
        model = _MockVerbModel()
        model._accumulate_verb_offset(0, 1, "causes", domain_id=0)
        assert len(model._verb_accum_buffer) == 1
        stem, offset, domain = model._verb_accum_buffer[0]
        assert stem == "caus"
        assert isinstance(offset, np.ndarray)
        assert offset.shape == (model.token_embed.weight.data.shape[1],)

    def test_accumulate_multiple(self):
        model = _MockVerbModel()
        model._accumulate_verb_offset(0, 1, "causes", domain_id=0)
        model._accumulate_verb_offset(1, 2, "causes", domain_id=0)
        model._accumulate_verb_offset(2, 3, "melts", domain_id=0)
        assert len(model._verb_accum_buffer) == 3

    def test_compute_verb_offsets_empty(self):
        model = _MockVerbModel()
        model._compute_verb_offsets()
        assert len(model._verb_offsets) == 0

    def test_compute_verb_offsets_with_data(self):
        model = _MockVerbModel()
        embed_dim = model.token_embed.weight.data.shape[1]
        model._verb_accum_buffer.append(("caus", np.ones(embed_dim, dtype=np.float32) * 0.5, 0))
        model._verb_accum_buffer.append(("caus", np.ones(embed_dim, dtype=np.float32) * 0.7, 0))
        model._verb_accum_buffer.append(("melt", np.ones(embed_dim, dtype=np.float32) * 0.3, 0))
        model._compute_verb_offsets()
        assert 0 in model._verb_offsets
        assert "caus" in model._verb_offsets[0]
        assert "melt" in model._verb_offsets[0]
        assert model._verb_offset_count[0]["caus"] == 2
        assert model._verb_offset_count[0]["melt"] == 1

    def test_rp_forward_verb_offset_no_offset(self):
        """When verb has no offset, return (None, 0)."""
        model = _MockVerbModel()
        result = model._rp_forward_verb_offset(0, "nonexistent", return_count=True)
        assert result is not None
        assert result[0] is None
        assert result[1] == 0

    def test_rp_forward_verb_offset_disabled(self):
        model = _MockVerbModel()
        model.use_verb_offset = False
        result = model._rp_forward_verb_offset(0, "causes", return_count=True)
        assert result is not None
        assert result[0] is None
        assert result[1] == 0

    def test_rp_forward_verb_offset_with_count(self):
        """With return_count=True, returns (logits, count, variance) tuple."""
        model = _MockVerbModel()
        model.current_domain_id = 0
        embed_dim = model.token_embed.weight.data.shape[1]
        model._verb_offsets[0] = {"melt": np.ones(embed_dim, dtype=np.float32) * 0.5}
        model._verb_offset_count[0] = {"melt": 10}
        model._verb_offset_variance[0] = {"melt": 0.01}
        # Use "melt" as verb_word — its stem is already "melt" (no suffix stripping)
        result = model._rp_forward_verb_offset(0, "melt", return_count=True)
        assert result is not None
        assert result[0] is not None
        assert result[1] == 10
        assert result[2] == 0.01
        logits, count, variance = result
        assert isinstance(logits, np.ndarray)


class TestVerbOffsetClustering:
    """Test verb offset clustering during sleep."""

    def test_clustering_with_similar_offsets(self):
        model = _MockVerbModel()
        embed_dim = model.token_embed.weight.data.shape[1]
        model._verb_offsets[0] = {
            "caus": np.array([1.0] + [0.0] * (embed_dim - 1), dtype=np.float32),
            "produ": np.array([0.9, 0.1] + [0.0] * (embed_dim - 2), dtype=np.float32),
            "melt": np.array([0.0, 1.0] + [0.0] * (embed_dim - 2), dtype=np.float32),
        }
        model._verb_offset_count[0] = {"caus": 10, "produ": 5, "melt": 8}
        model._verb_offset_variance[0] = {"caus": 0.01, "produ": 0.02, "melt": 0.01}
        model._cluster_verb_offsets(similarity_threshold=0.85)
        remaining = list(model._verb_offsets[0].keys())
        assert len(remaining) <= 2  # caus+produ merged, melt separate

    def test_clustering_empty(self):
        model = _MockVerbModel()
        model._cluster_verb_offsets()
        assert len(model._verb_offsets) == 0
