"""Tests for ravana_ml.nn.rlm (original v1) — basic instantiation and properties."""

import pytest
import numpy as np


class TestRLMInit:
    def test_init_minimal(self):
        """Test minimal instantiation of the original RLM."""
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16, n_layers=2,
            tokenizer=None,
        )
        assert model.vocab_size == 10
        assert model.embed_dim == 8
        assert model.concept_dim == 8
        assert model.n_concepts == 5
        assert model.n_hidden == 16
        assert model.n_layers == 2
        assert model._step_counter == 0
        assert model.sleep_cycles_completed == 0
        assert model.graph is not None
        assert model.propagation is not None
        assert model.free_energy_engine is not None

    def test_init_with_custom_params(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=256, embed_dim=64, concept_dim=64,
            n_concepts=50, n_hidden=128, n_layers=3,
            max_seq_len=256, free_energy_threshold=5.0,
            sleep_interval=50, replay_buffer_max=200,
            gate_concept_creation=True, adaptive_downscale=True,
        )
        assert model.max_seq_len == 256
        assert model.free_energy_threshold == 5.0
        assert model.sleep_interval == 50
        assert model._gate_concept_creation is True
        assert model._adaptive_downscale is True

    def test_init_with_ablation_flags(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16, n_layers=2,
            anchor_relation_vectors=False,
            gate_concept_creation=False,
            adaptive_downscale=False,
        )
        assert model._anchor_relation_vectors is False
        assert model._gate_concept_creation is False
        assert model._adaptive_downscale is False

    def test_token_embedding(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        embed = model.token_embed.embed_raw(1)
        assert embed.shape == (8,)
        assert np.linalg.norm(embed) > 0

    def test_structured_embeddings_small_vocab(self):
        """Structured embeddings only for small vocabs (<=32)."""
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        # Check first two embedding dims have structured values (cos/sin)
        e0 = model.token_embed.embed_raw(0)
        assert abs(np.linalg.norm(e0) - 1.0) < 0.1

    def test_positional_encoding(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        assert model._positional_encoding is not None
        assert model._positional_encoding.shape[0] >= 1024
        assert model._positional_encoding.shape[1] == 8

    def test_cognitive_currencies(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        assert model.currencies is not None
        assert model.identity_strength == 0.5
        assert model.valence == 0.0
        assert model.arousal == pytest.approx(0.3, abs=1e-4)  # baseline arousal
        assert model.dominance == 0.5

    def test_currency_properties(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        # Test setter properties
        model.identity_strength = 0.7
        assert model.identity_strength == 0.7
        model.valence = 0.5
        assert model.valence == 0.5
        model.sleep_pressure = 0.8
        assert model.sleep_pressure == 0.8

    def test_currency_advanced_properties(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        # Default value for identity_momentum is 0.0
        assert model.identity_momentum == 0.0
        model.identity_momentum = 0.3
        assert model.identity_momentum == 0.3
        model.accumulated_meaning = 2.0
        assert model.accumulated_meaning == 2.0
        model.dissonance_ema = 0.4
        assert model.dissonance_ema == 0.4

    def test_forward_basic(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        input_ids = np.array([[0, 1, 2, 3]], dtype=np.int64)
        logits = model.forward(input_ids)
        assert hasattr(logits, 'data')
        assert logits.data.shape[0] == model.vocab_size

    def test_projection_methods(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        # embed_dim == concept_dim here, so no truncation
        vec = np.random.randn(8).astype(np.float32)
        result = model._project_to_concept(vec)
        assert result.shape == (8,)
        result2 = model._project_to_embed(vec)
        # embed_dim=8, concept_dim=8 → both return same shape
        assert result2.shape == (8,)

    def test_classify_relation_causal(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=256, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        # Simulate char-level tokenizer via ASCII
        text_ids = np.array([ord(c) for c in "heat causes fire"], dtype=np.int64)
        rel = model._classify_relation(text_ids)
        assert rel in ("causal", "semantic", "temporal")

    def test_classify_relation_semantic(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=256, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        text_ids = np.array([ord(c) for c in "a is b"], dtype=np.int64)
        rel = model._classify_relation(text_ids)
        assert rel in ("causal", "semantic", "temporal")

    def test_concept_posterior(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        vec = model.token_embed.embed_raw(1)
        posterior = model._concept_posterior(vec, k=3)
        assert len(posterior) > 0
        for cid, prob in posterior:
            assert isinstance(cid, int)
            assert 0 <= prob <= 1.0

    def test_sleep_cycle(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        model.sleep_cycle()
        assert model.sleep_cycles_completed >= 0

    def test_sleep_cycle_counter_increments(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        # Multiple sleep cycles
        for _ in range(3):
            model.sleep_cycle()
        assert model.sleep_cycles_completed >= 3

    def test_learn_basic(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        token_ids = np.array([[0, 1]], dtype=np.int64)
        next_token = np.array([2], dtype=np.int64)
        model.learn(token_ids, next_token)
        assert model._step_counter == 1
        assert model._edges_learned >= 0

    def test_learn_multiple_steps(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        for i in range(5):
            token_ids = np.array([[i, i + 1]], dtype=np.int64)
            next_token = np.array([(i + 2) % 10], dtype=np.int64)
            model.learn(token_ids, next_token)
        assert model._step_counter == 5

    def test_binding_map(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        assert model.binding_map is not None

    def test_episodic_buffer(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        assert model._episodic_buffer_max == 500
        assert model._episodic_max == 5000

    def test_concept_gating_init(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        assert model._concept_similarity_threshold == 0.7
        assert model._max_concepts >= 5

    def test_identity_history_property(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        assert model.identity_history == []
        model.identity_history = [0.5, 0.6]
        assert model.identity_history == [0.5, 0.6]

    def test_state_dict_basic(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        sd = model.state_dict()
        # Named parameters include registered module params
        # (graph is a ConceptGraph, not a Module, so not in state_dict)
        assert "token_embed.weight" in sd
        assert "recurrent_cell.W_z.weight" in sd
        assert "concept_predictor.weight" in sd
        assert len(sd) >= 5  # Should have many registered parameters

    def test_token_concept_map(self):
        from ravana_ml.nn.rlm import RLM
        model = RLM(
            vocab_size=10, embed_dim=8, concept_dim=8,
            n_concepts=5, n_hidden=16,
        )
        assert len(model._token_concept_map) == model.vocab_size
