"""Tests for RLMv2 with contrastive relation predictor."""

import pytest
import numpy as np
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer


class TestRPContrastive:
    """Test RLMv2 with relation predictor contrastive learning."""

    @pytest.fixture
    def model_and_tokenizer(self):
        all_texts = [
            "heat causes expansion",
            "fire produces warmth",
            "kindness leads to trust",
            "anger causes conflict",
            "ice causes expansion",
            "light produces warmth",
            "honesty leads to trust",
            "sadness causes conflict",
            "friction produces heat",
            "gravity pulls objects",
            "rain causes growth",
            "cold freezes water",
            "patience creates understanding",
            "honesty builds respect",
            "generosity creates gratitude",
            "cold melts ice",
        ]
        tokenizer = WordTokenizer()
        for t in all_texts:
            tokenizer.encode(t)

        model = RLMv2(
            vocab_size=tokenizer.vocab_size, embed_dim=64,
            concept_dim=64, n_concepts=100
        )
        model.disable_spreading_activation = True
        model.use_rp_for_analogy = True
        model.use_rp_hidden = True
        model.use_rp_contrastive = True
        return model, tokenizer

    def test_init(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        assert model.vocab_size == tokenizer.vocab_size
        assert model.graph is not None

    def test_forward_no_crash(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        text = "heat causes expansion"
        input_ids = np.array(tokenizer.encode(text), dtype=np.int64)
        logits = model.forward(input_ids)
        assert hasattr(logits, 'data')
        assert logits.data.shape[0] == model.vocab_size

    def test_learn_no_crash(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        facts = [
            ("heat causes ", "expansion"),
            ("fire produces ", "warmth"),
        ]
        for inp, tgt in facts:
            input_ids = np.array(tokenizer.encode(inp), dtype=np.int64)
            target_ids = np.array(tokenizer.encode(tgt), dtype=np.int64)
            model.set_domain(0)
            res = model.learn(input_ids, target_ids)
            assert "loss" in res

    def test_set_domain(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        model.set_domain(0)
        assert model.current_domain_id == 0
        model.set_domain(1)
        assert model.current_domain_id == 1

    def test_multi_domain_training(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        facts = [
            ("heat causes ", "expansion", 0),
            ("fire produces ", "warmth", 0),
            ("kindness leads to ", "trust", 1),
            ("honesty builds ", "respect", 1),
        ]
        for inp, tgt, domain in facts:
            input_ids = np.array(tokenizer.encode(inp), dtype=np.int64)
            target_ids = np.array(tokenizer.encode(tgt), dtype=np.int64)
            model.set_domain(domain)
            res = model.learn(input_ids, target_ids)
            assert "loss" in res
        assert model._step_counter == len(facts)

    def test_held_out_prediction(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        # Train on domain 0
        for inp, tgt in [("heat causes ", "expansion"), ("fire produces ", "warmth")]:
            model.set_domain(0)
            model.learn(
                np.array(tokenizer.encode(inp), dtype=np.int64),
                np.array(tokenizer.encode(tgt), dtype=np.int64),
            )
        # Test prediction
        test_input = np.array(tokenizer.encode("ice causes "), dtype=np.int64)
        model.set_domain(0)
        logits = model.forward(test_input)
        pred_id = int(np.argmax(logits.data))
        assert 0 <= pred_id < model.vocab_size
