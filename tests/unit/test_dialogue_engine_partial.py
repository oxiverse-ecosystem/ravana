"""Tests for ravana_grace.dialogue.dialogue_engine — basic functionality."""

import pytest
from unittest.mock import MagicMock
from ravana_grace.core.dialogue_context import Triple
from ravana_grace.dialogue.dialogue_engine import (
    DialogueEngine, DialogueEngineConfig, DialogueTurnRecord,
)


class TestDialogueEngineConfig:
    def test_defaults(self):
        cfg = DialogueEngineConfig()
        assert cfg.decay_rate == 0.92
        assert cfg.tokenizer_name == "word"
        assert cfg.sleep_pressure_threshold == 2.0


class TestDialogueEngine:
    def test_init(self):
        de = DialogueEngine(user_id="test")
        assert de.user_id == "test"
        assert de.turn_count == 0
        assert de.dialogue_context is not None
        assert de.repair is not None

    def test_init_with_components(self):
        from ravana_ml.graph import ConceptGraph
        g = ConceptGraph(dim=8, max_nodes=100)
        de = DialogueEngine(user_id="test", graph=g)
        assert de.graph is g
        assert de.repair.graph is g

    def test_parse_input_to_triples(self):
        de = DialogueEngine(user_id="test")
        triples = de._parse_input_to_triples("heat causes expansion", [0, 1, 2])
        assert len(triples) >= 1

    def test_parse_input_to_triples_with_rlm(self):
        # Without RLM, uses fallback parsing
        de = DialogueEngine(user_id="test")
        triples = de._parse_input_to_triples("heat causes expansion", [0, 1, 2])
        if triples:
            assert triples[0].subject == "heat"

    def test_generate_fallback_response(self):
        de = DialogueEngine(user_id="test")
        t = Triple(subject="heat", relation="causes", relation_type="causal", object="expansion")
        response, response_triples = de._generate_fallback_response(t, [("heat", 0.9)])
        assert len(response) > 0

    def test_accumulate_free_energy(self):
        de = DialogueEngine(user_id="test")
        t = Triple(subject="heat", relation="causes", relation_type="causal", object="expansion")
        de._accumulate_free_energy([t])
        assert de._free_energy_accumulator > 0

    def test_get_state(self):
        de = DialogueEngine(user_id="test")
        state = de.get_state()
        assert state["user_id"] == "test"
        assert state["turn_count"] == 0
        assert "free_energy" in state

    def test_reset_conversation(self):
        de = DialogueEngine(user_id="test")
        de.turn_count = 10
        de._free_energy_accumulator = 5.0
        de.reset_conversation()
        assert de.turn_count == 0
        assert de._free_energy_accumulator == 0.0

    def test_set_user_id(self):
        de = DialogueEngine(user_id="old_user")
        de.set_user_id("new_user")
        assert de.user_id == "new_user"
        assert de.dialogue_context.user_id == "new_user"

    def test_get_turn_history(self):
        de = DialogueEngine(user_id="test")
        history = de.get_turn_history()
        assert history == []

    def test_repr(self):
        de = DialogueEngine(user_id="test")
        r = repr(de)
        assert "DialogueEngine" in r

    def test_modulate_with_emotion_no_engine(self):
        de = DialogueEngine(user_id="test")
        activations = {"heat": 0.8, "expansion": 0.5}
        result = de._modulate_with_emotion(activations)
        # Without emotion engine, returns same dict
        assert result == activations

    def test_check_sleep_needed_initial(self):
        de = DialogueEngine(user_id="test")
        assert de._check_sleep_needed() is False
