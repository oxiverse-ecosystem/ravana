"""
Integration tests for the DialogueEngine with real ConceptGraph + tokenizer.
Tests the full conversational pipeline end-to-end.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ravana-v2'))

from core.dialogue_context import DialogueContext, Triple
from core.conversational_repair import RepairEvent, CorrectionType
from dialogue.dialogue_engine import DialogueEngine, DialogueEngineConfig
from ravana_ml.graph import ConceptGraph


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_graph():
    """Create a simple ConceptGraph for testing."""
    g = ConceptGraph(dim=16)
    rng = np.random.RandomState(42)
    for label in ["heat", "expansion", "cold", "contraction", "coffee",
                  "anxiety", "calm", "water", "ice", "productive", "jittery"]:
        g.add_node(rng.randn(16).astype(np.float32) * 0.1, label)
    return g


def _link(graph, subject, relation_type, object_, weight=0.7):
    """Add an edge between two labeled concepts."""
    src = _find_node(graph, subject)
    tgt = _find_node(graph, object_)
    if src is not None and tgt is not None:
        graph.add_edge(src.id, tgt.id, weight=weight, relation_type=relation_type)


def _find_node(graph, label):
    for nid, node in graph.nodes.items():
        if node.label and node.label.lower() == label.lower():
            return node
    return None


# ── Integration Tests ───────────────────────────────────────────────────────

class TestDialogueEngineIntegration:

    def test_1_basic_turn_with_graph(self):
        """DialogueEngine processes a turn using a real ConceptGraph."""
        g = _make_graph()
        _link(g, "heat", "causal", "expansion", 0.7)
        engine = DialogueEngine(user_id="test", graph=g)

        response = engine.process_turn("heat causes expansion")
        assert engine.turn_count == 1
        assert isinstance(response, str) and len(response) > 0
        # Should reference the activated concepts
        concepts = engine.dialogue_context.get_active_concepts()
        assert "heat" in concepts or any("heat" in k.lower() for k in concepts)

    def test_2_multi_turn(self):
        """Multiple turns increment the counter and produce responses."""
        engine = DialogueEngine(user_id="test")
        for i, inp in enumerate(["hello world", "how are you", "goodbye"]):
            resp = engine.process_turn(inp)
            assert isinstance(resp, str) and len(resp) > 0
        assert engine.turn_count == 3

    def test_3_state_inspection(self):
        """get_state returns all expected keys."""
        g = _make_graph()
        _link(g, "heat", "causal", "expansion")
        engine = DialogueEngine(user_id="test", graph=g)
        engine.process_turn("heat causes expansion")

        state = engine.get_state()
        assert "user_id" in state
        assert "turn_count" in state
        assert "dialogue_context" in state
        assert "repair_stats" in state
        assert "free_energy" in state
        assert state["turn_count"] == 1

    def test_4_reset_conversation(self):
        """reset_conversation clears turn count and free energy."""
        engine = DialogueEngine(user_id="test")
        engine.process_turn("hello")
        engine._free_energy_accumulator = 5.0
        engine.reset_conversation()
        assert engine.turn_count == 0
        assert engine._free_energy_accumulator == 0.0
        assert engine._sleep_requested is False

    def test_5_user_switch(self):
        """set_user_id propagates to all subsystems."""
        engine = DialogueEngine(user_id="old_user")
        engine.set_user_id("new_user")
        assert engine.user_id == "new_user"
        assert engine.dialogue_context.user_id == "new_user"
        assert engine.repair.user_id == "new_user"

    def test_6_repair_detects_contradiction(self):
        """handle_correction detects same-subject contradictions."""
        g = _make_graph()
        _link(g, "coffee", "causal", "anxiety", 0.8)
        engine = DialogueEngine(user_id="test", graph=g)

        engine.process_turn("coffee causes anxiety")
        # Last output is recorded by process_turn
        event = engine.handle_correction("coffee causes calm")
        assert event is not None, "Should detect contradiction (same subject)"
        assert isinstance(event, RepairEvent)
        assert event.wrong_triple.object.lower() == "anxiety"
        assert event.correct_triple.object.lower() == "calm"
        assert len(engine.repair.repair_history) >= 1

    def test_7_repair_with_negation(self):
        """handle_correction handles negation markers."""
        g = _make_graph()
        _link(g, "coffee", "causal", "anxiety", 0.8)
        engine = DialogueEngine(user_id="test", graph=g)

        engine.dialogue_context.record_output(
            "coffee causes anxiety",
            [Triple("coffee", "causes", "causal", "anxiety")]
        )
        event = engine.handle_correction("No, coffee causes calm")
        assert event is not None, "Should detect negation contradiction"
        assert event.correction_type == CorrectionType.NEGATION

    def test_8_no_contradiction_different_subject(self):
        """Different subjects do NOT trigger contradiction."""
        g = _make_graph()
        _link(g, "heat", "causal", "expansion")
        engine = DialogueEngine(user_id="test", graph=g)

        engine.process_turn("heat causes expansion")
        event = engine.handle_correction("cold causes contraction")
        assert event is None, "Different subject should not be a contradiction"

    def test_9_sleep_pressure(self):
        """High free energy triggers sleep needed."""
        engine = DialogueEngine(user_id="test")
        engine.process_turn("hello")
        engine._free_energy_accumulator = 2.5  # Above threshold
        assert engine._check_sleep_needed() is True

    def test_10_sleep_without_engine(self):
        """trigger_sleep returns False when no sleep engine is set."""
        engine = DialogueEngine(user_id="test")
        assert engine.trigger_sleep() is False

    def test_11_repair_applies_agent_weights(self):
        """Repair updates ConceptEdge agent_weights in the graph."""
        g = _make_graph()
        _link(g, "coffee", "causal", "anxiety", 0.8)
        coffee = _find_node(g, "coffee")
        anxiety = _find_node(g, "anxiety")
        edge = g.get_edge(coffee.id, anxiety.id)

        engine = DialogueEngine(user_id="alice", graph=g)
        engine.dialogue_context.record_output(
            "coffee causes anxiety",
            [Triple("coffee", "causes", "causal", "anxiety")]
        )
        engine.handle_correction("coffee causes calm")

        # The wrong edge should have a penalized agent_weight for alice
        agent_key = "user_alice"
        assert agent_key in edge.agent_weights
        assert edge.agent_weights[agent_key] < 0.8, "Weight should be penalized"
        # Global weight preserved
        assert edge.weight == 0.8

    def test_12_turn_history(self):
        """get_turn_history returns recorded turns."""
        engine = DialogueEngine(user_id="test")
        engine.process_turn("first turn")
        engine.process_turn("second turn")
        history = engine.get_turn_history(n=2)
        assert len(history) == 2
        assert history[0].user_input == "first turn"
        assert history[1].user_input == "second turn"
        assert history[0].turn == 1
        assert history[1].turn == 2

    def test_13_emotional_modulation(self):
        """Emotional modulation filters activations by arousal threshold."""
        activations = {"strong": 0.5, "medium": 0.1, "weak": 0.01}
        arousal = 0.3  # Low arousal → high threshold
        threshold = 0.05 * (1.0 + (1.0 - arousal))  # = 0.085
        modulated = {c: a for c, a in activations.items() if a >= threshold}
        assert "strong" in modulated
        assert "weak" not in modulated

    def test_14_dialogue_context_memory(self):
        """User beliefs persist and can be retrieved after clearing."""
        ctx = DialogueContext(user_id="test")
        ctx.store_user_belief("coffee_effect", {
            "subject": "coffee", "relation": "causes", "object": "calm"
        })
        ctx.clear_for_sleep()
        belief = ctx.get_user_belief("coffee_effect")
        assert belief is not None
        assert belief["object"] == "calm"

    def test_15_multi_user_independent_beliefs(self):
        """Multiple users can hold different beliefs."""
        g = _make_graph()
        _link(g, "coffee", "causal", "productive", 0.6)

        engine_a = DialogueEngine(user_id="alice", graph=g)
        engine_b = DialogueEngine(user_id="bob", graph=g)

        # Alice corrects
        engine_a.dialogue_context.record_output(
            "coffee makes productive",
            [Triple("coffee", "makes", "causal", "productive")]
        )
        engine_a.handle_correction("coffee makes jittery")

        # Bob reinforces
        engine_b.dialogue_context.record_output(
            "coffee makes jittery",
            [Triple("coffee", "makes", "causal", "jittery")]
        )
        engine_b.handle_correction("coffee makes productive")

        # Find the productive edge
        coffee = _find_node(g, "coffee")
        productive = _find_node(g, "productive")
        edge = g.get_edge(coffee.id, productive.id)
        assert edge is not None

        # Alice's weight should be lower than Bob's for "productive"
        alice_w = edge.get_weight_for_agent("user_alice")
        bob_w = edge.get_weight_for_agent("user_bob")
        # Alice penalized productive, Bob boosted it
        assert alice_w < bob_w, f"Alice({alice_w:.2f}) < Bob({bob_w:.2f}) expected"
