"""Tests for ravana_grace.core.dialogue_context."""

import pytest
import time
from ravana_grace.core.dialogue_context import (
    DialogueContext, ActiveSubgraph, Triple, DialogueState,
)


class TestTriple:
    def test_init(self):
        t = Triple(subject="heat", relation="causes", relation_type="causal", object="expansion")
        assert t.subject == "heat"
        assert t.relation == "causes"
        assert t.object == "expansion"

    def test_hash(self):
        t1 = Triple(subject="a", relation="is", relation_type="semantic", object="b")
        t2 = Triple(subject="a", relation="is", relation_type="semantic", object="b")
        assert hash(t1) == hash(t2)

    def test_defaults(self):
        t = Triple(subject="a", relation="is", relation_type="semantic", object="b")
        assert t.confidence == 0.5
        assert t.source_agent == "system"
        assert t.epistemic_status == "fact"


class TestActiveSubgraph:
    def test_init(self):
        sg = ActiveSubgraph()
        assert sg.decay_rate == 0.92
        assert sg.activation_threshold == 0.01

    def test_inject_triple(self):
        sg = ActiveSubgraph()
        t = Triple(subject="heat", relation="causes", relation_type="causal", object="expansion")
        sg.inject([t])
        assert len(sg) == 1
        assert "heat" in sg._active_concepts

    def test_decay_reduces_salience(self):
        sg = ActiveSubgraph()
        t = Triple(subject="heat", relation="causes", relation_type="causal", object="expansion")
        sg.inject([t])
        sg.decay()
        assert sg._active_concepts["heat"] < 1.0

    def test_get_active_edges(self):
        sg = ActiveSubgraph()
        t = Triple(subject="heat", relation="causes", relation_type="causal", object="expansion")
        sg.inject([t])
        edges = sg.get_active_edges()
        assert len(edges) == 1

    def test_get_active_concepts(self):
        sg = ActiveSubgraph()
        t = Triple(subject="heat", relation="causes", relation_type="causal", object="expansion")
        sg.inject([t])
        concepts = sg.get_active_concepts(threshold=0.01)
        assert "heat" in concepts

    def test_reset(self):
        sg = ActiveSubgraph()
        t = Triple(subject="heat", relation="causes", relation_type="causal", object="expansion")
        sg.inject([t])
        sg.reset()
        assert len(sg) == 0
        assert len(sg._active_concepts) == 0


class TestDialogueContext:
    def test_init(self):
        dc = DialogueContext(user_id="test_user")
        assert dc.user_id == "test_user"
        assert dc.turn_count == 0
        assert dc.active_subgraph is not None

    def test_process_turn(self):
        dc = DialogueContext(user_id="test_user")
        t = Triple(subject="heat", relation="causes", relation_type="causal", object="expansion")
        activations = dc.process_turn("heat causes expansion", [t])
        assert dc.turn_count == 1
        assert isinstance(activations, dict)

    def test_record_output(self):
        dc = DialogueContext(user_id="test_user")
        t = Triple(subject="heat", relation="causes", relation_type="causal", object="expansion")
        dc.record_output("Heat causes expansion.", [t])
        assert dc.last_output == "Heat causes expansion."

    def test_get_context(self):
        dc = DialogueContext(user_id="test_user")
        ctx = dc.get_context()
        assert ctx["user_id"] == "test_user"
        assert ctx["turn_count"] == 0
        assert "active_concepts" in ctx

    def test_get_state(self):
        dc = DialogueContext(user_id="test_user")
        state = dc.get_state()
        assert isinstance(state, DialogueState)
        assert state.user_id == "test_user"

    def test_store_user_belief(self):
        dc = DialogueContext(user_id="test_user")
        dc.store_user_belief("science_key", {"subject": "heat", "object": "expansion"})
        belief = dc.get_user_belief("science_key")
        assert belief is not None
        assert belief["strength"] == 0.5
        assert belief["access_count"] == 1

    def test_update_existing_belief(self):
        dc = DialogueContext(user_id="test_user")
        dc.store_user_belief("key1", {"subject": "a"})
        dc.store_user_belief("key1", {"subject": "a"})
        belief = dc.get_user_belief("key1")
        assert belief["strength"] > 0.5  # Increased

    def test_get_all_user_beliefs(self):
        dc = DialogueContext(user_id="test_user")
        dc.store_user_belief("k1", {"subject": "a"})
        dc.store_user_belief("k2", {"subject": "b"})
        beliefs = dc.get_all_user_beliefs()
        assert len(beliefs) == 2

    def test_clear_for_sleep(self):
        dc = DialogueContext(user_id="test_user")
        t = Triple(subject="heat", relation="causes", relation_type="causal", object="expansion")
        dc.process_turn("heat causes expansion", [t])
        dc.record_output("test", [t])
        dc.clear_for_sleep()
        assert dc.last_output == ""
        assert len(dc.active_subgraph) == 0

    def test_repr(self):
        dc = DialogueContext(user_id="test_user")
        r = repr(dc)
        assert "DialogueContext" in r
