"""Tests for context-dependent edge gating in ConceptGraph.

Demonstrates how the same graph can represent fuzzy/ambiguous knowledge
without hardcoding, by activating different edges based on context.

Three types of ambiguity are tested:
1. Idiom: "spill the beans" means "reveal secrets" only in an idiom context
2. Metaphor: "time is money" maps to "save/waste" only in a metaphor context
3. Sensory ambiguity: "bank" maps to "river" in nature context, "money" in finance
"""

import pytest
import numpy as np
from ravana_ml.graph import ConceptGraph, ConceptEdge


class TestContextGating:
    """Test the context-dependent edge gating mechanism."""

    def _make_graph(self):
        """Create a small graph with context-dependent edges."""
        g = ConceptGraph(dim=16, max_nodes=100, max_edges=1000,
                        rng=np.random.RandomState(42))
        # Add nodes for ambiguous concepts
        bank = g.add_node(np.random.randn(16).astype(np.float32), label="bank")
        river = g.add_node(np.random.randn(16).astype(np.float32), label="river")
        money = g.add_node(np.random.randn(16).astype(np.float32), label="money")
        # Context-independent edge (always active)
        g.add_edge(bank.id, river.id, weight=0.3, relation_type="semantic",
                   confidence=0.3)
        # Context-dependent edges
        g.add_context_edge(bank.id, river.id, context_labels=["nature"],
                           weight=0.8, relation_type="semantic", confidence=0.8)
        g.add_context_edge(bank.id, money.id, context_labels=["finance"],
                           weight=0.9, relation_type="semantic", confidence=0.9)
        return g, bank, river, money

    def test_no_context_all_edges_active(self):
        """Without context set, all edges are active (no filtering)."""
        g, bank, river, money = self._make_graph()
        # No context set — all edges should be active
        edge_bank_river = g.get_edge(bank.id, river.id)
        edge_bank_money = g.get_edge(bank.id, money.id)
        assert g.is_edge_active(edge_bank_river)
        assert g.is_edge_active(edge_bank_money)

    def test_context_activates_matching_edges(self):
        """Setting context activates only matching context-labeled edges."""
        g, bank, river, money = self._make_graph()
        g.set_context(["nature"])
        # The nature-context bank->river edge should be active
        edge_bank_river = g.get_edge(bank.id, river.id)
        assert g.is_edge_active(edge_bank_river)
        # The finance-context bank->money edge should be inactive
        edge_bank_money = g.get_edge(bank.id, money.id)
        assert not g.is_edge_active(edge_bank_money)

    def test_context_switch_changes_active_edges(self):
        """Switching context changes which edges are active."""
        g, bank, river, money = self._make_graph()
        # Finance context
        g.set_context(["finance"])
        edge_bank_money = g.get_edge(bank.id, money.id)
        edge_bank_river = g.get_edge(bank.id, river.id)
        assert g.is_edge_active(edge_bank_money)
        assert not g.is_edge_active(edge_bank_river)
        # Switch to nature context
        g.set_context(["nature"])
        assert not g.is_edge_active(edge_bank_money)
        assert g.is_edge_active(edge_bank_river)

    def test_clear_context_reactivates_all(self):
        """Clearing context makes all edges active again."""
        g, bank, river, money = self._make_graph()
        g.set_context(["finance"])
        edge_bank_river = g.get_edge(bank.id, river.id)
        assert not g.is_edge_active(edge_bank_river)
        g.clear_context()
        assert g.is_edge_active(edge_bank_river)

    def test_context_independent_edges_always_active(self):
        """Edges without context_labels are always active regardless of context."""
        g = ConceptGraph(dim=16, max_nodes=100, max_edges=1000,
                        rng=np.random.RandomState(42))
        a = g.add_node(np.random.randn(16).astype(np.float32), label="a")
        b = g.add_node(np.random.randn(16).astype(np.float32), label="b")
        g.add_edge(a.id, b.id, weight=0.5, relation_type="semantic")
        g.set_context(["some_context"])
        edge = g.get_edge(a.id, b.id)
        assert g.is_edge_active(edge)

    def test_get_context_edges(self):
        """get_context_edges returns only edges with the given label."""
        g, bank, river, money = self._make_graph()
        nature_edges = g.get_context_edges("nature")
        finance_edges = g.get_context_edges("finance")
        assert len(nature_edges) >= 1
        assert len(finance_edges) >= 1
        # Verify the edges have the right context labels
        for edge in nature_edges:
            assert "nature" in edge.context_labels
        for edge in finance_edges:
            assert "finance" in edge.context_labels


class TestIdiomAmbiguity:
    """Test that idiomatic expressions are handled via context gating.

    "Spill the beans" literally means spilling legumes, but idiomatically
    means revealing secrets. The same graph can represent both meanings
    by activating different edges based on whether an idiom context is set.
    """

    def test_idiom_context_gating(self):
        g = ConceptGraph(dim=16, max_nodes=100, max_edges=1000,
                        rng=np.random.RandomState(42))
        spill = g.add_node(np.random.randn(16).astype(np.float32), label="spill")
        beans = g.add_node(np.random.randn(16).astype(np.float32), label="beans")
        secrets = g.add_node(np.random.randn(16).astype(np.float32), label="secrets")
        # Literal meaning (always available but weak)
        g.add_edge(spill.id, beans.id, weight=0.3, relation_type="semantic",
                   confidence=0.3)
        # Idiomatic meaning (only in idiom context)
        g.add_context_edge(spill.id, secrets.id, context_labels=["idiom"],
                           weight=0.85, relation_type="semantic", confidence=0.85)
        # Without idiom context, spill->secrets is inactive
        g.set_context(["literal"])
        edge_spill_secrets = g.get_edge(spill.id, secrets.id)
        assert not g.is_edge_active(edge_spill_secrets)
        # With idiom context, spill->secrets is active
        g.set_context(["idiom"])
        assert g.is_edge_active(edge_spill_secrets)
        # Literal edge is always active (context-independent)
        edge_spill_beans = g.get_edge(spill.id, beans.id)
        assert g.is_edge_active(edge_spill_beans)


class TestMetaphorAmbiguity:
    """Test that metaphors are handled via context gating.

    "Time is money" is a metaphor that maps time-related concepts to
    money-related concepts (save, waste, spend). This mapping should only
    be active in a metaphor context.
    """

    def test_metaphor_context_gating(self):
        g = ConceptGraph(dim=16, max_nodes=100, max_edges=1000,
                        rng=np.random.RandomState(42))
        time_n = g.add_node(np.random.randn(16).astype(np.float32), label="time")
        money = g.add_node(np.random.randn(16).astype(np.float32), label="money")
        save = g.add_node(np.random.randn(16).astype(np.float32), label="save")
        waste = g.add_node(np.random.randn(16).astype(np.float32), label="waste")
        # Metaphor mapping: time -> money (only in metaphor context)
        g.add_context_edge(time_n.id, money.id, context_labels=["metaphor"],
                           weight=0.8, relation_type="analogical", confidence=0.8)
        # Metaphor mapping: time -> save (only in metaphor context)
        g.add_context_edge(time_n.id, save.id, context_labels=["metaphor"],
                           weight=0.7, relation_type="analogical", confidence=0.7)
        # Without metaphor context, time->money is inactive
        g.set_context(["literal"])
        edge_time_money = g.get_edge(time_n.id, money.id)
        assert not g.is_edge_active(edge_time_money)
        # With metaphor context, time->money is active
        g.set_context(["metaphor"])
        assert g.is_edge_active(edge_time_money)
        assert g.is_edge_active(g.get_edge(time_n.id, save.id))


class TestSensoryAmbiguity:
    """Test that sensory/polysemous ambiguity is handled via context gating.

    "Bank" can mean a financial institution or a river bank. The same
    graph node can activate different neighbors depending on context.
    """

    def test_sensory_ambiguity_context_gating(self):
        g = ConceptGraph(dim=16, max_nodes=100, max_edges=1000,
                        rng=np.random.RandomState(42))
        bank = g.add_node(np.random.randn(16).astype(np.float32), label="bank")
        river = g.add_node(np.random.randn(16).astype(np.float32), label="river")
        water = g.add_node(np.random.randn(16).astype(np.float32), label="water")
        money = g.add_node(np.random.randn(16).astype(np.float32), label="money")
        interest = g.add_node(np.random.randn(16).astype(np.float32), label="interest")
        # Nature context: bank -> river, bank -> water
        g.add_context_edge(bank.id, river.id, context_labels=["nature"],
                           weight=0.8, relation_type="semantic", confidence=0.8)
        g.add_context_edge(bank.id, water.id, context_labels=["nature"],
                           weight=0.7, relation_type="semantic", confidence=0.7)
        # Finance context: bank -> money, bank -> interest
        g.add_context_edge(bank.id, money.id, context_labels=["finance"],
                           weight=0.9, relation_type="semantic", confidence=0.9)
        g.add_context_edge(bank.id, interest.id, context_labels=["finance"],
                           weight=0.8, relation_type="semantic", confidence=0.8)
        # Nature context activates river/water, not money/interest
        g.set_context(["nature"])
        assert g.is_edge_active(g.get_edge(bank.id, river.id))
        assert g.is_edge_active(g.get_edge(bank.id, water.id))
        assert not g.is_edge_active(g.get_edge(bank.id, money.id))
        assert not g.is_edge_active(g.get_edge(bank.id, interest.id))
        # Finance context activates money/interest, not river/water
        g.set_context(["finance"])
        assert not g.is_edge_active(g.get_edge(bank.id, river.id))
        assert not g.is_edge_active(g.get_edge(bank.id, water.id))
        assert g.is_edge_active(g.get_edge(bank.id, money.id))
        assert g.is_edge_active(g.get_edge(bank.id, interest.id))


class TestContextGatedSpreadingActivation:
    """Test that spreading activation respects context gating."""

    def test_spreading_respects_context(self):
        """Edges inactive in current context should not propagate activation."""
        g = ConceptGraph(dim=16, max_nodes=100, max_edges=1000,
                        rng=np.random.RandomState(42))
        a = g.add_node(np.random.randn(16).astype(np.float32), label="a")
        b = g.add_node(np.random.randn(16).astype(np.float32), label="b")
        c = g.add_node(np.random.randn(16).astype(np.float32), label="c")
        # a -> b (always active)
        g.add_edge(a.id, b.id, weight=0.8, relation_type="semantic")
        # a -> c (only in "special" context)
        g.add_context_edge(a.id, c.id, context_labels=["special"],
                           weight=0.8, relation_type="semantic")
        # Without context, a->c is active — use relation_type to force fine-grained path
        g.clear_context()
        g.reset_activation()
        g.activate(a.id, 1.0)
        g.spread_activation(steps=2, k_active=5, decay=0.5, relation_type="semantic")
        # Both b and c should have activation
        assert g.nodes[b.id].activation > 0.01
        assert g.nodes[c.id].activation > 0.01
        # With "other" context, a->c is inactive
        g.set_context(["other"])
        g.reset_activation()
        g.activate(a.id, 1.0)
        g.spread_activation(steps=2, k_active=5, decay=0.5, relation_type="semantic")
        # b should have activation, c should not
        assert g.nodes[b.id].activation > 0.01
        assert g.nodes[c.id].activation < 0.01

    def test_spreading_with_matching_context(self):
        """With matching context, context-labeled edges should propagate."""
        g = ConceptGraph(dim=16, max_nodes=100, max_edges=1000,
                        rng=np.random.RandomState(42))
        a = g.add_node(np.random.randn(16).astype(np.float32), label="a")
        b = g.add_node(np.random.randn(16).astype(np.float32), label="b")
        c = g.add_node(np.random.randn(16).astype(np.float32), label="c")
        g.add_edge(a.id, b.id, weight=0.8, relation_type="semantic")
        g.add_context_edge(a.id, c.id, context_labels=["special"],
                           weight=0.8, relation_type="semantic")
        # With "special" context, a->c should propagate
        g.set_context(["special"])
        g.reset_activation()
        g.activate(a.id, 1.0)
        g.spread_activation(steps=2, k_active=5, decay=0.5)
        assert g.nodes[c.id].activation > 0.01


class TestContextGatedInference:
    """Test that multi-hop inference respects context gating."""

    def test_inference_respects_context(self):
        """Context-inactive edges should not be traversed during inference."""
        g = ConceptGraph(dim=16, max_nodes=100, max_edges=1000,
                        rng=np.random.RandomState(42))
        a = g.add_node(np.random.randn(16).astype(np.float32), label="a")
        b = g.add_node(np.random.randn(16).astype(np.float32), label="b")
        c = g.add_node(np.random.randn(16).astype(np.float32), label="c")
        # a -> b (always active)
        g.add_edge(a.id, b.id, weight=0.8, relation_type="semantic")
        # b -> c (only in "special" context)
        g.add_context_edge(b.id, c.id, context_labels=["special"],
                           weight=0.8, relation_type="semantic")
        # Without context, b->c is active, so a->b->c should be found
        g.clear_context()
        results = g.infer_chain(a.id, max_hops=2)
        result_ids = [r[0] for r in results]
        assert c.id in result_ids
        # With "other" context, b->c is inactive, so a->b->c should NOT be found
        g.set_context(["other"])
        results = g.infer_chain(a.id, max_hops=2)
        result_ids = [r[0] for r in results]
        assert c.id not in result_ids


class TestContextEdgePickleCompatibility:
    """Test that context edges survive pickle round-trips."""

    def test_context_edges_pickle_roundtrip(self):
        """Context labels and context should survive pickling."""
        import pickle
        g = ConceptGraph(dim=16, max_nodes=100, max_edges=1000,
                        rng=np.random.RandomState(42))
        a = g.add_node(np.random.randn(16).astype(np.float32), label="a")
        b = g.add_node(np.random.randn(16).astype(np.float32), label="b")
        g.add_context_edge(a.id, b.id, context_labels=["test_context"],
                           weight=0.7, relation_type="semantic")
        g.set_context(["test_context"])
        # Pickle and unpickle
        data = pickle.dumps(g)
        g2 = pickle.loads(data)
        # Context should be preserved
        assert g2.get_context() == {"test_context"}
        # Edge should be active
        edge = g2.get_edge(a.id, b.id)
        assert edge is not None
        assert g2.is_edge_active(edge)
        assert edge.context_labels == ["test_context"]

    def test_backward_compatible_unpickle(self):
        """Graphs saved before context gating should load without errors."""
        import pickle
        # Create a graph without context features
        g = ConceptGraph(dim=16, max_nodes=100, max_edges=1000,
                        rng=np.random.RandomState(42))
        a = g.add_node(np.random.randn(16).astype(np.float32), label="a")
        b = g.add_node(np.random.randn(16).astype(np.float32), label="b")
        g.add_edge(a.id, b.id, weight=0.5, relation_type="semantic")
        # Simulate an old pickle by removing context attributes
        data = pickle.dumps(g)
        g2 = pickle.loads(data)
        # Should have default empty context
        assert g2.get_context() == set()
        # All edges should be active (no context filtering)
        edge = g2.get_edge(a.id, b.id)
        assert g2.is_edge_active(edge)
