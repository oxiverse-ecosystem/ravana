"""Tests for the explanation-chain assembler and its provenance gate.

The load-bearing test is `test_unprovenanced_chain_is_not_assertable`: the
live graph's causal edges are co-occurrence-derived and measurably false
(`gravity -> earth`, `air -> sound`), so a chain that grades 1.000
structurally must still be declined.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ravana.learn.explanation import (  # noqa: E402
    ExplanationChain, Hop, _hop_provenance, assertable_chain, enumerate_chains,
    grade_chain, provenance_trusted, score_chain,
)


class _Node:
    def __init__(self, label):
        self.label = label


class _Edge:
    def __init__(self, relation_type, weight=0.8, confidence=0.9, provenance=""):
        self.relation_type = relation_type
        self.weight = weight
        self.confidence = confidence
        self.provenance = provenance


class _Graph:
    """adj: {node_id: (label, [(target_id, edge), ...])}"""

    def __init__(self, adj):
        self._adj = {k: v[1] for k, v in adj.items()}
        self._nodes = {k: _Node(v[0]) for k, v in adj.items()}

    def get_outgoing(self, nid):
        return list(self._adj.get(nid, []))

    def get_node(self, nid):
        return self._nodes.get(nid)


class TestEnumeration:
    def test_only_causal_edges_are_traversed(self):
        g = _Graph({
            0: ("a", [(1, _Edge("causal")), (2, _Edge("semantic"))]),
            1: ("b", [(3, _Edge("causal"))]),
            2: ("c", [(3, _Edge("causal"))]),
            3: ("d", []),
        })
        chains = enumerate_chains([0], g, min_steps=2, max_steps=3)
        assert chains, "a 2-hop causal path exists"
        for c in chains:
            assert all(h.relation_type == "causal" for h in c.hops)

    def test_min_steps_is_respected(self):
        g = _Graph({0: ("a", [(1, _Edge("causal"))]), 1: ("b", [])})
        assert enumerate_chains([0], g, min_steps=2, max_steps=3) == []

    def test_no_repeated_nodes(self):
        g = _Graph({
            0: ("a", [(1, _Edge("causal"))]),
            1: ("b", [(0, _Edge("causal"))]),
        })
        for c in enumerate_chains([0], g, min_steps=2, max_steps=4):
            labels = [h.to_label for h in c.hops]
            assert len(labels) == len(set(labels))

    def test_longer_chains_are_penalised(self):
        short = [Hop("a", "causal", "b", 0.8, 0.9, "web")]
        long = short + [Hop("b", "causal", "c", 0.8, 0.9, "web")]
        assert score_chain(long) < score_chain(short)


class TestProvenanceGate:
    def test_web_and_definition_are_trusted(self):
        assert provenance_trusted(Hop("a", "causal", "b", provenance="web"))
        assert provenance_trusted(Hop("a", "causal", "b", provenance="definition"))

    def test_unprovenanced_is_not_trusted(self):
        assert not provenance_trusted(Hop("a", "causal", "b", provenance=""))

    def test_reads_the_graph_source_metadata(self):
        """The graph stamps edge_kind in source_metadata, not in an attribute.

        The first version read `edge.provenance`, which does not exist, so
        every web fact looked unprovenanced and was declined.
        """
        hop = Hop("smoking", "causal", "lung cancer", 0.5, 0.9,
                  source_metadata={"edge_kind": "web_fact",
                                   "source": "wikipedia"})
        assert provenance_trusted(hop)
        assert _hop_provenance(hop) == "web"

    def test_co_occurrence_metadata_is_not_trusted(self):
        hop = Hop("gravity", "causal", "earth", 0.85, 0.9,
                  source_metadata={"edge_kind": "co_occurrence"})
        assert not provenance_trusted(hop)

    def test_unprovenanced_chain_is_not_assertable(self):
        """The measured failure: structural adequacy 1.000 on a false chain."""
        chain = ExplanationChain(
            subject="gravity", direction="forward",
            hops=[Hop("gravity", "causal", "earth", 0.85, 0.90, ""),
                  Hop("earth", "causal", "life", 0.85, 0.90, "")],
            score=-1.7, runner_up_score=-1.8)
        grade = grade_chain(chain)
        assert grade["adequacy"] > 0.9, "structure alone looks perfect"
        assert grade["verdict"] == "decline", \
            "provenance must override a perfect structural grade"
        assert grade["provenance_trusted"] is False

    def test_one_untrusted_hop_sinks_the_chain(self):
        chain = ExplanationChain(
            subject="a", direction="forward",
            hops=[Hop("a", "causal", "b", 0.9, 0.9, "web"),
                  Hop("b", "causal", "c", 0.9, 0.9, "")],
            score=0.0, runner_up_score=0.0)
        assert not assertable_chain(chain)

    def test_all_trusted_chain_asserts(self):
        chain = ExplanationChain(
            subject="a", direction="forward",
            hops=[Hop("a", "causal", "b", 0.9, 0.9, "web"),
                  Hop("b", "causal", "c", 0.9, 0.9, "definition")],
            score=0.0, runner_up_score=-0.5)
        assert assertable_chain(chain)
        assert grade_chain(chain)["verdict"] == "assert"

    def test_empty_chain_declines(self):
        assert grade_chain(None)["verdict"] == "decline"
        assert not assertable_chain(None)
