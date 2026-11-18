"""Conversational teaching must produce provenance-trusted causal edges.

Regression for the flat learning curve: a taught mechanism used to create
14 causal edges with edge_kind='unknown', so assertable_chain declined all of
them. The fix routes asserted propositions through the schema-consistency
gate with a user-source record; the gate itself is unchanged.

Brain grounding (see docs/LEARNING_CURVE_CONSOLIDATION.md):
  McClelland, McNaughton & O'Reilly (1995), Psychological Review 102(3):419-457
  Squire & Zola (1996), PNAS 93(24):13515-13522
  Johnson & Raye (1981), Psychological Review 88(1):67-85
"""
import os
import sys

import numpy as np
import pytest

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_PROJ, "ravana", "src"),
           os.path.join(_PROJ, "ravana_ml", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.chain_walker import ChainWalkerMixin  # noqa: E402
from ravana.graph.engine import GraphEngine  # noqa: E402
from ravana.learn.intake import KnowledgeIntake  # noqa: E402
from ravana.learn.explanation import (  # noqa: E402
    assertable_chain, enumerate_chains, grade_chain,
)


MECHANISM = ("fog happens because chill causes condensation, "
             "and condensation causes fog.")
DEFINITION = ("marble is a kind of rock and refers to a mineral "
              "with a distinctive form.")


class _Shim:
    """Mirror the engine's _EngineGraphShim: _all_labels reads the host index."""

    def __init__(self, host):
        self.host = host

    @property
    def _all_labels(self):
        return {k: v[0] for k, v in self.host._concept_keywords.items() if v}

    @property
    def graph(self):
        return self.host.graph


class _Host(ChainWalkerMixin):
    """Minimal host exposing only what the mixin methods need."""

    def __init__(self):
        ge = GraphEngine(dim=16, seed=7)
        self.graph = ge.graph
        self._concept_keywords = {}
        self._concept_labels = set()
        self._intake = KnowledgeIntake(ge=_Shim(self))
        self._glove_vector = lambda label: np.zeros(16, dtype=np.float32)

    def _get_knowledge_intake(self):
        return self._intake


def _chain_for(host, label):
    ids = [nid for nid, node in host.graph.nodes.items()
           if (getattr(node, "label", "") or "").lower() == label]
    return [c for c in enumerate_chains(ids, host.graph, direction="backward",
                                        min_steps=2, max_steps=4, limit=8)
            if grade_chain(c).get("verdict") == "assert" and assertable_chain(c)]


def test_asserted_propositions_are_precise():
    host = _Host()
    assert host._asserted_causal_propositions(MECHANISM) == [
        ("chill", "condensation"), ("condensation", "fog")]
    assert host._asserted_causal_propositions(DEFINITION) == []


def test_question_asserts_nothing():
    host = _Host()
    assert host._consolidate_asserted_causal_facts(
        "why does fog happen?") == 0
    assert host._consolidate_asserted_causal_facts(
        "can you explain why fog occurs?") == 0


def test_teaching_writes_trusted_assertable_chain():
    host = _Host()
    written = host._consolidate_asserted_causal_facts(MECHANISM)
    assert written == 2
    chains = _chain_for(host, "fog")
    assert chains, "no trusted 2-hop chain after teaching"
    assert chains[0].labels == ["chill", "condensation", "fog"]
    for hop in chains[0].hops:
        meta = hop.source_metadata
        assert meta["edge_kind"] == "curated"      # gate's user-source map
        assert meta["source"] == "conversation"
        assert meta["source_type"] == "user"
        assert meta["operations"] == ["taught-in-conversation"]


def test_definition_does_not_become_a_causal_chain():
    host = _Host()
    assert host._consolidate_asserted_causal_facts(DEFINITION) == 0
    assert _chain_for(host, "marble") == []
    for _, edge in ((nid, e) for nid in list(host.graph.nodes)
                    for _, e in host.graph.get_outgoing(nid)):
        assert getattr(edge, "relation_type", "") != "causal"


def test_provenance_survives_pickle_round_trip():
    import pickle
    host = _Host()
    host._consolidate_asserted_causal_facts(MECHANISM)
    restored = pickle.loads(pickle.dumps(host.graph))
    ids = [nid for nid, node in restored.nodes.items()
           if (getattr(node, "label", "") or "").lower() == "fog"]
    chains = [c for c in enumerate_chains(ids, restored, direction="backward",
                                          min_steps=2, max_steps=4, limit=8)
              if grade_chain(c).get("verdict") == "assert" and assertable_chain(c)]
    assert chains and chains[0].labels == ["chill", "condensation", "fog"]
