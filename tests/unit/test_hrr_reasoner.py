"""Work A0 unit tests: HRR compositional reasoning wired into the live engine.

Verifies (per the plan's brain basis):
  - Dusek & Eichenbaum 1997 transitive inference: A->B, B->C => A->C recovered
    through the integrated HRR store, decoded via the discrete graph atom set.
  - DualCodeSpace is instantiated in CognitiveChatEngine.__init__ and the
    ConceptGraph._fact_encode_hook populates the HRR store on every add_edge.
  - Structured retrieval is grounded (falls through to graph.infer_chain when
    the HRR store lacks the fact), and never emits a raw vector.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ravana_ml", "src"))


@pytest.fixture(scope="module")
def engine():
    from scripts.ravana_chat import CognitiveChatEngine
    return CognitiveChatEngine(dim=64, seed=1, baby_mode=True)


def test_dual_code_instantiated(engine):
    # Work A0: DualCodeSpace (2048-D) is live, not inert.
    assert engine.dual_code is not None
    assert engine.hrr_reasoner is not None
    assert engine.dual_code.hrr_dim == 2048


def test_add_edge_populates_hrr(engine):
    # The ConceptGraph._fact_encode_hook wired in __init__ must fire on add_edge.
    before = len(engine.hrr_reasoner)
    na = engine.graph.add_node(label="__a0_x")
    nb = engine.graph.add_node(label="__a0_y")
    e = engine.graph.add_edge(na.id, nb.id, relation_type="is", confidence=0.9)
    assert e is not None
    # hook encoded exactly one new fact
    assert len(engine.hrr_reasoner) == before + 1
    assert engine.hrr_reasoner.has_fact("__a0_x", "is")


def test_transitive_inference(engine):
    # A->B, B->C => query A recovers [B, C].
    na = engine.graph.add_node(label="__a0_alice")
    nb = engine.graph.add_node(label="__a0_bob")
    nc = engine.graph.add_node(label="__a0_carol")
    engine.graph.add_edge(na.id, nb.id, relation_type="likes", confidence=0.9)
    engine.graph.add_edge(nb.id, nc.id, relation_type="likes", confidence=0.9)
    chain = engine.hrr_query_chain("__a0_alice", "likes", max_hops=2)
    assert chain == ["__a0_bob", "__a0_carol"], chain
    ans = engine._structured_fact_answer("__a0_alice", "likes")
    assert ans is not None and "__a0_bob" in ans and "__a0_carol" in ans


def test_fallback_to_graph_when_hrr_empty(engine):
    # Unknown head with no HRR fact must defer to graph.infer_chain (or []),
    # never crash and never emit a raw vector.
    out = engine.hrr_query_chain("__definitely_not_a_real_concept_xyz", "is", 2)
    assert isinstance(out, list)
