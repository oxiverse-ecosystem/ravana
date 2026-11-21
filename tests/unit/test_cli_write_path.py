"""Regression tests for the CLI write-time criterion fix (P2).

Two measured defects blocked A-teaching criterion (before sleep):
1. A cause with two different effects was treated as a CONTRADICTION, so the
   second taught hop was buffered and never reached the graph. Causal relations
   are many-to-many; slot exclusivity belongs to functional relations.
2. When a fact asserted a causal relation on a pre-existing SEMANTIC edge,
   _write_edge bumped confidence but never upgraded relation_type, so the chain
   detector skipped the hop. Reconsolidation must rewrite the trace's type.
"""
from __future__ import annotations

import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (PROJ, os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana_ml", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.web.openie import Fact
from ravana.graph.engine import GraphEngine
from ravana.learn.intake import (SchemaConsistencyGate, KnowledgeIntake,
                                 Provenance)

DIM = 16


def _engine(labels):
    ge = GraphEngine(dim=DIM, seed=7)
    for lab in labels:
        n = ge.graph.add_node(vector=np.zeros(DIM, dtype=np.float32), label=lab)
        ge._all_labels[lab] = n.id
    return ge


def _causal(ge, subj, obj, conf=0.6):
    a, b = ge._all_labels[subj], ge._all_labels[obj]
    return ge.graph.add_edge(source=a, target=b, weight=0.5,
                             relation_type="causal", confidence=conf)


def test_causal_effects_are_not_a_contradiction():
    ge = _engine(["cooling", "condensation", "moisture"])
    _causal(ge, "cooling", "condensation", conf=0.9)
    gate = SchemaConsistencyGate()
    fact = Fact(subject="cooling", relation="causes", obj="moisture",
                confidence=0.7)
    prov = Provenance(source="http://teach.local/dew", source_type="seed")
    assert gate.detect_contradiction(fact, ge, prov) is None
    v = gate.evaluate(fact, ge, prov, salience=1.0)
    assert v.route != "episodic_buffer"


def test_functional_relation_still_contradicts():
    # The relaxation must NOT remove one-to-one slot exclusivity from
    # functional relations (the legitimate use of contradiction detection).
    ge = _engine(["paris", "france", "germany"])
    a, b = ge._all_labels["paris"], ge._all_labels["france"]
    ge.graph.add_edge(source=a, target=b, weight=0.5,
                      relation_type="is_a", confidence=0.9)
    gate = SchemaConsistencyGate()
    fact = Fact(subject="paris", relation="is_a", obj="germany", confidence=0.7)
    assert gate.detect_contradiction(fact, ge, Provenance()) is not None


def test_existing_semantic_edge_upgraded_to_causal():
    ge = _engine(["moisture", "dew"])
    a, b = ge._all_labels["moisture"], ge._all_labels["dew"]
    e = ge.graph.add_edge(source=a, target=b, weight=0.5,
                          relation_type="semantic", confidence=0.65)
    assert e.relation_type == "semantic"
    ki = KnowledgeIntake(ge=ge)
    fact = Fact(subject="moisture", relation="causes", obj="dew", confidence=0.7)
    prov = Provenance(source="http://teach.local/dew", source_type="seed")
    v = ki.gate.evaluate(fact, ge, prov, salience=1.0)
    ki.intake(fact, prov, salience=1.0)
    assert e.relation_type == "causal", v.route
    # Never downgrade a causal edge when a semantic fact arrives.
    fact2 = Fact(subject="moisture", relation="is_a", obj="dew", confidence=0.7)
    ki.intake(fact2, prov, salience=1.0)
    assert e.relation_type == "causal"


def test_seed_source_type_is_declared_explicitly():
    from ravana.web.web_to_graph import WebToGraph
    w = WebToGraph.__new__(WebToGraph)
    p_web = w._provenance("http://random.example/page")
    p_seed = w._provenance("http://teach.local/x", source_type="seed")
    assert p_web.source_type == "web" and p_web.base_reliability() == 0.5
    assert p_seed.source_type == "seed" and p_seed.base_reliability() >= 0.7
