"""
Pin the derived (non-hand-listed) behaviors introduced in the de-hardcoding
audit:
  * _derive_definition_purge: closed-class seed + graph-derived abstract
    attractors (not a frozen 50-word list).
  * AbstractionEngine._walk_hierarchy: graph IS_A/level ascent is primary,
    ABSTRACTION_HIERARCHY is only a fallback seed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ravana_ml", "src"))

from ravana.chat.engine import CognitiveChatEngine
from ravana_ml.graph import ConceptGraph, ConceptNode
from ravana.core.abstraction_engine import AbstractionEngine, AbstractionConfig


def _bare_engine():
    eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
    # Minimal graph so _derive_definition_purge has something to walk.
    g = ConceptGraph(dim=16)
    # abstract hub: high degree, high abstraction_degree
    hub = g.add_node(label="meaning", vector=np.zeros(16, dtype=np.float32))
    hub.abstraction_degree = 0.9
    hub.level = 3
    leaf = g.add_node(label="purpose", vector=np.zeros(16, dtype=np.float32))
    for _ in range(5):
        n = g.add_node(vector=np.zeros(16, dtype=np.float32))
        g.add_edge(hub.id, n.id, weight=0.5)
    g.add_edge(leaf.id, hub.id, weight=0.5)
    eng.graph = g
    return eng


def test_derive_definition_purge_includes_closed_class_and_derived_attractors():
    eng = _bare_engine()
    purge = eng._derive_definition_purge()
    # Closed-class universal seed present.
    assert "you" in purge and "i" in purge and "they" in purge
    # Derived abstract attractor ("meaning" is a high-degree, high-abstractness
    # hub) is included WITHOUT being hand-listed.
    assert "meaning" in purge
    # A low-degree concrete leaf is NOT purged.
    assert "purpose" not in purge


def test_definition_purge_falls_back_to_seed_without_graph():
    eng = CognitiveChatEngine.__new__(CognitiveChatEngine)
    eng.graph = None
    purge = eng._derive_definition_purge()
    assert "you" in purge
    assert "meaning" not in purge  # nothing derived without a graph


def test_walk_hierarchy_prefers_graph_over_seed():
    g = ConceptGraph(dim=16)
    child = g.add_node(label="cat", vector=np.zeros(16, dtype=np.float32))
    parent = g.add_node(label="animal", vector=np.zeros(16, dtype=np.float32))
    parent.level = 2
    g.add_edge(child.id, parent.id, relation_type="isa", weight=0.8)
    eng = AbstractionEngine(g, AbstractionConfig(max_hierarchy_depth=3))
    path = eng._walk_hierarchy("cat")
    # Graph edge is the primary path; "animal" is reached from the graph.
    assert "cat" in path and "animal" in path


def test_walk_hierarchy_seed_fallback_when_no_graph_edges():
    g = ConceptGraph(dim=16)
    g.add_node(label="justice", vector=np.zeros(16, dtype=np.float32))
    eng = AbstractionEngine(g, AbstractionConfig(max_hierarchy_depth=3))
    path = eng._walk_hierarchy("justice")
    # No graph hierarchy -> seeded fallback supplies superordinates.
    assert "justice" in path
    assert any(p != "justice" for p in path)
