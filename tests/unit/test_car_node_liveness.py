"""Regression tests for the CAR k=2 stale-node-id root cause.

Measured on 7/10 sealed k=2 items: junk-node pruning deleted freshly taught
concept nodes while their ids stayed in the label index, so later writes
attached curated causal hops to DEAD endpoints. Enumeration walks live nodes
only -> raw_chains=0 in-process. Fixes: (F4) never prune a node carrying an
earned causal edge; (F1) sync the label index after pruning; (F2) resolve
labels through live nodes only at the write choke point.
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

from ravana_ml.graph import ConceptGraph
from ravana.web.web_to_graph import WebToGraph


def _node(g, label):
    n = g.add_node(vector=np.zeros(8, dtype=np.float32), label=label)
    return n.id


def test_trusted_causal_node_survives_junk_prune():
    g = ConceptGraph(dim=8)
    a, b, x = _node(g, "cause_a"), _node(g, "effect_b"), _node(g, "target_x")
    e1 = g.add_edge(a, b, weight=0.5, relation_type="causal", confidence=0.5)
    e1.source_metadata = {"edge_kind": "curated", "source": "http://teach.local/x/sourceA"}
    e2 = g.add_edge(b, x, weight=0.5, relation_type="causal", confidence=0.5)
    e2.source_metadata = {"edge_kind": "curated", "source": "http://teach.local/x/sourceA"}
    # All three are degree 1/1 singletons not adjacent to a hub; an unprovenanced
    # node with the same shape would be pruned.
    junk = _node(g, "random_scrape_token")
    count, labels, hubs = g.prune_low_degree_junk_nodes(tau_low=1, tau_high=8)
    assert a in g.nodes and b in g.nodes and x in g.nodes
    assert junk not in g.nodes
    assert "random_scrape_token" in labels


def test_unprovenanced_singleton_still_pruned():
    g = ConceptGraph(dim=8)
    junk = _node(g, "one_shot_token")
    count, labels, hubs = g.prune_low_degree_junk_nodes(tau_low=1, tau_high=8)
    assert junk not in g.nodes


def test_ensure_node_ignores_dead_index_id():
    """A stale first id must not be reused as an edge endpoint."""
    class _GE:
        def __init__(self):
            self.graph = ConceptGraph(dim=8)
            self._all_labels = {}
            self._concept_keywords = {}
            self._promote_min_sources = 1
            self._junk_theta = 0.5

    ge = _GE()
    live = ge.graph.add_node(vector=np.zeros(8, dtype=np.float32), label="tmp").id
    ge.graph.remove_node(live)
    ge._concept_keywords["bruise"] = [live]          # stale id
    ge._all_labels["bruise"] = live
    w = WebToGraph.__new__(WebToGraph)
    w.ge = ge
    fresh = w._ensure_node("bruise", source_url="http://teach.local/bruise/sourceA")
    assert fresh is not None
    assert ge.graph.get_node(fresh) is not None
    assert fresh != live
    assert live not in ge._concept_keywords.get("bruise", [])


def test_trusted_definition_node_survives_junk_prune():
    """Measured cause of the grounded re-learning T1->T2 loss: a just-taught
    is_a/semantic fact with earned provenance was pruned by the junk sweep."""
    g = ConceptGraph(dim=8)
    a, b = _node(g, "asteroid_fact"), _node(g, "rocky_body")
    e = g.add_edge(a, b, weight=0.5, relation_type="is_a", confidence=0.6)
    e.source_metadata = {"edge_kind": "web_fact", "source": "http://grounded.local/asteroid"}
    junk = _node(g, "scrape_noise_token")
    g.prune_low_degree_junk_nodes(tau_low=1, tau_high=8)
    assert a in g.nodes and b in g.nodes
    assert junk not in g.nodes


def test_fix_source_present():
    src = open(os.path.join(PROJ, "ravana", "src", "ravana", "chat",
                            "engine_generation.py"), encoding="utf-8").read()
    assert "_ids_before - _ids_after" in src
