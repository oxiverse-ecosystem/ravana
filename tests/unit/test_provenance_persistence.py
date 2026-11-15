"""Pin the provenance-persistence contract.

MEASURED DEFECT: `ConceptEdge.__getstate__` popped `source_metadata` with the
comment "heavy unused fields". It is the ONLY record of an edge's provenance
(`edge_kind`: web_fact / curated / co_occurrence), so every save+load silently
erased it:

    persisted edges: 380, edge_kind counts: {'(absent)': 380}

Every provenance gate then read "unprovenanced" after a reload.
"""
import os
import pickle
import sys

import pytest

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (PROJ, os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _graph():
    from ravana.graph.engine import GraphEngine
    return GraphEngine()


def _stamp(g, kind="web_fact", source="test"):
    from ravana.web.web_to_graph import WebToGraph
    w = WebToGraph(g, source=source)
    w.learn_text("Smoking causes lung cancer.", source_url=source)
    return list(g.graph.edges.values())[0]


class TestProvenanceSurvivesPickle:
    def test_web_fact_stamp_survives_round_trip(self):
        g = _graph()
        e = _stamp(g, "web_fact")
        assert (e.source_metadata or {}).get("edge_kind") == "web_fact"

        g2 = pickle.loads(pickle.dumps(g.graph))
        e2 = list(g2.edges.values())[0]
        assert (getattr(e2, "source_metadata", None) or {}).get(
            "edge_kind") == "web_fact", \
            "provenance was dropped by __getstate__"

    def test_metadata_is_a_dict_after_reload(self):
        g = _graph()
        _stamp(g)
        g2 = pickle.loads(pickle.dumps(g.graph))
        e2 = list(g2.edges.values())[0]
        assert isinstance(getattr(e2, "source_metadata", None), dict)

    def test_unstamped_edge_still_loads(self):
        """An edge with no metadata (a pre-existing persisted edge) must not
        break on reload -- it becomes an empty dict, not None."""
        g = _graph()
        w = __import__("ravana.web.web_to_graph", fromlist=["WebToGraph"])
        # build an edge with no stamp at all
        g.graph.add_edge(0, 1, weight=0.5, relation_type="semantic")
        g2 = pickle.loads(pickle.dumps(g.graph))
        e2 = list(g2.edges.values())[0]
        assert getattr(e2, "source_metadata", None) is not None

    def test_heavy_fields_still_excluded(self):
        """The fix must not start pickling the circular parent_graph."""
        g = _graph()
        _stamp(g)
        e = list(g.graph.edges.values())[0]
        state = e.__getstate__()
        assert "parent_graph" not in state
        assert "source_metadata" in state
