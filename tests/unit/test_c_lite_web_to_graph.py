"""
Tests for C-lite: web -> OpenIE -> typed graph + EFE knowledge-gap sketch.
"""

import os
import sys

import numpy as np
import pytest

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))

from ravana.graph.engine import GraphEngine
from ravana.web.openie import OpenIEExtractor, Fact
from ravana.web.web_to_graph import WebToGraph, KnowledgeGap


def _seed_ge() -> GraphEngine:
    ge = GraphEngine(dim=64, glove_vecs=None)
    for w in ["water", "earth", "sun", "paris", "france", "plant", "leaf",
              "fire", "smoke", "ice", "star", "city", "country"]:
        vec = np.random.RandomState(hash(w) % 1000).randn(64).astype("float32")
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        node = ge.graph.add_node(vector=vec, label=w)
        ge._all_labels[w] = node.id
    return ge


def _typed_edge_count(ge: GraphEngine) -> int:
    labels = list(ge._all_labels.keys())
    n = 0
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            sa, sb = ge._all_labels[la], ge._all_labels[lb]
            for a, b in ((sa, sb), (sb, sa)):
                e = ge.graph.get_edge(a, b)
                if e is not None and e.relation_type in (
                        "is_a", "has_property", "causes", "located_in", "part_of"):
                    n += 1
    return n


class TestOpenIEExtractor:
    def test_extracts_is_a(self):
        ex = OpenIEExtractor()
        facts = ex.extract("Water is a chemical compound. The sun is a star.")
        rels = {f.relation for f in facts}
        assert "is_a" in rels
        subs = {f.subject for f in facts}
        assert "water" in subs

    def test_extracts_causes_and_has_property(self):
        ex = OpenIEExtractor()
        facts = ex.extract("Fire causes smoke. A plant has a leaf.")
        rels = {f.relation for f in facts}
        assert "causes" in rels
        assert "has_property" in rels

    def test_skips_non_factual_sentence(self):
        ex = OpenIEExtractor()
        facts = ex.extract("Hello there, how are you today?")
        assert facts == []


class TestWebToGraph:
    def test_writes_typed_edges_no_dim_change(self):
        ge = _seed_ge()
        w2g = WebToGraph(ge, source="test")
        dim_before = ge.dim
        n = w2g.learn_text("Water is a chemical compound. Paris is located in France.")
        assert ge.dim == dim_before
        assert n > 0
        assert _typed_edge_count(ge) > 0

    def test_dedup_does_not_explode(self):
        ge = _seed_ge()
        w2g = WebToGraph(ge, source="test")
        w2g.learn_text("Fire causes smoke.")
        n2 = w2g.learn_text("Fire causes smoke. Fire causes smoke.")
        assert n2 == 0  # repeated fact dedups (Hebbian strengthen, not duplicate)

    def test_efe_gap_sparse_topic_higher(self):
        ge = _seed_ge()
        w2g = WebToGraph(ge, source="test")
        w2g.learn_text("Water is a chemical compound. The earth is a planet.")
        gap_known = w2g.knowledge_gap("water")
        gap_novel = w2g.knowledge_gap("quantum entanglement")
        assert gap_novel.efe > gap_known.efe
        assert gap_novel.is_curiosity_target()

    def test_knowledge_gap_dataclass(self):
        g = KnowledgeGap(topic="x", known_edges=1, efe=5.0)
        assert g.is_curiosity_target(1.0)
        assert not g.is_curiosity_target(10.0)


class TestWebLearnerIntegration:
    def test_web_learner_has_c_lite_hooks(self):
        # Importability + the hooks exist without instantiating the heavy learner
        import ravana.web.learner as wl
        assert hasattr(wl.WebLearner, "knowledge_gap")
        assert hasattr(wl.WebLearner, "last_fact_count")
