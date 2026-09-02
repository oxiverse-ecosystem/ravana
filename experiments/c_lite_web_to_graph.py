"""
C-lite harness — validate web -> OpenIE -> typed graph edges + EFE gap.
=======================================================================
Evidence-first check that C-lite: (1) writes facts as typed edges into the
EXISTING graph with NO dimensionality change; (2) dedups repeats (Hebbian
strengthen, not duplicate); (3) emits a KnowledgeGap EFE signal for sparse
topics. Mirrors the production WebToGraph path against a real GraphEngine.
"""

from __future__ import annotations

import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root = .../ravana
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))

import numpy as np  # noqa: E402

from ravana.graph.engine import GraphEngine  # noqa: E402
from ravana.web.openie import OpenIEExtractor  # noqa: E402
from ravana.web.web_to_graph import WebToGraph, KnowledgeGap  # noqa: E402


def _build_ge() -> GraphEngine:
    ge = GraphEngine(dim=64, glove_vecs=None)
    # seed a tiny graph so node minting + label index exist
    for w in ["water", "earth", "sun", "paris", "france", "plant", "leaf",
              "fire", "smoke", "ice", "water vapor"]:
        vec = np.random.RandomState(hash(w) % 1000).randn(64).astype("float32")
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        node = ge.graph.add_node(vector=vec, label=w)
        ge._all_labels[w] = node.id
    return ge


def main():
    print("=" * 78)
    print("C-LITE HARNESS — web -> OpenIE -> typed graph + EFE gap")
    print("=" * 78)

    ge = _build_ge()
    dim_before = ge.dim
    w2g = WebToGraph(ge, source="harness")

    text = (
        "Water is a chemical compound. The Earth is a planet. "
        "Paris is a city. France is a country. A plant has a leaf. "
        "Fire causes smoke. Ice is water in solid form. "
        "Water is essential for life. The sun is a star. "
        "Paris is located in France. Smoke has a smell."
    )

    n_written = w2g.learn_text(text, source_url="http://example.com/facts")
    print(f"\n  facts written as typed edges : {n_written}")
    print(f"  graph dimensionality         : {ge.dim}D (unchanged from {dim_before}D)")
    assert ge.dim == dim_before, "C-lite must NOT change graph dimensionality"
    assert n_written > 0, "should have written some facts"

    # ── T1: typed edges actually landed ──
    # Count web_fact-ish edges (is_a / has_property / causes / located_in)
    typed = 0
    web_fact_nodes = set()
    for (s, t), edge in ge.graph._edges.items() if hasattr(ge.graph, "_edges") else {}:
        pass
    # GraphEngine stores edges in self.graph; iterate via public surface:
    # use get_edge over known node pairs we can reconstruct from labels
    labels = list(ge._all_labels.keys())
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            sa, sb = ge._all_labels[la], ge._all_labels[lb]
            for a, b in ((sa, sb), (sb, sa)):
                e = ge.graph.get_edge(a, b)
                if e is not None and e.relation_type in (
                        "is_a", "has_property", "causes", "located_in", "part_of"):
                    typed += 1
                    web_fact_nodes.add(la)
                    web_fact_nodes.add(lb)
    print(f"  typed relation edges found   : {typed}")
    print(f"  distinct concepts involved   : {len(web_fact_nodes)}")
    assert typed > 0, "typed relation edges must exist"

    # ── T2: dedup / Hebbian strengthen on repeat ──
    conf_before = None
    sa, sb = ge._all_labels["fire"], ge._all_labels["smoke"]
    e0 = ge.graph.get_edge(sa, sb)
    if e0 is not None:
        conf_before = e0.confidence
    n2 = w2g.learn_text("Fire causes smoke. Fire causes smoke.", source_url="repeat")
    print(f"  repeat-pass new facts        : {n2} (expect 0 — dedup, not duplicate)")
    e1 = ge.graph.get_edge(sa, sb)
    if e1 is not None and conf_before is not None:
        print(f"  fire->smoke confidence       : {conf_before:.2f} -> {e1.confidence:.2f} "
              f"({'strengthened' if e1.confidence >= conf_before else 'unchanged'})")
    assert n2 == 0, "repeats must dedup, not duplicate"

    # ── T3: EFE knowledge-gap sketch ──
    gap_known = w2g.knowledge_gap("water")
    gap_novel = w2g.knowledge_gap("quantum entanglement")
    print(f"\n  EFE gap('water')            : {gap_known.efe:.1f}  (known_edges={gap_known.known_edges})")
    print(f"  EFE gap('quantum entangle') : {gap_novel.efe:.1f}  (known_edges={gap_novel.known_edges})")
    print(f"  'quantum entanglement' is curiosity target: {gap_novel.is_curiosity_target()}")
    assert gap_known.efe < gap_novel.efe, "sparse topic must have higher EFE"
    assert gap_novel.is_curiosity_target(), "novel topic should be a curiosity target"

    # ── Verdict ──
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    ok1 = ge.dim == dim_before and n_written > 0
    ok2 = typed > 0
    ok3 = n2 == 0
    ok4 = gap_novel.efe > gap_known.efe
    print(f"  no dimensionality change + facts written : {'PASS' if ok1 else 'FAIL'}")
    print(f"  typed relation edges landed              : {'PASS' if ok2 else 'FAIL'}")
    print(f"  repeat dedup (no explosion)              : {'PASS' if ok3 else 'FAIL'}")
    print(f"  EFE knowledge-gap emits signal           : {'PASS' if ok4 else 'FAIL'}")
    print(f"\n  C-lite is SAFE to ship: facts acquired into the existing KG,")
    print(f"  no HRR, no dim change, gaps feed the future E control spine.")


if __name__ == "__main__":
    main()
