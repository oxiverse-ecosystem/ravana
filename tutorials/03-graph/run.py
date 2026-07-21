"""Tutorial 03: Concept Graph Operations — build and inspect a concept graph.

Standalone tutorial (no dependency on 01-02), but concepts build on what
you learned about the graph role in the chat engine.

Usage:
    python tutorials/03-graph/run.py
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "ravana_ml", "src"))

from ravana_ml.graph import ConceptGraph, ConceptEdge


def main() -> None:
    # 1. Create a standalone concept graph
    graph = ConceptGraph(dim=64, max_nodes=1000)
    print("=== Concept Graph Demo ===")

    # 2. Add nodes with random 64-D vectors
    # In the real engine, these come from GloVe embeddings
    a = graph.add_node(
        vector=np.random.randn(64).astype(np.float32),
        label="trust",
    )
    b = graph.add_node(
        vector=np.random.randn(64).astype(np.float32),
        label="courage",
    )
    c = graph.add_node(
        vector=np.random.randn(64).astype(np.float32),
        label="fear",
    )

    # 3. Add typed edges with weights
    # Causal: trust -> courage (trust enables courage)
    graph.add_edge(a, b, relation_type="causal", weight=0.7, confidence=0.8)
    # Contrastive: courage <-> fear
    graph.add_edge(b, c, relation_type="contrastive", weight=0.6, confidence=0.7)
    # Semantic: trust -> fear (both are emotions)
    graph.add_edge(a, c, relation_type="semantic", weight=0.3, confidence=0.4)

    print(f"\nnodes: {len(graph.nodes)}")
    print(f"edges: {len(graph.edges)}")

    # 4. Walk outgoing edges from "trust"
    trust_node = graph.get_node(a)
    print(f"\nOutgoing from '{trust_node.label if trust_node else '?'}':")
    for target_id, edge in graph.get_outgoing(a):
        target_node = graph.get_node(target_id)
        target_label = target_node.label if target_node else "?"
        print(f"  -> {target_label:12s}  {edge.relation_type:12s}  w={edge.weight:.2f}  conf={edge.confidence:.2f}")

    # 5. Spread activation demo
    graph.activate(a, 1.0)  # seed activation at "trust"
    # In the real engine, _spread_and_collect does 3-hop spread
    print("\nActivation propagated in engine via 3-hop spread + relation bias")
    print("(See chain_walker.py -> _spread_and_collect)")

    # 6. Edge types summary
    print("\nSupported relation types:")
    edge_types_seen = set()
    for (_, _), edge in graph.edges.items():
        edge_types_seen.add(edge.relation_type)
    for t in sorted(edge_types_seen):
        print(f"  {t}")


if __name__ == "__main__":
    main()
