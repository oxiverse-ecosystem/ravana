"""
Graph Scalability Benchmark for RAVANA

Measures performance of graph operations at different scales:
  - find_similar() latency: brute-force vs FAISS index
  - spread_activation() per-step time
  - sleep cycle time
  - hebbian_update() time

Graph sizes: 1K, 5K, 10K, 50K nodes

Usage:
    python experiments/experiment_scaling_benchmark.py
    python experiments/experiment_scaling_benchmark.py --quick   # fewer sizes
"""

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import time
import json
from ravana_ml.graph import ConceptGraph


def populate_graph(graph: ConceptGraph, n_nodes: int, n_edges_per_node: int = 5):
    """Populate a graph with random nodes and edges."""
    rng = np.random.RandomState(42)
    nodes = []
    for i in range(n_nodes):
        vec = rng.randn(graph.dim).astype(np.float32)
        vec /= (np.linalg.norm(vec) + 1e-15)
        node = graph.add_node(vec, label=f"n_{i}")
        nodes.append(node)

    # Add random edges
    for i in range(n_nodes):
        targets = rng.choice(n_nodes, size=min(n_edges_per_node, n_nodes - 1), replace=False)
        for t in targets:
            if t != i:
                rel_type = rng.choice(["semantic", "causal", "temporal", "analogical"])
                graph.add_edge(nodes[i].id, nodes[t].id, weight=rng.uniform(0.1, 0.9),
                              relation_type=rel_type)

    # Mark vectors dirty to trigger rebuild
    graph._vectors_dirty = True
    return nodes


def benchmark_find_similar(graph: ConceptGraph, n_queries: int = 100):
    """Benchmark find_similar() with a random query vector."""
    rng = np.random.RandomState(99)
    queries = [rng.randn(graph.dim).astype(np.float32) for _ in range(n_queries)]

    # Warm up (rebuild vector matrix)
    graph.find_similar(queries[0], k=5)

    t0 = time.perf_counter()
    for q in queries:
        graph.find_similar(q, k=5)
    dt = time.perf_counter() - t0
    return dt / n_queries * 1000  # ms per query


def benchmark_spread_activation(graph: ConceptGraph, nodes, n_steps: int = 100):
    """Benchmark spread_activation() on random active nodes."""
    rng = np.random.RandomState(99)
    # Activate some nodes
    active_ids = [nodes[i].id for i in rng.choice(len(nodes), size=min(5, len(nodes)), replace=False)]
    for nid in active_ids:
        graph.nodes[nid].activation = 0.8
        graph._active_nodes.add(nid)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        graph.spread_activation(steps=1)
    dt = time.perf_counter() - t0
    return dt / n_steps * 1000  # ms per step


def benchmark_hebbian(graph: ConceptGraph, nodes, n_updates: int = 100):
    """Benchmark hebbian_update() on random edges."""
    rng = np.random.RandomState(99)
    edge_keys = list(graph.edges.keys())
    if not edge_keys:
        return 0.0

    t0 = time.perf_counter()
    for _ in range(n_updates):
        key = edge_keys[rng.randint(len(edge_keys))]
        graph.hebbian_update(key[0], key[1], coactivation=0.8, lr=0.01)
    dt = time.perf_counter() - t0
    return dt / n_updates * 1000


def benchmark_consolidate(graph: ConceptGraph, nodes):
    """Benchmark consolidate_vectors()."""
    # Activate some nodes to populate incremental set
    for n in nodes[:min(10, len(nodes))]:
        graph._activated_since_sleep.add(n.id)

    t0 = time.perf_counter()
    graph.consolidate_vectors()
    dt = time.perf_counter() - t0
    return dt * 1000  # ms


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fewer sizes")
    args = parser.parse_args()

    sizes = [1000, 5000, 10000] if args.quick else [1000, 5000, 10000, 50000]
    dim = 32  # same as paper baseline

    print("=" * 70)
    print("  RAVANA Graph Scalability Benchmark")
    print("=" * 70)
    print(f"  Embedding dim: {dim}")
    print(f"  Graph sizes: {sizes}")
    print()

    results = []

    for n_nodes in sizes:
        print(f"  --- {n_nodes} nodes ---")
        graph = ConceptGraph(dim=dim)

        t0 = time.perf_counter()
        nodes = populate_graph(graph, n_nodes, n_edges_per_node=5)
        pop_time = time.perf_counter() - t0
        n_edges = len(graph.edges)
        print(f"    Populate: {pop_time:.2f}s ({n_edges} edges)")

        # find_similar
        fs_time = benchmark_find_similar(graph, n_queries=50)
        print(f"    find_similar: {fs_time:.2f} ms/query")

        # Check if FAISS is being used
        faiss_active = graph._use_faiss and graph._faiss_index is not None
        print(f"    FAISS active: {faiss_active}")

        # spread_activation
        sa_time = benchmark_spread_activation(graph, nodes, n_steps=50)
        print(f"    spread_activation: {sa_time:.2f} ms/step")

        # hebbian_update
        hb_time = benchmark_hebbian(graph, nodes, n_updates=50)
        print(f"    hebbian_update: {hb_time:.3f} ms/update")

        # consolidate_vectors
        cv_time = benchmark_consolidate(graph, nodes)
        print(f"    consolidate_vectors: {cv_time:.2f} ms")

        results.append({
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "find_similar_ms": round(fs_time, 2),
            "spread_activation_ms": round(sa_time, 2),
            "hebbian_update_ms": round(hb_time, 3),
            "consolidate_ms": round(cv_time, 2),
            "faiss_active": faiss_active,
        })
        print()

    # Scaling analysis
    if len(results) >= 2:
        print("  Scaling analysis (find_similar):")
        for i in range(1, len(results)):
            r0, r1 = results[i-1], results[i]
            size_ratio = r1["n_nodes"] / r0["n_nodes"]
            time_ratio = r1["find_similar_ms"] / max(r0["find_similar_ms"], 0.001)
            print(f"    {r0['n_nodes']} -> {r1['n_nodes']}: "
                  f"size x{size_ratio:.0f}, time x{time_ratio:.1f}")

    # Save
    out_path = os.path.join(_PROJECT_ROOT, "revisions", "scaling_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
