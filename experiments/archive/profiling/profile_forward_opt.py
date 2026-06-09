"""Profile the optimized forward() pass.

Simulates hard-boost training: same context, multiple forward() calls.
Measures per-call time to see if vectorization + caching helps.
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ravana_ml.nn.rlm_v2 import RLMv2

def build_graph(model, n_nodes=185, n_edges=337):
    rng = np.random.RandomState(42)
    dim = model.concept_dim
    node_ids = []
    for i in range(n_nodes):
        vec = rng.randn(dim).astype(np.float32) * 0.1
        norm = np.linalg.norm(vec)
        if norm > 0: vec /= norm
        node = model.graph.add_node(vector=vec, label=f"node_{i}")
        node_ids.append(node.id)

    rel_types = ["causal", "semantic", "temporal", "possessive", "analogical", "contextual"]
    for i in range(n_edges):
        src = node_ids[rng.randint(0, n_nodes)]
        tgt = node_ids[rng.randint(0, n_nodes)]
        if src == tgt: tgt = node_ids[(src + 1) % n_nodes]
        rt = rel_types[rng.randint(0, len(rel_types))]
        edge = model.graph.add_edge(source=src, target=tgt, weight=0.5, relation_type=rt)
        rv = rng.randn(dim).astype(np.float32) * 0.1
        rv_norm = np.linalg.norm(rv)
        if rv_norm > 0: rv /= rv_norm
        edge.relation_vector = rv
    return node_ids

def main():
    model = RLMv2(vocab_size=500, embed_dim=64, concept_dim=64, n_concepts=200, sleep_interval=999999)
    build_graph(model, n_nodes=185, n_edges=337)
    print(f"Graph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")

    token_ids = np.array([1, 5, 10], dtype=np.int64)

    # Warm up
    for _ in range(3):
        model.forward(token_ids)

    # Benchmark: 100 forward() calls (simulates hard-boost loop)
    n_calls = 100
    start = time.perf_counter()
    for _ in range(n_calls):
        model.forward(token_ids)
    elapsed = time.perf_counter() - start
    print(f"\n{n_calls} forward() calls: {elapsed:.3f}s total, {elapsed/n_calls*1000:.3f}ms per call")

    # Now simulate hard-boost with learn() between calls (cache invalidation)
    model2 = RLMv2(vocab_size=500, embed_dim=64, concept_dim=64, n_concepts=200, sleep_interval=999999)
    build_graph(model2, n_nodes=185, n_edges=337)

    # Train a bit so graph stabilizes
    for i in range(20):
        model2.learn(token_ids, np.array([10], dtype=np.int64))

    # Now measure forward+learn pairs (cache invalidates each time)
    n_calls = 100
    start = time.perf_counter()
    for _ in range(n_calls):
        model2.forward(token_ids)
        model2.learn(token_ids, np.array([10], dtype=np.int64))
    elapsed = time.perf_counter() - start
    print(f"\n{n_calls} forward+learn pairs: {elapsed:.3f}s total, {elapsed/n_calls*1000:.3f}ms per pair")
    print(f"  (forward+learn per pair: {elapsed/n_calls*1000:.3f}ms)")

    # Measure just forward() with stable graph (cache should hit)
    # Reset the version so next forward rebuilds cache
    model2._node_matrix_version = -1
    start = time.perf_counter()
    for _ in range(n_calls):
        model2.forward(token_ids)
    elapsed = time.perf_counter() - start
    print(f"\n{n_calls} forward() calls (stable graph, cache warm): {elapsed:.3f}s total, {elapsed/n_calls*1000:.3f}ms per call")

if __name__ == "__main__":
    main()
