"""Profile learn_fast() vs learn() — the key speed comparison."""
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
    token_ids = np.array([1, 5, 10], dtype=np.int64)
    target_ids = np.array([10], dtype=np.int64)

    # ── Profile learn() ──
    model1 = RLMv2(vocab_size=500, embed_dim=64, concept_dim=64, n_concepts=200, sleep_interval=999999)
    build_graph(model1)
    # Warm up
    for _ in range(3):
        model1.learn(token_ids, target_ids)

    n = 200
    start = time.perf_counter()
    for _ in range(n):
        model1.learn(token_ids, target_ids)
    t_learn = (time.perf_counter() - start) / n

    # ── Profile learn_fast() ──
    model2 = RLMv2(vocab_size=500, embed_dim=64, concept_dim=64, n_concepts=200, sleep_interval=999999)
    build_graph(model2)
    for _ in range(3):
        model2.learn_fast(token_ids, target_ids)

    start = time.perf_counter()
    for _ in range(n):
        model2.learn_fast(token_ids, target_ids)
    t_fast = (time.perf_counter() - start) / n

    print(f"learn():      {t_learn*1000:.3f}ms per call")
    print(f"learn_fast(): {t_fast*1000:.3f}ms per call")
    print(f"Speedup:      {t_learn/t_fast:.1f}x")
    print()

    # ── Benchmark projections ──
    hard = 39
    epochs = 1500
    timeout = 600

    for boost in [300, 50, 30, 10]:
        for label, per_call in [("learn", t_learn), ("learn_fast", t_fast)]:
            calls = boost * hard * epochs
            t = calls * per_call
            ok = "✓" if t <= timeout else "✗"
            print(f"  {ok} boost={boost}, {label}: {calls:>12,} calls → {t:>8.0f}s ({t/60:.1f}min)")
    print()
    # Find max boost that fits in timeout with learn_fast
    for boost in [300, 200, 100, 50, 30, 20, 10, 5]:
        calls = boost * hard * epochs
        t = calls * t_fast
        if t <= timeout:
            print(f"  Max boost with learn_fast (all {hard} hard): {boost}x → {t:.0f}s")
            break
    # With sampling
    for sampled in [20, 15, 10, 5]:
        for boost in [300, 200, 100, 50, 30]:
            calls = boost * sampled * epochs
            t = calls * t_fast
            if t <= timeout:
                print(f"  Max boost with learn_fast ({sampled} sampled hard): {boost}x → {t:.0f}s")
                break
        else:
            continue
        break

if __name__ == "__main__":
    main()
