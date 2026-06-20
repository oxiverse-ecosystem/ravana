"""
Minimal graph scaling benchmark — directly tests ConceptGraph operations.
Measures latency at 1K->10K nodes.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ravana_ml.graph import ConceptGraph

rng = np.random.RandomState(42)

def benchmark_find_similar(graph, n_queries=200):
    vec = rng.randn(graph.dim).astype(np.float32)
    times = []
    for _ in range(n_queries):
        t0 = time.perf_counter()
        graph.find_similar(vec, k=5)
        times.append(time.perf_counter() - t0)
    p50 = np.median(times) * 1000
    p95 = np.percentile(times, 95) * 1000
    return p50, p95

print("=" * 50)
print("RAVANA - Graph Scaling Benchmark")
print("=" * 50)

sizes = [1000, 5000, 10000]
results = []

for size in sizes:
    graph = ConceptGraph(dim=32, max_nodes=size + 1000)
    n_edges = 0
    for i in range(size):
        v = rng.randn(32).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        graph.add_node(vector=v, label=f"c{i}")
    nids = list(graph.nodes.keys())
    edge_target = min(size * 5, 50000)
    for i in range(edge_target):
        s, t = int(nids[i % len(nids)]), int(nids[(i + 1) % len(nids)])
        if s != t and (s, t) not in graph.edges:
            graph.add_edge(s, t, weight=0.8, relation_type="semantic")
            n_edges += 1
    p50, p95 = benchmark_find_similar(graph, n_queries=100)
    results.append({"nodes": len(graph.nodes), "edges": n_edges, "p50_ms": round(p50, 3), "p95_ms": round(p95, 3)})
    print(f"  {len(graph.nodes):>6} nodes, {n_edges:>6} edges  |  find_similar p50={p50:.3f}ms  p95={p95:.3f}ms")

print("\n" + "=" * 50)
print("RESULTS (JSON)")
print(json.dumps(results, indent=2))

# --- False-positive probe ---
print("\n" + "=" * 50)
print("FALSE POSITIVE PROBE - 3 domains")
print("=" * 50)
graph2 = ConceptGraph(dim=32, max_nodes=500)
domains = {
    "animals": ["dog","cat","bird","fish","rabbit","hamster"],
    "colors":  ["red","blue","green","yellow","purple","orange"],
    "fruits":  ["apple","banana","grape","mango","kiwi","pear"],
}
nids_by_domain = {}
for dname, items in domains.items():
    nids_by_domain[dname] = []
    for word in items:
        v = rng.randn(32).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        node = graph2.add_node(vector=v, label=word)
        nids_by_domain[dname].append(node.id)
for dname, nids in nids_by_domain.items():
    for i in range(len(nids)):
        j = (i + 1) % len(nids)
        graph2.add_edge(int(nids[i]), int(nids[j]), weight=0.9, relation_type="semantic")

total_intrusions = 0
n_probes = 0
for dname, nids in nids_by_domain.items():
    for i, nid in enumerate(nids):
        node = graph2.get_node(int(nid))
        if node is None:
            continue
        matches = graph2.find_similar(node.vector, k=10)
        intrusions = sum(1 for mid, sim in matches if int(mid) not in nids_by_domain[dname])
        total_intrusions += intrusions
        n_probes += 1

avg = total_intrusions / max(1, n_probes)
print(f"  Avg cross-domain intrusions in top-10: {avg:.2f}")
print(f"  (lower is better; < 5 is good, 0 is perfect)")
print("\nDone - all data collected for Reddit reply")
