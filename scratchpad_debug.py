import numpy as np
from ravana_ml.graph import ConceptGraph
g = ConceptGraph(dim=16, max_nodes=100, max_edges=1000, rng=np.random.RandomState(42))
a = g.add_node(np.random.randn(16).astype(np.float32), label="a")
b = g.add_node(np.random.randn(16).astype(np.float32), label="b")
c = g.add_node(np.random.randn(16).astype(np.float32), label="c")
g.add_edge(a.id, b.id, weight=0.8, relation_type="semantic")
g.add_context_edge(b.id, c.id, context_labels=["special"], weight=0.8, relation_type="semantic")
g.clear_context()
results = g.infer_chain(a.id, max_hops=2)
print("results", results)
result_ids = [r[0] for r in results]
print("result_ids", result_ids)
print("c.id", c.id)
print("c in?", c.id in result_ids)
