"""Profile learn() sub-steps with the optimized forward()."""
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
    build_graph(model)

    token_ids = np.array([1, 5, 10], dtype=np.int64)
    target_ids = np.array([10], dtype=np.int64)

    # Warm up
    for _ in range(3):
        model.forward(token_ids)
        model.learn(token_ids, target_ids)

    # Profile: break learn() into sub-steps
    n = 200
    t_fwd = t_decompose = t_concepts = t_classify = t_hebbian = t_rest = 0

    for _ in range(n):
        t0 = time.perf_counter()
        logits = model.forward(token_ids)
        t1 = time.perf_counter()
        t_fwd += t1 - t0

        # Inline the learn() sub-steps
        full_triple_ids = token_ids.tolist() + [target_ids[0]]
        t2 = time.perf_counter()
        subject_ids, relation_ids, object_ids = model.decompose_triple(full_triple_ids)
        t3 = time.perf_counter()
        t_decompose += t3 - t2

        if not subject_ids:
            continue

        subject_tid = subject_ids[0]
        subject_embed = model.token_embed.weight.data[subject_tid]
        object_embed = model.token_embed.weight.data[target_ids[0]]
        t4 = time.perf_counter()

        subject_cid = model._get_or_create_concept(subject_tid, subject_embed)
        object_cid = model._get_or_create_concept(target_ids[0], object_embed)
        t5 = time.perf_counter()
        t_concepts += t5 - t4

        rel_type_idx, rel_type_embed = model._classify_relation_learned(relation_ids)
        t6 = time.perf_counter()
        t_classify += t6 - t5

        # Hebbian + edge operations
        edge = model.graph.edges.get((subject_cid, object_cid))
        if edge is None:
            edge = model.graph.add_edge(source=subject_cid, target=object_cid,
                                        weight=0.1, relation_type=RELATION_TYPES[rel_type_idx])
        edge.weight = min(1.0, edge.weight + 0.05)
        edge.confidence = min(1.0, edge.confidence + 0.02)
        t7 = time.perf_counter()
        t_hebbian += t7 - t6

    total = t_fwd + t_decompose + t_concepts + t_classify + t_hebbian
    print(f"Profile of {n} learn() sub-steps:")
    print(f"  forward():        {t_fwd/n*1000:6.3f}ms  ({t_fwd/total*100:5.1f}%)")
    print(f"  decompose:        {t_decompose/n*1000:6.3f}ms  ({t_decompose/total*100:5.1f}%)")
    print(f"  get/create concept: {t_concepts/n*1000:6.3f}ms  ({t_concepts/total*100:5.1f}%)")
    print(f"  classify relation:  {t_classify/n*1000:6.3f}ms  ({t_classify/total*100:5.1f}%)")
    print(f"  hebbian/edge:     {t_hebbian/n*1000:6.3f}ms  ({t_hebbian/total*100:5.1f}%)")
    print(f"  ─────────────────────────────")
    print(f"  Total per learn():  {total/n*1000:6.3f}ms")

if __name__ == "__main__":
    main()
