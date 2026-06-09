"""
Profile learn() in rlm_v2.py to find the actual performance bottleneck.

Creates a model with ~185 nodes and ~337 edges, then times a single learn()
call broken down by sub-step.
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ravana_ml.nn.rlm_v2 import RLMv2, RELATION_TYPES


def build_graph(model, n_nodes=185, n_edges=337):
    """Populate the model's graph with n_nodes concept nodes and n_edges edges."""
    rng = np.random.RandomState(42)
    dim = model.concept_dim

    # Create nodes
    node_ids = []
    for i in range(n_nodes):
        vec = rng.randn(dim).astype(np.float32) * 0.1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        node = model.graph.add_node(vector=vec, label=f"node_{i}")
        node_ids.append(node.id)

    # Create edges (random source→target pairs)
    rel_types = ["causal", "semantic", "temporal", "possessive", "analogical", "contextual"]
    for i in range(n_edges):
        src = node_ids[rng.randint(0, n_nodes)]
        tgt = node_ids[rng.randint(0, n_nodes)]
        if src == tgt:
            tgt = node_ids[(rng.randint(0, n_nodes))]
        rt = rel_types[rng.randint(0, len(rel_types))]
        edge = model.graph.add_edge(source=src, target=tgt, weight=0.5, relation_type=rt)
        # Give edges a relation_vector (normally set during learning)
        rv = rng.randn(dim).astype(np.float32) * 0.1
        rv_norm = np.linalg.norm(rv)
        if rv_norm > 0:
            rv /= rv_norm
        edge.relation_vector = rv

    # Set all node activations to something non-zero (simulates post-forward state)
    for nid, node in model.graph.nodes.items():
        node.activation = rng.uniform(0.0, 0.5)

    return node_ids


def profile_learn():
    """Profile a single learn() call with detailed sub-step timing."""
    vocab_size = 500
    embed_dim = 64
    concept_dim = 64

    print("Creating model...")
    model = RLMv2(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        concept_dim=concept_dim,
        n_concepts=200,
        sleep_interval=999999,  # disable auto-sleep during profiling
    )

    print("Building graph (~185 nodes, ~337 edges)...")
    node_ids = build_graph(model, n_nodes=185, n_edges=337)
    actual_nodes = len(model.graph.nodes)
    actual_edges = len(model.graph.edges)
    print(f"  Graph: {actual_nodes} nodes, {actual_edges} edges")

    # Create a sample input: "heat causes expansion" → token_ids=[1, 5, 10]
    token_ids = np.array([1, 5, 10], dtype=np.int64)
    target_ids = np.array([10], dtype=np.int64)

    # ── Warm-up call (first call may be slow due to caches, JIT, etc.) ──
    print("\nWarm-up call...")
    model.learn(token_ids, target_ids)
    print("  Done.")

    # ── Profiled learn() call ──
    print("\n" + "=" * 60)
    print("PROFILING SINGLE learn() CALL")
    print("=" * 60)

    timings = {}

    # We monkey-patch key methods to time them individually
    # This gives us exact per-method timings without modifying source code

    orig_forward = model.forward
    orig_decompose = model.decompose_triple
    orig_classify = model._classify_relation_learned
    orig_get_or_create = model._get_or_create_concept
    orig_update_classifier = model._update_relation_classifier
    orig_project_to_concept = model._project_to_concept

    def timed_forward(token_ids_inner):
        t0 = time.perf_counter()
        result = orig_forward(token_ids_inner)
        t1 = time.perf_counter()
        timings['forward'] = timings.get('forward', 0) + (t1 - t0)
        return result

    def timed_decompose(token_ids_inner):
        t0 = time.perf_counter()
        result = orig_decompose(token_ids_inner)
        t1 = time.perf_counter()
        timings['decompose_triple'] = timings.get('decompose_triple', 0) + (t1 - t0)
        return result

    def timed_classify(rel_ids):
        t0 = time.perf_counter()
        result = orig_classify(rel_ids)
        t1 = time.perf_counter()
        timings['_classify_relation_learned'] = timings.get('_classify_relation_learned', 0) + (t1 - t0)
        return result

    def timed_get_or_create(token_id, embed_vec):
        t0 = time.perf_counter()
        result = orig_get_or_create(token_id, embed_vec)
        t1 = time.perf_counter()
        timings['_get_or_create_concept'] = timings.get('_get_or_create_concept', 0) + (t1 - t0)
        return result

    def timed_update_classifier(rel_ids, true_idx):
        t0 = time.perf_counter()
        result = orig_update_classifier(rel_ids, true_idx)
        t1 = time.perf_counter()
        timings['_update_relation_classifier'] = timings.get('_update_relation_classifier', 0) + (t1 - t0)
        return result

    def timed_project(embed_vec):
        t0 = time.perf_counter()
        result = orig_project_to_concept(embed_vec)
        t1 = time.perf_counter()
        timings['_project_to_concept'] = timings.get('_project_to_concept', 0) + (t1 - t0)
        return result

    model.forward = timed_forward
    model.decompose_triple = timed_decompose
    model._classify_relation_learned = timed_classify
    model._get_or_create_concept = timed_get_or_create
    model._update_relation_classifier = timed_update_classifier
    model._project_to_concept = timed_project

    # Also time the forward's internal spreading activation and scoring
    # by breaking it down: we time forward separately, then time just the
    # "post-forward" portion of learn()

    # Reset counters
    model._train_correct = 0
    model._train_total = 0

    t_total_start = time.perf_counter()
    result = model.learn(token_ids, target_ids)
    t_total_end = time.perf_counter()

    total_time = t_total_end - t_total_start

    print(f"\nTotal learn() time: {total_time*1000:.3f} ms")
    print(f"Result: loss={result['loss']:.4f}, accuracy={result['accuracy']:.1%}")
    print(f"\nBreakdown by sub-step:")
    print("-" * 50)

    # Sort by time descending
    sorted_timings = sorted(timings.items(), key=lambda x: -x[1])
    accounted = 0
    for name, t in sorted_timings:
        pct = t / total_time * 100
        print(f"  {name:40s} {t*1000:8.3f} ms  ({pct:5.1f}%)")
        accounted += t

    unaccounted = total_time - accounted
    pct = unaccounted / total_time * 100
    print(f"  {'(rest of learn: Hebbian, predictive coding, etc.)':40s} {unaccounted*1000:8.3f} ms  ({pct:5.1f}%)")
    print("-" * 50)
    print(f"  {'TOTAL':40s} {total_time*1000:8.3f} ms  (100.0%)")

    # ── Now profile the forward() internals more granularly ──
    print("\n" + "=" * 60)
    print("FORWARD PASS INTERNAL BREAKDOWN")
    print("=" * 60)

    # Remove monkey-patches temporarily for internal profiling
    model.forward = orig_forward

    # Time the key sub-operations of forward() individually
    # We can't easily instrument inside forward() without modifying source,
    # so we'll do it by profiling the expensive graph operations directly.

    # 1. Phase 0: similarity-based priming (iterates ALL nodes)
    print("\nSimulating forward() sub-steps with {0} nodes, {1} edges:".format(actual_nodes, actual_edges))

    # Time: iterate all nodes for cosine similarity (Phase 0 in forward)
    subject_vec = model.graph.nodes[list(model.graph.nodes.keys())[0]].vector
    sv_norm = np.linalg.norm(subject_vec)
    t0 = time.perf_counter()
    for cid, node in model.graph.nodes.items():
        nv_norm = np.linalg.norm(node.vector)
        if nv_norm > 0:
            sim = float(np.dot(subject_vec, node.vector) / (sv_norm * nv_norm))
    t1 = time.perf_counter()
    print(f"  Phase 0: similarity priming (all {actual_nodes} nodes): {(t1-t0)*1000:.3f} ms")

    # Time: spread_activation (3 steps, sparse or dense)
    for node in model.graph.nodes.values():
        node.activation = 0.0
    model.graph.activate(list(model.graph.nodes.keys())[0], amount=1.0)
    t0 = time.perf_counter()
    model.graph.spread_activation(steps=3, k_active=10, decay=0.3)
    t1 = time.perf_counter()
    print(f"  spread_activation (3 steps, k=10):       {(t1-t0)*1000:.3f} ms")

    # Time: Phase 2 relation-aware spreading (2 iterations over all nodes+edges)
    t0 = time.perf_counter()
    for _ in range(2):
        to_activate = []
        for nid, node in model.graph.nodes.items():
            if node.activation < 0.005:
                continue
            for tgt_id, edge in model.graph.get_outgoing(nid):
                if edge.relation_type == "causal":
                    tgt_node = model.graph.get_node(tgt_id)
                    if tgt_node is not None:
                        to_activate.append((tgt_id, node.activation * 0.5))
        for nid, amount in to_activate:
            model.graph.activate(nid, amount=amount)
    t1 = time.perf_counter()
    print(f"  Phase 2: relation-aware spreading (2x):  {(t1-t0)*1000:.3f} ms")

    # Time: scoring active nodes against edges (Phase scoring)
    active_nodes = [(nid, node) for nid, node in model.graph.nodes.items() if node.activation > 0.01]
    t0 = time.perf_counter()
    matching_targets = {}
    for nid, node in active_nodes:
        outgoing = model.graph.get_outgoing(nid)
        for tgt_id, edge in outgoing:
            base_score = node.activation * edge.weight * edge.confidence
            if tgt_id in matching_targets:
                matching_targets[tgt_id] = max(matching_targets[tgt_id], base_score)
            else:
                matching_targets[tgt_id] = base_score
    t1 = time.perf_counter()
    print(f"  Scoring active→edges ({len(active_nodes)} active):  {(t1-t0)*1000:.3f} ms")

    # Time: batch concept-to-token scoring (matmul)
    if matching_targets:
        batch_targets = []
        for tgt_cid, score in matching_targets.items():
            tgt_node = model.graph.get_node(tgt_cid)
            if tgt_node is not None:
                batch_targets.append((tgt_cid, score, tgt_node.vector))

        if batch_targets:
            t0 = time.perf_counter()
            tgt_vecs = np.stack([tv[2] for tv in batch_targets])
            tgt_embeds = model.concept_to_embed(tgt_vecs).data
            tgt_norms = np.linalg.norm(tgt_embeds, axis=1)
            token_embeds = model.token_embed.weight.data
            token_norms = np.linalg.norm(token_embeds, axis=1)

            normed_tgt = tgt_embeds.copy()
            valid_tgt = tgt_norms > 0
            normed_tgt[valid_tgt] /= tgt_norms[valid_tgt, np.newaxis]
            normed_tok = token_embeds.copy()
            valid_tok = token_norms > 0
            normed_tok[valid_tok] /= token_norms[valid_tok, np.newaxis]
            sim_matrix = normed_tgt @ normed_tok.T
            t1 = time.perf_counter()
            print(f"  Batch concept→token scoring ({len(batch_targets)} targets × {vocab_size} vocab): {(t1-t0)*1000:.3f} ms")

    # Time: _get_or_create_concept (includes find_similar)
    t0 = time.perf_counter()
    embed_vec = model.token_embed.weight.data[1]
    model._get_or_create_concept(1, embed_vec)
    t1 = time.perf_counter()
    print(f"  _get_or_create_concept (cold):             {(t1-t0)*1000:.3f} ms")

    # Time: find_similar (called from _nearest_concept)
    concept_vec = model.subject_proj(embed_vec.reshape(1, -1)).data.flatten()
    t0 = time.perf_counter()
    model.graph.find_similar(concept_vec, k=1)
    t1 = time.perf_counter()
    print(f"  graph.find_similar (k=1, {actual_nodes} nodes): {(t1-t0)*1000:.3f} ms")

    # Time: predictive coding loop (iterates ALL edges)
    t0 = time.perf_counter()
    relevant_cids = {list(model.graph.nodes.keys())[0], list(model.graph.nodes.keys())[1]}
    for (src_id, tgt_id), edge in list(model.graph.edges.items()):
        if src_id not in relevant_cids and tgt_id not in relevant_cids:
            continue
        src_node = model.graph.nodes.get(src_id)
        tgt_node = model.graph.nodes.get(tgt_id)
        if src_node is None or tgt_node is None:
            continue
    t1 = time.perf_counter()
    print(f"  Predictive coding loop (iter {actual_edges} edges, filter): {(t1-t0)*1000:.3f} ms")

    # Time: vector arithmetic / analogy (iterates all edges for relation vectors)
    t0 = time.perf_counter()
    rvs = []
    for (src, tgt), edge in model.graph.edges.items():
        if edge.relation_type == "causal" and hasattr(edge, 'relation_vector') and edge.relation_vector is not None:
            rvs.append(edge.relation_vector)
    if rvs:
        avg_rv = np.mean(rvs, axis=0)
    t1 = time.perf_counter()
    print(f"  Vector arithmetic (scan {actual_edges} edges for rvs): {(t1-t0)*1000:.3f} ms")

    # ── Scaling test: what happens at 1000+ edges? ──
    print("\n" + "=" * 60)
    print("SCALING TEST: learn() time vs graph size")
    print("=" * 60)

    for n_nodes, n_edges in [(50, 100), (185, 337), (500, 1000), (1000, 2000)]:
        m = RLMv2(vocab_size=500, embed_dim=64, concept_dim=64, n_concepts=200, sleep_interval=999999)
        build_graph(m, n_nodes=n_nodes, n_edges=n_edges)
        # Warm-up
        m.learn(np.array([1, 5, 10], dtype=np.int64), np.array([10], dtype=np.int64))
        # Timed
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            m.learn(np.array([1, 5, 10], dtype=np.int64), np.array([10], dtype=np.int64))
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = np.mean(times) * 1000
        mn = np.min(times) * 1000
        print(f"  {len(m.graph.nodes):5d} nodes, {len(m.graph.edges):5d} edges → learn(): avg={avg:.2f} ms, min={mn:.2f} ms")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)
    print("""
Key findings from the profiling above:

1. forward() dominates learn() — it contains:
   - Phase 0: O(N) cosine similarity scan over ALL nodes
   - spread_activation: 3 propagation steps  
   - Phase 2: 2x iteration over all nodes + edges (relation-aware)
   - Scoring: O(active × edges) edge traversal
   - Batch scoring: matmul of (n_targets × vocab_size)

2. Predictive coding loop: iterates ALL edges (O(E)) even though
   only ~4 relevant nodes are involved — filter is cheap but iteration
   is O(E).

3. Vector arithmetic: scans ALL edges to collect relation vectors.

4. At 300x hard-example boost, learn() is called thousands of times
   per epoch, so even 1ms per call adds up to seconds.

Main bottleneck candidates:
- O(N) node scans in forward() (Phase 0, Phase 2)
- O(E) predictive coding edge scan
- O(E) relation vector scan
- Repeated _project_to_concept calls (matmul each time)
""")


if __name__ == "__main__":
    profile_learn()
