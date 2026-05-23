"""Profile RAVANA learn() to find actual bottlenecks."""
import cProfile
import pstats
import io
import time
import numpy as np
import sys

sys.path.insert(0, '.')

from ravana_ml.nn.rlm import RLM

def make_rlm(vocab_size=500, embed_dim=64, concept_dim=64, n_hidden=128):
    return RLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        concept_dim=concept_dim,
        n_concepts=1000,
        n_hidden=n_hidden,
    )

def profile_individual_operations():
    """Time individual graph operations."""
    rlm = make_rlm()
    rng = np.random.RandomState(42)
    
    # Build up graph
    for _ in range(500):
        tid = rng.randint(0, 500)
        rlm.learn(np.array([tid]), np.array([rng.randint(0, 500)]))
    
    N = len(rlm.graph.nodes)
    E = len(rlm.graph.edges)
    print(f"=== Individual Operation Timing (N={N}, E={E}) ===")
    
    vec = rng.randn(rlm.concept_dim).astype(np.float32)
    
    # 1. find_similar(k=1)
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        rlm.graph.find_similar(vec, k=1)
        times.append(time.perf_counter() - t0)
    print(f"find_similar(k=1):  {np.median(times)*1000:.3f}ms")
    
    # 2. find_similar(k=5)
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        rlm.graph.find_similar(vec, k=5)
        times.append(time.perf_counter() - t0)
    print(f"find_similar(k=5):  {np.median(times)*1000:.3f}ms")
    
    # 3. spread_activation
    rlm.graph.reset_activation()
    rlm.graph.bind_input(vec, k=7)
    times = []
    for _ in range(50):
        saved_active = set(rlm.graph._active_nodes)
        t0 = time.perf_counter()
        rlm.graph.spread_activation(steps=2, k_active=7, decay=0.5)
        times.append(time.perf_counter() - t0)
    print(f"spread_activation(2): {np.median(times)*1000:.3f}ms")
    
    # 4. prune_edges
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        rlm.graph.prune_edges(threshold=0.05)
        times.append(time.perf_counter() - t0)
    print(f"prune_edges:       {np.median(times)*1000:.3f}ms")
    
    # 5. form_edges
    rlm.graph.reset_activation()
    rlm.graph.bind_input(vec, k=7)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        rlm.graph.form_edges()
        times.append(time.perf_counter() - t0)
    print(f"form_edges:        {np.median(times)*1000:.3f}ms")
    
    # 6. homeostatic_downscale
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        rlm.graph.homeostatic_downscale()
        times.append(time.perf_counter() - t0)
    print(f"homeostatic_downscale: {np.median(times)*1000:.3f}ms")
    
    # 7. _rebuild_vector_matrix
    rlm.graph._vectors_dirty = True
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        rlm.graph._rebuild_vector_matrix()
        times.append(time.perf_counter() - t0)
        rlm.graph._vectors_dirty = True
    print(f"_rebuild_vector_matrix: {np.median(times)*1000:.3f}ms")
    
    # 8. forward() node_sims loop (the explicit loop in forward())
    z = rng.randn(rlm.concept_dim).astype(np.float32)
    z_norm = z / (np.linalg.norm(z) + 1e-15)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        node_sims = []
        for nid, node in rlm.graph.nodes.items():
            sim = np.dot(node.vector, z_norm)
            node_sims.append((nid, sim))
        times.append(time.perf_counter() - t0)
    print(f"forward_node_sims_loop: {np.median(times)*1000:.3f}ms")
    
    # 9. vs find_similar equivalent
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        rlm.graph.find_similar(z, k=7)
        times.append(time.perf_counter() - t0)
    print(f"find_similar(k=7):  {np.median(times)*1000:.3f}ms")

def profile_learn_steps(n_steps=200):
    """Run N learn steps and profile with cProfile."""
    rlm = make_rlm()
    rng = np.random.RandomState(42)
    
    # Warm up
    for _ in range(100):
        tid = rng.randint(0, 500)
        rlm.learn(np.array([tid]), np.array([rng.randint(0, 500)]))
    
    N = len(rlm.graph.nodes)
    E = len(rlm.graph.edges)
    print(f"\nAfter warmup: {N} nodes, {E} edges")
    
    pr = cProfile.Profile()
    pr.enable()
    
    t0 = time.perf_counter()
    for i in range(n_steps):
        tid = rng.randint(0, 500)
        next_tid = rng.randint(0, 500)
        rlm.learn(np.array([tid]), np.array([next_tid]))
    elapsed = time.perf_counter() - t0
    
    pr.disable()
    
    print(f"\n{n_steps} learn steps in {elapsed:.3f}s ({elapsed/n_steps*1000:.1f}ms/step)")
    print(f"Nodes: {len(rlm.graph.nodes)}, Edges: {len(rlm.graph.edges)}")
    
    # Top by cumulative
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(30)
    print("\n=== TOP 30 BY CUMULATIVE TIME ===")
    print(s.getvalue())
    
    # Top by tottime (self)
    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats('tottime').print_stats(30)
    print("=== TOP 30 BY TOTAL TIME (self) ===")
    print(s2.getvalue())

if __name__ == '__main__':
    profile_individual_operations()
    profile_learn_steps(200)
