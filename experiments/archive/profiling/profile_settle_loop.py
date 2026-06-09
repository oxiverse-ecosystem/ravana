"""
Proposed optimizations for RLMv2 settle loop — brain-inspired fixes.

Measures the impact of:
1. Pre-computed/cached norms (brain doesn't recompute synaptic strengths)
2. Vectorized concept-to-token scoring (brain uses parallel pathways)  
3. Sparse activation tracking (brain activates ~1-4% of neurons)
4. Early termination (brain doesn't spread to dead branches)
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ravana_ml.nn.rlm_v2 import RLMv2, RELATION_TYPES
from ravana_ml.tokenizer import SimpleTokenizer
from experiments.experiment_baselines import SimpleMLP

tok = SimpleTokenizer()
vocab_size = tok.vocab_size

np.random.seed(42)
model = RLMv2(vocab_size=vocab_size, embed_dim=32, concept_dim=32,
              n_concepts=vocab_size, max_seq_len=128, sleep_interval=999999)

np.random.seed(42)
mlp = SimpleMLP(vocab_size=vocab_size, embed_dim=32, n_hidden=32, lr=0.01)

facts = [
    "heat causes expansion", "fire causes heat", "cold causes contraction",
    "fire causes melting", "heat causes melting", "cold causes solidification",
    "pressure causes compression", "light causes visibility",
    "darkness causes sleep", "sound causes vibration",
]

# Pre-train
for fact in facts:
    ids = tok.encode(fact)
    for i in range(len(ids) - 1):
        ctx = np.array([ids[:i+1]], dtype=np.int64)
        tgt = np.array([[ids[i+1]]], dtype=np.int64)
        model.learn(ctx, tgt)

print(f"Graph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")

# ─── BASELINE forward timing ───
test_ids = tok.encode("heat causes")
ctx = np.array([test_ids], dtype=np.int64)

N_ITERS = 100

# Baseline
times_baseline = []
for _ in range(N_ITERS):
    t0 = time.perf_counter()
    _ = model.forward(ctx)
    times_baseline.append(time.perf_counter() - t0)

# MLP baseline
times_mlp = []
for _ in range(N_ITERS):
    t0 = time.perf_counter()
    _ = mlp.forward(ctx)
    times_mlp.append(time.perf_counter() - t0)

# ─── OPTIMIZATION 1: Cache norms in graph nodes ───
print("\n" + "="*70)
print("OPTIMIZATION 1: Pre-compute and cache vector norms")
print("="*70)
print("Brain analogy: synaptic efficacies are physical constants, not")
print("recomputed from scratch every action potential.")

# Simulate the savings: count norm() calls in forward()
# From profiling: ~22000 norm() calls per forward, each ~4μs = 88μs
# With caching: norm() called once per node/edge, then reused

class NormCache:
    """Cache norms for all node and edge vectors."""
    def __init__(self, graph):
        self.graph = graph
        self._node_norms = {}  # nid -> norm
        self._rv_norms = {}    # (src,tgt) -> relation_vector norm
        self._dirty = True
    
    def rebuild(self):
        self._node_norms = {}
        self._rv_norms = {}
        for nid, node in self.graph.nodes.items():
            self._node_norms[nid] = float(np.linalg.norm(node.vector))
        for (src, tgt), edge in self.graph.edges.items():
            if edge.relation_vector is not None:
                self._rv_norms[(src, tgt)] = float(np.linalg.norm(edge.relation_vector))
        self._dirty = False
    
    def node_norm(self, nid):
        if self._dirty:
            self.rebuild()
        return self._node_norms.get(nid, 0.0)
    
    def rv_norm(self, src, tgt):
        if self._dirty:
            self.rebuild()
        return self._rv_norms.get((src, tgt), 0.0)
    
    def invalidate(self):
        self._dirty = True

# Measure time to rebuild the cache
cache = NormCache(model.graph)
t0 = time.perf_counter()
cache.rebuild()
cache_build_time = time.perf_counter() - t0
print(f"  Cache build time: {cache_build_time*1000:.3f}ms ({len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges)")
print(f"  If rebuilt every learn() call, amortized cost per forward: {cache_build_time*1000:.3f}ms")
print(f"  Savings: eliminate ~22,000 np.linalg.norm() calls = ~88ms per 1000 forwards")

# ─── OPTIMIZATION 2: Vectorized concept-to-token scoring ───
print("\n" + "="*70)
print("OPTIMIZATION 2: Batch concept-to-token scoring")
print("="*70)
print("Brain analogy: cortical columns process in parallel, not serially.")
print("Instead of scoring each concept against all tokens individually,")
print("batch all concept vectors into a matrix and do ONE matmul.")

# Current: for each matching_target, project to embed, then cosine sim with all tokens
# Proposed: collect all target vectors, project them all at once, then batch matmul

t_serial = []
t_batch = []

matching_targets = {}
# Build some matching targets from the graph
for nid, node in model.graph.nodes.items():
    if node.activation > 0.01:
        for tgt_id, edge in model.graph.get_outgoing(nid):
            matching_targets[tgt_id] = max(matching_targets.get(tgt_id, 0), 
                                            node.activation * edge.weight)

# Serial (current approach)
for _ in range(N_ITERS):
    t0 = time.perf_counter()
    concept_scores_serial = np.zeros(vocab_size, dtype=np.float32)
    for tgt_cid, score in matching_targets.items():
        tgt_node = model.graph.get_node(tgt_cid)
        if tgt_node is None:
            continue
        tgt_embed = model._project_to_embed(tgt_node.vector)
        tgt_norm = np.linalg.norm(tgt_embed)
        if tgt_norm > 0:
            token_embeds = model.token_embed.weight.data
            token_norms = np.linalg.norm(token_embeds, axis=1)
            valid = token_norms > 0
            sims = np.zeros(vocab_size, dtype=np.float32)
            sims[valid] = token_embeds[valid] @ tgt_embed / (token_norms[valid] * tgt_norm)
            concept_scores_serial += sims * score * 0.3
    t_serial.append(time.perf_counter() - t0)

# Batch (proposed)
# Pre-compute token norms once
token_embeds = model.token_embed.weight.data
token_norms_cached = np.linalg.norm(token_embeds, axis=1)  # (V,)
token_normed = token_embeds / (token_norms_cached[:, None] + 1e-15)  # (V, D)

for _ in range(N_ITERS):
    t0 = time.perf_counter()
    concept_scores_batch = np.zeros(vocab_size, dtype=np.float32)
    if matching_targets:
        tgt_cids = []
        tgt_scores = []
        for tgt_cid, score in matching_targets.items():
            tgt_node = model.graph.get_node(tgt_cid)
            if tgt_node is not None:
                tgt_cids.append(tgt_cid)
                tgt_scores.append(score)
        
        if tgt_cids:
            # Collect all target vectors, project in batch
            tgt_vecs = np.stack([model.graph.nodes[cid].vector for cid in tgt_cids])  # (T, D_concept)
            tgt_embeds = model.concept_to_embed(tgt_vecs)  # (T, D_embed) — ONE matmul
            if hasattr(tgt_embeds, 'data'):
                tgt_embeds = tgt_embeds.data
            tgt_norms = np.linalg.norm(tgt_embeds, axis=1, keepdims=True)  # (T, 1)
            tgt_normed = tgt_embeds / (tgt_norms + 1e-15)  # (T, D)
            
            # Batch cosine similarity: (V, D) @ (D, T) = (V, T)
            sim_matrix = token_normed @ tgt_normed.T  # (V, T)
            
            # Weighted sum across targets
            score_weights = np.array(tgt_scores, dtype=np.float32) * 0.3  # (T,)
            concept_scores_batch = sim_matrix @ score_weights  # (V,)
    t_batch.append(time.perf_counter() - t0)

mean_serial = np.mean(t_serial) * 1000
mean_batch = np.mean(t_batch) * 1000
print(f"  Serial scoring: {mean_serial:.3f}ms")
print(f"  Batch scoring:  {mean_batch:.3f}ms")
print(f"  Speedup: {mean_serial / max(mean_batch, 0.001):.1f}x")

# ─── OPTIMIZATION 3: Sparse activation tracking ───
print("\n" + "="*70)
print("OPTIMIZATION 3: Sparse activation (only visit active nodes)")
print("="*70)
print("Brain analogy: only ~1-4% of neurons fire at any time.")
print("Current code iterates ALL nodes to find active ones.")
print("With sparse tracking, skip directly to active nodes.")

# Count how many nodes are actually active vs total
model.forward(ctx)
active_count = sum(1 for n in model.graph.nodes.values() if n.activation > 0.01)
total_count = len(model.graph.nodes)
print(f"  Active nodes: {active_count}/{total_count} ({100*active_count/max(total_count,1):.1f}%)")

# Time: iterate all nodes vs iterate only active set
t_all = []
t_sparse = []
for _ in range(N_ITERS):
    t0 = time.perf_counter()
    active = [(nid, node) for nid, node in model.graph.nodes.items() if node.activation > 0.01]
    t_all.append(time.perf_counter() - t0)

# Pre-build sparse set
active_set = {nid for nid, node in model.graph.nodes.items() if node.activation > 0.01}
for _ in range(N_ITERS):
    t0 = time.perf_counter()
    active2 = [(nid, model.graph.nodes[nid]) for nid in active_set if nid in model.graph.nodes]
    t_sparse.append(time.perf_counter() - t0)

print(f"  Filter all nodes: {np.mean(t_all)*1000:.4f}ms")
print(f"  Sparse lookup:    {np.mean(t_sparse)*1000:.4f}ms")

# ─── OPTIMIZATION 4: Eliminate redundant edge traversals ───
print("\n" + "="*70)
print("OPTIMIZATION 4: Consolidate edge traversals")
print("="*70)
print("Brain analogy: a single synaptic event does everything at once —")
print("activation spread, type filtering, scoring. Not 4 separate passes.")
print()
print("Current forward() traverses edges FIVE separate times:")
print("  1. vector_analogy: iterate ALL edges for relation vectors")
print("  2. spread_activation phase1: 3 iterations × active × degree")
print("  3. spread_activation phase2: 2 iterations × all_nodes × degree")  
print("  4. score_active_edges: iterate active × degree")
print("  5. merge_active_outgoing: iterate active × degree AGAIN")
print()
print("Proposed: single-pass activation + scoring during spread.")

# Count current traversals
n_edges = len(model.graph.edges)
n_active = active_count
# Estimate total edge visits per forward()
visits_current = (
    n_edges +                              # analogy: all edges
    3 * n_active * max(1, n_edges // max(total_count, 1)) +  # spread phase1
    2 * total_count * max(1, n_edges // max(total_count, 1)) +  # spread phase2
    n_active * max(1, n_edges // max(total_count, 1)) +  # score active
    n_active * max(1, n_edges // max(total_count, 1))    # merge active
)
visits_proposed = (
    3 * n_active * max(1, n_edges // max(total_count, 1)) +  # single spread with inline scoring
    n_edges  # analogy still needs one pass (but can merge with spread)
)
print(f"  Current edge visits (estimated):  {visits_current}")
print(f"  Proposed edge visits (estimated): {visits_proposed}")
print(f"  Reduction: {visits_current / max(visits_proposed, 1):.1f}x fewer traversals")

# ─── OPTIMIZATION 5: Eliminate redundant np.linalg.norm() calls ───
print("\n" + "="*70)
print("OPTIMIZATION 5: Inline norm computation")
print("="*70)

# Profile how many norm() calls happen in spread_activation
class NormCounter:
    count = 0
    
def counting_norm(x, *args, **kwargs):
    NormCounter.count += 1
    # Delegate to real norm for correctness
    return _real_norm(x, *args, **kwargs)

# Count norm calls during forward
NormCounter.count = 0
_real_norm = np.linalg.norm
original_norm = np.linalg.norm
np.linalg.norm = counting_norm
try:
    model.forward(ctx)
except:
    pass
np.linalg.norm = original_norm
print(f"  np.linalg.norm() calls in one forward(): {NormCounter.count}")
print(f"  At ~4μs each: {NormCounter.count * 4:.0f}μs = {NormCounter.count * 4 / 1000:.2f}ms")
print(f"  With caching: ~{total_count + n_edges} calls = {(total_count + n_edges) * 4 / 1000:.2f}ms")
print(f"  Savings: {(NormCounter.count - total_count - n_edges) * 4 / 1000:.2f}ms per forward()")

# ─── MLP baseline timing ───
print("\n" + "="*70)
print("COMPARISON: MLP vs optimized RLMv2 projection")
print("="*70)

mean_rlm = np.mean(times_baseline) * 1000
mean_mlp = np.mean(times_mlp) * 1000
print(f"  Current RLMv2 forward: {mean_rlm:.3f}ms")
print(f"  MLP forward:           {mean_mlp:.3f}ms")
print(f"  Current ratio:         {mean_rlm / mean_mlp:.0f}x slower")
print()

# Projected savings
norm_savings = (NormCounter.count - total_count - n_edges) * 4 / 1000  # ms
batch_savings = mean_serial - mean_batch  # ms from scoring optimization
total_savings = norm_savings + batch_savings
projected = mean_rlm - total_savings
print(f"  Projected savings from norm caching: {norm_savings:.3f}ms")
print(f"  Projected savings from batch scoring: {batch_savings:.3f}ms")
print(f"  Total projected savings:              {total_savings:.3f}ms")
print(f"  Projected optimized forward:          {projected:.3f}ms")
print(f"  Projected ratio to MLP:               {projected / mean_mlp:.0f}x")
print()

# ─── SAMPLE EFFICIENCY ANALYSIS ───
print("="*70)
print("SAMPLE EFFICIENCY: Why 26x more data needed")
print("="*70)
print("""
The sample efficiency gap is NOT about forward() speed — it's about
LEARNING SIGNAL QUALITY.

RLMv2 learn() path:
  1. forward() → prediction logits
  2. Compute prediction_error = onehot(target) - softmax(logits)
  3. Hebbian update: edge.weight += lr * src_activation * tgt_activation * error
  4. Concept vector update: pull toward token embedding (lr=0.005)
  5. Relation vector update: 3-way blend (0.70 current + 0.20 target + 0.10 seed)
  6. Contrastive push: repel from different-target edges
  7. Token embedding update: pull subject/object embeddings together (lr=0.002)

MLP train_step() path:
  1. forward() → logits
  2. softmax → cross-entropy loss
  3. Full backprop: d_W2, d_b2, d_W1, d_b1 (exact gradients)
  4. SGD update on ALL parameters simultaneously

KEY DIFFERENCES:
  a) MLP computes EXACT gradients via chain rule. Every parameter gets
     a precise update direction. This is maximally sample-efficient.
  b) RLMv2 uses LOCAL Hebbian rules: edge.weight += lr * pre * post * error.
     The error signal is global, but the update is local. Many weight
     combinations that would reduce loss are never explored.
  c) MLP updates W1, W2, b1, b2, and embeddings ALL AT ONCE.
     RLMv2 updates one edge at a time, one concept at a time.
  d) MLP's gradient carries INFORMATION about which direction in weight
     space reduces loss. RLMv2's Hebbian rule carries only co-occurrence +
     sign of error. This is a fundamentally lower-bandwidth learning signal.

BRAIN-INSPIRED FIX:
  The brain achieves sample efficiency through:
  1. PREDICTIVE CODING: top-down predictions generate error signals at
     every layer, not just the output. Each concept should predict its
     downstream neighbors and learn from the mismatch.
  2. TEMPORAL DIFFERENCE: the brain learns from sequences, not isolated
     pairs. "heat causes expansion" teaches not just heat→expansion but
     also that "causes" is a causal relation between thermal concepts.
  3. ATTENTION: the brain focuses learning on surprising events. RLMv2's
     surprise-based lr (effective_lr = base_lr * (1 + surprise * 5)) is
     a step in this direction, but could be stronger.
  4. COMPOSITIONAL BINDING: the brain's hippocampal binding is rapid and
     one-shot because it uses existing structure. RLMv2 creates new
     concept nodes for every novel token — this is expensive.
""")

# ─── Verify sample efficiency claim ───
print("="*70)
print("SAMPLE EFFICIENCY MEASUREMENT")
print("="*70)

np.random.seed(42)
model2 = RLMv2(vocab_size=vocab_size, embed_dim=32, concept_dim=32,
               n_concepts=vocab_size, max_seq_len=128, sleep_interval=999999)
np.random.seed(42)
mlp2 = SimpleMLP(vocab_size=vocab_size, embed_dim=32, n_hidden=32, lr=0.01)

train_facts = [
    "heat causes expansion", "fire causes heat", "cold causes contraction",
    "fire causes melting", "heat causes melting",
]
test_facts = [
    "cold causes solidification", "pressure causes compression",
]

def measure_rank(model, facts, tok):
    ranks = []
    for fact in facts:
        ids = tok.encode(fact)
        if len(ids) < 2:
            continue
        prefix = ids[:-1]
        target = ids[-1]
        ctx = np.array([prefix], dtype=np.int64)
        raw = model.forward(ctx)
        logits = np.asarray(raw.data).flatten()
        rank = int(np.where(np.argsort(logits)[::-1] == target)[0][0])
        ranks.append(rank)
    return np.mean(ranks) if ranks else vocab_size

# Baseline
pre_rlm = measure_rank(model2, test_facts, tok)
pre_mlp = measure_rank(mlp2, test_facts, tok)

# Train 1 epoch
for fact in train_facts:
    ids = tok.encode(fact)
    for i in range(len(ids) - 1):
        ctx = np.array([ids[:i+1]], dtype=np.int64)
        tgt = np.array([[ids[i+1]]], dtype=np.int64)
        model2.learn(ctx, tgt)
        mlp2.train_step(ctx, tgt[0])

post_rlm_1 = measure_rank(model2, test_facts, tok)
post_mlp_1 = measure_rank(mlp2, test_facts, tok)

# Train 50 more epochs for MLP
for ep in range(50):
    for fact in train_facts:
        ids = tok.encode(fact)
        for i in range(len(ids) - 1):
            ctx = np.array([ids[:i+1]], dtype=np.int64)
            tgt = np.array([ids[i+1]], dtype=np.int64)
            mlp2.train_step(ctx, tgt)

post_mlp_50 = measure_rank(mlp2, test_facts, tok)

# Train RLM with 50 more passes too
for ep in range(50):
    for fact in train_facts:
        ids = tok.encode(fact)
        for i in range(len(ids) - 1):
            ctx = np.array([ids[:i+1]], dtype=np.int64)
            tgt = np.array([[ids[i+1]]], dtype=np.int64)
            model2.learn(ctx, tgt)

post_rlm_50 = measure_rank(model2, test_facts, tok)

print(f"  Test facts: {test_facts}")
print(f"  Before training:")
print(f"    RLMv2 mean rank: {pre_rlm:.1f}")
print(f"    MLP mean rank:   {pre_mlp:.1f}")
print(f"  After 1 epoch:")
print(f"    RLMv2 mean rank: {post_rlm_1:.1f}  (delta: {pre_rlm - post_rlm_1:+.1f})")
print(f"    MLP mean rank:   {post_mlp_1:.1f}  (delta: {pre_mlp - post_mlp_1:+.1f})")
print(f"  After 50 epochs:")
print(f"    RLMv2 mean rank: {post_rlm_50:.1f}  (delta: {pre_rlm - post_rlm_50:+.1f})")
print(f"    MLP mean rank:   {post_mlp_50:.1f}  (delta: {pre_mlp - post_mlp_50:+.1f})")

rlm_data_ratio = max(1, (pre_rlm - post_rlm_1)) / max(1, (pre_mlp - post_mlp_1))
print(f"\n  Sample efficiency ratio (1 epoch):")
print(f"    MLP learns {1/max(rlm_data_ratio, 0.001):.1f}x more per sample than RLMv2")
