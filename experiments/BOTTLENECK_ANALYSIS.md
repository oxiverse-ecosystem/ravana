"""
RLMv2 SETTLE LOOP BOTTLENECK ANALYSIS
======================================

Profiling results from experiments/profile_settle_loop.py on the RAVANA
settle loop in ravana_ml/nn/rlm_v2.py.

TIMING RESULTS (forward pass):
  RLMv2 forward():  ~2.3ms
  MLP forward():    ~0.016ms
  Ratio:            ~144x slower

WHERE TIME IS SPENT IN forward():
  concept_to_token_scoring   37-46%   (0.6-0.9ms)  <<<< BIGGEST BOTTLENECK
  spread_activation_phase1   25-29%   (0.5ms)
  score_active_edges          8-10%   (0.16ms)
  vector_analogy              6-9%    (0.14ms)
  spread_activation_phase2    5-6%    (0.1ms)
  classify_relation           5-9%    (0.08-0.16ms)
  everything else             ~3%

ROOT CAUSE #1: np.linalg.norm() CALLS
  319 calls to np.linalg.norm() in a single forward() pass.
  At ~4μs each, that's 1.28ms — over HALF the forward time.
  These happen inside:
    - spread_activation: edge relation_vector norm per edge per step
    - concept_to_token scoring: token embedding norms, target norms
    - vector analogy: node vector norms for cosine similarity
    - lateral inhibition: pairwise vector norms
  
  FIX: Pre-compute norms once, invalidate on mutation. Reduces to ~117 calls.
  SAVINGS: ~0.81ms per forward (35% reduction).

ROOT CAUSE #2: SERIAL CONCEPT-TO-TOKEN SCORING
  For each activated concept (~10-20), compute cosine similarity against
  ALL vocab_size tokens. Current: loop doing individual project+dot.
  
  FIX: Batch all concept vectors into a matrix, project once (one matmul),
  then batch cosine similarity: token_normed @ targets_normed.T.
  
  Serial: 0.733ms  →  Batch: 0.065ms  =  11.3x speedup.
  SAVINGS: ~0.67ms per forward (29% reduction).

ROOT CAUSE #3: REDUNDANT EDGE TRAVERSALS
  forward() traverses the edge set 5 separate times:
    1. vector_analogy: ALL edges for relation vectors
    2. spread_activation phase1: 3 steps × active × degree
    3. spread_activation phase2: 2 steps × ALL nodes × degree
    4. score_active_edges: active × degree
    5. merge_active_outgoing: active × degree AGAIN
  
  FIX: Merge scoring into the spread loop. During each spread step,
  accumulate matching_targets inline. Eliminate passes 4+5.
  SAVINGS: ~2.1x fewer edge traversals.

ROOT CAUSE #4: LEARN() CALLS FORWARD() INTERNALLY
  learn() calls forward() to get prediction, then does Hebbian updates.
  Total learn() time ≈ 2x forward + updates ≈ 5ms.
  MLP train_step() ≈ 0.2ms.
  
  This means per-sample wall-clock is ~25x slower than MLP.

PROJECTED OPTIMIZED PERFORMANCE:
  Current forward:    2.3ms
  After norm caching: 1.5ms  (-35%)
  After batch scoring: 0.8ms (-55% combined)
  After merged passes: 0.6ms (-15% combined)
  Projected ratio to MLP: ~38x (from 144x)

============================================================================
SAMPLE EFFICIENCY: WHY 26x MORE DATA
============================================================================

The sample efficiency gap is a DIFFERENT problem from the speed gap.
It's about LEARNING SIGNAL QUALITY, not compute.

RLMv2 learns via LOCAL Hebbian rules:
  edge.weight += lr * src_activation * tgt_activation * error
  
MLP learns via GLOBAL backpropagation:
  Every parameter gets an exact gradient via chain rule.

The Hebbian rule carries less information per sample:
  - Only updates the specific (src, tgt) edge that was involved
  - No credit assignment to intermediate concepts
  - No gradient flow through the graph topology
  - Each sample updates ~1-3 edges; MLP updates all ~10k params

WHAT THE BRAIN DOES DIFFERENTLY:
  1. PREDICTIVE CODING: Every layer generates predictions and learns
     from the mismatch. Not just the output layer. This gives each
     concept its own local error signal, dramatically increasing the
     bandwidth of the learning signal per sample.
  
  2. TEMPORAL CONTINUITY: The brain learns from temporal sequences,
     not isolated (input, target) pairs. "heat causes expansion"
     generates a stream of prediction errors at each moment, not
     just one at the end.
  
  3. ATTENTION-GATED PLASTICITY: Learning rate is modulated by
     attention/salience. Unexpected events get amplified learning.
     RLMv2's surprise-based lr is a start, but it's modulating
     a single global lr, not per-synapse.
  
  4. STRUCTURAL PLASTICITY: The brain doesn't just change weights —
     it grows new synapses, prunes unused ones, and myelinates
     frequently-used pathways. RLMv2's graph has fixed topology
     per sample; only edge weights change.

  5. SLEEP CONSOLIDATION: The brain replays and reorganizes during
     sleep. RLMv2 has this mechanism but it's set to sleep_interval=100
     which may be too infrequent for small datasets.

SPECIFIC STRUCTURAL CHANGES TO CLOSE THE GAP:

  A) Add predictive coding to forward():
     During spread activation, each node predicts its downstream
     neighbors. Prediction errors propagate backward and strengthen
     edges that resolve the error. This gives every active edge
     a learning signal, not just the final output edge.

  B) Batch gradient accumulation:
     Instead of updating edges one at a time, accumulate gradient-like
     signals across all active edges in one pass, then apply them.
     This mimics how backprop distributes credit.

  C) Concept-level cross-entropy:
     Instead of Hebbian on individual edges, compute cross-entropy
     loss over the concept_scores distribution and backprop through
     the concept_to_embed projection. This gives a global signal
     that reaches all activated concepts.

  D) Reduce forward() overhead:
     After the above compute optimizations (norm caching, batch scoring),
     forward() drops to ~0.6ms. The remaining 38x gap vs MLP is
     inherent to the graph-based architecture and would require
     FAISS for similarity search, sparse activation matrices, and
     C extensions for the spread loop.
"""
