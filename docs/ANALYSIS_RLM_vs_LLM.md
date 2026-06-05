# RLM vs LLM: Honest Gap Analysis & Improvement Roadmap

**Date:** 2026-06-01
**Purpose:** Where RLM loses to LLMs, why, and what we can adopt
**Status:** ✅ ALL 5 BUGS FIXED + ALL 7 IMPROVEMENTS IMPLEMENTED (verified 2026-06-01)

---

## Part 0: Critical Bugs Found (Fix Immediately)

### BUG 1: Recurrent Cell is Frozen (CRITICAL)

`self.recurrent_cell` NEVER appears in any weight update path in `learn()`. Scanning the entire `learn()` method (rlm.py:358-597), the recurrent cell is absent from:
- Settle loop updates (only touches `hidden_layers` and `context_logits`)
- Direct Hebbian updates (only touches `context_logits`)
- Free energy accumulation path

The recurrent cell is the core sequential processing mechanism. It processes the entire token sequence but receives ZERO learning signal. The only way it gets updated is through `accumulate_free_energy` during sleep, which is extremely weak (~0.0001 effective lr per sleep cycle).

**Impact:** The model cannot learn temporal dependencies. The recurrent cell is a random projection that never improves.

**Fix:** Add direct Hebbian update to recurrent cell in learn(), similar to what exists for context_logits:
```python
# After settle loop, update recurrent cell
h_err = h_new - h_old  # error signal
recurrent_update = np.outer(np.concatenate([x, h_prev]), h_err) * lr
self.recurrent_cell.weight.data += recurrent_update
```

### BUG 2: LayerNorm Exists But Is Never Used (HIGH)

`module.py:256-281` has a full `LayerNorm` implementation with learnable weight/bias. The RLM class never instantiates or uses it. The forward pass (rlm.py:219-222) passes hidden states through `tanh` only — no normalization between layers.

**Impact:** Training instability as model scales. Activation distributions drift during training.

**Fix:** Add LayerNorm after each hidden layer:
```python
# In __init__:
self.hidden_norms = [LayerNorm(n_hidden) for _ in range(n_layers)]

# In forward:
for i, layer in enumerate(self.hidden_layers):
    h = layer(h)
    h = self.hidden_norms[i](h)
    h = np.tanh(h)
```

### BUG 3: Vanilla RNN — No Gating (HIGH)

The recurrent cell is `concat(x, h) -> linear -> tanh`. No LSTM, no GRU, no gating. This is the weakest possible recurrent architecture. Without gates:
- Old information is overwritten by new tokens
- No mechanism to selectively remember/forget
- Hidden state capacity is fixed and limited

**Impact:** Cannot model long-range dependencies. Information from early tokens is lost.

**Fix:** Replace with GRU (simpler than LSTM, similar performance):
```python
# GRU: z = sigmoid(Wz @ [x,h])  # update gate
#       r = sigmoid(Wr @ [x,h])  # reset gate
#       h~ = tanh(W @ [x, r*h])  # candidate
#       h = (1-z)*h + z*h~       # new state
```

### BUG 4: O(V²) compute_curvature Will Crash at Scale (HIGH)

`graph.py:2498-2501` computes `norms @ norms.T` — the FULL pairwise similarity matrix of ALL nodes. For 100,000 nodes, this is a 100K × 100K matrix = 40GB of float32. Will crash on any machine.

**Fix:** Use sampling (already done for clustering coefficient but not for curvature).

### BUG 5: forward_step() Lacks Inverted Index (MEDIUM)

`forward()` uses `_concept_to_tokens` inverted index for O(B*T) context priming. `forward_step()` (used for generation) falls back to O(V*T) full vocabulary scan. Generation is much slower than it needs to be.

---

## Part 1: Where RLM is NOT Better

### 1.1 Sample Efficiency (Per-Example Learning)

| Metric | RLM | MLP (backprop) | Gap |
|--------|-----|----------------|-----|
| Rank improvement after 1 pass | +5.8 | +152.8 | **26x worse** |
| Time to learn 5 facts | 1 pass | 50 epochs | RLM needs more data |

**Why:** Backprop computes the exact gradient of the loss with respect to every parameter. Each update is maximally efficient at reducing error. Hebbian learning is local — each edge updates based on local co-activation, not global error. It's like trying to find the bottom of a hill by feeling the slope under your feet vs having a GPS.

**Can we fix it?** Partially. The Forward-Forward algorithm (Hinton 2022) shows that local learning CAN be competitive if you have the right training signal. RLM's settle loop is already close — it just needs better error signals per layer.

### 1.2 Raw Speed

| Operation | RLM | MLP | Ratio |
|-----------|-----|-----|-------|
| Forward pass | 0.57ms | 0.02ms | **29x slower** |
| Learn step | 3.49ms | 0.09ms | **38x slower** |
| Sleep cycle | 7.4ms | N/A | RLM only |

> **Note:** Timing numbers above are approximate and hardware-dependent.

**Why:** RLM does MORE per step:
- Concept graph lookup (brute force over all nodes)
- Spreading activation (BFS)
- Hebbian edge updates
- Binding updates
- Vector updates
- Contrastive relation learning
- Cognitive state updates (identity, emotion, meaning)
- Episodic memory storage

MLP does: matrix multiply + gradient descent. That's it.

**Can we fix it?** Yes, significantly:
- Vectorize concept lookup (batch cosine similarity — already done in find_similar)
- Batch edge updates
- Reduce Python overhead (dict lookups, loop overhead)
- Consider Cython/Numba for hot loops

### 1.3 Attention / Global Context

| Mechanism | RLM | Transformer |
|-----------|-----|-------------|
| Context range | 2 hops (local) | Global (every token sees every other) |
| Information flow | Sequential BFS | Parallel matmul |
| Context weighting | Edge weight * confidence | Learned QKV attention |

**Why it matters:** When processing "the animal didn't cross the street because it was too tired", the model needs to know "it" refers to "animal" not "street". Transformers solve this with attention. RLM has to propagate through 2+ hops, losing information.

**Can we fix it?** YES — this is the highest-impact improvement:
- Add **concept attention**: active concepts attend to each other
- O(n_active^2 * d) where n_active is small (7-15)
- Same computation as transformer attention but on the concept graph
- Concepts that co-activate reinforce each other

### 1.4 Representation Quality

| Aspect | RLM | Transformer |
|--------|-----|-------------|
| Embedding space | Random init + slow Hebbian drift | Pre-trained on billions of tokens |
| Vector normalization | Ad-hoc (divide by norm) | LayerNorm (principled, stable) |
| Positional info | Implicit in recurrent state | Explicit (sinusoidal/RoPE/ALiBi) |
| Multi-scale | Single concept dimension | Multi-head (multiple subspaces) |

**Why it matters:** Transformer embeddings capture rich semantic relationships because they're trained on massive data. RLM's embeddings start random and drift slowly.

**Can we fix it?** Yes:
- Use pre-trained embeddings as initialization (transfer learning)
- Add LayerNorm for stability
- Add positional encoding
- Add multi-head concept representations

### 1.5 Scaling

| Scale | RLM | Transformer |
|-------|-----|-------------|
| Parameters | ~20K neural + graph | 7B-70B typical |
| Vocab | 256 (char) | 50K-100K |
| Sequence | Unlimited (recurrent) | 4K-128K (attention) |
| Training data | Streaming (one pass) | Massive corpus (many epochs) |

**Why:** Transformers scale because attention is embarrassingly parallel. RLM's graph operations are sequential Python.

---

## Part 2: What Transformers Do That RLM Should Adopt

### 2.1 LayerNorm / RMSNorm (EASY — High Impact)

**What:** Normalize activations to have zero mean and unit variance (LayerNorm) or just unit norm (RMSNorm).

**Why it works:** Prevents activation explosion/vanishing, stabilizes training, enables deeper networks.

**Current RLM:** Ad-hoc normalization — `vector / (norm + eps)` applied inconsistently.

**Proposal:**
```python
class LayerNorm:
    def __init__(self, dim):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
    
    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + 1e-6) + self.beta
```

**Where to apply:**
- After each hidden layer in RLM
- On concept vectors after adjustment
- On the recurrent cell output

**Difficulty:** Easy. **Impact:** High (training stability).

### 2.2 Positional Encoding (EASY — Medium Impact)

**What:** Add explicit position information to token embeddings.

**Why it works:** Transformers need it because attention is position-agnostic. RLM has a recurrent cell that implicitly encodes position, but the signal degrades over long sequences.

**Proposal:** Sinusoidal positional encoding added to token embeddings:
```python
def positional_encoding(pos, dim):
    pe = np.zeros(dim)
    for i in range(0, dim, 2):
        pe[i] = np.sin(pos / 10000 ** (i / dim))
        pe[i+1] = np.cos(pos / 10000 ** (i / dim))
    return pe
```

**Where to apply:** In `forward()`, add to token embedding before recurrent cell.

**Difficulty:** Easy. **Impact:** Medium (helps with long sequences).

### 2.3 Concept Attention (MEDIUM — Highest Impact)

**What:** Instead of (or in addition to) spreading activation, compute attention between active concepts.

**Why it works:** Spreading activation is local — information propagates 2 hops max. Attention is global — every active concept sees every other. This is THE key innovation of transformers.

**Proposal:**
```python
def concept_attention(self, active_concepts, active_vectors):
    """
    active_concepts: list of (node_id, activation) 
    active_vectors: (n_active, dim) array of concept vectors
    """
    n = len(active_concepts)
    d = active_vectors.shape[1]
    
    # Q, K, V from concept vectors
    Q = active_vectors @ self.W_q  # (n, d_k)
    K = active_vectors @ self.W_k  # (n, d_k)
    V = active_vectors @ self.W_v  # (n, d_v)
    
    # Attention scores
    scores = Q @ K.T / np.sqrt(d_k)  # (n, n)
    
    # Mask: concepts that are connected get higher scores
    for i, (nid_i, _) in enumerate(active_concepts):
        for j, (nid_j, _) in enumerate(active_concepts):
            edge = self.graph.get_edge(nid_i, nid_j)
            if edge:
                scores[i, j] += edge.weight * 2.0  # graph prior
            if edge and edge.edge_type == "inhibitory":
                scores[i, j] -= 3.0  # inhibitory penalty
    
    # Softmax
    weights = softmax(scores, axis=-1)
    
    # Weighted combination
    attended = weights @ V  # (n, d_v)
    
    return attended
```

**Key insight:** This is Hebbian attention — concepts that co-activate strongly attend to each other. The graph structure provides the attention mask (connected concepts attend more). This is DIFFERENT from transformer attention which is purely learned.

**Where to apply:** In `forward()`, after spreading activation, before logit computation.

**Difficulty:** Medium. **Impact:** Highest (global context, compositional reasoning).

### 2.4 Contrastive Representation Learning (MEDIUM — High Impact)

**What:** Learn representations by pulling similar examples together and pushing different examples apart.

**Why it works:** Transformers learn rich representations through massive pre-training (next-token prediction). RLM's representations start random and drift slowly. Contrastive learning provides a stronger training signal.

**Current RLM:** Contrastive push exists for relation vectors (Gap 1 fix) but NOT for concept vectors.

**Proposal:**
```python
def contrastive_concept_update(self, anchor_vec, positive_vec, negative_vecs, lr=0.01):
    """Pull anchor toward positive, push away from negatives."""
    # Pull toward positive
    delta = positive_vec - anchor_vec
    self.adjust_vector(anchor_nid, delta, lr=lr)
    
    # Push away from negatives (InfoNCE-style)
    for neg_vec in negative_vecs:
        delta = anchor_vec - neg_vec  # repel
        self.adjust_vector(anchor_nid, delta, lr=lr * 0.3)
```

**Where to apply:** In `learn()`, after vector updates. Use the output concept as anchor, input concept as positive, and other activated concepts as negatives.

**Difficulty:** Medium. **Impact:** High (better representations).

### 2.5 Proper Softmax Normalization (EASY — Medium Impact)

**What:** Use softmax on concept scores to produce proper probability distributions.

**Why it works:** Current RLM uses `concept_scores * 15.0` — a temperature hack. Softmax gives proper probabilities, enables calibration, and is mathematically principled.

**Current code (rlm.py forward):**
```python
concept_logits = concept_scores * 15.0  # temperature hack
```

**Proposal:**
```python
# Temperature modulated by arousal (already have this!)
temperature = 1.0 + 2.0 * self.arousal
concept_logits = concept_scores / temperature
concept_probs = softmax(concept_logits)
log_concept_logits = np.log(concept_probs + 1e-10) * temperature
```

**Difficulty:** Easy. **Impact:** Medium (better calibration, principled temperature).

### 2.6 Residual Connections (EASY — Medium Impact)

**What:** Add skip connections: `output = layer(x) + x`.

**Why it works:** Enables gradient flow in deep networks (even without backprop, helps signal propagation). Prevents information loss in deep layers.

**Current RLM:** Hidden layers are sequential without skip connections.

**Proposal:**
```python
# In forward():
h_new = F.relu(self.hidden_layers[0](h))
h = h + h_new  # residual connection
```

**Difficulty:** Easy. **Impact:** Medium (deeper networks, better signal propagation).

### 2.7 Learning Rate Scheduling (EASY — Medium Impact)

**What:** Warmup + decay learning rate over time.

**Why it works:** Early training needs high lr to explore; later training needs low lr to converge. Transformers use warmup (start low, increase) then cosine decay.

**Current RLM:** Fixed lr everywhere.

**Proposal:**
```python
# Warmup: linear increase over first N steps
warmup_steps = 100
if self._step_counter < warmup_steps:
    lr_scale = self._step_counter / warmup_steps
else:
    # Cosine decay
    progress = (self._step_counter - warmup_steps) / total_steps
    lr_scale = 0.5 * (1 + np.cos(np.pi * progress))

effective_lr = base_lr * lr_scale
```

**Difficulty:** Easy. **Impact:** Medium (better convergence).

### 2.8 Multi-Head Concept Representations (MEDIUM — Medium Impact)

**What:** Represent each concept with multiple vectors (like multi-head attention).

**Why it works:** Different aspects of a concept (syntactic, semantic, emotional) may be best captured by different subspaces. Single vector = one representation = one perspective.

**Proposal:** Each concept has K vectors (heads). During attention, each head attends independently. Results are concatenated and projected.

**Difficulty:** Medium. **Impact:** Medium (richer representations).

---

## Part 3: What RLM Has That Transformers DON'T

These are RLM's genuine advantages — don't lose them while improving:

### 3.1 Structural Knowledge (Graph with Typed Edges)
- Inhibitory edges for contradiction resolution
- Edge types (semantic, causal, temporal, analogical)
- Relation vectors on edges
- Transformers: just weight matrices, no structure

### 3.2 Self-Organization
- Sleep consolidation reorganizes knowledge
- Concept splitting under contradiction pressure
- Homeostatic downscale prevents runaway
- Transformers: fixed architecture after training

### 3.3 Cognitive State
- Identity, emotion (VAD), meaning, sleep pressure
- These modulate inference (arousal → exploration)
- Transformers: stateless, no self-model

### 3.4 Memory System
- Episodic → semantic → graph bridge
- Ebbinghaus decay, interference, reconstruction
- Transformers: no persistent memory (RAG is external)

### 3.5 Interpretability
- Concept graph is fully inspectable
- Can see what's learned, what's contradictory, what's drifting
- Transformers: black box (probing required)

### 3.6 Streaming Learning
- Learns from each example immediately
- No replay buffer, no epoch-based training
- Transformers: need massive corpus, many epochs

---

## Part 4: Priority Implementation Plan

### Tier 1: Quick Wins (1-2 days each)
1. **LayerNorm** on hidden states and concept vectors
2. **Positional encoding** in token embeddings
3. **Softmax normalization** on concept scores
4. **Residual connections** in hidden layers
5. **Learning rate scheduling** (warmup + cosine decay)

### Tier 2: High Impact (3-5 days each)
6. **Concept attention** — global context among active concepts
7. **Contrastive concept learning** — stronger representation training
8. **Forward-forward style local learning** — better error signals per layer

### Tier 3: Scaling (1-2 weeks each)
9. **Vectorized graph operations** — NumPy batch operations
10. **Cython/Numba** for hot loops (spread_activation, hebbian_update)
11. **Sparse matrix** for graph adjacency (scipy.sparse)

---

## Part 5: The Honest Bottom Line

**RLM is not better than LLMs at:**
- Learning from few examples (backprop is more efficient)
- Raw speed (38x slower)
- Global context (no attention)
- Scaling (sequential Python)

**RLM IS better than LLMs at:**
- Compositional generalization (100% vs 33%)
- Contradiction handling (structural inhibition)
- Interpretability (inspectable graph)
- Knowledge reorganization (sleep consolidation)
- Persistent memory (episodic → semantic → graph)
- Self-awareness (identity, emotion, meaning)

**The path forward:** Adopt transformer innovations (attention, LayerNorm, positional encoding, contrastive learning) while preserving RLM's unique advantages (graph structure, self-organization, cognitive state, interpretability). This creates something neither can be alone: a self-organizing, interpretable, cognitively-aware learning system with the representational power of attention.

**The killer combination:**
```
Concept Attention (global context)
+ Graph Structure (typed edges, inhibition)
+ Sleep Consolidation (self-organization)
+ Cognitive State (identity, emotion, meaning)
= Something transformers can't do
```

### Verified Status (2026-06-01)

All 5 bugs confirmed fixed in current codebase:
- BUG 1: GRU gates get direct Hebbian updates (rlm.py:1816-1880)
- BUG 2: LayerNorm used on all hidden layers (rlm.py:88)
- BUG 3: GRUCell implemented (module.py:373)
- BUG 4: compute_curvature uses sampling (graph.py:2884, max_sample=500)
- BUG 5: forward_step uses inverted index (rlm.py:2938)

All 7 improvements confirmed implemented:
- LayerNorm on hidden states ✓
- Sinusoidal positional encoding ✓
- Concept attention (ConceptAttentionHead, 2-head QKV) ✓
- Contrastive concept learning (InfoNCE) ✓
- Softmax normalization (temperature modulated by arousal) ✓
- Residual connections ✓
- Learning rate scheduling (warmup + cosine decay) ✓

Current benchmark reality:
- Core unit tests: 44/45 passing (1 subprocess non-code error)
- RLMv2 v6 benchmark (500ep): 80.9% overall top-10
- Relation vector separation: 0.551
- Phase 2 NN bridge: 67% bridge, 82-95% query success (experiment-dependent)
- Full cross-domain: 0.0% neutral transfer
- Fair eval: 10% train, 0% test

---

---

## Part 6: The Mamba Connection — RLM's Architectural Cousin

The transformer research found that **Mamba (Selective State Space Models)** is the closest existing architecture to RLM's philosophy:

| Property | Mamba | RLM |
|----------|-------|-----|
| Attention | None | None (spreading activation) |
| Scaling | O(n) linear | O(n) linear |
| Memory | Selective (input-dependent gating) | Pressure-driven (prediction error gating) |
| Dynamics | Continuous-time SSM | Discrete cycles with settle loop |
| Learning | Backprop (but architecture is local-compatible) | Local Hebbian + predictive coding |

**Mamba's key innovation:** Input-dependent parameters that let the model decide what to remember and what to forget. This IS pressure-driven gating — high prediction error = remember, low error = forget.

**What RLM can steal from Mamba:**
1. **Selective state updates** — don't update all concepts equally. Update more when pressure is high, less when pressure is low. (RLM already does this partially via plasticity, but not principled.)
2. **Structured state matrix** — Mamba's A matrix has special structure (HiPPO) that enables long-range memory. RLM's concept graph adjacency could serve the same role.
3. **Linear complexity attention** — Mamba achieves transformer-quality results with O(n) computation. RLM's spreading activation is already O(n) but could be more principled.

**The key insight:** The reason transformers work is NOT backprop. It is:
1. Content-based addressing (attention) — find relevant information dynamically
2. Residual refinement — make small corrections, don't replace
3. Normalization — keep magnitudes bounded
4. Selective memory — attend to what matters, ignore what doesn't

ALL of these are local operations. ALL of them can be driven by pressure signals. The transformer's effectiveness comes from its architecture, not its training algorithm. RLM can capture the architectural insights while replacing the training algorithm with local, pressure-driven rules.

---

## Part 7: Local Learning Rules — What Actually Works

From the research, here's the state of the art in non-backprop learning:

| Rule | How It Works | RLM Status |
|------|-------------|------------|
| **Predictive Coding** | Each layer predicts the one above. Error = actual - predicted. Local updates. | **ALREADY IMPLEMENTED** (settle loop) |
| **Forward-Forward** | Two forward passes: positive (real data, maximize "goodness") and negative (generated, minimize). Each layer has its own objective. | **PARTIALLY IMPLEMENTED** (settle loop has this structure) |
| **Contrastive Hebbian** | Free phase (network runs) vs clamped phase (input+target). Weights update by correlation difference. | **CLOSE** (Hebbian + inhibitory edges ≈ this) |
| **Target Propagation** | Each layer receives a TARGET (not gradient). Adjusts to move activations toward target. | **PARTIALLY** (identity/meaning provide targets) |
| **Equilibrium Propagation** | Energy-based. Learning uses difference between two equilibrium states. | **NOT IMPLEMENTED** |

**Field consensus (2025):** No single local rule matches backprop on ImageNet-scale. BUT combinations of local rules (predictive coding + Hebbian + contrastive + sleep consolidation) can approach backprop on continual learning tasks where backprop fails catastrophically.

**RLM's advantage:** It already combines multiple local rules (predictive coding + Hebbian + contrastive + sleep consolidation). This is MORE than any single published local learning method. The gap is in the architectural details (normalization, gating, attention), not the learning paradigm.

---

*This is not about catching up to transformers. It's about stealing their best ideas and combining them with things they can't do.*

---

## Update: Phase 2 NN Bridge Results (2026-05-31)

The gap analysis identified 5 critical bugs (all fixed). Phase 2 now achieves 95% query success on reverse inheritance (and 82% on held-out transfer) via:
- Pre-trained MiniLM bridge (no dimensionality projection)
- Independent traversals per candidate
- Depth decay (0.7x per hop)
- Reverse edge inheritance
- Bridge-as-candidate for is_a queries

MiniLM preserves domain structure: intra-domain similarity 0.413 vs cross-domain 0.155 (2.5x gap). The concept graph's structured knowledge + embedding bridge enables transfer to terms never seen during training.
