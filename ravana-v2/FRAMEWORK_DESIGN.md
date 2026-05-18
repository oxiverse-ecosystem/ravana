# RAVANA Cognitive ML Framework — Design Specification

> A CPU-native ML framework where learning emerges from pressure-driven self-organization, not gradient descent.
> This is NOT a neural network framework. No autograd. No backprop. No loss functions. No GPU required.

---

## Part I: Why PyTorch/TensorFlow Need GPUs

### Core Computational Bottlenecks

| Operation | PyTorch/TF Cost | Why It's Expensive |
|-----------|----------------|-------------------|
| **Matrix Multiply (GEMM)** | O(n³) FLOPs per layer | `y = Wx + b` — a 4096×4096 layer is 69B FLOPs per forward pass |
| **Backpropagation** | 2× forward memory + compute | Must store ALL intermediate activations for chain rule reversal |
| **Autograd Graph** | O(nodes) memory per iteration | Builds a DAG tracing every tensor operation; graph rebuilt every step |
| **Batch Processing** | O(batch_size × params) | GD averages gradients across N samples — N simultaneous forward+backward passes |
| **Memory Bandwidth** | TB/s required | Moving 10B+ parameter weights between HBM and compute units every step |
| **Precision** | FP16/FP32 tensor ops | Full-precision floating point on every operation |

### Why GPUs Win Here

GPUs are specialized for **SIMT** (Single Instruction, Multiple Thread):
- ~16,000 CUDA cores on H100 vs ~16 CPU cores — 1000× parallelism
- Tensor Cores do 4×4 matrix multiply-accumulate in 1 cycle
- HBM3 gives 3.35 TB/s bandwidth vs DDR5's ~50 GB/s
- Result: **100-1000× speedup** for matrix-heavy workloads

### The Hidden Cost: Backpropagation Memory

```
Forward pass:  store all activations  →  GPU VRAM full
Backward pass: read activations back   →  compute gradients
Gradient step: apply optimizer (Adam)  →  update weights
```

A 70B parameter model needs:
- ~140 GB for weights (FP16)
- ~280 GB for activations (depends on sequence length)
- ~140 GB for optimizer states (Adam: 2× params)
- ~560 GB total → requires 8× H100 (80GB each)

**This is why nobody trains LLMs on CPU — the memory bandwidth alone makes it infeasible.**

---

## Part II: RAVANA's Fundamental Difference

### Not "Smaller Neural Network" — Different Physics Entirely

| PyTorch/TensorFlow | RAVANA |
|-------------------|--------|
| Loss function minimized | Pressure self-organized |
| Gradient descent | Governor regulation |
| Weight matrices | Concept graph + state vectors |
| Backpropagation | Hebbian co-activation |
| Autograd DAG | Localized constraint satisfaction |
| Batch normalization | Sleep consolidation |
| Weight decay | Identity regularization |
| GPU-parallel matmuls | CPU-sequential state updates |

### Why RAVANA Runs on CPU

**1. No Matrix Multiplies**
The governor operates on scalars and small vectors (5-10 dims). No GEMM anywhere.
- `governor.regulate()` = ~50 scalar ops: sigmoids, dampening, clamping
- `identity.update()` = ~20 scalar ops: momentum, penalty, bonus
- `resolution.compute()` = ~15 scalar ops: partial credit accumulation

**2. No Autograd Graph**
Each step is independent. No need to remember what happened 100 steps ago for gradient computation.
- Memory per step: O(1) — just current state vector
- PyTorch Transformer: O(layers × hidden_dim × seq_len) — grows with model size

**3. Sparse Activation**
At any timestep, only K=3-5 concepts are active out of thousands.
- Hebbian updates only on active concepts — O(K²) per step
- Compare: Neural network activates ALL neurons every forward pass

**4. Memory IS the Model**
Semantic memory is a graph (nodes + edges), not a weight matrix.
- Storage scales with O(concepts + edges), not O(hidden_dim²)
- Retrieval is graph walk, not matrix multiply

**5. Sleep = Consolidation, Not Another Epoch**
Sleep is a periodic maintenance phase, not repeated gradient steps.
- 4 stages: analysis → compression → contradiction resolution → integration
- Each stage operates on localized hot spots, never the full set

**6. Discrete Bottlenecks**
Mode selection, strategy choice, intent selection — all discrete (1-of-N).
- Compare: neural networks need continuous differentiable everywhere
- Discrete operations are cheap (comparisons, not multiplications)

### Hypothetical Compute Comparison

| Task | PyTorch (GPU) | RAVANA (CPU) |
|------|--------------|--------------|
| 10K training steps | ~1 min on A100 | ~2 sec on i7 |
| 1M parameter model | ~100 MB (weights + grads + opt) | ~5 MB (concept graph + edges) |
| Inference latency | 1-10 ms (batched) | 0.1-1 ms (single step) |
| Memory during training | O(params × precision × 3) | O(state_vector + active_concepts) |
| Power per step | ~400W (GPU) | ~15W (CPU) |

---

## Part III: Framework API Design

### Core Design Principle

> `State` is the only mutable object. All operations are pure functions on State.
> Learning happens through pressure accumulation, not parameter optimization.

### Abstraction Layers

```
┌─────────────────────────────────────────────────────────┐
│                    USER API LAYER                        │
│  learn(predictions, outcomes) → state update             │
│  predict(input) → output                                │
│  sleep() → consolidation                                │
│  query(memory) → knowledge                              │
├─────────────────────────────────────────────────────────┤
│                    COGNITIVE LAYER                        │
│  Governor (constraint satisfaction)                      │
│  Identity (self-concept stabilization)                   │
│  Resolution (continuous partial credit)                  │
│  Strategy (deliberate mode selection)                    │
│  Intent (dynamic objective generation)                   │
│  Belief (multi-hypothesis reasoning)                     │
├─────────────────────────────────────────────────────────┤
│                    PHYSICS LAYER                          │
│  Pressure accumulation → Δ state                          │
│  Hebbian co-activation → edge updates                     │
│  Sleep consolidation → topology reorganization             │
│  Homeostatic regulation → center-seeking forces            │
├─────────────────────────────────────────────────────────┤
│                    STATE LAYER                             │
│  CognitiveState {dissonance, identity, wisdom, cycle}     │
│  ConceptGraph {nodes, edges, confidences}                  │
│  MemorySystem {episodic, semantic, working}                │
│  GovernorState {clamp_rate, mode, diagnostics}             │
└─────────────────────────────────────────────────────────┘
```

### Proposed Public API

```python
# === TRAINING (the framework) ===

# 1. Initialize
framework = CognitiveFramework()
state = framework.initialize(
    concept_dim=10,           # state vector dimensionality per concept
    max_concepts=1000,        # global concept pool size
    k_active=5,               # top-K sparse activation
    governor_config=GovernorConfig(
        max_dissonance=0.95,
        target_dissonance=0.30,
        min_dissonance=0.15
    )
)

# 2. Train on sequential data
for episode, (input_vec, target_vec) in enumerate(data_stream):
    # Perception → concept activation
    active_concepts = framework.perceive(state, input_vec)
    
    # Prediction (forward pass, no loss function)
    predictions = framework.predict(state, active_concepts)
    
    # Learn from outcome (pressure-based update, not backprop)
    state = framework.learn(
        state,
        predictions=predictions,
        outcomes=target_vec,
        episode=episode
    )
    
    # Periodic consolidation
    if episode % 100 == 0:
        state = framework.sleep(state)

# 3. Inference
result = framework.infer(state, input_vec)
# Returns: {concepts, confidences, coherence, dissonance}

# === MEMORY QUERY ===

# Semantic knowledge retrieval
knowledge = framework.query(state, "what predicts X?")
# Returns: graph neighborhood around X concept

# Episodic recall
memories = framework.recall(state, dissonance_threshold=0.7)
# Returns: past states with high dissonance

# === INSPECTION ===

# Cognitive dashboard
report = framework.diagnose(state)
# Returns: coherence, pressure_map, clamp_rate, mode, wisdom_level
```

### Why This API Is CPU-Native

1. `perceive()` — similarity match against concept pool: O(concepts × dim) = ~10K ops
2. `predict()` — Hebbian activation spread across active edges: O(K²) = ~25 ops
3. `learn()` — governor regulation + pressure update: ~100 scalar ops
4. `sleep()` — localized perturbation on high-pressure zones: O(hot_spots)  
5. `infer()` — same as perceive + predict, no learning: ~10K ops

No operation involves matrix multiplication larger than 10×10.
No operation requires storing intermediate results for backpropagation.
No operation benefits from GPU parallelism (too few FLOPs, overhead dominates).

---

## Part IV: Learning Mechanism Detail

### Pressure = Learning Signal

```
traditional loss:        L = (y_pred - y_true)²
pressure accumulation:   ΔP = error × salience × (1 - confidence)

where:
  - error ∈ [0, 1]: prediction mistake magnitude
  - salience ∈ [0, 1]: relationship importance
  - confidence ∈ [0, 1]: belief certainty in the prediction
```

High-confidence failures → high pressure → forces reorganization.
Low-confidence failures → low pressure → ignores noise.
This replaces the loss function.

### Governor = Optimizer

```
traditional optimizer:   w -= lr × ∇L
governor regulation:     Δstate = clamp(Δproposed, constraints)

pipeline:
  1. Hard constraints (absolute ceilings/floors)
  2. Predictive dampening (slow before wall)
  3. Boundary pressure (sigmoid soft resistance)
  4. Center-seeking (homeostatic pull to target)
  5. Mode regulation (exploration vs recovery shaping)
  6. Final clamp (absolute enforcement)
```

No learning rate. No momentum (separate from identity). No weight decay.
The governor IS the optimizer.

### Identity = Regularizer

```
traditional weight decay:     L += λ||w||²
identity stabilization:       Iₜ = Iₜ₋₁ + ΔI_regulated + bonus - penalty

where:
  - ΔI_regulated: governor-approved identity change
  - bonus: +0.08 per successful resolution (streak-multiplied)
  - penalty: -0.08 per failed prediction (fixed)
  - recovery bias: growth boost when I < 0.5
  - stability damping: shrink delta when I > 0.85
```

Identity prevents catastrophic forgetting without weight decay.
It's a structural self-concept, not a numeric penalty term.

### Memory = Model Storage

```
traditional parameters:  W ∈ R^(d_in × d_out) — continuous weight matrix
RAVANA parameters:        ConceptGraph(V, E) — graph with sparse edges

storage comparison:
  - 10K parameter neural net: 40 KB (FP32) — dense, all used
  - 10K concept RAVANA: 5-15 KB — sparse, only active edges stored
  
inference cost:
  - Neural net: full matmul through all layers (always O(params))
  - RAVANA: activate K concepts, walk edges (O(K) active, K << concepts)
```

---

## Part V: Inference vs LLM

### How RAVANA Does Inference (NOT token generation)

RAVANA inference is **pattern completion through concept activation**:

```python
def infer(state, input_pattern):
    # 1. Perception: match input to concepts
    activations = []
    for concept in state.concepts:
        similarity = cosine(input_pattern, concept.weights)
        activations.append((concept.id, similarity))
    
    # 2. Sparse selection: top-K only
    active = top_k(activations, K=5)
    
    # 3. Activation spread via prediction edges (Hebbian)
    for _ in range(3):  # 3 propagation steps
        for concept in active:
            neighbors = state.graph.get_neighbors(concept.id)
            for neighbor, edge in neighbors:
                neighbor.activation += edge.weight * concept.activation * 0.5
        active = top_k(all_concepts, K=7)  # slight expansion allowed
    
    # 4. Integrated response
    return {
        "primary_concepts": [c.id for c in active[:3]],
        "confidence": mean([c.confidence for c in active]),
        "coherence": compute_coherence(active, state.graph),
        "dissonance": state.dissonance
    }
```

**Key differences from LLM inference:**
- LLM: autoregressive token generation (O(seq_len) sequential matmuls)
- RAVANA: single-pass concept activation spread (O(K) graph walks)
- LLM: output is token probabilities (next-token prediction)
- RAVANA: output is concept activations (pattern completion)
- LLM: context window limits working memory
- RAVANA: separates working memory (temporal) from semantic (permanent)

---

## Part VI: Comparison Summary

| | PyTorch/TensorFlow | RAVANA |
|---|---|---|
| **Core operation** | Matrix multiply | Governor regulation |
| **Learning signal** | Loss gradient | Pressure accumulation |
| **Optimizer** | SGD/Adam | Constraint-based regulation |
| **Model storage** | Weight matrices | Concept graph |
| **Memory during training** | Activations + grads + opt states | Current state only |
| **Hardware need** | GPU (CUDA cores, HBM) | CPU (any x86/ARM) |
| **Precision** | FP32/FP16/BF16 | FP32 (or even FP16) |
| **Parallelism needed** | Massive SIMT | Single-threaded fine |
| **Training memory** | O(params × 3 × precision) | O(concepts × state_dim) |
| **Inference** | Dense forward pass | Sparse concept spread |
| **Scaling** | Bigger models | More concepts |
| **Human analogy** | Pattern matcher | Cognitive organism |

---

## Part VII: Implementation Roadmap

### Phase 0 — Core Physics Engine (completed in v2)
- [x] Governor with hard/soft/center-seeking regulation
- [x] Identity engine with momentum and recovery bias
- [x] Resolution engine with partial credit accumulation
- [x] State manager orchestrating the pipeline
- [x] Clamp diagnostics for governor transparency

### Phase 1 — Cognitive Framework API (this design)
- [ ] `CognitiveFramework` class wrapping all modules
- [ ] `perceive()` — input → concept activation
- [ ] `predict()` — Hebbian activation spread
- [ ] `learn()` — pressure-based governor update
- [ ] `sleep()` — 4-stage consolidation
- [ ] `infer()` — inference (no state change)
- [ ] `query()` — semantic/episodic memory access

### Phase 2 — Concept Graph Engine
- [ ] Sparse concept pool with top-K activation
- [ ] Hebbian edge updates (co-activation → weight change)
- [ ] Prediction edge creation/pruning
- [ ] Confidence-gated contradiction pressure

### Phase 3 — Sleep Consolidation
- [ ] Pressure threshold triggering
- [ ] Localized perturbation (≤2 hops from contradiction)
- [ ] 4-stage sleep with abort-on-instability
- [ ] Tier-0 identity protection during sleep

### Phase 4 — Real-World Training
- [ ] Sequential data ingestion
- [ ] Pattern extraction → concept formation
- [ ] Long-term knowledge accumulation
- [ ] Curriculum difficulty ramping

---
