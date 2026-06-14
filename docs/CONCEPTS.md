# Core Concepts — Theoretical Foundations

> **Deep dive into the theoretical principles** behind RAVANA's pressure-driven cognitive architecture.

---

## Table of Contents

1. [Pressure-Driven Learning](#pressure-driven-learning)
2. [Free Energy & Prediction Error](#free-energy--prediction-error)
3. [ConceptGraph — Structured Knowledge](#conceptgraph--structured-knowledge)
4. [Hebbian & Anti-Hebbian Plasticity](#hebbian--anti-hebbian-plasticity)
5. [Spreading Activation](#spreading-activation)
6. [Sleep Consolidation (SWS + REM)](#sleep-consolidation-sws--rem)
7. [GRACE Governor — Regulation](#grace-governor--regulation)
8. [VAD Emotion Engine](#vad-emotion-engine)
9. [Triple Decomposition (RLMv2)](#triple-decomposition-rlmv2)
10. [Verb-Stem Offset Predictor](#verb-stem-offset-predictor)
11. [Intrinsic Motivation & Meaning](#intrinsic-motivation--meaning)
12. [Human Memory Architecture](#human-memory-architecture)
13. [Global Workspace — Consciousness Bottleneck](#global-workspace--consciousness-bottleneck)
14. [Dual Process Theory](#dual-process-theory)
15. [Comparative Analysis](#comparative-analysis)

---

## Pressure-Driven Learning

### The Core Thesis

```
Traditional ML:     Loss(θ) → ∇L(θ) → θ ← θ - η∇L    (gradient descent)
RAVANA:             Prediction Error → Free Energy → Self-Organization → Equilibrium
```

**Key insight**: In biological systems, learning is not driven by an external loss function computed over a dataset. Instead, *internal prediction errors create pressure (free energy)*, and the system self-organizes to reduce that pressure. This is **active inference** (Friston, 2010) implemented in software.

### Why Pressure, Not Gradients?

| Gradient Descent | Pressure-Driven |
|------------------|-----------------|
| Requires differentiable path | Works with discrete, non-differentiable ops |
| Centralized error signal | Distributed, local error signals |
| Catastrophic forgetting | Sleep consolidation prevents forgetting |
| Fixed architecture | Self-organizing structure (graph growth) |
| No intrinsic motivation | Meaning/identity/emotion as first-class |

### Mathematical Formulation

Let the system state be `s ∈ S`, prediction `p = f(s)`, target `t`.

**Traditional**: `L = ||p - t||²`, `Δθ = -η ∂L/∂θ`

**RAVANA**: 
```
Free Energy F(s) = Σ_i w_i * |p_i - t_i|  (5 channels)
Pressure = ∂F/∂s  (not computed analytically — accumulated locally)
Δs ∝ -Pressure  (via Hebbian plasticity, inhibition, sleep)
```

The system settles into **attractors** (low free energy states) rather than minimizing a global loss.

---

## Free Energy & Prediction Error

### Five-Channel Accumulator

Free energy is not a scalar — it's a **5-dimensional vector** tracking different error types:

```python
FreeEnergy = {
    "semantic":      # Concept-level prediction error
    "linguistic":    # Token/sequence prediction error  
    "episodic":      # Memory retrieval error
    "contradiction": # Belief conflict error
    "abstraction":   # Hierarchical compression error
}
```

Each channel has independent dynamics:
- **Accumulation**: `FE_channel += error * salience`
- **Decay**: `FE_channel *= (1 - rate)` per cycle
- **Threshold**: Sleep triggered when `total_FE > threshold`

### Local vs Global

In backprop, error flows backward through the *entire* network. In RAVANA:

```
Prediction error at node N → FE at N → Hebbian update on N's edges
                                                         ↓
                                              No global gradient needed
```

**Each concept node tracks its own `prediction_free_energy`** — truly local learning signal.

---

## ConceptGraph — Structured Knowledge

### Why Graphs, Not Matrices?

| Dense Weight Matrix | ConceptGraph |
|---------------------|--------------|
| Fixed size | Dynamic growth |
| O(n²) parameters | O(edges) — sparse |
| No semantics | Typed edges (causal, temporal, ...) |
| Black box | Inspectable, queryable |
| Catastrophic interference | Structural separation |

### Node Design: Dual Vectors

Each concept has **two vectors**:

```
active_vector   — Fast plastic, tracks recent experience
core_vector     — Slow anchor, identity-preserving
genesis_vector  — Original, for drift measurement
```

**Why?** Prevents catastrophic drift while allowing adaptation. During sleep, active vectors consolidate toward core vectors.

### Edge Design: Rich Typed Relations

```
Edge = {
    source, target,
    weight ∈ [0,1],
    confidence ∈ [0,1],
    relation_type ∈ {semantic, causal, temporal, analogical, contextual, inferred},
    edge_type ∈ {excitatory, inhibitory},
    relation_vector ∈ ℝ^16,      # learned relational embedding
    predicate_token_id,          # verb-level discrimination
    # Consolidation
    fisher_importance,           # EWC for lifelong learning
    posterior_alpha, posterior_beta,  # Bayesian weight uncertainty
    # Multi-agent
    agent_weights: {agent_id: weight},
    source_metadata: {epistemic_status, is_user_statement, ...}
}
```

### Hierarchical Abstraction

```
Level 0 (Leaf):    "heat", "expansion", "melts"
Level 1 (Cluster): [heat, fire, burns] → "thermal_process"
Level 2 (Abstract): [thermal_process, chemical_reaction] → "transformation"
```

Created during sleep via **co-activation clustering** — concepts that fire together abstract together.

---

## Hebbian & Anti-Hebbian Plasticity

### Hebbian: "Fire Together, Wire Together"

```python
def hebbian_update(source, target, coactivation, lr=0.03):
    # coactivation = act_source * act_target
    edge.weight = min(1.0, edge.weight + lr * coactivation)
    edge.confidence = min(1.0, edge.confidence + lr * 0.1)
    
    # Update relation vector toward (target - source)
    rel = target.vector - source.vector
    edge.relation_vector = 0.9 * edge.relation_vector + 0.1 * rel
```

**Prediction**: After learning A→B, activating A spreads activation to B.

### Anti-Hebbian: Competition

```python
def anti_hebbian_update(source, target, coactivation, lr=0.02):
    # Weakens edges between simultaneously active but unrelated concepts
    edge.weight = max(0.0, edge.weight - lr * coactivation)
```

**Function**: Implements **lateral inhibition** — prevents everything connecting to everything.

### Structural Plasticity: Graph Topology Changes

```python
def structural_step():
    # Prune: remove edges with weight < prune_threshold AND confidence < 0.1
    # Form: create edges between frequently co-activated nodes (no existing edge)
```

---

## Spreading Activation

### Algorithm

```python
def spread_activation(graph, active_nids, steps=3, k_active=5, decay=0.3):
    for step in range(steps):
        # 1. Collect from neighbors
        for (src, tgt), edge in graph.edges.items():
            if src in active_nids:
                if edge.edge_type == "excitatory":
                    graph.nodes[tgt].activation += graph.nodes[src].activation * edge.weight * decay
                elif edge.edge_type == "inhibitory":
                    graph.nodes[tgt].activation -= graph.nodes[src].activation * edge.weight * decay
        
        # 2. Top-k competition (global inhibition)
        active_nids = top_k_nodes_by_activation(graph, k_active)
        
        # 3. Decay all
        for nid in active_nids:
            graph.nodes[nid].activation *= (1 - decay)
    
    return active_nids
```

### Relation-Type Filtering (RLMv2)

```python
def spread_by_relation(graph, source_nid, relation_type, steps=2):
    # Only traverse edges matching relation_type
    for (src, tgt), edge in graph.edges.items():
        if src == source_nid and edge.relation_type == relation_type:
            graph.nodes[tgt].activation += ...
```

Enables **"heat" --causes--> ?"** queries by filtering for `causal` edges.

---

## Sleep Consolidation (SWS + REM)

### Biological Inspiration

| Sleep Stage | Brain Function | RAVANA Implementation |
|-------------|----------------|----------------------|
| **SWS** (Slow-Wave) | Hippocampal replay, synaptic downscaling | Structured replay + homeostatic downscale |
| **REM** | Creative recombination, emotional processing | Dream sabotage (counterfactuals, valence flips) |

### Four-Stage Sleep Cycle

```python
def execute_sleep_cycle():
    # Stage 1: Topology Analysis
    pressure_zones = identify_high_pressure_areas()
    
    # Stage 2: Pattern Compression
    strengthen_coactivation_clusters(pressure_zones)
    apply_dream_sabotage()  # REM-style perturbations
    
    # Stage 2.5: Abstraction Compression
    hierarchical_merge_frequent_coactivations()
    
    # Stage 3: Contradiction Resolution
    rewire_weakest_edges_in_pressure_zones()
    
    # Stage 3.5: Model Update (Hippocampal Replay)
    replay_memories_through_graph()
    update_embedding_vectors()
    prune_unreinforced_edges()
    
    # Stage 4: Integration
    if coherence_dropped > 5%: rollback()
    reduce_pressure()
```

### Dream Sabotage (REM)

```python
def apply_dream_sabotage(memories, rate=0.2):
    for mem in sample(memories):
        if random() < 0.20:      # Counterfactual reversal
            reverse_causal_direction(mem)
        if random() < 0.10:      # Emotional flip
            flip_valence(mem)
        if random() < failure_oversample:  # Failure oversampling
            replay_failure(mem)
```

**Why?** Prevents overfitting, tests robustness, enables creative generalization.

### Interleaved Replay (Lifelong Learning)

```python
class ReplayBuffer:
    def __init__(self, max_size=500):
        self.domain_buffers = defaultdict(list)  # domain → experiences
    
    def add(self, experience, domain_tag):
        self.domain_buffers[domain_tag].append(experience)
    
    def sample_for_sleep(self, n=20):
        # Sample proportionally from ALL domains
        return balanced_sample(self.domain_buffers, n)
```

**Result**: Eliminates catastrophic forgetting (12% → 0% retention drop).

---

## GRACE Governor — Regulation

### Why a Governor?

In gradient descent, stability comes from:
- Weight decay (L2)
- Gradient clipping
- Learning rate schedules
- Batch normalization

**All are heuristic hacks.** RAVANA makes regulation **explicit, first-class, and diagnostically observable.**

### Four Regulatory Layers

```
┌─────────────────────────────────────────────────────────┐
│ 1. HARD CONSTRAINTS (Non-negotiable)                    │
│    dissonance ∈ [0.15, 0.95]                            │
│    identity ∈ [0.10, 0.95]                              │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 2. PREDICTIVE DAMPENING (Look-ahead)                    │
│    dd *= 1 - max(0, predicted_dissonance - target) * k  │
│    "Slow down before you hit the wall"                  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 3. BOUNDARY PRESSURE (Air resistance)                   │
│    dd *= (1 - pressure)^boundary_k  near limits         │
│    Prevents overshoot                                    │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 4. CENTER-SEEKING (Homeostatic pull)                    │
│    dd += -k * (current - target)                        │
│    Always pulls toward healthy setpoint                 │
└─────────────────────────────────────────────────────────┘
```

### Diagnostic Observability

```python
gov.get_clamp_report()
# ==================================================
# 📊 CLAMP DIAGNOSTICS REPORT
# ==================================================
# Alignment Score: 87.3% (higher = better)
#   Upstream suggestions: 1,247
#   Total clamps applied: 159
# 
# By Variable:
#   Dissonance: 89 clamps (7.1% rate)
#   Identity:   70 clamps (5.6% rate)
# 
# By Layer:
#   Hard constraints: 112
#   Final clamp:      47
# 
# Mean correction: 0.0234
# Significant corrections (>5%): 3 (1.9%)
# ==================================================
```

**This is unprecedented** — you can *see* exactly how often and why the governor overrides upstream modules.

---

## VAD Emotion Engine

### Differential Equations

Valence, Arousal, Dominance evolve via coupled ODEs (Euler integration, dt=1):

```
dV/dt = -γ_v * V + β_v * (reward - punishment)
dA/dt = -γ_a * A + β_a * |prediction_error| + β_a2 * novelty
dD/dt = -γ_d * D + β_d * (control - helplessness)
```

**Parameters** (configurable):
- `γ` = decay rates (return to baseline)
- `β` = sensitivity to inputs

### Modulation of Inference

```python
def get_inference_modulation(vad):
    mod = {}
    if vad.arousal > 0.7:
        mod["exploration_bonus"] = 0.2      # High arousal → explore
    if vad.valence > 0.5:
        mod["prediction_confidence"] = 1.2  # Positive → trust predictions
    if vad.dominance > 0.6:
        mod["concept_activation"] = 1.15    # High dominance → stronger concepts
    return mod
```

### Emotion as Compute, Not Afterthought

| Traditional ML | RAVANA |
|----------------|--------|
| Emotion = cosmetic output | Emotion = internal state modulating learning |
| Fixed temperature | Dynamic exploration via arousal |
| No valence | Valence gates prediction trust |
| No dominance | Dominance scales concept activation |

---

## Triple Decomposition (RLMv2)

### From Sequences to Triples

```
Traditional:  "heat causes expansion" → [h, e, a, t, _, c, a, u, s, e, s, ...]
RLMv2:        "heat causes expansion" → (subject="heat", relation="causes", object="expansion")
```

### Architecture

```
Input text
    ↓
Decompose → (subject_token, relation_token, object_token)
    ↓
Classify relation_token → relation_type_embedding (CAUSAL, SEMANTIC, TEMPORAL, ...)
    ↓
Find subject concept in graph
    ↓
Spread activation from subject, FILTERED by relation_type
    ↓
Score activated nodes against ALL token embeddings
    ↓
Logits over vocabulary
```

### Why Triples?

| Sequence (Transformer) | Triple (RLMv2) |
|------------------------|----------------|
| Learns statistical co-occurrence | Learns *relational structure* |
| No explicit compositionality | Built-in compositionality: S + R ≈ O |
| O(n²) attention | O(edges) spreading activation |
| Black box | Inspectable graph paths |
| Poor OOD generalization | Analogy via verb-stem offsets |

---

## Verb-Stem Offset Predictor

### The Insight

For a given verb, the **subject→object offset is consistent**:

```
heat --causes--> expansion
fire --causes--> heat
ice --melts--> water
  ↓
offset("causes") ≈ expansion - heat ≈ heat - fire
offset("melts") ≈ water - ice
```

### Implementation

```python
# During training, accumulate per-verb offsets
global_relation_priors[verb_token] = E[target_embed - source_embed | verb]

# During inference
predicted_object = source_embed + global_relation_priors[query_verb]
```

### Results

| Metric | Before | After |
|--------|--------|-------|
| RP-only cross-domain top-10 | 3.3% | **6.7%** |
| Same-subject different-verb | Failed | Works |

---

## Intrinsic Motivation & Meaning

### Meaning Equation

```
M = w₁(-D) + w₂(I) + w₃(P) × (1 + κ × E)
```

Where:
- `D` = dissonance (lower = better)
- `I` = identity strength
- `P` = prediction accuracy
- `E` = cognitive effort invested
- `κ` = effort multiplier

**Interpretation**: Meaning emerges from *reducing dissonance*, *maintaining identity*, *accurate prediction*, *weighted by effort*.

### Why Intrinsic?

- No external reward function needed
- Self-sustaining exploration
- Aligns with cognitive science (Ryan & Deci, Self-Determination Theory)

---

## Human Memory Architecture

### Three-Store Model

```
┌─────────────────┐     Sleep Consolidation     ┌─────────────────┐
│   WORKING       │ ─────────────────────────▶  │   EPISODIC      │
│  (7±2 items,    │    Replay + decay +         │  (5000,        │
│   immediate)    │    Ebbinghaus               │   half-life 1hr)│
└─────────────────┘                             └────────┬────────┘
                                                         │
                                    ┌────────────────────┘
                                    ▼ Sleep Consolidation
                                                         │
                                              ┌──────────┴──────────┐
                                              │    SEMANTIC         │
                                              │  (10000, permanent) │
                                              └─────────┬───────────┘
                                                        │
                                        Bridge to ConceptGraph
```

### Episodic → Semantic Consolidation

```python
def consolidate():
    for episodic_memory in active_episodic:
        if memory.importance > threshold and memory.access_count > 2:
            # Create semantic memory
            semantic = SemanticMemory(
                content=episodic.content,
                strength=episodic.importance * 0.5,
                source_episodes=[episodic.id],
            )
            # Vector drift toward concept centroid
            semantic.vector = blend(episodic.vector, concept_centroid)
```

### Ebbinghaus Forgetting

```python
def apply_decay():
    for mem in episodic_memories:
        age_hours = (now - mem.timestamp) / 3600
        mem.confidence *= exp(-age_hours / half_life_hours)
        if mem.confidence < 0.05:
            forget(mem)
```

### Interference

```python
def recall(query, limit=10):
    # Similar memories interfere
    candidates = vector_search(query)
    for c in candidates:
        c.confidence -= interference_factor * sum(
            similarity(c, other) * other.confidence 
            for other in candidates if other != c
        )
    return top_k(candidates, limit)
```

---

## Global Workspace — Consciousness Bottleneck

### Competitive Broadcast

```python
class GlobalWorkspace:
    def submit(content):
        # Content competes for broadcast slot
        # Winner broadcasts to ALL modules
        pass
```

### Architecture

```
Module A → [Content 1, act=0.8] ─┐
Module B → [Content 2, act=0.6] ─┤ COMPETITION (capacity=4)
Module C → [Content 3, act=0.9] ─┘
                              ↓
                    WINNER broadcasts globally
                              ↓
              All modules receive broadcast content
              (top-down attention, belief update, etc.)
```

### Why Bottleneck?

- **Serializes** parallel module outputs
- **Prevents** chaotic multi-module conflict
- **Enables** meta-cognition (inspecting broadcast content)

---

## Dual Process Theory

### System 1 / System 2

| Property | System 1 (Fast) | System 2 (Slow) |
|----------|-----------------|-----------------|
| Speed | ~ms | ~seconds |
| Effort | Low | High |
| Trigger | Low dissonance, high identity | High dissonance, low identity |
| Mechanism | Hebbian spread, pattern match | Deliberative simulation, planning |
| Error rate | Higher | Lower |

### Routing Decision

```python
def route(dissonance, identity, novelty, stakes, time_pressure):
    if dissonance < 0.3 and identity > 0.7:
        return SYSTEM1_FAST
    elif dissonance > 0.6 or stakes > 0.5:
        return SYSTEM2_SLOW
    else:
        return SYSTEM1_FAST  # default
```

---

## Comparative Analysis

### RAVANA vs Transformers

| Aspect | Transformer (LLM) | RAVANA |
|--------|-------------------|--------|
| **Learning** | Backprop on massive data | Hebbian + sleep on streaming data |
| **Architecture** | Fixed layers, attention | Dynamic graph, spreading activation |
| **Memory** | Weights + context window | ConceptGraph + episodic/semantic + sleep |
| **Generalization** | Statistical (next-token) | Relational (analogy, composition) |
| **Compute** | GPU, massive parallel | CPU, NumPy, sequential |
| **Forgetting** | Catastrophic (fine-tune) | Solved (interleaved replay) |
| **Interpretability** | Low (attention viz) | High (graph inspection) |
| **Emotion/Identity** | None | First-class citizens |

### RAVANA vs Other Local Learning

| Method | Plasticity | Structure | Sleep | Regulation |
|--------|-----------|-----------|-------|------------|
| **RAVANA** | Hebbian + anti-Hebbian + structural | Dynamic typed graph | SWS+REM+sabotage | GRACE Governor |
| **Equilibrium Prop** | Contrastive Hebbian | Fixed | No | Implicit |
| **Forward-Forward** | Layer-wise | Fixed | No | Implicit |
| **Predictive Coding** | Error-driven | Fixed hierarchy | No | Implicit |
| **NEF/Spaun** | Hebbian | Fixed vocab | No | Basal ganglia |

---

## Key References

### Theoretical Foundations

1. **Free Energy Principle** — Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*.
2. **Active Inference** — Friston et al. (2017). "Active inference: a process theory." *Neural Computation*.
3. **Hebbian Learning** — Hebb, D.O. (1949). *The Organization of Behavior*.
4. **Spreading Activation** — Collins, A.M. & Loftus, E.F. (1975). "A spreading-activation theory of semantic processing." *Psychological Review*.
5. **Sleep Consolidation** — Diekelmann, S. & Born, J. (2010). "The memory function of sleep." *Nature Reviews Neuroscience*.
6. **Global Workspace** — Baars, B.J. (1997). *In the Theater of Consciousness*.
7. **Dual Process** — Kahneman, D. (2011). *Thinking, Fast and Slow*.
8. **Self-Determination Theory** — Ryan, R.M. & Deci, E.L. (2000). "Self-determination theory and the facilitation of intrinsic motivation." *American Psychologist*.

### RAVANA-Specific

- `docs/RAVANA_REPORT.md` — Technical paper: "From Zero to Generalization"
- `docs/PAPER_DRAFT.md` — Academic draft: "Pressure-Driven Self-Organization"
- `docs/SCIENCE_DIRECT_MANUSCRIPT.md` — "Beyond Reward Maximization"
- `docs/ANALYSIS_RLM_vs_LLM.md` — Gap analysis vs Transformers
- `docs/EXTERNAL_AUDIT.md` — External collaborator audit

---

## See Also

- [Architecture Overview](ARCHITECTURE.md)
- [ML Framework](ML_FRAMEWORK.md)
- [Cognitive Core](COGNITIVE_CORE.md)
- [Experiments](EXPERIMENTS.md)