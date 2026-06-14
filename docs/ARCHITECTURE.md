# RAVANA Architecture Reference

> **Complete technical reference for the RAVANA cognitive architecture.** This document covers all three layers of the system, their interactions, and the theoretical foundations.

---

## Table of Contents

1. [Overview](#overview)
2. [Three-Layer Architecture](#three-layer-architecture)
3. [Layer 1: `ravana_ml/` — ML Framework](#layer-1-ravana_ml--ml-framework)
4. [Layer 2: `ravana-v2/` — GRACE Cognitive Core](#layer-2-ravana-v2--grace-cognitive-core)
5. [Layer 3: `ravana/` — Unified Package](#layer-3-ravana--unified-package)
5. [Data Flow & Integration](#data-flow--integration)
6. [Key Algorithms](#key-algorithms)
7. [Configuration Reference](#configuration-reference)

---

## Overview

RAVANA implements **pressure-driven self-organization** as an alternative to gradient-based learning. The core thesis:

```
Traditional ML:  Loss → Gradient → Weight Update
RAVANA:          Prediction Error → Pressure (Free Energy) → Self-Organization → Equilibrium
```

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **No backpropagation** | Hebbian/anti-Hebbian plasticity + structural plasticity |
| **CPU-native, NumPy-only** | Single dependency: `numpy>=1.20` |
| **Concept-first representation** | ConceptGraph with typed edges, not dense matrices |
| **Sleep as consolidation** | SWS + REM cycles with dream sabotage |
| **Cognition as regulation** | GRACE Governor with hard constraints |
| **Emotion as compute** | VAD (Valence/Arousal/Dominance) differential equations |

---

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LAYER 3: ravana/                          │
│                  Unified pip-installable package                 │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────────┐   │
│  │   nn/       │  │ cognitive/  │  │ graph/ propagation/    │   │
│  │  (RLM)      │  │Framework +  │  │ world/ lab/            │   │
│  │             │  │ 23+ modules │  │                        │   │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬─────────────┘   │
└─────────│───────────────│─────────────────────│──────────────────┘
          │               │                     │
          ▼               ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: ravana_ml/ (5,200+ lines)    LAYER 2: ravana-v2/ (19,500+ lines) │
│  ┌────────────────────────────┐      ┌────────────────────────────────┐    │
│  │ • graph.py (ConceptGraph)  │      │ • GRACE Phase A–O (27 modules) │    │
│  │ • nn/rlm.py (RLM v1)       │      │ • CognitiveCycle orchestration │    │
│  │ • nn/rlm_v2.py (RLM v2)    │      │ • Persistent human memory      │    │
│  │ • nn/module.py             │      │ • Sleep/dream consolidation    │    │
│  │ • tensor.py                │      │ • VAD emotion engine           │    │
│  │ • free_energy.py           │      │ • Global workspace             │    │
│  │ • plasticity.py            │      │ • Meta-cognition & belief      │    │
│  │ • propagation.py           │      │ • Social epistemology          │    │
│  │ • tokenizer.py             │      │ • Meaning/intrinsic motivation │    │
│  │ • currencies.py            │      │ • Planning & world model       │    │
│  └────────────────────────────┘      └────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: `ravana_ml/` — ML Framework

The ML framework provides a PyTorch-compatible API built entirely on NumPy.

### Core Modules

#### `graph.py` — ConceptGraph (3,678 lines)

The central knowledge representation. A heterogeneous graph with:

**Nodes (ConceptNode):**
- `vector` — active vector (fast-adapting, plastic)
- `core_vector` — identity anchor (slow, drift-resistant)
- `genesis_vector` — original vector for drift tracking
- `activation` — current spreading activation level
- `salience`, `prediction_free_energy`, `stability`, `confidence`
- Hierarchical fields: `level`, `parent`, `children`, `abstraction_degree`
- Temporal fields: `last_activated`, `activation_history`, `temporal_context`

**Edges (ConceptEdge):**
- Typed: `semantic`, `causal`, `temporal`, `analogical`, `contextual`, `inferred`, `inhibitory`
- Weight ∈ [0, 1] with sign (inhibitory = negative)
- `confidence`, `stability`, `prediction_free_energy`
- `forward_pred_count` / `backward_pred_count` for bidirectional tracking
- `predicate_token_id` — verb-level discrimination
- `relation_vector` — learned relational embedding
- EWC & Bayesian posterior: `fisher_importance`, `posterior_alpha/beta`
- Multi-agent weights: `agent_weights`, `source_metadata`

**Key Operations:**
```python
graph = ConceptGraph(dim=64, max_nodes=10000)
nid = graph.add_node(vector, label="heat")
graph.add_edge(source, target, weight=0.5, relation_type="causal")
graph.activate(nid, amount=1.0)
graph.spread_activation(steps=3, k_active=5, decay=0.3)
similar = graph.find_similar(vector, k=10)
graph.hebbian_update(source, target, coactivation, lr=0.03)
graph.homeostatic_downscale(protection_threshold=0.8)
graph.reconcile_contradictions()
```

#### `nn/module.py` — Neural Modules (473 lines)

PyTorch-like module hierarchy without autograd:

```python
class Module:
    def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
    def parameters(self): ...
    def state_dict(self): ...
    def load_state_dict(self, sd): ...
    def register_module(self, name, module): ...

class Linear(Module):          # weight: (out, in), no bias by default
class Embedding(Module):       # token → vector
class LayerNorm(Module):       # mean/var normalization
class GRUCell(Module):         # gated recurrent unit
class ConceptAttentionHead(Module):  # multi-head QKV over concepts
```

All params are `Parameter` (wraps `StateTensor` with `salience`, `free_energy`, `stability`, `decay`).

#### `nn/rlm.py` — RLM v1 (3,931 lines)

Recursive Learning Model with predictive coding:

- **Token embedding** → positional encoding → **GRU** → hidden layers → **concept predictor**
- **5-path logit blend**: concept attention, context logits, RP analogy, sparse concept, direct latent
- **Settling loop**: 5 iterations of predictive coding state updates
- **Free energy**: 5-channel accumulator (semantic, linguistic, episodic, contradiction, abstraction)
- **Sleep**: interleaved replay (domain-tagged), contrastive hidden states, BP-trained RP
- **Cognitive state**: unified via `CognitiveCurrencies`

#### `nn/rlm_v2.py` — RLM v2 (5,013 lines)

Triple decomposition architecture:

```
Input: "heat causes expansion"
       ↓ decompose
(subject="heat", relation="causes", object="expansion")
       ↓ classify relation
"causes" → CAUSAL type embedding
       ↓ spread activation
subject_node → (filter by CAUSAL) → activated target nodes
       ↓ score against vocab
Logits over vocabulary
```

Key differences from v1:
- No character-level GRU
- Spreading activation as **sole** inference mechanism
- Learned relation type embeddings (not keyword-based)
- Hebbian learning on `(subject @ relation_type) → object`
- Verb-stem offset predictor for cross-domain generalization

#### `free_energy.py` — FreeEnergyAccumulator (90 lines)

Five independent channels tracking prediction errors:

```python
class FreeEnergyAccumulator:
    def accumulate_semantic(self, error, salience=1.0): ...
    def accumulate_linguistic(self, error, salience=1.0): ...
    def accumulate_episodic(self, error, salience=1.0): ...
    def accumulate_contradiction(self, error, salience=1.0): ...
    def accumulate_abstraction(self, error, salience=1.0): ...
    def total(self) -> float: ...
    def decay(self, rate=0.1): ...
```

#### `plasticity.py` — Plasticity Rules (77 lines)

```python
class HebbianPlasticity:      # Δw = lr * pre * post
class AntiHebbianPlasticity:  # Δw = -lr * pre * post (competition)
class StructuralPlasticity:   # prune weak edges, form co-activation edges
```

#### `propagation.py` — PropagationEngine (78 lines)

Spreading activation with typed edge filtering:

```python
engine = PropagationEngine(graph)
engine.get_prediction(active_nids, top_k=5)  # traverse edges, filter by type
engine.get_activation_vector(nids)           # weighted average of vectors
engine.measure_coherence(active_nids)        # mean pairwise similarity
```

#### `tokenizer.py` — Tokenizers

- `WordTokenizer` — dynamic vocab, word-level (fastest for cognitive exps)
- `BPETokenizer` — tiktoken/GPT-2 (requires `tiktoken`)
- `SimpleTokenizer` — char-level fallback (256 vocab)
- `PixelTokenizer` — image → token sequence (28×28 → 784 tokens)

#### `currencies.py` / `currency.py` — Cognitive Currencies

Unified cognitive state management replacing scattered scalars.

#### `tensor.py` — Tensor Abstractions

- `RawTensor` — numpy wrapper with device API
- `StateTensor` — adds cognitive metadata (salience, free_energy, stability, decay)

#### `embedder.py` — LearnedEmbedder

Character n-gram + random projection (64-dim) for OOV handling.

---

## Layer 2: `ravana-v2/` — GRACE Cognitive Core

GRACE = **Governance, Reflection, Adaptation, Constraint, Exploration**

### Phase Architecture

| Phase | Module | Lines | Function |
|-------|--------|-------|----------|
| **A** | `governor.py` | 743 | Central regulation — hard constraints, predictive dampening, boundary pressure, center-seeking |
| **A** | `identity.py` | — | Momentum-based self-concept with recovery bias |
| **A** | `resolution.py` | — | Continuous partial credit toward wisdom events |
| **B** | `adaptation.py` | — | Policy learning from clamp events |
| **C** | `strategy.py` | — | 4 exploration modes: AGGRESSIVE, SAFE, STABILIZE, RECOVER |
| **C** | `strategy_learning.py` | — | Meta-learning over mode outcomes |
| **D** | `intent.py` | — | Dynamic objectives evolving from outcomes |
| **D.5** | `planning.py` | — | Micro-planner with simulated futures |
| **E** | `environment.py` | — | Non-stationary world (boundary shifts, noise drift, goal flips) |
| **F** | `predictive_world.py` | — | Neural world model, adaptive surprise threshold |
| **F.5** | `belief_reasoner.py` | — | Competing hypotheses with confidence decay |
| **G** | `active_epistemology.py` | — | Value of Information (VoI) calculation |
| **G.5** | `surgical_probes.py` | — | Targeted intervention experiments |
| **J** | `hypothesis_generation.py` | — | Generate hypotheses from anomalies |
| **J.1** | `occam_layer.py` | — | Hypothesis discipline / complexity penalization |
| **K** | `emotion.py` | 234 | VAD differential equations |
| **K.5** | `empathy.py` | — | Other-mind modeling |
| **L** | `sleep.py` | 703 | 4-stage SWS + REM with dream sabotage |
| **L.5** | `dual_process.py` | — | System 1 / System 2 routing |
| **M** | `meaning.py` | 224 | Intrinsic motivation: M = w1(-D) + w2(I) + w3(pred) × (1 + κ×effort) |
| **N** | `global_workspace.py` | — | Competitive broadcast, consciousness bottleneck |
| **O** | `human_memory.py` | 2,321 | Persistent episodic/semantic memory, Ebbinghaus decay, interference |
| **P** | `dialogue_context.py` | — | Conversation tracking, active subgraphs |
| **P** | `conversational_repair.py` | — | Correction handling, repair events |

### Governor — Central Regulation

```python
from core.governor import Governor, GovernorConfig, RegulationMode

config = GovernorConfig(
    max_dissonance=0.95,
    min_dissonance=0.15,
    target_dissonance=0.30,
    center_target=0.50,
    boundary_k=12.0,
    use_smoothed_dissonance=True,
    smoothing_alpha=0.2,
)

gov = Governor(config)

# Every state change MUST pass through governor
signals = CognitiveSignals(
    dissonance_delta=0.1,
    identity_delta=0.05,
    trend=0.02,              # dD/dt prediction
    predicted_dissonance=0.35,
)
output = gov.regulate(
    current_dissonance=0.3,
    current_identity=0.7,
    signals=signals,
    episode=100
)
# output.dissonance_delta (regulated)
# output.identity_delta (regulated)
# output.mode (RegulationMode.NORMAL/EXPLORATION/RESOLUTION/RECOVERY/PLATEAU)
```

**Four regulatory layers:**
1. **Hard constraints** — absolute ceilings/floors (non-negotiable)
2. **Predictive dampening** — look-ahead regulation (slow before hitting wall)
3. **Boundary pressure** — air resistance near limits
4. **Center-seeking force** — homeostatic pull to target

### Sleep Consolidation

```python
from core.sleep import SleepConsolidation, SleepConfig, SleepStage

config = SleepConfig(
    pressure_threshold=0.2,
    counterfactual_rate=0.20,    # 20% memory reversal
    emotional_flip_rate=0.10,    # 10% valence flip
    failure_oversample_factor=1.5,
)

sleep = SleepConsolidation(config)
sleep.accumulate_pressure(delta)

if sleep.should_sleep():
    record = sleep.execute_sleep_cycle(
        episode=100,
        state_snapshot=state.snapshot(),
        episodic_memories=memories,
        emotion_engine=emotion_engine,
        coherence_fn=lambda s: 1.0 - s["dissonance"],
        graph=concept_graph,
    )
```

**Four stages:**
1. **Topology Analysis** — identify high-pressure zones
2. **Pattern Compression** — strengthen consistent co-activation clusters
3. **Contradiction Resolution** — rewire weakest edges in pressure zones
4. **Integration** — merge, stabilize, rollback if coherence drops >5%

**Dream sabotage (REM):**
- Counterfactual reversal (20%)
- Emotional valence flip (10%)
- Failure oversampling (1.5×)

### Human Memory Engine

```python
from core.human_memory import HumanMemoryEngine, HumanMemoryConfig

memory = HumanMemoryEngine(config)

# Store
mem_id = memory.store(content="heat causes expansion", importance=0.8, tags="causal,physics")

# Recall
results = memory.recall(query="heat", limit=5, min_confidence=0.3)

# Sleep operations
memory.sleep_replay(state_snapshot)
memory.apply_decay()          # Ebbinghaus forgetting curve
memory.consolidate()          # episodic → semantic
memory.bridge_to_graph(graph) # semantic memories → ConceptGraph edges
```

---

## Layer 3: `ravana/` — Unified Package

Single pip-installable package re-exporting both codebases:

```bash
pip install -e ravana/   # numpy only
```

```python
import ravana as torch          # PyTorch-compatible API
from ravana.nn import RLM       # RLMv2 (triple decomposition)
from ravana.cognitive import CognitiveFramework  # Full cognitive system
from ravana.graph import ConceptGraph
from ravana.propagation import PropagationEngine
```

**Structure:**
```
ravana/
├── __init__.py          # import ravana as torch
├── nn/
│   └── __init__.py      # from ravana_ml.nn import RLMv2 as RLM
├── cognitive/
│   └── framework.py     # CognitiveFramework (wires L1 + L2)
├── graph/               # re-exports ravana_ml.graph
├── propagation/         # re-exports ravana_ml.propagation
├── world/               # simulation environments
├── lab/                 # analysis tools
└── pyproject.toml       # numpy>=1.20
```

---

## Data Flow & Integration

### Cognitive Cycle (Framework API)

```python
from ravana.cognitive import CognitiveFramework

fw = CognitiveFramework()
state = fw.initialize()

for episode, (input_vec, target_vec) in enumerate(dataset):
    # 1. PERCEIVE: input → active concepts
    concepts = fw.perceive(state, input_vec)
    
    # 2. PREDICT: spreading activation → predicted vector
    predictions = fw.predict(state, concepts)
    
    # 3. LEARN: pressure → governor → identity → emotion → meaning
    state = fw.learn(state, predictions, target_vec, episode)
    
    # 4. SLEEP (periodic): consolidation + memory replay
    if episode % 100 == 0:
        state = fw.sleep(state)

# Inference (no state change)
result = fw.infer(state, test_input_vec)
```

### RLMv2 Standalone Usage

```python
from ravana.nn import RLM
from ravana_ml.tokenizer import WordTokenizer
import numpy as np

tokenizer = WordTokenizer()
tokenizer.encode("heat causes expansion")

model = RLM(
    vocab_size=tokenizer.vocab_size,
    embed_dim=64,
    concept_dim=64,
    n_concepts=100,
    sleep_interval=50,
)

# Learn
input_ids = np.array(tokenizer.encode("heat causes"), dtype=np.int64)
target_ids = np.array(tokenizer.encode("expansion"), dtype=np.int64)
model.learn(input_ids, target_ids)

# Sleep consolidation
model.sleep_cycle()

# Generate
logits = model.forward(input_ids)
```

---

## Key Algorithms

### Spreading Activation (PropagationEngine)

```python
def spread_activation(graph, active_nids, steps=3, k_active=5, decay=0.3):
    for _ in range(steps):
        # 1. Collect activation from neighbors via edges
        for (src, tgt), edge in graph.edges.items():
            if src in active_nids and edge.edge_type == "excitatory":
                graph.nodes[tgt].activation += graph.nodes[src].activation * edge.weight * decay
            elif src in active_nids and edge.edge_type == "inhibitory":
                graph.nodes[tgt].activation -= graph.nodes[src].activation * edge.weight * decay
        
        # 2. Top-k competition (k_active most active survive)
        active_nids = top_k_active_nodes(graph, k_active)
        
        # 3. Decay all activations
        for nid in active_nids:
            graph.nodes[nid].activation *= (1 - decay)
```

### Hebbian Learning with Relation Vectors

```python
def hebbian_update(graph, source, target, coactivation, lr=0.01):
    edge = graph.get_edge(source, target)
    if edge:
        # Strengthen weight
        edge.weight = min(1.0, edge.weight + lr * coactivation)
        edge.confidence = min(1.0, edge.confidence + lr * 0.1)
        
        # Update relation vector toward (target_vec - source_vec)
        rel_vec = graph.nodes[target].vector - graph.nodes[source].vector
        edge.relation_vector = 0.9 * edge.relation_vector + 0.1 * rel_vec
        edge.relation_vector /= (np.linalg.norm(edge.relation_vector) + 1e-15)
```

### Verb-Stem Offset Predictor (RLMv2)

```python
def verb_offset_predict(query_concept, verb_token, graph):
    """subject_embed + offset(query_verb) ≈ target_embed"""
    # offset(verb) = avg(target - subject) over all training pairs with this verb
    offset = graph.global_relation_priors.get(verb_token)
    if offset is not None:
        predicted = query_concept + offset
        return graph.find_similar(predicted, k=10)
    return []
```

### VAD Emotion Dynamics

```python
# Differential equations (Euler integration, dt=1):
# dV/dt = -γ_v * V + β_v * (reward - punishment)
# dA/dt = -γ_a * A + β_a * |prediction_error|
# dD/dt = -γ_d * D + β_d * (control - helplessness)

# Modulation of inference:
if A > 0.7:  # high arousal → exploration
    exploration_bonus += 0.2
if V > 0.5:  # positive valence → trust predictions
    prediction_confidence *= 1.2
if D > 0.6:  # high dominance → stronger concepts
    concept_activation *= 1.15
```

---

## Configuration Reference

### FrameworkConfig (CognitiveFramework)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `concept_dim` | 64 | Concept vector dimensionality |
| `max_concepts` | 10000 | Maximum concept nodes |
| `k_active` | 5 | Top-k active concepts |
| `hebbian_lr` | 0.03 | Hebbian learning rate |
| `anti_hebbian_lr` | 0.02 | Anti-Hebbian learning rate |
| `propagation_steps` | 3 | Activation spread iterations |
| `propagation_decay` | 0.5 | Per-step decay |
| `initial_identity` | 0.5 | Initial identity strength |

### GovernorConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_dissonance` | 0.95 | Hard ceiling |
| `min_dissonance` | 0.15 | Hard floor |
| `target_dissonance` | 0.30 | Homeostatic target |
| `center_target` | 0.50 | Identity target |
| `boundary_k` | 12.0 | Boundary pressure steepness |
| `use_smoothed_dissonance` | True | EMA smoothing |
| `smoothing_alpha` | 0.2 | EMA alpha |

### SleepConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pressure_threshold` | 0.2 | Sleep trigger |
| `counterfactual_rate` | 0.20 | Dream: reversal rate |
| `emotional_flip_rate` | 0.10 | Dream: valence flip |
| `failure_oversample_factor` | 1.5 | Dream: failure replay |
| `coherence_drop_threshold` | 0.05 | Rollback threshold |

---

## See Also

- [Getting Started](GETTING_STARTED.md) — Quickstart tutorial
- [ML Framework](ML_FRAMEWORK.md) — ravana_ml deep dive
- [Cognitive Core](COGNITIVE_CORE.md) — ravana-v2 module reference
- [Unified Package](UNIFIED_PACKAGE.md) — ravana/ API
- [Core Concepts](CONCEPTS.md) — Theory & foundations
- [API Reference](API_REFERENCE.md) — Complete function/class reference
- [Experiments](EXPERIMENTS.md) — Running benchmarks
- [Advanced Topics](ADVANCED_TOPICS.md) — Customization & extension
- [Tutorials](TUTORIALS.md) — Step-by-step guides
- [Developer Guide](DEVELOPER_GUIDE.md) — Contributing