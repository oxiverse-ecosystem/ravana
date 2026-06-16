# RAVANA Unified Package (`ravana/`)

> **Single pip-installable package** re-exporting both `ravana_ml/` and `ravana-v2/` with a clean PyTorch-compatible API.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [API Structure](#api-structure)
4. [PyTorch Compatibility Layer](#pytorch-compatibility-layer)
5. [RLM — Recursive Learning Models](#rlm--recursive-learning-models)
6. [CognitiveFramework — Full Cognitive System](#cognitiveframework--full-cognitive-system)
7. [Graph & Propagation](#graph--propagation)
8. [World & Lab Utilities](#world--lab-utilities)
9. [Serialization](#serialization)
10. [Development](#development)

---

## Overview

```
ravana/                          # pip install -e ravana/
├── __init__.py                  # Main exports, import ravana as torch
├── nn/
│   └── __init__.py              # RLMv2 as RLM, functional
├── cognitive/
│   ├── __init__.py              # Exports CognitiveFramework
│   └── framework.py             # Top-level API wiring L1 + L2
├── graph/                       # Re-exports ravana_ml.graph
├── propagation/                 # Re-exports ravana_ml.propagation
├── world/
│   └── __init__.py              # Simulation environments
├── lab/
│   └── __init__.py              # Analysis tools
└── pyproject.toml               # numpy>=1.20
```

**Design goals:**
- Single import: `import ravana as torch`
- Zero-config: works out of the box with `numpy` only
- Full access: all L1 + L2 modules reachable
- PyTorch-like: familiar API for ML practitioners

---

## Installation

```bash
# From repository root
pip install -e ravana/

# Or install all layers from root
pip install -e .

# Verify
python -c "import ravana as torch; print(torch.__version__)"
# 0.1.0
```

**Dependencies:** `numpy>=1.20` (only hard dependency)

**Optional:** `pip install tiktoken` for BPE tokenizer

---

## API Structure

### Main Imports

```python
import ravana as torch

# Tensor operations (PyTorch-like)
from ravana import (
    tensor, zeros, ones, randn, eye, arange, stack, cat, from_numpy,
    RawTensor, StateTensor, Parameter, Tensor,
    device, cuda, cuda_is_available,
    is_tensor, no_grad, save, load,
)

# Submodules
from ravana import nn, cognitive, graph, propagation, world, lab

# Direct class imports
from ravana.nn import RLM, Linear, Embedding, LayerNorm, GRUCell, ConceptAttentionHead
from ravana.cognitive import CognitiveFramework, FrameworkConfig, FrameworkState
from ravana.graph import ConceptGraph, ConceptNode, ConceptEdge, ConceptBindingMap
from ravana.propagation import PropagationEngine
```

### Version

```python
import ravana
ravana.__version__  # '0.1.0'
```

---

## PyTorch Compatibility Layer

### Tensors

```python
import ravana as torch
import numpy as np

# Creation
x = torch.tensor([1., 2., 3.])           # StateTensor
x = torch.tensor(np.array([1,2,3]))      # from numpy
x = torch.zeros(3, 4)
x = torch.ones(3, 4)
x = torch.randn(3, 4)
x = torch.eye(3)
x = torch.arange(10)
x = torch.stack([t1, t2], dim=0)
x = torch.cat([t1, t2], dim=1)
x = torch.from_numpy(np.array([1,2,3]))

# Properties (same as PyTorch)
x.shape, x.ndim, x.dtype, x.device
x.data           # underlying np.ndarray
x.T              # transpose

# Operations
x + y, x - y, x * y, x / y
x @ y            # matmul
x.reshape(...), x.view(...)
x.squeeze(), x.unsqueeze(dim)
x.sum(dim), x.mean(dim)
x.max(dim), x.min(dim)
x.argmax(dim), x.argmin(dim)

# Cognitive metadata (RAVANA-specific)
x.salience       # float
x.free_energy    # float
x.stability      # float
x.decay          # float
```

### Device API

```python
torch.device        # Device('cpu')
torch.cuda          # Device('cuda')
torch.cuda_is_available  # False

# Usage
x = torch.tensor([1,2,3], device=torch.device)
model.to(torch.device)   # no-op (CPU only)
```

### Context Managers

```python
with torch.no_grad():
    # No gradient tracking (compatibility, no-op)
    y = model(x)
```

### Save/Load

```python
# Generic save (auto-detects model type)
torch.save(model, "checkpoint.pkl")      # pickle
torch.save(model, "checkpoint.zip")      # human-readable zip

# Generic load
model = torch.load("checkpoint.pkl")
model = torch.load("checkpoint.zip")

# Load state dict into existing model
torch.load(model, "checkpoint.pkl")
```

---

## RLM — Recursive Learning Models

### RLMv2 (Default, Triple Decomposition)

```python
from ravana.nn import RLM
from ravana_ml.tokenizer import WordTokenizer
import numpy as np

tok = WordTokenizer()
tok.encode("heat causes expansion")

model = RLM(
    vocab_size=tok.vocab_size,
    embed_dim=64,
    concept_dim=64,
    n_concepts=100,
    max_seq_len=128,
    sleep_interval=50,
    gate_concept_creation=True,
    anchor_relation_vectors=True,
    latent_dim=96,
    hidden_dim=128,
)

# Learn
inp = np.array(tok.encode("heat causes"), dtype=np.int64)
tgt = np.array(tok.encode("expansion"), dtype=np.int64)
model.learn(inp, tgt)

# Sleep
model.sleep_cycle()

# Forward
logits = model.forward(inp)          # (vocab_size,)
probs = torch.softmax(logits, dim=-1)

# Top-k predictions
top_k = model.get_top_predictions(inp, k=5)
```

#### RLMv2 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | required | Token vocabulary size |
| `embed_dim` | 64 | Token embedding dim |
| `concept_dim` | 64 | Concept vector dim |
| `n_concepts` | 100 | Initial concept nodes |
| `max_seq_len` | 128 | Positional encoding length |
| `sleep_interval` | 100 | Learn steps between sleep |
| `gate_concept_creation` | True | Similarity threshold for new concepts |
| `anchor_relation_vectors` | True | Deterministic relation vector seeds |
| `latent_dim` | 96 | World model latent dim |
| `hidden_dim` | 128 | World model hidden dim |

**Note:** For the entity-specific adapter to work correctly (enables test-time adaptation for held-out subjects), `latent_dim` should equal `embed_dim`. If they differ, the model automatically projects embeddings to latent space before applying the adapter, but this adds computational overhead. For optimal performance and simplicity, set `latent_dim=embed_dim` (e.g., both 64).

#### Key Attributes (Post-Training)

```python
model.graph                    # ConceptGraph
model.sleep_cycles_completed   # int
model.total_free_energy        # float
model.conceptual_accuracy      # float
model._train_correct, model._train_total  # counters
```

### RLM v1 (Predictive Coding, GRU)

```python
from ravana_ml.nn import RLM as RLMv1

model = RLMv1(
    vocab_size=1000,
    embed_dim=64,
    concept_dim=64,
    n_concepts=200,
    n_hidden=128,
    n_layers=3,
    max_seq_len=128,
    free_energy_threshold=8.0,
    sleep_interval=100,
)

model.learn(input_ids, target_ids)
model.sleep_cycle()
logits = model.forward(input_ids)
```

---

## CognitiveFramework — Full Cognitive System

```python
from ravana.cognitive import CognitiveFramework, FrameworkConfig, FrameworkState
import numpy as np

# Configure (all optional, have sensible defaults)
config = FrameworkConfig(
    concept_dim=64,
    max_concepts=10000,
    k_active=5,
    governor_config=None,        # GovernorConfig()
    emotion_config=None,         # VADConfig()
    sleep_config=None,           # SleepConfig()
    meaning_config=None,         # MeaningConfig()
    dual_process_config=None,    # DualProcessConfig()
    gw_config=None,              # GWConfig()
    human_memory_config=None,    # HumanMemoryConfig()
    hebbian_lr=0.03,
    anti_hebbian_lr=0.02,
    propagation_steps=3,
    propagation_decay=0.5,
    initial_identity=0.5,
)

fw = CognitiveFramework(config)
state = fw.initialize()

# Training loop
for episode, (input_vec, target_vec) in enumerate(dataset):
    # 1. PERCEIVE
    concepts = fw.perceive(state, input_vec)
    
    # 2. PREDICT
    predictions = fw.predict(state, concepts)
    
    # 3. LEARN
    state = fw.learn(
        state, predictions, target_vec, episode,
        difficulty=0.5,
        effort=0.3,
    )
    
    # 4. SLEEP (periodic)
    if episode % 100 == 0:
        state = fw.sleep(state)

# Inference (no state change)
result = fw.infer(state, test_input_vec)
# {
#   "concepts": [...],
#   "predicted_concepts": [...],
#   "predictions": np.ndarray,
#   "confidences": [...],
#   "coherence": float,
#   "dissonance": float,
#   "recalled_memories": [...],
# }

# Semantic query
query = fw.query(state, concept_id=42)
# {
#   "concept": {...},
#   "neighbors": [...],
#   "edges": [...],
# }

# Save/Load
fw.save("checkpoint.pkl")
fw = CognitiveFramework.load("checkpoint.pkl")
fw.rebridge()  # sync consolidated memories → graph edges
```

### FrameworkConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `concept_dim` | 64 | Concept vector dimensionality |
| `max_concepts` | 10000 | Maximum concept nodes |
| `k_active` | 5 | Top-k active concepts |
| `hebbian_lr` | 0.03 | Hebbian learning rate |
| `anti_hebbian_lr` | 0.02 | Anti-Hebbian learning rate |
| `propagation_steps` | 3 | Activation spread iterations |
| `propagation_decay` | 0.5 | Per-step decay |
| `initial_identity` | 0.5 | Starting identity strength |
| `governor_config` | None | GovernorConfig() if None |
| `emotion_config` | None | VADConfig() if None |
| `sleep_config` | None | SleepConfig() if None |
| `meaning_config` | None | MeaningConfig() if None |
| `dual_process_config` | None | DualProcessConfig() if None |
| `gw_config` | None | GWConfig() if None |
| `human_memory_config` | None | HumanMemoryConfig() if None |

### FrameworkState (Immutable Snapshot)

```python
state.dissonance           # float
state.identity             # float
state.wisdom               # float
state.meaning              # float
state.cycle                # int
state.episode              # int
state.vad                  # VADState or None
state.emotional_label      # str
state.total_free_energy    # float
state.mode                 # str (regulation mode)
state.processing_route     # str (system1_fast / system2_slow)
state.sleep_cycles         # int
state.gw_broadcast_source  # str or None

# Serializable
snapshot = state.snapshot()
```

---

## Graph & Propagation

```python
from ravana.graph import ConceptGraph, ConceptNode, ConceptEdge, ConceptBindingMap
from ravana.propagation import PropagationEngine

# ConceptGraph
graph = ConceptGraph(dim=64, max_nodes=10000)

# Nodes
nid = graph.add_node(vector, label="heat")
node = graph.get_node(nid)
# node.vector, node.core_vector, node.activation, node.salience, ...

# Edges
graph.add_edge(source, target, weight=0.5, relation_type="causal")
edge = graph.get_edge(source, target)
# edge.weight, edge.confidence, edge.relation_type, edge.relation_vector, ...

# Activation
graph.activate(nid, 1.0)
graph.spread_activation(steps=3, k_active=5, decay=0.3)
graph.reset_activation()

# Similarity
similar = graph.find_similar(vector, k=10)  # [(nid, sim), ...]

# Learning
graph.hebbian_update(source, target, coactivation, lr=0.03)

# Structural
pruned, formed = graph.structural_step()
graph.homeostatic_downscale(protection_threshold=0.8)
graph.reconcile_contradictions()
graph.form_inhibitory_edges()

# Input binding
active_nids = graph.bind_input(input_vector, k=5)

# Propagation Engine
engine = PropagationEngine(graph)
predicted = engine.get_prediction(active_nids, top_k=5)
act_vec = engine.get_activation_vector(predicted)
coherence = engine.measure_coherence(active_nids)
```

---

## World & Lab Utilities

### World (Simulation Environments)

```python
from ravana.world import GridWorld, ContinuousWorld, SymbolicWorld

# GridWorld — discrete 2D environment
env = GridWorld(width=10, height=10, n_agents=1)
obs = env.reset()
obs = env.step(action)  # action: 0=up, 1=down, 2=left, 3=right

# ContinuousWorld — continuous control
env = ContinuousWorld(dim=2, action_dim=2)
obs = env.reset()
obs = env.step(action)  # action: np.ndarray(2,)

# SymbolicWorld — discrete state, symbolic actions
env = SymbolicWorld(states=["A", "B", "C"], actions=["move", "wait"])
obs = env.step("move")
```

### Lab (Analysis Tools)

```python
from ravana.lab import (
    analyze_concept_graph,
    plot_activation_dynamics,
    compute_coherence_trajectory,
    visualize_sleep_cycle,
    diagnose_learning,
)

# Graph analysis
stats = analyze_concept_graph(graph)
# {"n_nodes": ..., "n_edges": ..., "avg_degree": ..., "clustering": ..., "modularity": ...}

# Activation dynamics
plot_activation_dynamics(graph, save_path="activation.png")

# Coherence trajectory
trajectory = compute_coherence_trajectory(fw, episodes=1000)

# Sleep diagnostics
visualize_sleep_cycle(sleep_engine, save_path="sleep.png")

# Learning diagnosis
report = diagnose_learning(model, test_data)
```

---

## Serialization

### Unified Save/Load

```python
import ravana as torch

# Any model
torch.save(model, "model.pkl")
torch.save(model, "model.zip")  # human-readable

model = torch.load("model.pkl")
model = torch.load("model.zip")

# State dict into existing model
torch.load(existing_model, "model.pkl")
```

### RLM-Specific

```python
# RLMv2 / RLMv1
model.save("rlm.pkl")
model = RLM.load("rlm.pkl")

model.save_zip("rlm.zip")
model = RLM.load_zip("rlm.zip")
```

### CognitiveFramework

```python
fw.save("cognitive.pkl")
fw = CognitiveFramework.load("cognitive.pkl")
fw.rebridge()  # critical: sync memories → graph
```

### ConceptGraph

```python
import pickle

# Save
with open("graph.pkl", "wb") as f:
    pickle.dump(graph, f)

# Load
with open("graph.pkl", "rb") as f:
    graph = pickle.load(f)
```

---

## Development

### Project Structure

```
ravana/
├── __init__.py              # Main package exports
├── nn/
│   └── __init__.py          # from ravana_ml.nn import RLMv2 as RLM
├── cognitive/
│   ├── __init__.py          # from .framework import CognitiveFramework
│   └── framework.py         # CognitiveFramework implementation
├── graph/                   # symlink or re-export to ravana_ml.graph
├── propagation/             # symlink or re-export to ravana_ml.propagation
├── world/
│   └── __init__.py          # environments
├── lab/
│   └── __init__.py          # analysis tools
└── pyproject.toml           # [project] dependencies: numpy>=1.20
```

### Adding New Exports

Edit `ravana/__init__.py`:

```python
from ravana_ml import NewClass, new_function

__all__ = [
    # ... existing
    'NewClass', 'new_function',
]
```

Edit `ravana/nn/__init__.py` for neural modules:

```python
from ravana_ml.nn import NewModule

__all__ = ['NewModule', 'RLM', 'Linear', ...]
```

### Running Tests

```bash
# ML framework tests
python -m pytest tests/ -v

# Cognitive core tests  
python -m pytest ravana-v2/tests/ -v

# Unified package tests
python -c "import ravana as torch; print('Import OK')"
python -c "from ravana.cognitive import CognitiveFramework; fw = CognitiveFramework(); print('Framework OK')"
```

### Building Distribution

```bash
pip install build
python -m build ravana/
# dist/ravana-0.1.0.tar.gz
# dist/ravana-0.1.0-py3-none-any.whl
```

---

## See Also

- [Architecture Overview](ARCHITECTURE.md)
- [ML Framework](ML_FRAMEWORK.md)
- [Cognitive Core](COGNITIVE_CORE.md)
- [Getting Started](GETTING_STARTED.md)
- [API Reference](API_REFERENCE.md)