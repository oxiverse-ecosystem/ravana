# RAVANA ML Framework (`ravana_ml/`)

> **Complete reference for the ML framework layer** — PyTorch-compatible API, NumPy-only, pressure-driven learning.

---

## Table of Contents

1. [Overview](#overview)
2. [Tensor Abstractions](#tensor-abstractions)
3. [Neural Modules (`nn/`)](#neural-modules-nn)
4. [ConceptGraph](#conceptgraph)
5. [Free Energy & Plasticity](#free-energy--plasticity)
6. [Propagation Engine](#propagation-engine)
7. [Tokenizers](#tokenizers)
8. [Cognitive Currencies](#cognitive-currencies)
9. [RLM v1 — Recursive Learning Model](#rlm-v1--recursive-learning-model)
10. [RLM v2 — Triple Decomposition](#rlm-v2--triple-decomposition)
11. [Serialization](#serialization)

---

## Overview

`ravana_ml/` is a **standalone ML framework** with:

- **Single dependency**: `numpy>=1.20`
- **PyTorch-compatible API**: `import ravana as torch`
- **No autograd/backprop**: learning via free energy + Hebbian plasticity
- **CPU-native**: no GPU required

```
ravana_ml/
├── __init__.py           # Main exports, device API, save/load
├── tensor.py             # RawTensor, StateTensor, Parameter
├── nn/
│   ├── __init__.py       # Module, Linear, Embedding, LayerNorm, GRUCell, RLM
│   ├── module.py         # Base classes
│   ├── functional.py     # relu, softmax, cross_entropy, etc.
│   ├── rlm.py            # RLM v1 (predictive coding, GRU)
│   └── rlm_v2.py         # RLM v2 (triple decomposition, GloVe)
├── graph.py              # ConceptGraph (3,678 lines)
├── propagation.py        # PropagationEngine
├── free_energy.py        # FreeEnergyAccumulator (5 channels)
├── plasticity.py         # Hebbian, AntiHebbian, Structural
├── tokenizer.py          # Word, BPE, Simple, Pixel tokenizers
├── currencies.py         # CognitiveCurrencies (unified state)
├── currency.py           # create_rlm_currency()
├── embedder.py           # LearnedEmbedder (char n-gram)
├── episode_injector.py   # Knowledge injection
├── relation_ontology.py  # Relation type hierarchy
└── lab/, world/          # Analysis tools, environments
```

---

## Tensor Abstractions

### `RawTensor` — NumPy Wrapper

```python
from ravana_ml.tensor import RawTensor, tensor, zeros, ones, randn, eye, arange, stack, cat, from_numpy
import numpy as np

# Creation (PyTorch-like)
x = tensor([1, 2, 3])                    # 1D
x = tensor([[1, 2], [3, 4]])             # 2D
x = zeros(3, 4)                          # zeros
x = ones(3, 4)                           # ones
x = randn(3, 4)                          # normal(0,1)
x = eye(3)                               # identity
x = arange(10)                           # [0..9]
x = stack([t1, t2], dim=0)               # stack
x = cat([t1, t2], dim=1)                 # concatenate
x = from_numpy(np.array([1,2,3]))        # from numpy

# Properties
x.shape          # tuple
x.ndim           # int
x.dtype          # numpy dtype
x.device         # Device('cpu')
x.data           # underlying np.ndarray

# Operations
x + y, x - y, x * y, x / y
x @ y             # matmul
x.T               # transpose
x.reshape(...)
x.squeeze(), x.unsqueeze(dim)
x.sum(dim), x.mean(dim)
x.max(dim), x.min(dim)
x.argmax(dim), x.argmin(dim)
```

### `StateTensor` — Cognitive Tensor

Extends `RawTensor` with cognitive metadata:

```python
from ravana_ml.tensor import StateTensor, Parameter

# StateTensor adds:
st = StateTensor(data=np.array([1., 2., 3.]))
st.salience       # float: importance weight (default 1.0)
st.free_energy    # float: accumulated prediction error
st.stability      # float: resistance to change (0=fragile, 1=stable)
st.decay          # float: time-based decay rate

# Parameter — for learnable weights
param = Parameter(data=np.randn(10, 5))
param.grad        # None (no autograd!)
# Learning via: param.data += delta  (manual Hebbian updates)
```

### Device API (Compatibility)

```python
from ravana_ml import device, cuda, cuda_is_available

device        # Device('cpu')
cuda          # Device('cuda')
cuda_is_available  # False (CPU-only)

# Usage
x = tensor([1,2,3], device=device)
model.to(device)    # no-op (CPU only)
```

---

## Neural Modules (`nn/`)

### Base Classes

```python
from ravana_ml.nn.module import Module, Sequential, Parameter

class Module:
    def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
    def forward(self, *args, **kwargs): raise NotImplementedError
    def parameters(self) -> List[Parameter]: ...
    def named_parameters(self) -> List[Tuple[str, Parameter]]: ...
    def state_dict(self) -> Dict[str, np.ndarray]: ...
    def load_state_dict(self, sd: Dict[str, np.ndarray]): ...
    def register_module(self, name: str, module: 'Module'): ...
    def modules(self): ...
    def train(self, mode: bool = True): ...
    def eval(self): ...

class Sequential(Module):
    def __init__(self, *modules): ...
```

### Linear

```python
from ravana_ml.nn import Linear

# No bias by default (matches cognitive architecture)
layer = Linear(in_features=64, out_features=32, bias=False)

# Forward
x = torch.randn(5, 64)      # (batch, in)
y = layer(x)                # (batch, out)

# Weight access
layer.weight.data           # (out, in) StateTensor
layer.weight.grad           # Always None — NO backprop!

# Manual update (Hebbian style)
layer.weight.data += lr * (pre.T @ post)  # outer product
```

### Embedding

```python
from ravana_ml.nn import Embedding

emb = Embedding(num_embeddings=1000, embedding_dim=64)
ids = torch.tensor([1, 5, 10, 42])        # (seq_len,)
vecs = emb(ids)                           # (seq_len, 64)

# Access
emb.weight.data      # (vocab_size, embed_dim) StateTensor
```

### LayerNorm

```python
from ravana_ml.nn import LayerNorm

ln = LayerNorm(normalized_shape=64, eps=1e-5)
x = torch.randn(32, 64)
y = ln(x)
```

### GRUCell

```python
from ravana_ml.nn import GRUCell

gru = GRUCell(input_size=64, hidden_size=128)
h = torch.zeros(1, 128)
x = torch.randn(1, 64)
h_next = gru(x, h)
```

### ConceptAttentionHead

```python
from ravana_ml.nn import ConceptAttentionHead

# Multi-head attention over concept vectors
attn = ConceptAttentionHead(concept_dim=64, vocab_size=1000, n_heads=2)
concept_vecs = torch.randn(10, 64)    # 10 active concepts
logits = attn(concept_vecs)           # (vocab_size,) predictions
```

---

## ConceptGraph

### Node: `ConceptNode`

```python
from ravana_ml.graph import ConceptNode

node = graph.get_node(nid)

# Core vectors
node.vector             # active vector (plastic, fast)
node.core_vector        # identity anchor (stable, slow)
node.genesis_vector     # original vector (for drift tracking)

# State
node.activation         # current spreading activation
node.salience           # importance (0-1)
node.prediction_free_energy  # local prediction error
node.stability          # resistance to change
node.confidence         # reliability of this concept
node.contradiction_count # how often contradicted
node.fatigue            # usage-based fatigue

# Hierarchy
node.level              # 0=leaf, >0=abstract
node.parent             # parent concept ID
node.children           # set of child concept IDs
node.abstraction_degree # 0=raw, 1=fully compressed

# Temporal
node.last_activated     # timestamp
node.activation_history # list of timestamps (max 100)
node.temporal_context   # context vector at last activation

# Properties
node.effective_activation  # activation * (1 - fatigue)
node.drift_magnitude       # ||vector - genesis_vector||
node.age()                 # time since creation
node.recency_score()       # exp decay from last activation
node.frequency_score(window) # fraction active in window
node.plasticity            # 1 - stability
```

### Edge: `ConceptEdge`

```python
from ravana_ml.graph import ConceptEdge

edge = graph.get_edge(source, target)

# Core
edge.source, edge.target
edge.weight             # [0,1], clamped
edge.confidence         # [0,1]
edge.stability          # resistance to change
edge.prediction_free_energy
edge.prediction_count   # total predictions
edge.forward_pred_count # A→B successes
edge.backward_pred_count # B→A successes

# Type system
edge.edge_type          # "excitatory" | "inhibitory"
edge.relation_type      # "semantic" | "causal" | "temporal" | "analogical" | "contextual" | "inferred"
edge.shortcut           # context→target edges (exempt from competition)

# Relational
edge.relation_vector    # learned relational embedding (dim=16)
edge.predicate_token_id # verb token ID that created this edge (-1 if none)

# Consolidation
edge.fisher_importance  # EWC importance for old tasks
edge.old_weight         # weight snapshot at domain boundary

# Bayesian posterior (Beta distribution)
edge.posterior_alpha    # 1 + successes
edge.posterior_beta     # 1 + failures
edge.posterior_mean     # alpha / (alpha + beta)
edge.posterior_uncertainty # variance

# Multi-agent
edge.agent_weights      # {agent_id: weight}
edge.source_metadata    # {epistemic_status, is_user_statement, ...}

# Properties
edge.effective_weight   # weight (negative if inhibitory)
edge.plasticity         # 1 - stability
```

### Binding: `ConceptBinding` / `ConceptBindingMap`

```python
from ravana_ml.graph import ConceptBinding, ConceptBindingMap

# Binding: token ↔ concept with confidence
binding = ConceptBinding(token_id=42, concept_id=7, confidence=0.8, source="learned")
binding.reinforce(0.05)
binding.decay(0.01)
binding.strength        # confidence * (1 - decay/10)

# Map: manages all bindings
bmap = ConceptBindingMap()
bmap.bind(token_id=42, concept_id=7, confidence=0.8)
bmap.get_concepts(token_id=42, min_confidence=0.1)
bmap.get_tokens(concept_id=7, min_confidence=0.1)
bmap.best_concept(token_id=42)
bmap.best_token(concept_id=7)
bmap.is_ambiguous(token_id=42, threshold=0.3)
bmap.ambiguity_score(token_id=42)  # entropy-based 0-1
```

### Graph Operations

```python
from ravana_ml.graph import ConceptGraph

graph = ConceptGraph(dim=64, max_nodes=10000)

# Nodes
nid = graph.add_node(vector, label="heat")
node = graph.get_node(nid)
graph.remove_node(nid)

# Edges
graph.add_edge(source, target, weight=0.5, relation_type="causal")
edge = graph.get_edge(source, target)
graph.remove_edge(source, target)

# Activation
graph.activate(nid, amount=1.0)
graph.reset_activation()
graph.spread_activation(steps=3, k_active=5, decay=0.3)

# Similarity
similar = graph.find_similar(vector, k=10)  # [(nid, cosine_sim), ...]

# Learning
graph.hebbian_update(source, target, coactivation, lr=0.03)
graph.anti_hebbian_update(source, target, coactivation, lr=0.02)

# Structural plasticity
pruned, formed = graph.structural_step()

# Homeostasis
w_before, w_after = graph.homeostatic_downscale(
    protection_threshold=0.8, downscale_factor=0.8
)

# Contradiction resolution
reconciled = graph.reconcile_contradictions()

# Inhibitory edges
inhibitory_formed = graph.form_inhibitory_edges()

# Hierarchical abstraction
graph.create_abstraction_cluster(child_ids, parent_label)

# Input binding
active_nids = graph.bind_input(input_vector, k=5)

# Stats
len(graph.nodes)
len(graph.edges)
graph.version          # increments on structural changes
```

---

## Free Energy & Plasticity

### FreeEnergyAccumulator (5 Channels)

```python
from ravana_ml.free_energy import FreeEnergyAccumulator

fe = FreeEnergyAccumulator(graph)

# Accumulate per channel
fe.accumulate_semantic(error=0.5, salience=1.0)
fe.accumulate_linguistic(error=0.3, salience=0.8)
fe.accumulate_episodic(error=0.7, salience=1.0)
fe.accumulate_contradiction(error=0.9, salience=1.0)
fe.accumulate_abstraction(error=0.2, salience=0.5)

# Total free energy
total = fe.total()

# Decay over time
fe.decay(rate=0.1)

# Per-node free energy (for targeted sleep)
fe.get_node_free_energy(nid)
```

### Plasticity Rules

```python
from ravana_ml.plasticity import HebbianPlasticity, AntiHebbianPlasticity, StructuralPlasticity

hebbian = HebbianPlasticity(graph, lr=0.03)
anti_hebbian = AntiHebbianPlasticity(graph, lr=0.02)
structural = StructuralPlasticity(graph, prune_threshold=0.005, form_threshold=0.3)

# Applied automatically during learning/sleep
hebbian.update(source, target, coactivation)
anti_hebbian.update(source, target, coactivation)
pruned, formed = structural.step()
```

---

## Propagation Engine

```python
from ravana_ml.propagation import PropagationEngine

engine = PropagationEngine(graph)

# Predict from active concepts
predicted_nids = engine.get_prediction(active_nids, top_k=5)

# Activation vector (weighted average)
act_vec = engine.get_activation_vector(predicted_nids)

# Coherence measure
coherence = engine.measure_coherence(active_nids)
```

---

## Tokenizers

```python
from ravana_ml.tokenizer import WordTokenizer, BPETokenizer, SimpleTokenizer, PixelTokenizer, get_tokenizer

# WordTokenizer — dynamic vocab, fastest for cognitive experiments
tok = WordTokenizer()
ids = tok.encode("heat causes expansion")  # builds vocab on the fly
text = tok.decode(ids)
tok.vocab_size

# BPETokenizer — tiktoken/GPT-2 (requires tiktoken)
tok = BPETokenizer("gpt2")
ids = tok.encode("heat causes expansion")

# SimpleTokenizer — char-level fallback (256 vocab)
tok = SimpleTokenizer()

# PixelTokenizer — image data
tok = PixelTokenizer()
img_tokens = tok.encode_image(np.random.rand(28, 28))
label_token = tok.encode_label(3)

# Factory
tok = get_tokenizer("word")      # default
tok = get_tokenizer("bpe")       # GPT-2
tok = get_tokenizer("simple")    # char-level
```

---

## Cognitive Currencies

```python
from ravana_ml.currencies import CognitiveCurrencies
from ravana_ml.currency import create_rlm_currency

# Unified cognitive state
currencies = CognitiveCurrencies()

# Scalars (all have history tracking)
currencies.identity_strength
currencies.identity_momentum
currencies.identity_history
currencies.valence, currencies.arousal, currencies.dominance  # VAD
currencies.accumulated_meaning
currencies.meaning_history
currencies.sleep_pressure
currencies.sleep_pressure_threshold
currencies.regulation_mode
currencies.dissonance_ema

# RLM-compatible currency
currency = create_rlm_currency()
currency.identity_strength = 0.7
currency.valence = 0.3
# ... mirrors RLM scalar properties
```

---

## RLM v1 — Recursive Learning Model

```python
from ravana_ml.nn import RLM

model = RLM(
    vocab_size=1000,
    embed_dim=64,
    concept_dim=64,
    n_concepts=200,
    n_hidden=128,
    n_layers=3,
    max_seq_len=128,
    free_energy_threshold=8.0,
    sleep_interval=100,
    replay_buffer_max=500,
    replay_n_samples=20,
    anchor_relation_vectors=True,
    gate_concept_creation=True,
    adaptive_downscale=True,
    deep_sleep_every=1,
)

# Learn
input_ids = np.array([1, 5, 10], dtype=np.int64)  # "heat causes"
target_ids = np.array([42], dtype=np.int64)       # "expansion"
model.learn(input_ids, target_ids)

# Sleep consolidation
model.sleep_cycle()

# Forward
logits = model.forward(input_ids)           # (seq_len, vocab_size)
```

### Key Components

| Component | Description |
|-----------|-------------|
| `token_embed` | Embedding(vocab, embed_dim) |
| `recurrent_cell` | GRUCell(embed_dim, n_hidden) |
| `hidden_layers` | n_layers × Linear + LayerNorm |
| `concept_predictor` | Linear(n_hidden, concept_dim) |
| `context_logits` | Linear(n_hidden, vocab_size) |
| `concept_attn_head` | ConceptAttentionHead |
| `graph` | ConceptGraph |
| `propagation` | PropagationEngine |
| `free_energy_engine` | FreeEnergyAccumulator |

### 5-Path Logit Blend

```python
# Final logits = weighted sum of 5 paths:
# 1. Concept attention (concept → vocab)
# 2. Context logits (GRU hidden → vocab)
# 3. RP analogy (source + relation prior → vocab)
# 4. Sparse concept predictor (latent → bottleneck → vocab)
# 5. Direct latent → vocab (domain-specific heads)
```

### Sleep Cycle

```python
model.sleep_cycle()

# Does:
# 1. Graph reconciliation
# 2. Structural plasticity (prune/form)
# 3. Inhibitory edge formation
# 4. Homeostatic downscale
# 5. Interleaved replay (domain-tagged)
# 6. Contrastive hidden state learning
# 7. BP-trained relation predictor update
# 8. Free energy decay
# 9. Memory bridge (episodic → semantic → graph)
```

---

## RLM v2 — Triple Decomposition

```python
from ravana_ml.nn import RLMv2 as RLM

model = RLM(
    vocab_size=tokenizer.vocab_size,
    embed_dim=64,
    concept_dim=64,
    n_concepts=100,
    max_seq_len=128,
    sleep_interval=50,
    gate_concept_creation=True,
    anchor_relation_vectors=True,
)

# Learn triple
input_ids = np.array(tok.encode("heat causes"), dtype=np.int64)
target_ids = np.array(tok.encode("expansion"), dtype=np.int64)
model.learn(input_ids, target_ids)

# Generation
logits = model.forward(input_ids)
```

### Architecture

```
Input:  "heat causes expansion"
         ↓
Decompose: (subject="heat", relation="causes", object="expansion")
         ↓
Classify:  "causes" → CAUSAL type embedding
         ↓
Spread:    subject_node → (filter by CAUSAL) → target nodes
         ↓
Score:     activated nodes vs all token embeddings
         ↓
Logits:    vocab_size logits
```

### GloVe Embeddings

```python
# Automatically loads on first use:
# data/glove/glove.6B.100d.txt → projected to embed_dim via QR
# Cache: data/glove/projected_{vocab_size}_{embed_dim}.npy

# Download if missing:
# wget -O data/glove/glove.6B.100d.txt http://nlp.stanford.edu/data/glove.6B.100d.txt
```

### Verb-Stem Offset Predictor

```python
# For cross-domain generalization:
# offset(verb) = avg(target - subject) over all training pairs with this verb
# predicted = subject_embed + offset(query_verb)

model.use_rp_for_analogy = True  # enable
```

---

## Serialization

### Save/Load (Unified API)

```python
import ravana as torch

# Save model (auto-detects format)
torch.save(model, "checkpoint.pkl")      # pickle
torch.save(model, "checkpoint.zip")      # human-readable zip

# Load
model = torch.load("checkpoint.pkl")
model = torch.load("checkpoint.zip")

# Load state dict into existing model
torch.load(model, "checkpoint.pkl")
```

### RLM-Specific

```python
# RLM has custom save/load
model.save("rlm_checkpoint.pkl")
model = RLM.load("rlm_checkpoint.pkl")

model.save_zip("rlm_checkpoint.zip")
model = RLM.load_zip("rlm_checkpoint.zip")
```

### CognitiveFramework

```python
fw.save("cognitive_checkpoint.pkl")
fw = CognitiveFramework.load("cognitive_checkpoint.pkl")
fw.rebridge()  # sync consolidated memories → graph edges
```

---

## Performance Tips

1. **Use WordTokenizer** — 5× faster than char-level for cognitive experiments
2. **Reduce dimensions** for quick experiments: `embed_dim=32, concept_dim=32, n_concepts=50`
3. **Disable sleep** during prototyping: `sleep_interval=999999`
4. **Pre-allocate graphs** — `max_nodes=vocab_size * 2` avoids resizing
5. **Cache norms** — RLMv2 caches `token_embed_norms` per forward pass

---

## See Also

- [Architecture Overview](ARCHITECTURE.md)
- [Cognitive Core](COGNITIVE_CORE.md)
- [Unified Package](UNIFIED_PACKAGE.md)
- [Core Concepts](CONCEPTS.md)
- [API Reference](API_REFERENCE.md)