# Tutorial 03: Concept Graph Operations

**Part of a 7-tutorial progression. Standalone — no dependency on 01-02.**

## What you'll learn

- Build a `ConceptGraph` directly (standalone — no chat engine needed)
- Understand the **graph as the primary knowledge store** (not a weight matrix)
- Add nodes with embedding vectors and typed edges
- Walk outgoing edges and inspect edge attributes
- Understand **spreading activation** — the sole inference mechanism

---

## Run it

```bash
python tutorials/03-graph/run.py
```

---

## Deep dive: ConceptGraph architecture

**File:** `ravana_ml/src/ravana_ml/graph.py` (3,678 lines — the largest file in `ravana_ml`)

### What it is

The `ConceptGraph` is a **heterogeneous, typed, weighted graph** that stores all
knowledge. It is NOT a weight matrix, NOT a vector database, and NOT a
key-value store. It's a **cognitive graph** inspired by how the brain's semantic
network is organized (Collins & Loftus 1975; Binder & Desai 2011; Patterson,
Nestor & Rogers 2007).

### Node structure

```python
@dataclass
class ConceptNode:
    id: int                # unique numeric ID
    label: Optional[str]   # human-readable name (e.g., "trust", "courage")
    vector: np.ndarray     # 64-D GloVe embedding (the concept's "meaning")
    node_type: ConceptNodeType  # active / core / genesis
    stability: float       # 0-1, resistance to modification (higher = more stable)
    activation: float      # current spread activation level (0-1)
    confidence: float      # belief certainty (0-1)
```

The **vector** is the concept's semantic content — its position in a 64-D
semantic space. Two concepts with similar vectors have similar meanings.
This is the **distributional semantics** hypothesis: meaning is defined by
position in semantic space.

The **stability** determines how easily a concept can be modified. Core
concepts (like "i", "think", "know") have high stability. Newly learned
concepts have low stability (easily overwritten).

The **activation** is transient — it's set during graph walking and cleared
after each response. It's the computational analog of neural firing rate.

### Edge structure

```python
@dataclass
class ConceptEdge:
    weight: float              # connection strength (0-1)
    relation_type: str         # causal / semantic / temporal / contrastive / analogical
    confidence: float          # how certain we are this edge is correct
    prediction_free_energy: float  # prediction error on this edge
    prediction_count: int      # how many times this edge has been evaluated
    source: str                # seed / web / user / sleep / inference
```

**Weight vs confidence:**
- **Weight** (0-1): How strong the connection is. Higher = more influence during spread.
  A weight of 0.7 means 70% of the source node's activation passes to the target.
- **Confidence** (0-1): How certain the system is that this edge is correct.
  A seed edge (from the corpus) has high confidence. A web-scraped edge has low
  confidence until verified by multiple sources.

**Prediction free-energy** is the engine's learning signal. When the engine uses
an edge to predict a relationship and gets it wrong, the prediction error is
stored on the edge. During sleep, high-PE edges are strengthened or pruned.

### Relation types

| Type | Vector prototype | Semantic | Example |
|------|-----------------|----------|---------|
| `causal` | cause, lead, trigger, produce | A causes B | heat → expansion |
| `semantic` | relate, connect, associate | A is related to B | trust → honesty |
| `temporal` | after, before, during | A happens before B | sunrise → daylight |
| `contrastive` | but, however, unlike | A is opposite of B | love ↔ hate |
| `analogical` | like, similar, metaphor | A is like B | memory ↔ filing cabinet |
| `contextual` | part of, member of | A is part of B | wheel → car |

Relation types are **descriptive labels**, not functional categories. The
actual functional behavior is determined by edge **weight** (which controls
activation flow), not by the type label.

### Vector prototypes

Each relation type has a **prototype vector** — the mean GloVe vector of
seed words representing that relation:

```python
proto_map = {
    "causal": ["cause", "lead", "trigger", "produce", "because", "therefore", ...],
    "contrastive": ["but", "however", "unlike", "opposite", ...],
    "temporal": ["when", "after", "before", "during", "then", ...],
    ...
}
```

When the engine needs to infer a relation type for a new edge, it computes the
**relational direction vector** (target_vector - source_vector) and finds the
closest prototype via cosine similarity.

---

## Deep dive: spreading activation

**File:** `ravana/src/ravana/chat/chain_walker.py` → `_spread_and_collect()`

Spreading activation is the **sole inference mechanism** in RAVANA. There is
no attention mechanism, no matrix multiplication, no transformer layer.

### The algorithm

```
1. Activate seed concept(s) with activation = 1.0
2. For each hop (up to 3):
   a. For each active node:
      - Follow outgoing edges
      - Transfer: signal = node.activation × edge.weight × edge.confidence × decay
      - Apply relation bias: signal ×= relation_preference[edge.relation_type]
      - Apply topic relevance gate: suppress semantically distant concepts
      - Apply degree suppression: high-degree concepts get penalized
   b. Accumulate signals at target nodes
3. Collect all nodes with activation > 0.05 (up to 12)
4. Clear all activations (reset for next turn)
```

### The decay formula

```
decay = 0.7^hop
```

After 3 hops: decay = 0.7³ = 0.343 — meaning a signal from the seed concept
reaches 3-hop neighbors at 34% strength.

### The topic relevance gate (pattern separation)

**Files:** `ravana/src/ravana/chat/synaptic_dynamics.py`

During spread, the engine computes a **subject vector** (the averaged vector
of the seed concepts). Any candidate association whose vector is too distant
from the subject vector gets suppressed:

```python
semantic_sim = cosine(subject_vector, candidate_vector)
signal *= relevance_suppression_dual(semantic_sim)
```

This prevents cross-topic bleeding — e.g., "water" (learned from a love
conversation) leaking into a "blockchain" response.

### The degree suppression

High-degree concepts (e.g., "thing", "way", "person" — concepts connected to
everything) are penalized because they're semantically bland and would dominate
spread otherwise:

```python
signal *= degree_suppression(degree, max_degree, semantic_sim)
```

This is the computational analog of the **fan effect** (Anderson 1974): facts
connected to many concepts are harder to retrieve because of interference.

### Contradiction detection

**File:** `ravana/src/ravana/chat/chain_walker.py` → `_is_contradictory()`

The engine maintains a **contradiction map** built from antipodal GloVe pairs
(cosine < -0.15). If the engine is about to assert a belief that contradicts
a previously logged assertion, it flags it:

```python
# "trust is fragile" contradicts with "trust is strong" if fragile ↔ strong
contradiction_map["trust"].add("fragile")
if new_assertion_target in contradiction_map[source]:
    → contradiction detected → resolve during sleep
```

---

## Why the graph instead of a weight matrix?

This is the most important design decision in RAVANA. Here's the comparison:

| Property | ConceptGraph | Neural network weight matrix |
|----------|-------------|------------------------------|
| **Storage** | Sparse — only active edges stored | Dense — every parameter stored |
| **Memory** | 10K nodes + 100K edges ≈ 15 KB | 10K × 10K matrix = 400 MB (FP32) |
| **Inference** | Graph walk (O(K) where K=active nodes) | Matrix multiply (O(N²)) |
| **Interpretability** | Every edge is human-readable | Opaque floating-point numbers |
| **Relation types** | Typed (causal, semantic, etc.) | Not possible in a single matrix |
| **Learning** | Hebbian updates on active edges | Backprop through entire matrix |
| **Forgetting** | Natural pruning of unused edges | Catastrophic (without replay) |

For a cognitive system that needs to **learn continuously** without catastrophic
forgetting, the graph is the natural choice. The brain doesn't store knowledge
in a weight matrix either — it stores it in **synaptic connections** between
neurons, which is exactly what a graph is.

---

## What the code does step by step

```python
# 1. Create a standalone ConceptGraph
graph = ConceptGraph(dim=64, max_nodes=1000)
```

This creates an empty graph. No GloVe, no PMI, no pretrained knowledge.
Just a container for nodes and edges.

```python
# 2. Add nodes
a = graph.add_node(vector=np.random.randn(64).astype(np.float32), label="trust")
```

`add_node()` returns the node ID (a unique integer). The vector is the
concept's position in semantic space. In the real engine, this comes from
GloVe. Here we use random vectors for demonstration.

```python
# 3. Add typed edges
graph.add_edge(a, b, relation_type="causal", weight=0.7, confidence=0.8)
```

`add_edge()` creates a directed edge from source to target with:
- **weight**: how much activation passes through (0-1)
- **relation_type**: what kind of relationship
- **confidence**: how certain we are
- **prediction_free_energy**: initialized to 0 (set during learning)

```python
# 4. Walk outgoing edges
for target_id, edge in graph.get_outgoing(a):
    target_node = graph.get_node(target_id)
```

`get_outgoing()` returns all edges where node `a` is the source. Each edge
carries weight, relation_type, confidence, and prediction_free_energy.

```python
# 5. Seed activation
graph.activate(a, 1.0)
```

In the real engine, this is the entry point for spreading activation.
Activation spreads through edges to neighboring nodes.

---

## Expected output (annotated)

```
=== Concept Graph Demo ===

nodes: 3
└── We added 3 concepts: trust, courage, fear

edges: 3
└── We created 3 relationships between them

Outgoing from 'trust':
  -> courage       causal        w=0.70  conf=0.80
  -> fear          semantic      w=0.30  conf=0.40
└── trust has 2 outgoing edges:
    → courage (causal, strong confidence)
    → fear (semantic, weaker confidence)

Supported relation types:
  causal
  contrastive
  semantic
└── The types we used in this demo
```

---

## Key source files reference

| Component | File (relative to repo root) |
|-----------|------------------------------|
| ConceptGraph class | `ravana_ml/src/ravana_ml/graph.py` (line 1047) |
| ConceptNode | `ravana_ml/src/ravana_ml/graph.py` (line 41) |
| ConceptEdge | `ravana_ml/src/ravana_ml/graph.py` (line 149) |
| Spreading activation | `ravana/src/ravana/chat/chain_walker.py` (line ~300, `_spread_and_collect`) |
| Relation prototypes | `ravana/src/ravana/chat/chain_walker.py` (line ~240, `_init_relation_prototypes`) |
| Synaptic dynamics / gates | `ravana/src/ravana/chat/synaptic_dynamics.py` |
| Graph engine (chat wrapper) | `ravana/src/ravana/graph/engine.py` (line 309) |

---

## Design philosophy notes

1. **The graph is the model.** There are no hidden layers, no attention heads,
   no transformer blocks. All knowledge is explicit, typed, and inspectable.
2. **Spreading activation replaces attention.** Instead of computing attention
   scores over all tokens, RAVANA propagates activation through typed edges
   with biologically-plausible decay and gating.
3. **Typed edges enable causal reasoning.** Because edges have relation types,
   the engine can follow only causal edges when answering "what causes X" —
   something a dense weight matrix cannot do.
4. **The graph grows continuously.** Unlike a fixed-size weight matrix, the
   graph can add new nodes and edges at any time without retraining.

---

## Next tutorial

[**Tutorial 04: Continuous Learning**](../04-continuous-learning/) — see how
the graph grows as the system searches the web autonomously.
