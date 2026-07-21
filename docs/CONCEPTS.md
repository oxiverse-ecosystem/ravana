# Theoretical Foundations

RAVANA replaces the standard ML paradigm (loss functions → gradients → weight update)
with a **pressure-driven self-organization** paradigm. This document explains the key
concepts.

---

## 1. Free Energy & Pressure

### The core idea

In standard ML: `Loss(prediction, target) → ∇Loss → θ ← θ - η∇Loss`

In RAVANA: `Prediction Error → Free Energy (Pressure) → Self-Organization`

**Free energy** (also called prediction error or surprise) is the difference between
what the system expected and what actually happened. This difference creates
**pressure** — a drive to change.

### How it's implemented

`ravana_ml.free_energy.FreeEnergyAccumulator` tracks 5 channels of prediction error:

| Channel | What it measures |
|---------|-----------------|
| **Semantic** | How well the graph predicts concept relationships |
| **Linguistic** | How well the decoder predicts the next word |
| **Episodic** | How well the system recalls past interactions |
| **Contradiction** | How many conflicting beliefs exist |
| **Abstraction** | How well higher-level schemas predict lower-level patterns |

### Why pressure replaces loss

| Loss function | Pressure accumulation |
|---------------|---------------------|
| `L = (y_pred - y_true)²` | `ΔP = error × salience × (1 - confidence)` |
| Fixed gradient | Adaptive: high confidence failure → high pressure |
| Backpropagated globally | Localized to the relevant edge/node |
| Requires autograd graph | No graph needed — O(1) per step |

**Key principle:** High-confidence failures create high pressure, forcing reorganization.
Low-confidence failures create low pressure — the system ignores noise.

---

## 2. Hebbian Plasticity

### The core idea

> "Neurons that fire together, wire together." — Donald Hebb (1949)

In RAVANA: when two concepts are activated simultaneously, the edge between them
strengthens. This is the **primary learning mechanism** — no backpropagation needed.

### Variants

| Type | File | Behavior |
|------|------|----------|
| **Hebbian** | `ravana_ml/plasticity.py` | Co-activation → weight increase |
| **Anti-Hebbian** | `ravana_ml/plasticity.py` | Co-activation → weight decrease (for inhibition) |
| **Structural** | `ravana_ml/plasticity.py` | New edge creation between co-activated nodes |

### How it's used

```python
# In ravana_ml.nn.rlm_v2: edge weight updates during sleep/consolidation
# In ravana.chat.engine: the Plasticity class wraps graph-level Hebbian updates
```

---

## 3. The Governor (Constraint Satisfaction)

The Governor (in `ravana_grace.core.governor.Governor`) replaces the optimizer.

### Pipeline (4 layers of regulation)

1. **Hard constraints** — absolute ceilings/floors (non-negotiable)
2. **Predictive dampening** — slow before hitting a wall
3. **Boundary pressure** — sigmoid soft resistance near edges
4. **Center-seeking** — homeostatic pull to target zone

### Key equations

```python
# Hard clamp
clamped = min(max(proposed, floor), ceiling)

# Predictive dampening
if proposed > wall * 0.9:
    dampening = (wall - proposed) / (wall * 0.1)
    damped = proposed * dampening

# Center-seeking (homeostatic pull)
pull = (target - current) * homeostatic_strength
state += pull
```

### Comparison to optimizers

| Traditional | Governor |
|------------|----------|
| Learning rate | Clamp rate |
| Momentum | Identity momentum |
| Weight decay | Identity regularization |
| Gradient clipping | Hard constraints |
| Batch normalization | Sleep consolidation |

---

## 4. Identity — The Regularizer

`ravana_grace.core.identity.IdentityEngine` maintains a **self-concept** that prevents
catastrophic forgetting.

### Dynamics

```python
I_t = I_{t-1} + ΔI_regulated + bonus - penalty
```

Where:
- `ΔI_regulated`: Governor-approved identity change
- `bonus`: +0.08 per successful resolution (streak-multiplied)
- `penalty`: -0.08 per failed prediction (fixed)
- `recovery_bias`: Growth boost when `I < 0.5`
- `stability_damping`: Shrink delta when `I > 0.85`

### Why it works

Identity is a **structural self-concept**, not a numeric penalty term. It creates a
homeostatic anchor — the system resists changes that would violate its core
self-model. This is the cognitive analog of weight decay, but emergent rather than
imposed.

---

## 5. Sleep Consolidation

Sleep is **not** a pause in learning — it IS the learning. The `sleep_cycle()`
method replaces `optimizer.step()`.

### 4-stage sleep (ravana_grace.core.sleep.SleepConsolidation)

| Stage | Name | What happens |
|-------|------|-------------|
| 1 | **Analysis** | Scan graph for high-pressure zones (edges with high prediction error) |
| 2 | **Compression** | Strengthen low-error edges, weaken noisy ones |
| 3 | **Contradiction resolution** | Resolve contradictory belief pairs |
| 4 | **Integration** | Merge episodic patterns into semantic structure |

### SWS + REM (ravana_ml.nn.rlm_v2_sleep)

- **SWS** (Slow-Wave Sleep): Structural stabilization, hierarchical compression,
  inhibitory edge formation, hippocampal replay
- **REM**: Creative recombination — 20% counterfactual reversals, 10% emotional
  valence flipping, 1.5× failure oversampling

### Sleep-Time Interleaved Replay

Domain-tagged experiences are buffered during training and replayed during SWS+REM.
This **eliminates catastrophic forgetting entirely** (12% → 0% retention drop) in
lifelong streaming benchmarks.

---

## 6. VAD Emotion Engine

`ravana_grace.core.emotion.VADEmotionEngine` models affect as a 3D continuous space:

| Dimension | Range | What it means |
|-----------|-------|--------------|
| **Valence** | -1 to +1 | Pleasantness (sad → happy) |
| **Arousal** | 0 to 1 | Alertness (calm → excited) |
| **Dominance** | 0 to 1 | Control (submissive → in control) |

### Effects on inference

- **High arousal** → exploration mode, more diverse associations
- **Positive valence** → trust predictions, optimism bias
- **High dominance** → stronger concept activations, more confident answers

### Dynamics

Emotion evolves through differential equations (not lookup tables):

```python
d_valence/dt = η_valence × (current_valence - resting_valence) + emotional_input
d_arousal/dt = η_arousal × (current_arousal - resting_arousal) + arousal_input
```

---

## 7. Concept Graph (ravana_ml.graph.ConceptGraph)

The graph is the **primary knowledge store** — not a weight matrix.

### Node structure

```python
@dataclass
class ConceptNode:
    id: int
    label: Optional[str]
    vector: np.ndarray          # 64-D GloVe projection
    node_type: ConceptNodeType  # active / core / genesis
    stability: float            # resistance to modification
    activation: float           # current spread activation level
    confidence: float           # belief certainty
```

### Edge structure

```python
@dataclass
class ConceptEdge:
    weight: float               # connection strength (0-1)
    relation_type: str          # causal / semantic / temporal / contrastive / analogical
    confidence: float           # how certain we are this edge is correct
    prediction_free_energy: float  # prediction error (drives learning)
    source: str                 # seed / web / user / sleep / inference
```

### Relation types

| Type | Meaning | Example |
|------|---------|---------|
| `causal` | A causes B | heat → expansion |
| `semantic` | A is related to B | trust → honesty |
| `temporal` | A happens before B | sunrise → daylight |
| `contrastive` | A is opposite of B | love ↔ hate |
| `analogical` | A is like B | memory ↔ filing cabinet |
| `contextual` | A is part of B | wheel → car |
| `inferred` | Derived from other edges | (computed) |

---

## 8. The Neural Decoder

`ravana_ml.nn.neural_decoder.NeuralDecoder` is a **small GRU** that generates
language conditioned on a concept embedding. It is NOT a language model in the
traditional sense — it is a **realization model** that translates graph activations
into word sequences.

### Architecture

```
Graph walk embedding (64-D)
       │
       ▼
  GRU (64-D hidden)
       │
       ▼
  Softmax over vocabulary
       │
       ▼
  Word sequence
```

### Training

- Trained **online** (no offline corpus required)
- Sampled-softmax with early stopping on cross-entropy
- Sleep consolidation replaces gradient steps
- Self-conditioning cheat was removed — honest CE only

---

## 9. RLMv2 — Triple Decomposition

`ravana_ml.nn.rlm_v2.RLMv2` decomposes sentences into (subject, relation, object)
triples. This enables **vector arithmetic analogy**:

```
subject_embed + offset(verb) ≈ target_embed
```

### Verb-stem offset

Each verb gets its own offset vector:

```python
offset("causes") = avg(target_embed - subject_embed) over all "causes" triples
```

This means the model can generalize: if it knows `heat causes expansion`, it can
infer `cold causes contraction` by applying the `causes` offset to `cold`.

### Spreading activation as inference

RLMv2 does NOT use attention or matrix multiplication. Inference is **spreading
activation** through the graph: activate seed concepts, propagate through typed
edges, collect activated neighbors.

---

## 10. Curiosity Drive

`ravana.learn.curiosity.CuriosityEngine` selects what to learn next — not from
a fixed curriculum, but from the system's own learning state.

### Selection criteria

| Signal | What it measures |
|--------|-----------------|
| **Prediction error** | Concepts where the graph predictions are wrong |
| **Novelty** | Concepts with few visited edges |
| **Contradiction** | Pairs of beliefs that conflict |
| **Serendipity** | Unexpected co-activations |

### Formula

```python
curiosity_score = w_pred * prediction_error + w_novel * novelty + w_contra * contradiction
```

The concept with the highest score is selected for web research on the next idle cycle.
