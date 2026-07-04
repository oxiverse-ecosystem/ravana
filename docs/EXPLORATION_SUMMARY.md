# RAVANA Cognitive Framework Exploration

## What is RAVANA?

A **pressure-driven cognitive ML framework** where learning emerges from self-organization, not gradient descent.

### Key Innovations:
- **No backpropagation** — Hebbian/Anti-Hebbian plasticity + sleep consolidation
- **ConceptGraph** — Typed edges (semantic, causal, temporal, analogical, contextual, inferred)
- **Sleep Consolidation** — SWS (pattern compression, abstraction, contradiction resolution) + REM (dream sabotage)
- **VAD Emotion Engine** — 3D affective state (Valence, Arousal, Dominance) modulating inference
- **GRACE Governor** — 4-layer constraint-based regulation (no SGD/Adam)
- **RLMv2 Triple Decomposition** — (subject, relation, object) with verb-stem offset arithmetic
- **Test-Time Entity Adapter** — Recovers 5-12% → 93-100% held-out generalization
- **Continuous Web Learning** — Autonomous curiosity-driven background learning with multi-API search
- **Theory of Mind** — UserModel tracks goals, preferences, emotional state, and relationship depth per user
- **Emotional Mirroring** — User arousal modulates generation temperature, verbosity, and concept breadth

## Demo Results

### RLMv2 Cross-Domain Transfer
| Metric | Result |
|--------|--------|
| Cross-domain transfer Top-1 | **75-100%** |
| Held-out Science Top-1 (adapted) | **93.8%** (n=16) |
| Held-out Social Top-1 (adapted) | **95-100%** (n=20) |
| Catastrophic forgetting (A→B→C) | **0.167** drop |
| Sleep interleaved replay retention | **0% drop** |

### Graph Scaling (Measured, Not Extrapolated)
| Nodes | `find_similar` p50 | `spread_activation` | Consolidation (full / incremental) |
|-------|-------------------|---------------------|-------------------------------------|
| 1K    | **0.021 ms**      | 0.78 ms             | 1.82 ms / 0.20 ms |
| 5K    | **0.044 ms**      | —                   | — |
| 10K   | **0.059 ms**      | 1.15 ms             | 27.98 ms / 2.14 ms |
| 50K   | **0.66 ms**       | —                   | 49.20 ms / 3.12 ms (15.8× speedup) |

### Test Suite
**1456+ tests passing** across CI (30), unit (1310), integration (95), GRACE (16), and generation (5).

## How It Works (RLMv2)
```
input "heat causes expansion"
    → decompose: (subject="heat", relation="causes", object="expansion")
    → classify relation: "causes" → CAUSAL type embedding
    → spread activation from subject, filtered by CAUSAL edges
    → score activated nodes against all token embeddings
    → blended with verb-offset: subject_embed + offset("causes") ≈ expansion_embed
    → return logits over vocabulary
```

## What Makes This Special?

| Traditional ML | RAVANA |
|---------------|--------|
| Loss functions → gradients | Free energy → self-organization |
| Backpropagation | Hebbian co-activation |
| Weight decay / optimizer | Identity regulation + Governor constraints |
| Batch/Layer normalization | Sleep consolidation (SWS+REM) |
| Dense weight matrices | ConceptGraph (sparse, typed, inspectable) |
| Global attention | Spreading activation (local, typed, bounded) |
| Static embeddings | GloVe + verb-stem offset + entity adapter |
| GPU required | CPU-native (NumPy only) |
| No emotion/identity | VAD emotion + Identity momentum + Meaning engine |

## Architecture Overview

RAVANA has **three layers**:

1. **`ravana_ml/`** — ML Framework (5,200+ lines): NumPy-only PyTorch-like API with ConceptGraph, RLMv1/v2, Neuromodulator, plasticity rules, free energy accumulation
2. **`ravana-v2/`** — GRACE Cognitive Core (19,500+ lines): 27-phase architecture (A–P) with Governor, Identity, Sleep, VAD Emotion, Human Memory, Global Workspace, Dual Process, Belief Reasoning, Active Epistemology, Meta-cognition, Social Epistemology
3. **`ravana/`** — Modular Chat Package: Continuous web learning, decoder-first generation, PFC discourse planning, Basal Ganglia gating, cerebellar n-gram fluency, syntactic cell assembly, SurfaceRealizer (morphology/agreement), Theory of Mind (UserModel), Emotional Mirroring, multi-user BeliefStore, curiosity-driven exploration

## Key Insights

1. **Verb offsets are the primary cross-domain driver** — `offset("causes")` is consistent across domains
2. **Test-time entity adapter adaptation** is essential for held-out generalization (5%→95%)
3. **Sleep eliminates catastrophic forgetting** via domain-tagged interleaved replay
4. **Continuous web learning** enables knowledge growth without retraining
5. **Theory of Mind** personalizes responses based on user goals, knowledge, and emotional state

## Try It Yourself

```python
# RLMv2 — Language model
from ravana.nn import RLM
from ravana_ml.tokenizer import WordTokenizer
import numpy as np

tok = WordTokenizer()
tok.encode("heat causes expansion")

model = RLM(vocab_size=tok.vocab_size, embed_dim=64, concept_dim=64, n_concepts=100)
inp = np.array(tok.encode("heat causes"), dtype=np.int64)
tgt = np.array(tok.encode("expansion"), dtype=np.int64)
model.learn(inp, tgt)
model.sleep_cycle()
logits = model.forward(inp)
print(f"Predictions: {tok.decode([np.argmax(logits)])}")
```

```python
# Full CognitiveFramework
from ravana.cognitive import CognitiveFramework

fw = CognitiveFramework()
state = fw.initialize()
for episode, (inp_vec, tgt_vec) in enumerate(training_data):
    concepts = fw.perceive(state, inp_vec)
    predictions = fw.predict(state, concepts)
    state = fw.learn(state, predictions, tgt_vec, episode)
    if episode % 100 == 0:
        state = fw.sleep(state)
result = fw.infer(state, test_vec)
```

```bash
# Interactive chat with web learning
python -m ravana.chat
```

## Files to Explore

| Layer | Key Files |
|-------|-----------|
| **ML Framework** | `ravana_ml/src/ravana_ml/graph.py` (ConceptGraph), `ravana_ml/src/ravana_ml/nn/rlm_v2.py` (RLMv2), `ravana_ml/src/ravana_ml/nn/neural_decoder.py` (decoder), `ravana_ml/src/ravana_ml/nn/neuromodulator.py` (ACh/NE/DA/5-HT) |
| **GRACE Core** | `ravana-v2/src/ravana_grace/core/governor.py` (central regulator), `ravana-v2/src/ravana_grace/core/emotion.py` (VAD engine), `ravana-v2/src/ravana_grace/core/sleep.py` (sleep consolidation) |
| **Modular Chat** | `ravana/src/ravana/chat/engine.py` (CognitiveChatEngine), `ravana/src/ravana/chat/chain_walker.py` (graph walking), `ravana/src/ravana/chat/user_model.py` (ToM), `ravana/src/ravana/web/learner.py` (web learning) |
| **Experiments** | `experiments/experiment_cross_domain.py` (transfer), `experiments/experiment_user_model.py` (personalization), `scripts/external_benchmark.py` (external benchmarks) |
| **Tests** | `scripts/test_emotional_mirror.py` (P2 emotional tracking), `scripts/test_theory_of_mind.py` (P1 ToM), `scripts/validate_held_out_generalization.py` (held-out validation) |