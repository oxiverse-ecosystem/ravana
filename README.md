# RAVANA — Recursive Learning Model

> **A pressure-driven cognitive ML framework where learning emerges from self-organization, not gradient descent.**
>
> CPU-native • No GPU required • NumPy only

---

## What Is RAVANA?

RAVANA is a **cognitive architecture research project** that proposes *pressure-driven self-organization* as an alternative to gradient descent for learning. Instead of minimizing loss functions via backpropagation, RAVANA learns through **cognitive dissonance minimization**: prediction errors create internal pressure, and the system self-organizes to reduce that pressure.

```
Traditional ML:     loss = f(prediction, target) → gradient → update weights
RAVANA:             prediction error → pressure → self-organization → equilibrium
```

| Concept | Traditional ML | RAVANA |
|---------|---------------|--------|
| Learning signal | Loss function → gradient | **Pressure** (dissonance, contradiction) |
| Optimization | Gradient descent / backprop | **Governor** regulation + Hebbian plasticity |
| Stability | Weight decay, regularization | **Identity** + Homeostatic sleep consolidation |
| Consolidation | Batch normalization | **Sleep** (SWS + REM) |
| Architecture | Dense weight matrices | **ConceptGraph** (typed edges, inhibition) |
| Context | Attention (global) | **Spreading activation** (local, bounded) |
| Memory | Weights (static) or RAG | Episodic → Semantic → Graph bridge |

---

## Architecture: Three-Layer Package

The project is organized into three layers totaling **~51,700 lines across 225 Python files**.

### Layer 1: `ravana_ml/` — ML Framework (5,200+ lines, 18 files)

A PyTorch-compatible API surface built entirely on NumPy. The **only hard dependency** is `numpy>=1.20`.

```python
import ravana as torch
x = torch.tensor([1, 2, 3])
model = torch.nn.Linear(10, 10)
y = model(x)
model.accumulate_free_energy(y - target)
model.sleep_cycle()  # ← instead of optimizer.step()
```

| Module | Lines | Purpose |
|--------|-------|---------|
| `graph.py` | 3,678 | `ConceptGraph` — Hebbian/anti-Hebbian plasticity, spreading activation, hierarchical abstraction, inhibitory edges, concept splitting, relation vectors |
| `nn/rlm_v2.py` | 2,874 | **RLMv2** — Triple decomposition architecture (subject, relation, object) with verb-stem offset predictor, GloVe embeddings, sleep-time interleaved replay |
| `nn/module.py` | 473 | Module base, Linear (no backprop!), GRUCell, LayerNorm, ConceptAttentionHead (multi-head QKV) |
| `nn/rlm.py` | 3,931 | **RLM** — Recursive Learning Model. Predictive coding settle loop, GRU, native cognitive state (identity, VAD emotion, meaning), episodic memory, sleep consolidation |
| `tensor.py` | 385 | `RawTensor` + `StateTensor` with salience, free_energy, stability, decay fields |
| `tokenizer.py` | 208 | `WordTokenizer` (default), `BPETokenizer` (tiktoken/GPT-2), `SimpleTokenizer` (char-level) |
| `free_energy.py` | 90 | 5-channel free energy accumulator (semantic, linguistic, episodic, contradiction, abstraction) |
| `currencies.py` | 250 | Unified cognitive currency system — identity, emotion, meaning, sleep pressure, dissonance |
| `episode_injector.py` | 276 | Structured knowledge injection into the concept graph |
| `relation_ontology.py` | 231 | Multi-level relation hierarchy (Family > Sub-family > Predicate) |
| `embedder.py` | 81 | LearnedEmbedder — character n-gram + random projection (64-dim) |
| `plasticity.py` | 77 | Hebbian, anti-Hebbian, and structural plasticity rules |
| `propagation.py` | 78 | Activation spreading engine over the concept graph |
| `nn/functional.py` | 118 | Functional API (relu, softmax, cross_entropy, etc.) |

### Layer 2: `ravana-v2/` — Cognitive Core (82 files, 27 core modules, 19,505 lines)

The **GRACE architecture** (Governance, Reflection, Adaptation, Constraint, Exploration) implements high-level cognitive functions across ordered development phases.

```python
from ravana.cognitive import CognitiveFramework

fw = CognitiveFramework()
state = fw.initialize()
concepts = fw.perceive(state, input_vec)      # input → active concepts
predictions = fw.predict(state, concepts)      # Hebbian spread → predictions
state = fw.learn(state, predictions, outcomes) # pressure + governor + emotion
state = fw.sleep(state)                        # 4-stage consolidation + memory
result = fw.infer(state, input_vec)            # inference with memory bias
```

| Phase | Module | What It Does |
|-------|--------|-------------|
| **A** | `governor.py` (743) | Central regulation — hard constraints, predictive dampening, boundary pressure, mode regulation |
| **A** | `identity.py` | Momentum-based self-concept with recovery bias |
| **A** | `resolution.py` | Continuous partial credit toward wisdom events |
| **B** | `adaptation.py` | Policy learning from clamp events |
| **C** | `strategy.py` | 4 exploration modes (AGGRESSIVE, SAFE, STABILIZE, RECOVER) |
| **D** | `intent.py` | Dynamic objectives evolving from outcomes |
| **E** | `environment.py` | Non-stationary world (boundary shifts, noise drift, goal flips) |
| **F** | `predictive_world.py` | Neural network world model with adaptive surprise threshold |
| **F.5** | `belief_reasoner.py` | Competing hypotheses with confidence decay |
| **G** | `active_epistemology.py` | Value of Information (VoI) calculation |
| **H** | `meta_cognition.py` (435) | Self-awareness, bias detection, mode recommendation |
| **I** | `social_epistemology.py` (760) | Multi-agent belief conflict, trust scoring, deception detection |
| **K** | `emotion.py` (234) | VAD (Valence, Arousal, Dominance) via differential equations |
| **L** | `sleep.py` (703) | 4-stage SWS + REM with dream sabotage |
| **M** | `meaning.py` (224) | Intrinsic motivation: `M = w1(-D) + w2(I) + w3(prediction) × (1 + κ × effort)` |
| **N** | `global_workspace.py` | Competitive broadcast system, consciousness bottleneck |
| **O** | `human_memory.py` (2,321) | Persistent episodic/semantic memory with Ebbinghaus decay, interference, reconstructive recall |

### Layer 3: `ravana/` — Unified Package (10 files, 855 lines)

A single pip-installable package that re-exports both codebases:

```
ravana/
├── __init__.py          → import ravana as torch  (re-exports all)
├── nn/                  → RLM imports from ravana_ml.nn
├── cognitive/           → CognitiveFramework + 23+ modules from ravana-v2/core/
├── graph/ propagation/  → ConceptGraph from ravana_ml
├── world/ lab/          → Simulations and experiments
└── pyproject.toml       → pip install -e ravana/
```

---

## Key Innovations

### 🧠 Pressure-Driven Learning

No loss functions, no gradients, no backpropagation. Learning is driven by **free energy** — a five-channel accumulator tracking semantic, linguistic, episodic, contradiction, and abstraction prediction errors. The system self-organizes to reduce this pressure through Hebbian plasticity, inhibitory edge formation, concept splitting, and sleep-phase consolidation.

### 🕸️ ConceptGraph with Typed Edges

A heterogeneous graph of concept nodes connected by typed relation edges (semantic, causal, temporal, inferred, inhibitory). Each concept node carries an active vector (fast-adapting) and core vector (slow anchor). Each edge carries a weight, confidence, relation vector, and bidirectional prediction counts for structural relation inference.

### 💤 Sleep Consolidation

Two-phase sleep modeled on mammalian sleep stages:

- **SWS (Slow-Wave Sleep)**: Structural stabilization, adaptive homeostatic downscale, hierarchical abstraction compression, inhibitory edge formation, memory replay through graph
- **REM (Rapid Eye Movement)**: Creative recombination through 20% counterfactual reversals, 10% emotional valence flipping, 1.5x failure oversampling

### 🔄 Sleep-Time Interleaved Replay

Domain-tagged experiences are buffered during training and replayed during SWS+REM sleep cycles. This eliminates catastrophic forgetting entirely (12% → 0%) in lifelong streaming benchmarks.

### 🏛️ GRACE Governor

Closed-loop regulation system with four layers of control: predictive dampening (foresight), soft boundary pressure (air resistance), center-seeking homeostasis, and hard constitutional clamps. Every state change must pass through the Governor.

### 🎭 VAD Emotion Engine

3D affective state (Valence, Arousal, Dominance) computed via differential equations. Emotions modulate inference: high arousal → exploration, positive valence → trust concept predictions, high identity → stronger concept signal.

### 🧩 RLMv2 — Triple Decomposition Architecture

Brain-inspired architecture that decomposes input into (subject, relation_type, object) triples instead of character sequences. Enables vector arithmetic analogy (`subject_embed + avg_relation_vector ≈ target_embed`) and relation-aware spreading activation for cross-domain generalization.

### 📖 GloVe Semantic Embeddings (NEW — 2026-06-07)

Token embeddings initialized from pre-trained GloVe vectors (100D) projected via QR-based orthogonal projection. Replaces character n-gram embeddings — enables genuine semantic relationships for verb-stem offset prediction.

### 🎯 Verb-Stem Offset Predictor (NEW — 2026-06-07)

A new inference path replacing bilinear `W_rel @ subject` with verb-conditioned vector arithmetic:

```
offset(verb) = avg(target - subject) over all training pairs using that verb
predicted_embed = subject_embed + offset(query_verb)
```

Each verb gets its own offset, enabling same-subject different-verb predictions. **RP-only cross-domain accuracy: 3.3% → 6.7% top-10.**

---

## Quick Start

### Installation

```bash
# Install the unified package (NumPy only)
pip install -e ravana/

# Or if you need tiktoken (optional, for BPE tokenizer)
pip install tiktoken
```

### ML Framework Basics

```python
import ravana as torch

# Create tensors
x = torch.tensor([1.0, 2.0, 3.0])

# Build a model
model = torch.nn.Linear(3, 2)
y = model(x)

# Learn without backpropagation!
target = torch.tensor([0.5, 0.3])
model.accumulate_free_energy(y - target)
model.sleep_cycle()  # ← consolidation instead of optimizer.step()
```

### RLMv2 — Language Model

```python
from ravana.nn import RLM
from ravana_ml.tokenizer import WordTokenizer
import numpy as np

# Setup
tokenizer = WordTokenizer()
text = "heat causes expansion"
tokenizer.encode(text)

model = RLM(vocab_size=tokenizer.vocab_size, embed_dim=64, concept_dim=64)

# Learn a fact
input_ids = np.array(tokenizer.encode("heat causes "), dtype=np.int64)
target_ids = np.array(tokenizer.encode("expansion"), dtype=np.int64)
model.learn(input_ids, target_ids)

# Sleep to consolidate
model.sleep_cycle()
```

### Cognitive Framework

```python
from ravana.cognitive import CognitiveFramework

fw = CognitiveFramework()
state = fw.initialize()

# Full cognitive cycle
concepts = fw.perceive(state, input_vec)
predictions = fw.predict(state, concepts)
state = fw.learn(state, predictions, outcomes)
state = fw.sleep(state)    # consolidation with memory bridge

# Save/load
fw.save("checkpoint.pkl")
fw = CognitiveFramework.load("checkpoint.pkl")
fw.rebridge()              # sync consolidated memories → graph edges
```

---

## Project Structure

```
ravana/
├── README.md                          # ← You are here
├── ravana/                            # Unified pip-installable package
│   ├── __init__.py                    # import ravana as torch
│   ├── nn/                            # RLM re-exports
│   ├── cognitive/                     # CognitiveFramework + 23+ modules
│   └── pyproject.toml                 # pip install -e . (dep: numpy)
├── ravana_ml/                         # ML framework (~5,200 lines)
│   ├── nn/
│   │   ├── rlm.py                     # RLM (predictive coding, GRU, cognitive state)
│   │   ├── rlm_v2.py                  # RLMv2 (triple decomposition, GloVe, verb-offset)
│   │   ├── module.py                  # Module, Linear, GRUCell, LayerNorm, ConceptAttention
│   │   └── functional.py              # relu, softmax, cross_entropy
│   ├── graph.py                       # ConceptGraph (3,678 lines)
│   ├── tensor.py                      # RawTensor, StateTensor
│   ├── tokenizer.py                   # WordTokenizer, BPETokenizer, SimpleTokenizer
│   ├── currencies.py / currency.py    # Cognitive currency system
│   ├── free_energy.py                 # 5-channel free energy accumulator
│   ├── plasticity.py                  # Hebbian + anti-Hebbian rules
│   └── propagation.py                 # Activation spreading engine
├── ravana-v2/                         # GRACE cognitive core (~19,500 lines)
│   ├── core/
│   │   ├── governor.py                # Central regulation
│   │   ├── identity.py                # Self-concept with momentum
│   │   ├── emotion.py                 # VAD emotion engine
│   │   ├── sleep.py                   # Sleep consolidation (SWS + REM)
│   │   ├── memory.py                  # Episodic, semantic, working memory
│   │   ├── human_memory.py            # Persistent SQLite memory (2,321 lines)
│   │   ├── dual_process.py            # System 1 / System 2
│   │   ├── meaning.py                 # Intrinsic motivation
│   │   ├── belief_reasoner.py         # Competing hypotheses
│   │   ├── meta_cognition.py          # Self-awareness, bias detection
│   │   ├── social_epistemology.py     # Multi-agent trust + deception
│   │   ├── active_epistemology.py     # Value of Information
│   │   ├── global_workspace.py        # Consciousness bottleneck
│   │   └── ... (15 more modules)
│   ├── tests/                         # v2 unit tests
│   ├── experiments/phases/            # Phase runners (B through J)
│   └── training/pipeline.py           # Training orchestration
├── tests/                             # ML framework tests (17 files)
├── experiments/                       # Cross-domain and Phase 4 experiments
├── scripts/                           # Analysis and profiling tools
├── docs/                              # Full documentation
│   ├── ARCHITECTURE.md                # Complete architecture reference
│   ├── ARCHITECTURE_v2.md             # GRACE architecture design
│   ├── RAVANA_STATUS.md               # Current codebase status and benchmarks
│   ├── RAVANA_REPORT.md               # Technical paper / research report
│   ├── PAPER_DRAFT.md                 # Academic paper draft
│   ├── SCIENCE_DIRECT_MANUSCRIPT.md   # ScienceDirect manuscript
│   ├── ANALYSIS_RLM_vs_LLM.md         # RLM vs Transformer gap analysis
│   └── EXTERNAL_AUDIT.md              # External LLM collaborator audit
├── data/glove/                        # Pre-trained GloVe embeddings
│   └── glove.6B.100d.txt              # (not included, download required)
└── results/                           # Benchmark outputs and diagnostics
```

---

## Running Tests

```bash
# ML framework tests (17 files)
python -m pytest tests/ -v

# RLMv2 unit tests
python -m pytest tests/test_rlm_v2.py -v

# Cognitive core tests
python -m pytest ravana-v2/tests/ -v

# All tests
python -m pytest tests/ ravana-v2/tests/ -v
```

---

## Running Experiments

```bash
# Cross-domain transfer experiment
python experiments/experiment_cross_domain.py

# Phase 4 integrated experiment (triplet margin + wake-sleep)
python experiments/experiment_phase4_integrated.py

# RLMv2 triple benchmark
python experiments/experiment_triple_benchmark_v6.py

# RP-only diagnostic tests (moved to tests/)
python tests/test_rp_only.py
python tests/test_rp_contrastive.py
python tests/test_structural_transfer.py

# Cognitive architecture experiments
cd ravana-v2 && python experiments/runner.py
python experiments/phases/run_phase_b.py
```

---

## Benchmarks

| Benchmark | Result | Status |
|-----------|--------|--------|
| Within-domain top-1 accuracy | **100%** (was 0%) | ✅ |
| RLMv2 triple benchmark (47 triples, 500 epochs) | **80.9% overall top-10** | ✅ |
| RLMv2 cross-domain causal top-10 | **75%** | ✅ |
| Cross-domain probes (optimized config) | **95% top-1 / 100% top-10** | ✅ |
| Cross-domain probes (neutral, zero-shot) | **10.0% top-10** | ⚠️ |
| RP-only verb-offset cross-domain | **6.7% top-10** (was 3.3%) | ✅ Improved |
| Lifelong retention (15k, replay + EWC + Bayesian) | **47.6% with 0% forgetting** | ✅ |
| Catastrophic forgetting eliminated | **12% → 0%** | ✅ Solved |
| Memory replay: Domain A retention | **0% → 100% top-10** | ✅ |
| Graph-aware encoder alignment | **33.3% → 100% traversal** | ✅ |
| Sleep cycle optimization | **656ms → 255ms** (2.6x) | ✅ |

---

## Documentation

| Document | Description |
|----------|-------------|
| `docs/ARCHITECTURE.md` | Complete architecture reference — GRACE, RLMv2, Phases A–O, GloVe, verb-stem offset |
| `docs/RAVANA_STATUS.md` | Current codebase status, benchmarks, and latest results (225 files, ~51,700 lines) |
| `docs/RAVANA_REPORT.md` | Full technical paper — from 0% to 95% cross-domain, catastrophic forgetting elimination |
| `docs/PAPER_DRAFT.md` | Academic paper draft — "From Zero to Generalization" |
| `docs/SCIENCE_DIRECT_MANUSCRIPT.md` | ScienceDirect manuscript — "Beyond Reward Maximization" |
| `docs/ANALYSIS_RLM_vs_LLM.md` | Honest gap analysis: RLM vs Transformers, Mamba, and local learning rules |
| `docs/ARCHITECTURE_v2.md` | GRACE architecture design motivation and critique responses |
| `docs/EXTERNAL_AUDIT.md` | External LLM collaborator audit and findings |

---

## License

MIT — Built for the RAVANA-AGI-Research initiative.

---

## About

RAVANA is an open research project exploring an alternative paradigm for machine learning — one where cognition emerges from internal pressure, not gradient optimization; where knowledge self-organizes through Hebbian plasticity and sleep consolidation, rather than being molded by backpropagation; and where identity, emotion, and meaning are first-class citizens of the learning dynamics, not afterthoughts.

**The core thesis**: *"Cognition as pressure-driven self-organization."*

This is the center of gravity. Everything else is implementation detail.
