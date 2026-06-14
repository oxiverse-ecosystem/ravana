# RAVANA — Recursive Learning Model

> **A pressure-driven cognitive ML framework where learning emerges from self-organization, not gradient descent.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-orange)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)]()

---

## What Is RAVANA?

RAVANA is a **cognitive architecture research project** proposing *pressure-driven self-organization* as an alternative to gradient descent for machine learning. Instead of minimizing loss functions via backpropagation, RAVANA learns through **cognitive dissonance minimization**: prediction errors create internal pressure (free energy), and the system self-organizes to reduce that pressure.

```
Traditional ML:     Loss(θ) = f(prediction, target) → ∇Loss → θ ← θ - η∇Loss
RAVANA:             Prediction Error → Free Energy (Pressure) → Hebbian Plasticity + Sleep → Equilibrium
```

### Core Philosophy

| Traditional ML | RAVANA |
|----------------|--------|
| **Loss functions** → gradients | **Free energy** (pressure) → self-organization |
| **Backpropagation** | **Hebbian/anti-Hebbian plasticity** + structural plasticity |
| **Weight decay, regularization** | **Identity** + homeostatic sleep consolidation |
| **Batch/layer normalization** | **SWS + REM sleep cycles** with dream sabotage |
| **Dense weight matrices** | **ConceptGraph** (typed edges, inhibition, hierarchy) |
| **Global attention** | **Spreading activation** (local, bounded, typed) |
| **Static weights / RAG** | **Episodic → Semantic → Graph bridge** |
| **External reward functions** | **Intrinsic motivation** (Meaning = f(-Dissonance, Identity, Prediction)) |
| **No emotion/identity** | **VAD emotion engine** + momentum-based identity |

---

## Key Innovations

### 🧠 Pressure-Driven Learning
No loss functions, no gradients, no backpropagation. Learning driven by **5-channel free energy accumulator** tracking semantic, linguistic, episodic, contradiction, and abstraction prediction errors.

### 🕸️ ConceptGraph with Typed Edges
Heterogeneous graph of concept nodes with **active/core/genesis vectors** connected by typed relation edges (semantic, causal, temporal, analogical, contextual, inferred, inhibitory). Each edge carries weight, confidence, relation vector, predicate token, EWC importance, and Bayesian posterior.

### 💤 Sleep Consolidation (SWS + REM)
Two-phase sleep modeled on mammalian sleep:
- **SWS**: Structural stabilization, adaptive homeostatic downscale, hierarchical abstraction compression, inhibitory edge formation, hippocampal replay
- **REM**: Creative recombination via 20% counterfactual reversals, 10% emotional valence flipping, 1.5× failure oversampling

### 🔄 Sleep-Time Interleaved Replay
Domain-tagged experiences buffered during training, replayed during SWS+REM. **Eliminates catastrophic forgetting entirely** (12% → 0% retention drop) in lifelong streaming benchmarks.

### 🏛️ GRACE Governor
Closed-loop regulation with four layers: hard constraints (non-negotiable), predictive dampening (look-ahead), boundary pressure (air resistance), center-seeking homeostasis. **Fully observable** with clamp diagnostics.

### 🎭 VAD Emotion Engine
3D affective state (Valence, Arousal, Dominance) via differential equations. Modulates inference: high arousal → exploration, positive valence → trust predictions, high dominance → stronger concepts.

### 🧩 RLMv2 — Triple Decomposition Architecture
Brain-inspired: decomposes input into **(subject, relation_type, object)** triples. Enables vector arithmetic analogy: `subject_embed + offset(verb) ≈ target_embed`. **Spreading activation is the sole inference mechanism**.

### 📖 GloVe Semantic Embeddings
Token embeddings initialized from pre-trained GloVe (100D) projected via QR-based orthogonal projection. Replaces character n-gram embeddings for genuine semantic relationships.

### 🎯 Verb-Stem Offset Predictor
New inference path: `offset(verb) = avg(target - subject)` over training pairs. Each verb gets its own offset. **RP-only cross-domain top-10: 3.3% → 6.7%**.

---

## Architecture: Three-Layer Package

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: ravana/ (Unified Package)                                                  │
│  pip install -e ravana/  •  import ravana as torch                                  │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │ nn/RLM       │  │ cognitive/       │  │ graph/         │  │ world/, lab/     │  │
│  │ (RLMv2)      │  │ CognitiveFramework│  │ propagation/  │  │ analysis tools   │  │
│  └──────┬───────┘  └────────┬─────────┘  └───────┬────────┘  └────────────────┘  │
└─────────│───────────────────│──────────────────────│──────────────────────────────┘
          │                   │                      │
          ▼                   ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: ravana_ml/ (~5,200 lines, 18 files)        LAYER 2: ravana-v2/ (~19,500 lines, 27 modules) │
│ ┌──────────────────────────────────────────────┐  ┌──────────────────────────────────────────┐   │
│ │ graph.py (3,678) — ConceptGraph              │  │ GRACE Phases A–P:                      │   │
│ │ nn/rlm_v2.py (5,013) — RLMv2 (triple)        │  │ A: Governor, Identity, Resolution       │   │
│ │ nn/rlm.py (3,931) — RLM v1 (predictive)      │  │ B: Adaptation                          │   │
│ │ nn/module.py (473) — Modules (no backprop!)  │  │ C: Strategy (4 modes) + Meta-learning  │   │
│ │ tensor.py — StateTensor (salience, FE, etc.) │  │ D: Intent + D.5: Planning              │   │
│ │ free_energy.py — 5-channel accumulator       │  │ E: Non-stationary Environment           │   │
│ │ plasticity.py — Hebbian/Anti-Hebbian/Struct. │  │ F: World Model + F.5: Belief Reasoner   │   │
│ │ propagation.py — Spreading activation        │  │ G: Active Epistemology + G.5: Probes    │   │
│ │ tokenizer.py — Word/BPE/Simple/Pixel         │  │ J: Hypotheses + J.1: Occam Layer        │   │
│ │ currencies.py — Unified cognitive state      │  │ K: VAD Emotion + K.5: Empathy          │   │
│ │ embedder.py — Char n-gram embedder           │  │ L: Sleep (4-stage) + L.5: Dual Process  │   │
│ └──────────────────────────────────────────────┘  │ M: Meaning + N: Global Workspace        │   │
│                                                   │ O: Human Memory (2,321 lines)           │   │
│                                                   │ P: Dialogue & Repair                     │   │
│                                                   └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/ravana.git
cd ravana
pip install -e ravana/          # NumPy only
pip install tiktoken            # Optional: BPE tokenizer
```

### PyTorch-Style API (No Backprop!)

```python
import ravana as torch

x = torch.tensor([1.0, 2.0, 3.0])
model = torch.nn.Sequential(
    torch.nn.Linear(3, 16),
    torch.nn.LayerNorm(16),
    torch.nn.Linear(16, 2)
)

y = model(x)
model.accumulate_free_energy(y - target)
model.sleep_cycle()  # ← replaces optimizer.step()
```

### RLMv2 — Language Model

```python
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
```

### Cognitive Framework — Full Agent

```python
from ravana.cognitive import CognitiveFramework

fw = CognitiveFramework()
state = fw.initialize()

for episode, (inp_vec, tgt_vec) in enumerate(training_data):
    concepts = fw.perceive(state, inp_vec)
    predictions = fw.predict(state, concepts)
    state = fw.learn(state, predictions, tgt_vec, episode)
    if episode % 100 == 0:
        state = fw.sleep(state)

result = fw.infer(state, test_vec)  # Inference without state change
```

---

## Benchmarks

| Benchmark | Result | Status |
|-----------|--------|--------|
| Within-domain top-1 accuracy | **100%** (was 0%) | ✅ |
| RLMv2 triple benchmark (47 triples, 500 epochs) | **80.9% overall top-10** | ✅ |
| RLMv2 cross-domain causal top-10 | **75%** | ✅ |
| Cross-domain probes (optimized) | **95% top-1 / 100% top-10** | ✅ |
| Cross-domain probes (neutral, zero-shot) | **10% top-10** | ⚠️ |
| RP-only verb-offset cross-domain top-10 | **6.7%** (was 3.3%) | ✅ Improved |
| Lifelong retention (15k, replay+EWC+Bayesian) | **47.6% with 0% forgetting** | ✅ |
| Catastrophic forgetting eliminated | **12% → 0%** | ✅ Solved |
| Memory replay: Domain A retention | **0% → 100% top-10** | ✅ |
| Graph-aware encoder alignment | **33.3% → 100% traversal** | ✅ |
| Sleep cycle optimization | **656ms → 255ms (2.6×)** | ✅ |

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Complete architecture reference — GRACE, RLMv2, Phases A–P, GloVe, verb-stem offset |
| [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) | Quickstart tutorial — install to first model in 5 minutes |
| [`docs/ML_FRAMEWORK.md`](docs/ML_FRAMEWORK.md) | ravana_ml deep dive — tensors, modules, ConceptGraph, RLM v1/v2 |
| [`docs/COGNITIVE_CORE.md`](docs/COGNITIVE_CORE.md) | ravana-v2 reference — all 27 GRACE modules, configurations |
| [`docs/UNIFIED_PACKAGE.md`](docs/UNIFIED_PACKAGE.md) | ravana/ package — PyTorch API, CognitiveFramework, serialization |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | Complete function/class reference for all three layers |
| [`docs/CONCEPTS.md`](docs/CONCEPTS.md) | Theoretical foundations — pressure, free energy, Hebbian, sleep, VAD, triples |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Running experiments — cross-domain, Phase 4, RLMv2 benchmarks, cognitive phases |
| [`docs/ADVANCED_TOPICS.md`](docs/ADVANCED_TOPICS.md) | Customization — tokenizers, plasticity, sleep, governor, multi-agent, lifelong |
| [`docs/TUTORIALS.md`](docs/TUTORIALS.md) | 8 step-by-step tutorials — Hello World to visualization |
| [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) | Contributing — code standards, testing, architecture principles, release process |

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

## Running Experiments

```bash
# Cross-domain transfer
python experiments/experiment_cross_domain.py

# Phase 4 integrated (triplet margin + wake-sleep)
python experiments/experiment_phase4_integrated.py

# RLMv2 triple benchmark
python experiments/experiment_triple_benchmark_v6.py

# Diagnostic tests
python tests/test_rp_only.py
python tests/test_rp_contrastive.py
python tests/test_structural_transfer.py

# Cognitive architecture experiments
cd ravana-v2 && python experiments/runner.py
```

---

## Project Structure

```
ravana/
├── README.md                    # This file
├── ravana/                      # Unified pip-installable package
│   ├── __init__.py              # import ravana as torch
│   ├── nn/                      # RLM re-exports
│   ├── cognitive/               # CognitiveFramework + wiring
│   ├── graph/                   # ConceptGraph re-exports
│   ├── propagation/             # PropagationEngine re-exports
│   ├── world/                   # Simulation environments
│   ├── lab/                     # Analysis tools
│   └── pyproject.toml           # numpy>=1.20
├── ravana_ml/                   # ML Framework (~5,200 lines)
│   ├── nn/
│   │   ├── rlm.py               # RLM v1 (predictive coding, GRU)
│   │   ├── rlm_v2.py            # RLM v2 (triple decomposition, GloVe)
│   │   ├── module.py            # Module, Linear, GRUCell, Attention
│   │   └── functional.py        # relu, softmax, cross_entropy
│   ├── graph.py                 # ConceptGraph (3,678 lines)
│   ├── tensor.py                # RawTensor, StateTensor, Parameter
│   ├── tokenizer.py             # Word, BPE, Simple, Pixel tokenizers
│   ├── currencies.py            # CognitiveCurrencies
│   ├── free_energy.py           # 5-channel accumulator
│   ├── plasticity.py            # Hebbian, Anti-Hebbian, Structural
│   ├── propagation.py           # Spreading activation engine
│   └── ... (embedder, episode_injector, relation_ontology)
├── ravana-v2/                   # GRACE Cognitive Core (~19,500 lines)
│   ├── core/                    # 27 Phases A–P modules
│   │   ├── governor.py          # Central regulation (743 lines)
│   │   ├── identity.py          # Momentum-based self-concept
│   │   ├── sleep.py             # 4-stage SWS+REM (703 lines)
│   │   ├── emotion.py           # VAD differential equations
│   │   ├── human_memory.py      # Episodic/Semantic (2,321 lines)
│   │   ├── global_workspace.py  # Consciousness bottleneck
│   │   ├── meaning.py           # Intrinsic motivation
│   │   ├── dual_process.py      # System 1 / System 2
│   │   └── ... (20 more modules)
│   ├── experiments/phases/      # Phase runners
│   └── tests/                   # Unit tests
├── experiments/                 # Cross-domain & Phase 4 experiments
├── tests/                       # ML framework tests (17 files)
├── scripts/                     # Analysis & profiling tools
├── docs/                        # Full documentation (12 files)
├── data/glove/                  # GloVe embeddings (download required)
└── results/                     # Benchmark outputs & diagnostics
```

---

## License

MIT — Built for the RAVANA-AGI-Research initiative.

---

## About

RAVANA explores an alternative paradigm for machine learning — where cognition emerges from **internal pressure**, not gradient optimization; where knowledge **self-organizes** through Hebbian plasticity and sleep consolidation; and where **identity, emotion, and meaning** are first-class citizens of the learning dynamics.

**The core thesis**: *"Cognition as pressure-driven self-organization."*

This is the center of gravity. Everything else is implementation detail.

---

## Citation

If you use RAVANA in research, please cite:

```bibtex
@misc{ravana2026,
  title={RAVANA: Pressure-Driven Self-Organization for Cognitive Machine Learning},
  author={RAVANA Research Team},
  year={2026},
  url={https://github.com/your-org/ravana}
}
```

---

## Links

- **Documentation**: [`docs/`](docs/)
- **Architecture Paper**: [`docs/RAVANA_REPORT.md`](docs/RAVANA_REPORT.md)
- **Academic Draft**: [`docs/PAPER_DRAFT.md`](docs/PAPER_DRAFT.md)
- **Gap Analysis**: [`docs/ANALYSIS_RLM_vs_LLM.md`](docs/ANALYSIS_RLM_vs_LLM.md)
- **External Audit**: [`docs/EXTERNAL_AUDIT.md`](docs/EXTERNAL_AUDIT.md)