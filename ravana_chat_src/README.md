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

## Two Architectures, One Codebase

RAVANA now contains **two complementary cognitive architectures**:

### 1. **RAVANA v2** — GRACE Cognitive Core (`ravana-v2/`)
The original GRACE (General Recursive Adaptive Cognitive Engine) with 27 phases (A–P) implementing:
- **Governor regulation** (Phases A–C) — Hard constraints, predictive dampening, center-seeking homeostasis
- **Strategy & Intent** (Phases D–J) — 4 strategy modes, planning, hypotheses, Occam layer
- **World Model & Epistemology** (Phases F–G.5) — Belief reasoning, active probes, VoI-driven action
- **Emotion & Sleep** (Phases K–L.5) — VAD emotion, empathy, 4-stage sleep, dual-process
- **Meaning & Consciousness** (Phases M–N) — Intrinsic motivation, global workspace
- **Human Memory & Dialogue** (Phases O–P) — Episodic/semantic persistence, conversational repair

### 2. **RAVANA Modular** — Decoder-First Chat (`ravana/`) **NEW**
A streamlined, package-oriented architecture focused on **continuous web learning** and **decoder-first generation**:
- **CognitiveCore** (`ravana/core/`) — VAD emotion, Identity, Meaning, Dual-process, Global Workspace, Meta-cognition, Sleep, **Multi-user BeliefStore**
- **GraphEngine** (`ravana/graph/`) — Concept graph, spreading activation, hippocampal indexing, curiosity scoring
- **DecoderEngine** (`ravana/decoder/`) — Neural decoder, vocab building, seed corpus + web article training
- **WebLearner** (`ravana/web/`) — Multi-API search (circuit breaker), background learning, curiosity-driven exploration (Phase 18)
- **BootstrapManager** (`ravana/bootstrap/`) — Consolidated concept seeding (3 methods unified)
- **RLMv2 Decomposed** (`ravana/nn/rlm/`) — RelationPredictor, PropagationEngine, Plasticity
- **ChatInterface** (`ravana/chat/`) — CLI, response pipeline, reasoning loop

---

## Key Innovations (Shared)

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

### 🏛️ GRACE Governor (v2 only)
Closed-loop regulation with four layers: hard constraints (non-negotiable), predictive dampening (look-ahead), boundary pressure (air resistance), center-seeking homeostasis. **Fully observable** with clamp diagnostics.

### 🎭 VAD Emotion Engine
3D affective state (Valence, Arousal, Dominance) via differential equations. Modulates inference: high arousal → exploration, positive valence → trust predictions, high dominance → stronger concepts.

### 🧩 RLMv2 — Triple Decomposition Architecture
Brain-inspired: decomposes input into **(subject, relation_type, object)** triples. Enables vector arithmetic analogy: `subject_embed + offset(verb) ≈ target_embed`. **Spreading activation is the sole inference mechanism**.

### 📖 GloVe Semantic Embeddings
Token embeddings initialized from pre-trained GloVe (100D) projected via QR-based orthogonal projection. Replaces character n-gram embeddings for genuine semantic relationships.

### 🎯 Verb-Stem Offset Predictor
New inference path: `offset(verb) = avg(target - subject)` over training pairs. Each verb gets its own offset. **RP-only cross-domain top-10: 3.3% → 6.7%**.

### 🌐 Continuous Web Learning (Modular only)
`WebLearner` fetches → extracts → trains decoder online. Knowledge grows **without retraining from scratch**. Curiosity-driven (free energy + contradiction + novelty + serendipity).

### 👥 Multi-User Belief Merging (Modular only)
`BeliefStore` tracks *who believes what*, detects contradictions, merges across users — `cross_reference_users()`, `find_agreement()`, `find_disagreement()`.

---

## Quick Start

### Installation

```bash
# Primary (Codeberg)
git clone https://codeberg.org/oxiverse/ravana.git
# Mirror (GitHub)
git clone https://github.com/oxiverse-ecosystem/ravana.git
cd ravana
pip install -e ravana/          # NumPy only (modular package)
pip install tiktoken            # Optional: BPE tokenizer
```

### Modular Package — Decoder-First Chat (NEW)

```bash
# Interactive chat with web learning
python -m ravana.chat

# Batch mode for testing
python -m ravana.chat --chat "hello|what is trust|explain oxiverse|bye" --strategy

# With ablation flags
python -m ravana.chat --chat "what causes rain" --no-vad --no-rlm --no-beliefs --no-curiosity
```

**Key flags:**
- `--no-vad` — Disable VAD emotion modulation
- `--no-rlm` — Disable RLMv2 triple verification
- `--no-beliefs` — Disable belief store
- `--no-curiosity` — Disable autonomous curiosity-driven learning
- `--mode` — `stochastic` | `deterministic` | `exploratory`
- `--trace` — Print chain traces
- `--data-dir` — Custom data directory
- `--user` — Multi-user isolation

### GRACE Cognitive Core (v2)

```python
from ravana_v2.cognitive import CognitiveFramework

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

### RLMv2 — Language Model (Shared)

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
    latent_dim=64,  # Set equal to embed_dim for entity adapter compatibility
)
inp = np.array(tok.encode("heat causes"), dtype=np.int64)
tgt = np.array(tok.encode("expansion"), dtype=np.int64)
model.learn(inp, tgt)
model.sleep_cycle()
logits = model.forward(inp)
```

**Note:** The `latent_dim` parameter controls the world model latent dimension. For the entity-specific adapter (enables test-time adaptation for held-out subjects) to work without additional projection overhead, set `latent_dim=embed_dim`. If they differ, the model automatically projects embeddings to latent space before applying the adapter.

---

## Benchmarks (External)

All benchmarks are run via external benchmarking infrastructure. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for reproduction instructions and expected results.

### External Benchmark Harness (NEW)

```bash
# Quick run (PCX + Graph, ~2 min)
python external_benchmark.py --quick

# Full suite (PCX + Lifelong + Graph, ~10 min)
python external_benchmark.py

# Individual surfaces
python external_benchmark.py --quick --skip-lifelong    # PCX + Graph
python external_benchmark.py --quick --skip-pcx        # Lifelong + Graph
```

**Key external validation results:**

| Metric | Result |
|--------|--------|
| Cross-domain transfer Top-1 | **75.0%** |
| Cross-domain transfer Top-10 | **100%** |
| Held-out Science Top-1 / Top-10 | 8.3% / 25.0% (n=12) |
| Held-out Social Top-1 / Top-10 | 0.0% / 8.3% (n=36) |
| Graph Inference P95 / P99 | 2.7 ms / 2.9 ms |
| Graph Peak Memory / Throughput | 0.3 MB / 556 QPS |
| W_rel Causal / Semantic Alignment | 0.68 / 0.55 |
| Lifelong forgetting (permuted MNIST) | **0%** (with sleep) |
| Within-domain triple top-10 | 80.9% |

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
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Running experiments — cross-domain, Phase 4, RLMv2 benchmarks, cognitive phases, **modular ablations, per-triple eval** |
| [`docs/ADVANCED_TOPICS.md`](docs/ADVANCED_TOPICS.md) | Customization — tokenizers, plasticity, sleep, governor, multi-agent, lifelong |
| [`docs/TUTORIALS.md`](docs/TUTORIALS.md) | 8 step-by-step tutorials — Hello World to visualization |
| [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) | Contributing — code standards, testing, architecture principles, release process |

---

## Running Tests

```bash
# Modular package tests (NEW)
python -m pytest tests/test_cognitive_rlm.py -v
python -m pytest tests/test_dialogue_system.py -v
python -m pytest tests/test_dialogue_engine_integration.py -v
python -m pytest tests/ -k "not slow" -v

# RLMv2 unit tests (shared)
python -m pytest tests/test_rlm_v2.py -v
python -m pytest tests/test_rp_only.py -v
python -m pytest tests/test_rp_contrastive.py -v
python -m pytest tests/test_structural_transfer.py -v

# GRACE cognitive core tests (v2)
python -m pytest ravana-v2/tests/ -v

# All tests
python -m pytest tests/ ravana-v2/tests/ -v
```

**Current status: 169 tests passing** (15 cognitive + 49 dialogue + 15 integration + ML framework tests)

---

## Running Experiments

### Modular Package Experiments (NEW)

```bash
# Ablation study — all config combinations
python scripts/run_ablation.py --queries "hello|what is trust|explain oxiverse|bye" --runs 3
python scripts/run_ablation.py --quick --modes  # Baseline + single flag removals + mode variants

# Per-triple evaluation harness
python scripts/triple_eval.py --data-dir ./data --output triple_eval_report.json
python scripts/triple_eval.py --triples-file my_triples.json --output report.json
```

### GRACE / ML Framework Experiments

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

### Custom Experiments

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md#custom-experiments) for templates and configuration reference.

---

## Project Structure

```
ravana/
├── README.md                           # This file
├── ravana/                             # Unified pip-installable package (NEW)
│   ├── __init__.py                     # import ravana
│   ├── core/                           # CognitiveCore
│   │   ├── emotion.py                  # VADEmotionEngine (VAD dynamics)
│   │   ├── identity.py                 # IdentityEngine (self-coherence)
│   │   ├── meaning.py                  # MeaningEngine (dissonance reduction)
│   │   ├── dual_process.py             # DualProcessController (System 1/2)
│   │   ├── global_workspace.py         # GlobalWorkspace (broadcast)
│   │   ├── meta_cognition.py           # MetaCognition (bias detection)
│   │   ├── sleep.py                    # SleepConsolidation (hippocampal replay)
│   │   ├── belief_store.py             # BeliefStore (multi-user merging)
│   │   └── __init__.py
│   ├── graph/                          # GraphEngine
│   │   ├── engine.py                   # Concept graph, spreading activation, hippocampal indexing
│   │   └── __init__.py                 # + backward compat: ConceptGraph, ConceptNode, etc.
│   ├── decoder/                        # DecoderEngine
│   │   ├── engine.py                   # Neural decoder, vocab, training, generation
│   │   └── __init__.py
│   ├── web/                            # WebLearner
│   │   ├── learner.py                  # Multi-API search, background learning, curiosity
│   │   └── __init__.py
│   ├── bootstrap/                      # BootstrapManager
│   │   ├── manager.py                  # Consolidated: auto_expand, seed_from_curiosity, bootstrap_domain
│   │   └── __init__.py
│   ├── nn/rlm/                         # RLMv2 Decomposed
│   │   ├── relation_predictor.py       # RelationPredictor (W_rel, verb-stem offset)
│   │   ├── propagation.py              # PropagationEngine (multi-phase spread)
│   │   ├── plasticity.py               # Plasticity (Hebbian/Anti-Hebbian/Structural)
│   │   └── __init__.py
│   ├── chat/                           # ChatInterface
│   │   ├── interface.py                # CLI, response pipeline, reasoning loop
│   │   └── __init__.py
│   └── pyproject.toml                  # numpy>=1.20
├── ravana_ml/                          # ML Framework (~5,200 lines)
│   ├── nn/rlm_v2.py                    # RLMv2 (triple decomposition, GloVe)
│   ├── graph.py                        # ConceptGraph (3,678 lines)
│   ├── tensor.py                       # StateTensor / RawTensor / Parameter
│   ├── plasticity.py                   # Hebbian, Anti-Hebbian, Structural
│   └── ... (tokenizer, embedder, free_energy, currencies, propagation)
├── ravana-v2/                          # GRACE Cognitive Core (~19,500 lines)
│   ├── core/                           # 27 Phases A–P modules
│   │   ├── governor.py                 # Central regulation
│   │   ├── identity.py                 # Momentum-based self-concept
│   │   ├── sleep.py                    # 4-stage SWS+REM
│   │   ├── emotion.py                  # VAD differential equations
│   │   ├── human_memory.py             # Episodic/Semantic (2,321 lines)
│   │   ├── global_workspace.py         # Consciousness bottleneck
│   │   ├── meaning.py                  # Intrinsic motivation
│   │   ├── dual_process.py             # System 1 / System 2
│   │   └── ... (20 more modules)
│   ├── experiments/phases/             # Phase runners
│   └── tests/                          # Unit tests
├── experiments/                        # Cross-domain & Phase 4 experiments (18 files)
├── scripts/                            # Analysis & evaluation tools
│   ├── run_ablation.py                 # Ablation study runner (NEW)
│   ├── triple_eval.py                  # Per-triple evaluation harness (NEW)
│   └── ...
├── tests/                              # ML framework tests (17 files)
├── docs/                               # Full documentation (12 files)
├── data/glove/                         # GloVe embeddings (download required)
├── results/                            # Benchmark outputs & diagnostics
├── ablation_results.json               # Ablation study outputs
├── triple_eval_report.json             # Per-triple diagnostic reports
└── RAVANA_DEV_PLAN.md                 # Development roadmap
```

---

## Ablation Study Framework (NEW)

The modular package includes a systematic ablation runner (`scripts/run_ablation.py`):

```bash
# Full factorial (4 flags × 3 modes = 48 configs)
python scripts/run_ablation.py --queries "hello|what is trust|explain oxiverse|bye" --runs 3

# Quick mode (baseline + 4 single removals + 2 mode variants = 7 configs)
python scripts/run_ablation.py --quick --modes

# Output: ablation_results.json + markdown comparison table
```

**What it tests:**
| Flag | Component |
|------|-----------|
| `--no-vad` | VAD emotion modulation |
| `--no-rlm` | RLMv2 triple verification |
| `--no-beliefs` | Multi-user belief store |
| `--no-curiosity` | Autonomous curiosity-driven learning |

**Metrics tracked:** Success rate, avg latency, total time, error modes per config.

---

## Per-Triple Evaluation Harness (NEW)

Detailed diagnostics instead of averaged metrics (`scripts/triple_eval.py`):

```bash
# Default 10 triples (semantic, causal, contrastive)
python scripts/triple_eval.py --data-dir ./data --output triple_eval_report.json

# Custom triples
python scripts/triple_eval.py --triples-file my_triples.json --output report.json
```

**Per-triple metrics:**
- **Relation type**: semantic / causal / contrastive / possessive / temporal / analogical
- **Prediction error (PE)**: how well the model predicts the object
- **Confidence**: model's confidence in the prediction
- **Top-1 / Top-5 accuracy**: exact match / in top 5
- **Rank**: position of correct object in predictions
- **Graph edge**: weight, confidence, type, `prediction_free_energy`
- **Source**: where learned — `seed` | `web` | `user` | `sleep` | `inference` | `unknown`

**Aggregated diagnostics:**
- Overall metrics + by relation type + by source
- Edge health summary (avg weight, confidence, PE, type distribution)

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
  url={https://codeberg.org/oxiverse/ravana}
}
```

---

## Links

- **Documentation**: [`docs/`](docs/)
- **External Audit**: [`docs/EXTERNAL_AUDIT.md`](docs/EXTERNAL_AUDIT.md)