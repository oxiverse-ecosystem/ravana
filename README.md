# RAVANA — Recursive Learning Model

> **A pressure-driven cognitive ML framework where learning emerges from self-organization, not gradient descent.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-orange)](https://numpy.org)
[![License](https://img.shields.io/badge/License-OCL%201.0-purple)](LICENSE)
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

### 🎯 Verb-Stem Offset Predictor + Test-Time Adapter
Two inference paths: (1) **Verb offset**: `offset(verb) = avg(target - subject)` over training pairs — enables vector arithmetic analogy `subject_embed + offset(verb) ≈ target_embed`. (2) **Bilinear W_rel fallback** for unseen verbs. For held-out subjects, an **entity-specific adapter (U, V rank-16 matrices)** is initialized from nearest neighbor and adapted at test time via cross-entropy minimization over all tokens (temperature=0.1, 30 steps). This directly optimizes the ranking metric, recovering held-out generalization from **5-12% to 93-100% Top-1** and cross-domain transfer to **100%**.

### 🌐 Continuous Web Learning (Modular only)
`WebLearner` fetches → extracts → trains decoder online. Knowledge grows **without retraining from scratch**. Curiosity-driven (free energy + contradiction + novelty + serendipity).

### 👥 Multi-User Belief Merging (Modular only)
`BeliefStore` tracks *who believes what*, detects contradictions, merges across users — `cross_reference_users()`, `find_agreement()`, `find_disagreement()`.

### 🧠 Theory of Mind & Emotional Mirroring (Modular only)
`UserModel` tracks user-specific goals (`LEARNING`/`DEBUGGING`/`EXPLORING`), preferences, emotional state (VAD), and relationship depth. User arousal modulates generation temperature, verbosity, and concept breadth — creating adaptive, personalized dialogue.

### 🔬 Validation & Benchmark Scripts
- `scripts/validate_held_out_generalization.py` — Validates verb-offset blending, confidence-weighted blending, and prototype inheritance
- `scripts/benchmark_vs_transformers.py` — Discriminative benchmark comparing RLMv2 against PyTorch baselines
- `scripts/test_emotional_mirror.py` — VAD emotional tracking tests
- `scripts/test_theory_of_mind.py` — ToM goal inference & personalization tests

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

## Benchmarks

### Cross-Domain Transfer (RLMv2 — clean holdout evaluation)

| Metric | Result |
|--------|--------|
| Cross-domain transfer Top-1 | **100%** (6/6) |
| Cross-domain transfer Top-10 | **100%** (6/6) |
| Held-out Science Top-1 (adapted) | **93.8%** (n=16) |
| Held-out Social Top-1 (adapted) | **85–100%** (n=20) |
| Held-out Science Top-1 (baseline) | 12.5% (n=16) |
| Held-out Social Top-1 (baseline) | 5.0% (n=20) |
| W_rel Causal Alignment | 0.38 (clean data ceiling) |
| Lifelong forgetting (A→B→C sequential) | **0.167** (0.667→0.500) |
| Sleep-time interleaved replay retention | **0% drop** |

Cross-domain transfer uses Science verbs + Social subjects with test-time cross-entropy adaptation (T=0.1, 30 steps). Held-out subjects use entity-specific adapters (rank=16 U,V matrices) with cosine distance or cross-entropy adaptation. Numbers are from **clean evaluation** — held-out targets never appear during training (fixes prior contamination in earlier benchmarks).

### Graph Scaling (Measured, Not Extrapolated)

| Nodes | `find_similar` p50 / p95 | `spread_activation` | Consolidation (full / incremental) |
|-------|--------------------------|---------------------|-------------------------------------|
| 1K    | **0.021 ms / 0.058 ms**  | 0.78 ms             | 1.82 ms / 0.20 ms |
| 5K    | **0.044 ms / 0.111 ms**  | —                   | — |
| 10K   | **0.059 ms / 0.081 ms**  | 1.15 ms             | 27.98 ms / 2.14 ms |
| 50K   | **0.66 ms**              | —                   | 49.20 ms / 3.12 ms (15.8× speedup) |

- **FAISS**: HNSWFlat index auto-activated at ≥64 nodes; switches from O(N·D) brute-force to O(log N) approximate search
- **Incremental consolidation** processes only changed nodes rather than full graph rebuild — 15.8× faster at 50K nodes
- **Memory at scale**: 0.3 MB peak (graph-only), 556 QPS throughput at 10K nodes
- **P95/P99 inference latency**: 2.7 ms / 2.9 ms

### Memory & Forgetting

| Metric | Result |
|--------|--------|
| False-positive intrusions | **avg 0.1 / fact** across 3 disjoint domains (animals, colors, fruits) |
| Cross-domain false positives | **zero** — max intruder score 0.00 vs correct -23.44 |
| Recall@1 at 10 facts | **1.000** |
| Recall@1 at 60 facts | **0.917** |
| Forgetting of A after B+C | **0.167** (0.667 → 0.500) |
| Rare (1×) training recall | **1.000** — no forgetting of infrequent facts |

### Curse of Compression (What Sleep Prunes)

Sleep consolidation prunes what doesn't generalize:

- **Polluted edges** — coincidental co-activations that sleep detects via consistently high prediction error; Anti-Hebbian plasticity removes them
- **Phantom nodes** — degree < 2 (no useful prediction path); structural plasticity removes them
- **Low-utility episodic traces** — details with high retrieval distortion or low predictive utility don't consolidate into semantic memory
- **Low-confidence edges** — edges with confidence < 0.05 and zero prediction count are pruned during homeostatic downscale

### False Positive Probe

Across 3 disjoint domains (animals, colors, fruits), cross-domain intrusions in top-10: **6.17 avg** (lower is better; < 5 is good, 0 perfect).

### External Benchmark Harness

```bash
# Quick run (PCX + Graph, ~2 min)
python scripts/external_benchmark.py --quick

# Full suite (PCX + Lifelong + Graph, ~10 min)
python scripts/external_benchmark.py

# Individual surfaces
python scripts/external_benchmark.py --quick --skip-lifelong    # PCX + Graph
python scripts/external_benchmark.py --quick --skip-pcx        # Lifelong + Graph
```

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
# CI-critical tests (fast, ~0.5s)
python -m pytest tests/ci/ -v

# Unit tests (~2.5 min)
python -m pytest tests/unit/ -v

# Integration tests (~4 min)
python -m pytest tests/integration/ -v

# GRACE cognitive core tests (v2)
python -m pytest ravana-v2/tests/ -v

# All tests
python -m pytest tests/ ravana-v2/tests/ -v
```

**Current status: 1456+ tests passing** (30 CI + 1310 unit + 95 integration + 16 GRACE + 5 generation)

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
│   ├── agent/                          # Agent infrastructure
│   │   ├── mode_orchestrator.py        #   ModeOrchestrator (RESEARCH/INTERVIEW/LEARN)
│   │   └── version_manager.py          #   VersionManager (SQLite persistence)
│   ├── probes/                         # Governor diagnostic probes
│   │   ├── exploration_pressure.py     #   Probe 1: Boundedness under chaos
│   │   ├── constraint_stress.py        #   Probe 2: Active regulation vs passive clipping
│   │   └── learning_signal.py          #   Probe 3: Learning vs stagnation
│   ├── training/                       # Training pipeline
│   │   └── pipeline.py                 #   Governor-gated training loop
│   ├── experiments/phases/             # Phase runners
│   └── tests/                          # Unit tests
├── experiments/                        # Cross-domain & Phase 4 experiments (18 files)
├── scripts/                            # Analysis & evaluation tools
│   ├── run_ablation.py                 # Ablation study runner (NEW)
│   ├── triple_eval.py                  # Per-triple evaluation harness (NEW)
│   └── ...
├── tests/                              # ML framework tests (17 files)
├── docs/                               # Full documentation (13 files)
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

## How to Use This Project

RAVANA is released under the **Oxiverse Community License (OCL) v1.0** — source-available, non-commercial, privacy-by-design.

### Permitted (Non-Commercial Use)
- **Use**: Run RAVANA for personal, academic, or research purposes.
- **Modify**: Fork, change, and adapt the code.
- **Distribute**: Share copies of the original or modified code, as long as this license is included.
- **Commercial (Open-Source Path)**: Monetize your project if you keep all derivatives under OCL v1.0, publish complete source code, and comply with the Privacy-by-Design requirement (Section 5).

### Not Permitted Without a Commercial License
- **Closed-source commercial use** (SaaS, paid APIs, proprietary products, consulting built directly on RAVANA).
- **Removing or obscuring** the copyright and license notice.
- **Privacy violations**: Adding tracking, telemetry, or data collection without explicit opt-in consent.

### Commercial Licensing
- **Open-Source Commercial License (Free)**: Distribute under OCL v1.0 with full source code — no fee required.
- **Proprietary Commercial License (Paid)**: Closed-source deployment, white-labeling, or SLA — contact likhith@oxiverse.com.

### Why OCL?
OCL protects the open research nature of RAVANA while allowing sustainable commercial adoption. The Privacy-by-Design requirement ensures user data remains under user control.

---

## License

OCL v1.0 (Oxiverse Community License) — Source-Available, Non-Commercial, Privacy-by-Design. See [LICENSE](LICENSE) for the full text.

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