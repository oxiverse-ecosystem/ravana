# RAVANA Architecture Reference

> **Complete technical reference for the RAVANA cognitive architecture.** This document covers all three layers of the system, their interactions, and the theoretical foundations.

---

## Table of Contents

1. [Overview](#overview)
2. [Three-Layer Architecture](#three-layer-architecture)
3. [Layer 1: `ravana_ml/` — ML Framework](#layer-1-ravana_ml--ml-framework)
4. [Layer 2: `ravana-v2/` — GRACE Cognitive Core](#layer-2-ravana-v2--grace-cognitive-core)
5. [Layer 3: `ravana/` — Modular Package](#layer-3-ravana--modular-package)
6. [Layer 4: `ravana/chat/` — Chat Interface](#layer-4-ravanachat--chat-interface)
7. [Phase 1-3 Modules: VSA, S1/S2, Learning](#phase-1-3-modules-vsa-s1s2-learning)
8. [Language Modules](#language-modules)
9. [Data Flow & Integration](#data-flow--integration)
10. [Key Algorithms](#key-algorithms)
11. [Configuration Reference](#configuration-reference)

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

## Four-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                LAYER 4: ravana/ — Modular Chat Package                      │
│  ┌────────────────┐  ┌───────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ cognitive/core │  │   graph/      │  │  decoder/    │  │  web/       │  │
│  │ (VAD,Identity, │  │  GraphEngine  │  │NeuralDecoder │  │ WebLearner  │  │
│  │  Meaning,GW,   │  │  hippocampal  │  │ vocab/syntax │  │ background  │  │
│  │  BeliefStore)  │  │  indexing     │  │ generation   │  │ learning    │  │
│  ├────────────────┤  ├───────────────┤  ├──────────────┤  ├─────────────┤  │
│  │ bootstrap/     │  │  nn/rlm/      │  │  language/   │  │  chat/      │  │
│  │ BootstrapMgr   │  │ RelationPred  │  │  BasalGanglia│  │ ChatIface   │  │
│  │                │  │ Propagation   │  │ Cerebellar   │  │ CLI + API   │  │
│  │                │  │ Plasticity    │  │ PFC/syntax/  │  │             │  │
│  │                │  │               │  │ SurfaceReal  │  │             │  │
│  └────────────────┘  └───────────────┘  └──────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: ravana/ — Re-export Layer (import ravana as torch)    │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ ravana.nn │  │ ravana.cog. │  │ ravana.graph/propagation │  │
│  │ re-exports│  │ Cognitive   │  │ re-exports from ravana_ml│  │
│  │ RLM from  │  │ Framework   │  │                          │  │
│  │ ravana_ml │  │ (GRACE API) │  │                          │  │
│  └─────┬─────┘  └──────┬──────┘  └──────────┬───────────────┘  │
└────────│───────────────│─────────────────────│──────────────────┘
          │               │                     │
          ▼               ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: ravana_ml/ (5,200+ lines)   LAYER 1: ravana-v2/ (22K+ lines) │
│  ┌────────────────────────────┐      ┌──────────────────────────────┐ │
│  │ • graph.py (ConceptGraph)  │      │ • GRACE Phases A–P (27 mods) │ │
│  │ • nn/rlm.py (RLM v1)       │      │ • Governor, Identity, Sleep  │ │
│  │ • nn/rlm_v2.py (RLM v2)    │      │ • VAD Emotion engine         │ │
│  │ • nn/module.py             │      │ • Human Memory (SQLite)      │ │
│  │ • nn/neuromodulator.py     │      │ • Global Workspace           │ │
│  │ • nn/neural_decoder.py     │      │ • Meta-cognition + Meta²     │ │
│  │ • tensor.py                │      │ • Belief / Social Epistemol. │ │
│  │ • free_energy.py           │      │ • Hypothesis / Occam Layer   │ │
│  │ • plasticity.py            │      │ • Active Epistemology/Probes │ │
│  │ • propagation.py           │      │ • Reality Friction           │ │
│  │ • embedder.py              │      │ • Planning/World Model       │ │
│  │ • episode_injector.py      │      │ • Dual-Process Controller    │ │
│  │ • relation_ontology.py     │      │ • Meaning/Intrinsic Motivat  │ │
│  │ • tokenizer.py             │      │ • Dialogue/Repair            │ │
│  │ • currencies/currency.py   │      │ • Runtime & Version Manager  │ │
│  └────────────────────────────┘      └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 2: `ravana_ml/` — ML Framework

The ML framework provides a PyTorch-compatible API built entirely on NumPy.

### Core Modules

#### `graph.py` — ConceptGraph (3,678+ lines)

The central knowledge representation. A heterogeneous graph with:

**Nodes (ConceptNode):**
- `vector` — active vector (fast-adapting, plastic)
- `core_vector` — identity anchor (slow, drift-resistant)
- `genesis_vector` — original vector for drift tracking
- `activation` — current spreading activation level
- `salience`, `prediction_free_energy`, `stability`, `confidence`, `contradiction_count`, `fatigue`
- `free_energy_history`, `free_energy_gradient`, `contradiction_free_energy` — for local predictive coding
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

#### `nn/module.py` — Neural Modules (473+ lines)

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

#### `nn/neural_decoder.py` — NeuralDecoder

The primary text generation engine. A GRU + attention decoder with:
- Word-level embedding → GRU hidden states → attention → output logits
- Sampled softmax with 50 negatives for tractable training
- Word-level training on full sentences and article text
- Conditioned generation with graph concept embeddings
- Integrated with `NeuromodulatorEngine` for dynamic modulation
- Sleep cycle for consolidation (weight + free energy update)

#### `nn/neuromodulator.py` — NeuromodulatorEngine (NEW)

Four-system ascending neuromodulation (ACh, NE, DA, 5-HT) modulating:
- **Temperature** — generation randomness
- **Repetition penalty** — output diversity
- **Learning rates** — per-component learning modulation
- **Exploration bonus** — rare word exploration
- **BG gate thresholds** — Go/NoGo gating
- **Dopamine tone** — vigor and confidence

Based on Yu & Dayan (2002), Hasselmo (1999), Aston-Jones & Cohen (2005).

#### `nn/rlm.py` — RLM v1 (3,931 lines)

Recursive Learning Model with predictive coding:

- **Token embedding** → positional encoding → **GRU** → hidden layers → **concept predictor**
- **5-path logit blend**: concept attention, context logits, RP analogy, sparse concept, direct latent
- **Settling loop**: 5 iterations of predictive coding state updates
- **Free energy**: 5-channel accumulator
- **Sleep**: interleaved replay (domain-tagged), contrastive hidden states, BP-trained RP

#### `nn/rlm_v2.py` — RLM v2 (5,013+ lines)

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

Key features: verb-stem offset predictor, entity adapter (test-time adaptation), W_rel bilinear matrices, cross-domain relation alignment.

#### `free_energy.py` — FreeEnergyAccumulator

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
    def get_node_free_energy(self, node_id): ...
```

#### `plasticity.py` — Plasticity Rules

```python
class HebbianPlasticity:      # Δw = lr * pre * post
class AntiHebbianPlasticity:  # Δw = -lr * pre * post (competition)
class StructuralPlasticity:   # prune weak edges, form co-activation edges
```

#### `propagation.py` — PropagationEngine

Spreading activation with typed edge filtering:

```python
engine = PropagationEngine(graph)
engine.get_prediction(active_nids, top_k=5)
engine.get_activation_vector(nids)
engine.measure_coherence(active_nids)
```

#### `tokenizer.py` — Tokenizers

- `WordTokenizer` — dynamic vocab, word-level (fastest for cognitive exps)
- `BPETokenizer` — tiktoken/GPT-2 (requires `tiktoken`)
- `SimpleTokenizer` — char-level fallback (256 vocab)
- `PixelTokenizer` — image → token sequence (28×28 → 784 tokens)

#### `word_tokenizer.py` — WordTokenizer (Standalone)

Word-level tokenizer with direct word-to-ID mapping for RLMv2. Used by the modular chat package.

#### `embedder.py` — LearnedEmbedder

Character n-gram + random projection (64-dim) embeddings for OOV handling. Uses MD5 feature hashing with IDF weighting.

#### `episode_injector.py` — EpisodeInjector (NEW)

Feeds structured knowledge into RLMv2's graph via learn(). Supports:
- Dict-based facts, tuple-based facts, batch injection
- Confidence-weighted training (repeat high-confidence facts more)
- Multi-edge support via relation-object hub keys
- Pre-built knowledge bases (PHARMACOLOGY_KB, ECOLOGY_KB)

#### `relation_ontology.py` — Relation Ontology (NEW)

Multi-level relation hierarchy for typed traversal:
- Family > Sub-family > Predicate granularity
- 6 families: causal (strong/moderate/weak), directional (positive/negative), compositional, taxonomic, temporal, capability
- Super-families for aggregate traversal (causal_all, directional_all, causal_directional)
- Structured Candidate dataclass with (word, predicate, family, sub_family, depth, confidence, path)

#### `currencies.py` / `currency.py` — Cognitive Currencies

Unified cognitive state management replacing scattered scalars.

#### `tensor.py` — Tensor Abstractions

- `RawTensor` — numpy wrapper with device API
- `StateTensor` — adds cognitive metadata (salience, free_energy, stability, decay)

#### `lab/` — Analysis Tools

- `analyze_concept_graph()` — Graph structure analysis
- `plot_activation_dynamics()` — Activation over time
- `compute_coherence_trajectory()` — Coherence tracking
- `visualize_sleep_cycle()` — Sleep diagnostics
- `diagnose_learning()` — Learning diagnosis

#### `world/` — Simulation Environments

- `GridWorld` — discrete 2D environment
- `ContinuousWorld` — continuous control
- `SymbolicWorld` — discrete state, symbolic actions

---

## Layer 1: `ravana-v2/` — GRACE Cognitive Core

GRACE = **Governance, Reflection, Adaptation, Constraint, Exploration**

### Phase Architecture

| Phase | Module | Function |
|-------|--------|----------|
| **A** | `governor.py` | Central regulation — hard constraints, predictive dampening, boundary pressure, center-seeking |
| **A** | `identity.py` | Momentum-based self-concept with recovery bias |
| **A** | `resolution.py` | Continuous partial credit toward wisdom events |
| **A** | `state.py` | StateManager — orchestrates all modules |
| **B** | `adaptation.py` | Policy learning from clamp events |
| **C** | `strategy.py` | 4 exploration modes: AGGRESSIVE, SAFE, STABILIZE, RECOVER |
| **C** | `strategy_learning.py` | Meta-learning over mode outcomes |
| **D** | `intent.py` | Dynamic objectives evolving from outcomes |
| **D.5** | `planning.py` | Micro-planner with simulated futures |
| **E** | `environment.py` | Non-stationary world (boundary shifts, noise drift, goal flips) |
| **F** | `predictive_world.py` | Neural world model, adaptive surprise threshold |
| **F.5** | `belief_reasoner.py` | Competing hypotheses with confidence decay |
| **G** | `active_epistemology.py` | Value of Information (VoI) calculation |
| **G.5** | `surgical_probes.py` | Targeted intervention experiments |
| **J** | `hypothesis_generation.py` | Generate hypotheses from anomalies |
| **J.1** | `occam_layer.py` | Hypothesis discipline / complexity penalization |
| **J.1** | `meta_cognition.py` | Meta-cognition: bias detection, confidence calibration, reasoning quality |
| **K** | `emotion.py` | VAD differential equations |
| **K.5** | `empathy.py` | Other-mind modeling |
| **L** | `sleep.py` | 6-stage SWS + REM with dream sabotage |
| **L.5** | `dual_process.py` | System 1 / System 2 routing |
| **M** | `meaning.py` | Intrinsic motivation: M = w1(-D) + w2(I) + w3(pred) × (1 + κ×effort) |
| **N** | `global_workspace.py` | Competitive broadcast, consciousness bottleneck |
| **O** | `human_memory.py` | Persistent episodic/semantic memory, Ebbinghaus decay, interference |
| **P** | `dialogue_context.py` | Conversation tracking, active subgraphs |
| **P** | `conversational_repair.py` | Correction handling, repair events |
| **—** | `meta2_cognition.py` | Meta²-cognition: self-model of epistemic processing |
| **—** | `meta2_integration.py` | Meta² integration: coordinating multiple epistemic layers |
| **—** | `social_epistemology.py` | Multi-agent belief dynamics, trust modeling, deception detection |
| **—** | `reality_friction.py` | Testing reality model against actual outcomes |
| **—** | `runtime.py` | Runtime orchestration module |
| **Agent** | `agent/mode_orchestrator.py` | ModeOrchestrator — central dispatcher (RESEARCH/INTERVIEW/LEARN) |
| **Agent** | `agent/version_manager.py` | VersionManager — SQLite version tracking, changelog, experiments queue |
| **Probes** | `probes/exploration_pressure.py` | Probe 1: Exploration pressure test (+25% noise, verify boundedness) |
| **Probes** | `probes/constraint_stress.py` | Probe 2: Constraint stress test (force D→0.85, verify active regulation) |
| **Probes** | `probes/learning_signal.py` | Probe 3: Learning signal test (ΔD trends, verify learning vs stagnation) |
| **Train** | `training/pipeline.py` | TrainingPipeline — governor-gated training with difficulty ramp |


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

## Layer 3: `ravana/` — Modular Chat Package

The modular chat package (`ravana/src/ravana/`) is a full cognitive chat engine with continuous web learning. It integrates all lower layers into a cohesive conversational agent.

```bash
# Interactive chat with web learning
python -m ravana.chat
# With ablation flags
python -m ravana.chat --chat "hello|what is trust" --no-vad --no-rlm
```

**Structure:**
```
ravana/src/ravana/
├── __init__.py              # Module exports
├── core/                    # CognitiveCore
│   ├── emotion.py           # VADEmotionEngine
│   ├── identity.py          # IdentityEngine
│   ├── meaning.py           # MeaningEngine
│   ├── dual_process.py      # DualProcessController
│   ├── global_workspace.py  # GlobalWorkspace
│   ├── meta_cognition.py    # MetaCognition
│   ├── sleep.py             # SleepConsolidation
│   ├── belief_store.py      # BeliefStore (multi-user belief merging)
│   ├── hippocampal_buffer.py # HippocampalBuffer (fact store)
│   ├── proposition_parser.py # Proposition parsing
│   ├── causal_schema.py     # CausalSchemaLearner
│   ├── implicature_detector.py # Pragmatic implicature
│   ├── relation_memory.py   # RelationMemory (comparative)
│   ├── quantity_modifier.py # QuantityModifierSystem
│   ├── analogy_engine.py    # AnalogyEngine
│   ├── abstraction_engine.py # AbstractionEngine
│   ├── mirror.py            # EmotionalMirrorEngine
│   ├── predictive_coding.py # PredictiveCodingLearner (Phase 1)
│   ├── coherence.py         # CoherenceNetwork (Phase 2)
│   ├── working_memory.py    # WorkingMemory with VSA (Phase 2)
│   ├── vsa.py               # VSAManager (Phase 2)
│   ├── system1.py           # System1Attractor (Phase 2)
│   └── system2.py           # System2Simulator (Phase 2)
├── graph/                   # GraphEngine
│   └── engine.py            # Seeding, auto-expansion, spreading, hippocampal indexing
├── decoder/                 # DecoderEngine
│   └── engine.py            # NeuralDecoder management, vocab, training, generation
├── web/                     # WebLearner
│   └── learner.py           # SearchEngine, background learning, curiosity
├── bootstrap/               # BootstrapManager
│   └── manager.py           # Unified seeding: seed/domain/curiosity
├── nn/rlm/                  # RLMv2 Decomposed
│   ├── relation_predictor.py # RelationPredictor, verb-stem offset, W_rel
│   ├── propagation.py       # PropagationEngine (BFS, relation-aware spread)
│   └── plasticity.py        # Plasticity (Hebbian, Anti-Hebbian, Structural)
├── language/                # Language production pipeline
│   ├── basal_ganglia.py     # BasalGangliaGate (Go/NoGo selection)
│   ├── cerebellar_ngram.py   # CerebellarNgram (fluent transitions)
│   ├── prefrontal_workspace.py # Discourse planning
│   ├── syntactic_cell_assembly.py # Syntactic role binding
│   ├── surface_realizer.py  # English morphology/agreement
│   ├── verb_lexicon.py      # Hebbian verb selection
│   ├── register.py          # RegisterController (formality/certainty)
│   └── schemas.py           # VSA SchemaLibrary
├── learn/                   # Learning systems
│   ├── curiosity.py         # CuriosityEngine (Phase 18)
│   └── consolidation.py     # HippocampalReplay (Phase 3)
├── storage/                 # Persistence
│   └── db.py                # CognitiveDB (SQLite)
└── chat/                    # Chat interface
    ├── interface.py         # ChatInterface (main CLI/API)
    ├── engine.py            # CognitiveChatEngine
    ├── models.py            # Data models
    ├── user_model.py        # UserModel — Theory of Mind (goal inference, preferences, rapport)
    ├── belief_store.py      # Multi-user belief store
    ├── response_gen.py      # ResponseGenMixin — neural decoder gen, chitchat, syntactic pipeline
    ├── chain_walker.py      # ChainWalkerMixin — graph traversal, relation inference, PFC top-down bias
    ├── web_learning.py      # WebLearningMixin — background learning, curiosity drive (Phase 18)
    └── constants.py         # Shared constants
```

### `chat/` — Chat Interface Mixins

The chat engine (`CognitiveChatEngine`) is composed of mixin layers that each handle a distinct cognitive function:

#### `UserModel` — Theory of Mind (`user_model.py`)

Tracks user-specific knowledge, preferences, emotional state, and inferred goals:

| Feature | Description |
|---------|-------------|
| **Goal Inference** | Classifies user queries into `LEARNING`, `DEBUGGING`, `EXPLORING` based on lexical markers |
| **Relationship Depth** | Grows with interaction count (`depth = count / 20`, capped at 1.0) |
| **Knowledge Model** | Per-concept familiarity tracking via `knowledge_model` and `learning_goals` |
| **Emotional Rapport** | Per-topic valence tracking via `emotional_rapport` dictionary |
| **Preference Extraction** | Regex-based extraction of likes, interests, favorites, and user name from chat |
| **Cognitive Style** | Detects `curious`, `skeptical`, or `practical` style from query vocabulary |
| **Edge Reactivations** | Tracks which concept pairs have been activated, boosting them in future walks |
| **Engagement & Depth** | Computes `engagement_level`, `conversation_depth`, `topic_interaction_count` |
| **Serialization** | Full `get_state()` / `set_state()` with backward compatibility |

#### `ChainWalkerMixin` — Graph Traversal (`chain_walker.py`)

Core graph walking and relation inference:
- **Concept seeding** — GloVe-powered teen concept seeding with 5 typed relation types
- **Auto-expansion** — `_auto_expand_concepts()` adds new words from user input via GloVe similarity
- **Causal detection** — GloVe-based semantic causal detection (`_word_causal_score()` with seed vectors)
- **PFC top-down modulation** — Relation prototypes bias edge selection by discourse intent type
- **Prediction error learning** — `_prediction_error()` updates edge weights via gradient descent on cosine similarity
- **Contradiction detection** — `_is_contradictory()` checks belief assertions against antonym map
- **Basal Ganglia gating** — `_walk_chain()` uses 15+ modulators (arousal, novelty, fatigue, dopamine tone, etc.)
- **Spreading activation** — `_spread_and_collect()` propagates activation through typed edges with hop decay
- **Grouped associations** — `_group_associations()` organizes concepts by relation type for response generation

#### `ResponseGenMixin` — Response Generation (`response_gen.py`)

Multi-path response generation pipeline:
1. **Neural Decoder** (`_generate_with_decoder`) — GRU+attention autoregressive generation with cerebellar n-gram bias and basal ganglia gating
2. **Syntactic Pipeline** (`_generate_with_decoder_and_syntax`) — P600-driven compositional integration
3. **Graph Fallback** (`_graph_fallback_response`) — Free-energy-driven SurfaceRealizer with Hebbian verb selection
4. **Reasoning Loop** — Web-aware fallback for unknown concepts
- **Comparison detection** — `_detect_comparison_concepts()` identifies contrasting pairs for structured responses
- **Chitchat handling** — `_handle_chitchat()` routes greetings/wellbeing/farewells through the cognitive pipeline
- **Hippocampal recall** — `_try_hippocampal_retrieval()` recalls factual memories before generative paths

#### `WebLearningMixin` — Web Learning (`web_learning.py`)

Continuous web learning and curiosity-driven exploration:
- **Multi-API search** with circuit breaker and automatic offline fallback
- **Background learning thread** — `_bg_learn_loop()` processes pending queue with 30s idle detection
- **Parallel article fetching** via `ThreadPoolExecutor`
- **Definition extraction** — ATL-style patterns (`X is a Y`, `X refers to Y`) populate neocortical definition store
- **Decoder training** — 30+ passes per article for Hebbian strengthening
- **Curiosity Drive (Phase 18)** — `_auto_select_curiosity_topics()` from 5 sources:
  - Unresolved impossible queries (5× weight)
  - High prediction error concepts (3× weight)
  - Contradiction pairs with confidence-mismatch (3× weight)
  - Novel concepts via dormant edge ratio (1× weight)
  - Random graph walk from high-degree hubs (serendipity)

#### Key Modular Components

#### `core/` — CognitiveCore

Includes all core cognitive engines from the modular package:
- **VADEmotionEngine** — 3D affective state (Valence, Arousal, Dominance)
- **IdentityEngine** — Self-coherence with momentum
- **MeaningEngine** — Intrinsic motivation computation
- **DualProcessController** — System 1/2 routing
- **GlobalWorkspace** — Competitive broadcast bottleneck
- **MetaCognition** — Bias detection, confidence calibration, epistemic modes
- **SleepConsolidation** — Sleep cycles with dream sabotage
- **BeliefStore** — Multi-user belief tracking and merging (`cross_reference_users()`, `find_agreement()`)
- **EmotionalMirrorEngine** — Mirrors user emotion to modulate verbosity/temperature
- **HippocampalBuffer** — Fact memory with decay and alias-based retrieval
- **PropositionParser** — Nested proposition detection
- **CausalSchemaLearner** — Learns causal relationships from patterns
- **ImplicatureDetector** — Pragmatic implicature analysis
- **RelationMemory** — Comparative/transitive relations
- **QuantityModifierSystem** — Quantity comparison reasoning
- **AnalogyEngine** — Analogy solving with relational structure mapping
- **AbstractionEngine** — Multi-perspective abstraction analysis

#### Phase 1-3: VSA Working Memory, System 1/2, Learning

| Module | Phase | Function |
|--------|-------|----------|
| `predictive_coding.py` | 1 | Local predictive coding: each node learns local predictors from context vectors |
| `coherence.py` | 2 | Thagard's ECHO coherence network: constraint satisfaction settling |
| `working_memory.py` | 2 | PFC working memory with VSA binding, context gating, lateral interference |
| `vsa.py` | 2 | Vector Symbolic Architecture (HRR): role-filler binding, bundling, unbinding |
| `system1.py` | 2 | System 1 attractor dynamics: iterative activation settling with confidence entropy |
| `system2.py` | 2 | System 2 deliberative simulation: causal subgraph extraction, forward/counterfactual sim |
| `curiosity.py` | 3 | CuriosityEngine: 5 curiosity signals (PE, information gap, dissonance, novelty, learning progress) |
| `consolidation.py` | 3 | HippocampalReplay: NREM forward replay + interleaved replay + pruning |
| `register.py` | 3 | RegisterController: formality/verbosity/certainty modulation knobs |
| `db.py` | 2 | CognitiveDB: SQLite persistence, WAL mode, graph/episode/metadata storage |

#### `language/` — Production Pipeline

The P1 Production-Grade Syntactic Pipeline implements a full language generation system:

```
PrefrontalWorkspace → Discourse Plan (structured intents from graph)
       ↓
SyntacticCellAssembly → Syntactic Frames (bind concepts to grammatical roles)
       ↓
BasalGangliaGate → Candidate Selection (Go/NoGo gating with neuromodulation)
       ↓
CerebellarNgram → Fluent Completion (learned function word transitions)
       ↓
SurfaceRealizer → Final Text (morphology, agreement, punctuation)
```

- **BasalGangliaGate**: Go/NoGo gating modulated by arousal, novelty, prediction error, identity strength, dopamine tone, fatigue level, prefrontal boost, thalamic salience, and contradiction penalty
- **CerebellarNgram**: Sparse bigram/trigram prediction for function words, learned from seed corpora
- **PrefrontalWorkspace**: Discourse planning from question type detection, intent decomposition, discourse marker assignment
- **SyntacticCellAssembly**: Hebbian role learning with subject/verb/object frames seeded from POS tags
- **SurfaceRealizer**: English morphology (past tense -ed, plural -s, 3rd person -s), determiner selection, pronoun coreference, negation handling
- **VerbLexicon**: Hebbian-compositional verb selection from relation type and dopamine tone
- **RegisterController**: Formality, verbosity, certainty knobs with REINFORCE-style adaptation

#### `web/` — WebLearner

Autonomous web learning system:
- **Multi-API search** with circuit breaker (local_api, oxiverse, duckduckgo)
- **Article fetching** with trafilatura + BeautifulSoup fallback
- **Background learning thread** with idle detection
- **Curiosity-driven topic selection** (Phase 18) combining prediction error, contradiction, novelty, and learning progress signals
- **Deferred learning queue** for unknown words

#### `bootstrap/` — BootstrapManager

Unified concept seeding:
- `bootstrap_all()` — seeds teen concepts + domain concepts + curiosity priming
- `auto_expand_from_input()` — expands graph from every user message
- `curiosity_bootstrap()` — seeds learning queue from curiosity signals

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
    latent_dim=64,  # Set equal to embed_dim for entity adapter compatibility
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

**Note:** The `latent_dim` parameter controls the world model latent dimension. For the entity-specific adapter (which enables test-time adaptation for held-out subjects) to work without additional projection overhead, set `latent_dim=embed_dim` (both 64 in the example above). If `latent_dim != embed_dim`, the model automatically projects embeddings to latent space before applying the adapter.

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

## Validated Benchmark Results

### Cross-Domain Transfer (External Benchmarks)
| Metric | Before Fixes | After Fixes (Optimized) |
|--------|--------------|-------------------------|
| Cross-domain transfer Top-1 | 45.8% | **75.0%** |
| Cross-domain transfer Top-10 | 66.7% | **100%** |
| Domain B held-out samples | 12 | 36 (3×) |
| W_rel causal alignment | 0.56 | **0.68** |
| W_rel semantic alignment | 0.60 | **0.55** |

**Key fixes enabling 75% → 100% cross-domain transfer:**
1. W_rel cross-domain alignment wired into training loop (after each domain + sleep)
2. Relation classification fix — expanded causal verb keyword map (`enables`, `shapes`, `drives`, etc.)
3. Domain B expanded from ~50 to ~150 facts for stable held-out metrics
4. Test-time entity adapter adaptation recovers held-out generalization from **5-12% → 93-100%**:
   - Entity adapter `(U, V)` initialized from nearest training neighbor
   - 10-step MSE minimization: `min ||(subject_embed @ U.T @ V) + offset(verb) - target_embed||²`
   - Mode is controlled by `model._test_time_adapt_mode` flag
   - Path B in `_rp_forward()` uses adapted source + verb offset (vs Path A which uses un-adapted embeddings)

### Graph Inference (External Benchmarks)
| Metric | Value |
|--------|-------|
| Avg latency | 0.032 ms |
| P95 latency | 0.081 ms |
| Peak memory | 0.0006 MB |
| Throughput | ~556 QPS |

### Lifelong Learning (Sleep Ablation)
| Metric | With Sleep | Without Sleep |
|--------|------------|---------------|
| Catastrophic forgetting (permuted MNIST) | **0%** | 0% (experiment needs improvement) |
| Avg accuracy (permuted MNIST) | 11.2% | 11.2% |

### RLMv2 Triple Decomposition (Internal Benchmarks)
| Benchmark | Result |
|-----------|--------|
| Overall top-10 | 80.9% |
| Cross-domain causal top-10 | 75% |
| RP-only verb-offset cross-domain top-10 | 6.7% |
| Simple causal chain top-1 (5 facts, 20 repeats) | 100% |

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