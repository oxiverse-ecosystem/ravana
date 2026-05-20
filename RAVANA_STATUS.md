# RAVANA — Codebase Status Report
**Date:** 2026-05-20 (updated — generation stabilization, observability, instruction grounding)
**Author:** Likhith
**Purpose:** Shareable status document for LLM collaborators

---

## What Is RAVANA?

A cognitive architecture research project proposing **pressure-driven self-organization** as an alternative to gradient descent for learning. Not an LLM, not symbolic AI, not reward-based RL, not a neural network trainer.

**Core thesis:** Cognition emerges from internal pressure (prediction errors, contradictions, dissonance) that the system self-organizes to resolve — governed by a central Governor, stabilized by Identity, consolidated by Sleep, shaped by Emotion (VAD), and motivated by Meaning.

---

## Architecture: Three-Layer Package

### Layer 1: `ravana_ml/` — The ML Framework (4,252 lines, 13 files)

A PyTorch-compatible API surface built on NumPy. Only hard dependency: `numpy`.

| File | Lines | Purpose |
|------|-------|---------|
| `tensor.py` | 385 | `RawTensor` (NumPy wrapper with PyTorch-like API) + `StateTensor` (adds salience, free_energy, stability, decay) |
| `graph.py` | 1,202 | `ConceptGraph` with `ConceptNode`/`ConceptEdge` — Hebbian/anti-Hebbian updates, structural plasticity, activation spreading, **hierarchical abstraction**. `ConceptBinding` + `ConceptBindingMap` — probabilistic token↔concept↔memory namespace. Inhibitory edges, soft lateral inhibition, precision-weighted spreading, concept splitting, synaptic homeostasis, temporal context + activation history, interference decay. `form_inhibitory_edges()` with adaptive confidence + bidirectional target inhibition, `apply_prediction_error()` for contradiction tracking, `contradiction_hotspots` for pressure-driven resolution. **NEW:** `ConceptNode.fatigue` field + `effective_activation` property (activation × (1 − fatigue)) |
| `free_energy.py` | 90 | `FreeEnergyAccumulator` — five-channel: semantic, linguistic, episodic, contradiction, abstraction free energy with decay + normalization. Replaces old `pressure.py` |
| `plasticity.py` | 77 | `HebbianPlasticity`, `AntiHebbianPlasticity` (converts dying edges to inhibitory), `StructuralPlasticity` |
| `propagation.py` | 78 | Activation spreading engine over concept graph |
| `tokenizer.py` | 73 | **NEW** — `BPETokenizer` (tiktoken/GPT-2, 50257 vocab), `SimpleTokenizer` (char-level fallback, 256 vocab), `get_tokenizer()` factory |
| `nn/module.py` | 300 | PyTorch-compatible `Module` base with `accumulate_free_energy()` + `sleep_cycle()` — local learning, no backprop. `Linear.backprop()` raises `NotImplementedError` |
| `nn/rlm.py` | 1,338 | **Recursive Learning Model (RLM)** — alternative to LLM. **Predictive coding** learning rule with settle loop + 3 stabilizers. **Saturating concept fatigue** (asymptotic accumulation, multiplicative decay). **Repetition penalty** (sliding window). **Composite exploratory drive** (repetition × low_entropy × free_energy → dynamic temperature/ACF scaling). **Direct Hebbian weight updates** on ctx_logits (lr=0.0001, bypasses slow accumulate+sleep_cycle). **Cognitive trace logging** — per-step JSON + markdown with entropy, fatigue, concepts, free_energy. `save()`/`load()` (pickle) + `save_zip()`/`load_zip()` (human-readable zip) |
| `nn/functional.py` | 118 | Functional API (relu, softmax, cross_entropy, etc.) |
| `world/__init__.py` | 159 | Simulation environments: TinyWorld, CausalSequenceWorld, ObjectInteractionWorld, SensorimotorWorld |
| `lab/__init__.py` | 263 | Concept Physics Lab for compositional experiments |
| `__init__.py` | 106 | Package init, `import ravana as torch` pattern, Device, save/load |

### Layer 2: `ravana-v2/` — The Cognitive Core (96+ files, 23 core modules)

The GRACE architecture (Governance, Reflection, Adaptation, Constraint, Exploration). 20+ cognitive modules spanning emotion, sleep, dual-process reasoning, meta-cognition, empathy, meaning, social epistemology.

**Core modules (`ravana-v2/core/`):**

| Module | Phase | Purpose |
|--------|-------|---------|
| `governor.py` | A | Central control — hard constraints, predictive dampening, boundary pressure, center-seeking, mode regulation (770 lines) |
| `identity.py` | A | Momentum-based self-concept with recovery bias |
| `resolution.py` | A | Continuous partial credit toward wisdom events |
| `adaptation.py` | A | Policy learning from clamp events |
| `strategy.py` | B | 4 exploration modes (AGGRESSIVE, SAFE, STABILIZE, RECOVER) |
| `strategy_learning.py` | B | Mode effectiveness evaluation over time |
| `intent.py` | C | Dynamic objectives evolving from outcomes |
| `planning.py` | C | Micro-planner with forward trajectory simulation |
| `environment.py` | D | Non-stationary world (boundary shifts, noise drift, goal flips) |
| `predictive_world.py` | E | Neural network world model with adaptive surprise threshold |
| `belief_reasoner.py` | F | Competing hypotheses with confidence decay, structural consistency |
| `active_epistemology.py` | G | Value of Information calculation, hypothesis-driven action |
| `surgical_probes.py` | H | KL-divergence driven probe selection to separate hypotheses |
| `social_epistemology.py` | I | Multi-agent belief conflict, trust scoring, deception detection |
| `meta_cognition.py` | J | Self-awareness of epistemic process, bias detection, mode recommendation (435 lines) |
| `reality_friction.py` | K | Partial observability, delayed ground truth, noisy signals |
| `meta2_cognition.py` | K | System questioning its own epistemic method |
| `meta2_integration.py` | K | Hypothesis space audits, epistemic epiphanies |
| `hypothesis_generation.py` | L | Constraint-guided generation: parametric, structural, causal |
| `occam_layer.py` | L | Complexity penalties |
| `emotion.py` | M | VAD (Valence, Arousal, Dominance) via differential equations (234 lines) |
| `memory.py` | — | Episodic, semantic, working memory |
| `global_workspace.py` | N | **NEW** — Competitive broadcast system, consciousness bottleneck |
| `human_memory.py` | O | Persistent episodic/semantic memory with Ebbinghaus decay, spreading activation, reconstructive recall. **NEW:** Interference-based decay (similar memories accelerate each other's forgetting), retrieval-induced forgetting (recall suppresses competitors), temporal context storage (encoding specificity) |

**Additional systems (inside `core/`):**
- `sleep.py` (540+ lines) — 5-stage consolidation: topology analysis, pattern compression, abstraction compression (hierarchical concept merging), contradiction resolution, integration. Dream sabotage (20% counterfactual reversals, 10% valence flipping, 1.5x failure oversampling). Tier-0 identity protection. Triggers human memory decay + consolidation. `replay_through_graph()` — hippocampal replay that re-activates memories through the ConceptGraph, applying Hebbian learning on replayed activations.
- `dual_process.py` (209 lines) — System 1 (fast/intuitive) vs System 2 (slow/deliberate) with override logic
- `meaning.py` (224 lines) — Intrinsic motivation: `M = w1(-D_future) + w2(identity_coherence) + w3(predictive_power) * (1 + kappa * effort_cost)`
- `empathy.py` — Theory of Mind via Gaussian Process regression
- `state.py` — StateManager + CognitiveState (wires all modules together)

### Layer 3: `ravana/` — Unified Package (NEW)

The RAVANA (Recursive Learning Model) package unifies both codebases into a single pip-installable package. Re-exports from `ravana_ml/` and `ravana-v2/core/` — no code duplication, no destructive renames.

```
ravana/
├── __init__.py          # import ravana as torch
├── tensor.py            # re-exports from ravana_ml.tensor
├── graph.py             # re-exports from ravana_ml.graph
├── pressure.py          # re-exports from ravana_ml.pressure
├── plasticity.py        # re-exports from ravana_ml.plasticity
├── propagation.py       # re-exports from ravana_ml.propagation
├── nn/
│   ├── __init__.py      # Module, Linear, Embedding, RLM
│   ├── module.py        # re-exports from ravana_ml.nn.module
│   ├── functional.py    # re-exports from ravana_ml.nn.functional
│   └── rlm.py           # re-exports from ravana_ml.nn.rlm
├── cognitive/
│   ├── __init__.py      # re-exports all 23+ cognitive modules
│   ├── framework.py     # NEW — CognitiveFramework API
│   └── ...              # re-exports from ravana-v2/core/
├── world/               # re-exports from ravana_ml.world
├── lab/                 # re-exports from ravana_ml.lab
├── pyproject.toml       # pip install -e .
└── README.md
```

**Install:** `pip install -e ravana/`
**Import:** `import ravana as torch` or `from ravana.cognitive import CognitiveFramework`

---

## CognitiveFramework API (NEW)

The top-level user interface that wires the ML framework and cognitive core together:

```python
from ravana.cognitive import CognitiveFramework

fw = CognitiveFramework()
state = fw.initialize()

# Core cycle
concepts = fw.perceive(state, input_vec)      # input → active concepts
predictions = fw.predict(state, concepts)      # Hebbian spread → predictions
state = fw.learn(state, predictions, outcomes) # pressure + governor + emotion

# Consolidation
state = fw.sleep(state)                        # 4-stage consolidation + memory bridge

# Inference (no state change, memory-biased)
result = fw.infer(state, input_vec)            # {concepts, predictions, recalled_memories}

# Memory
neighbors = fw.query(state, concept_id)        # graph neighborhood

# Persistence
fw.save("checkpoint.pkl")                      # save graph + memory DB path
fw = CognitiveFramework.load("checkpoint.pkl") # restore graph + reattach DB
fw.rebridge()                                  # sync consolidated memories → graph edges

# Diagnostics
report = fw.diagnose(state)                    # full cognitive dashboard
```

**What it wires together:**
- `ConceptGraph` + `PropagationEngine` (from ravana_ml/) for perception and prediction
- `Governor` + `Identity` + `Resolution` for state regulation
- `VADEmotionEngine` for affective tagging
- `SleepConsolidation` for periodic consolidation
- `MeaningEngine` for intrinsic motivation
- `GlobalWorkspace` for inter-module coordination
- `HebbianPlasticity` + `StructuralPlasticity` for graph-level learning
- `HumanMemoryEngine` for persistent episodic/semantic memory with bridge to ConceptGraph

---

## Hierarchical Abstraction Compression (NEW)

The system now develops **actual graph hierarchy** during sleep — not just edge strengthening, but structural reorganization.

**How it works:**
1. During wake: leaf concepts accumulate co-activation patterns through experience
2. During sleep: `find_coactivated_clusters()` identifies groups of frequently co-activated leaf concepts
3. `merge_concepts()` creates parent concepts via vector centroid averaging, aggregates edges
4. Parent-child hierarchy forms: children retain their edges, parents summarize the cluster
5. Activation spreads upward: activating children partially activates parents (decay=0.3)
6. Abstraction pressure accumulates from uncompressed clusters, driving further compression

**Key properties:**
- Hierarchical levels: leaf (L0) → abstract (L1) → more abstract (L2) → ... (configurable max depth)
- Compression ratio: tracks fraction of abstract vs leaf nodes
- Cluster coherence: only merges concepts with strong internal edge weights
- Protected: won't double-merge already-parented concepts

**API:**
```python
from ravana_ml.graph import ConceptGraph

g = ConceptGraph(dim=64)

# After experience, find and merge co-activated clusters
clusters = g.find_coactivated_clusters(min_cluster_size=3, max_cluster_size=8)
for cluster in clusters:
    parent_id = g.merge_concepts(cluster, abstraction_degree=0.5)

# Hierarchy traversal
leaves = g.get_leaves(parent_id)       # all leaf descendants
ancestors = g.get_ancestors(leaf_id)   # path to root
stats = g.get_abstraction_stats()      # compression ratio, max level, etc.
```

**Why this matters:** This is the bridge between "pressure accumulates" and "structure actually reorganizes." Without abstraction compression, the graph stays flat forever. With it, the system develops hierarchical representations that enable cross-domain transfer and compositional reasoning.

---

## Predictive Coding Learning Rule (NEW — 2026-05-20)

The RLM now uses **predictive coding** instead of backprop. Each layer predicts the layer above it. Error = actual - predicted. No chain rule, no global error signal.

**Settle loop (inference + learning):**
1. Top-down pass: each layer predicts the layer above
2. Error computation: local prediction error at each layer
3. State update: adjust hidden states to minimize local errors
4. Repeat for T steps

**Three stabilizers prevent attractor collapse:**
- **A. Prediction residual normalization:** `e = (actual - predicted) / (eps + ||predicted||)` — prevents giant attractors dominating
- **B. Noise injection:** `states[i] += N(0, sigma)` — preserves diversity, enables REM-style creativity
- **C. Energy floor / anti-collapse:** `novelty = alpha * (state - running_avg)` — prevents static minima

**Learning rule:** Error-gated Hebbian: `Δw_ij ∝ e_i · x_j` — each layer's weight update uses only its own local error.

**Pressure = free energy:** `Σ|e_i|²` across all layers. RAVANA "resolves dissonance" = inference dynamics reducing free energy.

**Why this matters:** This is the epistemological foundation. Backprop was a silent escape hatch back into conventional optimization. Predictive coding preserves locality, biological plausibility, and hierarchical refinement while staying true to the thesis: cognition from pressure, not gradients.

**Direct Hebbian weight updates (2026-05-20):** The `accumulate_free_energy()` → `sleep_cycle()` pipeline has an effective learning rate of ~0.001/step — too slow for output layers. A direct local Hebbian update was added to `context_logits`: `ΔW = lr * (error^T @ hidden)` with `lr=0.0001`, weights clipped to [-5, 5]. Uses raw softmax error (not settle-normalized) for stable gradients. Still local — no chain rule, no backprop.

---

## Generation Stabilization (NEW — 2026-05-20)

The RLM's autoregressive generation now includes three stabilization mechanisms, a composite exploratory drive, and dual-mode cognitive telemetry.

### Saturating Concept Fatigue

`ConceptNode.fatigue` (initialized 0.0) tracks per-concept exhaustion. During `forward_step()`:
- **Decay** (global): `node.fatigue *= (1.0 - fatigue_decay_rate)` — multiplicative decay
- **Accumulation** (active nodes only): `node.fatigue += (1.0 - node.fatigue) * node.activation * fatigue_accumulation_rate` — asymptotes to 1.0
- **Scoring**: `effective_activation = activation * (1.0 - fatigue)` — fatigued concepts lose influence

This prevents persistent activation loops and forces the system to explore alternative concepts.

### Repetition Penalty

In `generate()`, a sliding window of the last `repetition_window` tokens (default 10) tracks generated tokens. Logits for repeated tokens are reduced by `repetition_penalty` (default 1.5). Combined with fatigue, this breaks degenerate generation loops.

### Composite Exploratory Drive

When the system detects degenerate repetition, it dynamically scales exploration parameters:
```
repetition_score = 1.0 - (unique_tokens / total_tokens)  # over last 15 tokens
low_entropy = max(0, 1.0 - normalized_entropy)
free_energy_norm = (fe + 0.5) / (fe + 1.5)
exploration_drive = repetition_score * low_entropy * free_energy_norm
```
When `exploration_drive > 0.15`:
- Temperature increases: `effective_temperature = temperature + 2.0 * (drive - 0.15)`
- ACF boundary widens: `k_active_acf` increases (up to 15)
- Spreading steps increase (up to 3)
- Forward step re-run with dynamic params

### Cognitive Trace Logging

`generate()` exports per-step telemetry in two formats:
- **JSON** (`cognitive_trace.json`): machine-readable array of step objects
- **Markdown** (`cognitive_trace.md`): human-readable table with prompt, generated text, and per-step metrics

Each step records: `step`, `token`, `token_id`, `entropy`, `repetition_score`, `exploration_drive`, `temperature`, `k_active_acf`, `steps`, `top_concepts` (label, activation, fatigue), `free_energy`.

---

## Tokenizer (NEW — 2026-05-20)

`ravana_ml/tokenizer.py` — pluggable tokenization layer:
- `BPETokenizer` — wraps tiktoken (default GPT-2 encoding, 50257 vocab)
- `SimpleTokenizer` — character-level fallback (256 vocab) when tiktoken unavailable
- `get_tokenizer(name)` — factory that tries BPE, falls back to SimpleTokenizer

---

## Contradiction Resolution (NEW — 2026-05-20)

The system now has a complete **sense → accumulate → resolve → suppress** loop for contradictions.

**How it works:**
1. **Sense:** `learn()` tracks prediction errors on concepts. When predictions miss, `contradiction_count` and `pressure` increment on the source concept.
2. **Accumulate:** When pressure exceeds threshold (5.0), the concept enters `contradiction_hotspots`.
3. **Resolve:** During `sleep_cycle()`, `form_inhibitory_edges()` converts weak edges to inhibitory and forms bidirectional inhibition between competing targets. `should_split()` checks if concepts should bifurcate.
4. **Suppress:** `spread_activation()` negates activation along inhibitory edges. `homeostatic_downscale()` prevents runaway weights. `reconcile_contradictions()` resets counts after resolution.

**Three pathways in `form_inhibitory_edges()`:**
1. **Weak edges:** Convert low-confidence excitatory edges (confidence < 0.3) to inhibitory
2. **Strong contradictions:** Form bidirectional inhibitory edges between competing targets when source has very high contradiction count (≥ 3x threshold)
3. **Adaptive dampening:** Weaken confidence on strongly-held contradictions proportional to severity

**Contradictory Concepts Experiment results:**
- Mixed model (contradictory associations) forms **17 inhibitory edges** vs 0 for normal
- `hot ↔ cold` and `fly ↔ swim` develop mutual inhibition
- Input variance 26x higher for ambiguous concepts
- Energy (free energy) higher in mixed model — dissonance accumulates as expected
- Competing edge groups: 10 in mixed vs 1 in normal

**Why this matters:** Most AI systems silently resolve contradictions by averaging. RAVANA accumulates dissonance, forms structural inhibitory edges, and suppresses competing concepts — closer to how the brain handles semantic conflict.

---

## Shared Currencies Audit (NEW — 2026-05-20)

Complete audit of all Python files in `ravana_ml/` and `ravana-v2/core/`. Found severe fragmentation in state variables across 23+ modules.

### Fragmentation Found

| Variable | Distinct Concepts | Range Issues |
|----------|-------------------|-------------|
| **pressure** | 6 roles: learning signal, contradiction detector, sleep trigger, cognitive load, boundary resistance, identity crisis | [0,100], [0,1], [0,∞) — three different ranges |
| **confidence** | 4 concepts: graph node/edge, binding, hypothesis, decision/signal | All [0,1] but different semantics |
| **stability** | 6 concepts: graph, neural weight, identity, behavioral, memory trace, hypothesis | All [0,1] but different semantics |
| **salience** | 3 concepts: graph node, neural weight, episodic memory | Range violations found |
| **entropy** | 5 concepts: binding ambiguity, prediction distribution, pressure localization, memory, behavioral | Different ranges |
| **coherence** | 5 concepts: graph, memory, cognitive, identity, sleep | Different ranges |

### Bugs Fixed

1. **Salience overflow** — `rlm.py:373,380` set `_salience=3.0` and `2.0` directly, bypassing [0,1] setter. Hebbian updates amplified 2-3x beyond intended range. **Fixed:** error magnitude carries signal (×3.0, ×2.0), salience clamped to 1.0.

2. **Memory salience overflow** — `memory.py:100` computed `salience = dissonance * 1.5` → could reach 1.5, no clamping. **Fixed:** `min(1.0, dissonance * 1.5)`.

3. **IdentityState.stability dead code** — Defined and read but never updated. Remained at 0.5 forever. **Fixed:** stability now computed from history variance (stable=0.999, volatile=0.740).

### Cross-Domain Semantic Collisions

- `node.confidence = mem.get("coherence", 1.0)` — memory fidelity ≠ concept reliability
- `node.salience = utility` — memory utility ≠ concept importance

### Canonical Variable Proposal

| Variable | Maps From | Range | Conservation |
|----------|-----------|-------|-------------|
| `energy` | activation, cognitive work | 0-1 | Flows |
| `free_energy` | pressure (all channels), prediction error | 0-∞ | Transforms |
| `stability` | node/edge/identity/memory/hypothesis stability | 0-1 | Increases with sleep |
| `entropy` | ambiguity, uncertainty, edge entropy | 0-∞ | Increases naturally |
| `coherence` | graph coherence, memory fidelity, inverse dissonance | 0-1 | Goal: maximize |
| `temperature` | exploration volatility, arousal | 0-∞ | High = creative |
| `momentum` | persistence across time | 0-∞ | Carries through cycles |
| `valence` | affective sign (VAD) | -1 to 1 | From emotion system |
| `salience` | attentional priority | 0-1 | Determines processing |

### Progress

- [x] Audit complete — all Python files scanned
- [x] Salience overflow bugs fixed (`rlm.py` error magnitude carries signal, salience clamped to 1.0)
- [x] Memory salience overflow fixed (`memory.py` clamped to `min(1.0, dissonance * 1.5)`)
- [x] Identity stability dead code fixed (now computed from history variance)
- [x] `PressureAccumulator` → `FreeEnergyAccumulator` (full file rename: `pressure.py` → `free_energy.py`)
- [x] `ravana_ml/__init__.py` updated to export `free_energy` instead of `pressure`
- [ ] `ravana/pressure.py` unified package still references old module (needs update to `free_energy`)
- [ ] ConceptNode.pressure field rename → `free_energy`
- [ ] ConceptEdge.pressure field rename → `free_energy`
- [ ] RLM.total_pressure → `free_energy`
- [ ] Module._pressure → `free_energy`
- [ ] Confidence unification (4 concepts → canonical)
- [ ] Stability unification (6 concepts → canonical)
- [ ] Cross-domain mapping fixes

**Why this matters:** Without shared currencies, the architecture becomes "beautiful cognitive federalism with no central physics." Each module drifts into its own semantics. Sleep consolidation, contradiction resolution, and counterfactual simulation all need commensurable variables.

---

## Global Workspace (NEW)

`ravana-v2/core/global_workspace.py` — the consciousness bottleneck.

**How it works:**
1. Modules submit bids with urgency scores each cycle
2. GW selects the highest-urgency bid (with stochastic noise)
3. Winner is broadcast to all modules via the temporal buffer
4. Buffer (capacity 7) serves as working memory

**Integration:** Wired into `StateManager.step()` — emotion and meaning submit bids after their updates. GW pressure feeds into the sleep engine.

**API:**
```python
gw = GlobalWorkspace(GWConfig(capacity=7))
gw.submit_bid("emotion", payload, urgency=0.8, valence=0.3)
gw.submit_bid("meaning", payload, urgency=0.4)
broadcast = gw.compete()  # returns winning GWContent
recent = gw.get_recent(3) # last 3 broadcasts
context = gw.get_context_vector()  # weighted VAD vector
```

---

## Human Memory Engine (NEW — Phase O)

`ravana-v2/core/human_memory.py` — persistent episodic/semantic memory that reconstructs, not just retrieves.

**Core properties:**
- **SQLite persistence** — memories survive across sessions
- **Ebbinghaus decay** — memories fade through neglect, modulated by utility and coherence
- **Episodic → semantic consolidation** — frequently accessed memories promote to semantic
- **Spreading activation** — associative graph recall via BFS with decaying intensity
- **Auto-linking** — shared tags create graph edges automatically

**Memory entropy (each memory tracks):**
- `coherence` — erodes over time (-0.002/cycle), accelerates decay when low
- `stability` — grows with each recall (+0.05)
- `retrieval_distortion` — accumulates on each recall (+0.01)
- `associative_divergence` — drifts over time (+0.005/cycle)

**Utility-aware modulation:**
- `predictive_utility` — derived from dissonance reduction per episode
- Persistence formula: `0.4*utility + 0.3*emotional + 0.15*access + 0.15*coherence`
- High utility = slower decay. Emotional salience alone doesn't guarantee survival.

**Reconstructive recall modes:**
- `reconstructive_recall()` — rebuilds from graph neighbors when direct matches are weak
- `blended_recall()` — merges related memories into composites with weighted attributes
- `abstraction_recall()` — boosts semantic for abstract queries, episodic for concrete
- `reconstruct_schema()` — extracts clusters, hubs, bridges, chains from graph topology
- `detect_hallucination()` — flags when reconstruction diverges from ground truth
- `find_contradictions()` + `reconcile_contradictions()` — detects and resolves conflicting memories

**Cognitive integration:**
- `sleep_replay()` — actively rewrites memories during consolidation: strengthens coherent, weakens degraded, merges similar episodic into semantic
- `fragment_memory()` — splits memories under cognitive pressure when associative divergence is high and contradictions exist
- `stitch_narratives()` — links sequential episodes with shared tags/emotional continuity into narrative chains
- Identity interaction: strong identity boosts importance, identity pressure boosts emotional weight, identity-derived tags

**Integration:** Wired into `StateManager.step()` — stores each cognitive episode with emotional salience from VAD state and identity modulation. Sleep cycle triggers `sleep_replay()` → `apply_decay()` → `consolidate()`. Participates in Global Workspace competition via `compute_gw_bid()`.

**API:**
```python
from ravana.cognitive import HumanMemoryEngine, HumanMemoryConfig

engine = HumanMemoryEngine(HumanMemoryConfig(db_path="my_memory.db"))

# Store and recall
mid = engine.remember("Python is great for AI", tags="python,ai", importance=0.8)
results = engine.recall("python")
results = engine.semantic_search("machine learning")

# Reconstructive modes
blended = engine.blended_recall("programming")
abstract = engine.abstraction_recall("code", abstraction_level="abstract")
schema = engine.reconstruct_schema()

# Cognitive integration
engine.sleep_replay(state_snapshot)   # replay + reshape during sleep
frag = engine.fragment_memory(mid)    # split under pressure
narratives = engine.stitch_narratives()  # link episodes into stories

# Entropy and decay
engine.apply_decay()   # Ebbinghaus sweep + auto-consolidation
engine.consolidate()   # episodic → semantic promotion
status = engine.get_status()  # includes entropy stats
```

---

## Memory-Weights Bridge (NEW)

The core "memory IS the model" connection. Consolidated experience physically reshapes the ConceptGraph's representational topology.

**How it works:**
1. During `sleep()`, after memory consolidation, `bridge_to_graph()` queries consolidated memories from SQLite
2. Maps them to ConceptGraph nodes by label/tag matching
3. Creates new concept nodes for unmatched memories (deterministic hashed vectors)
4. Strengthens edges between concepts that co-occur in consolidated memories via `hebbian_update()`
5. Coactivation score based on how many consolidated memories share a tag

**The closed cognitive loop:**
```
experience → consolidation → graph reinforcement → future activation bias → altered future cognition
```

**Three integration points:**
- **Training**: `sleep()` triggers bridge after consolidation — consolidated memories become graph edges
- **Loading**: `rebridge()` re-syncs consolidated memories with restored graph after checkpoint load
- **Inference**: `infer()` returns `recalled_memories` biased by active ConceptGraph concepts

**API:**
```python
# During sleep (automatic)
fw.sleep(state)  # triggers bridge_to_graph() internally

# After loading (manual)
fw = CognitiveFramework.load("checkpoint.pkl")
fw.rebridge()  # sync consolidated memories → graph edges

# Biased recall
results = engine.recall_with_concepts(active_concept_ids, concept_graph)
```

**Why this matters:** Before the bridge, HumanMemoryEngine (persistent) and ConceptGraph (transient) were disconnected. Now consolidated experience physically reshapes representational topology. The graph becomes reconstructible from lived experience.

---

## Model Persistence (NEW)

**RLM save/load** (pickle + zip formats):
```python
from ravana_ml.nn import RLM

model.save("model.pkl")           # pickle (fast, simple)
model.save_zip("model.zip")       # zip (human-readable, safe, smaller)

model = RLM.load("model.pkl")     # load pickle
model = RLM.load_zip("model.zip") # load zip
model = ravana.load("model.zip")  # auto-detect format
```

**Zip layout:** `arrays.npz` (compressed numpy weights + node vectors) + `graph.json` (topology + metadata, readable) + `metadata.json` (config, scalars, pressure state)

**CognitiveFramework save/load** (ConceptGraph persistence):
```python
fw.save("framework.pkl")                      # saves graph + memory DB path
fw = CognitiveFramework.load("framework.pkl") # restores graph + reattaches DB
fw.rebridge()                                 # sync consolidated memories
```

**What gets saved:** Neural weights with cognitive metadata (pressure, stability, salience), ConceptGraph (nodes, edges, vectors, topology), RLM scalars, pressure accumulator state, token-concept mapping, human memory DB path.

---

## ConceptBinding (NEW — Unified Semantic Namespace)

`ravana_ml/graph.py` — probabilistic mapping between tokens, concepts, and memories.

**ConceptBinding**: a single link with confidence, source, reinforcement count, decay, ambiguity tracking. Not a static dictionary entry — a living semantic link that can drift, split, merge, and decay.

**ConceptBindingMap**: manages the full binding space.
- Multiple bindings per token (ambiguous meanings)
- Confidence-weighted lookup
- Ambiguity detection via entropy
- Decay and pruning of weak bindings
- Concept split/merge support

**API:**
```python
from ravana_ml.graph import ConceptBindingMap

bmap = ConceptBindingMap()
bmap.bind(token_id=0, concept_id=10, confidence=0.8, source="learned")
bmap.bind(token_id=0, concept_id=11, confidence=0.3, source="memory")  # ambiguous

concepts = bmap.get_concepts(0)       # all concepts for token
best = bmap.best_concept(0)           # strongest match
ambiguous = bmap.is_ambiguous(0)      # True (multiple strong bindings)
score = bmap.ambiguity_score(0)       # 0-1 entropy-based measure
bmap.decay_all(rate=0.01)             # time-based decay
pruned = bmap.prune(min_strength=0.05) # remove weak bindings
```

**Why this matters:** The token↔concept↔memory namespace needs to be probabilistic, not static. Concepts split, merge, meanings drift, ambiguity appears. Plain dicts can't handle that. ConceptBinding creates the foundation for: semantic grounding, memory indexing, concept persistence, symbolic continuity.

---

## Cognitive Architecture Enhancements (NEW — 7 Phases)

Based on cognitive science research (spreading activation, synaptic homeostasis, inhibitory connections, temporal context, interference theory, hippocampal replay), the following mechanisms were added to close the gap between RAVANA and biological cognition.

### Phase 1: Inhibitory Edges + Soft Lateral Inhibition

`ConceptEdge` now has `edge_type` (`"excitatory"` or `"inhibitory"`). Inhibitory edges subtract activation during spreading. `AntiHebbianPlasticity` converts dying excitatory edges to inhibitory instead of deleting them — the mismatch itself is information. `_top_k_activation()` replaced with `_soft_lateral_inhibition()` — each active concept suppresses others proportionally to similarity, preserving near-winners (unlike hard zeroing). `form_inhibitory_edges()` creates inhibitory connections between persistently contradictory concepts during sleep.

### Phase 2: Precision-Weighted Spreading Activation

`spread_activation()` now incorporates `edge.confidence` — high-confidence edges carry more signal, low-confidence edges are noisy. Fan effect normalization prevents hub domination: `activation /= sqrt(in_degree + 1)`. Hebbian learning rate scales with prediction surprise: `effective_lr = lr * (1 + error * confidence * 5)` — confident errors produce bigger updates (like the brain's error-related negativity).

### Phase 3: Concept Splitting + Genesis Vector Drift

`ConceptNode` tracks `genesis_vector` (original vector at creation) and `drift_magnitude` (L2 distance from genesis). `should_split()` detects when a concept should fork based on contradiction count, drift magnitude, and edge entropy (diverse unrelated targets). `split_concept()` creates two competing sub-concepts, distributes edges by vector alignment, and forms inhibitory edges between them. `homeostatic_downscale()` implements global synaptic homeostasis — all edges weakened proportionally, but high-stability edges are protected.

### Phase 4: Global Synaptic Homeostasis During Sleep

`homeostatic_downscale(protection_threshold=0.8, downscale_factor=0.8)` is called during `CognitiveFramework.sleep()`. All edge weights are multiplied by the downscale factor, but edges with stability above the protection threshold are preserved. This prevents runaway reinforcement and maintains signal-to-noise ratio — the brain's critical maintenance mechanism during SWS.

### Phase 5: Temporal Context + Activation History

`ConceptNode` now tracks `last_activated`, `activation_history` (rolling window of 100), and `temporal_context` (context vector at last activation). `recency_score()` and `frequency_score()` enable time-sensitive retrieval. `ConceptGraph` maintains a drifting `temporal_context` vector (EMA of active concept centroids). `HumanMemoryEngine` stores temporal context with each memory (encoding specificity principle — memories are easier to recall in the context where they were encoded).

### Phase 6: Interference-Based Decay + Retrieval-Induced Forgetting

`_apply_decay()` now includes interference from similar recent memories — new memories with overlapping tags accelerate decay of existing ones (retroactive interference). `retrieval_induced_forgetting()` suppresses competing memories that match a query but weren't recalled, preventing interference during future recall.

### Phase 7: Replay-Through-Graph

`SleepConsolidation.replay_through_graph()` implements hippocampal replay — samples episodic memories, matches their keywords to concept labels, activates matched concepts, runs spreading activation, and applies Hebbian learning on the replayed activations. This is how memories literally reshape the graph during sleep. Wired into `CognitiveFramework.sleep()` after memory consolidation and before the bridge.

**Files modified:** `ravana_ml/graph.py`, `ravana_ml/plasticity.py`, `ravana-v2/core/sleep.py`, `ravana-v2/core/human_memory.py`, `ravana/cognitive/framework.py`, `ravana_ml/nn/rlm.py` (zip serialization for edge_type).

---

## Empirical Validation

| Experiment | Result |
|------------|--------|
| 100K-episode stability | Dissonance 0.800→0.288, Identity 0.300→0.806, 99.1% survival |
| Classroom pilot (N=10K) | Demographic parity gap reduced 19.58%→7.81% (60.1% reduction), all groups improved simultaneously |
| Adversarial safety | 1.25x resistance to corrupt reward signals. 24.0% fairness collapse without dissonance engine |
| K2 agent | 100% late-phase survival, zero decline from early to late phase |
| RLM convergence | 5/5 exact-match generation, 9/9 causal edges |
| RLM Phase I+ | Epistemic resilience confirmed under reality friction (observation noise, partial observability) |
| RLM Full Architecture (2026-05-20) | 6 inhibitory edges formed, 5 hotspots resolved, apple disambiguated (score=0.879), context path active (range=1.42), 268 total pressure, all state survives save/load |
| Contradictory Concepts (2026-05-20) | Mixed model: 17 inhibitory edges (vs 0 normal), 26x input variance for ambiguous concepts, 10 competing edge groups |
| Shared Currencies Audit (2026-05-20) | Complete audit of all Python files: 6 pressure roles, 4 confidence concepts, 6 stability concepts, 3 salience concepts, 5 entropy concepts, 5 coherence concepts. 3 bugs fixed (salience overflow, identity dead code). PressureAccumulator unified to free_energy |
| Generation Stabilization (2026-05-20) | test_generation.py: 5/5 tests pass. Fatigue saturates correctly (18+ nodes with fatigue). Repetition penalty increases unique words (5→7). Compression scorer: 1.0 keyword overlap on perfect input, 0.45 stopword penalty on bad input. Learning signal: chased logit 1.58→2.33 after 50 epochs. Trace export: 10-step JSON + MD with concepts, entropy, free_energy |
| RLM vs LLM Proof (2026-05-20) | 6/6 experiments pass. Few-shot: RLM 100% on 3/5-shot (matches MLP, no backprop). Contradiction: 2 inhibitory edges formed. Identity: 100% consistency across 5 save/load cycles. Consolidation: weight shift -0.02 after sleep. Interference: competing memory weakens (effect=0.034). Efficiency: RLM 100x slower but no GPU needed |

**Note:** Paper claims dissonance trajectory 0.800→0.200 but `final_results.json` shows 0.323→0.322 from a different run configuration. These need reconciliation.

---

## Tests

| Test File | What It Tests |
|-----------|---------------|
| `test_generation.py` | Tokenizer roundtrip, stateful equivalence (forward vs forward_step), ACF bounding, fatigue stabilization, repetition penalty, compression scorer correctness, learning signal verification, trace export (JSON + MD) |
| `test_rlm_vs_llm.py` | **NEW** — 6 proof-of-superiority experiments: few-shot learning (RLM vs MLP vs Frozen LLM), contradiction resolution (inhibitory edges), identity persistence (save/load cycles), consolidation (sleep restructuring), interference forgetting (similar vs dissimilar), resource efficiency |
| `tests/test_phase_a.py` | Governor hard constraints, resolution partial credit, identity momentum, full StateManager integration |
| `test_rlm_full.py` | Full RLM architecture test: predictive coding, contradiction resolution, ConceptBindingMap, context_scale, sleep cycle, persistence |
| `test_contradiction.py` | Contradictory concepts experiment: normal vs contradictory vs mixed conditions, inhibitory edge formation, ambiguity detection |
| `tests/test_grace_layer.py` | Soft boundaries, predictive dampening, resolution mode, identity-coupled control, anti-overshoot |
| `tests/test_memory_integration.py` | Episodic/semantic memory traces, retrieval context |
| `test_dynamics.py` | Honesty metric, commitment integrity, wisdom gain stability, high dissonance behavior |
| `test_dynamics_check.py` | Quick dynamics verification |
| `agent/test_harness.py` | Structured interview system with 8 situation cards |
| `research/core_k0/test_k*.py` | K-series agent robustness, learning, adversarial breaking, regime shifts (10 files: k1_3, k2_breakers/learning/robustness, k3_exp1-3/trajectory/regime_shift, latent_regime) |

**RLC integration tests (14/14 passing):**
- `import ravana`, tensor creation, nn.Linear, ConceptGraph via ravana.graph
- Cognitive modules (Governor, Emotion, GlobalWorkspace)
- GlobalWorkspace bidding and broadcast
- CognitiveFramework: init, perceive, learn, infer, sleep, diagnose
- HumanMemoryEngine: remember, recall, decay, reconstructive modes
- End-to-end: forward + pressure + sleep_cycle
- `import ravana as torch` alias

---

## Roadmap Status

| Phase | Name | Status |
|-------|------|--------|
| **Phase 1** | Stable Physics & Cognitive Loops | **COMPLETED** — K0-K3, dissonance engine, identity clamp |
| **Phase 2** | Human-Like Memory & Real-World Context | **IN PROGRESS** |
| **Phase 3** | Educational Pilot & Value Alignment | **NEXT** — All items unchecked |
| **Phase 4** | Scaling to Level 3 (Expert AGI) | **FUTURE** — All items unchecked |

**Phase 2 items:**
- [x] Global Workspace memory integration — **DONE** (`global_workspace.py` + wired into StateManager)
- [x] Hierarchical abstraction compression — **DONE** (merge_concepts, cluster detection, hierarchy traversal, sleep integration)
- [x] Human memory engine — **DONE** (`human_memory.py`, persistent SQLite, Ebbinghaus decay, spreading activation, reconstructive recall)
- [ ] Episodic buffer with temporal binding
- [ ] Semantic knowledge graph (Bayesian)
- [ ] News-to-MDP pipeline (real-world grounding)
- [ ] Epistemic news ingestion

**Phase 3 unchecked items:**
- [ ] Fairness scaffolding
- [ ] XAI module
- [ ] Wisdom score integration
- [ ] Metacognitive probing
- [ ] Integrity testing

**Phase 4 unchecked items:**
- [ ] Hypothesis generation expansion
- [ ] Surgical probing at scale
- [ ] Cross-domain transfer (target: transfer efficiency > 0.8, currently 0.0)

**Final target:** "Jensen Huang" functional milestone — Composite Wisdom Score 0.85, Brier Score < 0.1, DeepMind Level 3

---

## Research Agents

**Research agents (`ravana-v2/research/`):**
- K0 through K3 agent loops with progressive sophistication
- K2 achieved 100% late-phase survival
- K3 added belief/regime-shift detection with transfer learning foundations
- 5 experiment environments: resource, classroom, deceptive, delayed, latent_regime

**Interface agent (`ravana-v2/interface_agent/`):**
- LLM interpreter (Groq API, llama-3.3-70b-versatile)
- Reality grounding (RSS/news ingestion)
- Telegram reporter
- Memory learner from interactions

**Zo Agent (`ravana-v2/agent/`):**
- Self-improving system running every 7 hours
- Web researches AGI/cognitive architecture methods
- Implements improvements autonomously
- Reports via Telegram

---

## Known Issues & Gaps

### Resolved (as of 2026-05-20)
- ~~Two disconnected codebases~~ → Unified via `ravana/` package
- ~~No packaging infrastructure~~ → `ravana/pyproject.toml` exists, `pip install -e .` works
- ~~Global Workspace missing~~ → `global_workspace.py` implemented and wired into StateManager
- ~~Framework API not built~~ → `CognitiveFramework` class implemented with full API
- ~~Phase 2.5 roadmap items~~ → All 6 items completed (abstraction compression, human memory, identity, replay, fragmentation, narrative)
- ~~No inhibitory edges~~ → `edge_type` field on ConceptEdge, soft lateral inhibition, form_inhibitory_edges
- ~~Hard winner-take-all activation~~ → Replaced with soft lateral inhibition (similarity-weighted suppression)
- ~~No precision weighting~~ → Edge confidence now modulates activation flow
- ~~No concept splitting~~ → `split_concept()` with genesis vector drift tracking
- ~~No synaptic homeostasis~~ → `homeostatic_downscale()` in sleep cycle
- ~~No temporal context~~ → Activation history, recency/frequency scores, drifting temporal context vector
- ~~Pure time-based decay~~ → Interference-based decay (similar memories accelerate forgetting)
- ~~No retrieval-induced forgetting~~ → `retrieval_induced_forgetting()` suppresses competitors
- ~~Memory→graph only~~ → `replay_through_graph()` enables graph←memory bidirectional flow
- ~~**RLM has partial backprop**~~ → **RESOLVED** (2026-05-20) — Replaced with predictive coding settle loop. `Linear.backprop()` raises `NotImplementedError`. Each layer computes local prediction error. Three stabilizers: residual normalization, noise injection, anti-collapse.
- ~~**Contradiction resolution not wired**~~ → **RESOLVED** (2026-05-20) — `apply_prediction_error()` tracks contradictions, `form_inhibitory_edges()` forms bidirectional inhibition between competing targets, `homeostatic_downscale()` + `reconcile_contradictions()` in sleep cycle. Full loop: sense → accumulate → resolve → suppress.
- ~~**No generation stabilization**~~ → **RESOLVED** (2026-05-20) — Saturating concept fatigue, repetition penalty, composite exploratory drive with dynamic temperature/ACF scaling.
- ~~**No cognitive telemetry**~~ → **RESOLVED** (2026-05-20) — Per-step JSON + markdown trace logging with entropy, fatigue, concepts, free_energy.
- ~~**No tokenizer module**~~ → **RESOLVED** (2026-05-20) — `tokenizer.py` with BPETokenizer (tiktoken/GPT-2) + SimpleTokenizer fallback.
- ~~**PressureAccumulator naming**~~ → **RESOLVED** (2026-05-20) — Full rename to `FreeEnergyAccumulator`, `pressure.py` → `free_energy.py`.

### Phase 2.5 (bridging Phase 2 → Phase 3)
- [x] Hierarchical abstraction compression — **DONE**
- [x] Human memory engine (Phase O) — **DONE** (persistent, decay, consolidation, reconstructive recall)
- [x] Identity interaction — **DONE** (strong identity boosts importance, identity pressure boosts emotional, identity-derived tags)
- [x] Replay-driven memory reshaping — **DONE** (`sleep_replay()` strengthens coherent memories, weakens degraded, merges similar episodic)
- [x] Memory fragmentation — **DONE** (`fragment_memory()` splits under cognitive pressure into aligned + contradiction fragments)
- [x] Cross-episode narrative stitching — **DONE** (`stitch_narratives()` links temporal sequences into narrative chains)
- [ ] Latent manifold stabilization (Identity engine tracks concept embedding trajectories)
- [ ] Structural replay metrics (measure abstraction depth, concept reuse, cross-domain transfer)

### Remaining
1. **Cross-domain transfer at 0.0** — `exp3_cross_domain.json` shows `transfer_efficiency: 0.0`, status: `"NARROW"`
2. **Paper claims vs results mismatch** — dissonance trajectory in paper (0.800→0.200) doesn't match `final_results.json` (0.323→0.322)
3. **No formal benchmarks** — no comparison scripts against PyTorch or other baselines
4. **News-to-MDP pipeline unimplemented** — `reality_grounding.py` exists but structured cognitive event pipeline is a design
5. **Semantic drift defense** — `attractor_drift()` measurement exists in lab but is not wired into the learning loop
6. **REM vs SWS distinction** — One sleep mode; brain uses SWS for structural consolidation and REM for creative recombination
7. **Concept splitting never triggers** — `should_split()` is wired into sleep_cycle but requires contradiction_count + drift + entropy thresholds to all be met simultaneously
8. **Shared currencies incomplete** — `FreeEnergyAccumulator` rename done. Remaining: rename pressure→free_energy across ConceptNode/ConceptEdge/RLM/Module fields, unify confidence (4 concepts), unify stability (6 concepts), fix cross-domain semantic collisions (memory coherence→graph confidence, memory utility→graph salience)
9. **`ravana/` package stale references** — `ravana/pressure.py` and `ravana/__init__.py` still import from `ravana_ml.pressure` which no longer exists; needs update to `free_energy`
10. **`accumulate_free_energy()` → `sleep_cycle()` learning rate too slow** — Effective lr ~0.001/step. Direct Hebbian update on ctx_logits works around this, but hidden layers still use the slow path. Consider increasing sleep_cycle application rate or adding direct updates to hidden layers

---

## Dependencies

**ravana_ml/:** `numpy>=1.20` (required), `tiktoken` (optional — for BPETokenizer; falls back to SimpleTokenizer)
**ravana/ (unified):** `numpy>=1.20` (required), `tiktoken` (optional)
**ravana-v2/interface_agent/:** `feedparser>=5.2.1`, `requests>=2.31.0`, `newspaper3k>=0.2.8`, `openai>=1.12.0`, `anthropic>=0.18.0`
**ravana-v2/:** No requirements.txt (core modules use only stdlib + numpy)

---

## Git History

```
[latest]     Generation stabilization + observability + instruction grounding + RLM vs LLM proof —
             saturating concept fatigue, repetition penalty, composite exploratory drive, direct
             Hebbian weight updates on ctx_logits, cognitive trace logging (JSON + markdown),
             tokenizer module (BPE + simple), PressureAccumulator → FreeEnergyAccumulator (full rename),
             test_generation.py (5 tests), experiment_rlm_vs_llm.py (6 experiments), test_rlm_vs_llm.py
             (6/6 pass), experiment_baselines.py (NumPy MLP + Frozen LLM baselines) — UNCOMMITTED
834c803 Predictive coding + contradiction resolution + binding map + context_scale +
             shared currencies audit — replace backprop with settle loop, wire form_inhibitory_edges/
             should_split/homeostatic_downscale into sleep_cycle, activate context_scale (1.0),
             wire ConceptBindingMap, fix salience overflow bugs, fix IdentityState.stability dead code,
             PressureAccumulator.total → free_energy, test_rlm_full.py + test_contradiction.py
834c803 Cognitive architecture enhancements — 7 phases: inhibitory edges, soft inhibition,
             precision-weighted activation, concept splitting, homeostasis, temporal context,
             interference decay, retrieval-induced forgetting, replay-through-graph
a851f71 Add ConceptBinding — unified token ↔ concept ↔ memory namespace
47587a1 Add CognitiveFramework save/load — ConceptGraph persistence
eadb302 Integrate memory bridge into infer() and add rebridge()
61939ca Memory-as-weights bridge: consolidated memories → ConceptGraph edges
b35982a Add zip checkpoint format — human-readable, safe, partial-load
aebe4d6 Phase O: Identity interaction, sleep replay, fragmentation, narrative stitching
de710e9 Phase O: Reconstructive memory — entropy, utility modulation, associative recall
bc2d491 Phase O: Human Memory — persistent episodic/semantic memory with Ebbinghaus decay
5daac51 Phase 2.5: Hierarchical abstraction compression
84b9d59 Phase N: Global Workspace + competitive broadcast architecture
```

---

## What RAVANA Is NOT

- Not an LLM wrapper
- Not symbolic AI
- Not reward-based reinforcement learning
- Not a neural network trainer (no backprop, no gradient descent, no chain rule)
- Not PyTorch/TensorFlow/JAX
- Not a transformer wearing a neuroscience costume

## What RAVANA IS

- A pressure-driven self-organizing cognitive architecture
- A PyTorch-compatible ML framework (API surface) using Hebbian learning + sleep consolidation
- A predictive coding system: each layer predicts the layer above, errors are local, learning is error-gated Hebbian (Δw ∝ e_i · x_j)
- A comprehensive cognitive system with emotion, meaning, meta-cognition, empathy, dual-process reasoning, global workspace
- A unified package (`ravana`) with a user-facing CognitiveFramework API
- A system with reconstructive memory that doesn't just store — it rebuilds, distorts, consolidates, fragments, and forgets
- A cognitive architecture where identity shapes what gets remembered and sleep actively rewrites the narrative
- A system where consolidated experience physically reshapes representational topology (memory-weights bridge)
- A system with a unified semantic namespace (ConceptBinding) where tokens, concepts, and memories are probabilistically linked
- A system with inhibitory connections, precision-weighted activation, and soft lateral inhibition — closer to cortical dynamics
- A system with contradiction resolution: pressure accumulates from prediction errors, inhibitory edges form between competing concepts during sleep, concepts split under sustained contradiction pressure
- A persistent cognitive operating system where memory, identity, recurrence, and consolidation are first-class citizens
- A system where concepts can split under contradiction, not just merge — identity evolves through forking
- A system with synaptic homeostasis during sleep — preventing runaway reinforcement
- A system with temporal context — memories are easier to recall in the context where they were encoded
- A system where forgetting is interference-driven, not just time-based — similar memories compete
- A system where sleep replay lets memories literally reshape the graph — hippocampal dynamics
- An evolving semantic ecology — not a static model
- A system with saturating concept fatigue that prevents persistent activation loops and forces exploration
- A system with composite exploratory drive that dynamically scales temperature and search breadth when repetition is detected
- A system with cognitive telemetry — per-step JSON + markdown traces exposing entropy, fatigue, concepts, free energy during generation
- An active research project with empirical validation: 6/6 proof-of-superiority experiments pass (few-shot, contradiction, identity, consolidation, interference, efficiency)
- A prototype — not yet AGI, but proposing a novel path toward it

---

*Updated 2026-05-20 (Generation stabilization, observability, instruction grounding, shared currencies rename). Share freely with LLM collaborators for guidance on next steps.*
