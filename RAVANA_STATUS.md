# RAVANA — Codebase Status Report
**Date:** 2026-05-19 (updated — cognitive architecture enhancements)
**Author:** Likhith
**Purpose:** Shareable status document for LLM collaborators

---

## What Is RAVANA?

A cognitive architecture research project proposing **pressure-driven self-organization** as an alternative to gradient descent for learning. Not an LLM, not symbolic AI, not reward-based RL, not a neural network trainer.

**Core thesis:** Cognition emerges from internal pressure (prediction errors, contradictions, dissonance) that the system self-organizes to resolve — governed by a central Governor, stabilized by Identity, consolidated by Sleep, shaped by Emotion (VAD), and motivated by Meaning.

---

## Architecture: Three-Layer Package

### Layer 1: `ravana/` — The ML Framework (2,155 lines, 12 files)

A PyTorch-compatible API surface built on NumPy. Only hard dependency: `numpy`.

| File | Lines | Purpose |
|------|-------|---------|
| `tensor.py` | 386 | `RawTensor` (NumPy wrapper with PyTorch-like API) + `StateTensor` (adds salience, pressure, stability, decay) |
| `graph.py` | 900+ | `ConceptGraph` with `ConceptNode`/`ConceptEdge` — Hebbian/anti-Hebbian updates, structural plasticity, activation spreading, **hierarchical abstraction**. `ConceptBinding` + `ConceptBindingMap` — probabilistic token↔concept↔memory namespace. **NEW:** Inhibitory edges, soft lateral inhibition, precision-weighted spreading, concept splitting, synaptic homeostasis, temporal context + activation history, interference decay |
| `pressure.py` | 85 | `PressureAccumulator` — semantic, linguistic, episodic, contradiction, **abstraction** pressure with decay + normalization |
| `plasticity.py` | 80 | `HebbianPlasticity`, `AntiHebbianPlasticity` (**NEW:** converts dying edges to inhibitory), `StructuralPlasticity` |
| `propagation.py` | 79 | Activation spreading engine over concept graph |
| `nn/module.py` | 272 | PyTorch-compatible `Module` base with `accumulate_pressure()` + `sleep_cycle()` — replaces backprop |
| `nn/rlm.py` | 700+ | **Recursive Learning Model (RLM)** — alternative to LLM, uses concept graphs + Hebbian plasticity + pressure-driven sleep cycles. **NEW:** `save()`/`load()` (pickle) + `save_zip()`/`load_zip()` (human-readable zip) for complete checkpoint |
| `nn/functional.py` | 119 | Functional API (relu, softmax, cross_entropy, etc.) |
| `world/__init__.py` | 160 | Simulation environment for testing |
| `lab/__init__.py` | 264 | Concept Physics Lab for compositional experiments |
| `__init__.py` | 74 | Package init, `import ravana as torch` pattern |

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

**Additional systems outside core/:**
- `sleep.py` (540+ lines) — 5-stage consolidation: topology analysis, pattern compression, **abstraction compression** (hierarchical concept merging), contradiction resolution, integration. Dream sabotage (20% counterfactual reversals, 10% valence flipping, 1.5x failure oversampling). Tier-0 identity protection. Triggers human memory decay + consolidation. **NEW:** `replay_through_graph()` — hippocampal replay that re-activates memories through the ConceptGraph, applying Hebbian learning on replayed activations.
- `dual_process.py` (209 lines) — System 1 (fast/intuitive) vs System 2 (slow/deliberate) with override logic
- `meaning.py` (224 lines) — Intrinsic motivation: `M = w1(-D_future) + w2(identity_coherence) + w3(predictive_power) * (1 + kappa * effort_cost)`
- `empathy.py` — Theory of Mind via Gaussian Process regression

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

**Note:** Paper claims dissonance trajectory 0.800→0.200 but `final_results.json` shows 0.323→0.322 from a different run configuration. These need reconciliation.

---

## Tests

| Test File | What It Tests |
|-----------|---------------|
| `tests/test_phase_a.py` | Governor hard constraints, resolution partial credit, identity momentum, full StateManager integration |
| `tests/test_grace_layer.py` | Soft boundaries, predictive dampening, resolution mode, identity-coupled control, anti-overshoot |
| `tests/test_memory_integration.py` | Episodic/semantic memory traces, retrieval context |
| `test_dynamics.py` | Honesty metric, commitment integrity, wisdom gain stability, high dissonance behavior |
| `test_dynamics_check.py` | Quick dynamics verification |
| `agent/test_harness.py` | Structured interview system with 8 situation cards |
| `research/core_k0/test_k*.py` | K-series agent robustness, learning, adversarial breaking, regime shifts |

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

### Resolved (as of 2026-05-19)
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
1. ~~**Shared sleep cycle**~~ → **RESOLVED** — `replay_through_graph()` enables graph←memory bidirectional flow
2. ~~**Runaway reinforcement damping**~~ → **RESOLVED** — `homeostatic_downscale()` prevents runaway reinforcement; edge entropy regularization via soft lateral inhibition
3. **Cross-domain transfer at 0.0** — `exp3_cross_domain.json` shows `transfer_efficiency: 0.0`, status: `"NARROW"`
4. **RLM has partial backprop** — `rlm.py` line 314 uses `context_logits.backprop()` despite "no backprop" claim. Limited to context logits head only.
5. **Paper claims vs results mismatch** — dissonance trajectory in paper (0.800→0.200) doesn't match `final_results.json` (0.323→0.322)
6. **No formal benchmarks** — no comparison scripts against PyTorch or other baselines
7. **News-to-MDP pipeline unimplemented** — `reality_grounding.py` exists but structured cognitive event pipeline is a design
8. **Semantic drift defense** — `attractor_drift()` measurement exists in lab but is not wired into the learning loop
9. **ConceptBindingMap not fully wired** — `disambiguate()` and `split_bindings()` exist but ConceptBindingMap is not yet instantiated by ConceptGraph or CognitiveFramework
10. **REM vs SWS distinction** — One sleep mode; brain uses SWS for structural consolidation and REM for creative recombination

---

## Dependencies

**ravana/:** `numpy` (only)
**ravana/:** `numpy>=1.20` (only)
**ravana-v2/interface_agent/:** `feedparser>=5.2.1`, `requests>=2.31.0`, `newspaper3k>=0.2.8`, `openai>=1.12.0`, `anthropic>=0.18.0`
**ravana-v2/:** No requirements.txt (core modules use only stdlib + numpy)

---

## Git History

```
[latest]     Cognitive architecture enhancements — 7 phases: inhibitory edges, soft inhibition,
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
- Not a neural network trainer (no backprop as primary learning mechanism)
- Not PyTorch/TensorFlow/JAX

## What RAVANA IS

- A pressure-driven self-organizing cognitive architecture
- A PyTorch-compatible ML framework (API surface) using Hebbian learning + sleep consolidation
- A comprehensive cognitive system with emotion, meaning, meta-cognition, empathy, dual-process reasoning, global workspace
- A unified package (`ravana`) with a user-facing CognitiveFramework API
- A system with reconstructive memory that doesn't just store — it rebuilds, distorts, consolidates, fragments, and forgets
- A cognitive architecture where identity shapes what gets remembered and sleep actively rewrites the narrative
- A system where consolidated experience physically reshapes representational topology (memory-weights bridge)
- A system with a unified semantic namespace (ConceptBinding) where tokens, concepts, and memories are probabilistically linked
- A system with inhibitory connections, precision-weighted activation, and soft lateral inhibition — closer to cortical dynamics
- A system where concepts can split under contradiction, not just merge — identity evolves through forking
- A system with synaptic homeostasis during sleep — preventing runaway reinforcement
- A system with temporal context — memories are easier to recall in the context where they were encoded
- A system where forgetting is interference-driven, not just time-based — similar memories compete
- A system where sleep replay lets memories literally reshape the graph — hippocampal dynamics
- An evolving semantic ecology — not a static model
- An active research project with empirical validation in constrained environments
- A prototype — not yet AGI, but proposing a novel path toward it

---

*Updated 2026-05-19 (Memory-weights bridge, ConceptGraph persistence, ConceptBinding). Share freely with LLM collaborators for guidance on next steps.*
