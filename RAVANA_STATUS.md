# RAVANA — Codebase Status Report
**Date:** 2026-05-19 (updated)
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
| `graph.py` | 564 | `ConceptGraph` with `ConceptNode`/`ConceptEdge` — Hebbian/anti-Hebbian updates, structural plasticity, activation spreading, **hierarchical abstraction** (merge_concepts, cluster detection, hierarchy traversal) |
| `pressure.py` | 85 | `PressureAccumulator` — semantic, linguistic, episodic, contradiction, **abstraction** pressure with decay + normalization |
| `plasticity.py` | 70 | `HebbianPlasticity`, `AntiHebbianPlasticity`, `StructuralPlasticity` |
| `propagation.py` | 79 | Activation spreading engine over concept graph |
| `nn/module.py` | 272 | PyTorch-compatible `Module` base with `accumulate_pressure()` + `sleep_cycle()` — replaces backprop |
| `nn/rlm.py` | 376 | **Recurrent Latent Module** — working language model using concept graphs, Hebbian learning, competitive inhibition, sleep cycles |
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
| `human_memory.py` | O | **NEW** — Persistent episodic/semantic memory with Ebbinghaus decay, spreading activation, reconstructive recall |

**Additional systems outside core/:**
- `sleep.py` (486 lines) — 5-stage consolidation: topology analysis, pattern compression, **abstraction compression** (hierarchical concept merging), contradiction resolution, integration. Dream sabotage (20% counterfactual reversals, 10% valence flipping, 1.5x failure oversampling). Tier-0 identity protection. Triggers human memory decay + consolidation.
- `dual_process.py` (209 lines) — System 1 (fast/intuitive) vs System 2 (slow/deliberate) with override logic
- `meaning.py` (224 lines) — Intrinsic motivation: `M = w1(-D_future) + w2(identity_coherence) + w3(predictive_power) * (1 + kappa * effort_cost)`
- `empathy.py` — Theory of Mind via Gaussian Process regression

### Layer 3: `rlc/` — Unified Package (NEW)

The RLC (Recursive Learning Model) package unifies both codebases into a single pip-installable package. Re-exports from `ravana/` and `ravana-v2/core/` — no code duplication, no destructive renames.

```
rlc/
├── __init__.py          # import rlc as torch
├── tensor.py            # re-exports from ravana.tensor
├── graph.py             # re-exports from ravana.graph
├── pressure.py          # re-exports from ravana.pressure
├── plasticity.py        # re-exports from ravana.plasticity
├── propagation.py       # re-exports from ravana.propagation
├── nn/
│   ├── __init__.py      # Module, Linear, Embedding, RLM
│   ├── module.py        # re-exports from ravana.nn.module
│   ├── functional.py    # re-exports from ravana.nn.functional
│   └── rlm.py           # re-exports from ravana.nn.rlm
├── cognitive/
│   ├── __init__.py      # re-exports all 23+ cognitive modules
│   ├── framework.py     # NEW — CognitiveFramework API
│   └── ...              # re-exports from ravana-v2/core/
├── world/               # re-exports from ravana.world
├── lab/                 # re-exports from ravana.lab
├── pyproject.toml       # pip install -e .
└── README.md
```

**Install:** `pip install -e rlc/`
**Import:** `import rlc as torch` or `from rlc.cognitive import CognitiveFramework`

---

## CognitiveFramework API (NEW)

The top-level user interface that wires the ML framework and cognitive core together:

```python
from rlc.cognitive import CognitiveFramework

fw = CognitiveFramework()
state = fw.initialize()

# Core cycle
concepts = fw.perceive(state, input_vec)      # input → active concepts
predictions = fw.predict(state, concepts)      # Hebbian spread → predictions
state = fw.learn(state, predictions, outcomes) # pressure + governor + emotion

# Consolidation
state = fw.sleep(state)                        # 4-stage consolidation

# Inference (no state change)
result = fw.infer(state, input_vec)            # {concepts, predictions, coherence}

# Memory
neighbors = fw.query(state, concept_id)        # graph neighborhood

# Diagnostics
report = fw.diagnose(state)                    # full cognitive dashboard
```

**What it wires together:**
- `ConceptGraph` + `PropagationEngine` (from ravana/) for perception and prediction
- `Governor` + `Identity` + `Resolution` for state regulation
- `VADEmotionEngine` for affective tagging
- `SleepConsolidation` for periodic consolidation
- `MeaningEngine` for intrinsic motivation
- `GlobalWorkspace` for inter-module coordination
- `HebbianPlasticity` + `StructuralPlasticity` for graph-level learning

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
from ravana.graph import ConceptGraph

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

**Integration:** Wired into `StateManager.step()` — stores each cognitive episode with emotional salience from VAD state. Sleep cycle triggers `apply_decay()` + `consolidate()`. Participates in Global Workspace competition via `compute_gw_bid()`.

**API:**
```python
from rlc.cognitive import HumanMemoryEngine, HumanMemoryConfig

engine = HumanMemoryEngine(HumanMemoryConfig(db_path="my_memory.db"))

# Store and recall
mid = engine.remember("Python is great for AI", tags="python,ai", importance=0.8)
results = engine.recall("python")
results = engine.semantic_search("machine learning")

# Reconstructive modes
blended = engine.blended_recall("programming")
abstract = engine.abstraction_recall("code", abstraction_level="abstract")
schema = engine.reconstruct_schema()

# Entropy and decay
engine.apply_decay()   # Ebbinghaus sweep + auto-consolidation
engine.consolidate()   # episodic → semantic promotion
status = engine.get_status()  # includes entropy stats
```

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
- `import rlc`, tensor creation, nn.Linear, ConceptGraph via rlc.graph
- Cognitive modules (Governor, Emotion, GlobalWorkspace)
- GlobalWorkspace bidding and broadcast
- CognitiveFramework: init, perceive, learn, infer, sleep, diagnose
- HumanMemoryEngine: remember, recall, decay, reconstructive modes
- End-to-end: forward + pressure + sleep_cycle
- `import rlc as torch` alias

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
- ~~Two disconnected codebases~~ → Unified via `rlc/` package
- ~~No packaging infrastructure~~ → `rlc/pyproject.toml` exists, `pip install -e .` works
- ~~Global Workspace missing~~ → `global_workspace.py` implemented and wired into StateManager
- ~~Framework API not built~~ → `CognitiveFramework` class implemented with full API

### Phase 2.5 (bridging Phase 2 → Phase 3)
- [x] Hierarchical abstraction compression — **DONE**
- [x] Human memory engine (Phase O) — **DONE** (persistent, decay, consolidation, reconstructive recall)
- [ ] Identity interaction — memories shaped by identity state, not just emotional/utility
- [ ] Replay-driven memory reshaping — sleep cycle actively rewrites memories, not just decay
- [ ] Memory fragmentation — single memory splits into parts under cognitive pressure
- [ ] Cross-episode narrative stitching — sequential episodes linked into coherent stories
- [ ] Latent manifold stabilization (Identity engine tracks concept embedding trajectories)
- [ ] Structural replay metrics (measure abstraction depth, concept reuse, cross-domain transfer)

### Remaining
1. **RLM has partial backprop** — `rlm.py` line 314 uses `context_logits.backprop()` despite "no backprop" claim. Limited to context logits head only.
2. **Cross-domain transfer at 0.0** — `exp3_cross_domain.json` shows `transfer_efficiency: 0.0`, status: `"NARROW"`
3. **Paper claims vs results mismatch** — dissonance trajectory in paper (0.800→0.200) doesn't match `final_results.json` (0.323→0.322)
4. **No formal benchmarks** — no comparison scripts against PyTorch or other baselines
5. **No root README** — docs split across 6+ files
6. **News-to-MDP pipeline unimplemented** — `reality_grounding.py` exists but structured cognitive event pipeline is a design

---

## Dependencies

**ravana/:** `numpy` (only)
**rlc/:** `numpy>=1.20` (only)
**ravana-v2/interface_agent/:** `feedparser>=5.2.1`, `requests>=2.31.0`, `newspaper3k>=0.2.8`, `openai>=1.12.0`, `anthropic>=0.18.0`
**ravana-v2/:** No requirements.txt (core modules use only stdlib + numpy)

---

## Git History

```
f2a65b3 RLM Phase I+ Pilot: Epistemic Resilience Confirmed
c23a148 Implement temporal context decay and hierarchical ambiguity resolution
00e45ba Architectural transition: separable cognitive forces and protected contextual causality
619f568 Pressure system now drives sleep: contradiction routing + edge metabolism
a0a5865 Concept Physics Lab + compositional experiment
782d2c2 RLM converges: 5/5 exact-match generation, 9/9 causal edges
d7d0f18 RAVANA framework: PyTorch-like API with pressure-driven learning + RLM
8437584 Initial commit: RAVANA v2 GRACE architecture + cognitive modules
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
- A unified package (`rlc`) with a user-facing CognitiveFramework API
- A system with reconstructive memory that doesn't just store — it rebuilds, distorts, consolidates, and forgets
- An active research project with empirical validation in constrained environments
- A prototype — not yet AGI, but proposing a novel path toward it

---

*Updated 2026-05-19 (Phase O: Human Memory). Share freely with LLM collaborators for guidance on next steps.*
