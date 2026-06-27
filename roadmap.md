# Ravana 2.0 Roadmap: CPU-Efficient Cognitive Architecture

> "The brain runs on ~20W and ~100Hz neural firing. It succeeds not through brute
> computation but through sparse, local, event-driven algorithms."
>
> **Principle:** Compute only what matters, only when needed, using local information.

---

## Existing Ecosystem (Do Not Rewrite)

The codebase already has **three published PyPI packages** (`ravana-ml`, `ravana-grace`, `ravana-chat`)
and ~18,000 lines of working code. The audit shows **~70% is reusable**:

| Package | Lines | Reuse | Role in v2 |
|---------|-------|-------|------------|
| `ravana_ml/graph.py` — **ConceptGraph** | 3,892 | ✅ As-is | Core graph with Hebbian plasticity & typed edges |
| `ravana_grace/core/` — **GRACE v2** | ~36 modules | ✅ As-is | Governor, StateManager, PredictiveWorldModel, BeliefReasoner |
| `ravana/src/ravana/language/` — **Language pipeline** | 3,525 | ✅ As-is | BasalGanglia, cerebellar n-grams, syntactic assembly, realizer |
| `ravana/src/ravana/core/` — **Cognitive primitives** | ~3,700 | ✅ Refactor | Emotion, identity, sleep, mirror neurons, causal schema |
| `ravana_ml/nn/rlm_v2*.py` — **RLM** | ~7 files | ✅ Refactor | Relation predictors (off custom tensor API → numpy) |
| `scripts/ravana_chat.py` — **Monolithic engine** | 9,462 | ❌ Decompose | Duplicates BeliefStore, SearchEngine, CognitiveResponseContext from core/ |
| `ravana_ml/tensor.py` — **Custom tensor API** | 389 | ❌ Deprecate | Use numpy/PyTorch directly |
| `ravana/__init__.py` — **Path surgery** | 158 | ❌ Replace | Flat package structure |

### Key Finding: Massive Duplication in `scripts/ravana_chat.py`

The 9,462-line monolith contains its own copies of:
- `BeliefStore` — duplicated from `ravana/src/ravana/core/belief_store.py`
- `CognitiveResponseContext` — duplicated from `ravana/src/ravana/chat/interface.py`
- `SearchEngine` — duplicated from `ravana/src/ravana/web/`
- `TEEN_CONCEPTS` (280 lines), `STOP_WORDS`, `WEB_GARBAGE` — in both scripts

**First quick win:** Delete these duplicates and import from the canonical modules.

---

## Neuroscience Framework

| Framework | Key Reference | What It Gives Us | Existing Code That Maps |
|-----------|---------------|------------------|------------------------|
| **Predictive Coding / Free Energy** | Friston (2010), PC-Network (NeurIPS 2024) | Local learning rules, no backprop | `ravana_ml/graph.py` plasticity rules, `ravana_ml/free_energy.py` |
| **Vector Symbolic Architectures** | Plate (1995), Kanerva (2009), Gayler (2003) | Compositional binding via simple vector ops | NEW — `holovec` library, or custom NumPy ops |
| **Semantic Pointer Architecture** | Eliasmith (2013), Spaun | Unified cognitive architecture on CPU | `ravana_grace/core/` Governor + StateManager pattern |
| **Coherence-Based Reasoning** | Thagard (1989, 2000), ECHO | Truth via constraint satisfaction | `ravana/src/ravana/core/belief_store.py` contradiction detection (weak) |
| **Dual-Process Theory** | Kahneman (2011), Common Model of Cognition | System 1 fast / System 2 slow | `ravana/src/ravana/core/dual_process.py` |
| **Hippocampal Replay** | Buzsáki (1998), Singh et al. (2022) | Sleep consolidation via replay | `ravana/src/ravana/core/sleep.py`, `ravana_grace/core/sleep.py` |
| **Fast Hebbian Plasticity** | Lansner, Fiebig (2017, 2020) | Working memory as temporary assemblies | `ravana/src/ravana/language/basal_ganglia.py` gating |
| **Lancaster Sensorimotor Norms** | Lynott et al. (2019) | Grounding without sensors | Replaces GloVe vectors in `ConceptGraph` |

---

## The 16-Point Redesign

### 1. Language Generation: Schema Completion (Replace Templates)

**Problem:** Current `_graph_fallback_response` uses hardcoded templates like
`"{subject} {verb} {label}"`. No learned generation.

**Neuroscience:** Broca's area activates a syntactic schema and fills slots from working
memory. Basal ganglia chunk action sequences into reusable units.

**Implementation:**
- ~20–50 learned utterance templates (S-V-O, S-V-O-Adv, S-V-Adv-O, etc.)
- Stored as VSA role-filler bindings
- Slots filled from concept node syntactic roles (learned via Hebbian statistics)
- Templates strengthen/weaken via counting after each interaction
- **CPU cost:** O(schemas × slots), ~10–50 μs per utterance

**Existing code to reuse:**
- `ravana/src/ravana/language/surface_realizer.py` (618 lines) — generative sentence production
- `ravana/src/ravana/language/syntactic_cell_assembly.py` (584 lines) — grammatical role binding
- `ravana/src/ravana/language/verb_lexicon.py` (740 lines) — Hebbian verb selection
- `ravana/src/ravana/language/basal_ganglia.py` (320 lines) — concept selection gating

**What to add:** Schema library with VSA binding for slot representation.

**Reference:** Levelt's "Speaking" model (1989); Dell's interactive activation model (1986).

---

### 2. Embedding Learning: Predictive Coding (No Backprop)

**Problem:** Current vectors are GloVe/hash initialized and never updated.
`_prediction_error` only adjusts edge weights, not node vectors.

**Neuroscience:** Predictive coding — each cortical area predicts activity of the area
below. Only prediction errors propagate. Learning is local.

**Implementation:**
```python
class ConceptNode:
    def __init__(self, label, dim=256):
        self.vector = np.random.randn(dim) * 0.1
        self.predictor = np.random.randn(dim, dim) * 0.01

    def predict(self, context_vector):
        return self.predictor @ context_vector

    def learn(self, context_vector, actual, lr=0.001):
        error = actual - self.predict(context_vector)
        self.predictor += lr * np.outer(error, context_vector)
        self.vector += lr * 0.1 * error
```
- No optimizer, no autograd, no backprop — just outer products
- **CPU cost:** O(nodes × dim²), nanoseconds per update

**Existing code to reuse:**
- `ravana_ml/graph.py` (3,892 lines) — already has node/edge structure, spreading activation
- `ravana_ml/plasticity.py` — Hebbian plasticity rules
- `ravana_ml/free_energy.py` — free energy computation

**What to add:** Per-node predictor matrix, local PC learning loop replacing the current
global `_prediction_error`.

**Reference:** Friston (2010); arXiv 2502.08860 (PC-Network, runs on laptop CPU);
CEREBRUM (github.com/nazmiefearmutcu/Cerebrum — backprop-free PC in pure NumPy).

---

### 3. Dialogue Coherence: Prefrontal Working Memory

**Problem:** Current `_response_context` is a flat list with no capacity limit, no decay,
no content-addressable retrieval.

**Neuroscience:** Prefrontal cortex maintains ~4–7 chunks via fast Hebbian plasticity.
Synapses rapidly strengthen to form temporary cell assemblies, then decay with activity.

**Implementation:**
```python
class WorkingMemory:
    def __init__(self, capacity=5, dim=256):
        self.slots = []          # (vector, timestamp, tag)
        self.capacity = capacity
        self.decay_rate = 0.1    # per retrieval

    def push(self, vector, tag="topic"):
        if len(self.slots) >= self.capacity:
            self.slots.pop(0)    # FIFO
        self.slots.append((vector, time.time(), tag))

    def retrieve(self, query):
        sims = [cosine_sim(query, v) for v, _, _ in self.slots]
        return self.slots[np.argmax(sims)]

    def decay(self):
        for i, (v, t, tag) in enumerate(self.slots):
            self.slots[i] = (v * (1 - self.decay_rate), t, tag)
```
- 5-slot hard limit forces forgetting
- Content-addressable retrieval by concept similarity
- Activity-dependent decay (STP/STDP biology)
- **CPU cost:** O(capacity × dim), <1 μs

**Existing code to reuse:**
- `ravana/src/ravana/core/hippocampal_buffer.py` (324 lines) — episodic buffer, already has `get_state/set_state`
- `ravana/src/ravana/language/prefrontal_workspace.py` (949 lines) — discourse planning

**What to add:** WorkingMemory class with VSA binding for context gating.

**Reference:** Baddeley's multicomponent model (2012); Fiebig & Lansner (2017, eNeuro 2020);
Cowan's 4-chunk limit.

---

### 4. Truth Verification: Coherence Constraint Satisfaction

**Problem:** Current `BeliefStore.detect_contradiction` only catches direct opposite-string
assertions. No semantic contradiction detection.

**Neuroscience:** Thagard's ECHO model — propositions are accepted/rejected by maximizing
coherence in a constraint satisfaction network.

**Implementation:**
```python
class CoherenceNetwork:
    def __init__(self):
        self.propositions = {}
        self.constraints = []   # (id1, id2, weight, sign)

    def settle(self, max_iter=100):
        activations = {p.id: 0.01 for p in self.propositions.values()}
        for _ in range(max_iter):
            for p1, p2, w, sign in self.constraints:
                activations[p1] += w * sign * activations[p2] * 0.1
            for pid in self.evidence_ids:
                activations[pid] = 1.0  # clamp evidence
            for pid in activations:
                activations[pid] *= 0.95  # decay
        return {pid: a for pid, a in activations.items() if a > 0.5}
```
- Proposition accepted if settles to activation > 0.5
- Contradictions = inhibitory constraints
- Evidence clamped to 1.0; competing hypotheses evaluated
- **CPU cost:** O(constraints × iterations), ~100 μs for 1K propositions × 100 iterations

**Existing code to reuse:**
- `ravana/src/ravana/core/belief_store.py` (242 lines) — belief tracking, `ContradictionWarning`
- `ravana/src/ravana/core/proposition_parser.py` (234 lines) — nested proposition extraction
- `ravana/src/ravana/core/causal_schema.py` (288 lines) — causal structure learning

**What to add:** CoherenceNetwork with VSA-based proposition encoding.

**Reference:** Thagard "Coherence as Constraint Satisfaction" (1998); ECHO implementation;
Thagard "Evaluating Explanations in Law, Science, and Everyday Life" (2006).

---

### 5. Compositional Semantics: Vector Symbolic Architectures

**Problem:** Current `_understand_query` does keyword → concept matching. "dog bites man"
and "man bites dog" produce the same bag-of-concepts.

**Neuroscience:** Neural synchrony binds features — equivalent to circular convolution in
VSA. Role-filler binding distinguishes "dog bites man" ≠ "man bites dog."

**Implementation:**
```python
DIM = 512

def bind(a, b):
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b))

def unbind(c, b):
    return np.fft.irfft(np.fft.rfft(c) * np.conj(np.fft.rfft(b)))

# "dog bites man" = BITE + R0⊙DOG + R1⊙MAN
# "man bites dog" = BITE + R0⊙MAN + R1⊙DOG  ← different vector
```
- MAP or FHRR binding: 3–4× faster than circular convolution (Carzaniga et al., PMLR 2025)
- **CPU cost:** O(DIM log DIM), ~5 μs per operation

**What to add:** Use `holovec` (github.com/Twistient/HoloVec) — NumPy backend, 8 VSA
models supported, 95%+ test coverage.

**Reference:** Plate (1995); Kanerva (2009); Gayler (2003); Carzaniga et al. (2025)
benchmarks.

---

### 6. Actual Reasoning: Dual-Process (System 1 + System 2)

**Problem:** Current `_walk_chain` is a weighted random walk — no actual inference.

**Neuroscience:** System 1 (fast, intuitive pattern completion) + System 2 (slow, mental
model simulation, counterfactuals).

**Implementation:**
```
Input → System 1 (attractor-based graph settle, <50ms)
  ↓ confidence > threshold? → Respond immediately
  ↓ else
System 2 (mental model simulation, 500ms–2s)
  - Extract causal subgraph (20–50 nodes)
  - Forward simulation: "if A then B then C..."
  - Counterfactual: flip edge weight, re-settle
  ↓ Respond
```
- System 1: constraint satisfaction settle (from point #4), entropy of final activations
  = confidence
- System 2: mental model via `causal_schema.py` forward simulation
- **CPU cost:** S1 ~10ms, S2 ~100ms (occasional)

**Existing code to reuse:**
- `ravana/src/ravana/core/dual_process.py` (106 lines) — System 1/2 routing controller
- `ravana/src/ravana/core/causal_schema.py` (288 lines) — causal simulation
- `ravana/src/ravana/core/abstraction_engine.py` (336 lines) — concept hierarchy walking
- `ravana_grace/core/predictive_world_model.py` — mental simulation (v2)

**What to add:** Threshold-gated System 1→System 2 escalation; counterfactual generation.

**Reference:** Kahneman (2011); Evans (2008); Johnson-Laird (2001); Conway-Smith & West
(2023).

---

### 7. Sensory Grounding: Sensorimotor Norm Vectors

**Problem:** All concepts anchored to GloVe/text with no experiential basis.

**Neuroscience:** Concepts grounded in sensorimotor experience — even abstract concepts
activate sensory/motor areas. "Grasp" activates hand areas; "warm" activates temperature.

**Implementation:** Replace GloVe vectors with **Lancaster Sensorimotor Norms** — 40K
English words rated on 11 dimensions:

```
Touch, Hear, Smell, Taste, Sight,
Foot/Leg, Hand/Arm, Mouth/Throat, Head, Torso,
plus Interoception
```

```python
sm_vector = lancaster_norms["grasp"]    # high Hand/Arm
concept_vector = np.concatenate([sm_vector, vsa_vector])  # 11 + 256 = 267-dim
```

- No download needed (embed CSV in package)
- **CPU-friendly:** 267-dim vectors (vs 300-dim GloVe)
- LANCASTER_NORMS.csv embedded in `ravana2/data/`

**Reference:** Lynott et al. (2019); Kennington (2021, CoNLL).

---

### 8. Working Memory Constraints

Working memory from point #3, with addition:

- **Context binding:** Each slot binds its content to a context vector via VSA binding.
  New topic → all slots unbound and decay.
- **Interference:** Similar items in adjacent slots mutually inhibit (cosine similarity →
  inhibition strength).

**Existing code to reuse:**
- `ravana/src/ravana/core/hippocampal_buffer.py` (324 lines) — pattern completion on
  partial cues
- `ravana/src/ravana/language/prefrontal_workspace.py` (949 lines) — discourse-level
  planning

---

### 9. Single-Threaded GIL: Event-Driven Async Loop

**Problem:** Background learning uses `threading.Thread` but GIL locks all Python
compute.

**Neuroscience:** Brain is parallel at ~100Hz per neuron. For CPU: never spin-wait,
never block on compute.

**Implementation:**
```python
class CognitiveLoop:
    def __init__(self):
        self.event_queue = asyncio.Queue()

    async def run(self):
        while True:
            event = await self.event_queue.get()
            if event.priority == "high":
                await self._process_high_priority(event)
            else:
                asyncio.create_task(self._process_background(event))
            await asyncio.sleep(0)  # yield
```
- All compute via NumPy (releases GIL via BLAS)
- Web fetches via `aiohttp` — true non-blocking I/O
- Idle = 0% CPU
- System 2 runs as `asyncio.create_task`, yielding after each step

---

### 10. Offline Consolidation: Hippocampal Replay

**Problem:** Current `_run_sleep_cycle` only adjusts edge weights by `stability` factor.
No replay, no reorganization.

**Neuroscience:** NREM sleep: hippocampus replays recent experiences (sharp-wave
ripples), strengthening cortical patterns. REM sleep: recombines experiences into new
generalizations.

**Implementation:**
```python
class SleepConsolidation:
    def __init__(self):
        self.replay_buffer = []    # (pair, context, weight, timestamp)
        self.consolidated = {}

    async def sleep_cycle(self):
        # Phase 1: NREM — replay recent 100 experiences
        for exp in self.replay_buffer[-100:]:
            self._strengthen_edges(exp)

        # Phase 2: Interleaved — mix new with old
        for old_exp in self.consolidated.values():
            if random() < 0.3:
                self._strengthen_edges(random.choice([old_exp, *sample]))

        # Phase 3: Prune weak connections (< 0.1)
        for concept, strength in self.consolidated.items():
            if strength < 0.1:
                self._prune(concept)
```
- Forward replay → preserves temporal order
- Reverse replay → enables causal inference
- **CPU cost:** ~500ms–2s during idle, then 0%

**Existing code to reuse:**
- `ravana/src/ravana/core/sleep.py` (188 lines) — consolidation + pruning
- `ravana_grace/core/sleep.py` (1,726 lines) — comprehensive v2 sleep with dream
  consolidation
- `ravana_ml/nn/rlm_v2_sleep.py` — RLM-specific sleep consolidation

**What to add:** Replay buffer with priority sampling; forward/reverse replay modes.

**Reference:** Buzsáki (1998); Singh, Norman & Schapiro (2022); De Vita (2025).

---

### 11. Web Scraping: Structured Extraction

**Problem:** Current regex HTML stripping is fragile. No readability algorithm.

**Fix:** Replace with `trafilatura`:

```python
import trafilatura

text = trafilatura.extract(html, output_format='txt',
                           favor_precision=True)
```

- Statistical heuristics, no JS rendering needed
- Falls back to readability-lxml
- CPU-only, ~100ms per page
- No ML dependency

---

### 12. Controlled Generation: Register Modulation

**Problem:** No control over factuality, formality, length, or style.

**Neuroscience:** Prefrontal cortex sets "registers" that modulate language production.

**Implementation:**
```python
class RegisterController:
    REGISTERS = {
        "casual":   {"formality": 0.2, "verbosity": 0.3, "certainty": 0.5},
        "didactic": {"formality": 0.7, "verbosity": 0.8, "certainty": 0.9},
        "terse":    {"formality": 0.5, "verbosity": 0.1, "certainty": 0.6},
    }
```
- Three knobs modulate schema completion
- Learned from user feedback via REINFORCE (7-parameter policy gradient)
- **CPU cost:** Negligible

**Reference:** Hymes (1974); register theory in sociolinguistics.

---

### 13. Metacognition: Epistemic Curiosity Engine

**Problem:** Current `_Step_8_meta_cognitive` only logs `AOT_horizon`. No directed
learning.

**Neuroscience:** Anterior cingulate monitors prediction error. High error →
norepinephrine release → switch from exploitation to exploration.

**Implementation:**
```python
class CuriosityEngine:
    def uncertainty_for(self, concept):
        pe = self.prediction_errors.get(concept, 1.0)
        visits = self.visit_counts.get(concept, 0)
        return pe * np.exp(-visits * 0.1)

    def suggest_exploration(self):
        return max(self.prediction_errors, key=self.uncertainty_for)
```
- Drives what `ravana_learn.py` searches for
- **CPU cost:** O(1) per concept

**Existing code to reuse:**
- `ravana/src/ravana/core/meta_cognition.py` (100 lines) — reasoning bias, confidence
- `ravana_grace/core/meta_cognition.py` (434 lines) — v2 metacognition
- `ravana_grace/core/meta2_cognition.py` — v2 meta²-cognition

**Reference:** Friston free energy principle; Oudeyer & Kaplan intrinsic motivation (2007).

---

### 14. Persistence: SQLite

**Problem:** Pickle is version-specific, fragile. JSON `default=str` silences errors.

**Fix:** SQLite with WAL mode:
```sql
CREATE TABLE concepts (id TEXT PRIMARY KEY, vector BLOB, sensorimotor BLOB, created_at REAL);
CREATE TABLE edges (source_id TEXT, target_id TEXT, weight REAL, edge_type TEXT);
CREATE TABLE episodes (id INTEGER PRIMARY KEY, timestamp REAL, content TEXT, concepts TEXT);
```
- ACID, cross-platform, version-safe
- NumPy vectors stored as float32 binary blobs
- WAL mode = concurrent reads during writes
- **CPU cost:** ~10 μs per operation

**What to add:** Migration script from pickle/JSON to SQLite.

---

### 15. Package Architecture: Clean Module Structure

**Problem:** Three overlapping packages with `sys.path` surgery.

**Fix:** Single clean `ravana2` package (new namespace — or replace `ravana` package with
flat structure):

```
ravana2/
├── __init__.py              # Clean exports, NO sys.path surgery
├── core/
│   ├── concepts.py          # ConceptNode, ConceptGraph (wraps ravana_ml)
│   ├── working_memory.py    # 5-slot PFC buffer
│   ├── coherence.py         # ECHO constraint satisfaction
│   ├── predictive_coding.py # Local Hebbian learning
│   └── sleep.py             # Hippocampal replay + consolidation
├── language/
│   ├── schemas.py           # Utterance schema library
│   ├── realizer.py          # Schema completion → text
│   └── register.py          # Register control
├── reasoning/
│   ├── system1.py           # Fast attractor-based completion
│   └── system2.py           # Slow mental model simulation
├── learn/
│   ├── curiosity.py         # Epistemic curiosity engine
│   ├── web.py               # trafilatura extraction
│   └── consolidation.py     # Offline replay
├── storage/
│   ├── db.py                # SQLite persistence
│   └── vectors.py           # Sensorimotor + VSA encodings
├── interface/
│   ├── chat.py              # CLI chat loop
│   └── api.py               # Optional REST API
└── data/
    ├── lancaster_norms.csv
    └── schemas.json
```

- Single `pyproject.toml`, `pip install -e .` works immediately
- Each sub-package imports from `ravana_ml` and `ravana_grace` by clear path

**Existing code mapping:**
| New module | Source |
|-----------|--------|
| `core/concepts.py` | `ravana_ml/graph.py` ConceptGraph |
| `core/working_memory.py` | NEW |
| `core/coherence.py` | NEW + `ravana/src/ravana/core/belief_store.py` |
| `core/predictive_coding.py` | NEW + `ravana_ml/plasticity.py` |
| `core/sleep.py` | `ravana_grace/core/sleep.py` |
| `language/schemas.py` | `ravana/src/ravana/language/surface_realizer.py` |
| `language/realizer.py` | `ravana/src/ravana/language/syntactic_cell_assembly.py` |
| `reasoning/system1.py` | `ravana/src/ravana/core/dual_process.py` + ConceptGraph traversal |
| `reasoning/system2.py` | `ravana_grace/core/predictive_world_model.py` |
| `learn/curiosity.py` | `ravana_grace/core/meta_cognition.py` |
| `learn/web.py` | NEW (trafilatura + existing `SearchEngine`) |
| `storage/db.py` | NEW (SQLite) |

---

### 16. Evaluation: Cognitive Benchmarks

**Problem:** No test suite for cognitive outputs. No factual accuracy, coherence, or
dialogue metrics.

**Implementation (built-in, computed during idle):**
- **Perplexity:** held-out concept sequence prediction (is the graph predictive?)
- **Coherence score:** attractor stability across 100 random resettlements
- **Contradiction rate:** inhibitory constraints triggered per 100 inputs
- **Learning efficiency:** prediction error reduction per new web article
- **Response diversity:** type-token ratio over last 100 responses

**Existing code to reuse:**
- `tests/` directory in multiple packages already exists
- `ravana_grace/tests/` has test infrastructure

---

## Quick Wins: Immediate First Steps

These can be done in parallel, in a single afternoon:

1. **Delete duplicates from `scripts/ravana_chat.py`**
   - `BeliefStore` → import from `ravana/src/ravana/core/belief_store.py`
   - `CognitiveResponseContext` → import from `ravana/src/ravana/chat/interface.py`
   - `SearchEngine` → import from `ravana/src/ravana/web/`
   - `TEEN_CONCEPTS`, `STOP_WORDS`, `WEB_GARBAGE` → move to JSON data files
   - **Savings:** ~800 lines removed

2. **Add `pyproject.toml` to `ravana_grace`**
   - Already published on PyPI (`ravana-grace` v0.2.3) but no pyproject.toml in repo
   - Reproduce the build config from `PYPI_PUBLISHING.md`

3. **Extract `CognitiveChatEngine` (8,378 lines)** into 4 focused files:
   - `ravana/src/ravana/chat/orchestrator.py` — main pipeline
   - `ravana/src/ravana/chat/chain_walker.py` — graph traversal
   - `ravana/src/ravana/chat/web_learner.py` — web search + learning
   - `ravana/src/ravana/chat/response_gen.py` — response generation
   - Keep the class interface identical — just relocate methods

4. **Fix imports** — remove `sys.path` surgery, use flat package references

---

## Implementation Roadmap

```
Phase 0 — Quick Wins (1 week) [COMPLETED]
  ├── Delete duplicates from scripts/ravana_chat.py (-800 lines) [x]
  ├── Move data constants to JSON files [x]
  ├── Add pyproject.toml to ravana_grace [x]
  └── Split CognitiveChatEngine into 4 files (no behavior change) [x]

Phase 1 — New Modules (2 weeks) [COMPLETED]
  ├── Predictive coding learner (core/predictive_coding.py) [x]
  ├── Coherence network (core/coherence.py) [x]
  ├── Working memory (core/working_memory.py) [x]
  └── SQLite persistence (storage/db.py) [x]

Phase 2 — Language + Reasoning (2 weeks) [COMPLETED]
  ├── VSA binding utility (core/vsa.py — wraps holovec) [x]
  ├── Schema completion realizer (upgrade surface_realizer) [x]
  ├── System 1 attractor dynamics (upgrade graph walk) [x]
  └── System 2 mental model simulation (upgrade causal_schema) [x]

Phase 3 — Learning + Consolidation (1 week)
  ├── Curiosity engine (learn/curiosity.py)
  ├── Hippocampal replay (learn/consolidation.py)
  ├── Register controller (language/register.py)
  └── trafilatura web extraction (learn/web.py)

Phase 4 — Integration (1 week)
  ├── Wire all new modules into orchestrator
  ├── Add evaluation benchmarks
  ├── Migration script: pickle → SQLite
  └── Tests for each new module
```

---

## CPU Budget Estimate

| Component | Time | Frequency |
|-----------|------|-----------|
| System 1 (graph settle, 200 iter) | ~10 ms | Every turn |
| System 2 (mental simulation) | ~100 ms | Occasional (<10% of turns) |
| Working memory update | <1 μs | Every turn |
| Schema completion | ~50 μs | Every turn |
| Predictive coding update | ~1 ms | Every turn |
| Sleep cycle | ~1–2 s | Every N turns (idle) |
| Web fetch | ~100–500 ms | Async, non-blocking |
| **Active compute per turn** | **~20 ms** | — |
| **Idle CPU** | **~0%** | Event loop waiting |

Total: **~1–5% CPU** on a modern laptop. Not 30W, but <1W sustained.
