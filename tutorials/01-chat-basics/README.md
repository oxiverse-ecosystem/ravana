# Tutorial 01: Chat Engine Basics

**Part of a 7-tutorial progression. This is where you start.**

## What you'll learn

- Create a `CognitiveChatEngine` — the main orchestrator
- Understand what happens **during initialization** (seeding, GloVe, PMI)
- Send queries and trace the **6-stage turn pipeline**
- Read graph diagnostics (nodes, edges, sleep cycles)
- Save brain state for the next tutorial

---

## Run it

```bash
python tutorials/01-chat-basics/run.py
```

---

## Deep dive: what happens when you create the engine

The line `CognitiveChatEngine(dim=64, seed=42, baby_mode=True)` triggers a cascade
that's worth understanding because it reveals the entire architecture's design.

### Stage 0: Thread safety & BLAS pinning

Before anything else, `ravana._numpy_threading` is imported. This pins BLAS/OpenMP
to 1 thread. Why? Windows has a known NumPy race condition (issue #27989) where
worker-thread BLAS calls can crash the main thread. The fix is to ensure all
numpy operations run single-threaded. If you see access-violation errors, this
is the first thing to check.

### Stage 1: Graph seeding via PMI (Pointwise Mutual Information)

**File:** `ravana/src/ravana/chat/chain_walker.py` → `_seed_concepts()`

The engine needs a starting vocabulary — you can't learn from nothing. RAVANA
uses **PMI bootstrapping** from a seed corpus (`data/corpora/teen_seeds.txt`,
~296 teen-level English sentences):

1. **Load corpus** → `PMISeeder.load_corpus()` reads sentences
2. **Compute PMI** → for every word pair in the corpus, compute:
   ```
   PMI(w1, w2) = log(P(w1, w2) / (P(w1) * P(w2)))
   ```
   High PMI = words that co-occur more than chance (e.g., "trust" and "honesty")
3. **Create graph nodes** → top-200 concepts become nodes with 64-D GloVe vectors
4. **Create edges** → top-300 PMI pairs become typed edges (initially "semantic")
5. **Auto-wire GloVe-similar concepts** → any two concepts with cosine > 0.65
   get a semantic edge (low confidence, marked "dormant")
6. **Build hub wiring** → minimal cognitive-action hubs (i → think → know → learn)
7. **POS tagging** → every concept gets a part-of-speech tag (verb/noun/adj)
   based on suffixes and known-word lists
8. **Contradiction map** → antipodal GloVe pairs (< -0.15 cosine) become
   contrastive entries in the contradiction detector

**Neuroscience parallel:** This mirrors how the infant brain bootstraps its
first semantic networks through statistical learning of co-occurring stimuli
(Saffran, Aslin & Newport 1996). PMI is the computational analog of
co-occurrence statistics.

### Stage 2: GloVe embedding projection

**File:** `ravana/src/ravana/chat/chain_walker.py` → `_glove_vector()`

Pre-trained GloVe (100-D) vectors are projected to 64-D via QR decomposition.
The projection matrix is cached after the first run. Words not in GloVe get
a composite vector from constituent sub-words (e.g., "blockchain" → "block"+"chain").

### Stage 3: GRACE module initialization

**Files:** `ravana-v2/src/ravana_grace/core/` (emotion.py, identity.py, meaning.py, etc.)

The engine instantiates 6 cognitive modules:

| Module | Source file | Initial state | Purpose |
|--------|-------------|---------------|---------|
| `VADEmotionEngine` | `emotion.py` | V=0.0, A=0.3, D=0.5 | 3D affect dynamics |
| `IdentityEngine` | `identity.py` | strength=0.25 | Self-concept stabilization |
| `MeaningEngine` | `meaning.py` | accumulated=0.0 | Intrinsic motivation |
| `DualProcessController` | `dual_process.py` | S1 default | Fast/slow reasoning routing |
| `GlobalWorkspace` | `global_workspace.py` | empty buffer | Conscious broadcast |
| `MetaCognition` | `meta_cognition.py` | calibrator ready | Bias/confidence monitoring |

### Stage 4: Language module initialization

**Files:** `ravana/src/ravana/language/` (basal_ganglia.py, cerebellar_ngram.py, etc.)

- **BasalGangliaGate**: Go/NoGo gating for response selection (dopamine-modulated)
- **CerebellarNgram**: Sparse sequence model for grammatical transitions
- **PrefrontalWorkspace**: Discourse planning — holds subject + associations
- **SyntacticCellAssembly**: Hebbian role learning with grammatical frames
- **SurfaceRealizer**: Rule-governed English morphology (pluralization, tense,
  determiner selection, capitalization)
- **VerbLexicon**: Semantic verb selection using GloVe vectors

### Stage 5: Sleep engine initialization

The sleep engine (`SleepConsolidation` from `ravana_grace`) is configured with:
- `pressure_threshold=0.3` — how much prediction error triggers sleep
- `counterfactual_rate=0.15` — REM-like recombination during sleep
- `emotional_flip_rate=0.08` — valence reversal during dreaming

Sleep runs every 20 turns regardless of pressure, and every 300 seconds (5 min)
if idle. This is the "instead of `optimizer.step()`" mechanism.

---

## Deep dive: what happens when you call process_turn()

The single call `engine.process_turn("what is trust")` triggers a **6-stage pipeline**:

### Stage 1: Brain-repair prepasses

**File:** `ravana/src/ravana/chat/brain_regions.py`

Before any processing, a set of **prepasses** run on the raw input:

1. **SelfModel.from_graph()** — determine "who am I" / "who is the user"
   from the graph structure (vmPFC analog)
2. **EpisodicIndex()** — FIRST/LAST/BY_ENTITY recall from past turns
   (hippocampal time-cell analog)
3. **classify_cause()** — classify the user's emotional cause using
   GloVe-centroid proximity (e.g., "I'm sad because..." → loss/thematic)
4. **select_empathy_frame()** — if the user disclosed emotion, select an
   empathy response frame (tightly gated — only fires on state-disclosure syntax)
5. **humor_is_coherent()** — if the input looks like humor, check it for
   coherence using the learned salad classifier ORed with rule-based detection
6. **parse_number_phrase()** / **mirror_deictic()** / **consult_internal()**

All prepasses are **fail-closed**: if uncertain, they return None/false.

### Stage 2: Intent routing & gating

**Files:** `ravana/src/ravana/chat/intent_router.py`, `coherence_gate.py`,
`monitor_gate.py`, `junk_scorer.py`

The engine classifies the input into one of 6 intent classes:

| Intent | Example | Routing |
|--------|---------|---------|
| `chitchat` | "hello" | Template response |
| `factual` | "what is trust" | Graph walk → decoder |
| `hypothetical` | "what if..." | Causal chain walk |
| `identity` | "who are you" | Self-model query |
| `comparative` | "X vs Y" | Comparative analysis |
| `OOD` | gibberish | **Abstain** (honest "I don't know") |

Before routing, three gates check the input:
1. **CoherenceGate** — GloVe-cosine coherence floor (reject gibberish)
2. **MonitorGate** — abstention check (high free-energy → abstain)
3. **JunkScorer** — word-salad / degenerate-text rejection

If any gate fires, the engine returns an honest "I don't know" — it never
confabulates. This is the **fail-closed grounding** principle.

### Stage 3: Graph walk

**File:** `ravana/src/ravana/chat/chain_walker.py` → `_spread_and_collect()`

Once the intent is known, the engine walks the ConceptGraph:

1. **Seed activation** → the query's subject concept(s) get activation=1.0
2. **3-hop propagation** → activation spreads through typed edges, decaying
   by 0.7 per hop
3. **Relation bias** → if the intent is causal (e.g., "what happens if"),
   causal edges get amplified via GloVe-vector prototype matching
4. **Topic relevance gate** → associations too semantically distant from
   the subject are suppressed (pattern separation, Yassa & Stark 2011)
5. **Top-12 collection** → the highest-activated concepts are returned

During propagation, the engine also applies:
- **Degree suppression**: high-degree concepts (e.g., "thing") are penalized
- **Post-hoc relevance filter**: final cosine check against subject vector
- **Recency boost**: recently learned concepts get 1.5x activation

### Stage 4: Neural decoder

**File:** `ravana_ml/src/ravana_ml/nn/neural_decoder.py`

The graph-walk embedding is fed to the `NeuralDecoder` — a 64-D GRU that
generates a word sequence conditioned on the concept vector:

```
Graph embedding (64-D) → GRU hidden state → softmax → next word
```

The decoder does NOT generate tokens autoregressively like an LLM. It
generates a **short phrase** that realizes the activated concepts in natural
language. The vocabulary is small (~100-500 words from the seed corpus).

### Stage 5: Surface realization

**File:** `ravana/src/ravana/language/surface_realizer.py`

The decoder's raw output is polished by the `SurfaceRealizer`:
- Pluralization ("cat" → "cats" where appropriate)
- Determiner selection ("a" vs "an")
- Capitalization (first word, proper nouns)
- Register adjustment (casual vs formal via `RegisterController`)

### Stage 6: Background web learning (async)

If the engine detected a knowledge gap (high free-energy on the query),
it queues a web search to run **in the background** (separate thread).
The search results will be available on the next turn.

---

## Reading the diagnostic output

After each query, you see:

```
[nodes=635, edges=15219, sleep=43]
```

| Metric | What it means | Why it matters |
|--------|---------------|----------------|
| **nodes** | Number of concept nodes in the graph | More nodes = more knowledge |
| **edges** | Number of typed relationships | More edges = denser connections |
| **sleep** | Number of sleep cycles completed | More sleep = better consolidation |

The numbers start high because the engine loaded from a previous save file
(635 words remembered). For a fresh engine, expect ~180-200 nodes.

**Nodes vs edges ratio:** A ratio of ~1:24 (635 nodes, 15219 edges) means
each concept is connected to ~24 others on average. This is the graph's
"connectivity density."

---

## The `_numpy_threading` import (important for Windows)

The very first import in `engine.py` is:

```python
import ravana._numpy_threading  # noqa: F401
```

This sets `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and registers a
`faulthandler` for crash diagnostics. Without this, Windows users may see
access-violation crashes when the background web-learning thread calls numpy
while the main thread is also in a numpy operation. The single-thread pinning
is a workaround for a known numpy race condition.

---

## Expected output (annotated)

```
  [GloVe] Loaded 400000 projected vectors from cache (100D -> 64D)
  └── GloVe cache hit — vectors were projected and saved on first run

  [Loaded] Remembered 635 words from before!
  └── Previous save file found — engine restored from data/ravana_weights.pkl

Q: what is trust
A: a legal that creates a lien on some specific item of inventory
  [nodes=635, edges=15219, sleep=43]
  └── The response quality depends on how much the engine has learned.
      Initial responses may seem odd — they improve with training.

Q: tell me about love
A: i wish i could write that...
  [nodes=635, edges=15219, sleep=43]

Q: what is memory
A: Memory is a powerful yet seemingly imperfect brain...
  [nodes=635, edges=15219, sleep=43]

[OK] State saved to data/ravana_weights.pkl
  └── Ready for Tutorial 02
```

---

## Design philosophy notes

1. **No LLM, no pretrained model.** Everything you see is generated by a
   tiny GRU decoder + concept graph walk. The system learns from scratch.
2. **Fail-closed.** If the engine doesn't know, it says "I don't know" rather
   than making something up. This is non-negotiable — see `monitor_gate.py`.
3. **Sleep replaces backprop.** The engine doesn't use gradient descent.
   Learning happens through free-energy minimization during sleep cycles.
4. **Graph is the knowledge store.** There are no weight matrices holding
   learned facts. Every fact is a typed edge between two concept nodes.

---

## Key source files reference

| Component | File (relative to repo root) |
|-----------|------------------------------|
| Engine init | `ravana/src/ravana/chat/engine.py` (line 182, `CognitiveChatEngine.__init__`) |
| Graph seeding | `ravana/src/ravana/chat/chain_walker.py` (line 38, `_seed_concepts`) |
| Turn processing | `ravana/src/ravana/chat/engine.py` (line ~700, `process_turn`) |
| Brain repair | `ravana/src/ravana/chat/brain_regions.py` |
| Intent routing | `ravana/src/ravana/chat/intent_router.py` |
| Coherence gate | `ravana/src/ravana/chat/coherence_gate.py` |
| GRACE modules | `ravana-v2/src/ravana_grace/core/` |
| Sleep consolidation | `ravana-v2/src/ravana_grace/core/sleep.py` |

---

## Next tutorial

[**Tutorial 02: Decoder Training**](../02-decoder-training/) — load this engine's
saved state, train the neural decoder, and watch the accuracy metrics improve.
