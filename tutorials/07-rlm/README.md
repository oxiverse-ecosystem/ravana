# Tutorial 07: RLM / RLMv2 Usage

**Part of a 7-tutorial progression. Final tutorial: ML model internals.**

## What you'll learn

- Import and instantiate `RLMv2` — the relation learner model
- Understand **triple decomposition**: (subject, relation, object)
- Learn how **verb-stem offsets** enable analogical reasoning
- See where RLMv2 fits in the broader system and how it benchmarks

---

## Run it

```bash
python tutorials/07-rlm/run.py
```

---

## Deep dive: what is RLMv2?

**RLMv2** = Recursive Learning Model, version 2.

**File:** `ravana_ml/src/ravana_ml/nn/rlm_v2.py` (line 214)

RLMv2 is a **triple decomposition model** — it learns relationships between
concepts as (subject, relation, object) triples:

```
"heat causes expansion"  →  (heat, causes, expansion)
"water is liquid"        →  (water, is, liquid)
"love is not hate"       →  (love, contrastive, hate)
```

### Why triples?

Triple decomposition is the **native format for the ConceptGraph**. Every typed
edge in the graph is already a triple:

```python
# Graph edge (source_node, target_node) = (subject, object)
edge.relation_type = "causal"  # = relation
# Full triple: (heat, causal, expansion)
```

By explicitly learning triples, RLMv2 can:
1. **Predict missing objects** — given (heat, causes, ?) → predict "expansion"
2. **Generalize via verb offsets** — apply the "causes" pattern to new subjects
3. **Transfer across domains** — learn causal patterns from science, apply to social

### Architecture

RLMv2 is composed of 5 mixins, each handling a different aspect:

| Mixin | File | Purpose |
|-------|------|---------|
| `EncoderMixin` | `rlm_v2_encoder.py` | Encodes input tokens into embeddings |
| `GraphMixin` | `rlm_v2_graph.py` | Graph operations (neighbor lookup, walk) |
| `RPMixin` | `rlm_v2_rp.py` | Relation prediction (W_rel matrix) |
| `VerbMixin` | `rlm_v2_verb.py` | Verb-stem offset computation |
| `SleepMixin` | `rlm_v2_sleep.py` | Sleep consolidation (SWS + REM) |

The model doesn't use attention or matrix multiplication. Its inference
mechanism is **spreading activation through the graph** — the same mechanism
used by the chat engine.

---

## Deep dive: verb-stem offsets

The key innovation in RLMv2 is the **verb-stem offset predictor**.

### The idea

Each verb in the vocabulary has a learned **offset vector**:

```
offset("causes") = avg(target_embed - subject_embed) for all "causes" triples
```

This offset captures the **semantic transformation** that the verb applies.
For example, if the model has seen:

```
(heat, causes, expansion)     →  target - subject = expansion_embed - heat_embed
(pressure, causes, compression) →  target - subject = compression_embed - pressure_embed
```

Then `offset("causes")` is the average of these differences — a vector that
represents "what it means for something to cause something else."

### Vector arithmetic analogy

Once offsets are learned, the model can do **analogical reasoning**:

```
subject_embed + offset(verb) ≈ predicted_target_embed
```

Example: If the model knows `heat + causes ≈ expansion`, it can infer
`cold + causes ≈ ?` by computing `cold_embed + offset("causes")` and finding
the nearest concept in embedding space.

### Example analogies the model can learn

| Known triple | Query | Predicted |
|-------------|-------|-----------|
| (heat, causes, expansion) | (cold, causes, ?) | contraction |
| (love, is, emotion) | (trust, is, ?) | emotion |
| (sunrise, then, daylight) | (fertilization, then, ?) | embryo |

### Cross-domain transfer

Because verb offsets are **verb-specific, not domain-specific**, the model can
transfer causal knowledge across domains:

1. Train on science domain: (heat, causes, expansion), (cold, causes, contraction)
2. Apply to social domain: (anger, causes, ?) → conflict (using the same "causes" offset)

This is measured by the `scripts/diagnose_transfer.py` benchmark.

---

## Deep dive: RLMv2 sleep consolidation

**File:** `ravana_ml/src/ravana_ml/nn/rlm_v2_sleep.py`

Like the chat engine, RLMv2 has its own sleep cycle — modeled on mammalian
SWS (Slow-Wave Sleep) and REM sleep.

### SWS Stage (structural consolidation)

During SWS, the model:
1. **Strengthens** low-prediction-error edges (they're correct, make them stronger)
2. **Weakens** high-prediction-error edges (they're wrong, reduce their influence)
3. **Forms inhibitory edges** — when two concepts are consistently NOT co-active,
   an inhibitory edge is formed to prevent spurious activation
4. **Specializes hubs** — high-degree concepts develop sharper selectivity

### REM Stage (creative recombination)

During REM, the model:
1. **Reverses 20% of edges** — counterfactual: what if the relationship were opposite?
2. **Flips emotional valence** on 10% of tagged concepts
3. **Oversamples failures** at 1.5× rate — learn more from mistakes
4. **Creates new prototype combinations** — blend two existing concepts

### Sleep-time interleaved replay

Domain-tagged experiences are buffered during training and replayed during
SWS+REM. This is the key mechanism that **eliminates catastrophic forgetting**
(12% → 0% retention drop) in lifelong streaming benchmarks.

---

## RLMv2 benchmark results

The model is evaluated by several benchmarks:

| Benchmark | Script | What it measures |
|-----------|--------|-----------------|
| Held-out generalization | `scripts/validate_held_out_generalization.py` | Can the model predict triples with unseen subjects/verbs? |
| Cross-domain transfer | `scripts/diagnose_transfer.py` | Can causal patterns from Science apply to Social? |
| Within-domain triple | `scripts/triple_eval.py` | Per-triple diagnostics: PE, confidence, rank, source |
| External benchmark | `scripts/external_benchmark.py` | PCX tasks, lifelong retention, graph latency |

Key results:

| Metric | Result |
|--------|--------|
| Cross-domain transfer Top-1 | 75.0% |
| Cross-domain transfer Top-10 | 100% |
| Held-out Science Top-1 | 8.3% (n=12) |
| Within-domain triple Top-10 | 80.9% |
| Lifelong forgetting (permuted MNIST) | 0% (with sleep) |
| Graph Inference P95 | 2.7 ms |

---

## Where RLMv2 is used in the codebase

| Script | How it uses RLMv2 |
|--------|-------------------|
| `scripts/benchmark_vs_transformers.py` | Held-out generalization, cross-domain transfer benchmarks |
| `scripts/external_benchmark.py` | Cross-domain profiling, large-graph latency |
| `scripts/validate_held_out_generalization.py` | Verb-offset adaptation checks |
| `scripts/diagnose_transfer.py` | Science→Social transfer analysis |
| `scripts/triple_eval.py` | Per-triple diagnostics |
| `experiments/experiment_cross_domain.py` | Cross-domain experiment harness |

---

## What the code does step by step

```python
# 1. Configure model dimensions
vocab_size = 500     # unique words in vocabulary
embed_dim = 64       # embedding dimension (matches graph dimension)
concept_dim = 64     # concept vector dimension

# 2. Instantiate RLMv2
model = RLMv2(
    vocab_size=vocab_size + 10,  # +10 for <unk>, <pad>, <sos>, <eos>, etc.
    embed_dim=embed_dim,
    concept_dim=concept_dim,
    n_concepts=200,
)
```

Constructor initializes:
- Embedding matrix: `vocab_size × embed_dim` (random, later replaced by GloVe)
- Concept graph: 200 dummy concepts
- Relation predictor: W_rel matrix for each relation type
- Verb-offset store: empty dict, populated during training
- Sleep counters: ready for consolidation

```python
# 3. Train on a triple (in normal usage)
model.learn(input_tokens, target_tokens)
```

`learn()` does:
1. Encode input tokens to embeddings
2. Predict target tokens via graph walk
3. Compute prediction error
4. Update edge weights via Hebbian plasticity
5. Accumulate error for sleep cycle

```python
# 4. Consolidate
model.sleep_cycle()
```

Runs the 2-stage sleep (SWS → REM) described above.

---

## Expected output (annotated)

```
=== RLMv2 - Relation Learner Model ===

  [Ontology] Injected 118 seed edges at confidence=0.25
  └── RLMv2 auto-seeds from the relation ontology on init

  Model type:       RLMv2
  Embed dimension:  64
  └── Word embeddings are 64-D (same as concept graph)

  Concept dimension:64
  Vocab size:       500

  RLMv2 decomposes sentences into (subject, relation, object) triples:
    "heat causes expansion"  ->  (heat, causes, expansion)

  This enables analogy via vector arithmetic:
    subject_embed + offset(verb) ~= target_embed
    heat + offset("causes") ~= expansion

  Used by:
    - scripts/benchmark_vs_transformers.py  - held-out benchmarks
    - scripts/validate_held_out_generalization.py  - verb-offset checks
    - scripts/external_benchmark.py  - cross-domain profiling
    - scripts/diagnose_transfer.py  - cross-domain transfer analysis
```

---

## Key source files reference

| Component | File (relative to repo root) |
|-----------|------------------------------|
| RLMv2 class | `ravana_ml/src/ravana_ml/nn/rlm_v2.py` (line 214) |
| Encoder mixin | `ravana_ml/src/ravana_ml/nn/rlm_v2_encoder.py` |
| Graph mixin | `ravana_ml/src/ravana_ml/nn/rlm_v2_graph.py` |
| Relation predictor | `ravana_ml/src/ravana_ml/nn/rlm_v2_rp.py` |
| Verb mixin | `ravana_ml/src/ravana_ml/nn/rlm_v2_verb.py` |
| Sleep mixin | `ravana_ml/src/ravana_ml/nn/rlm_v2_sleep.py` |
| Shared utilities | `ravana_ml/src/ravana_ml/nn/rlm_v2_common.py` |
| RLMv1 (original) | `ravana_ml/src/ravana_ml/nn/rlm.py` |

---

## Design philosophy notes

1. **Triples are the native format.** Everything in RAVANA is a (subject,
   relation, object) triple — the graph, the decoder, and RLMv2. This
   consistency is deliberate: there's only one data structure to understand.
2. **Verb offsets replace relation embeddings.** Instead of learning a single
   embedding for "causes," RLMv2 learns a transformation FROM the subject TO
   the target. This enables analogical transfer across domains.
3. **Sleep is not optional.** Without sleep consolidation, the model suffers
   from catastrophic forgetting. With sleep, it retains 100% across domain
   switches (verified by external benchmarks).
4. **Spreading activation is the inference mechanism.** No attention, no
   transformer blocks, no matrix multiplication. Just typed edges and
   propagation rules — the same mechanism used by the brain.

---

## Going further

To see RLMv2 in action with real training and benchmarks:

```bash
# Quick held-out generalization check
python scripts/validate_held_out_generalization.py

# Cross-domain transfer diagnostic
python scripts/diagnose_transfer.py

# Full benchmark suite
python scripts/benchmark_vs_transformers.py
python scripts/external_benchmark.py

# Per-triple evaluation
python scripts/triple_eval.py
```

---

## End of tutorial series

You've completed all 7 tutorials. You now know:
1. How the chat engine works end-to-end
2. How the neural decoder trains and generates text
3. How the concept graph stores and retrieves knowledge
4. How web learning works (curiosity-driven, background thread)
5. How to measure the system with experiments
6. How the GRACE cognitive modules regulate behavior
7. How RLMv2 learns and transfers relational knowledge

For next steps, see `docs/GETTING_STARTED.md` or explore the `experiments/`
directory.
