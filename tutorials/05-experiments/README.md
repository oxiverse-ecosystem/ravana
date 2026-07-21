# Tutorial 05: Experiment-Style Harness

**Part of a 7-tutorial progression. Layer 2 of 3: measurement.**

## What you'll learn

- Run a structured probe loop — the foundation of all RAVANA experiments
- Measure graph growth under controlled, repeatable conditions
- Write results to JSON for analysis
- Understand the experiment methodology used in `experiments/`

---

## Run it

```bash
python tutorials/05-experiments/run.py
```

---

## Deep dive: experiment methodology

### Why experiment design matters

RAVANA is a **cognitive architecture** — it has many interacting components
(graph, decoder, GRACE modules, web learning). To understand which components
are working and which aren't, you need **controlled experiments** with:

- **Independent variables**: what you change (query, configuration flags)
- **Dependent variables**: what you measure (graph size, response quality, latency)
- **Controls**: what you keep constant (seed, dimension, initial state)
- **Replication**: running the same probe multiple times to check consistency

### The probe loop pattern

Every experiment in RAVANA follows the same pattern:

```python
# 1. Create controlled initial state
engine = CognitiveChatEngine(dim=64, seed=42, ...)

# 2. Define probes (the independent variables)
probes = ["what is trust", "what is betrayal", ...]

# 3. Run each probe and record measurements
results = []
for probe in probes:
    engine.process_turn(probe)        # intervention
    results.append({
        "query": probe,
        "nodes": len(engine.graph.nodes),     # measurement
        "edges": len(engine.graph.edges),     # measurement
        "sleep": engine.sleep_cycles_completed,  # measurement
    })

# 4. Persist results
json.dump(results, "experiment_results/my_experiment.json")
```

This pattern is used by all 64 experiment files in `experiments/`.

### Probe design principles

Good probes are:

1. **Discriminating** — they should produce different outcomes for different
   system configurations (e.g., with and without VAD emotion)
2. **Repeatable** — same probe on same config should give similar results
3. **Interpretable** — you should be able to explain WHY the system gave a
   particular response (graph walk → decoder → realizer)
4. **Coverage-based** — probes should cover all 6 intent classes (chitchat,
   factual, hypothetical, identity, comparative, OOD)

### What the probes measure

| Metric | What it reveals | Typical range |
|--------|----------------|---------------|
| `nodes` | Knowledge breadth | 180 (fresh) → 1000+ (learned) |
| `edges` | Knowledge density | 130 (fresh) → 15000+ (learned) |
| `sleep` | Consolidation activity | 0 (fresh) → 50+ (after many turns) |
| `response` | Output quality | Subjective — evaluated by benchmarks |

### The seed parameter

The `seed=42` parameter controls the random number generator. Using the same
seed across runs ensures **reproducibility** — the same probes should produce
the same graph structure. This is critical for comparing experiments.

### The baby_mode flag

`baby_mode=True` means the engine starts fresh — it doesn't try to load
saved weights. This ensures every run starts from the same initial state.
Without it, the engine would load the accumulated knowledge from previous
runs, making cross-run comparisons meaningless.

---

## Understanding the results

### Example results

```json
[
  { "query": "what is trust",     "nodes": 635, "edges": 15405, "sleep": 44 },
  { "query": "what is betrayal",  "nodes": 635, "edges": 15406, "sleep": 44 },
  { "query": "what is loyalty",   "nodes": 635, "edges": 15406, "sleep": 44 },
  { "query": "what is memory",    "nodes": 635, "edges": 15406, "sleep": 44 }
]
```

### Interpreting the measurements

1. **Nodes stayed at 635** — all 4 probes used existing concepts. No new
   concepts were discovered because all 4 words (trust, betrayal, loyalty,
   memory) were already in the graph from the seed corpus.

2. **Edges increased by 1** (15405 → 15406) — query 2 ("betrayal") created
   a new edge from the existing graph walk. The other queries didn't add edges
   because their graph walks traversed existing paths.

3. **Sleep stayed at 44** — none of the queries triggered enough prediction
   error to cross the sleep threshold (0.3). The engine is stable on these
   well-known concepts.

### What would indicate problems

- **Nodes decreasing** → graph pruning is too aggressive (check sleep config)
- **Edges growing too fast** → web learning is adding junk edges (check quality gate)
- **Sleep increasing rapidly** → too much prediction error (check decoder accuracy)
- **No change at all** → graph is saturated (consider resetting or expanding vocabulary)

### Relation to real experiments

The real experiment suite (`experiments/experiment_cross_domain.py`) uses the
same pattern but with:
- **Two domains** (Science, Social) to measure cross-domain transfer
- **Held-out probes** to measure generalization
- **Baseline comparisons** (random vs trained)
- **Sleep intervention** (measure before/after consolidation)

---

## What the code does step by step

```python
# 1. Create a clean engine (fresh state for reproducible measurements)
engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
```

`baby_mode=True` ensures no saved weights are loaded. Every run starts from
the same initial state (GloVe + PMI-seeded graph).

```python
# 2. Define structured probes (the independent variables)
probes = ["what is trust", "what is betrayal", "what is loyalty", "what is memory"]
```

These 4 probes cover the "factual" intent class with related concepts (social
virtues). Real experiments would include probe from ALL intent classes.

```python
# 3. Run each probe and record graph statistics (the dependent variables)
for q in probes:
    engine.process_turn(q)
    results.append({
        "query": q,
        "nodes": len(engine.graph.nodes),
        "edges": len(engine.graph.edges),
        "sleep": engine.sleep_cycles_completed,
    })
```

Each `process_turn()` call runs the full 6-stage pipeline (brain-repair →
intent routing → graph walk → decoder → realizer → web learning). The
measurements are taken AFTER the turn completes, so they reflect any graph
changes the turn caused.

```python
# 4. Write results to JSON
path = os.path.join(ROOT, "experiment_results", "tutorial_persistence.json")
with open(path, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)
```

Results go to `experiment_results/` — the same directory used by the real
experiment suite. This makes it easy to compare tutorial results with
published benchmarks.

---

## Expected output (annotated)

```
=== Experiment-Style Harness Demo ===

  Probes: 4
    - what is trust
    - what is betrayal
    - what is loyalty
    - what is memory
  └── 4 controlled probes, all from the "factual" intent class

  [OK] what is trust                             nodes= 635  edges=15405  sleep=44
  [OK] what is betrayal                          nodes= 635  edges=15406  sleep=44
  └── "betrayal" created 1 new edge via graph walk

  [OK] what is loyalty                           nodes= 635  edges=15406  sleep=44
  [OK] what is memory                            nodes= 635  edges=15406  sleep=44
  └── "loyalty" and "memory" used existing paths

  [OK] Results written to experiment_results/tutorial_persistence.json

  Summary:
    Total nodes: 635
    Total edges: 15406
    Sleep cycles: 44
```

---

## Key source files reference

| Component | File (relative to repo root) |
|-----------|------------------------------|
| Experiment framework | `experiments/` (64 experiment files) |
| Cross-domain experiment | `experiments/experiment_cross_domain.py` |
| ARC benchmark | `experiments/benchmark_arc.py` |
| Ablation study | `experiments/experiment_ablation.py` |
| Sleep/memory experiment | `experiments/experiment_sleep_memory.py` |
| Result analysis | `scripts/plot_analysis.py` |

---

## Design philosophy notes

1. **Reproducibility is mandatory.** Every experiment uses a fixed seed,
   `baby_mode=True`, and structured probes. This ensures results are comparable
   across runs and configurations.
2. **Measure what matters.** Graph statistics (nodes, edges, sleep) are used
   because they directly reflect the system's learning state. Response quality
   is evaluated separately by dedicated benchmarks.
3. **JSON as the interchange format.** All experiments output JSON, which can
   be analyzed with any tool (Python, R, Excel). No proprietary format.
4. **Same pattern, different scales.** A 4-probe tutorial and a 200-probe
   experiment use the same code structure. Scale doesn't change methodology.

---

## Next tutorial

[**Tutorial 06: GRACE Governor Modules**](../06-governor/) — explore the
cognitive regulation modules that control emotion, identity, and reasoning.
