# Tutorial 04: Continuous Background Learning

**Part of a 7-tutorial progression. Layer 1 of 3 added on top of the mini-system.**

## What you'll learn

- Start background web learning — the system **teaches itself**
- Push topics to the learning queue and watch the graph grow
- Understand the **curiosity drive** — how the system chooses what to learn
- Trace the **web → graph pipeline**: search → extract → type → consolidate

---

## Run it

```bash
python tutorials/04-continuous-learning/run.py --cycles 5 --delay 2
```

---

## Deep dive: the curiosity-driven learning pipeline

The learning system has 4 stages:

```
Knowledge Gap      →  Topic Selection  →  Web Search      →  Graph Integration
(what don't I know)    (curiosity score)    (fetch + parse)     (edges + sleep)
```

### Stage 1: Detecting knowledge gaps

**File:** `ravana/src/ravana/learn/curiosity.py` → `CuriosityEngine`

The engine doesn't learn randomly. It selects topics based on a **curiosity score**
that combines 4 signals:

| Signal | Source | Formula | What it detects |
|--------|--------|---------|-----------------|
| **Prediction error** | Edge's `prediction_free_energy` | PE / max(PE) | Concepts where the graph makes wrong predictions |
| **Novelty** | Concept visit count | 1 / (1 + visits) | Underexplored concepts |
| **Contradiction** | Contradiction map size | contradictions / total | Conflicting beliefs |
| **Serendipity** | Unexpected co-activations | random boost | Random exploration |

The final score is a weighted combination:

```python
curiosity_score = w_pred * prediction_error
                + w_novel * novelty
                + w_contra * contradiction
                + w_serendipity * serendipity
```

The concept with the highest score is selected for web research. This is the
computational analog of the **epistemic curiosity** drive in humans (Loewenstein
1994; Berlyne 1966) — the desire to close information gaps.

### Stage 2: Topic to query

The selected concept is turned into a web search query. The engine also
expands the query with related terms:

- "trust" → "trust psychology definition" (expanded with domain context)
- "neural plasticity" → "neural plasticity brain learning" (if already in graph)

### Stage 3: Web search & extraction

**Files:** `ravana/src/ravana/web/learner.py` → `SearchEngine`

The search pipeline:

```
Topic → SearchEngine.fetch() → SearchEngine.extract() → WebToGraph.convert()
```

1. **DuckDuckGo search** (HTML scraping, no API key needed)
2. **HTML parsing** via BeautifulSoup or trafilatura
3. **OpenIE extraction** (`OpenIEExtractor`) — extract (subject, relation, object) triples
   from the fetched text
4. **Relevance scoring** — rank extracted triples by relevance to the topic
5. **Quality gating** — reject boilerplate, cookie notices, navigation text

The `SearchEngine` has a **circuit breaker** pattern: if a search engine fails
3 times in a row, it's disabled for 60 seconds. This prevents a single failed
search from blocking the entire learning thread.

### Stage 4: Graph integration

**File:** `ravana/src/ravana/web/web_to_graph.py` → `WebToGraph`

Each extracted triple becomes graph edges:

```
(heat, causes, expansion)
    → add_node("heat") if not exists
    → add_node("expansion") if not exists
    → add_edge(heat_id, expansion_id, relation_type="causal", weight=0.5)
```

The system also merges synonyms: if "heat" and "thermal energy" are GloVe-similar
(cosine > 0.7), they're treated as related concepts and edges are created
between them.

### Background thread architecture

The learning runs in a **background thread** so it never blocks the user:

```
Main thread:           process_turn() → immediate response
                       ↓
Background thread:     search → extract → graph → consolidate
                       (started by start_background_learning())
```

Thread safety is handled by three locks:
- `_vocab_lock` — reentrant lock for vocabulary access
- `_graph_lock` — reentrant lock for graph modification
- `_bg_lock` — standard lock for search count updates

The background thread also respects an **idle event**: when the user sends a
message, `_bg_idle_event` is set, signaling the background thread to pause
and prioritize the user's turn.

---

## Deep dive: the curiosity drive variants

### With curiosity enabled (default)

The engine autonomously selects what to learn from its own graph state.
Topics are scored by the `CuriosityEngine` and the highest-scoring concept
is searched next. This means the system chooses what interests it.

### With curiosity disabled (`--no-curiosity`)

The engine only learns topics you explicitly push to the queue. This is
deterministic — you control exactly what's researched.

### The learning queue

You can push topics to the queue at any time. The background thread pops
topics and processes them one by one. When the queue is empty and curiosity
is enabled, the engine auto-selects topics.

---

## What the code does step by step

```python
# 1. Start the background thread
engine.start_background_learning()
```

This launches a daemon thread that loops:
1. Pop a topic from the queue (or select via curiosity)
2. Search the web for the topic
3. Extract facts from search results
4. Add facts to the graph
5. Sleep for `--delay` seconds
6. Repeat

```python
# 2. Push topics
engine._bg_learning_queue.append("consciousness neuroscience")
```

Topics are strings that become search queries. The queue is a `List[str]`
processed in FIFO order.

```python
# 3. Poll graph growth
nodes = len(engine.graph.nodes)
edges = len(engine.graph.edges)
searches = engine._bg_search_count
```

`_bg_search_count` increments with each successful web search. You can track
learning activity even while the main thread is idle.

```python
# 4. Stop and save
engine.stop_background_learning()
engine.save()
```

`stop_background_learning()` sets a flag that causes the thread to exit
on its next iteration. It does NOT kill the thread — it's a cooperative stop.
`save()` then serializes the updated graph to disk.

---

## Expected output (annotated)

```
=== Continuous Background Learning Demo ===

  Queued 3 topics: consciousness neuroscience, human memory psychology, sleep dreaming
  └── These topics are pushed to the learning queue

  cycle=1/2  nodes=635 (+0)  edges=15219 (+0)  searches=337
  └── In this session, the graph didn't grow (searches were already completed)
  └── On a fresh run, you'd see nodes/edges increase each cycle

  [OK] State saved. Graph grew by 1 nodes and 125 edges.
  └── Even with no new nodes, sleep consolidated 125 edges
```

Why did the graph not grow in this run? The engine loaded from a saved state
that already had 337 searches completed. The searches completed quickly because
they were working with cached/similar topics. On a fresh engine (no save file),
each cycle would add nodes and edges as new concepts are discovered.

---

## Options reference

| Flag | Default | Effect |
|------|---------|--------|
| `--cycles` | 3 | Number of learning cycles before exit |
| `--delay` | 1.0 | Seconds between cycles (avoid rate-limiting) |
| `--no-curiosity` | off | Disable autonomous topic selection; only learn queued topics |

With `--cycles 0`, the system runs indefinitely until Ctrl+C.

---

## Design philosophy notes

1. **Learning is autonomous.** The system doesn't need a curated training set.
   It identifies gaps in its own knowledge and searches the web to fill them.
2. **Curiosity is a drive, not a schedule.** Learning isn't triggered by a timer
   — it's triggered by the internal state of the graph (prediction error, novelty,
   contradiction). This is the free-energy principle in action.
3. **The web is the corpus.** RAVANA doesn't download pre-built datasets.
   It learns from live web content, which means it's always up to date and
   can learn about any topic the user asks about.
4. **Thread safety is critical.** The graph can be modified by both the main
   thread (during process_turn) and the background thread (during learning).
   Three locks prevent race conditions.

---

## Key source files reference

| Component | File (relative to repo root) |
|-----------|------------------------------|
| Curiosity engine | `ravana/src/ravana/learn/curiosity.py` |
| Search engine | `ravana/src/ravana/web/learner.py` (line 134, `SearchEngine`) |
| Web extractor | `ravana/src/ravana/web/web_to_graph.py` (line 48, `WebToGraph`) |
| OpenIE extractor | `ravana/src/ravana/web/openie.py` (line 139, `OpenIEExtractor`) |
| Background thread | `ravana/src/ravana/chat/engine.py` (engine init, `_bg_*` fields) |
| Web learning mixin | `ravana/src/ravana/chat/web_learning.py` |

---

## Next tutorial

[**Tutorial 05: Experiment Harness**](../05-experiments/) — learn how to measure
the system's performance with a structured experiment harness.
