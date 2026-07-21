# Frequently Asked Questions

## Installation & Setup

### "No module named 'ravana_ml'" or "No module named 'ravana_grace'"

Make sure you're running `pip install -e .` from the **repo root** (the directory
containing `pyproject.toml`). The three packages live under `*/src/` and the
build system finds them automatically:

```
pip install -e .[full,dev]
```

If you're running scripts directly (not through installed packages), the scripts
insert `sys.path` entries automatically. But if you're importing from Python
directly, you need the editable install.

### "ModuleNotFoundError: No module named 'ravana.chat'"

The `ravana` package is installed from `ravana/src/ravana/`. If you see errors
about missing sub-packages, check:
1. You installed from the repo root (`pip install -e .`)
2. You're running Python from the repo root (for development path shims)

### "Can't find data/corpora/teen_seeds.txt"

The seed corpus is gitignored. Generate it:

```bash
python scripts/gather_teen_seeds.py
```

This creates `data/corpora/teen_seeds.txt` with ~296 teen-level English sentences.

### "pip install -e . fails with setuptools error"

Ensure you have a recent `setuptools` (>=68):

```bash
pip install --upgrade setuptools wheel
pip install -e .[full,dev]
```

---

## Runtime

### First run is very slow

This is expected. The first boot:
1. Downloads GloVe embeddings (822 MB) if not cached
2. Builds GloVe→64-D projection
3. Trains the attribute encoder
4. Seeds the concept graph from PMI statistics

Subsequent runs load from `data/ravana_weights.pkl` and are much faster.

### "The chatbot says 'I don't know' to everything"

That means the engine is working correctly — it's **abstaining** honestly rather
than confabulating. Try:
- Running `python scripts/train.py --mode test` first to seed with some knowledge
- Asking more concrete questions like "what is trust" or "tell me about love"
- Checking if web learning is working (the engine searches the web when it has gaps)

### Web search seems slow or times out

The engine has a 4-second socket timeout and a circuit breaker for failing search
engines. This is intentional — it prevents a single bad search from blocking the
user turn. If searches consistently fail:
- Check your internet connection
- The search engine uses DuckDuckGo HTML scraping (no API key needed)
- Set `--delay` higher if running background learning

### "Windows access violation error" on startup

This is a known NumPy/OpenBLAS threading issue (numpy #27989). The engine
auto-pins threads to 1, but some BLAS builds still encounter this. Try:

```bash
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
python scripts/ravana_chat.py
```

### Memory usage seems high for a "small" system

The concept graph grows as the system learns. After a long session with web
learning, it can reach thousands of nodes and edges. Run `--stats` to see the
current size. Sleep consolidation prunes low-utility edges.

---

## Development

### How do I add a new benchmark?

1. Create a new file in `experiments/` (e.g. `experiment_my_benchmark.py`)
2. Import from `ravana.chat.engine`, `ravana_ml`, or `ravana_grace` as needed
3. Add a corresponding runner in `scripts/` (e.g. `scripts/my_benchmark.py`)
4. Document it in [BENCHMARKS.md](BENCHMARKS.md)

### How do I extend the concept graph?

- Add new concepts: `graph.add_node(vector=vec, label="my_concept")`
- Add edges: `graph.add_edge(src_id, tgt_id, weight=0.7, relation_type="causal")`
- The engine auto-expands from user input via `_auto_expand_concepts()`

### How do I add a new brain-repair prepass?

Brain-repair prepasses live in `ravana/src/ravana/chat/brain_regions.py`. They run
BEFORE `process_turn` routing. Follow the existing pattern:
1. Add your function to `brain_regions.py`
2. Call it from `CognitiveChatEngine.process_turn()` before the intent router
3. Make it fail-closed (return None/false when unsure)
4. Add a test in `tests/unit/`

### How do I change the decoder?

The decoder (`ravana_ml.nn.neural_decoder.NeuralDecoder`) is a small GRU. To
change its architecture:
1. Modify `ravana_ml/src/ravana_ml/nn/neural_decoder.py`
2. The decoder uses `ravana_ml.nn.module` (not PyTorch) — the module is
   self-contained
3. Run `python scripts/train.py --mode test` to verify

### Can I use PyTorch instead of the native tensor framework?

The native tensor framework (`ravana_ml.tensor`) is a lightweight numpy wrapper
designed to make the decoder API familiar (`.forward()`, `.parameters()`, etc.).
You can replace it with PyTorch if needed, but you'll lose the sleep-consolidation
loop that replaces `optimizer.step()`.

---

## Concepts

### Why no backpropagation?

Backpropagation requires storing all intermediate activations for the chain rule.
This is memory-intensive and GPU-dependent. RAVANA replaces it with:
- **Pressure accumulation** (localized prediction error)
- **Hebbian updates** (co-activation → strength change)
- **Sleep consolidation** (global reorganization)

### Why a concept graph instead of a weight matrix?

The graph is **sparse** — only active edges are stored. A weight matrix is **dense**
— every parameter is stored and updated. The graph also enables:
- Typed relations (causal, temporal, semantic, etc.)
- Discrete reasoning (chain walking, not matrix multiply)
- Inspectable knowledge (you can read every edge)

### What is "sleep" in this context?

Sleep is a periodic consolidation phase that:
1. Analyzes prediction errors across the graph
2. Strengthens low-error edges, prunes noisy ones
3. Resolves contradictory beliefs
4. Integrates episodic patterns into semantic structure
5. Runs counterfactual recombination (REM analog)

It replaces the gradient descent step in traditional ML.

### What does "fail-closed" mean?

The system would rather say "I don't know" than emit confident nonsense. Every
gate (coherence, junk, salad, etc.) is designed so that if it malfunctions or
lacks data, it defaults to **rejection** rather than **admission**. This is
enforced throughout the codebase — see the design principles in `docs/README.md`.
