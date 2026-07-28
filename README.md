# RAVANA

> A **decoder-first ML cognitive architecture** that starts like a baby and
> learns continuously from conversation and the open web — **no LLM, no
> pretrained chat model.**

RAVANA stores knowledge as a **typed concept graph**, produces language with a
small **neural decoder** conditioned on graph walks, and orchestrates cognition
with a **20-phase GRACE governor** (phases A–P: identity, emotion, sleep, meaning,
theory-of-mind, metacognition, …). When it cannot answer honestly, it
**abstains** — confident-wrong is treated as high free-energy that would poison
the graph.

```
user text ──▶ brain-repair prepasses ──▶ intent/coherence/gate ──▶ graph walk
                                                              │
                                                              ▼
                                   neural decoder ──▶ surface realizer ──▶ response
                                                              │
                                          web-learning (gaps ──▶ new typed edges)
```

## Why

Most chat systems are thin wrappers over a giant pretrained model. RAVANA is an
experiment in the opposite direction: a small, inspectable system that
**learns in the loop**, grounds every claim in its graph, and is honest about
what it does not know.

## Install

Requires Python 3.10+.

```bash
pip install -e .[full,dev]      # editable install (also what CI runs)
# or, plain deps:
pip install -r requirements.txt
```

The core (`ravana` + `ravana_ml`) needs only `numpy` and `scipy`. The optional
extras (torch, web scraping, embeddings, plotting) are pulled in by `full`.

## Run

```bash
# Chat (interactive). The engine auto-learns from the web when it hits a gap.
python scripts/ravana_chat.py

# Train / promote the decoder
python scripts/train.py --mode phase2      # heavy seed + web + consolidate
python scripts/train.py --mode full        # same single-phase pipeline
python scripts/train.py --mode test        # quick diagnostic
python scripts/train.py --mode linggen     # LingGen sensorimotor promotion

# Autonomous background learning (no chat) — Ctrl+C to save
python scripts/ravana_learn.py
```

The first run needs `data/corpora/teen_seeds.txt` (gitignored). If absent,
rebuild it with `python scripts/gather_teen_seeds.py`.

## Repository map

| Path | What |
|------|------|
| `ravana/src/ravana/` | Chat engine: `CognitiveChatEngine` (composes 8 mixins in `chat/engine_*.py`), brain-repair prepasses, language generation, web learning, safety/consistency/abstention monitors. |
| `ravana_ml/src/ravana_ml/` | CPU-native ML substrate: tensors, `ConceptGraph`, `RLM`/`RLMv2`, neural decoder, embedders. |
| `ravana-v2/src/ravana_grace/` | GRACE 20-phase cognitive governor (A–P). |
| `scripts/` | Runnable entry points (chat, train, learn, benchmarks). |
| `experiments/` | Research harnesses used by the benchmarks. |
| `tests/` | pytest suite (`ci` / `unit` / `integration` / `eval`). |
| `docs/` | Architecture, modules, training, benchmarks, development guide. |

The three `src` trees are one integrated system; `scripts/ravana_chat.py`
imports from all of them.

## Tests

```bash
python -m pytest tests/ci/ -v --ci     # fast critical-path job (~15 min, soak tests)
python -m pytest tests/unit/ -q        # module-level
python -m pytest tests/integration/ -q # cross-module
python -m pytest tests/ --tb=short     # full suite
```

CI (`.github/workflows/ci.yml`) runs `pip install -e .[full,dev]` then the
`ci` / `unit` / `integration` jobs on Python 3.10.

## Benchmark results

Results are **current as of the latest commits**. Historical values from prior
experiments are archived separately.

### Cross-Domain transfer

| Metric | Result |
|--------|--------|
| Cross-domain transfer Top-1 | 100% (n=6) |
| Held-out Science Top-1 (post-sleep) | 93.8% (n=16) |
| Held-out Social Top-1 (post-sleep) | 80.0% (n=20) |
| Held-out vs baseline (Science) | 12.5% → 93.8% |
| Held-out vs baseline (Social) | 5.0% → 80.0% |
| Transfer probes Top-1/Top-10 | 59.5% / 73.8% |
| Sleep cycle conceptual accuracy | 90.2% |

### Graph scaling

| Graph size | `find_similar` p50 | `find_similar` p95 |
|-----------|-------------------|-------------------|
| 1K nodes | 0.021 ms | 0.025 ms |
| 5K nodes | 0.043 ms | 0.051 ms |
| 10K nodes | 0.071 ms | 0.191 ms |

### ARC Grounding Monitor

| Metric | Pre-ARC | Post-ARC |
|--------|---------|----------|
| Composite quality | 0.395 | 0.394 |
| HONEST-abstinence | 0.600 | 0.700 |
| Confabulation rate | 0.000 | 0.000 |
| Salad rate | 0.000 | 0.000 |
| **Verdict** | | ARC MAINTAINS QUALITY, IMPROVES HONESTY |

> *Results from a one-off measurement; see `docs/BENCHMARKS.md` for how to reproduce.*

### RLMv2 External Benchmarks (GRACE core)

| Metric | Result |
|--------|--------|
| Cross-domain transfer Top-1 | 75.0% |
| Cross-domain transfer Top-10 | 100% |
| Held-out Science Top-1 / Top-10 | 8.3% / 25.0% (n=12) |
| Within-domain triple top-10 | 80.9% |
| Lifelong forgetting (permuted MNIST) | 0% (with sleep) |
| Graph Inference P95 / P99 | 2.7 ms / 2.9 ms |
| Graph Peak Memory / Throughput | 0.3 MB / 556 QPS |
| W_rel Causal / Semantic Alignment | 0.68 / 0.55 |

> See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for how to reproduce each result.
> See [`experiment_results/`](experiment_results/) for raw JSON output files.

## Documentation

See [`docs/`](docs/README.md):

- [Architecture](docs/ARCHITECTURE.md) — turn-level data flow and the three packages.
- [Modules](docs/MODULES.md) — what lives in each package.
- [Training](docs/TRAINING.md) — `train.py` modes and the LingGen promotion gate.
- [Benchmarks](docs/BENCHMARKS.md) — every benchmark/diagnostic script and what it measures.
- [Development](docs/DEVELOPMENT.md) — layout, path shims, test commands, conventions.

## Benchmark results

RAVANA is evaluated end-to-end with `scripts/evaluate_ravana.py`, which trains a
fresh `dim=64` engine on TinyShakespeare (25 passes, no live web) and runs **all
nine** benchmark batteries — each in an isolated engine so no benchmark leaks
facts into another — then compares the trained model against a same-scale nanoGPT
on parameter efficiency. Full per-case output is written to
`data/eval_results.json`.

Latest live run (current `main`, `dim=64`, Shakespeare, 25 passes):

| Benchmark | Score |
|---|---|
| Lamp test (perceptual grounding) | 1.00 |
| Self-evaluation (metacognitive honesty) | 0.82 |
| Consult (advice / open Q&A) | 0.57 |
| Reasoning (LogiQA logical MCQ) | 0.37 |
| Temporal (TimeDial cloze) | 0.55 |
| LoCoMo (long-term episodic memory) | 0.34 |
| LongMemEval (cross-session memory) | 0.34 |
| Adversarial (AdvBench refusal) | 0.40 |
| Memory consistency (MemFail) | 0.70 |
| **Overall average** | **0.57** |

> Reasoning went 0.00 → 0.37 after a harness fix (LogiQA loader emits the
> `Options:` prefix) plus the Phase-3 HPC→PFC graph reasoner (structured
> premise mining + unit propagation + fail-closed entailment test) and an
> MC answer-frame discipline that stops a yes/no rule echo from answering a
> letter question and adds a forced-choice fluency fallback under forced
> choice. Consult went 0.10 → 0.57 from the Phase-1 ATL semantic graph
> (ConceptNet seed + goal-directed means-end advice) and LoCoMo 0.20 → 0.34
> from the Phase-2 encoding-specificity date binding + scoped temporal
> recall. Adversarial dropped 0.52 → 0.40 by design: the model now answers
> harmful "how to X" requests with helpful means-end advice instead of a
> hardcoded refusal (freedom over guardrails).

**RAVANA vs nanoGPT (comprehensive harness)** — same data, same `dim=64`
decoder, measured on parameter efficiency:

| Metric | nanoGPT | RAVANA |
|---|---|---|
| Parameters | 10,700,000 | 5,070,789 |
| % of nanoGPT params | 100% | **47.4%** |
| Params per data character | 9.60 | **4.55** (2.1× fewer) |

RAVANA reaches ~half of nanoGPT's parameter count while training the *same*
character-level decoder on the same corpus — the remaining parameters are the
brain-inspired cognitive substrate (concept graph, hippocampal buffer, belief
store, salience/decay) that the bare transformer lacks. The benchmark batteries
target those cognitive capacities (memory, temporal reasoning, refusal,
self-evaluation) rather than raw next-char perplexity, which is why a smaller
decoder is paired with a broader evaluation.

> Note: scores are **not** comparable to historical `data/eval_results.json`
> snapshots taken on the toy `train.py --mode test` corpus (vocab ≈ 96, ~50
> sentences). Those were a different training regime; this table is the current
> Shakespeare-trained configuration.

See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the per-benchmark methodology
and how to reproduce.

## Design principles

1. **Fail-closed grounding** — abstain rather than confabulate.
2. **No fixed thresholds where a distribution exists** — gates are data-derived.
3. **Continuous, curiosity-driven learning** — what to learn is selected from
   prediction error, novelty, and contradiction, not a fixed list.
4. **Learning without backprop** — `ravana_ml` minimizes free energy and
   consolidates during `sleep_cycle()`.

## License

Oxiverse Community License (OCL) v1.0 — see [LICENSE](LICENSE).
