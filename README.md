# RAVANA

> A **decoder-first ML cognitive architecture** that starts like a baby and
> learns continuously from conversation and the open web — **no LLM, no
> pretrained chat model.**

RAVANA stores knowledge as a **typed concept graph**, produces language with a
small **neural decoder** conditioned on graph walks, and orchestrates cognition
with a **27-phase GRACE governor** (identity, emotion, sleep, meaning,
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
| `ravana/src/ravana/` | Chat engine: `CognitiveChatEngine`, brain-repair prepasses, language generation, web learning. |
| `ravana_ml/src/ravana_ml/` | CPU-native ML substrate: tensors, `ConceptGraph`, `RLM`/`RLMv2`, neural decoder, embedders. |
| `ravana-v2/src/ravana_grace/` | GRACE 27-phase cognitive governor. |
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

Results are split into **verified this session** and **historical results**
from prior experiment outputs.

| Status | Notes |
|--------|-------|
| Verified | Reproduced by `scripts/scaling_benchmark.py` or by live `experiments/experiment_cross_domain.py` runs in this session. |
| Historical | Prior `experiment_results/*` values retained for reference, but not reproduced from checked-in outputs in this session. |

### Cross-Domain transfer

- **Verified**: cross-domain synthesis and held-out post-adapt results were reproduced from `experiments/experiment_cross_domain.py` and saved to `experiment_results/cross_domain_transfer.json`.
- **Cross-domain transfer Top-1/Top-10**: `100% (6/6)` — verified
- **Held-out Science Top-1 adapted**: `93.8% (n=16)` — verified
- **Held-out Social Top-1 adapted**: `85.0% (n=20)` — verified
- **Held-out baseline**: `12.5% (Science)` / `5.0% (Social)` — historical comparison values
- **W_rel Causal Alignment**: `0.38` — historical comparison value
- **Lifelong forgetting**: `0.167 (0.667→0.500)` — historical comparison value
- **Sleep-time interleaved replay retention**: `0% drop` — historical comparison value

### Graph scaling

Live scaling benchmark found only the 1K/5K/10K runs useful for this run; the
README’s 50K/15.8x claims are not reproduced here.

- `1K` nodes: `find_similar p50=0.021ms / p95=0.025ms`
- `5K` nodes: `find_similar p50=0.043ms / p95=0.051ms`
- `10K` nodes: `find_similar p50=0.071ms / p95=0.191ms`

### False positives and memory

Live runtime from this session shows cross-domain intrusions are imperfect
relative to the older “0.1 / fact” table.

- **Live false-positive probe**: avg `6.17` cross-domain intrusions in top-10
  (script default 3-domain probe in `scripts/scaling_benchmark.py`)
- **Earlier memory metrics**: historical tables were not found in checked-in
  output files for this session, including `Recall@1 at 10 facts = 1.000`,
  `Recall@1 at 60 facts = 0.917`, and `1.000` rare-fact recall.

### ARC grounding-monitor benchmark

Fresh run: `python -m experiments.benchmark_arc --output experiment_results/benchmark_arc.json`

- **Pre-ARC** composite quality: `0.395`
- **Post-ARC** composite quality: `0.394`
- **Pre-ARC** HONEST-abstinence: `0.600`
- **Post-ARC** HONEST-abstinence: `0.700`
- **Confabulation rate**: `0.000` both before and after
- **Salad rate**: `0.000`
- **Verdict from script**: `ARC IMPROVES QUALITY`

## Documentation

See [`docs/`](docs/README.md):

- [Architecture](docs/ARCHITECTURE.md) — turn-level data flow and the three packages.
- [Modules](docs/MODULES.md) — what lives in each package.
- [Training](docs/TRAINING.md) — `train.py` modes and the LingGen promotion gate.
- [Benchmarks](docs/BENCHMARKS.md) — every benchmark/diagnostic script and what it measures.
- [Development](docs/DEVELOPMENT.md) — layout, path shims, test commands, conventions.

## Design principles

1. **Fail-closed grounding** — abstain rather than confabulate.
2. **No fixed thresholds where a distribution exists** — gates are data-derived.
3. **Continuous, curiosity-driven learning** — what to learn is selected from
   prediction error, novelty, and contradiction, not a fixed list.
4. **Learning without backprop** — `ravana_ml` minimizes free energy and
   consolidates during `sleep_cycle()`.

## License

Oxiverse Community License (OCL) v1.0 — see [LICENSE](LICENSE).
