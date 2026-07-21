# Getting Started

A 5-minute quickstart to install and run RAVANA.

---

## Prerequisites

- **Python 3.10+** (verified on 3.14)
- **NumPy** and **SciPy** (automatically installed)

That's it for the core. Optional dependencies (web search, embeddings, plotting)
are installed via the `[full]` extra.

---

## Install

```bash
# Clone (from your preferred mirror)
git clone https://github.com/oxiverse-ecosystem/ravana.git
cd ravana

# Editable install (recommended for development)
pip install -e .[full,dev]

# Or minimal install (core only):
pip install -e .
# Or using requirements.txt:
pip install -r requirements.txt
```

> **Note:** The three packages (`ravana`, `ravana_ml`, `ravana_grace`) are all
> installed from the same repo — `pip install -e .` covers all three.

---

## First run: check it works

```bash
# Quick diagnostic (no web access needed)
python scripts/train.py --mode test
```

This creates a `CognitiveChatEngine`, trains on 50 seed sentences, generates
a few responses, and saves weights to `data/ravana_weights.pkl`.

Expected output (approximate):
```
  [PMI] Seeded 195 concepts, 137 connections (112 PMI-wired, 25 GloVe-wired)
  [Vocabulary] 96 words
  [train seed corpus] CE=4.21 top1=0.21 top5=0.43 — early stopped at 50 sentences
  [save] Saved to data/ravana_weights.pkl
```

---

## Chat with the system

```bash
# Interactive chatbot
python scripts/ravana_chat.py
```

Type queries like:
- `what is trust`
- `tell me about love`
- `what happens when you learn`
- `explain oxiverse`

The engine auto-learns from the web when it hits a knowledge gap.

### Key commands in the chat interface

| Command | Effect |
|---------|--------|
| `exit` / `quit` | Save and exit |
| `--stats` | Print graph statistics |
| `--trace` | Enable chain-walk tracing |
| `--reset` | Reset saved weights and start fresh |
| `--dim N` | Set concept dimension (default 64) |
| `--data-dir PATH` | Custom data directory |

---

## Train the decoder properly

```bash
# Heavy training on seed corpus + web learning (~1 hour)
python scripts/train.py --mode phase2

# Full training (~3-5 hours)
python scripts/train.py --mode full

# LingGen sensorimotor promotion
python scripts/train.py --mode linggen
```

See [TRAINING.md](TRAINING.md) for all options.

---

## Run the background learner

```bash
# Autonomous learning (no chat — Ctrl+C to save)
python scripts/ravana_learn.py --cycles 10 --delay 2
```

---

## Run the test suite

```bash
# Fast CI-critical tests (~15 min)
python -m pytest tests/ci/ -v

# Full unit tests (~2.5 min)
python -m pytest tests/unit/ -q

# Complete suite
python -m pytest tests/ --tb=short
```

---

## Tutorials (step-by-step)

The `tutorials/` directory contains 7 progressive exercises:

```bash
# Run them in order:
python tutorials/01-chat-basics/run.py          # Create the engine, send queries
python tutorials/02-decoder-training/run.py      # Train the neural decoder
python tutorials/03-graph/run.py                 # Build and inspect a concept graph
python tutorials/04-continuous-learning/run.py   # Background web learning
python tutorials/05-experiments/run.py           # Run an experiment harness
python tutorials/06-governor/run.py              # Instantiate GRACE modules
python tutorials/07-rlm/run.py                   # Use RLMv2 triple model
```

---

## Where to go next

| If you want to... | Read this |
|-------------------|-----------|
| Understand the architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| See how the packages map out | [MODULES.md](MODULES.md) |
| Learn about training | [TRAINING.md](TRAINING.md) |
| Run benchmarks | [BENCHMARKS.md](BENCHMARKS.md) |
| Contribute | [DEVELOPMENT.md](DEVELOPMENT.md) |
| Deep theory | [CONCEPTS.md](CONCEPTS.md) |
| Which architecture to use | [WHICH_ARCHITECTURE.md](WHICH_ARCHITECTURE.md) |

---

## Troubleshooting

**"No module named 'ravana_ml'"** — Make sure you ran `pip install -e .` from the
repo root, not from a subdirectory. The `setup.py`/`pyproject.toml` finds all three
packages automatically.

**"data/corpora/teen_seeds.txt not found"** — Run
`python scripts/gather_teen_seeds.py` to regenerate the seed corpus.

**Slow first run** — The first run downloads GloVe embeddings and builds the
attribute encoder cache. Subsequent runs use the cached files.

**Windows access violation errors** — Try setting environment variables
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` before running. The engine
auto-pins these internally, but some NumPy/OpenBLAS builds still need the env
vars.
