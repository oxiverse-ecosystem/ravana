# RAVANA Documentation

RAVANA is a **decoder-first ML cognitive architecture**: a system that starts
with a small "baby" vocabulary and learns continuously from conversation and the
open web — no LLM, no pretrained chat model. Knowledge is stored as a typed
concept graph; language is produced by a small neural decoder conditioned on
graph walks; cognition is orchestrated by a 20-phase "GRACE" governor (A–P).

This folder contains the engineered documentation. The code is the source of
truth; every claim here was checked against `ravana/`, `ravana_ml/`, and
`ravana-v2/` at the time of writing.

## Contents

| File | What it covers |
|------|----------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | 5-minute quickstart: install, first run, where to go next. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | End-to-end data flow: input → brain-repair → graph → decoder → response. Package relationship diagram (Mermaid). |
| [WHICH_ARCHITECTURE.md](WHICH_ARCHITECTURE.md) | Guide to `ravana/` (chat engine) vs `ravana-v2/` (GRACE governor) — which to use and why. |
| [MODULES.md](MODULES.md) | Map of the three source packages (`ravana`, `ravana_ml`, `ravana_grace`) and the key modules in each. |
| [CONCEPTS.md](CONCEPTS.md) | Theoretical foundations: pressure, free energy, Hebbian learning, governor, identity, sleep, VAD, RLMv2. |
| [TRAINING.md](TRAINING.md) | `scripts/train.py` modes (phase2 / full / test / linggen), the corpus, the decoder, and the LingGen promotion gate. |
| [BENCHMARKS.md](BENCHMARKS.md) | Every benchmark/diagnostic script under `scripts/` and `experiments/`, what it measures, and how to run it. |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Repo layout, the test suite, how to run it, the path shims, and contribution conventions. |
| [API_REFERENCE.md](API_REFERENCE.md) | Comprehensive class/function reference for all three packages. |
| [FAQ.md](FAQ.md) | Troubleshooting: installation, runtime, development issues. |

## Quick orientation

- **Run the chatbot:** `python scripts/ravana_chat.py`
- **First run:** see [GETTING_STARTED.md](GETTING_STARTED.md)
- **Train / promote:** `python scripts/train.py --mode <phase2|full|test|linggen>`
- **Autonomous learning:** `python scripts/ravana_learn.py`
- **Tests:** `python -m pytest tests/ci -v --ci` (fast) or `tests/` (full)
- **Install:** `pip install -e .[full,dev]` (see `requirements.txt`)

## Design principles (enforced in code)

1. **Fail-closed grounding.** When the system cannot honestly answer, it
   abstains. Confident-wrong is treated as high free-energy that would poison
   the graph (see `ravana/chat/coherence_gate.py`,
   `ravana/chat/junk_scorer.py`).
2. **No fixed thresholds where a distribution exists.** Gating decisions are
   driven by data-derived distributions, not hardcoded constants (brain-repair
   layer in `ravana/chat/brain_regions.py`).
3. **Continuous, adaptive learning.** The curiosity drive selects what to learn
   from prediction error, novelty, and contradiction — not a fixed topic list.
4. **Learning without backprop.** `ravana_ml` is a CPU-native tensor framework
   where learning emerges from free-energy minimization and sleep consolidation.
