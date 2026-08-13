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
| [STANCE_REVERSAL.md](STANCE_REVERSAL.md) | How the user changing their mind recodes a held stance (first-person reversal / retraction cues), verified against the live engine. |
| [ENTITY_LOCATION_RECALL.md](ENTITY_LOCATION_RECALL.md) | Capturing + surfacing a named thing's whereabouts (entity-keyed `location` fact, "where is X?" recall, correction, fail-closed on unknown). |
| [POSSESSION_REATTRIBUTION.md](POSSESSION_REATTRIBUTION.md) | Reverse-order pet naming + owner re-attribution (self/other boundary enforced at all four recall sources, no LLM). |
| [PET_NAME_RECALL.md](PET_NAME_RECALL.md) | Reverse pet lookup by name: answer "who is wren to me?" by reverse-indexing the pet store by VALUE, honoring rename + self/other boundary, no LLM. |
| [AGENT_SELF_STANCE.md](AGENT_SELF_STANCE.md) | RAVANA forms, records, and recalls its own stance on a discussed topic (grounded in the user's view, attenuated, persisted), and stays honestly silent with no evidence. |
| [CONTRASTIVE_SELF_OPINION.md](CONTRASTIVE_SELF_OPINION.md) | Binary "X versus Y" / "prefer A or B" self-opinion: RAVANA engages BOTH named sides through real state (split on the connective, resolve each side, compose), no single-token collapse. |
| [SELF_OPINION_RELATIVE_CLAUSE.md](SELF_OPINION_RELATIVE_CLAUSE.md) | Single-topic self-opinion whose topic is a relative clause: RAVANA resolves the CONTENT HEAD (e.g. "people who talk") instead of the last token ("theatres"), matching the mined stance key, no hollow-fallback collapse. |
| [FAQ.md](FAQ.md) | Troubleshooting: installation, runtime, development issues. |

## Quick orientation

- **Run the chatbot:** `python scripts/ravana_chat.py`
- **What it actually does:** see [`../README.md`](../README.md) → *"What RAVANA does"*
  for the user-facing capabilities (chat/identity, learning facts, stances,
  self-correction, recall, honest abstention) observed against the live engine.
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
