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
| [CAPABILITY_DURATION_MINING.md](CAPABILITY_DURATION_MINING.md) | How "i've been brewing beer for a decade" / "a few years" become a dated `since` fact recallable by `when did you start…` — four duration-mining blocks, verified against the live engine. |
| [CAPABILITY_POSSESSION_ATTRIBUTE_MINING.md](CAPABILITY_POSSESSION_ATTRIBUTE_MINING.md) | How "the cabin is a hand-hewn pine lodge with a sod roof" becomes a structured `cabin.madeof = pine` fact recallable as "your cabin is made of pine" (entity-scoped, fail-closed, seed-vocab), verified against the live engine. |
| [CAPABILITY_DATE_RECALL_PARAPHRASE.md](CAPABILITY_DATE_RECALL_PARAPHRASE.md) | How rotated/paraphrased date queries ("all this volcano stuff") still recall the right `since` fact (stem-linked `does`/`event` facts) and reply grammatically (morphological gerund), verified against the live engine. |
| [CAPABILITY_META_IDENTITY.md](CAPABILITY_META_IDENTITY.md) | How "do I seem like a real person to you" / "what am I to you" / "what have you learned about me" are answered from RAVANA's LIVE model of the user (name, stances, facts, identity strength/trend) — not a biographical fact or an episodic echo, verified against the live engine. |
| [CAPABILITY_USER_MODEL_AGGREGATION.md](CAPABILITY_USER_MODEL_AGGREGATION.md) | How "what have you picked up about me" / "describe me" / "what stands out about me" report the REAL learned profile (name, facts, beliefs, stance polarities) from the live durable stores — not degenerate uncertainty text, verified against the live engine. |
| [CAPABILITY_CATEGORY_ENUMERATION_RECALL.md](CAPABILITY_CATEGORY_ENUMERATION_RECALL.md) | How "name everyone in my family" / "name all my pets" / "who have i told you about" SCAN the live PersonalFactStore and enumerate every relative + pet it mined — not a `noted.` ack, verified against the live engine. |
| [CAPABILITY_USER_STANCE_RECALL.md](CAPABILITY_USER_STANCE_RECALL.md) | How "do you think i like spicy food or not?" reads the USER's own held stance (self/other boundary) — not RAVANA's empty stance — verified against the live engine. |
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
