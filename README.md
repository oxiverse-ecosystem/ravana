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

## What RAVANA does

RAVANA is not a fixed feature list — its user-facing capabilities are **derived
from its live self-model and memory stores**, and grow as it talks to you. The
behaviors below were each observed in a real in-process probe (dim=64, offline)
on this codebase:

- **Chats and discloses what it is.** Asked *"who are you?"* it replies from its
  identity model, not a script:
  `i'm ravana, cognitive architecture — an ai that learns by talking, not a person. what made you curious?`
- **Learns personal facts from conversation.** Told *"i live in berlin"* it
  stores a fact and confirms: `noted — i'll remember you live in berlin.`
  (persisted as `('i','location','berlin')`, confidence 0.7).
- **Forms and recalls stances.** Told *"i love coffee"* it records a stance
  (`coffee` polarity +1.0, confidence 0.65) and acknowledges:
  `good to know — you love coffee. i'll keep that in mind.`
- **Reverses a held stance.** If you later change your mind — *"i flipped, the
  reef tank is more work than joy"* — it **recodes** the stance you already held
  toward the opposite pole (`reef tank` +0.95 → −0.665) instead of leaving the
  stale one or stacking a contradiction. A flip on a topic you never stated an
  attitude about is a harmless no-op. See `docs/STANCE_REVERSAL.md`.
- **Corrects itself.** A later *"no, my cat's name is rex"* supersedes the
  earlier *"my cat's name is milo"* — both the old and new values are tracked in
  the fact store, and recall reflects the correction:
  `from what you've told me, you live in berlin; your cat is rex; …`
- **Learns what you *do*, not just who you are.** Told *"i tide-pool at low
  water and catalogue the anemones and limpets"* or *"i astrophotograph the milky
  way"* it captures the activity as a personal fact (`('i','does') ->
  "tide-pool ..."`) — including **novel and hyphenated-compound verbs** it had
  never seen, via an open-class (deny-list) capture rather than a frozen verb
  whitelist. The same turn is recognised as a self-disclosure and acknowledged,
  instead of leaking into a "i don't know that" knowledge query. See
  `docs/OPEN_CLASS_VERB_CAPTURE.md`.
- **Dates a year-only temporal start.** Told *"i have been firing my kiln since
  2017"* (with a session date set) it anchors the disclosure to **1 January 2017**
  and later answers *"when did i start firing"* with the grounded date —
   `you mentioned that around 1 January 2017.` — instead of returning empty.
  The anchor is pure date arithmetic over a seed cue set (no LLM, no hardcoded
  reply), gated behind a session date like every temporal-grounding feature. See
  `docs/DATE_GROUNDED_RECALL_YEAR_ANCHOR.md`.
- **Recalls what you told it.** *"what do you remember about me?"* surfaces the
  learned facts/stances (location, pet, likes) drawn from the durable stores.
- **Mines activity durations into dated facts.** Told *"i've been brewing beer
  for a decade"* (or *"a few years"*, *"two decades"*, *"several years"*, *"many
  years"*) it resolves the fuzzy span to a start year (`now − n`) and stores a
  `since` fact — then answers date queries through the **same** resolver as
  explicit years: `when did i start brewing beer` → `you started brew in 2016.`
  No per-phrase code; the resolver already knows how to read a `since` fact.
  See `docs/CAPABILITY_DURATION_MINING.md`.
- **Recalls the right dated fact even when you paraphrase.** A rotated query that
  shares no word with the stored activity still recalls it — *"what year did i
  start all this volcano stuff again"* → *"you started studying volcanoes back in
  2015."* — because the resolver links each `does`/`event` fact to the dated
  `since` activity by **morphological stem** (so *volcano* in a separate
  `start studying volcanoes` fact reaches the `study 2015` fact). The reply is
  also grammatical: the stored verb is realized as a **gerund** ("started
  **studying** volcanoes", not "started **study**"), and a redundant inceptive
  ("started studying…") is collapsed to the gerund. No LLM, no per-topic reply
  table. See `docs/CAPABILITY_DATE_RECALL_PARAPHRASE.md`.
- **Tells two activities apart when they share a verb but differ by object.**
  Told *"i've been building frames since 2019"* and *"i started building cabinets
  in 2021"*, it mines the object (`frames` / `cabinets`) into each dated fact and
  recalls the right one: *"when did i start building frames"* → *"you started
  building frames in 2019."*, and *"since what year have i been building
  cabinets"* → *"you started building cabinets in 2021."* Previously both returned
  the same (wrong) year because only the verb head was stored. No LLM, no
  per-topic reply table. See `docs/CAPABILITY_OBJECT_DISAMBIGUATED_DATE_RECALL.md`.
- **Mines possession-attribute disclosures into structured, correctable facts.**
  Told *"the cabin is a hand-hewn pine lodge with a sod roof"* it stores the
  material under the **entity** (`cabin.madeof = pine`), not a whole-sentence
  echo of you — so a later *"what's my cabin made of"* returns the clean
  structured answer *"your cabin is made of pine."* A feature noun after the
  material scopes the fact (*"my desk is oak frame"* → `desk.frame = oak`,
  recalled as *"your desk's frame is oak."*). A possession with no recognised
  material (*"the river is a fast mountain stream"*) is correctly **not** mined
  (fail-closed, no echo). The material/kind vocabulary is seed data that grows
  at runtime (`learn_material`) — no code change, no retraining, no LLM. See
  `docs/CAPABILITY_POSSESSION_ATTRIBUTE_MINING.md`.
- **Abstains when it has no settled view.** Asked *"what do you think about
  coffee?"* before forming its own position, it returns an honest non-answer
  rather than fabricating one:
  `i'm still figuring that out. i don't have a settled view on that yet — what do you think?`
- **Forms and recalls its own stance on a topic you've discussed.** Asked
  *"what do you think about chanterelles?"* after you've said *"i really love
  chanterelles"*, it answers from a stance it **derived and recorded as its own**
  (`i'm strongly for chanterelles.`) — grounded in your real learned view,
  attenuated (it leans, never copies), and persisted so it recalls the same stance
  next time. On a topic with no evidence it stays honestly silent
  (`i'm still figuring that out …`) instead of borrowing your opinion. No LLM, no
  retrain, no authored reply pool. See `docs/AGENT_SELF_STANCE.md`.
- **Engages BOTH sides of a binary self-opinion.** Asked *"what's your take on the
  sea versus the mountains?"* or *"do you prefer the countryside or the cities?"*,
  it splits on the contrastive connective (`versus` / `vs` / `or` / `over` /
  `rather than`) and resolves **each** side through its real stance state — e.g.
  `i'm for sea.; i'm still figuring out mountains.` — instead of collapsing to the
  last token and dropping the other side. A side with no view is answered honestly,
  never fabricated. No LLM, no retrain, no authored reply pool. See
  `docs/CONTRASTIVE_SELF_OPINION.md`.
- **Resolves a relative-clause topic to its content head in a self-opinion
  query.** Asked *"your honest read on people who talk in theatres?"* or
  *"what's your take on friends who keep their promises?"*, it no longer
  collapses the topic to the trailing last token (`theatres` / `promises`) — it
  resolves the **content head** (`people who talk` / `friends who keep`), which
  matches the stance key it mined from you, so it engages the real lean it
  learned (`i'm against people who talk.`) instead of the hollow
  `i'm still figuring that out`. Flat topics (`"your honest read on privacy"` ->
  `privacy`) are unchanged; an ungrounded relative clause stays honestly silent,
  never fabricated. No LLM, no retrain, no authored reply pool. See
  `docs/SELF_OPINION_RELATIVE_CLAUSE.md`.
- **Remembers and totals counts you disclose.** Told *"i keep twelve racing
  pigeons"* / *"i have three cats"* / *"i lost five hens"*, it stores each count as
  structured state (not free text) so it can answer *"how many racing pigeons do i
  keep?"* with `you have twelve racing pigeons.` and *"how many pets do i have in
  total?"* with `you have 21 pets in total.` (losses aren't counted as pets) — and
  a later *"it's seven hives now"* supersedes an earlier *"i keep six hives"*, so
  recall returns the corrected `you have seven hives.` No LLM, no retrain, no
  authored reply pool. See `docs/QUANTITY_MEMORY.md`.

These capabilities are backed by four durable stores — an **identity model**
(`IdentityEngine`), **stances** (`UserStanceStore`), **personal facts**
(`PersonalFactStore`), and **beliefs** (`BeliefStore`) — plus a **ConceptGraph**
world-model. The README's benchmark and architecture sections describe the
substrate; the stores above are what a user actually experiences.

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
- [Entity-Location Recall](docs/ENTITY_LOCATION_RECALL.md) — capturing + surfacing a named thing's whereabouts.
- [Quantity Memory](docs/QUANTITY_MEMORY.md) — capturing counts you disclose, answering "how many", totalling "in total", correcting online.
- [Reverse Pet Lookup by Name](docs/PET_NAME_RECALL.md) — answering "who is wren to me?" by reverse-indexing the pet store by the name value.
- [Agent Self-Stance](docs/AGENT_SELF_STANCE.md) — RAVANA forms, records, and recalls its own stance on a discussed topic (grounded in your view, attenuated, persisted), and stays honestly silent otherwise.

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
