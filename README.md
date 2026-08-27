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
- **Records its own opinions and answers "do you still feel that way?" from the
  record.** Asked its own view — *"what do you think about open source"* — it
  replies *"i strongly value open source. knowledge should be shared, not locked
  away."* and **records that stance durably** (survives save/load). A later
  *"do you still feel that way about open source?"* is answered **from that
  recorded stance** — *"yeah, i still strongly value open source — that hasn't
  shifted for me. knowledge should be shared, not locked away"* — not recomputed
  fresh. A revisit on a topic it never stated a view on is answered honestly
  (*"i don't actually have a recorded view on … from before"*) instead of
  fabricated. No LLM, no per-topic reply table. See
  `docs/CAPABILITY_AGENT_OWN_STANCE_PERSISTENCE.md`.
- **Reverses a held stance.** If you later change your mind — *"i flipped, the
  reef tank is more work than joy"* — it **recodes** the stance you already held
  toward the opposite pole (`reef tank` +0.95 → −0.665) instead of leaving the
  stale one or stacking a contradiction. A flip on a topic you never stated an
  attitude about is a harmless no-op. See `docs/STANCE_REVERSAL.md`.
- **Recodes a held stance on a FREE-FORM contradiction (no retraction keyword).**
  You don't have to say *"i flipped"* — an opposed restatement with no retraction
  cue still recodes the stance you hold: *"not all street art is good"* after
  *"i love street art"* moves `street art` from +0.95 to −0.275; *"actually i've
  gone off winter"* recodes the `silence` stance it already holds (the broader
  co-mention bridges via provenance). Detection is a seed reassessment-affect
  lexicon + `recode_stance_toward` (decisive blend toward the new value); a
  same-sign reassessment or a neutral utterance leaves the stance untouched, and
  there is no guessed reversal. No LLM, no retraining. See
  `docs/CAPABILITY_FREE_FORM_CONTRADICTION_RECODE.md`.
- **Links a broader-concept co-mention back to a held stance (provenance
  bridge).** Told *"i love the silence of deep winter"* it records the stance
  keyed on the subordinate head *silence* **and** keeps the salient broader
  concept *winter* it co-named as provenance. A later *"am i for or against
  winter?"* then resolves through that provenance to the held stance and answers
  *"from what you've told me, you're strongly for silence"* — instead of falling
  to the *"i don't have a read"* hedge it used before. Provenance is grown online
  from the real utterance and merged across encounters; there is no per-topic
  table and no retraining. The same bridge fixes the street-art reversal class
  (a reversal naming *street art* links to a stance keyed *murals*). See
  `docs/CAPABILITY_STANCE_PROVENANCE.md`.
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
  coffee?*" before forming its own position, it returns an honest non-answer
  rather than fabricating one:
  `i'm still figuring that out. i don't have a settled view on that yet — what do you think?`
- **Reflects on its model of you (meta-identity).** Asked *"do i seem like a
  real person to you"*, *"what am i to you"*, or *"what have you learned about
  me"*, it answers from its **live** accumulated model of you — your real name,
  the stances and facts it has picked up, and its own self-coherence — instead
  of a biographical fact lookup or an episodic echo:
  `i know you as Corvin. and from what you've told me i've picked up 2 stances you've shared and 1 facts about your life. you've let me see where you stand on things like oysters, surveillance. my own sense of self is still forming — my self-coherence sits around 0.25 and is holding steady.`
  Every word of content is read from runtime stores (no authored prose; the
  prior probe-tuned "feeling-real" frame was deleted). Fail-closed: a plain
  *"what's my name"* is not intercepted and still resolves from its own path. See
  `docs/CAPABILITY_META_IDENTITY.md`.
- **Reports the actual learned profile (content aggregation).** Asked *"what
  have you picked up about me"*, *"describe me"*, *"what stands out about me"*,
  *"tell me about myself"*, or *"what's your read on me"*, it surfaces the
  **real content** of its model of you — your name, where you're from, disclosed
  facts, stated beliefs, and the polarity of each stance it holds — read live
  from the durable stores:
  `here's what i've picked up about you so far: your name is corvin; you're from aldermoor in the hills; you grew village called aldermoor; you an astronomer who studies pulsars; on how you feel about things: you're strongly for sea; you're strongly against put.`
  This is distinct from meta-identity (which reports *counts + topics*, not the
  facts themselves). Previously these queries fell through to the
  graceful-uncertainty path and emitted degenerate text despite real facts being
  stored. Fail-closed: a brand-new user returns `None` and the honest path
  answers. No LLM, no per-topic reply table, no retraining. See
  `docs/CAPABILITY_USER_MODEL_AGGREGATION.md`.
- **Enumerates the entities it has learned in a category.** Asked *"name everyone
  in my family"*, *"name all my pets"*, or *"who have i told you about"* — queries
  with **no specific cue word** — it **scans its live PersonalFactStore** and
  lists every relative and pet it mined, drawn from the real stored facts:
  `you've told me about: your grandmother indira weaves baskets; your brother arjun climbs mountains; your cat is mochi; your dog is biscuit.`
  Previously these fell through to a generic acknowledgement (*"noted."*) because
  the cued-recall paths require a named entity. Category membership is decided by
  the **shared** lexicon helpers the miner and cued-recall already use, so all
  three paths agree on what counts as a relative/pet by construction (no
  duplicated word list). A brand-new user with nothing disclosed gets an honest
  *"you haven't told me about any family or pets yet."* instead of a fabricated
  list. No LLM, no per-topic reply table, no retraining. See
  `docs/CAPABILITY_CATEGORY_ENUMERATION_RECALL.md`.
- **Reads the USER's own held stance on a third-person query (self/other
  boundary).** Asked *"do you think i like spicy food or not?"* — where *you* are
  the attitude holder — it answers from **your** stored preference, not its own:
  `from what you've told me, you're strongly for spicy food.` (a disclosure of
  *"i hate cold coffee"* is later recalled the same way: *"you're strongly against
  cold coffee."*). Previously these matched the broad self-opinion gate and RAVANA
  answered from its *own* (empty) stance — the generic *"still figuring that out"*
  hedge — a self/other confusion. The topic is resolved the **same way the stance
  miner resolves it**, so a paraphrase (*"i adore jazz"* → query *"do you think i
  love jazz"*) still links to the held stance; the polarity is rendered as ONE word
  from the live store. Fail-closed: a topic you never stated a preference on, or a
  genuine question about *RAVANA's* own view, falls through to the normal path and
  is **not** answered with a fabricated stance. No LLM, no per-topic reply table,
  no retraining. See `docs/CAPABILITY_USER_STANCE_RECALL.md`.
- **Keeps a stance recallable when you name two activities in one breath.**
  A disclosure like *"i adore cold water swimming jumping"* used to mine a
  **run-on** stance key `cold water swimming jumping` that a later co-mention
  (*"am i still into cold water swimming?"*) could never bridge — so the stance
  was unrecallable. Now a morphological cut in `user_model._opinion_topic`
  (`user_model.py:3658`) truncates the object head at the first second-activity
  gerund, landing the key on the single salient activity (`cold water swimming`)
  while leaving single-activity objects (`mountain climbing`, `fossil hunting`)
  whole and still feeding the `does`/`event` fact miners through the same
  chokepoint. No per-topic rule, no retraining. See
  `docs/CAPABILITY_MULTI_ACTIVITY_STANCE_KEY.md`.
- **Recalls what it knows about a named relationship or person from open
  phrasing.** Asked *"tell me about my grandmother"*, *"who is my grandmother?"*,
  *"what does my grandmother do?"*, *"what do you know about my brother"*, or
  *"describe my niece priya"* — it reports the stored relationship/pet fact from
  the **same** open phrasing, not just a bare *"who is X"*:
  `your grandmother indira bakes sourdough bread.` (and *"who is theo?"* → *"your
  brother theo fixes bicycles."*). Pets are covered too (*"tell me about my cat"*
  → *"your cat is pixel."*). This needed two fixes: the relationship miner now
  stores the named fact regardless of name casing (it previously required a
  CAPITALIZED name and silently dropped lowercase chat names), and a new
  recall branch keys on the relationship word itself when phrased openly. The
  branch is gated on an interrogative frame so declarative disclosures (*"my
  friend is hurting"*) still reach the empathy router, and an unknown relative
  fails closed with honest uncertainty rather than a fabricated bio. No LLM, no
  per-person reply table, no retraining. See
  `docs/CAPABILITY_OPEN_ENDED_RELATIONSHIP_RECALL.md`.
- **Mines relationship attribute / enumeration disclosures (no activity verb, no capitalized name).** Told *"my grandmother yaya speaks three languages: greek, french, and italian"* — a lowercase relative, a non-activity verb (`speaks`), and a colon-enumeration — it now mines the combined-attr fact (`('i','grandmother yaya') -> 'speaks three languages: greek, french, and italian'`) and recalls it grammatically, enumeration intact, **without** a spurious copula: *"your grandmother yaya speaks three languages: greek, french, and italian."* A paraphrase (*"does yaya still speak those three languages?"*) resolves to the same stored fact. This generalizes the existing relationship miner's verb gate to a **seed** relation-verb lexicon (`speaks/works/studies/plays/…`), so other relationships (`"my uncle ravi works as a mechanic"`) mine the same way; location verbs stay owned by the location miner (no double-store), and a name-less/no-content disclosure (`"my grandmother"`) is correctly skipped (no degenerate fact). No LLM, no per-relationship reply table, no retraining. See `docs/CAPABILITY_RELATIONSHIP_ATTR_ENUMERATION_MINING.md`.
- **Recalls non-kin relationships (mentor / teacher / coach / friend) from open phrasing.** After a disclosure like *"my mentor Dr. Okonkwo taught me astronomy"*, asked *"who is my mentor?"*, *"tell me about my mentor"*, or *"what does my mentor do?"* — it reports the **full** relationship fact (`your mentor dr. okonkwo taught astronomy.`) from the same open phrasing as kin, with the full name + activity and no truncation. This needed a seed-vocabulary fix: non-kin role words (mentor, teacher, coach, friend, neighbour, boss, …) now live in the **shared** `relation_attrs` lexicon (single source of truth) instead of a duplicate local list, so the appositive-pet miner rejects them via its `relation_of()` guard instead of mis-storing *"my mentor Dr…"* as a bogus pet fact (`('i','mentor','dr')`) that truncated recall to *"your mentor is dr."* The role vocabulary is seed and grows at runtime via `learn_relation`. No LLM, no per-role reply table, no retraining. See `docs/CAPABILITY_NONKIN_ROLE_RECALL.md`.
- **Answers what *you* have told *it* — autobiographical recall of the USER.** Asked *"what will you remember most about me?"* it composes from your REAL profile (the most-confident learned fact/stance first, then a short tail), e.g. *"the thing that stands out most is your brother theo restores vintage radios."* Asked *"did i tell you i liked cold-weather hiking?"* it confirms from your REAL stance (*"yes — you told me you're uncertain about cold weather hiking. i've kept that."*) — and says *"not that i recall"* honestly when nothing maps. Asked *"earlier i told you i loved X. does that still fit, or have i changed?"* it reports your CURRENT (already-reconciled) stance, not a stale echo. This fixes a self/other boundary inversion: those queries used to be misrouted into RAVANA's own-reply echo store (returning *"i said: good to know you love…"* about the user's own disclosure). The answers are composed entirely from the live `personal_facts` / `opinions.stances` / `belief_store` — no authored prose, no per-topic table, no retraining. Genuine agent-self questions (*"what did you say about music?"*) still fall through untouched. See `docs/CAPABILITY_AUTOBIOGRAPHICAL_RECALL.md`.
- **Recalls a possession's name even when you PARAPHRASE the entity.** After *"i keep a sourdough starter i named doris"*, asked *"what did i name that sourdough culture on my counter?"* it links the paraphrase to the stored entity via cross-lemma GloVe cosine and answers *"your sourdough starter's name is doris."* — instead of leaking an unrelated "i"-scoped name fact (the R1 confabulation where it used to answer the best-friend's name). The linker shares the engine's seed GloVe embeddings, requires a verbatim head-word overlap as a confabulation bar, and **fails closed** (honest "i don't know" / no leak) when no stored entity clears the bar — so an unknown possession never gets a fabricated name. It runs before the self-profile scanners so a generic *"what is my name?"* is not hijacked by a possession. No LLM, no synonym table, no retraining. See `docs/CAPABILITY_ENTITY_LINKED_NAME_RECALL.md`.
- **Separates world-knowledge questions from autobiographical recall.** Asked
  *"what is cooking oil made of?"* it does **not** echo an unrelated stored fact
  about you (*"you enjoy cooking pasta on weekends"*) — the query is classified
  as a general-knowledge question and falls through to internal-knowledge / web /
  honest-uncertainty. The same phrase *"what is wrong with my car?"*, because it
  references your **own** disclosed entity (*my* car), is still answered from
  episodic memory (`gps`, `reboot`). The gate is a distribution-driven intent
  classifier (explicit recall markers + a personal-possessive reference), not a
  frozen topic list, so it generalizes across every subject and needs no
  retraining. Fail-open: a general knowledge question can never be answered by an
  autobiographical echo. See `docs/CAPABILITY_QUERY_INTENT_GATE.md`.
- **Withholds word salad about a subject it has never learned (D4).** The
  Situation-Model free-decode path used to restate a query's own near-neighbours
  as a "fact" about a subject RAVANA has *no* durable knowledge of (e.g. *"tired"*
  — no definition, no web source, not in the concept graph), and the grounding
  monitor accepted it because those neighbours are all GloVe-similar. Now an
  **unknown** subject — not in the concept graph / no definition / no web source
  — can no longer be grounded by free-association similarity alone: its utterance
  is withheld and the path falls back to honest uncertainty. A **known** concept
  (already learned, or with a seeded definition) still grounds a genuine answer,
  and a subject learned later online is re-admitted. No LLM, no per-topic reply
  table. See `docs/CAPABILITY_SM_UNKNOWN_SUBJECT_GROUNDING.md`.
- **Stops parroting your affect as its own reasoning (D3).** The in-prompt
  causal reasoner used to intercept a combined *"statement + question"* turn like
  *"that parking lot plan makes my blood boil. do you get why i'm furious?"*,
  mine your **affective statement** as a causal premise, and replay your own
  clause *"my blood boil"* verbatim as its reply — a source-monitoring failure.
  Now `parse_causal_edges` refuses to bind a premise whose **effect is a
  first-person affective self-report** (*"my blood boil"*, *"makes me furious"*,
  *"my heart races"*): that is a felt state, not a world-state transition, so the
  turn falls through to the genuine affective-response path. Detection is
  **seed-driven** — first-person pronoun (closed-class grammar set) + an
  affect-bearing word read from RAVANA's own learnable VAD lexicon (reused from
  the intent router, grown online via Hebbian learning), not a keyword table or
  authored prose. A legitimate world-state conditional (*"when you turn on the
  lamp, it lights up"*) still binds and answers. No LLM, no per-topic reply table,
  no retraining. See `docs/CAPABILITY_SOURCE_MONITORING_AFFECTIVE_ECHO.md`.
- **Recalls relationship disclosures made with an auxiliary verb (does/did + activity).** Told *"my cousin Jin does competitive speedcubing"* — where *"does"* is neither an activity verb nor a relation verb — it no longer drops the disclosure and later answers *"what does my cousin jin do"* with *"your cousin jin does competitive speedcubing."* (copula-free, not *"is does"*, and not the prior *"cousin is a bit outside what i know right now"*). The auxiliary is now a **third** verb class in the relationship miner (after activity verbs and relation verbs, both already generalized), opening the same capture path (name = tokens before it, value = aux + activity noun-phrase). The recall grammar rule that drops the copula for verb-phrase values (`is_verb_phrase`) now covers all three classes. The aux vocabulary is seed data that grows at runtime — no code change, no retraining, no LLM, no per-relationship reply table. See `docs/CAPABILITY_AUX_VERB_RELATIONSHIP_RECALL.md`.

These capabilities are backed by four durable stores
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
