# RAVANA causal-substrate code-map: where it can grow without confabulation

**Status:** read-only investigation at tree `6aabeda`. No code changed, no
harness edited, no `eng.save()`; no engine was booted for this document.
Evidence is `file:line` plus two direct read-only queries against the shipped
SQLite weights mirror (`weights/ravana_weights.db`, `metadata` table).
Companion docs that frame the decision: `BRAIN_REASONING_DESIGN.md` (production
criteria C1–C5), `BRAIN_CAUSAL_SUBSTRATE.md` (acquisition criteria G1–G5),
`AUTHORED_CAUSAL_SKELETON.md`, `WEB_NODE_ADMISSION_BLOCKED.md`,
`GAP_DRIVEN_WEB_LEARNING.md`, `WEB_LEARNING_CHAIN_VERIFIED.md`,
`PLAN_ROUTE_REPLICATION.md`. `PASSIVE_WEB_LEARNING.md` does not exist in this
tree.

> **Superseded (2026-11-19).** This bottom line was a read-only proposal at tree
> `6aabeda`, not current status. Production is sound; **acquisition is the
> deficit**. Only two subjects have a verified trusted 2-hop chain
> (`sun→light→photosynthesis`, `water vapour→clouds→rain`); the THINKING front
> was 19 offline / 41 web **at n=27** (`BRAIN_CAUSAL_SUBSTRATE.md:6-10`).
> **Current n=60:** honest offline thinking is **8/100** (primary 10, sealed 7)
> — see `BENCHMARK_60_ITEMS.md` §4. The web arm at n=60 timed out (2400 s, 207
> of 434 probes, no valid summary), so **no n=60 web score exists**; do not read
> the 41 above as current.
>
> This map proposed that the narrowest correct lever was a bounded offline
> pass promoting explicit causal propositions **already present in RAVANA's
> stored definitions** into provenance-stamped `definition`/`curated` edges,
> with verified chain shapes used only to choose which definitions to inspect.
> **That proposal was implemented, measured as a no-op, and REVERTED**:
> `WRITE_N=0`, `CANDIDATES=0` (only 8 of 583 definitions contain a causal-ish
> connective, most false positives). See `BRAIN_CAUSAL_SUBSTRATE.md` §10. More
> substrate must come from web reading with the network on or from authored
> curriculum edges — not from reinterpreting stored definitions.

---

## 1. Full path: web fact → trusted, assertable causal edge

### 1.1 Search provider (and why it is absent offline)

- `web/learner.py:85-133` `SearchConfig`; `:140-200` `SearchEngine.__init__`
  (local endpoint first, remote APIs as backups); `:323-545` `search()` with a
  whole-call deadline, per-session cache and fail-closed local-only mode.
- `:147` / `:323-340`: `RAVANA_OFFLINE=1` makes `search()` return `[]` with no
  network — the documented reproducible contract (`engine.py:1939`,
  `engine.py:14005-14024`).
- Ordinary ingest: `WebLearner.learn_from_web` `web/learner.py:701-744`;
  gap ingest: `WebLearningMixin.close_mechanism_gap`
  `chat/web_learning.py:715-725` builds the targeted query and calls
  `searcher.search(causal_query, max_results=cap)`.

### 1.2 OpenIE extraction

- `web/openie.py:38-121` `VERB_TO_RELATION`; the causal set is `:87-120`,
  including explicit process/phase-change verbs (`evaporate`, `condense`,
  `forms`, `scatter`, `leads`, `cause`, …) added after the measured zero-fact
  failure documented at `:74-86`.
- `:144-154` `Fact(subject, relation, obj, confidence, source_sentence)`.
- `:157-215` sentence split, tokenise, noun-chunk head selection; `:136-141`
  `NON_FACTUAL_HEADS`.
- `:219-255` `OpenIEExtractor.extract`. **Loss points here:** only the **first**
  mapped verb per sentence is used (`:226-230`); verb at token 0 is dropped
  (`:231`); interrogatives dropped (`:233-235`); subject/object must survive
  `_chunk` (`:240-243`) and pronoun/deixis filtering (`:244-246`); one fact per
  sentence; confidence is a flat 0.6/0.7 heuristic (`:247-250`).

### 1.3 `WebToGraph.learn_text` — relation mapping and edge write

- `web/web_to_graph.py:52-67` `RELATION_TO_EDGE`; `causes → causal` at `:65`
  (the PR #378 fix; without it no causal edge existed at all — see
  `WEB_LEARNING_CHAIN_VERIFIED.md:5-43`).
- `:206-298` `learn_text`: extracts facts (`:226`), ensures **both endpoint
  nodes** (`:229-230`), and **skips the whole fact** if either is `None`
  (`:231-232`). That single line is where most web facts die (see §1.4).
- With the engine's intake gate attached — `chat/engine.py:12671-12689`
  `_get_knowledge_intake`, used by `chat/web_learning.py:3010-3015` — the direct
  `add_edge` at `web_to_graph.py:285-297` is **bypassed**; facts go through
  `learn/intake.py:677-692` instead. Routes `fast_merge`/`novel_write`/`revise`
  write an edge (`:682-688`, `:694-739`); `buffer` writes **nothing**
  (`:689-690`). `written` is counted only for the writing routes
  (`web_to_graph.py:270-284`). Edge provenance is stamped from source type
  `"web"` → `edge_kind="web_fact"` (`intake.py:74-84`, `:727-738`).
- Without an intake gate, `learn_text` stamps `web_fact` directly at
  `web_to_graph.py:288-296`.

### 1.4 `_ensure_node` — the two-source promotion gate (primary bottleneck)

- `web/web_to_graph.py:126-204`. Junk gate `:146-165` (θ from
  `engine.py:2063` = 0.5); provisional buffer by **distinct `source_url` md5**
  `:166-204`; promotion only when `len(_prov[label]) >= k` at `:182-183`.
- `engine.py:2064`: `_promote_min_sources = 2`. A node read from **one** URL is
  held forever; no node → `learn_text:231` skips the fact → **no edge, no
  provenance record, no chain**. This is the dominant acquisition loss and the
  reason "most subjects lack a trusted 2-hop chain".
- The older degenerate-classifier root cause in `WEB_NODE_ADMISSION_BLOCKED.md`
  is **already fixed in this tree**: `chat/junk_scorer.py:536` requires
  `_classifier_has_signal` (`:549-586`), falling back to the structural formula.
  The remaining gate is convergence, not the junk scorer.
- Correct consequence: lowering `k` would write one-sighting edges and destroy
  the convergence property the trust gate measures — explicitly forbidden by
  `ravana-debugging` §"stamp provenance, do not loosen the gate" and by
  `BRAIN_CAUSAL_SUBSTRATE.md:317` (G2).

### 1.5 Gap-loop provenance on the write

- `chat/web_learning.py:727-806`: snapshot pre-existing causal edges (`:728-729`),
  collect distinct sources per `(s,causal predicate,o)` triple (`:735-754`),
  call `w2g.learn_text` per URL (`:755-756`), then re-stamp only newly written
  edges (`:768-801`) with `populate_provenance(edge_kind="web_fact",
  method="mechanism_gap_read", retrieval_conf, converging_sources)` plus
  `acquired_by="mechanism_gap"`, `gap_subject`, `gap_query` (`:788-796`) and
  evidence-scaled strength `conf + 0.15*(n_sources-1)` (`:775-798`, constants
  `:93-98`).
- **Measurement caution:** `close_mechanism_gap` reports `chain_hops` from an
  undirected connected-component count (`_gap_chain_hops`, `:819-857`), which
  is not the same object as a directed, provenance-trusted 2-hop chain. Per
  `ravana-debugging` §1, validate the probe before quoting its number.

### 1.6 Read path: enumeration → provenance → adequacy

- `learn/explanation.py:52-95` `Hop` / `ExplanationChain`.
- `:156-265` `enumerate_chains`: direction-aware traversal (`:204-207`), causal
  edges only (`:209`), simple-path/cycle guards (`:215-225`), min 2 hops
  (`:160`, `:245-252`).
- `:268-299` `_hop_provenance` reads `edge.source_metadata["edge_kind"]`:
  `web_fact → "web"` (`:285-286`); `definition`/`curriculum`/`curated` accepted
  (`:292-293`); `co_occurrence → ""` (`:294-295`).
- `:302-316` `provenance_trusted` accepts `web`, `definition`, `conversation`,
  `curriculum`, `curated`; `:319-328` `assertable_chain` fails closed (one bad
  hop sinks the chain); `:331-366` `grade_chain` computes adequacy
  `typed + conf − 0.5·gaps + 0.5·margin` and the `assert/hedge/decline` verdict.
- Consumers: `chat/engine_self_query.py:2436-2468` `_mechanism_chain_available`
  (a definition must not short-circuit a mechanism question);
  `response_gen.py:_why_how_causal_chain` (`:8245` per skill) and the
  `_final_emit_guard` pass-through for a gated explanation chain
  (`GAP_DRIVEN_WEB_LEARNING.md:72`).

### 1.7 Where facts are lost, ranked

1. **Node promotion needs 2 distinct sources** (`web_to_graph.py:149,182-183`,
   `engine.py:2064`) → one-page reads write zero edges (`learn_text:231`).
2. **Offline short-circuit** (`web_learning.py:690-692`) → the gap loop never
   supplies those two sources in the offline harness.
3. **Intake gate buffers** schema-surprising facts (`intake.py:689-690`), so
   even a promoted pair of nodes need not yield an edge.
4. **OpenIE is one-verb/one-fact-per-sentence** with head-noun and pronoun
   filters (`openie.py:226-246`) → most explanatory prose yields no usable
   triple.
5. **1 hop is never assertable** (`enumerate_chains` `min_steps=2`,
   `explanation.py:160,245`) → even a successful single web edge does not grow
   a subject's trusted chain.

---

## 2. Curated/physics seed edges and how they are stamped

- `chat/engine_reasoning.py:3185-3251` `_seed_physics_causal`, called at boot
  `engine.py:2214` (inside `try`, so failures are swallowed). Edge list
  `:3197-3241`: `sun→light` `:3198`, `sun→heat` `:3199`,
  `light→photosynthesis` `:3201`, `heat→climate` `:3206`, `water→life` `:3223`,
  and the `tree→fall→vibrate→air→sound` counterfactual chain `:3230-3240`.
  `gravity→earth` was deliberately **removed** `:3209-3222` as a false prior.
- Writer: `chat/engine_graph.py:887-941` `_ensure_relation`. Nodes are created
  with GloVe/hash vectors (`:898-916`) and stamped `edge_kind="curated",
  source="common_facts"` (`:914-915`). Edges are added with `confidence=0.9`
  (`:919-921`) and stamped the same way (`:922-924`); an already-existing edge
  is backfilled so the seed is idempotent w.r.t. provenance (`:925-941`).
- Consequence: these edges pass `_hop_provenance` (`explanation.py:292-293`)
  and, with confidence 0.9, clear `grade_chain`. This is the **only** path in
  the current tree that produces trusted chains offline.
- Honest limit: `curated` is a source class, not a truth guarantee —
  `AUTHORED_CAUSAL_SKELETON.md:38-60` and `BRAIN_CAUSAL_SUBSTRATE.md:433-436`
  require per-edge review. Do not blanket-stamp a fabricated edge.
- **Can a schema generalise to a new subject without a fresh web read? Yes —
  from the definition store, not from the template.** `engine.py:955`
  `self._definitions`, loaded at boot `engine.py:2207`; the shipped DB contains
  **583 definitions**, of which **115 contain a verb in the OpenIE
  `causes` lexicon** (read-only query of `weights/ravana_weights.db`,
  `metadata['definitions']`, cross-checked against `web/openie.py:87-120`).
  Concretely, `sky` is already stored as *"The sky appears blue because
  molecules in the air **scatter** shorter (blue) wavelengths of sunlight …
  — Rayleigh scattering"*; `scatter → causes` at `openie.py:110`. That is an
  explicit, already-sourced causal statement available offline — and `why does
  the sky appear blue` is a THINKING probe (`scripts/measure_fronts.py:180`).
  No two-source web read is needed **if** the definition itself is the source.

---

## 3. Gap registry and the gap→web loop under `RAVANA_OFFLINE=1`

- Registry: `_pending_learning_queue` `chat/engine.py:1052`,
  `_impossible_queries: List[FailedQuery]` `:1108`, model
  `chat/models.py:31`. Detection at `engine.py:11831-11847`
  (`_wants_mechanism` / `_got_mechanism` regexes); a `why/how` answer with a
  definition but no mechanism opens the gap. Append `:11853-11861`, queue
  `:11866-11870`, curiosity/sleep tagging `:11863-11886`.
- Call site: `engine.py:11902-11915` calls `self.close_mechanism_gap(...)`,
  guarded by `_mechanism_gap` **and** `self.baby_mode` at `:11849`, with a
  fail-open `try/except`.
- Loop: `chat/web_learning.py:661-817`. Hard offline short-circuit at
  **`:690-692`** returns `status="offline"` before any search; other bounds
  `:694-708` (2 sources, 4 reads/session, once per subject).
- **Does it fire in the harness? No.** `scripts/measure_fronts.py:248` and
  `:457-458` set `RAVANA_OFFLINE=1`; `search()` also returns `[]` offline
  (`web/learner.py:147,323-340`). The loop is entered only if `baby_mode` and a
  mechanism gap are detected, then immediately returns `offline`. The gap is
  recorded, curiosity/sleep pressure are raised, and the topic is queued — but
  **no search, no edges, and `_mark_mechanism_gap_resolved` (`:859-870`, called
  only on `status=="learned"` at `:814`) never runs**. The resolved case was
  demonstrated only with `RAVANA_OFFLINE=0` and a stub search
  (`GAP_DRIVEN_WEB_LEARNING.md:106-130`, subject `sonoluminescence`).

---

## 4. What a schema-based substrate change actually requires

`BRAIN_CAUSAL_SUBSTRATE.md:296-303` and G4 at `:319` are the binding
constraint: reuse the **relation shape** (`A→B→C` with process predicates),
but **relabelling an old chain's nodes is confabulation**. A schema instance
must be a new hypothesis with its own node IDs, its own edge objects, and its
own source; it may not inherit the template's evidence.

Requirements, in order:

1. **Candidate generation only.** A shape may select which definitions/queries
   to inspect, never admit an edge. `BRAIN_CAUSAL_SUBSTRATE.md:262-265,347-354`;
   tag pure schema instances `edge_kind="schema_hypothesis"`, which is **not**
   in `provenance_trusted` (`explanation.py:315-316`), so they are unassertable
   by construction.
2. **Independent source for each new hop.** For the offline case that source
   is the stored definition text itself, stamped `edge_kind="definition"`
   (`explanation.py:292-316` accepts it). For the web case it is ≥2 distinct
   `source_url`s on the **edge**, not just the node (G2,
   `BRAIN_CAUSAL_SUBSTRATE.md:317`).
3. **Write through the curated path** (`engine_graph.py:887-941`) so the edge
   gets `confidence=0.9` and a durable `source_metadata`; never through the
   node-promotion gate, which would throw the fact away.
4. **Review, not blanket trust.** Per-edge judgement as in
   `AUTHORED_CAUSAL_SKELETON.md`; a shape-congruent but false definition must
   not become a trusted edge.
5. **Leave the gates alone.** No change to `_promote_min_sources=2`,
   `provenance_trusted`, `assertable_chain`, `grade_chain`, or the adequacy
   threshold. Verify with `enumerate_chains` + `assertable_chain` +
   `grade_chain` on a frozen snapshot and an independent census (C1–C5 in
   `BRAIN_REASONING_DESIGN.md`; G3/G5 in `BRAIN_CAUSAL_SUBSTRATE.md`).

### Existing machinery that supports it (all already in-tree)

| Capability | File:line | How it helps |
|---|---|---|
| Definition → proposition reader | `learn/definition_proposition.py:118-173` (`_words` `:84`, `_head_noun` `:97`) | Structural reading of a stored definition; currently emits only `is_a`/`has_property` — the narrow extension point for a `causes` process predicate. Used at `response_gen.py:8150,8286`. |
| Process-verb lexicon + extractor | `web/openie.py:38-121`, `:219-255` | Already maps `evaporate/condense/scatter/forms/leads/… → causes`; can parse definition text with **no network**. |
| Definition store | `chat/engine.py:955`, boot `:2207`; 583 stored, 115 with causal verbs | The independent, already-sourced material for offline growth. |
| Provenance-stamped curated writer | `chat/engine_graph.py:887-941` | Creates nodes+edges with `edge_kind="curated"` / `source="common_facts"`; add `definition` source variant here. |
| Verified chain shapes | `chat/engine_reasoning.py:3197-3241`; reader `learn/explanation.py:156-265` | Templates for *selection* of analogous process language; the reader already knows a valid `A→B→C`. |
| Surface realisation | `learn/linearization.py:172-225` (`cause` construction), `:321` `message_to_plan`; `learn/committed_proposition.py:207,241` | Turns a committed causal edge into "A leads to B" without new generation machinery. |
| Schema store (heavier option) | `core/causal_schema.py:54-255` | State-transition schemas (`state_A/condition/state_B`), not graph-edge templates; would need a new edge-template view — not the narrow lever. |

---

## 5. The narrowest correct change

**Extend the offline seed path with definition-sourced causal edges.** After
definitions load (`engine.py:2207`) and `_seed_physics_causal` (`engine.py:2214`),
add one bounded pass that:

1. takes the two verified shapes (`sun→light→photosynthesis`,
   `water vapour→clouds→rain`) **only as selectors** for the process-predicate
   vocabulary to look for (`scatter`, `condense`, `evaporate`, `forms`,
   `leads`, … from `openie.py:87-120`);
2. runs the existing extractor over the **stored definition text**
   (`proposition_from_definition` extended to a `causes` case, or
   `OpenIEExtractor.extract` with a single-sentence guard) for a **reviewed**
   set of subjects;
3. resolves each object to an existing graph node and writes **new** edges via
   `_ensure_relation` (`engine_graph.py:887-941`) with
   `edge_kind="definition"`, `source="definition:<subject>"`, the definition
   text preserved in `source_metadata`, and confidence from the seed writer;
4. verifies with `enumerate_chains(min_steps=2)` → `assertable_chain` →
   `grade_chain`, plus the independent census; any instance that is only a
   relabelled template, or whose definition does not explicitly state the
   relation, is **not written** and the subject stays an honest gap.

This increases the count of subjects with a trusted chain **offline**, uses no
web read, adds no confabulated edge (every new hop has a real source that
states it), and leaves `_promote_min_sources=2`, `provenance_trusted`,
`assertable_chain` and `grade_chain` untouched. It is also the only option here
that fits the harness: the gap→web loop cannot grow the offline substrate at
all (§3), and lowering the promotion or trust gate would destroy the property
being measured (§1.4–1.6).

Explicitly **not** the change: writing `edge_kind="web_fact"` on a first
sighting, accepting `schema_hypothesis` as evidence, relabelling a verified
chain's nodes, or flipping `RAVANA_OFFLINE` in the benchmark harness to make a
number move.

### Verification checklist (cheap, single process, no `save()`)

- Cold `CognitiveChatEngine(data_dir=weights, user_suffix='')`; census
  `enumerate_chains` over the seeded subjects before/after with correctly
  iterated `graph.nodes.values()` (the dict trap in `ravana-debugging` §2).
- Print each new chain with per-hop `edge_kind` and `grade_chain` verdict.
- Negative controls: a subject with only a definition-shaped template match
  must return `decline`; `gravity→earth`-style unstated relations must not be
  written.
- Confirm `assertable_chain` still rejects `co_occurrence` and
  `schema_hypothesis` hops; confirm no `eng.save()` was called.

---

## Files consulted (read-only)

`docs/BRAIN_REASONING_DESIGN.md`, `docs/BRAIN_CAUSAL_SUBSTRATE.md`,
`docs/PLAN_ROUTE_REPLICATION.md`, `docs/GAP_DRIVEN_WEB_LEARNING.md`,
`docs/AUTHORED_CAUSAL_SKELETON.md`, `docs/WEB_NODE_ADMISSION_BLOCKED.md`,
`docs/WEB_LEARNING_CHAIN_VERIFIED.md`, `docs/DEFINITION_PROPOSITION_RECALL.md`;
`ravana/src/ravana/web/{learner,openie,web_to_graph}.py`;
`ravana/src/ravana/chat/{engine,engine_graph,engine_reasoning,engine_self_query,
web_learning,junk_scorer,response_gen}.py`;
`ravana/src/ravana/learn/{explanation,definition_proposition,linearization,intake,
committed_proposition}.py`; `ravana/src/ravana/core/causal_schema.py`;
`scripts/measure_fronts.py`; `weights/ravana_weights.db` (read-only SELECT).
