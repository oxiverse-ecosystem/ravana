# Brain → RAVANA: real-time knowledge acquisition and template-free production

**Status:** research + implementation. The intake gate described in §7 is
implemented (`ravana/src/ravana/learn/intake.py`, wired into
`ravana/src/ravana/web/web_to_graph.py`) and tested (`tests/unit/test_intake.py`,
29 tests). No existing test regressed.
**Scope:** how the brain (a) produces language without templates and (b) learns
from external text in real time — and what to replicate in RAVANA.
**Sources:** cited inline; full mechanism reports in
`reports/CONTINUAL_LEARNING_RESEARCH.md` and the knowledge-acquisition report.

---

## 0. TL;DR

Two questions were asked. They have different answers, and conflating them is
what made the earlier pivot hard.

1. **How does the brain avoid templates?** It does not store sentences. It
   stores *lemmas with syntactic features*, *constructions* (form–meaning
   pairings learned from use), and a *learned transition prior*, then produces
   text by **incremental selection under competitive gating** — function words
   and inflections are *computed from the frame*, not retrieved. Generalisation
   emerges from combinatorial structure, not from a bigger table.
2. **How does the brain learn from the web in real time?** It does not store
   the page. It builds a **situation model** (propositions integrated with
   prior knowledge), and the write is **gated by schema-consistency**:
   schema-consistent facts merge in one trial; schema-inconsistent facts are
   buffered and interleaved during replay; contradictions are resolved by an
   entrenchment ordering. Retrieval practice — not re-reading — is what makes
   it durable.

RAVANA already had most of the *production* machinery and most of the
*plasticity* machinery. What it was missing was the **write gate** between
ingestion and the graph, plus the **retrieval-practice schedule**. Those are
now implemented.

---

## 1. The two failure modes, stated precisely

### 1.1 The template failure

The emission path sampled authored strings. The pivot doc
(`LANGUAGE_PRODUCTION_PIVOT.md`) diagnosed this exactly: *"each component
exists, is qualitatively good, and is downstream of a gate that is closed."*

The brain's answer is not "a better generator". It is that **there is no
sentence-level store to compete with**:

| Brain | RAVANA before | RAVANA now |
|---|---|---|
| Lemma (meaning + syntax, no form) | `VerbLexicon.select_verb` returns a **verb phrase string** | unchanged (pivot M2, still open) |
| Constructions learned from use | `GenerativeEmitter`, `SyntacticCellAssembly` | exists, not the sole emit path |
| Transition prior | `CerebellarNgram` | exists, bypassed on social turns |
| Competitive token selection | `BasalGangliaGate` selects **concepts** only | unchanged (pivot M11) |
| Function words computed from frame | `SurfaceRealizer` | exists, but still samples authored variant menus |
| **Gate that retires the pools** | `use_linggen = False` (`engine.py:1234`) | **still `False`** |

The pivot's promotion gate (decoder CE ≤ template CE on held-out turns) was
never instrumented, so it never opened. **That remains the first task.**

### 1.2 The web-learning failure

The audit found the ingestion path was not producing *knowledge*:

- **Two disjoint paths.** A synchronous answer path (`_web_direct_answer` →
  verbatim snippet → `BeliefStore`) and an asynchronous learning path
  (`learn_from_web` → co-occurrence edges + OpenIE triples → `ConceptGraph`).
  Neither wrote to the other's store.
- **Text, not propositions.** The primary answering artifact was a verbatim
  200-char string (`web_learning.py:1760`), and `OpenIE`'s
  `Fact.source_sentence` was populated and then discarded.
- **No integration.** The same fact from two pages became three unrelated
  memories (a string, a co-occurrence edge, a triple) with no shared identity.
- **No learning signal.** All weight changes were fixed `+0.05` bumps or
  frequency counts. `GraphEngine.update_edge_from_error` (`graph/engine.py:1002`)
  was dead code.
- **Provenance not durable.** `source_metadata` was dropped by
  `storage/db.py:save_edge`; `_definitions` had no source field at all.
- **Web facts bypassed consolidation.** `TripletInferenceOperator` was built
  with `episodic=None` (`engine.py:1597`), so `Consolidator.extract_schemas`
  never saw web-derived triples.

This is precisely the difference Kintsch draws between *memory for a text* and
*learning from a text*.

---

## 2. The brain mechanisms, with evidence

Grades: **[S]** strong/replicated, **[M]** moderate, **[W]** weak/inferential.

### 2.1 Representation: surface → textbase → situation model

**[S]** Readers construct three levels: surface (exact wording), textbase (the
propositions of the text), and situation model (the state of affairs,
elaborated with prior knowledge). Verbatim memory decays within seconds while
gist persists (Sachs 1967; Kintsch et al. 1990). Kintsch (1994) states the
consequence directly:

> *"Remembering a text means that one can reproduce it... Learning from a text
> implies that one is able to use the information... in other ways."*

**Design rule.** Store propositions, never text. Give verbatim a fast decay.
The knowledge store is the situation-model graph.

### 2.2 The write gate: schema-consistency

**[S]** Schema-consistent information is learned in one trial without
interference; schema-inconsistent information must be interleaved or it
catastrophically interferes. Tse et al. (2011, *Science* 333:891–895) showed
rats with an established schema learned new associations in a single trial and
became hippocampus-independent within ~24 h. McClelland (2013, *JEPG*
142(4):1190–1210) showed the converse by simulation: *"complementary learning
systems and interleaved learning are necessary... if that information is not
consistent with the structure of known information."*

**[S]** Kumaran, Hassabis & McClelland (2016, *TiCS* 20(7):512–534) soften the
dichotomy in the important direction:

> *"neocortical learning can be rapid for information that is consistent with
> known structure"*

**Design rule.** The gate is not "slow vs fast learner". It is **one trial if
the fact fills existing slots; buffer-and-replay if it does not.**

### 2.3 Two systems, and why a single fast learner fails

**[S]** CLS: the hippocampus learns specifics rapidly and sparsely; the
neocortex learns structure gradually and distributively; interleaved replay
reconciles them (McClelland, McNaughton & O'Reilly 1995; Kumaran et al. 2016).
Catastrophic interference is directly demonstrated in connectionist networks
(McCloskey & Cohen 1989): after A-B then A-C, A-B dropped to ~1% within 5
trials.

**Design rule.** Append-only / sparse-update structures for facts; reserve
dense parameter change for an offline consolidation phase. Never overwrite a
monolithic embedding table on every ingest.

### 2.4 Elaboration determines durability

**[S]** Craik & Lockhart (1972); Craik & Tulving (1975): trace persistence is a
positive function of the depth and *elaboration* of processing, not of time
spent. *"It is the qualitative nature of a word's encoding which determines
retention, not processing time as such."*

**Design rule.** Durability should be a computable function of how much
existing structure a new fact activated — not a fixed decay constant.

### 2.5 Retrieval practice is the learning event

**[S]** Karpicke & Roediger (2008, *Science* 319:966–968): after one successful
recall, repeated *study* produced no measurable learning a week later, while
repeated *testing* produced ~80% recall vs ~36%. Roediger & Karpicke (2006):
retrieval practice *"enhances long-term retention... by slowing the rate of
forgetting."* Spacing/interleaving: Cepeda et al. (2006, 2008) meta-analyses.

**[M]** Boundary condition (Soderstrom, Kerr & Bjork 2016): when spacing is
controlled within subjects, restudying *does* help. The correct rule is
*retrieval beats re-reading; re-reading plus retrieval beats either alone.*

**Design rule.** A system that ingests a page once and never revisits it is
replicating the **worst** condition in this literature. Schedule self-retrieval
at expanding intervals.

### 2.6 Contradiction: revision by entrenchment

**[S]** AGM belief revision (Alchourrón, Gärdenfors & Makinson 1985): revision
removes the minimum necessary to restore consistency, guided by an
entrenchment ordering. AGM is a rationality benchmark, not a neural model —
use it as the engineering specification.

**[S]** Retrieval-induced forgetting (Anderson, Bjork & Bjork 1994): retrieving
one item actively suppresses its competitors sharing a cue. **Design rule:**
lateral inhibition is a deduplication mechanism, not just a forgetting one.

### 2.7 Updating: reconsolidation

**[M]** Nader, Schafe & LeDoux (2000, *Nature* 406:722–726): a reactivated
memory becomes labile and must be re-stabilised; the window is time-limited and
the update is gated by prediction error (Sevenster et al. 2013). Schiller et
al. (2010) showed human fear-memory updating inside the window, lasting ≥1 year.

**[M]** Contested in humans: Elsey, Van Ast & Kindt (2018, *Psych Bull*
144(8):797–848) — *"we cannot conclusively infer the presence of
reconsolidation at the neurobiological level from current evidence."*

**Design rule.** Treat the labile window as a design heuristic: updates are
bounded and localised, and the old value is kept as a superseded record rather
than deleted (so "you said X before" stays answerable).

### 2.8 Provenance is computed, not stored

**[S]** Johnson, Hashtroudi & Lindsay (1993, *Psych Bull* 114(1):3–28):
*"people do not typically directly retrieve an abstract tag or label that
specifies a memory's source, rather, activated memory records are evaluated and
attributed to particular sources through decision processes performed during
remembering."* Reality monitoring (Johnson & Raye 1981) distinguishes
perceived from generated records on the same feature dimensions.

**Design rule.** Every node carries record features (source, modality,
operations); trust is computed at query time. Keep *inferred* distinct from
*read*.

### 2.9 Offline abstraction — and its evidential limit

**[M]** Lewis & Durrant (2011): overlapping replay strengthens shared elements,
building schematic representations. **[W]** The claim that sleep is *necessary*
for abstraction is contested: Lerner & Gluck (2019) review many null results.

**Design rule.** Implement overlap-based abstraction; do **not** make it
sleep-dependent. Running the same algorithm online captures most of the benefit
and avoids an unestablished dependency.

### 2.10 Neuromodulatory gating

**[S]** Dopamine RPE (Schultz, Dayan & Montague 1997); LC-NE arousal/salience
(Aston-Jones & Cohen 2005); acetylcholine encoding-vs-retrieval mode
(Hasselmo). **[S]** Metaplasticity: Hebbian plasticity alone is unstable and
requires fast compensatory processes (Abraham & Bear 1996).

**Design rule.** Plasticity is gated by a computable scalar —
`|prediction error| × novelty × relevance` — not applied unconditionally.

---

## 3. Mechanism → RAVANA mapping

| # | Brain mechanism | RAVANA before | RAVANA now |
|---|---|---|---|
| 1 | Situation model (propositions, not text) | verbatim snippet store | **intake extracts propositions; verbatim is never the store** |
| 2 | Schema-consistency write gate | none — unconditional write | **`SchemaConsistencyGate`** |
| 3 | Two systems (fast/slow) | `HippocampalBuffer` + `ConceptGraph` | existed; now **connected by the gate** |
| 4 | Elaboration → durability | count/frequency decay | **`retention_prior` = f(elaboration, consistency)** |
| 5 | Retrieval practice | reinforcement on retrieval only | **expanding-interval schedule** (`due_for_review`) |
| 6 | Contradiction revision (AGM) | `BeliefStore.reconcile` (conf × recency) | **entrenchment + demote-and-supersede** |
| 7 | Reconsolidation window | `contradict`/`supersede` in fact stores | **bounded in-place edge update + superseded record** |
| 8 | Lateral inhibition / dedup | exact-match dedup only | **inhibitory edges already in `ravana_ml.graph`; near-dup via label/vector** |
| 9 | Provenance computed trust | in-memory `source_metadata`, non-durable | **`Provenance` with computed trust, durable metadata on every edge** |
| 10 | Neuromodulatory gating | decoder only; `_dopamine_tone` constant | **`salience` gates `effective_write`** |
| 11 | Metaplasticity | absent | partially: `ConceptNode.stability` exists; `regulate()` still uncalled |
| 12 | Offline abstraction | sleep schema extraction (narrow) | unchanged |
| 13 | Competitive token selection | `BasalGangliaGate` selects concepts | unchanged (pivot M11) |
| 14 | Construction learning from use | `GenerativeEmitter`, `SyntacticCellAssembly` | existed |

---

## 4. What already existed (credit where due)

The audit was surprised by how much was present:

- **`ravana_ml.graph.ConceptGraph`** already implements surprise-driven
  Hebbian updates with an EWC penalty and a Beta posterior
  (`hebbian_update:2070`), prediction-error free energy with contradiction
  counting and plasticity escalation (`apply_prediction_error:1992`), inhibitory
  contradiction edges (`form_inhibitory_edges:2181`), topology-aware structural
  importance (`compute_edge_structural_importance:2294`), concept splitting
  (`should_split:2378`, `split_concept:2428`), homeostatic downscaling
  (`homeostatic_downscale:2524`), and a fast-active/slow-core two-timescale
  vector (`adjust_vector:2022`).
- **`HippocampalBuffer`** already does pattern completion, retrieval
  reinforcement and turn-gated decay.
- **`BeliefStore`** already detects contradictions and reconciles by
  confidence × recency.
- **`core/sleep.py`** is wired and does replay, pruning, and REM
  counterfactual recombination.
- **`WebToGraph`** already gates node admission (junk score + reactivation
  across distinct sources) and does XdG context protection.

**The gap was not missing machinery. It was that the machinery was wired into
`ravana_ml/nn/rlm.py` and not into the chat engine's ingestion path, and there
was no gate between ingestion and the graph.**

---

## 5. What was implemented

### 5.1 `ravana/src/ravana/learn/intake.py`

**`Provenance`** — source, modality, operations, corroboration; `trust()` is
computed at query time from source reliability + corroboration + recency, with
inferred records discounted (reality monitoring). Serialises to a durable
metadata dict.

**`SchemaConsistencyGate`** — pure function of (fact, graph, provenance,
salience):
- `consistency()` — does the fact fill slots that already exist? (subject
  known, object known, subject already has this relation type, endpoints
  already linked)
- `elaboration()` — how much existing structure does it touch? (endpoint
  degree, shared neighbourhood, bridging bonus)
- `entrenchment()` — support × source trust × corroboration (AGM)
- `challenger_entrenchment()` — extractor confidence × source trust × whether
  the value is already a connected concept
- `detect_contradiction()` — same (subject, predicate) slot, different value
- `route()` — `fast_merge` / `episodic_buffer` / `revise`, guarded by three
  independent checks (below)
- `retention_prior()` — elaboration-driven durability prior

**`KnowledgeIntake`** — applies verdicts:
- `fast_merge` → typed edge, bounded update, durable provenance
- `episodic_buffer` → `HippocampalBuffer` with a retention-weighted confidence;
  with no buffer attached, retained at **low** confidence rather than dropped
  (the brain-faithful hippocampal-damage fallback)
- `revise` → demote the incumbent, record `superseded_by`, write the challenger
- schedules retrieval practice at an expanding interval; `due_for_review()`
  returns what is due; `record_review(recalled=...)` expands on success and
  resets on failure
- `salience × (0.35 + 0.65·elaboration)` is the gated write magnitude

Three guards keep the revise route honest — each was added because a test or a
smoke run showed it was missing:
1. **Support floor.** A challenger below `revise_min_entrenchment` (0.35) may
   not overturn anything, however weak the incumbent. Without it a noisy
   extractor displaces a real value with a bad one.
2. **Source-monitoring guard.** A single uncorroborated page may not overturn an
   incumbent; revision needs a high-reliability source (user/seed) or
   independent corroboration (≥ 0.5).
3. **Extractor confidence feeds the challenger score.** A triple the extractor
   is unsure about is a weak challenger.

**Known limit.** The gate is only as good as the extractor. `openie.py` is
deliberately crude — `is_a` covers *is / are / means / refers / called / define*
and its confidence heuristic is 0.6 or 0.7 by object length — so it emits junk
objects ("essential life"). The gate will faithfully resolve conflicts among
junk. Improving extraction is a prerequisite for the gate to show its value.

### 5.2 Wiring

`WebToGraph.learn_text(..., salience=1.0)` now routes every extracted fact
through the gate when one is attached. XdG context protection is preserved
unchanged. With no gate attached, the legacy unconditional write still runs,
so nothing else changes.

### 5.3 Graph plasticity wiring — `ravana/src/ravana/learn/plasticity.py`

The audit's biggest surprise: `ravana_ml.graph.ConceptGraph` already implements
the brain's plasticity mechanisms, and implements them well —

| primitive | what it does |
|---|---|
| `hebbian_update` | surprise-driven Hebbian delta with an EWC penalty and a Beta posterior |
| `apply_prediction_error` | free-energy accumulation, contradiction counting, plasticity escalation |
| `form_inhibitory_edges` | lateral inhibition between competing targets |
| `compute_edge_structural_importance` | topology-aware importance (Synaptic-Intelligence analogue) |
| `should_split` / `split_concept` | polysemy resolution by splitting |
| `homeostatic_downscale` | SHY-style synaptic homeostasis |
| `consolidate_vectors` | fast active vector → slow core vector (two timescales) |
| `record_path` / `compress_paths` | successful inference chains → shortcut edges |
| `reconcile_contradictions` | contradiction hotspots → stability / free-energy decay |

— and **every one was reachable only from `ravana_ml/nn/rlm.py`**. None was
called from `ravana/src/ravana/chat/`. `consolidate_vectors`, `compress_paths`,
`record_path`, `should_split`, `split_concept` and `apply_prediction_error`
had **zero callers** anywhere in the chat engine.

`GraphPlasticity` is the wiring, and it adds no new learning rule:

- **Per turn**, in `_spread_and_collect`, the edges that actually carried
  signal are collected. Each is corrected by a bounded gradient step toward
  the observed endpoint similarity — *only if* its weight poorly predicted
  that similarity, and only up to a per-pass budget, scaled by VAD arousal.
  Predictable edges cost nothing (Friston 2010; Chang et al. 2006). This is
  the graph's own prediction-error rule reached from the live retrieval path
  for the first time.
- **At sleep**, `_sleep_consolidate` runs the structural pass: inhibitory
  edges, concept splitting (rate-limited to 2/cycle, capped at 1.5× the
  concept count — the same guards `rlm.py` uses), two-timescale vector
  consolidation, SHY-style homeostatic downscale, contradiction
  reconciliation, and path compression. It runs after the golden-fact
  retention measurement so it cannot confound that metric.

Two handoffs close loops that were previously open:

- **Structural importance → EWC.** `homeostatic_downscale` already computes
  topology-aware importance (degree centrality, usage, sampled-BFS bridges —
  the Synaptic-Intelligence analogue) and caches it. The wiring then pushes
  that score into `edge.fisher_importance`, which is what `hebbian_update`'s
  EWC penalty actually reads. Before the handoff the importance was computed
  and discarded.
- **Usage → compression.** Every traversed edge is recorded via `record_path`;
  at sleep, chains used ≥ `compress_min_usage` times become shortcut edges.
  `record_path` had zero callers before this.

Both paths are exception-guarded and add zero cost when there is nothing to
do.

### 5.5 Retrieval practice driven from the engine

The schedule and the review logic already existed; nothing advanced the clock,
so nothing ever came due. Now `process_turn` calls `_run_retrieval_practice()`
at the **top** of every turn — deliberately not at the end-of-turn hook, which
the ~55 early-return paths (meta, temporal recall, greetings, arithmetic, …)
never reach. When a fact comes due, `_recall_from_own_memory` attempts the
retrieval from the graph plus the hippocampal buffer — the source text is
deliberately **not** consulted, because a false positive would reinforce
nothing — and `record_review` expands the interval on success and resets it to
the shortest on failure.

The engine also constructs its `KnowledgeIntake` over the existing
`_EngineGraphShim` and its own hippocampal buffer, and hands it to
`WebToGraph`, so web ingestion is gated by schema-consistency in normal
operation. A host without the getter (`ChatInterface`) keeps the legacy
unconditional write unchanged.

Both halves are needed: §5.1 builds the schedule, this section makes it run.

### 5.4 Verification

`tests/unit/test_intake.py` — 33 tests against the **real** `ConceptGraph`:
consistency ordering, elaboration ordering and the bridge bonus, contradiction
detection and re-sighting, all three routes, incumbent demotion and
supersession records, salience gating, provenance trust and round-trip,
durable provenance on the edge, retrieval-practice scheduling and
expand/reset, state round-trip, buffer routing, and the `WebToGraph` wiring in
both modes.

`tests/unit/test_c_lite_web_to_graph.py`, `test_phase1_provenance.py`,
`test_triplet_inference.py`, `test_e_active_inference.py`,
`test_self_directed_learning.py` — 37 passed.
`tests/integration/test_triplet_inference_wiring.py` — passed.
`tests/integration/test_unified_semantic_loop.py` — 2 failures, **pre-existing**
(verified by stashing the change and re-running: identical failures).

`tests/unit/test_plasticity.py` — 20 tests: surprise gating (predictable edges
untouched, surprising edges corrected toward observation), salience gating, the
per-pass budget, `apply_prediction_error` reaching the graph, path recording →
shortcut compression, two-timescale vector consolidation, contradiction
reconciliation, inhibitory-edge formation, concept splitting and its budget,
SHY downscale wiring + structural-importance→EWC handoff, structural protection
of critical edges, the disabled flag, state round-trip, and the
`_spread_and_collect` wiring in both directions.

`tests/unit/test_ravana_cognitive_graph.py`, `test_self_reference_guard.py`,
`test_language_production_mapping.py`, `tests/integration/test_sleep_episodic_replay.py`,
`tests/test_same_turn_profile.py` — 79 passed, 1 skipped.

`tests/unit/test_retrieval_practice.py` — 14 tests: the engine builds and
caches the intake, `_recall_from_own_memory` returns true only for a real
relation-typed match (wrong object, wrong relation type, and unknown subject
all return false), the episodic-buffer fallback, the schedule clock advancing
on every `process_turn` including early-return greeting paths, due facts being
reviewed, success expanding / failure resetting the interval, the no-due case,
trace output, and `WebToGraph` receiving the engine's intake.

**Full-suite regression check.** The suite was run on a clean tree and on this
tree, with nothing else competing for the machine:

| tree | passed | failed | skipped |
|---|---|---|---|
| clean (`git stash -u`) | 2757 | 1 | 23 |
| this branch | 2810 | 1 | 23 |
| this branch, after §5.5 | 2823 | 2 | 23 |

The first delta is exactly +53 = the 33 intake tests + the 20 plasticity tests;
the second is +13 net (14 new retrieval-practice tests, one of which replaced a
counted case). Both single failures are different, load-dependent tests
(`test_scaling::test_large_graph_construction` on the clean tree,
`test_d3_affective_echo::test_engine_still_reasons_legit_causal` on this one);
both pass in isolation. **No regression.**

**One pre-existing flake worth knowing about.**
`test_d3_affective_echo.py::test_engine_does_not_echo_affective_clause` fails
intermittently on `main` with and without this branch. Verified: it fails
*identically* on a clean `main` checkout (same response, same assertion), and
three consecutive runs on identical code gave fail / fail / pass — the passing
run took 24 s against 9 s for the failing ones, i.e. it waited out a network
timeout. The reply that trips the assertion carries live IntentForge results
(`[web:intentforge] … "It makes my blood boil! (The language of anger)"`), and
`SearchEngine` consults `http://localhost:4000` first, so the outcome depends on
whether a local IntentForge is running and what it returns. This is an
environment dependency in the test, not a behaviour change; it predates this
branch.

> An earlier sweep reported 102 failures — that run was executing concurrently
> with other pytest processes (SQLite lock contention plus xdist
> `ModuleNotFoundError`s). Don't run this suite concurrently with other test
> processes on this repo.

---

## 6. What remains

Ranked by leverage:

1. **Instrument the promotion gate.** Per-route decoder CE vs template CE on
   held-out turns. Nothing can retire a pool safely without it, and
   `docs/LINGGEN_CE_COMPARISON.md` already names it as the blocker. Still the
   first task.
2. **Bridge the answer path and the learning path.** `_web_direct_answer`
   stores a snippet in `BeliefStore` and never calls `WebToGraph`; the learning
   path never reads `BeliefStore`. A fact answered from the web and a fact
   learned from the web are still different objects.
3. ~~**Wire the existing graph plasticity into the chat path.**~~ — **done**,
   see §5.3. Per-turn surprise-gated correction, the sleep structural pass,
   SHY homeostatic downscale and the structural-importance → EWC handoff all
   run now. The one primitive still not driven from the chat path is
   `adjust_vector` (the per-node fast→slow vector update); `consolidate_vectors`
   covers it at sleep, but nothing moves a single node's vector online yet.
5. **Make provenance durable.** `storage/db.py:save_edge`/`load_edge` have no
   `source_metadata` column; `_definitions` has no source field. Until this is
   fixed, provenance survives only because the whole graph is pickled.
6. ~~**Drive the retrieval-practice schedule from the engine.**~~ — **done**,
   see §5.5. `due_for_review()` is now called every turn and the review
   result feeds back into the interval.
7. **Metaplasticity.** `ConceptNode.stability` is incremented but never gates
   anything; `ravana_ml/graph.py:regulate()` and `plasticity_boost` are uncalled.
8. **Collapse the duplicate `_graph_fallback_response`** (`chain_walker.py:2011`,
   `interface.py:1327`) — the pivot risk table flags this exact hazard.

---

## 7. Doctrine

What counts as a template, per the project's own docs and tests:

- **Acceptable constant** — closed-class/structural vocabulary, schema markers,
  hyperparameters, adaptive exemplars, short state-selected lexicon tokens.
- **Template** — authored reply prose, per-topic/per-entity tables, Q→A dicts,
  full-sentence templates, Cartesian products of authored strings.

Deciding test (from `CAPABILITY_EVALUATIVE_PREDICATE_POLARITY.md` §4):
*can RAVANA change this by itself, through experience?*

The intake gate is the write-side enforcement of that test: **a fact that
cannot be reconciled with existing structure is not written as if it were
established — it is buffered until replay can integrate it.**
