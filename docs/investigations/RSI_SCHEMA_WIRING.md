# RSI schema wiring — converge demonstrations, rank for an unseen subject (2026-11-23)

Status: **static code + fake unit tests only. No engine boot, no pytest by the
subagent — the parent runs both.** Files owned by this branch:
`ravana/src/ravana/chat/schema_induction.py` (helpers, from PR #482),
`ravana/src/ravana/chat/response_gen.py` (answer-path hook),
`tests/unit/test_schema_wiring.py`, this doc. `engine.py`, `models.py` and
`measure_learn5_rsi.py` were NOT edited.

## 0. Root cause this closes

`RSI_FOCUSED_LIVE.md` (#487) measured live near/far = **0.00** with foil
rejection intact. The demonstrations are stored as separate provenance-trusted
causal edges (`gale→rainbow`, `shower→climate`, `overcast→storm`, `dew→tempest`)
by `chain_walker.py:878-930` via `KnowledgeIntake.intake` (provenance
`source_type='user'` → `edge_kind='curated'`, `source='conversation'`), but the
query asks to apply the SAME relation to an UNSEEN subject (`thaw → mist`,
where `mist` was never taught as an edge into `thaw`). The existing H3 hook
(`RAVANA_EDGE_CHOICE=1`) can only retrieve a directly taught edge; it correctly
fail-closed. The missing mechanism is **abstraction of an unassertable schema
hypothesis from k converged demonstrations and its use to rank candidates for a
new subject**. This wiring adds that mechanism; it does not touch the harness.

## 1. Exact hook and control flow

Flag: **`RAVANA_SCHEMA=1`** (`schema_induction.SCHEMA_ENV_FLAG`,
`schema_enabled()`). Unset/anything-but-`1` → every new function returns
`None`/`[]` and the answer path is byte-identical. The earlier query-time
helpers keep their own flag `RAVANA_RSI_DECLARATION_STORAGE`; **that flag does
not enable this path** (a test asserts this).

1. `response_gen.py` imports the pure helpers from `.schema_induction` (no
   engine import, no state) and adds the `_schema_candidate` method plus two
   **module-level** read-only adapters (so duck-typed engines/fakes that expose
   only `.graph` / `._concept_keywords` work, and no sibling-method call can be
   swallowed by the hook's `except`):
   - `schema_trusted_edges_from_graph(graph)` — all provenance-trusted graph
     edges → `EdgeView`s, taking `edge_kind`/`source` from `source_metadata`.
   - `schema_association_scores_from_graph(graph, index, subject, candidates, pairs, glove_vector)` — cosine of stored node vectors (lazy GloVe
     fallback), supplied as DATA to the pure ranker. No threshold is read or
     lowered; a missing vector scores 0.
   - `_schema_candidate(raw_input, subject)` — the hook itself.
2. `_human_like_uncertainty()` calls `_schema_candidate` **immediately after**
   the existing `_trusted_edge_candidate` (H3) hook, via defensive
   `getattr(self, "_schema_candidate", None)` so duck-typed engines/fakes
   without it are unchanged. A returned string is emitted as the answer;
   `None` falls through to today's hedge. This is the hedge/reject surface,
   **after** CAR chain articulation has already had its chance — so chains and
   non-forced-choice turns are never diverted.
3. `_schema_candidate` order of operations (each step fail-closed):
   1. `schema_enabled()` false → `None`. No forced-choice frame / <2 options →
      `None`. Subject resolved from the question, with head-noun fallback,
      exactly as the H3 hook.
   2. **Directly taught edge decides → defer.** It builds incident
      `CandidateEdge`s and calls `select_trusted_candidate`; if that returns a
      candidate, `_schema_candidate` returns `None` and H3's answer stands.
      This is requirement (c): a hypothesis is never allowed to override a
      provenance-admitted observation.
   3. `traces_from_edges(self._schema_trusted_edges())` reuses the shared
      `_is_trusted_edge` provenance predicate. Each distinct trusted triple is
      one demonstration; duplicates do not inflate support.
   4. `induce_schema_hypotheses(traces)` — **only relations with >=2 distinct
      trusted traces** survive; every `SchemaHypothesis` is constructed with
      `assertable=False` and its `__post_init__` forces that flag off even if a
      caller asks for `True`. No edge/label is written or upgraded anywhere.
   5. `rank_candidates_for_unseen_subject(...)` keeps a hypothesis only if
      `admit_for_query()` finds an observed trusted trace of that relation
      ("schemas propose, sources admit"); a candidate must beat the congruence
      floor set by the weakest demonstrated instance and be a STRICT winner
      (ties → `None`). Scores come from the caller, so the ranker is pure.
4. The mechanism is **stateless**: all state is local to the call; a second
   turn on the same object with demonstrations removed yields `None`.

Requirement map: (a) `induce_schema_hypotheses` `min_support=2` over distinct
trace ids; (b) `SchemaHypothesis.assertable` is forced `False` and no write
path exists; (c) step 2 above; (d) hedge-surface placement + congruence floor +
strict margin leave foil/negative/CAR paths unchanged; flag off = step 1.

## 2. Brain map (verified citations)

| Mechanism implemented | Source (verified) | RAVANA analogue |
|---|---|---|
| Fast assimilation needs an EXISTING schema; one trial is not a schema | Tse et al. 2007, *Science* 316:76–82, doi:10.1126/science.1135935 | A single trusted edge licenses a candidate (H3) but is NOT promoted to a schema; >=2 converged edges are required |
| Fast episodic store + slow interleaved extraction of regularity | McClelland, McNaughton & O'Reilly 1995, *Psychol Rev* 102:419–457, doi:10.1037/0033-295X.102.3.419 | Each demonstration stays a separate trusted edge; the hypothesis is a derived, unassertable view over them (ANALOGY, not homology) |
| Pattern separation keeps episodes distinct | Yassa & Stark 2011, *Trends Neurosci* 34:515–525, doi:10.1016/j.tins.2011.06.006 | Distinct trace ids per triple; duplicate edges cannot inflate support |
| Relational mapping, not lexical overlap; relations map independent of content | Gentner 1983, *Cognitive Science* 7:155–170, doi:10.1207/s15516709cog0702_3 | Group by `relation_type`, not by shared words; a new subject can fill the slot |
| **Schema induction from concrete exemplars enables transfer to a new case** (additional mechanism) | **Gick & Holyoak 1983, *Cognitive Psychology* 15:1–38, doi:10.1016/0010-0285(83)90002-6** | `induce_schema_hypotheses` from k demos, then apply to the unseen query subject |
| **Mapping candidate roles by relational fit (constraint satisfaction)** (additional mechanism) | **Holyoak & Thagard 1989, *Cognitive Science* 13:295–355, doi:10.1207/s15516709cog1303_1** | `rank_candidates_for_unseen_subject` scores each option against the subject and requires a strict winner above the instance-derived floor |
| Schema congruence is also a source of FALSE memories | Webb, Turney & Dennis 2016, *Neuropsychologia* 93:61–75, doi:10.1016/j.neuropsychologia.2016.09.023 | Hypothesis is unassertable; a real observed trace must admit the relation; ties and below-floor candidates fail closed |

Citation verification: Gick & Holyoak 1983 verified via PsycNet record
1983-20819-001, UMich Deep Blue handle 2027.42/25331 and FORRT (DOI
10.1016/0010-0285(83)90002-6, pp. 1–38); Holyoak & Thagard 1989 verified via
Wiley `onlinelibrary.wiley.com/doi/10.1207/s15516709cog1303_1`, PsycNet record
1990-16295-001 and MIT OECS (DOI 10.1207/s15516709cog1303_1, pp. 295–355).
The five pre-existing citations were verified in `RSI_DECLARATION_STORAGE.md`.

### Support-vs-extrapolation ledger

| Claim used | Source | Verdict | Limit |
|---|---|---|---|
| >=2 converged demonstrations, not one, are needed before a schema may propose | Tse 2007; Gick & Holyoak 1983 | SUPPORT (conceptual) | Human behaviour; RAVANA's `min_support=2` is an operationalisation, not a measured human threshold |
| Keep demonstrations as distinct traces and derive a separate hypothesis | McClelland et al. 1995; Yassa & Stark 2011 | SUPPORT (analogy) | No hippocampal measurement in RAVANA; distinct trace ids are the whole claim |
| Rank candidates by relational fit to an unseen subject | Gentner 1983; Holyoak & Thagard 1989 | SUPPORT (framework) | Graph cosine is a weak stand-in for structural alignment; no claim of human-level analogy |
| The hypothesis must stay unassertable and be admitted by observed traces | Webb et al. 2016 | SUPPORT | Human false-memory paradigm; structural mapping to the gate only |
| Node-vector cosine as the proposal score | — | EXTRAPOLATION | Associative similarity is not relational structure; the A/B decides whether it is sufficient |
| Trusted-convergence floor from demonstrated pairs | — | EXTRAPOLATION | A design choice to fail closed, not a brain claim |
| Near/far failure was graded from the engine | RSI_FOCUSED_LIVE.md | SUPPORT (measured) | n=2 items/pool, focused mechanism probe; `engine_graded=True`, no API error |

## 3. Predictions (pre-registered; parent to measure, identical state, one process)

With `RAVANA_SCHEMA=1`, flag OFF byte-identical on the same store:

- **near/far > 0 at k=8** and **growth_headroom > 0** (k=8 minus k=1 positive —
  not a flat 1.00, not saturation).
- **foil/negative retained** at their current live rates: no candidate emitted
  for a foil subject or a negated demonstration, so the parser still records a
  rejection; negative metrics must not move down.
- The 1-demo and mismatched-relation conditions yield no hypothesis and hence
  no candidate (unit tests).
- A directly taught edge still wins: the hypothesis path defers (unit test).
- Fake oracle stays 1.00 with no `engine_answer` contribution; flag OFF output
  is byte-identical to today's (differential check before the A/B).

Honest scope caveat: under the current consolidation path
(`chain_walker._asserted_causal_propositions`) only CAUSE-family demonstrations
are written as trusted `causal` edges; other harness relations (PART_OF,
PRECEDES, SUPPORTS, OPPOSITE, RESEMBLES) currently produce no trusted demo
edges and therefore cannot form a hypothesis. The mechanism can only move the
CAUSE families until that write path is generalised — out of scope here
(`chain_walker.py` is not owned by this branch). Report the per-family split,
not only the aggregate.

## 4. Falsifiers (pre-registered — stop editing when they fire)

- **F0 — saturation:** if near/far is a flat **1.00** at every k with
  `growth_headroom == 0`, the front is saturated; a PASS is G6-flagged and is
  NOT capability evidence. Do not tune the probe to "make the number move".
- **F1 — articulation bottleneck:** if trusted demonstration edges are present
  (census), a hypothesis IS formed (`len(induce_schema_hypotheses) >= 1`), and
  the forced single-token answer surface still yields ~0 explicit decisions
  after echoed text is stripped, the gap is DOWNSTREAM of induction — in answer
  emission, not storage or hypothesis formation. **Stop editing** the read
  path and the hypothesis; the next work item is the decision surface.
- **F2 — source-admission leak:** if near/far improves on a store with no
  observed trusted edge (hypothesis alone), `admit_for_query` leaked; revert the
  schema branch (Webb et al. caution).
- **F3 — over-broad hook:** if foil/negative rejection or any non-forced-choice
  front moves, the congruence gate is too permissive; revert H4 rather than
  relaxing the harness.
- **F4 — direct-edge override:** if any case with a directly taught trusted
  edge emits a different candidate under `RAVANA_SCHEMA=1`, step 2 was bypassed;
  revert.

## 5. G6 caveat

A saturated or deterministic PASS does not demonstrate autonomous induction.
If both arms hit 1.00, or if a rule table hands the grader the right token by
construction, label the result **G6 FLAG — saturated/instrument-driven**, not
evidence the engine monitors its sources. Report honest deltas and the
per-family split; a small real gain with clean controls beats a flat 1.00 with
a weakened gate.

## 6. Unverified / left to the parent

- No engine boot and no pytest were run for this note (task constraint). The
  tests exist but must be run by the parent with `PYTHONPATH` pointed at this
  worktree.
- Whether graph node-vector cosine actually ranks `mist` above the other
  options for the live store is UNVERIFIED; the pure ranker is correct by
  construction, the score source is an extrapolation (ledger above).
- Whether trust-grouped hypotheses survive the live consolidation count at k=8
  (and thus whether near/far moves) is UNVERIFIED until the parent's A/B.
- The identity of the writer is inferred from an exact provenance-stamp match
  (`chain_walker.py:912-916` + `intake.py:80`), not a live call trace.
- No threshold, gate, `engine.py`, `models.py` or `measure_learn5_rsi.py` change
  was made.
