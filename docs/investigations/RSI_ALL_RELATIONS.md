# RSI: all relation families must be provenance-stamped at write time

Status: static + fake-test fix on `fix/rsi-all-relation-trust`. No engine was
booted and no benchmark file, gate, threshold or probe was changed.

## Evidence (observed, not inferred)

Parent's final focused live probe (`scratch/rsi_focus_live_rows.json`,
`RAVANA_SCHEMA=1`, natural-language demos, structural k=8): near = 0.50
(CAUSE `thaw->mist` correct, RESEMBLES `drizzle->chill` correct; PART_OF
`thunder->chill` and PRECEDES `cloud->glow` wrong), far = 0.00 across
CAUSE / PART_OF / PRECEDES / RESEMBLES, foil and negative all 0 (good).

Parent's census (`scratch/rsi_teach_form_probe2.log`): natural CAUSE demos
write 8/8 trusted edges; only the CAUSE family carries
`source_metadata.edge_kind` in `TRUSTED_EDGE_KINDS` with
`source='conversation'`. Other relation families are stored without a source
record, so schema induction cannot see them.

Code locations (line numbers at the fixed commit unless marked *pre-fix*):

* `scripts/measure_learn5_rsi.py:101-126` defines six families; this fix is
  scoped to the four named in the task (CAUSE, PART_OF, PRECEDES, RESEMBLES).
  `:542-556` writes each demo as a natural sentence (`"s phrase o."`) through
  `eng.process_turn`; `:612-628` builds the surface triples, with negative
  demos phrased `"not <phrase>"`.
* `ravana/src/ravana/chat/engine.py:10850-10862` (*pre-fix*) — the only
  declarative-consolidation call in a turn is
  `_consolidate_asserted_causal_facts(user_input)`, followed by
  `_extract_and_store_causal_relations(..., weak_associations=bool(consolidated))`.
* `ravana/src/ravana/chat/chain_walker.py:790-843` — `_ASSERTED_CAUSAL_RE`
  matches only causal verbs; `_asserted_causal_propositions` feeds only those
  pairs. `:878-930` — `_consolidate_asserted_causal_facts` is the **only** chat
  call site constructing `Provenance(source='conversation', source_type='user')`
  (`:912-916`) and calling `intake.intake` (`:916`).
* `ravana/src/ravana/chat/chain_walker.py:1273+` — `_create_causal_edge` writes
  edges via `graph.add_edge` with no `source_metadata`; it is the associative
  (untrusted-by-design) route reached by the non-causal demos.
* `ravana/src/ravana/chat/engine_generation.py:1373-1375` — the only other
  `intake.intake` call in chat is sleep replay with
  `source_type='retrieved'` -> `inferred`, not a conversational source.
* `ravana/src/ravana/learn/intake.py:79-85` maps `source_type='user'` to
  `edge_kind='curated'`; `:705-762` (`_write_edge`) persists that stamp;
  `:262-288` maps fact relations to graph relation types.
* `ravana/src/ravana/chat/schema_induction.py:68-72,96-115` defines the
  already-trusted kinds / `source='conversation'` predicate and reads both
  keys from `source_metadata`; `:316-351` (`traces_from_edges`) groups trusted
  edges by `relation_type`. `response_gen.py:196-218,7720-7732` adapts graph
  edges and induces/ranks hypotheses.

## Root cause

WRITE-side asymmetry, not a trust-gate or articulation defect. The turn
pipeline routes every conversational assertion through a CAUSE-only extractor
and a CAUSE-only provenanced intake write. PART_OF / PRECEDES / RESEMBLES
sentences therefore produce either an unprovenanced associative edge or no
edge. `schema_induction._is_trusted_edge` correctly refuses those edges, so
no hypothesis can form for those families and the near-transfer answer must
fall back to direct-edge/hedge behaviour.

INFERENCE (consistent with, but not directly measured by the parent's census):
some non-causal demos also land in the gate's contradiction/revise path
because `SchemaConsistencyGate.detect_contradiction` (`intake.py:507-547`)
excludes slot exclusivity only for causal relations. That path still writes a
provenanced edge, so it does not change the fix; it was not measured here.

## Change (narrowest write-side fix)

* `chain_walker.py:989-1029` — new `_asserted_relation_propositions`: parses
  declarative PART_OF / PRECEDES / RESEMBLES sentences (forward and inverse
  phrases), canonicalises role swaps, and rejects questions, hedges,
  negation/counterexamples, unknown phrases, junk and multiword arguments.
* `chain_walker.py:1031-1049` — new `_declarative_assertion_text` filters
  hedged / negated / interrogative clauses before the untouched legacy CAUSE
  extractor runs. This closes an observed junk write where
  "gale probably causes rainbow." produced `probably -> rainbow`; clean CAUSE
  sentences are passed through unchanged, so benchmark CAUSE behaviour is
  identical.
* `chain_walker.py:1052-1070` — new `_consolidate_asserted_relations`: calls
  the unchanged CAUSE path first, then the new path.
* `chain_walker.py:1072-1135` — new `_consolidate_asserted_non_causal_relations`:
  every accepted assertion goes through the same `intake.intake` with
  `Provenance(source='conversation', source_type='user',
  extractor='conversational-assertion')`, `salience=1.0`. It records the
  asserted family in provenance `operations` and, on the stamped edge, in
  `source_metadata['asserted_relation']`, then gives each family a distinct
  `relation_type` (`part_of`, `temporal`, `analogical`) so
  `traces_from_edges` cannot merge families into `semantic`. No trust
  predicate, confidence threshold, gate parameter, benchmark file or gold
  answer is touched. No item metadata is read. Nothing is written without an
  assertion; the CAUSE fact relation stays `causes` and its metadata stays
  byte-identical.
* `engine.py:10850-10868` — the turn calls the generalized hook, with an
  explicit fallback to the old CAUSE-only hook for duck-typed engines.

## Brain grounding

* **Source monitoring at encoding.** Johnson, Hashtroudi & Lindsay (1993),
  *Psychological Bulletin* 114(1):3-28, doi:10.1037/0033-2909.114.1.3 —
  source is an attribution computed from features of the record, and internally
  generated vs perceived information is distinguished at encoding. A
  conversational assertion must therefore carry its user-source features
  (`source`, `source_type`, `operations`) on the edge itself; provenance is
  not something the reader can reconstruct later. This is why the fix stamps
  at write time and does not relax the trust predicate.
* **Schema assimilation.** Tse, Bone & McClelland (2007), *Science*
  316(5828):1179-1183, doi:10.1126/science.1135935 — rapid learning of
  arbitrary new facts requires an existing schema and is gradual without one.
  Two or more converged, provenance-admitted demonstrations instantiate an
  *unassertable* hypothesis that can propose a candidate for an unseen subject;
  the stored demonstrations remain the only admitting evidence
  (`schema_induction.py:275-311`, `327-351`).

## Predictions / verification boundary

* P1 (this fix): with `RAVANA_SCHEMA=1` and natural demos, each of the four
  families now yields provenance-stamped edges; two converged demos per family
  yield one unassertable hypothesis per family. Verified by 28 fake-based
  tests in `tests/unit/test_rsi_all_relations.py` (no engine process): all
  four trusted edges, exact provenance keys, inverse role swap, rejection of
  unknown/hedged/negated/question forms, unchanged CAUSE fact and metadata,
  no-op with no assertion, per-family hypothesis at support 2 (not 1),
  direct-edge deferral, foil/negative rejection, flag-off inertness.
* P2 (NOT verified here): structural k=8 near should rise above 0.50 on the
  parent's four-item probe once PART_OF and PRECEDES edges are admitted.
* Separate gap (NOT addressed): far transfer may still need semantic transfer
  beyond provenance stamping. The far items use held-out domain vocabularies;
  a trusted hypothesis still requires an association signal from graph/GloVe
  data (`response_gen.schema_association_scores_from_graph`). A flat far
  score after this fix is a distinct work item, not evidence that the trust
  fix failed.
* No live engine was run in this worktree (parent is running PDR live); no
  live score, census or benchmark gate result is claimed here.

## Falsifier

If, after this change, trusted edges for PART_OF / PRECEDES / RESEMBLES are
still absent in a live census, dump each edge's `source_metadata` BEFORE and
AFTER the teaching turn, on the exact node ids resolved for the assertion,
and check whether the write route stamped `edge_kind='curated'` /
`source='conversation'` or whether a later consolidation overwrote it
(`intake._write_edge` calls `meta.update(...)` on existing edges). Do **not**
loosen `TRUSTED_EDGE_KINDS`, `schema_induction._is_trusted_edge`, the intake
gate thresholds, or the benchmark gates to make the metric move. If the stamp
is present but hypotheses still do not form, inspect the grouping key
(`relation_type` / `asserted_relation`) before touching the trust predicate.
