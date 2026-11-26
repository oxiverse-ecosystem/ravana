# Phase 3 — raw ingestion vs explicit `is_a` writer on the provenance-stamped Wikidata channel

**Verdict: DO NOT implement a narrow compositional parser.** On this channel the
structured fields are already parsed: a parser-free writer through the existing
`Fact` + `Provenance` + `KnowledgeIntake.intake` path preserves 86/86 source
facts, all 18 provenance fields, and 86/86 typed-slot reads after a simulated
cold reload. Raw `process_turn`-style ingestion writes **0** typed `is_a` edges
and preserves **0** provenance records. The two failures worth keeping are (a)
two substrate self-loops that the production writer correctly refuses and (b)
any parser that lowercases a source-native label silently breaks provenance
fidelity (84/86 and 69/86), so the structured-field arm is the only clean one.

## Scope and authority

- Branch `experiment/phase3-parser-ingestion`, worktree
  `C:/Users/Likhith/Documents/projects/ravana_p3_parser`, base `ac4ec88`.
- STATIC + IN-MEMORY ONLY. No `CognitiveChatEngine` instance (instrumented
  `__init__` call count = 0), no weights store, no network, no engine boot,
  no production edit, no threshold or gate change. Production module SHA-256
  hashes are identical before and after the run.
- Sole data input: `scratch/trak_channel_from_substrate.json`, SHA-256
  `18dac950d0f415539a5d9aa51e9859dd97d0fa7e9c284501322b70004d6372d0`.
  No item bank, `type_sets`, `world_model`, gold target or answer key was
  imported, opened or supplied to any scorer. AST guard + `open()` audit +
  import check are in the results JSON.
- Production functions called **unchanged**:
  `ChainWalkerMixin._consolidate_asserted_relations` (the hook `engine.py:10879`
  calls on every turn), `ChainWalkerMixin._typed_slot_assertions`,
  `ravana.web.openie.Fact`, `ravana.learn.intake.Provenance`,
  `KnowledgeIntake.intake` / `_write_edge`, `SchemaConsistencyGate`,
  `schema_induction._is_trusted_edge`, `typed_slot_views_from_graph`,
  `slot_keyed_forced_choice`, and `ConceptGraph` state serialization via
  `pickle.dumps`/`loads` (`__getstate__`/`__setstate__`).
- Scratch-only: the harness (`GraphEngine` + bare `ChainWalkerMixin` with an
  index-sync shim for `_all_labels`), the channel-provenance post-write stamp,
  the default-OFF narrow compositional parser, and the census/scoring code.
  Nothing under `ravana/src`, `scripts` or `tests` was modified.

## Brain-grounded design decision

The writer keeps **source monitoring** separate from item memory: trust is
re-derived at write time from record features, not from label similarity, and a
Wikidata record keeps its own URL/license/QIDs/span through the gate
(Johnson, Hashtroudi & Lindsay 1993, *Psychological Bulletin* 114:3–28).
Schema-consistent facts are admitted in one trial by the existing
consistency gate (Tse et al. 2011, *Science* 333:891–895; McClelland 2013,
*JEP:General*), so no new trust predicate is needed. Incremental clause-wise
binding motivates parsing only where the input is genuinely unstructured text
(Pylkkänen 2019, *Science* 366:62–66; Fedorenko et al. 2016, *PNAS*
113:E6256–E6262) — **this channel is already structured**, which is the
measured reason a parser buys nothing here. These citations motivate the
design; the experiment does not validate a neural analogue.

## Population and pre-registered exclusions

| Property | Count |
|---|---:|
| Channel rows (400 items x 8 facts) | 3200 |
| Unique source-native facts (subject, object, source, type QID, property, sense) | 88 |
| Self-loops (`branch is a branch`, `joint is a joint`) reported but excluded | 2 |
| Deployable/assertable denominator | 86 |

The self-loop exclusion was registered in the script **before** any arm ran.
Production `_ensure_assertion_node` / `_write_edge` refuse `sid == oid`; the
facts are logged, not silently dropped.

## Measured results

Facts are counted as exact `(subject, is_a, object)` identities. Parse metrics
use the 86 assertable surfaces rendered as `"<subject> is a <object>."`.
"Provenance full" requires **all 18** channel fields verbatim, plus production
`source`/`url`/`extractor`/`edge_kind`, `type_label`, `slot_label`, and a
trusted edge. Cold reload is a production `ConceptGraph` pickle state
round-trip, then a fresh census — no disk store and no engine boot.

| Arm | Parse TP | FP | FN | Precision | Recall | Typed is_a written | Provenance full (write) | Provenance full (reload) | Graph-derived read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A raw `process_turn` hook | 0 | 0 | 86 | 0.0000 | 0.0000 | 0/86 | 0/86 | 0/86 | 0/86 |
| B explicit writer, production parser | 70 | 0 | 16 | 1.0000 | 0.8140 | 70/86 | 69/86 | 69/86 | 70/86 |
| C explicit writer, scratch compositional parser | 86 | 0 | 0 | 1.0000 | 1.0000 | 86/86 | 84/86 | 84/86 | 86/86 |
| D explicit writer, **structured fields, no parser** | 86 | 0 | 0 | 1.0000 | 1.0000 | 86/86 | **86/86** | **86/86** | **86/86** |
| E negative control: same writer, no provenance | 86 | 0 | 0 | 1.0000 | 1.0000 | 86/86 | **0/86** | **0/86** | 86/86 |

Exact `findings_sha256`: `47dffb9bd1d4f1a59b076b43bdb1485f9bb55880ba382be78c5f726836c80d8b`
(stable under `PYTHONHASHSEED=0` and `1`; run metadata excluded from the hash).

### Where each arm actually stops

- **Raw path:** the default turn hook has no `is_a` family. It wrote **0** of
  86 edges. A positive control on the same hook wrote 1 edge from
  `rain causes flood.`, so the zero is a missing `is_a` write path, not a dead
  instrument. The GloVe associative path (`_extract_and_store_causal_relations`)
  is untrusted by design and carries no source record; it was not run.
- **Production parser:** 16/86 misses are exactly objects longer than its
  anchored ≤2-token regex (`particular anatomical entity`, `response to
  external stimulus`, `planar generalized triangle`, …). All emitted facts
  were correct (0 FP).
- **Case fidelity:** both text parsers lowercase the object before writing.
  `reward → Wikimedia article covering multiple topics` and
  `vessel → Void being` therefore lose the source-native `type_label` and fail
  the provenance check (B: 1 loss, C: 2 losses; `missing=[]`, the production
  fields and trust predicate were intact — only case differed). The structured
  arm writes the channel string verbatim and loses none.
- **Write alone is not provenance:** arm E wrote the same 86 edges through the
  same gate, but with a default `Provenance()` it preserved 0/86 channel
  records. This is the mutation control that makes the metric non-saturated.
- **No reload loss:** every arm's write-time provenance count equals its
  post-reload count. The stamp is in `ConceptEdge.source_metadata`, which
  production `__getstate__`/`__setstate__` retain.
- **Read side:** production `slot_keyed_forced_choice` picked the correct type
  on every written, above-floor typed slot with four disjoint graph-derived
  options; the unprovenanced arm reads edges but has no channel provenance to
  audit. Withdrawing one `type_label` makes the reader fail closed (`None`).

## Instrument validation

Instrument self-test passed: a known-good fact wrote a trusted typed edge,
survived the reload, and outranked an unwritten distractor; a question
(`X is a Y?`) was refused by both the production parser and the production
writer. Mutation controls passed: forging one `channel_type_qid`, or replacing
`type_label`, flips the provenance verdict; removing `type_label` flips the
read verdict to `None`; the default typed-slot hook is OFF. All pre-registered
falsifiers are **false**: F1 raw-suffices, F2 writer-fails, F3 parser-needed,
F4 reload-loss, F5 contamination, F6 instrument. `EXIT=0` on seeds 0 and 1.
Every probe's input and both outputs are in
`scratch/phase3_is_a_ingestion_run.log` (57 lines of auditable decisions).

## EVIDENCE

1. The raw turn hook writes 0/86 `is_a` facts and 0/86 provenance records on
   this channel, while its positive causal control writes 1 edge.
2. The production parser emits 70/86 facts with precision 1.0 and loses 16
   wider-NP objects; the scratch compositional parser emits 86/86.
3. A parser-free writer using the channel's own structured fields emits,
   stamps, reloads and reads **86/86** facts with all 18 provenance fields.
4. Neither text parser preserves source-native case on 2/86 labels; the
   structured arm does.
5. No confirmed fact loses provenance across the production pickle round-trip
   in any arm.

## INFERENCE (not measured here)

1. That this survives a real cold boot of `CognitiveChatEngine` with the live
   weights store — only a graph pickle round-trip was exercised.
2. That any benchmark answer front improves — no harness or benchmark ran.
3. That the 88-fact channel generalises to prose, multi-clause teaching turns
   or other substrates — this channel contains only single-clause `is_a`.
4. That the cited brain mechanisms map one-to-one onto this writer.

## Recommendation

**Do not implement a narrow compositional parser from this experiment.** For
the provenance-stamped channel, take `subject`/`object` from the structured
record and call the existing `Fact` + `Provenance` + `KnowledgeIntake.intake`
path, stamping the record's own fields at write time. Keep any integration
default-OFF, prove flag equivalence, and filter self-loops before writing.
Preserve source-native label case; do not normalize a provenance label through
a parser. If unstructured prose ever becomes the input, re-run this A/B on that
input shape before building a parser.

## Reproduce

```bash
cd C:/Users/Likhith/Documents/projects/ravana_p3_parser
PYTHONHASHSEED=0 C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/phase3_is_a_ingestion_experiment.py > scratch/phase3_is_a_ingestion_run.log 2>&1
```

Artifacts: `scratch/phase3_is_a_ingestion_experiment.py`,
`scratch/phase3_is_a_ingestion_results.json`,
`scratch/phase3_is_a_ingestion_run.log`.
