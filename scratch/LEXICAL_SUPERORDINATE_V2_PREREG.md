# H_struct v2 — pre-registered control correction (written BEFORE the v2 run)

## What v1 measured

`lexical_superordinate_experiment.py` v1 ran once with scoring constants
`DELTA=0.10`, `MARGIN=0.05`, `PERMUTE_OFFSET=137` frozen in source. Endpoints:
STRUCT 40/125 emits, 28/40 correct (0.700 precision); DEMO-WN 10/26 (0.385);
subject permutation 38/40 picks changed (0.95); gold permutation leaves picks
identical and drops accuracy to 0.100; primary 0.650 / sealed 0.750.

One pre-registered control FAILED: the fixed option tuple
`apple/banana/chair/table` produced 6 emits on graded rows (29 over all rows).

## Why this is an instrument defect, not a mechanism leak

The deployed channel is the WordNet 3.0 noun hypernym graph. In that graph the
"disjoint" tuple is NOT disjoint from several TRA subjects: `herb -> banana`
shares `herb.n.01` (depth 8), `vine -> banana` shares `vascular_plant.n.01`
(depth 7), `leader -> chair` shares `leader.n.01` (depth 7). A correct
lexical-semantic system is SUPPOSED to see those relations; rejecting them would
require suppressing genuine ontology evidence. `wn_disjoint_diagnostic.py`
prints all 29 cases with their lowest common subsumer and depth. No answer-key
field, domain tag or gold label enters the scorer (AST scan clean in v1).

## v2 change — controls ONLY, scoring and endpoints untouched

- Replace the invalid fixed tuple with four out-of-vocabulary tokens that have
  NO WordNet noun synset: `zzq_lex_a`, `zzq_lex_b`, `zzq_lex_c`, `zzq_lex_d`.
  Any emission there is a genuine fail-open defect.
- Retain the legacy tuple as a REPORTED DIAGNOSTIC (`legacy_disjoint_emits`),
  never as a positive criterion.
- Add order invariance: permuting the displayed candidates (offset 137) must
  leave the pick byte-identical.
- Keep P1–P6 and P8 exactly as v1; P7 now refers to the OOV tuple.

## v2 POSITIVE criterion (all required)

P1 AST clean, bank 0 errors, unmodified-input SHAs identical.
P2 STRUCT emits >= 20 / 125.
P3 STRUCT precision >= 0.60.
P4 STRUCT - DEMO-WN precision >= 0.15 on same cases.
P5 subject permutation changes >= 80% of emitted picks.
P6 gold permutation: all keys changed, picks identical, accuracy <= 0.40.
P7 OOV disjoint emits == 0; negated teaching emits == 0; unrelated subject
   emits == 0; candidate-order permutation changes 0 picks.
P8 |primary precision - sealed precision| <= 0.20.

If all hold, the mechanism is ELIGIBLE for a default-off implementation only.
It does NOT establish an RSI/TRA gate PASS: no engine was booted and no
production gate was measured.
