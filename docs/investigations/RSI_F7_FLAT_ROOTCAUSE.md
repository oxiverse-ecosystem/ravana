# RSI F7 flat result — root cause (M1+M2+M3)

**Verdict: READER_DEFERS, not a wiring or threshold failure.** The M3 hook fires
on every one of the 384 recorded RSI query turns, admits 1,360 provenance-
trusted M2 records, and emits **zero** picks. Every admitted record is a
*type* label (`"atmospheric condition"`, `"wind"`, …) from the independent
Wikidata snapshot; every RSI displayed option is a *relation-target object*
(`"sunset"`, `"typhoon"`, `"mist"`, `"hail"`, …). The reader's display-membership
filter therefore scores nothing and returns `no_admitted_subject_evidence`.
The forced-choice surface does not contain the label vocabulary the only
deployable evidence channel supplies.

M1 is a separate, earlier break on this front-end: its staging preconditions
are never satisfied by an RSI item, so `m1_act_calls = 0`, `staged_m1 = 0`
in every arm. M2 is the only live evidence channel and it is answering a
different question (type, not relation target).

## Evidence (boot-free, no engine boot, no store write)

Diagnostic: `scratch/rsi_f7_rootcause_probe.py`; raw output
`scratch/rsi_f7_rootcause_result.json`. It imports the **unmodified** staging
primitives (`scratch/m123_preflight.py`) and reader (`scratch/m3_graded_probe.py`),
replays the exact query prompt built by `scripts/measure_learn5_rsi.py:567-569`,
and reads the six recorded reports in `scratch/m123_out/rsi_*.json`.

### Four-way classification

| category | where | result |
|---|---|---|
| UNREACHABLE_WIRING | M3 hook in M123/SHAM | **ruled out**: `hook_calls=384`, `flag_on=384`, `production_calls=384`, `scorer_calls=384`, `hook_errors=0` |
| UNREACHABLE_WIRING | M1 staging on RSI | **confirmed**: `m1_act_calls=0`, `staged_m1=0`; RSI items have no `channel_evidence` and their teaching surfaces (`"gale causes rainbow."`) never match `pf.CLAIM_RE` (`X is a Y.`) |
| NO_SUBJECT_EVIDENCE | M2 snapshot | **mostly ruled out**: 468/480 full-bank query rows have admitted subject evidence; 12 have none |
| READER_DEFERS | M3 decision | **confirmed**: 1,360 records admitted in M123, `picks=0`, `emitted_picks=0`; replay reason `no_admitted_subject_evidence` on 473/480 rows |
| ARTICULATION/NO-ANSWER | grader | **not the blocker**: no hook emission exists to parse (`emitted_picks=0`); the 339 `engine_response_unparsed` rows are the production hedge path, unaffected by M3 |

### The decisive label-space census

Full bank (120 items × 4 conditions = 480 rows):

- 468 rows have subject-side type evidence admitted (Wikidata/CC0 snapshot).
- **7 rows** have any snapshot type label equal to a displayed option (and
  those 7 have no stored-confidence signal in the boot-free replay, so they
  fail `no_trusted_label_above_floor`).
- **473/480 rows** have no admitted label in the displayed option set.
- On the 4 executed items (16 query rows): 16/16 admitted, 0/16 label–option
  overlap, `no_admitted_subject_evidence` on 16/16.

Worked example (`rsi_000`, near, subject `thaw`): admitted evidence label
`"atmospheric condition"`; displayed options `sunset, typhoon, mist, hail`;
`read_graded_probe` returns `pick=None`, `reason=no_admitted_subject_evidence`.

### Reader is live (NON-DEPLOYABLE oracle control)

Replacing one scratch record's label with a displayed option (and supplying a
confidence, since no engine graph is available boot-free) makes the **same
unmodified reader** emit `pick="sunset"`, `reason=admitted_graded_probe_winner`.
This proves the hook, trust predicate, scoring and emission contract all work;
only the deployable label vocabulary is missing. That control injects answer-
side vocabulary and is **not** a fix or capability evidence.

### Negation / status admission ruled out

`veto_hits=0`, `records_rejected=0` in M123; `stage_arm_evidence` sets
`negated` only from `negated_pairs`, and `present()` stages `surfaces=[]` on
query turns (`m123_combined_runner.py:218`). Provenance admission succeeds for
all 1,360 records (`edge_kind=curated`, `source_type=external`, CC0). Neither
negation veto nor source-status rejection is suppressing a candidate.

## Why no narrow default-off fix is justified

RSI is an unseen-subject *relation-induction* benchmark: `build_items()`
(`scripts/measure_learn5_rsi.py:249-257, 369-371`) deliberately holds the query
subject and gold target out of the demonstrations. No deployable evidence for
`(query subject, relation, target)` exists in the snapshot, which supplies
only `is_a`/`instance_of` types, nor in the graph, which contains only the
*other* demonstrated pairs. Making M3 emit an RSI answer would require either
the item bank's gold key (contamination) or a new relation-induction mechanism
that forms a schema hypothesis from converged demonstrations and constrains it
to the displayed objects. That is a new capability, not a wiring repair, and
it is outside this task's no-item/no-scoring constraint. **Do not implement a
fix from this result.** The missing input is a deployable subject→relation-
target evidence channel for held-out subjects, which RSI's design intentionally
withholds.

## Brain-mechanism note (no new research needed)

The mechanism is already documented in the reader's own design
(`scratch/m3_graded_probe.py:30-58`): encoding specificity — Tulving & Thomson
(1973), *Psychological Review* 80(5):352-373, doi:10.1037/h0020071 — and
transfer-appropriate processing — Morris, Bransford & Franks (1977), *J. Verbal
Learning & Verbal Behavior* 16(5):519-533, doi:10.1016/S0022-5371(77)80016-9.
A cue encoded as a type cannot retrieve a relation-target proposition at test;
the mismatch observed here is exactly that prediction. Source monitoring
(Johnson, Hashtroudi & Lindsay 1993, *Psychol Bull* 114(1):3-28,
doi:10.1037/0033-2909.114.1.3) is why the correct response is to defer, not to
guess a displayed object. No additional literature search changes the diagnosis.

## Reproduce (boot-free)

```bash
cd C:/Users/Likhith/Documents/projects/ravana_rsi_f7_rootcause
PYTHONHASHSEED=0 RAVANA_OFFLINE=1 \
C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/rsi_f7_rootcause_probe.py
```

## Next live A/B command

Confirm the diagnosis under the shipped runner (it will reproduce F7 until a
deployable relation-target channel exists — do not read a flat rerun as a new
result):

```bash
cd C:/Users/Likhith/Documents/projects/ravana_m123_run
export PYTHONHASHSEED=0 RAVANA_OFFLINE=1
export TRAK_CHANNEL_JSON=C:/Users/Likhith/Documents/projects/ravana/scratch/trak_channel_from_substrate.json
PY=C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe
timeout 1800 "$PY" scratch/m123_combined_runner.py --stage rsi \
  --arm "M1+M2+M3" --tag m123_confirm \
  --base "$PWD/scratch/m123_data" --out "$PWD/scratch/m123_out/rsi_m123_confirm.json"
```

The decisive A/B after a deployable channel is built is the same invocation
with `M1+M2+M3` vs `SHAM` on fresh store copies, checking that near/far move
while foil/negative/unrelated stay at zero. Until then, no scoring change can
make this endpoint move.
