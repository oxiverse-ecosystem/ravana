# TRA experiment 2 — scratch trusted `is_a` write and fail-closed slot-keyed read

**Status: scratch experiment only. NO production code was edited.**
No file under `ravana/src/`, `ravana_ml/src/`, `scripts/` or `tests/` was
changed. Every engine import resolved at runtime to the **unmodified production
tree** `C:/Users/Likhith/Documents/projects/ravana` (main `dbed80d`):
`engine.py`, `schema_induction.py`, `chain_walker.py` and `openie.py` (the
`Fact` module) all resolved under that tree, asserted in the script. Four
independent **copies** of the weight store were used under
`C:/Users/Likhith/AppData/Local/hermes/cache/scratch/tra_slot_data/{type_arm,relation_arm,untrusted_arm,margin_arm}`.
Production `weights/ravana_weights.pkl` and `.db` SHA-256 were `181f1c6a824ed62c…`
and `2ff105143c3ae3c1…` at copy time and were byte-identical after all runs —
production weights were read from, never written to.

This is the "next work item" from `TRA_WRITE_READ_EXPERIMENT.md:205-210`: the
bounded scratch prototype of both missing halves, **not** the 400-family TRA
benchmark and **not** a LEARN-5 component claim.

## What was run

```
cd C:/Users/Likhith/Documents/projects/ravana_slot_exp
RAVANA_OFFLINE=1 PYTHONHASHSEED=0 \
  C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/tra_slot_write_read_experiment.py > scratch/tra_slot_run.log 2>&1
```

`EXIT=0`, zero recorded errors. 5 real engine boots: type-arm teach 2.24 s,
type-arm cold reload 2.82 s, relation arm 2.82 s, untrusted arm 2.92 s, margin
arm 3.17 s. `eng.save()` wrote 94,949 KB
to the **scratch** type-arm store. 10 teaching assertions in the type arm, 10
in the relation-only arm. No item bank, `world_model`, `type_sets`, `domain`,
`far_domain`, `gold_slot` or grader hook was supplied to the engine or the
scratch scorer.

## Instrument validation before any number (run 1 retracted)

Run 1 reported `typed_slots=0`, all positive picks `None`, and 6/6 gates
failing. The raw log showed the real cause: the write helper called the census
helper `_meta(e)`, which returns `dict(edge.source_metadata)` — a **copy**.
Mutating that copy never stamped the live edge, even though `intake.intake`
had already written all six `is_a` edges (`route=novel_write`, `new trusted is_a
= 6`). The probe, not the system, was wrong in both directions: it reported a
write failure when the write had succeeded, and a read failure when the read
had no stamped edge to find. Fixed by introducing `_writable_meta(e)` (the live
dict) and using it in the write helper and the forged-edge control. Thresholds
were **not** changed (`congruence_floor=0.20`, `strict_margin=0.10`, chosen in
the script before the first run). Run-1 artifacts are retained as
`scratch/tra_slot_results_invalid_r1.json` / `tra_slot_run_invalid_r1.log` and
their numbers are **retracted, not interpreted**.

A second validation: the strict-margin and congruence-floor branches were
never exercised by the main arm (only one candidate ever cleared the floor).
A fourth, explicitly labelled **instrument** arm on a fresh scratch copy wrote
two trusted typed edges for `mist`, equalised their stored confidences to
0.50 (metadata-only mutation, M-A/M-B style), and confirmed
`pick=None, reason=strict_margin_not_met`; lowering both to 0.10 gave
`pick=None, reason=no_trusted_typed_slot_above_floor`. This validates the
scorer's fail-closed branches; it is a mechanism check, not a capability
result.

## EVIDENCE

### Write half

1. **The write goes through the unchanged production intake API.** For each
   natural assertion the scratch helper calls the same sequence as the
   production relation-family writer (`chain_walker.py:1091-1105`):
   `_ensure_assertion_node` → `Fact(subject, relation, obj, confidence=0.9,
   source_sentence=text)` → `Provenance(source='conversation',
   source_type='user', extractor='scratch-tra-typed-assertion',
   operations=('taught-in-conversation','asserted-relation:IS_A'))` →
   `intake.intake(fact, prov, salience=1.0)`. All six `is_a` facts returned
   `route='novel_write'`; the unchanged trust gate admitted every one. No
   threshold or trust predicate was widened.
2. **6 trusted typed slots were written, 0 before.** Type arm census:
   `is_a` edges **169 → 175 → 175** (baseline → in-process → cold reload);
   trusted edges **170 → 180 → 180**; edges with a dedicated `type_label`
   stamp **0 → 6 → 6**. The six edges are
   `mist→weather phenomenon`, `hail→weather phenomenon`,
   `drizzle→weather phenomenon`, `election→civic event`,
   `ceremony→civic event`, `contract→civic event`.
3. **Provenance is real (C12).** Every one of the 6 typed edges carries
   `edge_kind='curated'`, `edge_source='conversation'`, `source_type='user'`,
   `extractor='scratch-tra-typed-assertion'`, `url=''`, and passes the
   unchanged `si._is_trusted_edge` predicate. Confidences on cold reload:
   `mist 0.3889`, `hail 0.4097`, `drizzle 0.4298`, `election 0.3889`,
   `ceremony 0.3638`, `contract 0.3769` — all above the 0.20 floor.
4. **The distinct relation type and slot labels are stored.**
   `relation_type='is_a'` for typed/category facts (distinct from the
   relation families `causal`/`part_of`/`temporal`/`analogical`), with
   `type_label=<category>` and `slot_label='type'`; part-of assertions wrote
   `slot_label='whole'`, `type_label='structured whole'`. (Experiment 1 found
   zero slot labels before teaching; this run measures 6 after.)
5. **Cold reload retains all of them (C9).** 6/6 typed triples survived
   `save()` + cold reload by exact `(subject, type_label, target)` identity;
   retention **1.00**, lost `[]`. Total edge count held at 18,952 across the
   reload.
6. **Control (a): relation assertions do not create typed slots.** A separate
   fresh store taught 10 relation assertions (`causes`/`part_of`/`precedes`/
   `resembles`) through `process_turn` only: **10** new provenance-trusted
   edges were written, **0** new typed slots. Relation edges and typed slots
   are therefore separable by the scratch write path.

### Read half

7. **The scratch scorer is slot-keyed, not cosine-keyed.** It reads only
   `eng.graph` edges and `source_metadata` on the cold-reloaded engine: an
   option is eligible only if a trusted `is_a` edge with a dedicated
   `type_label` goes from the subject to that exact slot, and the score is the
   **stored edge confidence** (an intake write signal). No GloVe vector,
   `world_model`, `type_sets`, gold slot or item metadata is read anywhere in
   `slot_keyed_forced_choice`.
8. **Positive picks and permutation stability.** For `mist` with options
   `[weather phenomenon, civic event, structured whole, similar form]` the
   scores were `{'weather phenomenon': 0.3889}` → pick **`weather phenomenon`**
   (margin 0.3889). For `election` → **`civic event`** (0.3889). Two
   permutations of the four options both returned `weather phenomenon`; all
   four recorded picks (`positive_mist`, `permutation_A`, `permutation_B`,
   `no_cross_turn_state_repeat`) are byte-identical.
9. **C10 disjoint control fails closed.** For `mist` with disjoint options
   `[apple, banana, chair, table]`: scores `{}`, eligible `[]`, pick **`None`**,
   reason `no_trusted_typed_slot_above_floor`. The unrelated subject
   `zeppelin` with the typed options also returned **`None`**. A subject with
   only relation teaching (`wheel`, `part_of`) returned **`None`** — relation
   edges cannot seed a type slot. The same query repeated after four
   intervening queries produced an identical output object (no cross-turn
   state, control (f)).
10. **Control (d): untrusted typed edge cannot seed.** On a fresh store a
    forged `zeppelin→is_a→weather phenomenon` edge was written through the
    same intake with `source='inference'`, `source_type='inference'`; it
    received `edge_kind='inferred'` and
    `si._is_trusted_edge(edge) = False`. The scorer for `zeppelin` returned
    **`None`**, while a genuinely trusted control edge for `hail` in the same
    store returned `weather phenomenon` — the scorer is alive and the trust
    filter, not an empty graph, caused the abstention.
11. **Production contrast on the same cold engine.** With the supported
    opt-in `RAVANA_SCHEMA=1`, the disjoint query for `mist` emitted **`banana`**
    (`C10_fail_closed=False`), reproducing Experiment 1's finding. The
    scratch scorer emits `None` for the identical option set. The two paths
    differ only in where the candidate signal comes from.

## INFERENCE (not established by the above)

- **I1** — The 2167→1310 node drop during the 10 teaching turns recurred; it
  did not delete any of the 6 trusted typed edges. Cause not isolated; same
  observation as Experiment 1 §I1, not a diagnosis.
- **I2** — The scratch helper's `is_a` *parser* and the `type_label`/`slot_label`
  post-write stamp are local adapters. `intake.intake` needed no edit, but
  promoting this to default behaviour would require production edits (an `is_a`
  phrase in `chain_walker.py:946-955` and write-time slot stamping). This
  experiment did not test those edits in place.
- **I3** — Whether the scratch scorer generalises to the TRA 6-slot enum,
  unseen near/far words, foil/negative items and k-sweeps is unmeasured. All
  options here were taught category labels; the positive margin had only one
  eligible candidate (the margin branch was validated separately, EVIDENCE
  instrument arm).
- **I4** — The 0.20 congruence floor and 0.10 strict margin are designer
  choices, not TRA-pre-registered thresholds. They were fixed before the run
  and never tuned after a result; no sensitivity sweep was performed.
- **I5** — The scorer is a standalone function, not wired into
  `process_turn`. It demonstrates a deployable *mechanism*, not an integrated
  answer path.

## Gate mapping (C9–C12 and the required controls)

| Gate / control | Measured | Verdict |
|---|---|---|
| **C9** cold-reload retention ≥ 0.80 | 6/6 = **1.00**; fresh engine read | **PASS** |
| **C10** type contrast: taught type separates candidates; disjoint ⇒ `None` | `mist→weather phenomenon` selected; disjoint ⇒ `None`; production flag-on ⇒ `banana` | **PASS in scratch; FAIL in shipped flag-on path** |
| **C11** anti-fake-backend, live API counters, fake-arm, query-only grading | real engine, real store, no fake oracle; but no `api_calls`/fake-arm harness | **NOT ASSESSED** — do not infer PASS |
| **C12** provenance integrity; no fabricated URL/span; gate unchanged | 6/6 edges `curated`/`conversation`/`user`, empty URL, unchanged `_is_trusted_edge` | **PASS** |
| Control (a) relation-only ⇒ no typed slots | 10 relation edges, 0 typed slots | PASS |
| Control (b) type assertions ⇒ typed slots | 6 typed slots | PASS |
| Control (c) disjoint ⇒ `None` | `None` | PASS |
| Control (d) untrusted typed edge cannot seed | `None`; forged edge trusted=False | PASS |
| Control (e) cold-reload retention | 6/6 survive | PASS |
| Control (f) no cross-turn state | repeated probe byte-identical | PASS |
| Instrument: margin / floor branches fire | tie ⇒ `None` (`strict_margin_not_met`); below floor ⇒ `None` | PASS |
| C1–C8, C13, M-A/M-B on taught items, 400-family bank | not run (bounded 2-arm probe) | NOT ASSESSED |

All 11 gates encoded in the script passed; `all_gates_pass=true`, `errors_count=0`.

## Recommendation

**IMPLEMENT — narrow, default-OFF, as a separate PR; DO NOT IMPLEMENT as a
default-on TRA capability or claim a TRA PASS.**

Why implement the two primitives: the two gaps named in Experiment 1 are now
closed on a real engine with clean controls. The `is_a` write succeeds through
the **unchanged** `intake.intake` with real `Provenance` (C12 PASS) and survives
cold reload (C9 PASS, 6/6); the scratch read uses stored slot edges instead of
GloVe cosine and fails closed on disjoint options (scratch C10 PASS), where the
shipped `RAVANA_SCHEMA=1` path still emits `banana`. No threshold, gate, item
or trust predicate was relaxed, and both control arms separate relation edges
from typed slots.

Why it must stay default-OFF and out of the LEARN-5 release rule for now:
C11 is not fully assessed (no fake-arm/API-counter harness), C1–C8/C13 and the
400-family k-sweep are unrun, n is small (one query subject per slot, 10
teaching turns), and the parser plus slot stamp are scratch adapters whose
in-place production form has not been measured. Promote only after the full
TRA harness reports those gates with the disjoint control wired in as a hard
instrument gate. If either the write-time stamp or the fail-closed read cannot
be reproduced in the integrated path, revert rather than lower a floor.

**Do not** cite this run as TRA PASS, LEARN-5 acquisition, or evidence the
engine already has typed slots by default.

## Honest caveats

- n = 2 positive subjects (`mist`, `election`), 3 three-way convergences, 8
  read probes, 6 boots. Mechanism probe, not a verdict; no Wilson CI or k-curve.
- All typed slots are category strings taught in this run. No unseen near/far
  gold word, foil or negated demo was tested (TRA P2/P3/C2/C5 unassessed).
- The copied store is one `weights/` snapshot; a different history changes the
  baseline is_a census (169 here).
- The margin check mutates stored confidences on a scratch copy; it validates
  the scorer's control flow only.
- Run-1 numbers are retracted (instrument defect above), not averaged in.

## Files

- `scratch/tra_slot_write_read_experiment.py` — scratch write + read + controls
- `scratch/tra_slot_results.json` — full per-stage censuses, probes, gates
- `scratch/tra_slot_run.log` — captured stdout/stderr with `EXIT=0`
- `scratch/tra_slot_results_invalid_r1.json`, `scratch/tra_slot_run_invalid_r1.log`
  — retracted run-1 instrument-defect artifacts, retained for audit only
- this document
