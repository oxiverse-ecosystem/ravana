# M3 — negation-aware graded-probe reader for SUPPLIED evidence

**Status:** SCRATCH + DEFAULT-OFF. No engine was booted; no store, gate,
threshold, item, prompt, gold answer, rubric, scoring function or production
file was changed. Worktree
`C:/Users/Likhith/Documents/projects/ravana_m3_reader`, branch
`feat/trak-m3-graded-probe-reader`, base `16c7738` (parent's checkout, which
contains `main` as an ancestor).

This is the M3 item of the shared root-cause map
(`UNIFIED_BRAIN_ROOT_CAUSE_MAP.md` §3): M1 is a provenance-bearing identity-keyed
act record, M2 is subject-side schema evidence, and M3 is the reader that must
fire. The map explicitly says M1 without M2 and M3 is "a store nobody reads".

## 1. Why the previous reader failed, and what changed

`TRA_K_READ_SURFACE_NEGATIVE.md` measured the earlier hook flat in all four
arms (`OFF = OFF2 = READ = SHAM = 0.500/0.500/0.000`, retention `0.000`). That
hook scanned the **live graph** for a redundant `type_label` stamp. When the
stamp was absent at the first graded read it failed closed, emitted a hedge,
and `immediate(k=8)=0` by construction. Its own design doc pre-registered the
fork: if READ and SHAM are both flat, the fact was **absent at that read** and
"no read hook can help". That branch was observed.

This reader does not re-scan the graph and does not re-tune the cue. **The
caller supplies the evidence.** It accepts only two evidence classes:

| class | what it is | admission gate |
|---|---|---|
| **M1** | identity-keyed, provenance-bearing prior-act record (`record_id`, `claim`, `prior_decision`, provenance) | unmodified `source_evaluation.evaluate_source(...) == ENDORSE` **and** unmodified provenance trust predicate |
| **M2** | subject-side typed-slot evidence binding the **subject** to a source-native type label | unmodified `schema_induction._is_trusted_edge` provenance predicate |

A familiar answer, a high edge confidence, an item-bank domain, `type_sets`,
`world_model` or a gold key cannot admit or veto anything: none of them is in
the reader's input signature.

## 2. Reader contract

Module: `scratch/m3_graded_probe.py`. Public surface:
`EvidenceBundle`, `ProbeDecision`, `read_graded_probe`, `graded_probe_pick`,
`wrong_tag_trust`, `new_counters`, `m3_enabled`.

### Inputs

```
EvidenceBundle:
  m1_records            tuple of provenance-bearing M1 act records
  m2_subject_evidence   tuple of subject-side typed-slot records

record keys (plain mappings; aliases in parentheses):
  record_id             stable identity key (audit only)
  subject               source-native subject, case preserved
  relation              is_a | type_of | instance_of (alias: relation_type, predicate)
  label                 SOURCE-NATIVE target label (alias: target, object)
  provenance            {edge_kind, source, source_type, source_id}
  confidence            stored edge confidence (graded signal)
  negated               explicit negation flag (optional)
  surface               optional teaching surface, scanned for explicit negation
  stamp / type_label    optional redundant stamp; must agree with label
  M1 only: source_status, prior_decision, invalidated, cue_present, origin_id
```

Accepting plain mappings is deliberate: a harness can stage M1/M2 evidence
without importing the reader or booting an engine, and the no-oracle-field
property is directly testable.

### Admission and veto (fixed order)

1. Relation must be one of the three `is_a` variants mirrored from
   `tra_k_items.ISA_PREDICATES`. An **explicit** empty/null relation is refused.
2. Provenance must pass the caller's trust predicate. READ supplies the
   unmodified `_is_trusted_edge`; SHAM supplies `wrong_tag_trust` (same
   signature, same call count, admits nothing).
3. M1 additionally calls the unmodified `evaluate_source`; anything other than
   `ENDORSE` (unknown/discredited source, no explicit credibility token,
   invalidation, missing origin) is refused with the evaluator's own rule.
4. The retrieval cue is the record's **own source-native label**. A stored
   `stamp`/`type_label` is accepted only if it agrees; a contradiction refuses
   the record. Case is preserved in the audit and the exact displayed option
   string is returned on a pick.
5. An admitted record with `negated=True`, or with a teaching surface containing
   an explicit negated copula, adds `(subject, label)` to the **veto set**.
   Positive support for a vetoed label is retained in the audit but removed
   from the eligible set. An untrusted negated record cannot veto.
6. Ranking uses the **imported production constants** `CONGRUENCE_FLOOR=0.20`
   and `STRICT_MARGIN=0.10`; no new threshold. Ties and below-floor scores
   defer. Ranking is deterministic (`-score`, then label key).

### Output

`ProbeDecision(pick, reason, subject, options, scores, vetoed, admitted,
rejected, congruence_floor, strict_margin, margin, source)`. `pick` is the bare
displayed source-native option, or `None`. `reason` is one of
`no_admitted_subject_evidence`, `no_trusted_label_above_floor`,
`all_candidates_negation_vetoed`, `strict_margin_not_met`,
`admitted_graded_probe_winner`, `deferred_to_production`.

### Default-off and structural deferral

`RAVANA_M3_GRADED_PROBE=1` (or explicit `enabled=True`) is required. With the
flag unset, `graded_probe_pick` returns `None` before touching production or
evidence. When on, it calls the caller-supplied **unmodified production**
reader first; a production pick is returned unchanged with
`source='production'` and the graded path performs **zero** evidence scans.
This makes overwrite by the new path structurally impossible and preserves the
existing safe behaviour on every path where production already answers.

### Matched SHAM / call-count control

`READ` uses `_is_trusted_edge`; `SHAM` uses `wrong_tag_trust`. Both execute the
same production call, the same scorer call, and the same per-record trust call
and scan. The self-test and unit test assert equality of `production_calls`,
`scorer_calls`, `trust_calls`, `records_scanned`, `m1_scanned`, `m2_scanned`
between the arms, with READ picking and SHAM returning `None`. Any difference
is a compute confound and invalidates the contrast.

## 3. Brain basis (verified citations)

Full machine-readable ledger: `scratch/m3_citation_ledger.json`. Verification
was against publisher/index records this session; where only an abstract or
index record was available the strength is marked accordingly.

| source | finding | consequence here |
|---|---|---|
| Tulving & Thomson (1973), *Psychol Rev* 80(5):352–373, doi:10.1037/h0020071 | encoding specificity: encoding operations determine which cues are effective | cue = the encoded source-native node label, not a parallel stamp |
| Morris, Bransford & Franks (1977), *JVLVB* 16(5):519–533, doi:10.1016/S0022-5371(77)80016-9 | transfer-appropriate processing | read the surface that overlaps encoding |
| Bransford & Johnson (1972), *JVLVB* 11(6):717–726, doi:10.1016/S0022-5371(72)80006-9 | a schema supplied **before** encoding changes recall | schema is a cue, not just a write-time aid |
| Anderson & Pichert (1978), *JVLVB* 17(1):1–12 (no DOI confirmed; ERIC ED142974) | a schema shift **at recall** alone changes retrieval | M2 subject-side evidence is a legitimate read input independent of M1 |
| Tse et al. (2007), *Science* 316:76–82, doi:10.1126/science.1135935 | one-trial assimilation needs an existing schema | M2 is a separate necessary condition, not a renamed M1 |
| MacMillan, Peer & Garnham (2001), *Memory & Cognition* 29(7):960–967, doi:10.3758/BF03195758 | explicit negation reduces concept accessibility | admitted negation vetoes that label |
| Giora (2008), doi:10.1515/9783110198843.2.129 | negated concepts are **not** always suppressed | veto is scoped to an explicitly negated admitted pair, not a global word ban |
| Johnson, Hashtroudi & Lindsay (1993), *Psychol Bull* 114(1):3–28, doi:10.1037/0033-2909.114.1.3 | item and source memory dissociate | trust from provenance, never from confidence/familiarity |
| Levelt, Roelofs & Meyer (1999), *Behav Brain Sci* 22(1):1–38, doi:10.1017/S0140525X99001776 | commitment to a proposition before form encoding | structural deferral, no overwrite of an admitted production pick |

These are computational analogies to a symbolic read path. No claim is made
that the brain has typed-slot edges, a `negated` flag or a forced-choice grader.
The physical-context result (Godden & Baddeley 1975), used in earlier TRA-K
docs, remains an extrapolation and is **not** relied on here.

## 4. Verification actually run (no engine boot)

```
# boot-free self-test
cd <worktree> && <main-repo>/.venv/Scripts/python.exe scratch/m3_graded_probe_self_test.py
=> self_test=PASS, 37/37 checks, boot_free.no_engine_construction_in_reader=true,
   no_harness_import=true, production_constants={floor:0.20, margin:0.10}

# unit tests
PYTHONPATH=<wt>/scratch;<wt>/ravana/src;<wt>/ravana_ml/src \
  <main-repo>/.venv/Scripts/python.exe -m pytest tests/unit/test_m3_graded_probe.py -q
=> 31 passed in 5.92s

# global-patch-leak check (m3 + two unrelated modules in ONE process)
=> 51 passed, 1 failed: tests/unit/test_brain_faithful_fixes.py::
   test_grounding_clause_segregation. Re-running that module ALONE reproduces
   the same failure on the untouched base commit; it is pre-existing and not
   caused by the M3 test module.
```

Both runs resolved `m3_graded_probe` and `ravana.chat.schema_induction` from
the worktree paths (checked by `test_imported_from_this_worktree` and an
explicit `__file__` probe), not from the parent's editable install.

## 5. Pre-registered falsifiers for the parent live A/B

Do not relax a threshold, change a prompt, or regrade a row to rescue a
falsifier. The instrument controls in `measure_tra_k` stay byte-identical.

| # | Falsifier | Measurement | Action |
|---|---|---|---|
| F1 | Boot-free self-test or unit suite fails | 37/37 and 31/31 | `INSTRUMENT INVALID — no score`; fix probe before any engine |
| F2 | SHAM returns a pick, or READ/SHAM call counters differ | SHAM pick `None`; all six call/scan counters equal | Control leaked through compute; revert contrast |
| F3 | Explicit negation does not veto a higher-confidence positive | veto set contains label, pick `None` (self-test covers) | Negative-control leak; reader non-deployable |
| F4 | Untrusted negated record vetoes, or gold/type_sets/world_model moves a pick | 0 such events | Oracle/trust leak; discard the arm |
| F5 | A production-admitted pick is overwritten or the new scorer runs on deferral path | `source='production'`, `scorer_calls=0` | Structural deferral broken |
| F6 | Default-off flag does any work | `None` with all counters zero | Flag equivalence broken; block adoption |
| F7 | Live READ `immediate(k=8)` stays `0.000` on both pools while SHAM is flat | require `>= 0.50` both, retention `>= 0.80`, foil/negative rise `<= 0.05` | Evidence absent again: name the missing subject-side input; **STOP**, do not tune |
| F8 | Live READ rises but SHAM rises too, or `api_calls['process_turn']` differs | SHAM `0.000`; byte-identical turn counts | Compute/carryover confound; invalidate arm deltas |
| F9 | Primary moves, sealed does not | both pools together | Store carryover/selection, not capability |
| F10 | Any pick is absent from the displayed source-native option set | 0 such picks | Instrument leak; non-deployable |
| F11 (G6) | READ saturates at `1.00` | mutation controls must move | Mechanism evidence only; flag G6, no capability claim |

## 6. Parent integration contract (NOT wired here)

The parent runs the live A/B in its own worktree. The intended boundary shape,
using only existing hooks, is:

```python
# per engine instance, in the parent's scratch backend only
from ravana.chat.schema_induction import _is_trusted_edge, slots_enabled
import m3_graded_probe as m3

def hook(raw_input, subject=None):
    if not slots_enabled():                 # existing production gate, unchanged
        return original(raw_input, subject)
    q, opts = parse_forced_choice(raw_input)         # production parser
    subj = resolve_subject(q, subject)               # existing head-noun fallback
    ev = stage_m1_m2_evidence(subj, opts)            # PARENT supplies evidence
    d = m3.graded_probe_pick(
        subj, opts, ev,
        production_pick=lambda s, o: original(raw_input, s),
        trust=_is_trusted_edge if arm == "READ" else m3.wrong_tag_trust,
        enabled=(arm in ("READ", "SHAM")),
        counters=rcounters)
    return None if d is None else d.pick
```

`stage_m1_m2_evidence` must derive M1 records from the engine's own prior-act
provenance and M2 subject-side evidence from the admitted channel facts, with
the same subject and the same source-native labels shown in `opts`. It must
never read the item bank's gold, `world_model`, `type_sets` or domain words.
The reader itself has no graph, engine, item or store dependency; if the
parent's staging is empty, the reader defers and the comparison is uninformative
rather than fabricated.

## 7. Honest boundaries

* This is a design + boot-free artifact. The live endpoint is **unmeasured**.
  The synthetic self-test graph is an oracle-shaped reachability check and is
  **non-deployable**; only a live READ gain with a flat SHAM would be evidence.
* Reusing `evaluate_source` for M1 means an M1 record with no explicit
  credibility token is refused even if it has strong stored confidence. That
  is a property of the unmodified gate, not a new threshold introduced here.
* The reader deliberately cannot repair an absent evidence channel. If F7
  fires again, the next work item remains a subject-side evidence input, not
  a fifth read-surface tweak.
