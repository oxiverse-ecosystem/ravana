# WFA E3 — prior-act re-staging scratch harness (live run handed off)

**Branch:** `invest/wfa-e3-prior-restage` (worktree
`C:/Users/Likhith/Documents/projects/ravana_wfa_e3`, based on `main` @
`66480ab`). Design source: `WFA_PRIOR_ENDORSEMENT_NEXT_EXPERIMENT.md` §4, E3.

## What was built

`scratch/measure_learn5_wfa_e3.py` is a SCRATCH driver that imports the
**unmodified** production harness (`scripts/measure_learn5_wfa.py`) and patches
exactly one module global: `run_internal_source_condition`. The replacement is
a copy of the production runner with the E3 manipulation made explicit and
auditable:

- at the first (prior-commitment) turn, immediately after
  `_is_process_turn(engine, prompt)`, it captures the exact object
  `engine._last_source_evaluation.decision` (no `.value` lookup, no
  `Decision.ENDORSE` constant, no item metadata);
- it forwards **that object itself** as `prior_decision` into the second
  turn's `context_from_graph_provenance(...)` call;
- it records the object's `id()`, a SHA-256 of `pickle.dumps(obj, protocol=4)`,
  and a runtime `ctx.prior_decision is prior` check in additive rows at
  `scratch/wfa_e3_staging_audit.jsonl`.

`WFA_E3_PRIOR_RESTAGE=0` disables the capture for a baseline arm (default `1`).
Nothing else is touched: no production file, gate, item, threshold, prompt,
expected decision or grading rule is edited. The production summary is written
by the unmodified `w.main()`; E3 audit rows are additive and read by no gate.

Note: the production runner already performs this capture; the scratch copy
makes the object identity observable and the manipulation toggleable so E3 can
be A/B'd and audited in the live run.

## Exact run commands (from the worktree root)

Boot-free unit-style check (no engine boot, no store open):

```bash
cd C:/Users/Likhith/Documents/projects/ravana_wfa_e3
RAVANA_OFFLINE=1 PYTHONHASHSEED=0 uv run python scratch/measure_learn5_wfa_e3.py --self-check
```

Live E3 run (parent only — not run when preparing this branch):

```bash
cd C:/Users/Likhith/Documents/projects/ravana_wfa_e3
RAVANA_SOURCE_EVAL=1 RAVANA_OFFLINE=1 PYTHONHASHSEED=0 \
  uv run python scratch/measure_learn5_wfa_e3.py
```

Baseline A/B arm (no prior act re-staged):

```bash
cd C:/Users/Likhith/Documents/projects/ravana_wfa_e3
RAVANA_SOURCE_EVAL=1 RAVANA_OFFLINE=1 PYTHONHASHSEED=0 \
  WFA_E3_PRIOR_RESTAGE=0 uv run python scratch/measure_learn5_wfa_e3.py
```

Optional focused subset: add `WFA_IS_MAX_PER_POOL=<n>` (must still be >=20/pool
for any pre-registered verdict). Env flags of the production harness are
unchanged (`WFA_IS_FRESH_ENGINE`, `IS_FLAG`, etc.).

## Boot-free check result (actually executed)

`verdict: E3_SELF_CHECK_OK` — all 12 checks true:

- `A_object_from_engine_audit`, `A_byte_identical_forward` (unique sentinel:
  `id()` match, pickle bytes match, `ctx.prior_decision is sentinel`),
  `A_first_turn_had_no_prior`, `A_invalidation_fired`;
- `B_no_item_parameter`, `B_item_fields_do_not_change_staging` (an item dict
  subclass that raises on every key except `claim`, plus a fully mutated copy,
  both give the identical staged fingerprint);
- `C_reproduces_ABSTAIN` / `C_flips_to_WITHDRAW` / `C_only_prior_changed`
  (real `evaluate_source` on a real in-memory `ConceptGraph`, no boot);
- `D_flag_off_baseline`, `D_no_live_engine_boot`;
- `E_grading_fields_match_production` (for all three conditions, every
  production observation field is byte-identical between the E3 copy and the
  production runner given the same engine audit object).

Wiring was also verified boot-free: after `install()`,
`w.run_internal_source_condition is e3_run_internal_source_condition` and
`w.main()` calls that module global.

## Expected live outcomes (pre-registered, not asserted here)

- If cross-turn bookkeeping was the loss: sealed discredited rate rises from
  0.80 to near 1.00 with unchanged stored provenance fingerprint.
- If it stays 0.80 while E2's fresh-engine arm passes: the prior ENDORSE is
  lost inside engine state, not in the harness.
- Any G6-saturation caveat from `WFA_E3_...` must still be reported with the
  internal-source number; no threshold is to be tuned to move this arm.

The scratch file is force-added (`git add -f`) despite the repo's scratch
gitignore rule, so the branch is runnable from a fresh checkout; production
code is unchanged in the diff.
