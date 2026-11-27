# WFA Phase 0 — runtime-blocked, not re-measured (2026-11-26)

The roadmap required the WFA full internal-source run to complete and its result
to be recorded. That did **not** happen after the 2026-11-26 fixes. This note
prevents an unfinished run from being read as a PASS or a new FAIL.

## Last complete WFA result (still authoritative)

Full internal-source run, before the 2026-11-26 engine performance fixes:
- primary credible/discredited/unknown `1.00/1.00/1.00`
- sealed `1.00/0.80/1.00`
- anti-tautology PASS; overall unsaturated FAIL (exit 1)
- sealed discredited `0.80 < 0.90`; PD/forced-choice gates failed
  (`R=0`, `pd_unparseable_rate=1.00`)

No threshold, item, scoring rule, or gate was changed.

## What was fixed and verified on 2026-11-26

- g05 in-place provenance overwrite confirmed; existing claim-ID anchor fix
  retained provenance (boot-free A/B).
- Prior-endorsement variable identified: four sealed discredited items had
  `invalidated` tags but no `prior_decision`; supplying `ENDORSE` flips
  ABSTAIN → WITHDRAW in a boot-free probe.
- Decision-routine scratch experiment: best candidate is the existing
  production evaluator (`evaluate_source → SourceDecision.token`); it fixes
  articulation but saturates at `R=1.00`, so it would still be G6-flagged.
- Engine performance:
  - PR #549 — indexed `_activate_from_input` label scan (13.596 s → 3.978 s
    forced-choice turn; activation 10.10 s → 0.115 s).
  - PR #550 — cached deterministic relation-vector templates with per-edge
    copies (2.353 s → 1.021 s turn; byte-identical graph state).
- Harness root cause: `WFA_IS_MAX_PER_POOL` and `WFA_IS_FRESH_ENGINE` applied
  only to the internal-source arm, which runs last; bounded runs died in the
  full 192-item PD arm.
- PR #552 added a default-off `WFA_BOUNDED_DIAGNOSTIC` that bounds gap+decoy
  before PD and stamps `full_bank_verdict=false` / `NO FULL-BANK VERDICT`.

## Why Phase 0 is still blocked

All bounded attempts ended with a hard timeout and no complete endpoint:

| Run | Settings | Result |
|---|---|---|
| E2 full limited | `WFA_IS_MAX_PER_POOL=20`, fresh engine | `WFA_E2_EXIT=124`, 46 boots, 5,364 lines |
| E2B focus | `WFA_IS_MAX_PER_POOL=6`, fresh engine | `WFA_E2B_EXIT=124`, 1 boot, 1,830 lines |
| E2C after activation fix | `WFA_IS_MAX_PER_POOL=6` | `WFA_E2C_EXIT=124`, 1 boot, 1,951 lines |
| E2D after both engine fixes | `WFA_IS_MAX_PER_POOL=6` | `WFA_E2D_EXIT=124`, 1 boot, 2,033 lines |
| E2E minimal | `WFA_IS_MAX_PER_POOL=2` | `WFA_E2E_EXIT=124`, 1 boot, 1,329 lines |
| Bounded diagnostic | `WFA_BOUNDED_DIAGNOSTIC=1`, `max=6` | `WFA_BOUNDED_DIAG_EXIT=124`, 8 boots, 385 lines |

The bounded diagnostic reached the PD header but did not complete before its
30-minute cap. No full internal-source endpoint was produced.

## Status

- **WFA verdict:** last complete run remains **FAIL**; no new verdict is claimed.
- **Phase 0:** **runtime-blocked**, not complete.
- **Not done:** no threshold/gate change, no inferred PASS, no fake-oracle
  capability claim. A future complete run must be scheduled with enough wall
  time or with a further harness-level profile; this document records the
  blocker honestly.


## UPDATE 2026-11-27 — Phase 0 completed

A complete full-bank run with the two-line regex-cache fix (no GloVe cache)
finished in 475.9 s with unchanged scoring: primary credible/discredited/unknown
0.985/0.923/1.000, sealed 0.955/0.627/1.000; anti-tautology PASS; overall FAIL.
Artifact: `scratch/full_B_reonly_summary.json`. Phase 0 is no longer blocked;
WFA remains FAIL. The historical bounded-run timeouts above are retained as
measurement evidence.


## UPDATE 2026-11-27 — prior-act trace full verification

A default-off `RAVANA_GLOVE_BOOT_MEMO` boot-payload cache was added. A full
272-item run with BOTH flags (`RAVANA_PRIOR_ACT_TRACE=1`,
`RAVANA_GLOVE_BOOT_MEMO=1`) completed on the GloVe-free store in 855.5 s
(exit 1, WFA still FAIL). Internal-source primary/sealed discredited
correctness is 1.000/1.000 (baseline 0.923/0.627); all 24 internal-source
rules pass. The GloVe-present run hit its 3600 s cap (exit 124) and was
reprofiled: per-edge HRR/FFT encoding, not GloVe boot, is the remaining
hotspot. See `WFA_GLOVE_BOOT_MEMO.md` for endpoints, controls and artifacts.
