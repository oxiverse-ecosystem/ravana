# Addendum — final live A/B outcome (R2 arrived after the stage-2 design)

Read-only from `ravana_trak_live_ab/scratch/trak_live_ab_all3.log` and
`scratch/trak_live_ab_data/report_*.json` / `summary_*.json`. No engine was
booted by this worktree. The stage-2 design in
`TRA_K_STAGE2_READ_SURFACE.md` was written before R2 finished and is
**triggered** by its result, not revised by it.

| arm | near / far (primary) | immediate | delayed | delayed_retention | controls | verdict | elapsed s |
|---|---|---|---|---|---|---|---|
| OFF | 0.000 / 0.000 | 0.000 | 0.000 | `null` | OK | genuine_fail | 474 |
| R1 | 0.333 / 0.333 | 0.000 | 0.000 | `null` | OK | genuine_fail | 414 |
| R1_CONF | 0.333 / 0.333 | 0.000 | 0.500 primary / 0.000 sealed | `null` | OK | genuine_fail | 718 |
| R1_WRONG | 0.333 / 0.333 | **0.500 primary** / 0.000 | 0.000 | **0.000 primary** | OK | genuine_fail | 843 |
| R2 | 0.333 / 0.333 | 0.000 | 0.000 | `null` | OK | genuine_fail | 668 |

All five arms: `harness_exit=1`, `delayed_option_audit.ok=true`,
`instrument_controls_ok=true`, `pass=false`, `interpretation=genuine_fail`.
`engine_response_unparsed` stays 83–95 of 240 rows.

## Reading the R2 failure

R2 = earned provenance restamp plus three matched confidence-boost cycles at
the save boundary. It did **not** move immediate or delayed (both 0.000), and
its unparsed count (95/240) is the worst of the five arms. Combined with the
ordering fact in the main doc — the manipulation is applied inside
`save_reload()`, after the immediate probe that forms the retention
denominator — this closes the replay-family explanation: no save-boundary
replay variant can create retention when the first graded read has already
failed. The stage-2 write-time re-encoding experiment stays the recommendation.

## Why R1_WRONG's immediate=0.5 is NOT a capability or causal result

R1_WRONG ran immediately after R1_CONF on the **same per-item stores**
(`TRAK_AB_DATA_DIR` was never set by the parent loop). It therefore inherited
R1_CONF's confidence=1.0 stamped edges and produced a positive immediate
endpoint, then its deliberately wrong `unknown/system` tag destroyed the
provenance at save and drove delayed to 0.000 (`delayed_retention=0.000`).
This is exactly the store-carryover confound flagged in the main doc, visible
in the instrument itself. Monotonically growing elapsed times
(474 → 414 → 718 → 843 → 668 s, with later arms paying for a larger
accumulated graph) corroborate it. Do not attribute the R1_CONF primary-only
delayed gain or the R1_WRONG immediate value to the manipulation until the
arms are re-run with one fresh store copy per arm (falsifier F9).

## Recommendation unchanged

1. Treat the whole replay family as a negative result for retention: missing
   replay is not the cause of zero retention; the delayed option-space defect
   is repaired; the live failure is a write-time/read-admission defect at the
   first graded read.
2. Run the stage-2 scratch A/B (OFF / S2 / SHAM) with `TRAK_AB_DATA_DIR` set
   per arm, exactly as in the main doc's command. Check F9 (byte-identical
   repeated baselines) before looking at S2.
3. Only if S2 raises immediate on both pools with SHAM flat and controls
   intact does a retention claim become measurable; otherwise report the
   negative and name the missing subject-specific type-evidence input.
