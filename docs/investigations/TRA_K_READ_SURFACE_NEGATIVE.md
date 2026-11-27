# TRA-K read-surface hook live A/B — negative result (2026-11-27)

Arms: `OFF`, `OFF2` (independent repeated OFF baseline), `READ`, `SHAM`;
`TRAK_MAX_PER_POOL=2`, one process per arm, isolated `TRAK_AB_DATA_DIR`,
identical bank/channel/seed. Boot-free self-test passed 15/15 before the run.

All arms ended `OVERALL: FAIL (genuine_fail)` with instrument controls intact.

| arm | primary transfer / near / far | sealed transfer / near / far | retention | growth |
|---|---|---|---|---|
| OFF | 0.500 / 0.500 / 0.000 | 0.500 / 0.500 / 0.000 | 0.000 | +0.000 |
| OFF2 | 0.500 / 0.500 / 0.000 | 0.500 / 0.500 / 0.000 | 0.000 | +0.000 |
| READ | 0.500 / 0.500 / 0.000 | 0.500 / 0.500 / 0.000 | 0.000 | +0.000 |
| SHAM | 0.500 / 0.500 / 0.000 | 0.500 / 0.500 / 0.000 | 0.000 | +0.000 |

## Verdict

- F8 (OFF == OFF2) passes: isolated-store baselines are byte-identical on the
  reported endpoints, so arm deltas are attributable to the hook.
- READ produces no gain over OFF; SHAM is also flat. The read-surface hook does
  not fix TRA-K retention or transfer.
- Combined with the six-arm replay A/B and the stage-2 write-time re-encode
  A/B, the consolidation/replay/read-surface family is ruled out for this
  harness. The remaining deficit is upstream subject-side input or engine
  articulation, not stored-fact admission.
- **Do not implement** the read-surface hook. No gate, threshold, scoring,
  item, prompt, or TRA/RSI rule was changed.

Raw log: `scratch/trak_read_surface_run.log`.
