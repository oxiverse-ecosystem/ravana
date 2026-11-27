# TRA-K stage-2 live A/B — negative result (2026-11-27)

Stage-2 tested write-time typed-slot re-encoding with isolated stores:
`OFF`, `S2`, `SHAM`; `TRAK_MAX_PER_POOL=2`, one process per arm, identical
bank/channel/seed, `TRAK_AB_DATA_DIR` set separately per arm. Boot-free
self-test passed before the live run.

All arms ended `OVERALL: FAIL (genuine_fail)` with instrument controls OK.

| arm | primary transfer / near / far | sealed transfer / near / far | retention | growth | foil (primary/sealed) |
|---|---|---|---|---|---|
| OFF | 0.500 / 0.500 / 0.000 | 0.500 / 0.500 / 0.000 | 0.000 | +0.000 | 0.500 / 0.000 |
| S2 | 0.500 / 0.500 / 0.000 | 0.500 / 0.500 / 0.000 | 0.000 | +0.000 | 1.000 / 0.000 |
| SHAM | 0.500 / 0.500 / 0.000 | 0.500 / 0.500 / 0.000 | 0.000 | +0.000 | 1.000 / 0.000 |

## Verdict

- No retention or transfer gain from S2.
- The only endpoint movement (primary foil 0.500 → 1.000) is reproduced by the
  compute-matched sham, so it is attributable to the overlay procedure, not to
  correct source-backed re-encoding.
- Combined with the six-arm replay A/B (all `genuine_fail`, shams moved), the
  consolidation/replay family is ruled out for TRA-K's zero retention.
- **Do not implement** stage-2. No gate, threshold, scoring, item or prompt was
  changed.

Raw log: `scratch/trak_stage2_run.log`.
Per-arm artifact paths printed in that log (`scratch/trak_stage2_data/<arm>`).
