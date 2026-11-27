# TRA-K replay live A/B — negative result (2026-11-27)

Six arms, focused `TRAK_MAX_PER_POOL=2`, one process per arm, identical bank,
channel, seed and controls. Harness subset was corrected to pin `k=8` in each
pool because `mutation_control()` requires a primary `k=8` item; without that
fix the tiny subset crashed with `StopIteration`.

All arms produced `OVERALL: FAIL (genuine_fail)` and `instrument_controls_ok`.

| arm | primary delayed | sealed delayed | primary growth | sealed growth |
|---|---:|---:|---:|---:|
| OFF | 0.000 | 0.000 | 0.000 | 0.000 |
| R1 (restamp) | 0.000 | 0.000 | 0.000 | 0.000 |
| R1_CONF (confound sham) | 0.500 | 0.000 | 0.000 | 0.000 |
| R1_WRONG (wrong-tag sham) | 0.000 | 0.000 | 0.500 | 0.000 |
| R2 (restamp + reactivation) | 0.000 | 0.000 | 0.000 | 0.000 |
| R2_WRONG | 0.000 | 0.000 | 0.000 | 0.000 |

## Verdict

- **No replay benefit**: R1 and R2 delayed remain `0.000` in both pools.
- **Falsifiers triggered**: the confound and wrong-tag shams move delayed or
  growth (`0.500`), so any apparent gain would not be attributable to correct
  replay provenance.
- The static replay gain (OFF `0.30/0.25` → R2 `0.45/0.45`) did not transfer to
  the live reader.
- **Do not implement** the replay-journal flag. No gate, threshold, scoring,
  item, or TRA/RSI change was made.

Raw log: `scratch/trak_live_ab_all3.log`.
Summaries: `scratch/trak_live_ab_data/summary_<arm>.json`.
