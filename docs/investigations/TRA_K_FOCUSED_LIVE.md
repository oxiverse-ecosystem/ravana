# TRA-K focused live — genuine FAIL (2026-11-26)

Run: `TRAK_CHANNEL_JSON=scratch/trak_channel_from_substrate.json
TRAK_MAX_PER_POOL=20 PYTHONHASHSEED=0 RAVANA_OFFLINE=1
uv run python scripts/measure_tra_k.py --live`

- Commit: `0829813`
- Selection: 40 families (20 primary / 20 sealed), mechanism probe, not a full
  benchmark verdict.
- Elapsed: `4999.1 s`; exit `1`.
- Log: `scratch/tra_k_focused3.log`.

## Instrument status

`instrument_controls_ok`, `c11_anti_fake_ok`, `subject_independence_ok`,
`channel_independence_ok`, `backend_live`, `backend_error_free`,
`saturation_flag_recorded`, and `mutations_passed_if_saturated` all pass.
Saturation gate **PASS** (no front in the saturated band); M-A/M-B moved `True`.
This is a valid measurement, not an instrument failure.

## Endpoints

| Split | transfer | near | far | foil | negative | chance | perm | mono | graded | coverage | retention | growth | ostensive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | 0.625 | 0.450 | 0.300 | 0.350 | 0.100 | 0.300 | 0.350 | 0.500 | 0.500 | 1.000 | 0.000 | +0.000 | 0.500 |
| sealed | 0.500 | 0.500 | 0.350 | 0.300 | 0.350 | 0.375 | 0.350 | 0.500 | 0.750 | 1.000 | 0.000 | +0.000 | 0.375 |

`interpretation=genuine_fail`, `OVERALL: FAIL`.

## Interpretation

This is the first properly specified TRA-K run: the channel supplies the
queried subject's own Wikidata `is_a` fact with provenance and deterministic
conflicts, and the graded options are source-native type labels. The engine
shows real but insufficient signal (primary transfer `0.625`), but fails the
pre-registered near/far/foil/negative/monotonicity/graded/retention criteria.
Retention is `0.000` and growth `+0.000`.

No gate or threshold was changed. The full 400-family run is **not** warranted
after a focused FAIL; it remains conditional on a clean focused result.
