# Focused live-engine RSI result (2026-11-23)

Instrument: PR #470 rewired `RealBackend.present()` to grade parsed
`process_turn` output instead of returning the demo triples. Full 120-item live
sweep needs ~43,200 `process_turn` calls (~30 h), so this probe uses the SAME
backend and code path on a deterministic **2 items per pool** subset
(384 rows, 1,440 live `process_turn` calls, one engine boot, no backend error).

Artifacts: `scratch/rsi_focus_live.py`, `scratch/rsi_focus_live4.log`,
`scratch/rsi_focus_live_result.json`, `scratch/rsi_focus_fake_rows.json`,
`scratch/rsi_focus_live_rows.json`.

## Result (harness's own `summarize()`)

| arm | pool | k=1 near/far/foil/negative | k=8 near/far/foil/negative | monotonic drop | growth |
|---|---|---|---|---|---|
| fake | primary | 1.0/1.0/1.0/1.0 | 1.0/1.0/1.0/1.0 | 0.0 | 0.0 |
| fake | sealed | 1.0/1.0/1.0/1.0 | 1.0/1.0/1.0/1.0 | 0.0 | 0.0 |
| **live** | primary | **0.0/0.0**/1.0/0.5 | **0.0/0.0**/1.0/1.0 | **0.5** | 0.0 |
| **live** | sealed | **0.0/0.0**/1.0/0.0 | **0.0/0.0**/1.0/1.0 | 0.0 | 0.0 |

`engine_response_unparsed = 26 / 1440` (1.8%). No API error. The graded
observation came from the engine in the live arm (`engine_graded=True`).

## Conclusion

The live endpoints are **not saturated**: near/far transfer from conversational
demonstrations is 0.00 on both pools at every k, while foil rejection is
preserved (1.00) and negative-rule rejection improves from 0.5/0.0 at k=1 to
1.0 at k=8. The fake oracle's uniform 1.00 was an artifact, exactly as
`G6_FLAGS_ANALYSIS.md` predicted. This is a real capability finding: RAVANA can
reject foils and counterexamples but does not induce the taught relation from
the demonstrations it is shown.

## Caveats (do not overclaim)

- n=2 items per pool, not the full 120-item design; this is a mechanism probe,
  not a suite verdict.
- 26/1,440 live responses were unparsed and scored as no-answer; a parser that
  recovered some of them could raise near/far somewhat, but the gap from 1.00
  is far larger than 1.8%.
- No cold reload/save distinction applies to this instrument; the live arm
  boots one engine and grades its emissions.
- No thresholds, gates or item banks were changed. The full sweep remains
  impractically slow and was intentionally replaced by this focused probe.
