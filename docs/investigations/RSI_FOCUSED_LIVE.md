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

## Re-run after the query-turn fix (2026-11-23, PR #478)

`scratch/rsi_focus_queryturn.log`, `scratch/rsi_focus_live_result.json`:
1,824 `process_turn` calls (demos **plus the query**, up from 1,440), 1 engine
boot, no API error. Fake oracle unchanged at 1.00 everywhere.

| pool | k=1 near/far/foil/negative | k=8 near/far/foil/negative | mono drop | growth |
|---|---|---|---|---|
| primary | 0.0/0.0/0.5/1.0 | 0.0/0.0/0.0/1.0 | **1.0** | 0.0 |
| sealed | 0.0/0.0/0.0/0.5 | 0.0/0.0/1.0/0.0 | 0.5 | 0.0 |

`engine_response_unparsed` rose to 177/1,824 (9.7%) because the new query
turn contributes its own hedged emissions.

**Conclusion unchanged and strengthened:** the zero near/far transfer was not
an artifact of grading demo reactions. With a real question asked, the engine
still never applies the demonstrated relation to the query candidate. Foil and
negative-rule rejection became noisy rather than clean, which is what an
engine without a stored schema hypothesis looks like when actually tested.
Per `RSI_LIVE_RULE_INDUCTION.md`, the remaining fix is engine-side: write
declarative demonstrations as trusted provenance-stamped edges and aggregate
them into an unassertable schema hypothesis before answer generation.

## Final focused run with the trusted-edge hook (2026-11-23)

`RAVANA_EDGE_CHOICE=1`, 1,824 `process_turn` calls: **near/far still 0.00** on
both pools; foil ~0.5, negative 0–0.5, 180/1,824 unparsed. Fake oracle
unchanged. The direct `rsi_hook_probe` (which taught `mist→thaw` itself) did
emit `mist`, but that probe was **not representative**: it taught the answer
edge, which the real instrument never does.

Inspection of `scratch/rsi_focus_live_rows.json` shows the actual structural
demonstrations for `rsi_000` are `gale→rainbow`, `shower→climate`,
`overcast→storm`, `dew→tempest` — none involves the query subject `thaw` or the
gold target `mist`. RSI therefore requires **inducing a relation from several
exemplars and applying it to a new subject**, not retrieving a taught edge.
The trusted-edge hook can only answer when a demonstrated edge already lands
on the subject; it correctly fails closed otherwise.

**Final root cause:** RAVANA stores each demonstration as a separate trusted
episode (proven) but does not abstract a schema hypothesis that licenses
applying the demonstrated relation to an unseen subject. Brain grounding:
schema assimilation requires an existing schema (Tse 2007), and pattern
separation keeps episodes distinct until a slow regularity is extracted
(McClelland et al. 1995). The unassertable `schema_induction.py` helpers added
earlier are the right primitive but are not wired into the answer path.

No threshold was changed. The remaining fix is a genuine mechanism: converge
k demonstrated instances of the same relation into an unassertable schema
hypothesis, then use it to rank candidates for a new subject while keeping
foil/negative rejection. This is engine-side work; do not tune the benchmark.
