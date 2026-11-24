# PDR repaired live run — verified, with a saturation caveat

## Measured result (2026-11-25)

`PYTHONHASHSEED=0 RAVANA_OFFLINE=1 uv run python scripts/measure_learn5_pdr.py --live`
completed `PDR_LIVE_FIXED_EXIT=0`. Backend is genuinely live: `backend=live`,
`live=True`, `api_calls={intake:3198, save:984, cold_reload:984,
process_turn:984}`, `backend_error=None`, `engine_observation_missing=0`.

All 8 instrument controls pass; primary n=40 and sealed n=42 both show
`orig0=1.00 orig5=1.00 intr5=0.00 forged=1.00 (prov_ok 1.00)
sleeper_dormant=1.00 sleeper_rej=1.00 durable=1.00 src=1.00 sense=1.00`.
Verdict: `PASS PDR -- meets all criteria`.

## Saturation flag — resolved by mutation control, not by tuning

Every boundary endpoint is exactly 1.00/0.00, so the pre-registered F1
saturation flag fired (`pdr_focus_intr`, `pdr_forged_reject` non-informative).
F1 requires the M-A/M-B mutation control before treating this as capability
evidence. `scratch/pdr_mut/fresh_probe.py` teaches one item into a fresh
isolated real-engine store, saves, cold-reloads, then mutates the REAL stored
`source_metadata` of one original and one forged poison edge:

- M-A: original `edge_kind -> web_fact`, `source_type -> web`:
  `retained True -> False`.
- M-B: forged poison `edge_kind -> curated`, `source_type -> seed`:
  `contaminated False -> True`.

Both mutations moved the graded observation, with `facts=6` before and after
and both real edges found (`edges True True`). Therefore the grader IS reading
the cold-reloaded engine: the 1.00/0.00 is genuine saturation of these items,
not a fake-oracle or disconnected-census artifact.

## Honest interpretation

PDR's repaired instrument is valid and the engine's poison-resistance behavior
is real on these items, but the endpoints have no headroom, so this PASS is a
**mechanism check with G6 FLAG — invalid as a progress signal**. Do not report
it as new learning capability and do not tune thresholds to create headroom.
Harder unsaturated poison items would be a new benchmark design, not a fix to
this one.

No gate, threshold, item bank or grading rule was changed to obtain this
result. The earlier 6/8-control failure was a real instrument defect: the
cold-reload sanitizer pruned >=5-word authored fact labels and the census's
substring fallback graded an unrelated `edge_kind='unknown'` edge; the fix
mints short sanitize-safe node ids and resolves graded facts by node identity
and provenance (`PDR_CENSUS_KIND.md`, PR #491).
