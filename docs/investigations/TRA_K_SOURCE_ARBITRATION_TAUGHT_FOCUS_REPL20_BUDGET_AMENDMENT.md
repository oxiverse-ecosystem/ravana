# TRA-K 20/pool replication — runtime-budget amendment (pre-result)

**Logged 2026-11-28T04:14:15Z, before any arm report, endpoint value, L1–L9
result or verdict was read.** This amends only the per-arm wall-clock budget in
`TRA_K_SOURCE_ARBITRATION_TAUGHT_FOCUS_REPL20_PREREG.md`; the decision rule,
falsifiers, thresholds, selection rule, gates, scoring and five-arm design are
unchanged.

## Observation (progress counters only)

The first launch (`START 2026-11-28T03:23:23Z`) ran the OFF arm under
`timeout 3600`. At 2026-11-28T04:06Z — ~43 minutes in — the runner's own
`ITEM_START` counter showed **17 of 40** graded items completed, with no arm
report written (`scratch/m123_ab_out_repl20/` contained only `run.log` and
`run_markers.txt`). Projected per-arm cost is ~100–110 minutes, so the
pre-registered 3600s cap would kill every arm mid-sweep and yield **no score**
at all. No endpoint values were inspected: the only log data read were the
`ITEM_START` count and arm timestamps.

## Amendment

* Per-arm `timeout` raised from **3600s to 9000s** (2.5 h) in
  `scratch/run_trak_arbitration_ab_repl20.sh`. Arms remain **sequential**, each
  with a fresh isolated store, `PYTHONHASHSEED=0`, `RAVANA_OFFLINE=1`,
  `--max-per-pool 20 --taught-focus`.
* The entire first attempt was discarded (no report existed); no partial state
  is carried forward. Store and output roots were deleted and recreated.
* L1–L9 and R1–R8, all thresholds, the shipped runner/evaluator and the
  replication-only scope declaration are byte-identical to the pre-registration.

If a reviewer treats 3600s as an inviolable hard cap, the correct verdict for
this replication is **RUNTIME-BLOCKED — no score**, not PASS and not FAIL. The
larger-budget run below is logged so it can never be mistaken for a run under
the originally registered 3600s budget.
