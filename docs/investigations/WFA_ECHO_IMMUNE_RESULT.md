# Echo-immune WFA rerun — final measurement (2026-11-22)

Instrument: PR #468 (echo-immune parser, per-item episodic reset, self-consistent
trusted list, forced-choice arm). `scratch/learn5_wfa_echo_immune2.log`,
`scratch/learn5_wfa_summary.json` (exit 1).

## Result

| pool | I | E | R = I−E | exclusion margin | withdrawal |
|---|---|---|---|---|---|
| primary | 0.0 | 0.0 | **0.0** | 0.0 | **0.0** |
| sealed | 0.0 | 0.0 | **0.0** | 0.0 | **0.0** |

New audit counters:

- `pd_unparseable_rate = 1.00` (both pools) → new hard fail
  `unparseable_ceiling_*`.
- `echo_marker_rate`: inclusion **0.766**, exclusion **0.026**,
  invalidation **0.977**, forced-choice **0.000**.
- `forced_choice_explicit_rate = 0.00`, `forced_choice_no_answer_rate = 1.00`.

## What the fixes accomplished, and what they showed

1. The **trusted-list defect is fixed**: exclusion echo collapsed from ~0.88
   to 0.026. A claim's own `source_id` now appears in the allow-list.
2. The **echo contamination is fixed for exclusion and forced choice**: both
   show 0.000/0.026 echo markers; the parser no longer counts prompt words as
   engine decisions.
3. **Inclusion and invalidation still echo heavily** (0.766/0.977), so those
   arms remain contaminated by recall of the harness's own phrasing.
4. Most importantly, with a single-token forced-choice surface and no echo
   possible, the engine produced **zero explicit Y/N/W decisions** — all
   responses were classified `no_answer`.

That is exactly the falsifier pre-registered in `WFA_FAILURE_ANALYSIS.md` §4
fix 4: *"token arm still yields <0.50 explicit decisions → the failure is
upstream of articulation, which is a mechanism work item, not a measurement
item."*

## Conclusion

WFA is now a valid instrument that FAILs for a real reason: RAVANA does not
implement a source-evaluation / explicit-decision routine. It cannot currently
say "I endorse this claim because its provenance is credible", "I abstain — my
source is unknown", or "I withdraw that claim now that the source is
discredited". Its answers are chatter, mirror or parametric completion
(`engine_memory.py:2230`, `engine.py:6003`, `:7257`, `:8049`).

Brain-grounded next mechanism (do NOT tune thresholds): add an explicit
source-status evaluation step before articulation, corresponding to source
monitoring (Johnson, Hashtroudi & Lindsay 1993, *Psychol Bull* 114:3–28,
doi:10.1037/0033-2909.114.1.3) and the source-invalidation reevaluation seen
even in toddlers (*Open Mind*, doi:10.1162/opmi_a_00038). The process-dissociation
bounds remain exactly as pre-registered.

**Retired claim:** the earlier statement that R=0.015/0.075 proved RAVANA lacks
source monitoring was based on echo-keyword counts and is void. This run is the
first clean measurement, and its conclusion is the same direction but for a
different, now-verified reason.
