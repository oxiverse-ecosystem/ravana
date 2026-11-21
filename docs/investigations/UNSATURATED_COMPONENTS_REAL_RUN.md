# Real-engine runs of the unsaturated WFA / PDR / RSI (2026-11-22)

Instruments merged in PRs #461 (WFA) and #462 (PDR/RSI). All runs used isolated
scratch state, `RAVANA_OFFLINE=1`, fixed hash seed, no network, no save into
`weights/`. Unit suites: WFA 28 passed; PDR+RSI 46 passed.

## WFA — FAIL (new confound found)

`scratch/learn5_wfa_unsaturated.log`, `scratch/learn5_wfa_summary.json`:

| pool | n_gap | wfa_valid | answer_only | delta | abstain | decoy_rej |
|---|---|---|---|---|---|---|
| primary | 65 | **1.000** | 0.000 | 1.000 | 1.000 | 1.000 |
| sealed | 67 | **1.000** | 0.000 | 1.000 | 1.000 | 1.000 |

Hard failures: **`answer_only_web_leaks: 162`** (all other hard-fail counters 0).
Saturation gate: **FLAG** — `wfa_valid=100` is non-informative.

Two honest conclusions:
1. Paraphrasing the gold claim and adding 60 near-miss decoys did **not**
   desaturate attribution on this system — the synonym-map rubric still admits
   every item.
2. The live answer-only arm is not web-free. Disabling *web access* does not
   remove facts the engine already internalised in its graph, so its answers
   still contain source-span content. That is a genuine confound in the
   attribution-vs-answer contrast, not a scoring bug; a true parametric-only
   arm needs a separate store, not just an offline flag.

The verdict is FAIL and no threshold was relaxed.

## PDR — PASS, still saturated

`scratch/learn5_pdr_unsaturated.log`: primary n=40 / sealed n=42;
`orig0=orig5=1.00, intr5=0.00, forged=1.00 (prov_ok 1.00),
sleeper_dormant=1.00, sleeper_rej=1.00, durable=1.00, src=sense=1.00`.
Saturation gate: **FLAG** — `pdr_forged_reject=100`, `pdr_focus_intr=100`.

The redesign now tests durable, provenance-forged and sleeper poison instead of
in-memory token substitution — a real coverage gain — but the current system
has zero headroom on all of them. Still not capability evidence.

## RSI — PASS, still saturated and flat

`scratch/learn5_rsi_unsaturated.log`: both pools `near=far=foil=negative=1.00`,
`counterex=0.00`, `mono_drop=0.000`, **`growth=+0.000`**, k=1/2/4/8 all 1.00.
Saturation gate: **FLAG** (`rsi_near=100`, `rsi_negative_reject=100`).
The merged instrument deliberately gates on "no adjacent-k drop" and reports
growth as headroom **without gating on it**, so a flat 1.00 curve still PASSes.

## What this means for the suite

PDR and RSI remain formally PASS, WFA is now FAIL, and the G6 caveat is
stronger, not weaker: adding durable/forged/sleeper poison and negative rules
expanded *coverage*, yet every endpoint still saturates. The
remaining work is not more rubric items — it is an instrument that can
actually be wrong on the current system (a true parametric-only arm, and a
growth-gated rule-induction task).
