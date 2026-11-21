# Grounded re-learning A/B (2026-11-21)

**Question:** the Nov-19 `web_fact` provenance beyond the pickle truncation is
permanently lost (`docs/investigations/DRIFT_AND_OCEAN.md`). Can RAVANA
re-acquire those facts with provenance, and does the gain survive a cold reload?

## Instrument (unchanged)

`measure_fronts.GROUNDED` (n=60: 20 easy + 20 medium + 20 hard), `judge_grounded`,
the repaired production weights copied into an isolated `scratch/` data_dir.
`weights/` was never a save target. One engine process, `RAVANA_OFFLINE=1`,
`PYTHONHASHSEED=0`.

Intervention: for each of the 60 probe subjects, one concise factual sentence
was written through the **real** `WebToGraph.learn_text` intake path with
`source_type="seed"`, a unique `source_url`, and a fixed ingest context.
No judge, threshold or rubric was changed. Script:
`scratch/grounded_relearn.py`; raw log `scratch/grounded_relearn.log`; result
`scratch/grounded_relearn_result.json`; per-probe log
`scratch/grounded_relearn_gates.jsonl`.

## Result

| stage | grounded score | Δ vs T0 |
|---|---|---|
| T0 baseline | **33** (20/60) | — |
| T1 after re-teaching (same process) | **55** (33/60) | **+22** |
| T2 after `eng.save()` + cold boot | **45** (27/60) | **+12** |

42/60 facts were written (16 easy, 15 medium, 11 hard; 22 primary / 20
sealed). Failure reasons at T0 were dominated by `deflecting` (10) and
`no_predicate;deflecting` (9) — consistent with the missing factual priors,
not a scoring artifact.

## What the cold-reload gap shows

Six subjects gained at T1 then lost the gain at T2: `asteroid`, `eclipse`,
`comet`, `tides`, `canyon`, `reef`. All six had exactly **one** fact written
at teach time. A first-pass post-reload inspection sampled the first four
incident edges for those subjects and saw only `semantic`/`inferred`/`temporal`
edges with `edge_kind='unknown'` and no `source`; a subject with a curated
edge (`oxygen`) retained its gain. That is a **hypothesis**, not a settled
finding — the sample did not enumerate all incident edges. The concrete
follow-up is to census every incident edge and its `source_metadata` after
reload; trust-gated retrieval failing closed is the leading explanation.

## Claim discipline

This demonstrates **directed re-teaching with provenance** can restore ~two
thirds of the lost grounded score in-process and ~one third durably. It does
**not** demonstrate web-driven discovery: the facts were authored and injected
through the teaching channel. It does not restore the historical 57 — the
remaining gap includes those six provenance-losing subjects, 18 subjects whose
single fact did not write, and the separate 20-hard-probe subject-resolution
parser gap (`subject -> "that"`), which re-teaching cannot fix.

Honest current numbers after this experiment: repaired production state is
still **33**; a re-taught isolated copy measures **45** cold.

## After the provenance-prune repair (2026-11-21, PR #459)

Census over ALL incident edges confirmed the hypothesis for 5/6 lost subjects:
`asteroid`, `eclipse`, `comet`, `canyon`, `reef` had **zero** `grounded.local`
edges at T2 — their newly taught `is_a`/`semantic` facts carried earned
provenance but were pruned by the sleep junk sweep during the intervening
60-turn front, because the F4 protection covered only causal relations.

Fix: protect every earned edge kind (`curated`/`web_fact`/`definition`/
`curriculum`) regardless of grammatical relation. Re-run of the identical A/B:

| stage | before fix | after fix |
|---|---|---|
| T0 baseline | 33 | **38** |
| T1 immediate | 55 | **60** |
| T2 cold reload | 45 | **52** |

T0 itself rose by 5: the baseline front run also triggers sleep pruning, and
broader protection now preserves five pre-existing facts that were previously
swept. Cold delta is +14 vs +12. The T1→T2 gap narrowed from 10 to 8 but is
not eliminated — the remaining loss is a separate open question, not claimed
as fixed.
