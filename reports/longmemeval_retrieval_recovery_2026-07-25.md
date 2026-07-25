# LongMemEval Retrieval-Ranking Recovery — 2026-07-25

## Summary

Recovered a LongMemEval regression introduced by the Tier 1.2/1.4 retrieval
work, using an additive, zero-regression ranking strategy. Certified on the
full 500-case oracle set.

| Metric | Value |
|--------|-------|
| Final 500-case (P0+P1+P2) | **0.136** (68/500 correct) |
| fix1 baseline | 0.1340 (67/500) |
| Delta | **+0.002** |
| Regressions vs fix1 | **0 / 500** |
| Gains vs fix1 | 1 (case #296) |

Scoring is binary (0.0 / 1.0), so 0.136 = exact case-pass-rate 68/500.

## Root cause (case-level diff, 120-subset)

The Tier 1.2/1.4 retrieval work replaced the baseline hippocampal ranking
tuple with an 8-tuple that pushed `density`, `dense_sim`, and `entity_binding`
into the **primary** sort key, and added a GloVe `_dense` tiebreaker to
`_best_date_hybrid`. This flipped 5 previously-correct cases (buffer indices
2,12,29,33,86 → LongMemEval cases 3,13,30,34,87) and dropped the first-120
slice from 0.1167 to 0.0833:

- Case 3  (bike vs car)           — generic "vehicle" fact won on density
- Case 13 (days between events)   — GloVe `_dense` tiebreak picked wrong date
- Case 30 (airline in Mar/Apr)    — "airline alliances" won on density
- Case 34 (cough vs skin tag)     — "bronchitis" won on embedding similarity
- Case 87 (grocery store)         — "buying in bulk" won on density

## Fix (commit 60e111a) — additive, zero-regression by construction

- **Phase 0** — restore baseline primary key `(active, matched, novel,
  confidence, turn_number)`; remove the GloVe `_dense` tiebreaker; the
  `_best_date_hybrid` key is now `(ov, _explicit)` with explicit-month
  absolute priority, `bkey = (0, 0)`.
- **Phase 1** — within-group reranker: after the baseline sort, re-rank ONLY
  within tied `(matched, novel)` groups by `(density, dense_sim_eff,
  entity_binding, confidence, turn_number)`. Inter-group order never changes,
  so it cannot regress vs the clean baseline.
- **Phase 2** — `dense_sim_eff = dense_sim * (1 + 0.5 * novel / len(fact_toks))`
  to down-weight related-but-generic facts inside a tied group.

### Verification ladder
- Phase 0 (120-subset): 0.116667 — **byte-identical** to the pre-regression
  baseline slice, 0 case diffs, all 5 flipped cases recovered to 1.0.
- Phase 1+2 (120-subset): 0.1167 — 0 regressions, 0 gains (safe no-op on this
  slice; baseline tuple already wins these groups).
- Full 500-case: **0.136**, +0.002 vs fix1, 0 regressions, 1 gain (#296
  "previous occupation" → correctly recalls "marketing specialist").

## Phase 3 (Tier 1.5 sequence recall) — ATTEMPTED, REVERTED

Ordering questions ("which X did I ... first, the A or the B?") remain
unsolved. Three implementations all failed:

1. Original regex required `happened|came|occurred|was` between entity and
   order word — real LongMemEval phrasings use arbitrary verbs
   ("take care of first", "attend first"), so it fired **0/120**.
2. Broadened regex + `retrieve_any()` probe: still 0/120 fires (facts lack
   `absolute_date` for these entities), AND regressed the slice 0.1167→0.1083
   because `retrieve_any()` has a **side-effect** (retrieval practice:
   `fact.confidence += 0.05`) that reinforced the losing candidate and
   corrupted the downstream hippocampal echo (case #33 flipped).
3. Read-only buffer scan (no side-effect): still 0/120 fires, still regressed
   to 0.10 (cases #32, #33 via echo path).

**Conclusion:** the blocker is upstream in fact ingestion — episodic facts
are not stored with `absolute_date` for the ordering entities, so any
date-comparison ordering handler is structurally impossible. Fixing this
belongs in the ingestion/date-grounding path, NOT the retrieval reranker.
`_answer_sequence_recall` was reverted to its original HEAD implementation
(commit 60e111a leaves it untouched).

Commit 435dae2 additionally removed the Tier 1.6 decoder-echo fallback (a
verified no-op: removing it left the 120-subset score unchanged).

## Files
- `data/eval_longmem_final500.json` — certified 500-case result (0.136)
- `data/eval_longmem_fix1.json` — baseline (0.1340)
- Commits: `60e111a` (recovery), `435dae2` (T1.6 removal) — both Likhithsai2580
