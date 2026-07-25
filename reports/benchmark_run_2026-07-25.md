# Full Benchmark Run — 2026-07-25

Ran the 4 canonical RAVANA benchmarks against CURRENT code (HEAD = ab59806,
2026-07-24 14:18). Two of them (MemFail, TimeDial) had stale artifacts that
predated commit dae8b4e (2026-07-24 12:52, "relative-date resolution +
bounded associative graph (OOM root cause)"). LongMemEval was also refreshed.
LoCoMo was already complete post-fix via the fresh-engine driver.

## Method
- Driver: `scripts/evaluate_ravana.py` with `--skip-train` (restores existing
  snapshot `data/ravana_eval_snapshot.pkl`, mtime 2026-07-22 15:35).
- Verified the snapshot/restore path works cleanly (rc=0) via a 3-case smoke
  before launching full runs. (Note: memory says this harness segfaults in
  this env with numpy 2.4.6/Py3.11 BLAS-OpenMP — that did NOT reproduce today;
  all three full runs exited rc=0.)
- Each benchmark run isolated, output to `data/eval_*_fresh.json`.

## Results (current code)

| Benchmark        | Cases | Score (fresh) | Score (stale) | Notes |
|------------------|-------|---------------|---------------|-------|
| Memory Consistency (MemFail) | 200 | **0.7553** | 0.7553 | unchanged |
| Temporal (TimeDial)          | 600 | **0.4738** | 0.4738 | unchanged |
| LongMemEval (oracle)         | 500 | **0.0960** | 0.126  | DROPPED |
| LoCoMo (fresh-engine)        |1986 | **0.0604** | —      | already post-fix |

Runtimes: MemFail 479s, TimeDial 50.7s, LongMemEval 2335s (~39 min).

## Key finding — LongMemEval regression
LongMemEval fell from 0.126 (stale artifact, `data/eval_longmem.json`) to
0.096 (fresh run, `data/eval_longmem_fresh.json`). The run is deterministic
(seed=42, no per-case shuffle in the LongMemEval loader), so this is a real
difference, not run-to-run noise.

The ONLY code change between the stale longmem artifact (produced 2026-07-24
13:33) and now is commit **ab59806** (2026-07-24 14:18):
"fix(locomo): entity-attribute recall — routing + subject-attribute pool +
predicate-value rerank" — which touches `ravana/src/ravana/chat/engine.py`
and `engine_reasoning.py` (the shared routing/retrieval path).

MemFail and TimeDial are byte-identical to their stale numbers, so the change
did not affect short-fact retention or temporal cloze. But LongMemEval — which
depends heavily on multi-session retrieval/recall — dropped ~24%. This strongly
suggests **ab59806's routing/recall changes helped LoCoMo but regressed
LongMemEval recall** (likely the subject-attribute pool / predicate-value
rerank is now over- or mis-routing on LongMemEval's longer haystacks, or the
entity-attribute recall path is shadowing the general recall path).

## Caveat on provenance
The stale artifacts embed a decoder CE (~1.27) that differs from the current
snapshot's restored CE (~1.15). Both runs used the same on-disk snapshot
(mtime 2026-07-22), so this is a readout discrepancy, not a weight change.
The score deltas above are the load-bearing signal; CE readouts are not.

## Recommended next steps
1. `git stash`/revert ab59806 and re-run LongMemEval to confirm it is the
   cause (clean A/B). If 0.126 returns, the regression is confirmed.
2. If confirmed, inspect `engine_reasoning.py` entity-attribute recall routing
   for a guard that is too aggressive on multi-session haystacks, or a
   predicate-value rerank that penalizes LongMemEval question types.
3. Consider splitting the recall fix so LoCoMo improvements don't regress
   LongMemEval (and vice-versa).

## FIX APPLIED — Tier-1 #1: max_edges crash (VERIFIED)
**Bug (confirmed by code read + re-run):** `_prune_weakest_edges` in
`ravana_ml/src/ravana_ml/graph.py` used bare `self.max_edges` on its early-return
check (line 2150). A pre-`max_edges` snapshot graph (restore skips `__init__`)
lacks that attribute. `add_edge` already guards with `getattr`, but once edge
count crosses the 60000 cap during multi-session haystack priming, `add_edge`
calls `_prune_weakest_edges`, whose first line does `len(self.edges) <=
self.max_edges` → AttributeError → caught by the runner → `[error: ...]` →
score 0. This killed exactly the 181 largest-haystack LongMemEval cases.

**Fix (commit 2b87973):** moved `_max_edges = getattr(self, "max_edges", 60000)`
to the top of `_prune_weakest_edges` and used it in the early return. 2-line change.

**Measured impact (re-run, rc=0, 500/500, 14727s):**
| Run | LongMemEval | crash/error cases |
|-----|-------------|-------------------|
| before fix (data/eval_longmem_fresh.json) | 0.0960 | 181 |
| after fix  (data/eval_longmem_fix1.json)  | **0.1340** | **0** |

Delta +0.038 (+40% relative). The 181 crash cases now run; ~38 of them score,
lifting the benchmark. 122 graph/edge/prune pytest still pass.

**Corrected root-cause read:** the 0.126→0.096 drop reported above was NOT a
regression from commit ab59806. It was the `max_edges` AttributeError crashing
36% of LongMemEval cases on the current snapshot. The "stale 0.126" baseline was
produced by a different engine snapshot that never hit the cap. After the fix,
current code = 0.134. The ab59806 regression hypothesis is now DISPROVEN for
LongMemEval — the drop was the crash, not a routing change.

Remaining LongMemEval failures (452→271 zero-score) are the non-crash class:
wrong-fact retrieval (lexical overlap picking a filler fact), temporal-interval
mismatches, and multi-session ordering — covered by Tiers 1.2+ (not yet done).

## Files
- Fresh run outputs: `data/eval_memfail_fresh.json`, `data/eval_timedial_fresh.json`,
  `data/eval_longmem_fresh.json`
- Fix-1 output: `data/eval_longmem_fix1.json` (+ `_fix1.log`)
- Logs: `data/eval_*_fresh.log`
- Already-complete (post-fix): `benchmark_results/locomo_full_freshengine.json`
  (overall 0.0604, n=1986)
- Fix commit: `2b87973` (ravana_ml/src/ravana_ml/graph.py)
