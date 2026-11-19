# RAVANA-7 v1 — 60-item front harnesses

Spec: `docs/BENCHMARKING_SURVEY.md` §5.2 (30 primary + 30 sealed replication,
stratified 20/20/20 easy / medium / hard). Applies to both
`scripts/measure_fronts.py` (offline) and `scripts/measure_fronts_web.py` (web).

## 1. What changed

- Each of the seven fronts now has **60 held-out items**: 30 `primary` +
  30 `sealed`, with exactly **10 easy / 10 medium / 10 hard per pool**
  (20/20/20 overall). The old 27-item lists are retained verbatim in both
  harnesses as `*_N27` provenance constants; they are disjoint from both new
  pools (asserted at build time).
- Every per-probe JSONL record now carries two new fields:
  `pool ∈ {primary, sealed, control}` and
  `difficulty ∈ {easy, medium, hard, control}`. All pre-existing fields, the
  per-probe JSONL logging, and the aggregate print block are unchanged.
- The summary additionally contains `primary_scores` and `sealed_scores`;
  `scores` remains the full-60 headline. Both arms print a primary-vs-sealed-
  vs-full-60 comparison so a sealed-set overfit check is possible.
- **No rubric was changed.** The judge functions, regex markers, thresholds,
  and all of `scripts/front_gates.py` are byte-identical to commit `6aabeda`
  (`git diff -- scripts/front_gates.py` is empty; no judge/threshold line in
  either harness was added or removed). `--gates-only` is unchanged and was
  verified to run and exit 0 on the new records.
- The scoring region between `def run_query_front(` and `def main(` (the region
  G4 compares) is character-identical across the two harnesses after whitespace
  normalization; it contains zero `RAVANA_OFFLINE` tokens and zero capability
  bonuses.

## 2. Sealed-set provenance (frozen before the scored run)

The sealed set is **held out by construction**: it was never used to make a
design decision, and no pass/fail outcome from it was inspected before the
single reported run. The procedure was fixed before any engine boot:

1. Author 60 new items per front (disjoint from the 27-item provenance lists),
   grouped as 20 `easy`, 20 `medium`, 20 `hard`. For code and language-learning
   the 60 novel symbols are `hfc60_01_fn … hfc60_60_fn` and
   `llc60_01_fn … llc60_60_fn`; the first/middle/last 20 are labelled
   easy/medium/hard and receive progressively longer signatures.
2. For each `(front, difficulty)` group, sort the 20 items by
   `sha256("RAVANA-7-v1|<front>|<difficulty>|<subject>")` encoded UTF-8.
   The first 10 are `primary`, the last 10 are `sealed`. This is deterministic,
   score-independent, and gives an exact 10/10/10 split inside each pool.
   The constant string `RAVANA-7-v1` and the hash rule are in
   `scripts/measure_fronts*.py` (`_seal_split`), so the split can be recomputed.
3. Because the rubric and gates were not edited at all, there was no tuning
   surface for the sealed items even in principle. The sealed items were run
   through the engine exactly once, in the measurement reported in §4.

Sealed manifest hashes (sha256 of canonical JSON of the 30 sealed records per
front, `sort_keys=True`, `ensure_ascii=False`; also saved at
`scratch/front60_manifest_hashes.json`):

| Front | Sealed n | sealed manifest sha256 (first 16) |
|---|---:|---|
| language understanding / conversation | 30 | `1efa5964cef179f2199` |
| grounded factual answers | 30 | `5c48df349efcbe9b2ee` |
| opinions / stance | 30 | `51124a9ab2d21a37ab9` |
| thinking / reasoning | 30 | `fa499f5769c77040e95` |
| speaking / surface realization | 30 | `09c6d26f6eb004f508b` |
| code generation | 30 | `166efc99411301fdeaf` |
| language learning (online) | 30 | `08e8778e4fab1549cf6` |

## 3. Run configuration

Commit `6aabeda`; Python 3.14.9 (repo `.venv`); `RAVANA_OFFLINE=1`,
`PYTHONHASHSEED=0`; one cold engine boot for all 420 probes in one process,
plus the two cold reloads required by G5; no `eng.save()` anywhere; weights
were read only. Elapsed 436.9 s. The web arm was **not** run here and is left
for the parent agent; an unpaired run is not yet a release result per §5.5.

## 4. Offline results (honest, n=60)

Full-60 is the headline `scores` field. Primary-only is the 30-item pool that
may be used for iteration; sealed is the held-out replication check.

| Front | Primary (n=30) | Sealed (n=30) | Full-60 (n=60) | n=27 baseline |
|---|---:|---:|---:|---:|
| language understanding / conversation | 87 | 100 | **93** | 85 |
| grounded factual answers | 10 | 23 | **17** | 78 |
| opinions / stance | 87 | 87 | **87** | 70 |
| thinking / reasoning | 10 | 7 | **8** | 19 |
| speaking / surface realization | 7 | 7 | **7** | 37 |
| code generation | 100 | 100 | **100** | 100 |
| language learning (online) | 80 | 80 | **80** | 80 |
| MEAN | 54 | 58 | **56** | 67 |

The drop from the 27-item baseline is real and is **not** a rubric change.
Hand-checking the raw records (per the RAVANA debugging rule) shows the engine
genuinely deflects on terms absent from its concept store, e.g.
`what is an asteroid` → `honestly, Asteroid is a bit outside what i know right
now.` (correctly FAILed by the unchanged grounded judge), and produces
single-clause non-causal replies for most new mechanism questions (correctly
FAILed by the unchanged thinking judge). The new primary and sealed pools are
different items from the old 27, so the two numbers are not a like-for-like
A/B; the n=27 column is provenance only.

Sealed-vs-primary deltas are +13 (conversation), +13 (grounded), 0 (opinion),
-3 (thinking), 0 (speaking), 0 (code), 0 (language learning). This is a single
reveal, not a trend estimate, but there is no uniform sealed advantage that
would indicate primary-set tuning (the rubric was not tuned in any case).
Scores that fall at n=60 are reported as measured; no rubric was relaxed to
recover the n=27 numbers.

## 5. Gate outcome

`G1 PASS`, `G2 PASS` (real 0.4233 − control 0.0000 = +0.4233 ≥ 0.30),
`G3 PASS`, `GC PASS` (0/14 control failures), `G4 PASS` (identical scoring
regions, 0 conditional tokens, 0 bonuses, earned == reported on all seven
fronts), `G5 PASS` (420 records, indices 1..60 per front, cold reload ok),
`G6 FLAG` — `code generation=100` is non-informative and must be refreshed
before it is used as a progress signal (non-blocking). Overall status
`PARTIAL (saturated fronts INVALID)`; no blocking gate failed, so the scores
are reportable with the saturation caveat.

## 6. Artifacts

- `scratch/front_probes_n30.jsonl` — 420 non-control records + 14 controls.
- `scratch/front_probes_n30_summary.json` — scores + primary/sealed scores.
- `scratch/front_probes_n30_gates.jsonl` — G1–G6 gate records.
- `scratch/front60_offline_run.log` — full stdout of the offline run.
- `scratch/front60_gates_only_offline.log` — `--gates-only` verification.
- `scratch/front60_manifest_hashes.json` — provenance hashes.

Reproduce: `RAVANA_OFFLINE=1 PYTHONHASHSEED=0 uv run python
scripts/measure_fronts.py`; gates: append `--gates-only`.
