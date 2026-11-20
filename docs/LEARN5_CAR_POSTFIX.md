# RAVANA-LEARN-5 / CAR — post-fix measurement (F1 + F3 + F5)

**Status: still FAIL, but the binding primary criterion is repaired.** Measured
2026-11-20 on branch `fix/weight-store-durability`, repaired weights,
`PYTHONHASHSEED=0`, `RAVANA_OFFLINE=1`, one process, isolated scratch data_dir.
`scratch/learn5_car_measured.json` (post-fix), `scratch/car_full_postfix.log`,
elapsed 1569 s. Pre-fix baseline preserved as
`scratch/learn5_car_measured_PREFIX_baseline.json` and documented in
`docs/LEARN5_CAR_MEASURED.md` (commit `43db6cd`).

## Fixes applied (from `docs/investigations/CAR_ROOT_CAUSE.md`)
- **F1 — provenance-preserving load-time re-seed** (`engine_graph.py`
  `_seed_curriculum_causal`): the curriculum re-seed no longer overwrites a
  conversation-taught `curated` episode record (source/acquired_by) with
  curriculum defaults. Relabels only untagged or authored-seed edges.
- **F3 — trust-first candidate selection** (`learn/explanation.py`
  `enumerate_chains`): partition by `assertable_chain` BEFORE ranking; trusted
  chains rank above untrusted; runner-up taken within the same trust partition.
  Score values, `grade_chain`, `assertable_chain` and all thresholds UNCHANGED.
- **F5 — shared-predicate guard wiring** (`engine_self_query.py`
  `_mechanism_chain_available`): uses `mechanism_intent` (admit + direction),
  limit 3→64, narrow longest-label resolution fallback only when the subject
  resolves to no node.

No trust gate, threshold, rubric, or hash split was changed.

## Measured curves (Δ_cold = cold − baseline, n=10 per k per pool)

| pool | k=0 | k=1 | k=2 | k=4 | k=8 | worst adjacent reversal |
|---|---|---|---|---|---|---|
| PRIMARY pre-fix | 0.0 | 0.7 | 0.2 | 0.5 | 0.8 | **−0.50 (k1→k2) FAIL** |
| PRIMARY post-fix | 0.0 | 0.3 | 0.5 | 0.4 | 0.6 | **−0.10 (k2→k4) PASS** |
| SEALED pre-fix | 0.0 | 0.4 | 0.6 | 0.6 | 0.5 | −0.10 (k4→k8) PASS |
| SEALED post-fix | 0.0 | 0.5 | 0.2 | 0.5 | 0.8 | **−0.30 (k1→k2) FAIL** |

## Criteria (design §3.6, machine-checked)

| # | criterion | pre-fix | post-fix |
|---|---|---|---|
| 1 | primary Δ_cold(k8) ≥ 0.25 | 0.80 PASS | 0.60 PASS |
| 4 | sealed Δ_cold(k8) ≥ 0.25 | 0.50 PASS | 0.80 PASS |
| 7 | primary no adjacent-k reversal > 0.10 | −0.50 **FAIL** | −0.10 **PASS** |
| 8 | sealed no adjacent-k reversal > 0.10 | −0.10 PASS | −0.30 **FAIL** |
| 2,3,5,6 | controls ≤ 0.05 | PASS | PASS (all 0.00) |
| 9,10 | gain survives cold reload | PASS | PASS |

**Overall: FAIL** — the failure MOVED from primary criterion 7 to sealed
criterion 8. The located primary drop points (provenance overwrite, rank
crowding, guard wiring) are repaired and the primary curve is monotonic within
the bound; the sealed k=2 rate dropped 0.6→0.2 (3 items at n=10).

## Answer reach (secondary endpoint)
- PRIMARY k=8 immediate answer reach **0.30** (pre-fix post-`527f401` was 0.40);
  identical-surface echo 0.50 (parroting risk persists).
- SEALED k=8 reach **0.50**, echo 0.10.
Reach is noisy at n=10 and did not uniformly rise; F5's guard wiring is
gate-preserving but the resolution fallback's benefit is not yet established at
this n.

## Honest interpretation
- The fixes are **gate-preserving** (F3 cannot leak untrusted chains into
  emission — the gate still declines them; unit tests
  `test_explanation`/`test_mechanism_intent`/`test_curriculum_scaffold`/
  `test_learn5_car` = 92 passed).
- The sealed reversal appearing as the primary reversal disappears is exactly
  the **n=10 item-composition sensitivity** the root-cause doc flagged
  (`CAR_ROOT_CAUSE.md` RC-3, limitation 2). It is a hypothesis at this n, not a
  established regression.
- **No further tuning was done to chase the sealed reversal** (skill §1: never
  edit rubrics/gates to restore a number; a shortfall is the next work item).
- **Required next step (F7): replicate CAR at n≥40 per pool, same config twice,
  fixed hash split.** Until then both the primary repair and the sealed reversal
  are n=10 observations. The binding release rule (all criteria, both pools)
  remains unmet.

## Reproduction
```
cd C:/Users/Likhith/Documents/projects/ravana
PYTHONPATH=scripts PYTHONHASHSEED=0 RAVANA_OFFLINE=1 uv run python scripts/measure_learn5_car.py
PYTHONPATH=scripts uv run python -m pytest tests/unit/test_learn5_car.py tests/unit/test_explanation.py tests/unit/test_mechanism_intent.py tests/unit/test_curriculum_scaffold.py -q
```