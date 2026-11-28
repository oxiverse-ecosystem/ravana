# TRA-K arbitration — taught-focus replication at 20/pool: PRE-REGISTRATION

**Status: PRE-REGISTERED at 2026-11-28T03:19:25Z, before any engine boot or live arm.**
Base `87f2ffd42ab9114d1155ec0bf49c8e983b5c530d` (PR #573, taught-focus mechanism PASS,
no full-bank claim). Isolated worktree
`C:/Users/Likhith/Documents/projects/ravana_trak_arb_repl20`, branch
`experiment/trak-arb-taught-repl20`. Prior evidence retained unchanged:
`TRA_K_SOURCE_ARBITRATION_TAUGHT_FOCUS_AB.md` (n=2/pool PASS_MECHANISM_ONLY) and
`TRA_K_SOURCE_ARBITRATION_TAUGHT_FOCUS_PREREG.md`.

## 0. What this is (and is not)

This is **replication only** of the n=2/pool taught-focus mechanism result at a
larger focused n. It is **not** a full-bank verdict, **not** a capability claim,
and **not** a production default-path change. The bank still has 400 items; this
run grades **40 families** (20 per pool), selected by the shipped
`--taught-focus` rule. The sealed pool is the bank's own sealed split, not an
independent replication cohort; a true independent cohort remains future work.
No gate, threshold, item definition, prompt, gold key, scorer, overlay decision
or production flag is changed.

## 1. Frozen surfaces (byte-identical to PR #573)

* `scratch/m123_combined_runner.py` — the shipped runner with `--taught-focus`
  and `focused_items_taught()`; **not edited**.
* `scratch/m123_preflight.py` — all **13 SHA pins** re-hashed every arm.
* `scratch/evaluate_trak_arbitration_live.py` — L1–L9 definitions, thresholds
  (0.80, +0.50, +0.05) and PASS rule imported unchanged; invoked only through a
  path-only wrapper that redirects `OUT` and reuses `ev.evaluate()`.
* `pf.ARM_FLAGS`, `RAVANA_SOURCE_ARBITRATION` default-off isolation, the
  `ARBITRATION` overlay, compute-matched wrong-tag/permuted `SHAM`, C1–C15 gates,
  `CONGRUENCE_FLOOR=0.20`, `STRICT_MARGIN=0.10`.
* `evaluate_trak_arbitration_live.py` writes its own `verdict.json` with the
  legacy `n_per_pool=2` literal; the authoritative replication artifact is
  `m123_ab_out_repl20/verdict_repl20.json`, which embeds the same L1–L9 results
  with corrected replication metadata and the R1–R8 checks.

## 2. The replication parameter (only change)

`--max-per-pool 20 --taught-focus` instead of 2. Boot-free verification of the
selector at this value (no engine, no store write) must confirm:

* 40 graded items total, exactly 20 per pool (`primary`/`sealed`);
* all graded items have `k >= 1`: 38 × k=1 and 2 × k=8 (one forced k=8 per pool);
* **no k=0 graded item** — k=0 remains only the cold `baseline` endpoint emitted
  by the unmodified `summarize()`, with `by_k["0"]["far"] is None` and
  `k0_minus_baseline is None` on both pools;
* selection uses only `{pool, k}` (answer-side stubs select identically).

This is checked before any live arm by `scratch/repl20_selector_check.py`, which
writes `scratch/trak_repl20_selector_check.json`.

## 3. Boot-free gates (must pass BEFORE any engine boot)

1. `python scratch/m123_preflight.py` — 13 pins, 22 preflight checks;
2. `python scratch/m123_combined_runner.py --self-test` — `ok=true`,
   `engine_booted=false`, `stores_written=false`, 13/13 pins, 22/22 preflight,
   8/8 arbitration, 5/5 taught-focus checks;
3. `python scratch/repl20_selector_check.py` — `ok=true`, selection as in §2.

Any failure → `INSTRUMENT INVALID`, no engine boot, no score.

## 4. Live arms

Five arms, run sequentially in ONE process chain (never concurrent engines),
each with its **own fresh seeded store copy** under
`scratch/m123_ab_data_repl20/trak_<tag>/` and one store per item:

| arm | tag |
|---|---|
| OFF | off1 |
| OFF2 | off2 |
| M1+M2+M3 | m123 |
| ARBITRATION | arb |
| SHAM (compute-matched wrong-tag/permuted) | sham |

Per arm: `timeout 3600`, `PYTHONHASHSEED=0`, `RAVANA_OFFLINE=1`,
`TRAK_CHANNEL_JSON` set to the shipped substrate channel, `--stage trak`,
`--arm <arm> --tag <tag> --max-per-pool 20 --taught-focus`. A `timeout 124` or a
missing `DONE` marker is an incomplete run, **not** a verdict; the last complete
arm report stays authoritative if a later arm dies.

## 5. Decision rule

Evaluate L1–L9 in order with the unmodified evaluator; first failing check is
the NO-GO reason. Then evaluate replication-specific checks:

* **R1** scale — all five reports have `max_per_pool==20`, `item_count==40`,
  `bank_count==400`, `focus_mode=="taught_only_k_ge_1"`, one k=8 per pool;
* **R2** no k=0 graded rows — every arm has `far_n==20` per pool,
  `by_k["0"]["far"] is None`, `k0_minus_baseline is None`, numeric cold baseline;
* **R3** fresh isolation — five distinct `arm_dir`/store roots, per-item stores
  present, `store_prefix_identical` true on every arm;
* **R4** controls — `failed_controls==[]`, C11 anti-fake OK, 13 pins match,
  G6/mutation status unchanged, no backend error;
* **R5** SHAM flat on **all** endpoints (near, far, delayed, retention, growth
  <= OFF + 0.05), zero emitted picks, module decisions identical to ARBITRATION;
* **R6** OFF == OFF2 on every reported endpoint on both pools;
* **R7** replication-specific replication of the mechanism: ARBITRATION far,
  near, delayed_retention and growth all >= 0.80 on both pools with SHAM flat
  (L7/L8 already require far >=0.80/gain >=0.50 and no regressions);
* **R8** scope — `full_bank_claim=false`, no capability claim in the verdict.

If L1–L9 and R1–R8 all pass → `REPLICATION_PASS_MECHANISM_ONLY` with
`n_per_pool=20`, `full_bank_claim=false`. If any falsifier fires → `NO-GO` with
the first failing check named. This remains a focused mechanism replication,
not a full-bank PASS and not retirement of CAR/WFA/RSI/TRA failures.

## 6. Exact commands

```bash
cd C:/Users/Likhith/Documents/projects/ravana_trak_arb_repl20
PY=C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe
export PYTHONHASHSEED=0 RAVANA_OFFLINE=1
export TRAK_CHANNEL_JSON="C:/Users/Likhith/Documents/projects/ravana_trak_arb_repl20/scratch/trak_channel_from_substrate.json"
"$PY" scratch/m123_preflight.py
"$PY" scratch/m123_combined_runner.py --self-test
"$PY" scratch/repl20_selector_check.py
bash scratch/run_trak_arbitration_ab_repl20.sh
"$PY" scratch/evaluate_trak_arbitration_repl20.py
```
