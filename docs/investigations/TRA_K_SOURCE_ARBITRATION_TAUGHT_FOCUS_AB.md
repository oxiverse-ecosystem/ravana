# TRA-K arbitration — taught-only focused rerun: mechanism PASS (no full-bank claim)

**Outcome: `PASS_MECHANISM_ONLY`.** L1–L9 all clear; first fired = none.
Focused **n=2/pool**, isolated stores, `timeout 1800` per arm,
`PYTHONHASHSEED=0`. **No full-bank claim and no production default-path change.**

Base `cf463ec` (PR #572). Worktree
`C:/Users/Likhith/Documents/projects/ravana_trak_arb_taught`, branch
`experiment/trak-arb-taught-focused`. Pre-registration:
`docs/investigations/TRA_K_SOURCE_ARBITRATION_TAUGHT_FOCUS_PREREG.md` (written
before any live run). Prior NO-GO (L7) evidence:
`docs/investigations/TRA_K_SOURCE_ARBITRATION_LIVE_AB.md` (PR #572), retained
unchanged.

## 1. Instrument repair actually performed

`scratch/m123_combined_runner.py` gained a default-off `--taught-focus` flag and
`focused_items_taught()`: per pool keep the forced k=8 item plus the first
k>=1, k!=8 item (bank order). Selection reads only `pool` and `k`. k=0 is no
longer a graded focus item; the cold k=0 baseline endpoint is still emitted for
both pools by the **unmodified** `summarize()`, and the full-bank fake
certification still supplies the k=0/control gates. No gate, threshold, item
definition, prompt, gold key, scorer, overlay decision or production flag was
changed; the 13 SHA pins still match worktree vs parent.

Graded focus (both arms identical): primary `tra_k_0020` (k=8), `tra_k_0018`
(k=1); sealed `tra_k_0000` (k=8), `tra_k_0010` (k=1). Legacy k=0 items were
`tra_k_0008` / `tra_k_0006`.

## 2. Boot-free self-test (before any engine boot)

`scratch/trak_arbitration_live_selftest.json`: `ok=true`,
`engine_booted=false`, `stores_written=false`, 13/13 pins match, 22/22 preflight
checks, 8/8 arbitration checks, **5/5 new taught-focus checks**:

* `taught_focus_ast_anti_leakage` — no answer-side token in the selector;
* `taught_focus_only_taught_items` — 2 items/pool, all k>=1, exactly one k=8/pool;
* `taught_focus_replaces_untought_k0` — legacy had k=0/pool, new focus does not;
* `taught_focus_ignores_answer_side_fields` — `{pool,k}`-only stubs select
  identically to the full bank (no key/gold leakage);
* `taught_focus_k0_baseline_reported` — unmodified `summarize()` on the taught
  focus via the shipped fake backend reports `baseline=0.0` on both pools,
  `far_n=2`, and no graded k=0 rows.

## 3. Live run

Five arms, each with its own seeded store copy (`scratch/m123_ab_data_taught/`),
`--max-per-pool 2 --taught-focus`; run 01:53:34Z → 03:05:01Z (~71 min, ~14 min/arm);
all `EXIT=0`, `ALLDONE` present; no `timeout 124`. All reports `status=OK`,
`scored=true`, `falsifiers_local=[]`, `failed_controls=[]`, C11 anti-fake OK,
pins unchanged, store prefixes identical.

| arm | far p/s | near p/s | delayed p/s | retention p/s | growth p/s | cold baseline p/s | emitted picks |
|---|---|---|---|---|---|---|---|
| OFF | 0.0 / 0.0 | 0.0 / 0.5 | – / – | – / – | 0 / 0 | 0.0 / 0.0 | 0 |
| OFF2 | 0.0 / 0.0 | 0.0 / 0.5 | – / – | – / – | 0 / 0 | 0.0 / 0.0 | 0 |
| M1+M2+M3 | 0.0 / 0.0 | 0.5 / 0.5 | 0.5 / 0.5 | 1.0 / 1.0 | 0.5 / 0.5 | 0.0 / 0.0 | 28 |
| **ARBITRATION** | **1.0 / 1.0** | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 83 |
| SHAM (compute-matched) | 0.0 / 0.0 | 0.0 / 0.5 | – / – | – / – | 0 / 0 | 0.0 / 0.0 | 0 |

Foil, negative and unrelated rates are 0.0 in every arm; F5 reject probe has
zero picks. ARBITRATION and SHAM have **identical** module decision counters
(self-loop 225, slot decisions 124, slot writes 124, arbitration calls 101,
supersedes 0, separations 101, identity slots 8) — SHAM emits no reader pick.
`OFF == OFF2` on every endpoint; flag set only in ARBITRATION.

## 4. L1–L9 (unchanged evaluator, path-only wrapper)

| # | check | result |
|---|---|---|
| L1 | instrument valid on all five arms | ✓ |
| L2 | AST anti-leak clean + boot-free self-test OK | ✓ |
| L3 | arbitration flag only in ARBITRATION | ✓ |
| L4 | OFF == OFF2 on all endpoints | ✓ |
| L5 | SHAM matched & flat (0 picks, far 0.0/0.0, decisions equal) | ✓ |
| L6 | mechanism fired (slot writes 124, separations 101, identity 8; M123 far 0.0) | ✓ |
| L7 | ARBITRATION far 1.0/1.0 >= 0.80; gain +1.0/+1.0 >= +0.50 | ✓ |
| L8 | near/growth/retention held, no reject regression | ✓ |
| L9 | cold baseline 0.0/0.0; 13 pins unchanged | ✓ |

`verdict.json`: `PASS_MECHANISM_ONLY`, `full_bank_claim=false`, `n_per_pool=2`.

## 5. Scope and next work item

This is a focused n=2/pool **mechanism** result on isolated stores: the
default-off arbitration module restores the graded far endpoint on taught
items with a compute-matched flat control, without touching k=0 nulls, gates or
the production default path. It is **not** a full-bank PASS and does not retire
the CAR/WFA/RSI/TRA failures. The held-out sealed pool here is the bank's own
sealed split, not an independent replication; replication at larger n with a
fixed seed is required before citing this as established.

Machine-readable: `scratch/m123_ab_out_taught/{trak_off1,trak_off2,trak_m123,
trak_arb,trak_sham}.json`, `scratch/m123_ab_out_taught/verdict.json`,
`scratch/trak_arbitration_live_selftest.json` (the latter two under
`scratch/`, force-added as the repo gitignores scratch artifacts).
