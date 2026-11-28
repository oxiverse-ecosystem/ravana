# TRA-K arbitration — taught-only focused subset: INSTRUMENT REPAIR pre-registration

**Status: PRE-REGISTERED before any live arm run.** Base `cf463ec`
(PR #572, `experiment(tra-k): live A/B of default-off source arbitration —
NO-GO (L7)`). Worktree
`C:/Users/Likhith/Documents/projects/ravana_trak_arb_taught`, branch
`experiment/trak-arb-taught-focused`.

## 0. What this is (and is not)

This is an **instrument-selection repair**, not a verdict rescue. The previous
run (`TRA_K_SOURCE_ARBITRATION_LIVE_AB.md`, PR #572) produced a clean mechanism
signal on taught items (far 2/2) but the pre-registered pooled far endpoint was
0.5/pool because the runner's `focused_items(..., 2)` selected one **k=8** and
one **untaught k=0** item per pool. k=0 receives only neutral chatter in this
harness; no channel assertion is taught, so an honest overlay has no deployable
evidence for it. Staging facts for k=0 was tried and **rejected** because it
produced cold `baseline=1.0`, `growth=0.0` and an F5 reject pick
(`INSTRUMENT INVALID — no score`).

The repair changes **which items enter the graded focus**: every graded item is
now taught at k>=1. This removes a structural zero from the denominator; it does
not change a gate, threshold, item definition, prompt, gold key, scorer,
overlay decision, or production engine default. No result from the prior run is
retracted or re-labelled.

## 1. The one change

`scratch/m123_combined_runner.py` gains a default-off `--taught-focus` flag and
`focused_items_taught(items, max_per_pool)`:

* per pool, keep the first **k=8** item (same forced item as the legacy rule);
* then keep the first **k>=1, k!=8** item per pool (bank order, deterministic;
  for `max_per_pool=2` this is k=1: `tra_k_0018` primary, `tra_k_0010` sealed);
* selection reads **only `pool` and `k`** — no gold, domain, answer-side or
  evidence-role field can affect it;
* without the flag the legacy `pf.focused_items` rule is byte-for-byte the
  default, so all earlier suites remain reproducible.

`k=0` is no longer a *graded* focus item, exactly as it was never a taught
level. The **cold k=0 baseline endpoint is still reported** for both pools:
the unmodified `summarize()` emits `baseline` for every selected item's cold
phase, and the unmodified full-bank fake certification still supplies the k=0
null/control gates. `k0_minus_baseline` on the focused live arm is `null` by
construction (no k=0 rows); no L1-L9 check reads it.

## 2. Unchanged (frozen) surfaces

* All 13 SHA pins in `scratch/m123_preflight.py` (including
  `scripts/measure_tra_k.py`, `scripts/tra_k_items.py`, the item bank, prompts,
  gold keys, `m3_graded_probe.py`) — re-hashed every run.
* `pf.ARM_FLAGS`, `RAVANA_SOURCE_ARBITRATION` default-off isolation, the
  `ARBITRATION` overlay and the compute-matched wrong-tag/permuted `SHAM`.
* `measure_tra_k.instrument_controls`, `decide`, `mutation_control`, all
  C1-C15 gates, `CONGRUENCE_FLOOR=0.20`, `STRICT_MARGIN=0.10`.
* `evaluate_trak_arbitration_live.py`: L1-L9 definitions, thresholds (+0.05,
  +0.50, 0.80, 0.05) and PASS rule are untouched — it is invoked through a
  path-only wrapper so its logic is byte-identical.
* `n_per_pool=2`, `full_bank_claim=False`, `timeout 1800`, `PYTHONHASHSEED=0`,
  one isolated seeded store copy per arm and per item.

## 3. Boot-free self-test gates (must pass BEFORE any engine boot)

`python scratch/m123_combined_runner.py --self-test` must additionally show, in
`trak_arbitration_live_selftest.json`, all checks OK with
`engine_booted=false`, `stores_written=false`:

1. `taught_focus_ast_anti_leakage` — no answer-side token in the new function;
2. `taught_focus_only_taught_items` — 2 items/pool, all k>=1, exactly one k=8
   per pool;
3. `taught_focus_replaces_untought_k0` — the legacy focus did contain k=0 per
   pool and the taught focus does not;
4. `taught_focus_ignores_answer_side_fields` — selection over `{pool,k}`-only
   item stubs equals selection over the full bank;
5. `taught_focus_k0_baseline_reported` — the **unmodified** `summarize()` on the
   taught focus (driven by the shipped fake backend, no engine) reports a
   numeric `baseline` for both pools, `far_n=2`, and no graded k=0 rows.

All pre-existing arbitration/preflight checks (flag isolation, AST anti-leak,
gold-mutation invariance, 13 pins, 22 preflight checks) must still pass. If any
new or old check fails → `INSTRUMENT INVALID`, no engine boot, no score.

## 4. Live decision rule (unchanged)

Five arms — `OFF`, `OFF2`, `M1+M2+M3`, `ARBITRATION`, `SHAM` — isolated stores,
`max-per-pool=2`, `timeout 1800` per arm. Evaluate L1-L9 in order; the first
failing check is the NO-GO reason.

If L1-L9 all clear **and** ARBITRATION far >=0.80 on both pools with SHAM flat
and controls intact → `PASS_MECHANISM_ONLY` with **no full-bank claim**. Any
L-check failing → `NO-GO`. The verdict remains a focused n=2/pool mechanism
probe, not a capability or full-bank result.

## 5. Exact commands

```bash
cd C:/Users/Likhith/Documents/projects/ravana_trak_arb_taught
PY=C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe
export PYTHONHASHSEED=0 RAVANA_OFFLINE=1
export TRAK_CHANNEL_JSON="C:/Users/Likhith/Documents/projects/ravana_trak_arb_taught/scratch/trak_channel_from_substrate.json"
"$PY" scratch/m123_preflight.py
"$PY" scratch/m123_combined_runner.py --self-test
bash scratch/run_trak_arbitration_ab_taught.sh
"$PY" scratch/evaluate_trak_arbitration_live_taught.py
```
