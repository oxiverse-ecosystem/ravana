# FINDING — bounded WFA (`WFA_IS_MAX_PER_POOL=2`) never terminates: the bound is applied only to the LAST arm

**Scope:** scratch-only investigation on branch `investigate/wfa-bounded-timeout-profile`,
worktree `C:/Users/Likhith/Documents/projects/ravana_wfa_profile`, branched from
`main` at `90623cc` (PR #549 + #550 merged). **No production file, gate, item,
scoring routine or threshold was edited.** No production PR is proposed.

## Headline

`WFA_IS_MAX_PER_POOL` is a bound on the **internal-source (IS) arm only**, and
the IS arm is the **last** thing `main()` does. The process-dissociation (PD)
arm that runs before it receives the **full** item bank and performs
**192 items / 708 responder calls / 840 `process_turn` calls** on one shared
engine. The bounded runs therefore die in the PD arm and **never reach** the
arm the bound was meant to bound. `WFA_IS_FRESH_ENGINE=1` likewise only changes
IS-arm engine construction and is also never reached.

Evidence from the three timed-out logs (`scratch/wfa_e2*_*.log` in main):

| log | exit | GloVe loads | `Loaded Remembered` | forced-choice prompts seen | IS-arm prompts | PD section header |
|---|---|---:|---:|---:|---:|---|
| `wfa_e2c_after_index_fix.log` | 124 (60 min) | 1 | 1 | 179 / 192 | 0 | absent |
| `wfa_e2d_after_two_fixes.log` | 124 (60 min) | 1 | 1 | 186 / 192 | 0 | absent |
| `wfa_e2e_min_probe.log` | 124 (30 min) | 1 | 1 | 125 / 192 | 0 | absent |

One engine boot each => `WFA_FRESH_ENGINE` was not set; the runs died inside
`run_pd_arms()` before the `--- process-dissociation arms ---` print at
`measure_learn5_wfa.py:2476` and before the `--- internal-source arm ---` print
at `:2562`.

## Static control-flow proof (`scripts/measure_learn5_wfa.py`)

`main()` order:

1. `:2329` web arm — `for it in items` (all 272; frozen provider, no engine).
2. `:2355` answer-only arm — `run_answer_only(items, ...)` (fake, no engine).
3. `:2422` **`run_pd_arms(items, records, arms)` — full `items`, no bound.**
4. `:2538` `run_internal_source_arms(is_items, ...)` — the only bounded arm.

The bound is read and consumed **only** at `:2498–2513`:

```python
max_is = int(os.environ.get("WFA_IS_MAX_PER_POOL", "0") or 0)
...
is_items = items
if max_is > 0:
    seen, keep, kept = Counter(), [], set()
    for it in items:
        if it["kind"] != "gap":
            kept.add(it["id"])          # ALL non-gap items kept regardless of bound
            continue
        if seen[it["pool"]] < max_is:
            keep.append(it); seen[it["pool"]] += 1; kept.add(it["id"])
    is_items = [it for it in items if it["id"] in kept]
```

`WFA_IS_FRESH_ENGINE` is consumed only at `:2522–2533`, inside the IS-arm
closure `_run`, which is never invoked on these runs.

Item-bank census (deterministic, `build_items()`, manifest
`119b9d91fe43a060…`): 272 items = 132 gap + 60 decoy + 60 closed_web +
20 internal. Only `gap` and `decoy` reach `run_pd_arms` (`:1496`).

| arm | items | responder calls | `process_turn` calls |
|---|---:|---:|---:|
| PD, unbounded (**actual**) | 192 | 708 | **840** |
| PD if current `max_is=2` selector were reused unchanged | 64 (4 gap + 60 decoy) | 216 | 200 |
| PD if bound extended to gap+decoy (true 2/pool) | 8 (4+4) | 28 | 32 |

`strip_echoes()` (`:1159–1173`) is a fixpoint `while` but strictly shortens the
string, so it is not an unbounded loop.

## cProfile — bounded 2/pool run, 180 s watchdog

Driver: `scratch/profile_bounded_timeout.py` (scratch, unmodified production
harness imported by path; a copy is committed alongside this report as
`WFA_BOUNDED_TIMEOUT_HARNESS_ROOT_CAUSE.profile.py`). Env: `RAVANA_OFFLINE=1`,
`PYTHONHASHSEED=0`, `WFA_IS_FRESH_ENGINE=1`, `WFA_IS_MAX_PER_POOL=2`. Watchdog
thread disables cProfile and dumps `pstats` at 180 s. Provenance printed at
runtime: harness and engine both resolved from the worktree. Full stats:
`scratch/profile_bounded_timeout.stats` (committed copy:
`WFA_BOUNDED_TIMEOUT_HARNESS_ROOT_CAUSE.stats`); raw log:
`scratch/profile_bounded_timeout.log` (1.4 MB, left uncommitted).

Totals: **208,432,148 function calls (201,808,510 primitive) in 179.996 s**.
Progress at the dump: `inc_exc=116`, `inv=57`, `fc=57` (≈58 of 192 PD items);
`process_turn` called **287** times.

Cumulative hotspots (of 180 s):

| cum s | % | ncalls | function |
|---:|---:|---:|---|
| 179.626 | 99.8 % | 1 | `measure_learn5_wfa.py:1488 run_pd_arms` |
| 179.351 | 99.6 % | 287 | `engine.py:8187 process_turn` |
| 83.681 | 46.5 % | 390,670 | `re/__init__.py:330 _compile` |
| 79.561 | 44.2 % | 79,892 | `re/_compiler.py:757 compile` |
| **65.437** | **36.4 %** | **488** | `user_model.py:1718 mine_personal_facts` |
| 30.279 | 16.8 % | 57 | `measure_learn5_wfa.py:1367 __call__` (forced choice) |
| 28.345 | 15.7 % | 3,073,760 | `numpy.fromnumeric.py:2245 clip` |
| 21.600 | 12.0 % | 156 | `core/system1.py:17 settle` |
| 19.190 | 10.7 % | 108 | `core/coherence.py:57 settle` |
| 12.651 | 7.0 % | 18 | `engine_generation.py:1214 _sleep_consolidate` |

By internal time the largest single leaf is also `mine_personal_facts`
(18.993 s tottime), followed by `re._parser._parse` (15.791 s) and
`re._compiler._compile` (14.867 s).

**Neither PR-fixed hotspot appears in the top 70:** `_activate_from_input`
(PR #549) and `ConceptEdge._init_relation_vector` (PR #550) are gone. The
remaining per-turn cost is production-engine regex/mining work, not the two
previously repaired functions, and not a harness loop bug.

## Root cause

Harness-level, structural: **the bounded flag is scoped to a trailing
diagnostic arm, so it does not bound the run.** All expensive engine work is in
the preceding, unbounded PD arm. Fixing engine throughput (PRs #549/#550)
only reduced the slope of an 840-turn loop; it did not reduce the loop length.
A "2/pool" run is therefore not a bounded run at all.

## Fix recommendation

A narrow harness-side fix **does exist, but only as a diagnostic mode**: it
cannot yield a valid scored verdict. Bounding PD items would break the
pre-registered gates `PD_n_gap_*>=60` and `PD_n_decoy_*>=30`
(`decide():1673–1674`), so a bounded run must be stamped non-verdict rather
than scored. No gate definition, item definition, scoring formula or threshold
needs editing; the full-bank path (`max_per_pool=0`) stays byte-identical.

Exact change (all in `scripts/measure_learn5_wfa.py`):

1. Hoist the deterministic selector at `:2503–2513` above `_boot_pd_engine()`
   (`:2392`) so `max_is` is known before `:2422`.
2. Extend the selector to bound `decoy` as well as `gap` per pool (currently
   every non-gap item is kept unconditionally at `:2506–2508`), producing one
   `bounded_items` list.
3. Pass `bounded_items` to **both** `run_pd_arms(...)` at `:2422` and
   `run_internal_source_arms(...)` at `:2538`. Keep the web and answer-only
   arms on the full 272-item bank — they use no engine, and the full bank is
   what preserves `build_items()` / `item_manifest_hash()` validation and the
   frozen hashes.
4. Add `summary["full_bank_verdict"] = (max_is == 0)` and
   `summary["bounded_per_pool"] = max_is`; when false, print
   `BOUNDED DIAGNOSTIC — no verdict` instead of PASS/FAIL. `decide()` and `TH`
   are untouched — they simply are not applied to a partial run.
5. Add a unit test that asserts, at `max_is=2`, the PD responder call count is
   28 and the IS call count is 12, and that at `max_is=0` both are unchanged
   from the current 708 / 120.

Expected result at 2/pool: ~32 PD turns + 12 IS conditions instead of 840 PD
turns, i.e. a run that terminates in minutes. If step 2 is skipped and the
current selector is reused verbatim, the win is 840 → 200 turns (4.2×), still
terminating.

If the parent wants an even smaller change, an alternative is a separate
`WFA_IS_ONLY=1` that skips the PD arm entirely for an IS-focused probe; this
also needs `full_bank_verdict=false` because `decide()` hard-fails on missing
arms (`:1658–1660`).

## Out of scope / follow-up

The per-turn `mine_personal_facts` + regex-compile cost is inside the
production engine. Filtering the harness's eight URL tokens before causal
extraction would change the graph and WFA outputs; it needs its own A/B with
gates and items unchanged (already flagged in `FINDING_REMAINING_STALL.md`).
Do not tune WFA thresholds to compensate for runtime.
