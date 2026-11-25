# TRA-K × Wikidata substrate integration — FAKE-arm verification

**Verdict: the FAKE arm is valid. No capability claim is made from it.**

This change plugs the independent Wikidata substrate
(`docs/investigations/SEMANTIC_SUBSTRATE.md`, `scratch/trak_substrate.json`,
980 CC0 P31/P279 facts) into the pre-registered TRA-K harness as the pluggable
`TRAK_CHANNEL_JSON` channel and verifies the fake/oracle arm end to end. It is
STATIC + fake-only: no engine boot, no network call, no store write, no
production edit under `ravana/src` or `ravana_ml/src`, and no gate/threshold
change.

## 1. Files

| File | Role |
|---|---|
| `scratch/trak_channel_from_substrate.py` | deterministic adapter: substrate → `TRAK_CHANNEL_JSON` |
| `scratch/trak_channel_from_substrate.json` | generated channel: 400 items × 8 facts = 3200 facts |
| `scripts/measure_tra_k.py` | one adapter fix in the **scratch** mutation regrade hook only (§5) |
| `tests/unit/test_measure_tra_k.py` | 6 new fake-only tests (53 pass total) |
| `scratch/trak_integration_fake_arm.log` | exact captured fake-arm stdout |
| `docs/investigations/TRAK_INTEGRATION.md` | this document |

The substrate commit (`962890a`, "build: independent semantic type substrate
for TRA-K") was cherry-picked onto this branch as `39ea03f` because it was on
`origin/goal/semantic-substrate`, not in this worktree's HEAD ancestry. All 8
substrate files are all-new, so the cherry-pick was conflict-free.

## 2. Exact commands (git-bash, native C:/ paths, main venv)

```bash
cd C:/Users/Likhith/Documents/projects/ravana_trak_integrate
VENV=/c/Users/Likhith/Documents/projects/ravana/.venv
export PYTHONHASHSEED=0 RAVANA_OFFLINE=1
export PYTHONPATH="C:/Users/Likhith/Documents/projects/ravana_trak_integrate;C:/Users/Likhith/Documents/projects/ravana_trak_integrate/scripts;C:/Users/Likhith/Documents/projects/ravana_trak_integrate/scratch;C:/Users/Likhith/Documents/projects/ravana/ravana/src;C:/Users/Likhith/Documents/projects/ravana/ravana_ml/src"

# 1. Generate the channel (deterministic; cap 8 facts/item == max(K_LEVELS))
$VENV/Scripts/python.exe scratch/trak_channel_from_substrate.py \
  --substrate scratch/trak_substrate.json \
  --out scratch/trak_channel_from_substrate.json --max-per-item 8

# 2. Fake arm (exit 1 is expected: pass=False by construction)
export TRAK_CHANNEL_JSON="C:/Users/Likhith/Documents/projects/ravana_trak_integrate/scratch/trak_channel_from_substrate.json"
$VENV/Scripts/python.exe scripts/measure_tra_k.py 2>&1 | tee scratch/trak_integration_fake_arm.log

# 3. Tests (single process; no concurrent pytest suites)
$VENV/Scripts/python.exe -m pytest tests/unit/test_measure_tra_k.py \
  -q -p no:cacheprovider --no-header

# 4. Module resolution (must be the worktree copies)
$VENV/Scripts/python.exe -c "import measure_tra_k,tra_k_items,trak_channel_from_substrate as a; print(measure_tra_k.__file__,tra_k_items.__file__,a.__file__)"
```

Adapter generation report (exact):

```json
{
  "facts": 3200,
  "items": 400,
  "ok": true,
  "overlap": {
    "answer_key_leak": 0,
    "facts": 3200,
    "oracle_domain_tag_leak": 0,
    "overlap_type_sets": 3200,
    "overlap_world_model": 0,
    "provenance_missing": 0
  },
  "problems": [],
  "validate_errors": 0
}
```

The generated file is byte-identical for `PYTHONHASHSEED=0` and `1`
(sha256 `b58c8165630dfd727e5a116a98b43d9adbfe9cdd28a434a83335ae836cf88081`).

## 3. Exact fake-arm output

Captured at `commit=39ea03f` (substrate cherry-pick; the harness fix was
already applied in the working tree at capture time). The integration commit
that follows contains only the adapter, its tests, artifacts and this doc —
no further harness behaviour change.

```
==============================================================================
TRA-K benchmark (pre-registered) -- backend=fake
==============================================================================
  commit=39ea03f families=400 primary=200 sealed=200
  hash_rule=RAVANA-TRA-K-v1|TRA-K|<difficulty>|<item_id>
  manifest_sha256 primary=c3e03c2cc664922b8b511722e00a35715a27688bbadfcde5af200bb457459637
  manifest_sha256 sealed =9f43014cefd4c6e66c1b7a08ddf971062e334baad0b7f19ee967837fd42f25af
  bank_sha256=0ed8bdc2ee1dd555adc4595868bce6794a14085c0d02761b8a957fc152e147cd
  k_levels=(0, 1, 2, 4, 8) one_k_per_item=True
  channel path=C:/Users/Likhith/Documents/projects/ravana_trak_integrate/scratch/trak_channel_from_substrate.json external=400 fallback=0 overlap={'overlap_type_sets': 3200, 'overlap_world_model': 0, 'facts': 3200, 'answer_key_leak': 0, 'oracle_domain_tag_leak': 0, 'provenance_missing': 0}

--- instrument controls C1-C15 (before any score) ---
  [ok] C1 -- channel near(k=8)>=0.90
  [ok] C2 -- foil channel reject>=0.90 and naive accept<=0.10
  [ok] C3 -- naive near-permuted>=0.15 and channel gap<=0.10
  [ok] C4 -- chance<=0.35 and far-chance>=0.20
  [ok] C5 -- negative channel reject>=0.85 and counterexample<=0.20
  [ok] C6 -- monotonic k-curve max drop<=0.05 (both pools)
  [ok] C7 -- graded near>=far>=foil max drop<=0.15 (both pools)
  [ok] C8 -- role coverage>=0.95 + grader fixed vectors
  [ok] C9 -- cold-reload retention>=0.80, fresh engine, counters>=1
  [ok] C10 -- type contrast flips pick; disjoint returns None
  [BAD] C11 -- anti-fake: live backend, real API counters, query-only grading
        backend=fake live=False error=None graded_from_query=False calls={'boot': 0, 'intake': 0, 'save': 400, 'cold_reload': 400, 'process_turn': 0}
  [ok] C12 -- provenance integrity; no fabricated URL/span; gate intact
  [ok] C13 -- immediate(k=8)-baseline>=0.15 and coverage loss<=0.05
  [ok] C14 -- unrelated subject never gets confident answer (HARD FAIL)
  [ok] C15 -- channel evidence overlaps oracle only benignly; no deployable use

--- decision rules (pre-registered, not tuned) ---
  [ok] primary_transfer>=0.75
  [ok] primary_near>=0.75
  [ok] primary_far>=0.70
  [ok] primary_far-chance>=0.20
  [ok] primary_foil_reject>=0.85
  [ok] primary_negative_reject>=0.80
  [ok] primary_permuted>=near-0.10
  [ok] primary_monotonic<=0.05
  [ok] primary_graded<=0.15
  [ok] primary_coverage>=0.95
  [ok] primary_retention>=0.80
  [ok] primary_control_delta<=0.05
  [ok] sealed_transfer>=0.75
  [ok] sealed_near>=0.75
  [ok] sealed_far>=0.70
  [ok] sealed_far-chance>=0.20
  [ok] sealed_foil_reject>=0.85
  [ok] sealed_negative_reject>=0.80
  [ok] sealed_permuted>=near-0.10
  [ok] sealed_monotonic<=0.05
  [ok] sealed_graded<=0.15
  [ok] sealed_coverage>=0.95
  [ok] sealed_retention>=0.80
  [ok] sealed_control_delta<=0.05
  [ok] sealed_transfer within 0.10
  [ok] sealed_near within 0.10
  [ok] sealed_foil_reject within 0.10
  [ok] instrument_controls_ok
  [FAIL] c11_anti_fake_ok
  [ok] subject_independence_ok
  [ok] channel_independence_ok
  [FAIL] backend_live
  [ok] backend_error_free
  [ok] saturation_flag_recorded
  [ok] mutations_passed_if_saturated

--- PRIMARY vs SEALED (channel arm) ---
  primary transfer=1.000 near=1.000 far=1.000 foil=1.000 neg=1.000 chance=0.000 perm=1.000 mono=0.000 graded=0.000 cov=1.000 ret=1.000 growth=+1.000 ostensive=1.0
  sealed  transfer=1.000 near=1.000 far=1.000 foil=1.000 neg=1.000 chance=0.000 perm=1.000 mono=0.000 graded=0.000 cov=1.000 ret=1.000 growth=+1.000 ostensive=1.0
  saturation gate: FLAG -- 2 front(s) non-informative: tra_k_near=100, tra_k_negative_reject=100
  mutation control M-A/M-B moved: True
  interpretation: mechanism_probe_only_fake_arm
  NOTE: fake backend cannot yield a capability claim (C11 BAD by construction).

OVERALL: FAIL (mechanism_probe_only_fake_arm)
artifacts: C:\Users\Likhith\Documents\projects\ravana_trak_integrate\scratch\tra_k_harness\tra_k_rows.jsonl , C:\Users\Likhith\Documents\projects\ravana_trak_integrate\scratch\tra_k_harness\tra_k_summary.json
elapsed 1.3s | DONE
```

## 4. Fake-arm numbers (from `scratch/tra_k_harness/tra_k_summary.json`)

| Field | Value |
|---|---|
| `backend` / `live` | `fake` / `False` |
| `families` | 400 (200 primary + 200 sealed) |
| channel `external_items` / `fallback_items` | **400 / 0** |
| channel facts | 3200 |
| `answer_key_leak` / `oracle_domain_tag_leak` / `provenance_missing` | **0 / 0 / 0** |
| `overlap_type_sets` / `overlap_world_model` | 3200 / 0 (benign, reported only) |
| C1–C10, C12–C15 | all `[ok]` |
| C11 | `[BAD]` by construction (`backend=fake`, `graded_from_query=False`, `boot=0`, `process_turn=0`) |
| `invalid_controls` | `[]` (C11 is excluded from invalidity by design) |
| `instrument_controls_ok` | `False` **only** because that field includes C11 |
| `rules.backend_live` / `rules.c11_anti_fake_ok` | `FAIL` / `FAIL` (expected) |
| `rules.subject_independence_ok` / `rules.channel_independence_ok` | `ok` / `ok` |
| `saturation_gate.status` | `FLAG` — `tra_k_near=100`, `tra_k_negative_reject=100` (G6) |
| `mutation_control` | M-A: match→reject, M-B: match→reject, `moved=True` |
| primary / sealed channel arm | transfer=near=far=foil=neg=1.000, chance=0.000, mono=0.000, cov=1.000 |
| `interpretation` | `mechanism_probe_only_fake_arm` |
| `pass` / OVERALL | `False` / `FAIL (mechanism_probe_only_fake_arm)` |

The 1.000s are the fake oracle saturating; G6 flags them as non-informative and
the mutation control proves the measured quantity is still connected to the
channel. They are **not** evidence of real-engine capability.

## 5. Deviations and adapter fixes

1. **Substrate commit cherry-picked**, not merged from main — see §1. No source
   file under `ravana/src` was touched.
2. **Scratch mutation regrade hook adapted** in `measure_tra_k.py`
   (`_channel_verdict`, `MUTATION_SENTINEL`). The original hook required a
   channel fact's `object` to equal the item's TRA-authored `type_label`. The
   substrate has **0/638** overlap with those six labels by design, so the hook
   returned `reject` *before* any mutation, G6's FLAG found a disconnected
   mutation control, and the run exited as `INSTRUMENT INVALID`. The hook now
   asks the channel-neutral question — is there still a provenance-stamped
   `is_a` fact naming a real type on a taught exemplar? M-A re-stamps the type
   to a sentinel and M-B removes the facts, so both move the verdict. The
   fallback channel's behaviour is unchanged. **No C1–C15 control, decision
   rule, `TH` threshold, or `front_gates` gate was edited.**
3. **Deterministic 8-fact-per-item cap** (`--max-per-item 8`, round-robin over
   taught demo words, substrate order within a subject, final sort by a
   provenance key). This matches the harness's `max(K_LEVELS)` teaching-surface
   cap and keeps the artifact at 3.4 MB. `--max-per-item 0` emits every
   deployable fact.
4. The adapter reads only the item's public `taught_exemplars` field to choose
   which global facts are deployable; `type_sets`, `world_model`, gold and
   `unseen_test_words` are never read as inputs. The harness's own answer-key
   filter is run as an output check only (and is additionally asserted by the
   new tests by poisoning those oracle fields and showing byte-identical
   output).

## 6. No capability claim

**The fake arm makes no capability claim.** It certifies only that the
instrument is wired correctly: the substrate channel loads, its provenance is
preserved, it carries zero answer-key leakage, the controls behave, and the
saturation mutation control is connected. Every 1.000 above comes from a
deterministic fake oracle that never boots the engine; C11 is BAD by
construction. Any claim about RAVANA's TRA-K performance requires the live
runs in §7.

## 7. Parent's next commands (NOT executed here — WFA is still running)

Focused live probe (40 items, mechanism check, not a full-benchmark verdict):

```bash
cd C:/Users/Likhith/Documents/projects/ravana_trak_integrate
VENV=/c/Users/Likhith/Documents/projects/ravana/.venv
export PYTHONHASHSEED=0 RAVANA_OFFLINE=1
export PYTHONPATH="C:/Users/Likhith/Documents/projects/ravana_trak_integrate;C:/Users/Likhith/Documents/projects/ravana_trak_integrate/scripts;C:/Users/Likhith/Documents/projects/ravana/ravana/src;C:/Users/Likhith/Documents/projects/ravana/ravana_ml/src"
export TRAK_CHANNEL_JSON="C:/Users/Likhith/Documents/projects/ravana_trak_integrate/scratch/trak_channel_from_substrate.json"
TRAK_MAX_PER_POOL=20 $VENV/Scripts/python.exe scripts/measure_tra_k.py --live
```

Full live (400 families):

```bash
unset TRAK_MAX_PER_POOL
$VENV/Scripts/python.exe scripts/measure_tra_k.py --live
```

Both use `trak_slots_flag()` to enable the two default-OFF TRA slot flags for
the live arm and restore their prior values on exit; the fake arm never enters
that context. Verify the flags are restored afterwards and re-run the C14/C15
probes on the live result before reading any number as a capability claim.

## 8. Blockers

None for the fake arm. Live runs are deliberately out of scope for this task
(WFA is still running), so no engine boot, store write or live verdict was
produced here.

