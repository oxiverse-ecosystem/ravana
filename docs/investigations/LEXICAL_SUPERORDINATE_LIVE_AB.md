# Lexical-superordinate default-OFF hook — live A/B result (NO-GO)

**Decision: FAIL / NO-GO. Keep `RAVANA_LEXICAL_SUPERORDINATE` unset.**
The wiring is **reachable** (OFF `calls == 0`, ON `calls > 0` and `emits > 0` on
both RSI and TRA), so this is **not** a wiring-failure stop. The hook emits, but
it does not improve the shipped endpoints on both pools, and it **breaks a
pre-registered negative control** (negated teaching). X6 failed live; X8 is not
satisfied. The hypothesis is refuted at the live endpoints.

This is the execution of the pre-registered plan in
`LEXICAL_SUPERORDINATE_WIRING.md` §5–§6. No gate, threshold (`DELTA=0.10`,
`MARGIN=0.05`), item bank, option list, answer key, `world_model`, type set,
domain tag, context model, store, prompt or scoring routine was changed. The
only manipulated variable is the module's existing flag. The engine was owned
by this agent; the parent booted none.

## Pre-flight (before any ON arm)

| Item | Value |
|---|---|
| Code revision | `origin/main` `61eda222` (PR #559 merged the default-off hook; #561 HEAD) |
| Worktree branch | `docs/lexical-superordinate-live-ab` from `origin/main` |
| WordNet 3.0 | `lexical_superordinate.wordnet_available() is True` |
| `RAVANA_WORDNET_DATA` | unset (nltk corpus discovered on default path) |
| Config, every arm | `RAVANA_OFFLINE=1`, `PYTHONHASHSEED=0`, all other `RAVANA_*` flags unset; ON adds only `RAVANA_LEXICAL_SUPERORDINATE=1` |
| Store snapshot | canonical `weights/ravana_weights.pkl` (SHA-256 `181f1c6a…0f80f`) and `.db` (`48057b89…e18`) copied fresh into a distinct `data_dir` per arm; hashes verified equal across arms |
| Repeated OFF baselines | **byte-identical** on all endpoints for RSI (2 arms) and TRA (2 arms) before any ON arm |

## Endpoint summary (harness's own `summarize()`; focused subsets)

Focused RSI = shipped 2 items/pool (384 rows, 1,824 `process_turn` calls, 1 boot).
Focused TRA = `TRA_MAX_PER_POOL=20` (1,120 rows, 5,170 `process_turn` calls, 80 boots incl. 40 cold reloads).

| Endpoint (OFF → ON) | RSI primary | RSI sealed | TRA primary | TRA sealed |
|---|---|---|---|---|
| near / transfer | 0.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 |
| far | 0.00 → **0.50** | 0.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 |
| foil reject | 0.00 → 0.00 | 0.375 → 0.375 | 0.05 → 0.05 | 0.05 → **0.00** |
| negative reject | 0.00 → 0.00 | 0.25 → **0.00** | 0.10 → 0.10 | 0.10 → 0.10 |
| chance / counterexample | flat | flat | flat | flat |

**Emit coverage** (headline for a precision route): RSI **48/384 = 12.5 %** of
gradeable forced-choice turns; TRA **88/1,120 = 7.9 %**. Paired delta on items
where both arms emitted is 0 for the graded endpoint (OFF never emits, so no
paired positive exists). The RSI primary `far +0.50` rests on 2 primary items,
is absent on sealed, and is accompanied by a sealed control regression — it is
not capability evidence.

## Reachability counters

| Arm | `calls` | `emits` | forced-choice turns | `process_turn` | `api_error` |
|---|---|---|---|---|---|
| RSI OFF1/OFF2 | 0 | 0 | 0 | 1,824 each | none |
| RSI ON | 1,824 | 48 | 384 | 1,824 | none |
| TRA OFF1/OFF2 | 0 | 0 | 0 | 5,170 each | none |
| TRA ON | 5,170 | 88 | 1,120 | 5,170 | none |

Wiring is reachable and the store yields usable lexical evidence on some items.
No tuning was attempted.

## Controls (pre-registered falsifiers)

| ID | Prediction | Observed | Verdict |
|---|---|---|---|
| X1 | OFF/ON byte-identical with no evidence | OFF baselines byte-identical; no side effect with no lexical evidence | PASS |
| X2 | `calls > 0`, `emits > 0` ON | as table above | PASS |
| X3 | untrusted edge cannot change pick | adapter reuses `traces_from_edges`; no untrusted edge was planted live | NOT EXECUTED (structural) |
| X4 | gold-key permutation inside option set: picks unchanged, accuracy collapses | hook has no gold-key parameter; picks gold-blind by construction; full accuracy-collapse sweep not re-run | PARTIAL / NOT EXECUTED |
| X5 | permuting query subject changes ≥80 % of picks | 48 emits all had correctly resolved subjects; option-order permutation probe emitted in neither order (inconclusive) | NOT EXECUTED at scale |
| X6 | negated teaching / unrelated subject / OOV → 0 emits | **negated teaching ON emits `breeze` (1/1); 16 RSI emits land on `rsi_002` negative (negated) causing sealed negative-reject 0.25→0.00.** Unrelated-subject 0 emits, OOV 0 emits | **FAIL** |
| X7 | direct trusted edge wins → hook defers | `already_bare_candidate` probe: 0 emits (defers) | PASS (probe) |
| X8 | ON−OFF > 0 on primary AND sealed with controls flat | RSI sealed no gain + control regression; TRA both pools flat + sealed foil regression | **FAIL** |
| X9 | equal `process_turn` counts and store snapshots | counts and start snapshots equal; ON wall-clock 1.24× (RSI) / ~1.3–1.4× (TRA, contention-confounded by a concurrent unrelated engine) | PASS (counts), ratio reported |

Additional probes: non-forced-choice (`hello there`) → 0 emits; CAR chain
articulation and chatter responses byte-identical OFF vs ON. WFA fronts were
not re-run; the hook is structurally inert on non-forced-choice turns (no
forced-choice parse, no emit).

## Why it fails (located, not tuned)

1. **Fail-closed contract broken on negation.** The 48 RSI ON emits are
   `ceremony` (far, role), `habit` (far, custom), `tempest` (negative, storm —
   sealed `rsi_002`). The `tempest` emits on a negated-teaching item are the
   16 that drive the sealed negative-reject regression. The adapter
   `lexical_teaching_pairs_from_graph` reads `(subject, target)` label pairs
   from provenance-trusted edges without any negation status, and `decide()`
   has no negation check; `RAVANA_LEXICAL_SUPERORDINATE=1` therefore permits an
   emission on a counterexample item. Fixing this would be a production code
   change to the hook/adapter — explicitly out of scope for this measurement
   and not attempted.
2. **Frame-word subject capture.** On 206/384 RSI forced-choice turns and
   208/400 (audit-capped) TRA turns the hook's staged subject is the frame word
   `true` (from "what is true of X?"), because `_subj_for_hooks` passes
   `_last_subject` and the hook's `subject in question` check accepts it. On
   those turns the score is the subject-independent comparator. All 48 real
   emits occurred on turns where the subject resolved correctly. This is a
   reachability-quality defect, not a reason to enable.
3. **No sealed positive.** The only positive delta (RSI primary far +0.50) is
   on 2 items, absent on sealed, and accompanied by a control regression and a
   negation failure. Per §5 decision rule the flag stays OFF.

## Compute parity

`process_turn` counts and start-state store hashes are equal across arms by
construction (equal work). ON pays extra WordNet lookups: RSI 1,478.6 s vs OFF
mean 1,194.5 s (**1.24×**); TRA 7,753.5 s vs OFF1 5,460.4 s (**1.42×**, but
OFF2 was 6,630.3 s and a concurrent unrelated engine was running, so the ratio
is contention-confounded). No bound was relaxed.

## Verdict

`OVERALL: FAIL — NO-GO`. Do not enable `RAVANA_LEXICAL_SUPERORDINATE`.
This is a scored, instrument-valid result (instrument controls certified by
the harness's own fake/oracle arms; the live arm is allowed to report FAIL).
The missing input for a future attempt is **negation-aware teaching-pair
extraction and subject-stage hygiene** in the hook's own adapter, plus a
subject-specific type signal in the live store — not a gate or threshold
change. Any future attempt must re-register this same A/B before running.

## Artifacts

Raw compact result JSONs and runner scripts are committed under
`docs/investigations/lexical_ab_artifacts/` (force-added: `*.json` is
git-ignored). Full logs and per-arm isolated stores remain in the worktree's
git-ignored `scratch/`.
