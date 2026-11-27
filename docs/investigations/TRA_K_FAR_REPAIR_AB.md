# TRA-K far-transfer repair A/B — executed result (F4 halt)

**Status: executed A/B, authoritative halt at F4. `SUPERSESSION NOT SUFFICIENT`.
No capability claim.** Worktree
`C:/Users/Likhith/Documents/projects/ravana_trak_far_ab`, branch
`experiment/trak-far-repair-ab`, base `816bca3` (PR #569 integration head),
runner commit `6b523c9`. No engine was booted for the boot-free phase; all six
live arms then ran in the shipped harness within the pre-registered 1800 s cap.

## 1. What ran

Pre-registered design: `docs/investigations/TRA_K_FAR_ROOTCAUSE.md` §5.
Runner: `scratch/m123_combined_runner.py --stage trak --arm <A> --tag <T>
--base <isolated> --out <report> --max-per-pool 2`, one process per arm,
`PYTHONHASHSEED=0`, `RAVANA_OFFLINE=1`, frozen `TRAK_CHANNEL_JSON`, **fresh
store copy per arm** via `seed_data_dir()`. Focused bank `n=2/pool` (4 items:
`tra_k_0020`, `tra_k_0008`, `tra_k_0000`, `tra_k_0006`) — a mechanism probe,
never a full-bank verdict.

| arm | manipulation | wall (s) | rc | arm status |
|---|---|---:|---:|---|
| OFF | none | 892.0 | 0 | OK / scored |
| OFF2 | independent repeat, fresh copy | 849.1 | 0 | OK / scored |
| M123 | DURABLE+SUBJECT_EVIDENCE+M3 | 854.2 | 0 | OK / scored |
| REPAIR | M123 + far misleading assertion → length-matched neutral turn | 848.1 | 0 | OK / scored |
| REPAIR_SELFLOOP | REPAIR + one self-loop re-affirmation (k≥1) | 863.1 | 0 | OK / scored |
| SHAM | M123 + `wrong_tag_trust` + permuted subject | 839.5 | 0 | OK / scored |
| ORACLE | boot-free, answer-side confidence injection, **non-deployable** | n/a | 0 | 4/4 picks |

All six arm reports have `scored=true`, no local falsifier, preflight OK,
pin hashes matched, controls OK, C11 anti-fake OK, backend error `None`.
Parent weight-store SHA-256 values are byte-identical to the preflight snapshot
before and after the sweep. `git diff --name-only HEAD -- scripts/ data/
ravana/src/ docs/investigations/` is **empty** (F10).

## 2. Endpoints (harness `summarize()`, primary | sealed)

| arm | far | far−chance | near | immediate | delayed | delayed retention | growth | foil | negative | chance |
|---|---|---|---|---|---|---|---|---|---|---|
| OFF | 0.000 / 0.000 | −0.250 / −0.250 | 0.000 / 0.500 | 0.000 / 0.000 | 0.000 / 0.000 | null / null | 0 / 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.25 / 0.25 |
| OFF2 | 0.000 / 0.000 | −0.250 / −0.250 | 0.000 / 0.500 | 0.000 / 0.000 | 0.000 / 0.000 | null / null | 0 / 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.25 / 0.25 |
| M123 | 0.000 / 0.000 | −0.250 / −0.250 | 0.500 / 0.500 | 0.500 / 0.500 | 0.500 / 0.500 | 1.000 / 1.000 | 0.5 / 0.5 | 0.000 / 0.000 | 0.000 / 0.000 | 0.25 / 0.25 |
| REPAIR | 0.000 / 0.000 | −0.250 / −0.250 | 0.500 / 0.500 | 0.500 / 0.500 | 0.500 / 0.500 | 1.000 / 1.000 | 0.5 / 0.5 | 0.000 / 0.000 | 0.000 / 0.000 | 0.25 / 0.25 |
| REPAIR_SELFLOOP | 0.000 / 0.000 | −0.250 / −0.250 | 0.500 / 0.500 | 0.500 / 0.500 | 0.500 / 0.500 | 1.000 / 1.000 | 0.5 / 0.5 | 0.000 / 0.000 | 0.000 / 0.000 | 0.25 / 0.25 |
| SHAM | 0.000 / 0.000 | −0.250 / −0.250 | 0.000 / 0.500 | 0.000 / 0.000 | 0.000 / 0.000 | null / null | 0 / 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.25 / 0.25 |

Row-level far-only rates (deployable `channel` arm, query / immediate /
delayed / k=0) are **0.000 / 0.000 / 0.000 / 0.000 in every arm**, including
both repair arms. `far REPAIR − M123 = +0.000` on **both** pools; oracle 4/4.

## 3. Controls

- **F2**: OFF and OFF2 endpoint tables are byte-identical on every key of
  both pools; both stores unchanged. No carryover.
- **F3 / SHAM**: `emitted_picks=0`, boot-free reject probe `0/12` picks on
  foil+negative+unrelated; hook/scorer/trust counters identical to M123
  (`160 / 160 / 1113`). `records_admitted` differs (`489` SHAM vs `1081`
  M123) exactly as wrong-tag trust requires; it is not one of the three
  counters F3 names.
- **F5 / F6**: near drops `0.000 / 0.000`; all four M3 arms pass the
  reject-condition probe with 0 picks; REPAIR control-probe deltas are 0.
- **F7**: k=0 far rows are 0.000 in all six arms — null control intact.
- **F8**: far is not saturated; G6 `FLAG`; mutation control did not falsely
  move. No capability claim arises from a saturated endpoint.
- **F9**: self-loop ids `tra_k_0000`, `tra_k_0006`; self-loop and non-self-loop
  far rows remain 0.000 under both repair arms; re-affirmation fired 8 times
  (4 items × 2 far-teaching phases) and added 8 `process_turn` calls. No
  option generator was changed from the gold key.
- **Compute match**: M123 and REPAIR both `turns=1520`, `query_turns=160`,
  `hook_calls=160`, `scorer_calls=160`, `emitted_picks=28`,
  `process_turn=1520`; `repair_misleading_replaced=4`,
  `repair_selffloop_reaffirmed=0`. REPAIR_SELFLOOP = +8 turns, as
  pre-registered.
- **ORACLE (non-deployable)**: on the **fresh M123 arm's** saved stores, the
  unmodified reader picks the displayed gold on 4/4 far rows
  (`market`, `market`, `class`, `association`) only after answer-side
  confidence is injected into a scratch evidence bundle. It proves the
  hook/scorer/emission path is reachable; it is not a capability result.

## 4. Falsifier order (authoritative halt F4)

| # | falsifier | result |
|---|---|---|
| F1 | pins + boot-free preflight + AST anti-leakage | PASS (13/13 pins, 27/27 self-test) |
| F2 | OFF == OFF2 on every endpoint | PASS |
| F3 | SHAM emits no reader pick; hook/scorer/trust counters match | PASS |
| **F4** | **far REPAIR−M123 ≥ +0.10 on both pools (oracle 4/4)** | **FAIL — halt** |
| F5 | near drop ≤0.05; reject-condition picks == 0 | PASS |
| F6 | REPAIR does not move foil/negative/unrelated | PASS |
| F7 | k=0 far rows unchanged | PASS |
| F8 | no uncaveated G6 saturation | PASS |
| F9 | self-loop ceiling characterised, not repaired | PASS (report only) |
| F10 | no production file / pin changed | PASS |

Rows after the halt are audit-only and cannot override F4. F4 is the decisive
fork: with the oracle reaching 4/4 and every other control clean, removing the
planted misleading assertion from **far** teaching alone is not sufficient.

## 5. Mechanism reading (why F4 failed)

A boot-free replay of the **REPAIR arm's own saved stores** (`rootcause_repair_
stores.json`) shows the guild slot is *still* superseded at grading time:

```
guild --is_a--> association         conf=0.0037  edge_kind=curated
guild --is_a--> student association conf=0.9872  edge_kind=curated
market --is_a--> economy            conf=0.9689  (no market→market edge)
```

The harness's `run_sweep` teaches conditions in the order `near, far, foil,
negative, unrelated` against the **same per-item store**. `channel_teaching_
surfaces(item)` is condition-independent, so the misleading guild assertion is
still taught during the unchanged `near` condition *before* the `far` query.
The far-only repair neutralises that assertion in the far turn (counter=4),
but it cannot delete the edge already written. `records_admitted` is therefore
identical for M123 and REPAIR (1081), as the verbatim misleading act remains
in the ledger.

This is not a reader-threshold defect and not a coverage defect: the oracle
shows the same evidence bundle, options and reader reach 4/4 when the gold
label's confidence is present. The missing input is the **positive
subject→type assertion that survives every teaching condition** (and, for the
self-loop items, a write that the typed-slot writer skips when `sid == oid`).
The next work item is the write/retention path, not the congruence floor, the
margin, the option generator, or the scorer. No mechanism is claimed to be
repaired here; the negative result is the deliverable.

## 6. Instrument correction (pre-result, validated)

The first evaluator draft scoped "pick" to the engine's row-level
`verdict=match` and compared `records_admitted` in F3. Validation on the
**OFF** arm (reader disabled, `emitted_picks` absent) showed the same
row-level negative rate (0.5) as M123, proving that metric is the engine's own
answer structure, not a reader emission. The evaluator was corrected to use
the runner's authoritative `emitted_picks` counter and its boot-free
`reject_condition_probe`, and to compare exactly the three counters F3 names.
No result, gate, threshold or score was changed; after correction F1-F3 and
F5-F10 pass and F4 is the sole halt.

## 7. Boundaries

- `n=2/pool`; the zero far rate is measured on four far rows and is a
  mechanism probe, not a full-bank capability verdict.
- ORACLE injects answer-side information and is **NON-DEPLOYABLE**; it proves
  reachability only.
- The runner shell script used an MSYS `$PWD` (`/c/...`), so arm reports were
  emitted under `C:/c/Users/.../m123_ab_out` and were copied verbatim into
  `scratch/m123_ab_out/` for this commit; no report was regenerated or edited.
- No gate, threshold, item, prompt, gold key or scorer changed. The 13 pinned
  files re-hash identically; production diff is empty.
