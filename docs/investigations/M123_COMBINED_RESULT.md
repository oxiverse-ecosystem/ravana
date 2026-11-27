# M1+M2+M3 combined experiment — executed result

**Decision: HALT after Stage B (RSI).** TRA (Stage C) was **not run** because
F7 fired on RSI with clean instruments. Pre-registration:
`docs/investigations/M123_COMBINED_PREREG.md`. Runner/preflight:
`scratch/m123_combined_runner.py`, `scratch/m123_preflight.py`.

Branch `experiment/m123-combined-run`, base `6c0f1ea` (PR #566 branch
`pr/m123-implementations`). No production file, gate, threshold, item, prompt,
gold key or scorer was edited. All six arms used fresh store copies, one
process each, the shipped harness `run_sweep()/summarize()` and shipped
controls. Instrument controls C1–C10/C12–C15 were certified on the shipped
full-bank fake arm; C11 was read from the live arm (capability FAIL is a score,
never `INSTRUMENT INVALID`).

## 0. Boot-free preflight

`scratch/m123_preflight.py` exited 0 with `PREFLIGHT OK`:
**21/21 checks**, engine booted = false, AST anti-leakage clean over all
arm-specific staging/decision functions (0 banned identifiers), 13 newline-
normalised SHA pins matched the parent checkout, gold/type/world-model mutation
left the staged fingerprint unchanged, full-bank `validate_items` = 0 errors,
channel overlap clean (`answer_key_leak=0`, `overlap_world_model=0`,
`provenance_missing=0`, `oracle_domain_tag_leak=0`), delayed option audit clean
and its planted-domain-word mutation detected. Six canonical store files were
hashed before copying. Artifact: `scratch/m123_preflight_result.json`.

## 1. Stage A — TRA-K focused (max-per-pool=2; 4 items, 2/pool, k=8+k=0)

Every rate below is the harness's own `summarize()` key. Graded row counts:
`near/far/foil/negative n=2` per pool at the per-condition aggregate; the
registered `k=8` point has **n=1 per condition per pool** in this focused
probe. No full-bank escalation was performed (§4.1 not fully PASS).

| arm | k=8 near | k=8 far | immediate (n=2/pool) | delayed (n=2/pool) | delayed_retention | growth | foil / neg reject |
|---|---|---|---|---|---|---|---|
| OFF (`off1`) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | null / null | 0.000 / 0.000 | 0.000 / 0.000 |
| OFF2 (`off2`) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | null / null | 0.000 / 0.000 | 0.000 / 0.000 |
| M1 (`m1`) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | null / null | 0.000 / 0.000 | 0.000 / 0.000 |
| M1+M2 (`m12`) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | null / null | 0.000 / 0.000 | 0.000 / 0.000 |
| **M1+M2+M3 (`m123`)** | **1.000 / 1.000** | 0.000 / 0.000 | **0.500 / 0.500** | **0.500 / 0.500** | **1.000 / 1.000** | **0.500 / 0.500** | 0.000 / 0.000 |
| SHAM (`sham`) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | null / null | 0.000 / 0.000 | 0.000 / 0.000 |

Values are primary / sealed. Raw deltas against OFF (primary / sealed):

- `D_off` (OFF2−OFF) = **0.000 / 0.000** on every graded endpoint (byte-identical reports).
- `D_1` (M1−OFF) = **0.000 / 0.000**; `D_2` (M1+M2−M1) = **0.000 / 0.000**.
- `D_3` (M123−M12) = **+0.500 / +0.500** on `immediate`, `delayed`, `growth`;
  `D_3` on k=8 `near` = **+1.000 / +0.000** (sealed near was already 0.500 at
  all-k aggregate, k8 primary/sealed both 0→1).
- `D_tot` (M123−OFF) = **+0.500 / +0.500** on `immediate` and `delayed`;
  `delayed_retention` 0→1.000; `growth` 0→+0.500.
- `D_sham` = **0.000 / 0.000** on every graded endpoint.

§4.1 reading: `immediate` and `delayed` clear the +0.10 bound on both pools,
retention (1.000) and growth (+0.500) clear theirs, coverage loss is 0.000,
foil/negative do not drop, controls/baselines/SHAM are clean. **But k=8 `far`
stays 0.000 on both pools** (the `delayed` aggregate gain is carried entirely
by `near`), and step attribution fails because `D_1 = D_2 = 0`: the result is
**combined-only, no per-mechanism claim**, and §4.1 is **not a full PASS**.
G6 = `FLAG` (non-informative fronts), M-A/M-B mutation control moved on all
arms. `engine_response_unparsed`: 119→112 (no F11 regression). Wall-clock
ratios vs OFF: M123 1.037, SHAM 1.100 (≤1.12). Turn/hook/scorer/trust counters
are byte-identical between M123 and SHAM (1520 / 160 / 160 / 1113); M123
emitted 28 bare source-native picks, SHAM 0. Fail-closed probe on saved stores:
0 picks on foil (0/12), negative (0/12) and unrelated-subject (0/12) rows.

## 2. Stage B — RSI focused (shipped 2 items/pool; n=8 per condition per pool)

| arm | near | far | foil | neg_reject | unparsed / query turns |
|---|---|---|---|---|---|
| OFF / OFF2 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.375 | 0.000 / 0.250 | 339 / 384 |
| M1 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.375 | 0.000 / 0.250 | 339 / 384 |
| M1+M2 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.375 | 0.000 / 0.250 | 339 / 384 |
| M1+M2+M3 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.375 | 0.000 / 0.250 | 339 / 384 |
| SHAM | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.375 | 0.000 / 0.250 | 339 / 384 |

Raw deltas: `D_off = D_1 = D_2 = D_3 = D_tot = D_sham = 0.000 / 0.000` on
`near` and `far`. 384 query turns were issued in every arm (M123/SHAM ran 384
scorer calls, 0 picks); `process_turn` = 1824 in every arm; unparsed rate is
identical, so F11 does not fire.  Instruments clean: preflight/AST/pins OK,
controls OK, backend error null, shutdown stores recorded. **F7 fires**:
`D_tot < +0.05` on the registered primary structural endpoint with controls OK.
Per §6, stop tuning this chain, report the negative, name the missing link, and
do not escalate to a full bank. Per the run order, the experiment halts here:
**Stage C (TRA) was not executed**.

## 3. Falsifier ledger (evaluated in §6 order)

| ID | TRA-K | RSI | verdict |
|---|---|---|---|
| F1 | AST 0 hits, pins matched, 21/21 self-test | same preflight | PASS |
| F2 | controls OK, audit OK, no backend error, M-A/M-B moved | controls OK, audit N/A, no backend error, no shipped mutation control | PASS |
| F3 | `D_off = 0` byte-identical | `D_off = 0` byte-identical | PASS |
| F4 | `D_sham = 0`, hook/scorer/turn counters identical | `D_sham = 0`, turn counters identical | PASS |
| F5 | 0 picks on foil/negative/unrelated (12/12 each) | 0 picks in SHAM; no negated-emit regression | PASS |
| F6 | gold mutation + AST/overlap clean | same | PASS |
| F7 | primary `delayed` D_tot=+0.500 → not fired | structural `near` D_tot=0.000 → **FIRED** | RSI STOP |
| F8 | D_tot=+0.500 (not partial) | D_tot=0 | N/A |
| F9 | primary and sealed `delayed` both move (+0.500); sealed near/transfer already at 0.500 ceiling before the arm | both pools flat | PASS (with ceiling caveat) |
| F10 | retention 1.000; foil/negative drop 0.000 | no positive claim | PASS |
| F11 | unparsed 119→112 (improved) | 339 in every arm | PASS |
| F12 | M123/OFF 1.037; SHAM/OFF 1.100 | 0.909 / 0.919 | PASS |
| F13 | G6 FLAG with M-A/M-B moved | G6 FLAG, no positive claim | PASS |
| F14 | no full-bank escalation (no full PASS) | no escalation after F7 | N/A |

## 4. What the result does and does not show

The deployed TRA-K gain requires all three increments in the same run: M1 and
M1+M2 are flat, M1+M2+M3 moves `immediate`/`delayed` on both pools and survives
cold reload, SHAM is flat with matched counters, and foil/negative/unrelated
remain fail-closed. That is **combined-only G6 mechanism evidence at n=2/pool**,
not a capability claim: k=8 far is flat, step attribution is absent, no full
bank was run, and the reader's positive picks are bare displayed options
produced by a scratch wrapper, not a production change. RSI is a genuine
negative with clean instruments: the same chain does not reach the RSI query
answer, and near/far remain byte-identical at fixed item+k — the missing link
is a query-side surface the reader can commit from on that front-end, not a
threshold, prompt or accumulation of replay cycles. TRA was not run because
F7 fired at RSI.

## 5. Reproduction

```bash
cd C:/Users/Likhith/Documents/projects/ravana_m123_run
export PYTHONHASHSEED=0 RAVANA_OFFLINE=1
export TRAK_CHANNEL_JSON=C:/Users/Likhith/Documents/projects/ravana/scratch/trak_channel_from_substrate.json
PY=C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe
"$PY" scratch/m123_preflight.py
for SPEC in "OFF:off1" "OFF2:off2" "M1:m1" "M1+M2:m12" "M1+M2+M3:m123" "SHAM:sham"; do
  A="${SPEC%%:*}"; T="${SPEC##*:}"
  timeout 1800 "$PY" scratch/m123_combined_runner.py --stage trak --arm "$A" --tag "$T" \
    --base "$PWD/scratch/m123_data" --out "$PWD/scratch/m123_out/trak_${T}.json" --max-per-pool 2
done
# RSI: same loop with --stage rsi, no --max-per-pool.
```

Runner fixes applied during execution (each re-run from a fresh store; earlier
reports discarded): (1) pass the production trust predicate by omission rather
than `trust=None`; (2) read graph-view `source` rather than `subject` when
census-ing stored confidence; (3) parse explicit `is not a` teaching surfaces
so negated pairs veto; (4) normalise RSI's tuple-shaped control checks; (5)
only fire F2 on a mutation control that exists and actually fails. The two
RSI OFF reports were produced before fix (5) and their status fields were
recomputed from the stored guard fields without re-running the engine; no
endpoint, counter or score was altered.
