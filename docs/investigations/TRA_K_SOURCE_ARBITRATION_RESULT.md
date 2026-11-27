# TRA-K far — source arbitration after F4: boot-free result and default-off module

**Verdict: pre-registered boot-free criteria A1–A9 CLEARED. A pure, default-off
module and 20 unit tests landed. NO capability claim. NO engine boot, NO live
A/B, NO wiring, NO gate/threshold/item/prompt/gold/scorer change.**

Worktree `C:/Users/Likhith/Documents/projects/ravana_trak_arbitration`, branch
`invest/trak-source-arbitration`, base `cd711c8` (PR #570 head). Runner
`scratch/trak_source_arbitration_experiment.py`; raw report
`scratch/trak_source_arbitration_result.json`; normalized SHA-256
`37bb0a7899f34210ac4e85acb9b68e98551d2dc002533c8603016f32eeb9d951`
(identical under `PYTHONHASHSEED=0` and `1`). Pre-registration written BEFORE
the run: `docs/investigations/TRA_K_SOURCE_ARBITRATION_PREREG.md`.

## 1. Mechanism

`intake._revise` demotes an incumbent typed fact on every conflict, using
recency rather than source quality. The benchmark's own licensed provenance
distinguishes the two assertions (`primary`/`item_description` for the admitted
type vs `alternate`/`type_description` for the planted conflict), but the
stored edge collapses both to `conversation/user/curated`. The candidate rule
restores that distinction at the slot decision:

1. `cred = LICENSE_CREDIBILITY[license] × SENSE_WEIGHT[sense] ×
   EVIDENCE_TIER_WEIGHT[kind]` — **only source fields**; no stored confidence,
   recency, label familiarity, item id, role or answer field.
2. A challenger with no strict credibility advantage (`> MARGIN = 0.05`), a
   tie, no corroboration, an explicit negation, or an unadmitted source does
   **not** demote the incumbent. It is written as a weak **separate** trace
   (`pattern separation`), leaving the incumbent slot at full strength.
3. A strictly better, corroborated, admitted challenger **may supersede**
   (`reconsolidation is an update mechanism, not a freeze`).
4. **Self-loop rule**: a source-admitted identity assertion `X is_a X` is a
   valid typed slot trace (`WRITE_IDENTITY_SLOT`); it is not skipped merely
   because both endpoints have the same id. A self-loop with no admitted
   source is skipped.

Brain citations (verified this task, full mapping in the pre-registration):
Johnson, Hashtroudi & Lindsay (1993) *Psychol Bull* 114:3–28,
doi:10.1037/0033-2909.114.1.3; Sperber et al. (2010) *Mind & Language*
25:359–393, doi:10.1111/j.1468-0017.2010.01394.x; Dautriche et al. (2021)
*Open Mind* 5:1–19, doi:10.1162/opmi_a_00038; Nader, Schafe & Le Doux (2000)
*Nature* 406:722–726, doi:10.1038/35021052; Hupbach et al. (2007) *Learn Mem*
14:47–53, doi:10.1101/lm.365707; Leutgeb et al. (2007) *Science* 315:961–966,
doi:10.1126/science.1135801; Yassa & Stark (2011) *Trends Neurosci*
34:515–525, doi:10.1016/j.tins.2011.06.006. These are computational analogies
to symbolic typed edges, not neural-localisation claims.

## 2. Boot-free measurements

The runner imports the UNMODIFIED `m123_preflight`, `m3_graded_probe`,
`tra_k_items`, `measure_tra_k`, `subject_type_evidence` and `source_evaluation`
primitives, reads saved M123 stores read-only, deep-copies evidence per arm,
and grades every arm through the unmodified M3 hook (production
`CONGRUENCE_FLOOR=0.20`, `STRICT_MARGIN=0.10` untouched). Focused bank
`n=2/pool`, four far rows: a mechanism probe, not a full-bank verdict.

| arm | far primary | far sealed | near p/s | foil p/s | negative p/s | unrelated p/s | far picks (item → label) |
|---|---:|---:|---|---|---|---|---|
| BASELINE | 0.000 | 0.000 | 0.0 / 0.5 | 0 / 0 | 0.5 / 0 | 0 / 0 | none |
| BASELINE2 | 0.000 | 0.000 | identical to BASELINE | identical | identical | identical | none |
| ARBITRATION | **1.000** | **1.000** | 1.0 / 1.0 | 0 / 0 | 0.5 / 0 | 0 / 0 | guild→association, rank→class, market→market ×2 |
| SHAM | 0.000 | 0.000 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | none (wrong-tag trust) |
| K0 | 0.000 | 0.000 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | none (null control) |
| ORACLE (non-deployable) | 1.000 | 1.000 | 4/4 near | — | — | — | answer-side confidence injection; reachability only |

Slot-level evidence: BASELINE demotes 6 incumbent labels on conflict (e.g.
`guild: association 0.7 → 0.425`, challenger cred 0.2625); ARBITRATION keeps
all six at `0.85` and writes challengers below the congruence floor. The
self-loop rule fires on both `market` rows (`market→market conf=0.85`).
All 8 recorded conflicts were replayed through the shipped pure module:
8/8 identical decisions, 0 mismatches.

## 3. Falsifiers (pre-registered; all PASS)

| # | check | result |
|---|---|---|
| A1 | AST anti-leak (9 new functions, extended banned-token set), 13/13 pin hashes, 56/56 snapshot source checks, module/scratch equivalence 8/8 | PASS |
| A2 | BASELINE == BASELINE2 on all far rows and scores | PASS |
| A3 | SHAM 0 picks; records_scanned/trust_calls/scorer_calls/production_calls and decision calls equal to ARBITRATION (141/141/20/20, 8) | PASS |
| A4 | ORACLE 4/4 far | PASS |
| A5 | Mechanism fired: 6 baseline demotions, 6 arbitration restorations | PASS |
| A6 | far ARBITRATION − BASELINE = +1.000 on both pools (≥+0.50 required) | PASS |
| A7 | no near drop; foil/negative/unrelated not increased | PASS |
| A8 | K0 far 0; pins unchanged | PASS |
| A9 | self-loop rule fires on both market rows | PASS |

**Evaluator correction (pre-result interpretation, no score/threshold change):**
the first draft of A3 compared `records_admitted` between SHAM and
ARBITRATION. Wrong-tag trust is *designed* to reduce admissions (61 vs 125),
so that counter is an admission outcome, not one of the pre-registered
"call counts". A3 was corrected to compare only call counters; all other
checks and every arm's numbers were unchanged. The first draft's endpoint
table also mixed conditions; corrected to filter by condition before rates.
Both corrections are recorded here rather than silently applied.

## 4. Limits / falsifiers for the parent

- n=2/pool; the four far rows are a mechanism probe, not a capability result.
- The write simulation is a single-pass replay of the harness's recorded
  channel stream; the live store accumulated 8 revisions and ends at
  `association conf=0.00093`. The simulation reproduces the *ordering* defect
  (incumbent below floor, challenger above) but not those exact floats.
- Primary negative rate is 0.5 in BASELINE **and** ARBITRATION (one `thaw`
  row carrying an M2-only label from a residual non-`is_a` view). It is
  pre-existing in this scratch construction, not caused by arbitration; the
  live F4 sweep had negative 0.000 in all arms. The parent must confirm the
  control on the real engine before treating this as a capability gain.
- The pure module is NOT wired into `process_turn`, `_revise` or the typed-slot
  writer. `RAVANA_SOURCE_ARBITRATION=1` only opts in to a caller-side wrapper;
  no live path calls it. No capability claim is made anywhere.
- The module takes credibility from the caller's provenance; it cannot verify
  that the caller supplied it honestly. Live wiring must stage it from the
  engine's own licensed source records, never from a benchmark role/gold key.

## 5. Delivered artifacts and exact commands

* `ravana/src/ravana/learn/source_arbitration.py` — pure, frozen dataclasses,
  no engine/benchmark import; default-off `enabled()`/`integration_hook()`.
* `tests/unit/test_source_arbitration.py` — 20 tests, all passing.
* Scratch runner + result JSON + this doc.

```bash
# unit tests (worktree paths; module path verified by printing __file__)
cd C:/Users/Likhith/Documents/projects/ravana_trak_arbitration
export PYTHONPATH="C:/Users/Likhith/Documents/projects/ravana_trak_arbitration/ravana/src;C:/Users/Likhith/Documents/projects/ravana_trak_arbitration/scripts"
C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe -m pytest tests/unit/test_source_arbitration.py -q

# boot-free replay (reads saved stores; writes only scratch JSON)
PYTHONHASHSEED=0 RAVANA_OFFLINE=1 \
TRAK_CHANNEL_JSON=C:/Users/Likhith/Documents/projects/ravana_trak_arbitration/scratch/trak_channel_from_substrate.json \
C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/trak_source_arbitration_experiment.py
```

### Exact parent A/B command (proposal — not executed here)

```bash
cd C:/Users/Likhith/Documents/projects/ravana
export PYTHONHASHSEED=0 RAVANA_OFFLINE=1
export TRAK_CHANNEL_JSON=C:/Users/Likhith/Documents/projects/ravana/scratch/trak_channel_from_substrate.json
PY=C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe
for SPEC in "OFF:off1" "OFF2:off2" "M1+M2+M3:m123" \
            "ARBITRATION:arb" "SHAM:sham"; do
  A="${SPEC%%:*}"; T="${SPEC##*:}"
  timeout 1800 "$PY" scratch/m123_combined_runner.py --stage trak --arm "$A" \
    --tag "$T" --base "$PWD/scratch/m123_ab_data" \
    --out "$PWD/scratch/m123_ab_out/trak_${T}.json" --max-per-pool 2
done
```

`--base` must be a **fresh copied store per arm** (`seed_data_dir()` in the
runner); use a new absolute base directory for each arm if the runner does not
already isolate it. `ARBITRATION` is deliberately NOT added to `pf.ARM_FLAGS`
in this worktree: live wiring is the parent's decision after it reproduces the
boot-free criteria. No gate, threshold, item, prompt, gold key or scorer was
changed to obtain any number above.
