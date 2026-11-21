# HANDOFF — RAVANA-LEARN-5 fixes (locked continuation artifact)

**Written:** 2026-11-21 ~05:30 IST, branch `fix/weight-store-durability`.
A fresh agent can execute the remaining phases without asking questions.
Read `docs/investigations/{CAR_ROOT_CAUSE,CLI_ROOT_CAUSE,DRIFT_AND_OCEAN,WEB_AND_SATURATION}.md`
for evidence; this file locks decisions and ordered next steps.

## Locked decisions

| # | decision |
|---|---|
| D1 | The grounded drift root cause is the **truncated pkl + provenance-less mirror**, NOT code (same-weights A/B byte-identical). Do not re-bisect code for it. |
| D2 | Durability fixes A/B/C (atomic save, load recovery, provenance mirror) are SHIPPED on this branch; never revert them to chase a score. |
| D3 | The Nov-19 web_fact provenance past the pickle cut is **permanently lost**; grounded 57 is not restorable by repair alone — it requires re-learning. Report 33 honestly until re-learning is measured. |
| D4 | CAR F1/F3/F5 are shipped; primary reversal repaired (−0.50→−0.10), sealed reversal (−0.30) is n=10 noise. **Do NOT tune gates/rubrics to remove it.** Next step is replication at n≥40 (F7), not more fixes. |
| D5 | CLI P1 = shared maintained snapshot + shared write-time capture; ON/OFF differ ONLY in the commit overlay (earned vs wrong tag). Maintenance is identical by construction. The ±10% wall band and all thresholds stay pre-registered; never relax them. |
| D6 | The OFF overlay must NEVER strip healthy trusted edges (only repair damaged hops). Violating this collapses ret_off to 0 and invalidates the contrast. |
| D7 | Web loop: the five read-schedule tweaks are exhausted (all reverted). The only untried axis is the **encoding gate** (curiosity/information-gap from EXTERNAL re-ask/correction signals, single provenance-stamped facts, schema-anchored). Ship default-OFF, A/B on frozen state, revert on no gain. |
| D8 | WFA/PDR/RSI 1.00 passes are G6-invalid. Redesigns are specified in WEB_AND_SATURATION.md §4; implement only after the CLI/CAR measurements settle. |
| D9 | Never run two engine processes concurrently (weight-store contention). Harness order: CLI run 5 → grounded re-measure → web A/B. |
| D10 | weights/ is never a save target for harnesses; isolated scratch data_dirs only. The repaired production pkl + mirror are the live state; corrupt original preserved as `weights/ravana_weights.pkl.corrupt-20261119`. |

## Current state (commits on `fix/weight-store-durability`)

- `56ddcdd` persistence durability (A/B/C) + ocean skeleton fix — verified.
- `878ed98` CLI overlay only repairs damaged hops + work counters + warmup.
- `bcc0315` CLI identical noop maintenance in both arms.
- `6cb1a70` CLI shared maintained snapshot (parity by construction).
- `c851255` CLI shared write-time capture (restores ON/OFF contrast).
- CAR F1/F3/F5 + docs committed earlier on the branch (see `git log`).

**Measurements so far**
- CAR post-fix (`scratch/learn5_car_measured.json`, `docs/LEARN5_CAR_POSTFIX.md`):
  primary Δ_cold curve 0,.3,.5,.4,.6 (worst reversal −0.10 PASS); sealed
  0,.5,.2,.5,.8 (−0.30 FAIL). Overall FAIL; failure moved primary→sealed.
- CLI run 2 (`scratch/learn5_cli_measured_INVALID_overlaybug.json`): INVALID
  (OFF stripped healthy edges, ret_off=0). Run 3: valid contrast (on−off 0.215,
  criterion 0.70, ret_on 0.929) but ratio 1.693. Run 4: parity 1.0 but contrast
  collapsed (OFF capture keys drifted). Run 5 (`scratch/cli_full_postfix5.log`,
  `scratch/learn5_cli_measured.json` when done): shared snapshot + capture —
  expected parity ≈1.0 AND contrast ≈0.2.
- Ocean: 17/17 counterfactual tests pass. Durability: 9/9 persistence tests pass.
  CAR/CLI unit: 92 + 16 pass.

## Ordered remaining phases

### Phase 1 — CLI run 5/6 verdict (DONE)
- 6 runs; the final (`scratch/learn5_cli_measured.json`, `docs/LEARN5_CLI_POSTFIX.md`)
  gives PRIMARY on−off **+0.215**, criterion **0.70**, ret_on **0.929**, wall
  ratio **1.000**, work counters equal, overlay 5/5; SEALED criterion **0.35**
  and retention 7/7 in both arms (ceiling → on−off 0). **Suite still FAIL.**

### Phase 2 — grounded re-measure post-CAR-fixes (DONE)
- `scratch/drift_grounded_postfix.json` = **33 (20/60), 0 flips** vs repaired
  baseline → F3/F5 falsifiers satisfied; no regression.

### Phase 3 — web encoding-gate mechanism (D7), default-OFF
- (Pre-work DONE) Both live and frozen web harnesses now support `--resume`
  keyed by `commit:weights-mtime`; `test_web_front_resume` = 3 passed. The frozen
  mean-77 is stale (measured on the degraded cold-start graph) and a full
  re-measure is infeasible in-session (~2 min/probe on the repaired graph).
- (Code DONE, A/B pending) `engine.py` tracks EXTERNAL curiosity signals
  (re-ask/correction), gates the synchronous FOK web read behind
  `RAVANA_CURIOSITY_GATE=1` (default OFF) with a 3-read budget.
  `test_curiosity_gate` = 6 passed; offline differential 6/6 identical.
  Keep OFF until the frozen-corpus A/B shows single-fact gain + flat negatives.
- A/B on the frozen fixture corpus (`scripts/measure_fronts_web_frozen.py`
  pattern), identical state, one process. Positive metric: provenance-stamped
  single-fact count + emitted reach on previously-failed subjects. Negative
  metric: untrusted-edge count + known-bad rejections unchanged.
- **Done when:** A/B measured; keep only if positive moves and negative flat;
  else revert and record the negative (skill §7a). Also re-measure the frozen
  web arm post-intent-fix (the mean-77 number is stale).

### Phase 4 — CAR replication at n≥40 (D4) — BLOCKED by bank size
- `_CANDIDATES` has 104 triples; design §3.5 needs 800 for ≥40/k/pool. An 8×
  authored-bank expansion would add ~700 more scaffolding items, changing what
  the instrument measures — NOT done silently. A `PYTHONHASHSEED=1` replication
  at the existing n=10 is DONE: primary −0.1 reproduced exactly, sealed −0.2
  (vs −0.3) — the sealed reversal is reproducible, not a fluke (see
  `docs/LEARN5_CAR_POSTFIX.md`).

### Phase 5 — unsaturated WFA/PDR/RSI (D8)
- Implement the redesigns in WEB_AND_SATURATION.md §4 (paraphrase entailment +
  live-engine answer-only arm; durable/forged/sleeper poison; engine-booted RSI
  with monotonic-k gate). **Done when:** each front no longer saturates at 1.00
  on the current system and the hash-split sealing is intact.

### Phase 6 — PR
- Push `fix/weight-store-durability`, open PR against main with the measurement
  tables + citations. Stage by explicit path; never `git add -A` (scratch and
  weights are untracked and must stay out). Verify remote branch exists before
  opening the PR.

## Verification commands
```
cd C:/Users/Likhith/Documents/projects/ravana
uv run python -m pytest tests/unit/test_learn5_cli.py tests/unit/test_learn5_car.py tests/unit/test_explanation.py tests/unit/test_mechanism_intent.py tests/unit/test_curriculum_scaffold.py tests/unit/test_db_persistence.py tests/unit/test_provenance_persistence.py -q
uv run python -m pytest tests/unit/test_chat_query_fixes.py -q -k "counterfactual_premise or conditional_detection"
```

## Risks / rollback
- Shared-maintenance cache (`_SHARED_MAINT`) couples sequential ON/OFF order
  (ON must run first; main() guarantees this). If arm order changes, the cache
  key still matches by teaching config, but verify ON computes before OFF.
- The repaired production pkl is 15.5 MB (salvage+merge). If a future load
  warns schema/checksum, the mirror now holds provenance — `[Load recover]`
  rehydrates safely. Rollback of any fix: `git revert <commit>`; never restore
  the corrupt pkl.
- Do not run the full unit suite concurrently with engine harnesses
  (16–22 min, contention). Targeted modules only during measurement phases.