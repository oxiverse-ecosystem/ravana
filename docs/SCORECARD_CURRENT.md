# RAVANA-LEARN-5 current scorecard (2026-11-25)

`main` is clean at `bfe99a6`. No gate, threshold or rubric was relaxed to alter
any verdict. A fake-oracle 1.00 is never reported as capability evidence; a
saturated real endpoint must pass a mutation control before it is called valid.

| Component | Verdict | Evidence | Caveat |
|---|---|---|---|
| CAR — conversational acquisition | **PASS (verified 2026-11-26)** | Fresh sealed rerun on current `main`: all 10 criteria pass, primary Δ8 1.0 / sealed 0.9, adjacent-k reversals 0.0/0.0, cold-reload gain retained | Earlier P2 `−0.4` reversal was the stale-node-ID bug, now fixed and confirmed |
| CLI — continual learning | **PASS** | primary on−off +0.267, sealed +0.154, compute ratio 1.000 | Sealed transfer saturated; REM flip 0.10 pre-registered calibration |
| WFA — web-fact attribution | **FAIL (Phase 0 complete 2026-11-27)** | Complete full-bank run after performance fixes (re-cache only, 475.9s): IS primary credible/discredited/unknown 0.985/0.923/1.000; sealed 0.955/0.627/1.000; anti-tautology PASS; overall FAIL | PD_R, exclusion margin/credible, invalidation-withdrawal and unparseable-ceiling gates fail; saturation FLAG. Artifact: `scratch/full_B_reonly_summary.json`. No gate or threshold changed. |
| PDR — poisoned-data resistance | **PASS — G6 FLAG** | Repaired live run: all controls OK, backend live, 3,198 intake / 984 save / 984 reload calls; endpoints 1.00/0.00; M-A and M-B mutations both moved the graded observation | Saturated: valid mechanism check, **invalid as progress signal**. Full 82-item run used n=40/42 |
| RSI — relational rule induction | **FAIL** | Focused live n=4/pool with schema wiring: near 0.50, far 0.00, foil/negative correctly rejected. Root cause: no transfer of a relation to an unseen subject | Fake backend 1.00 is a G6 artifact. Multiple wiring defects fixed; remaining gap is genuine knowledge/generalization |

**Suite verdict: FAIL** (release rule requires all five components PASS).

## TRA — new typed-role acquisition benchmark

- Full bank implemented: 400 families, 200/pool, 40 per k per pool, frozen hashes.
- Focused live run `TRA_MAX_PER_POOL=20`: instrument controls OK, live backend
  confirmed, mutation control moved, `interpretation=genuine_fail`, exit 1.
- Primary/sealed transfer **0.000**, near **0.000**, far **0.000**, retention
  **0.000**, growth **0.000**; foil 0.35/0.20, negative 0.50/0.40.
- Conclusion: provenance-stamped typed slots are written and retained for
  taught words, but do not generalize to an unseen subject. Full 400-family
  run has not been performed.

## TRA-K — explicit semantic type channel (new benchmark)

- Wikidata `P31`/`P279` substrate, CC0-1.0, 3,200 provenance-stamped facts.
- Harness corrections landed after live diagnosis: source-native type-label
  options, subject-specific type evidence with deterministic conflicts, and
  LF-normalized provenance hashing for CI determinism.
- Focused live run (`TRAK_MAX_PER_POOL=20`, 40 families): instrument controls,
  subject/channel independence, backend liveness, mutation control all OK;
  saturation gate PASS. Valid measurement.
- Primary transfer `0.625`, near `0.450`, far `0.300`, retention `0.000`;
  sealed transfer `0.500`, near `0.500`, far `0.350`, retention `0.000`.
  `interpretation=genuine_fail`, `OVERALL: FAIL`.
- Full 400-family run is conditional on a clean focused result and was not
  performed. See `docs/investigations/TRA_K_FOCUSED_LIVE.md`.

## Other fronts

Grounded **38** (historical 57 lost to a truncated pickle; unrecoverable
provenance), thinking **20**, speaking **8**, frozen web **77** (stale).
The Lamp Test was removed as uninformative.

## Discipline

Primary/sealed hash splits, fake-backend invalidation, live API counters,
instrument controls before scoring, mutation controls for saturated endpoints,
and additive JSON artifacts are enforced in the harnesses. No production
mechanism is implemented until a scratch experiment fixes the measured gap.
