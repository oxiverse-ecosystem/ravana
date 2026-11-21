# RAVANA-LEARN-5 — consolidated result

**Suite verdict: FAIL.** The pre-registered release rule in
`docs/BRAIN_LEARN5_DESIGN.md` requires all five components to PASS. Two fail
(CAR, CLI) and three pass only with saturation caveats. No threshold or rubric
was changed to alter any verdict.

## Scorecard

| component | verdict | headline primary / sealed |
|---|---|---|
| CAR — conversational acquisition | **FAIL** | trusted-chain delta at k=8 +0.80 / +0.50, survives cold reload; monotonicity reversal −0.50; emitted-answer reach ≤0.10 |
| WFA — web attribution | PASS (limited) | attribution-valid 1.00 / 1.00, zero fabricated URLs; saturated, partly tautological |
| PDR — poison resistance | PASS (limited) | poison intrusion 0.00 / 0.00, original retention 1.00 / 1.00; saturated, in-memory variants |
| CLI — continual learning | **FAIL** | sleep-on minus sleep-off 0.000 / 0.000; real retroactive interference exists but replay does not help |
| RSI — rule induction | PASS (limited) | exact 1.00 / 1.00, permuted 1.00 / 1.00, far transfer +0.79 over chance; saturated, flat across k |

## What the suite establishes

1. **RAVANA can acquire and retain trusted causal chains from teaching.**
   CAR's graph-level endpoint is +0.80 on primary at k=8 with both controls at
   0.00, Fisher p=0.00071, and the gain survives a cold reload. This is a real
   learning signal, not a static score.
2. **RAVANA does not use what it learned in generation.** Emitted-answer reach
   on the held-out paraphrase was ≤0.10. The learned structure exists in the
   graph but rarely reaches the answer.
3. **Sequential teaching causes genuine retroactive interference.** A-only
   retention 0.900/0.833 versus sequential 0.700/0.833; concurrent A+B is
   0.875/1.000. Interleaving protects, but the sleep replay routine does not:
   sleep-on and sleep-off are identical with compute matched.
4. **Attribution and source discipline work on the authored distribution.**
   WFA and PDR are clean, but both are saturated and use authored or in-memory
   artifacts — they validate bookkeeping, not open-ended capability.
5. **Structural matching works but shows no demonstration-count effect.**
   RSI is perfect on exact, permuted, foil and far transfer, flat across
   k=1/2/4 — a mechanism check, not learning growth.

## What it does not establish

- No component demonstrates open-ended natural-language learning from the web.
  Five earlier attempts to improve that loop failed; this suite does not
  reverse those negatives.
- Three passing components are G6-saturated and therefore invalid as progress
  signals. Re-run them with harder held-out items before citing them as gains.
- CAR's primary pool has n=10 per k, below the design target of 40. The
  monotonicity reversal may be noise, which is a reason to replicate, not to
  waive the FAIL.
- CLI's criterion rate (0.50/0.30) means the interference result rests on a
  subset; the A-in-B audit is clean (0 items).

## Next work, in order of evidence

1. **Make generation use learned chains** — the CAR answer-reach gap (≤0.10).
   This is the highest-value target: knowledge is acquired but not emitted.
2. **Replicate CAR at n≥40 per pool** to resolve the monotonicity reversal.
3. **Investigate why replay has no effect** in CLI, with interleaving as the
   positive control already in hand.
4. **Rebuild WFA/PDR/RSI with unsaturated items** before treating their PASS
   as capability evidence.

## Post-fix status (2026-11-21, branch `fix/weight-store-durability`)

This section updates the verdict above for work **on this branch**; no
pre-registered gate or threshold was changed. Details:
`docs/LEARN5_CAR_POSTFIX.md`, `docs/investigations/{CAR_ROOT_CAUSE,CLI_ROOT_CAUSE,DRIFT_AND_OCEAN,WEB_AND_SATURATION}.md`,
`docs/investigations/HANDOFF_LEARN5_FIXES.md`.

- **Score drift explained:** the grounded 57→33 drop was a **truncated
  authoritative pickle** (save killed mid-write) followed by a silent cold
  start, not a code regression (same-weights A/B byte-identical). pkl repaired
  from a salvage + mirror merge; atomic save / load recovery / provenance mirror
  shipped. The lost web_fact provenance cannot be reconstructed — grounded is
  honestly **33** until those facts are re-learned.
- **Ocean test: fixed** (genuine ocean water-cycle priors), 17/17 pass.
- **CAR:** primary monotonicity reversal **−0.50 → −0.10 (criterion 7 PASS)**
  via F1/F3/F5 (provenance-preserving re-seed, trust-first ranking, shared
  mechanism-predicate guard). A sealed reversal (−0.30) appeared at n=10; no
  gate was tuned to chase it, and n≥40 replication is required. Suite still FAIL.
- **CLI:** the reverted fix's 1.122 compute ratio was a real confound — restoring
  edges mid-phase diverged every O(state) maintenance loop. P1 now shares ONE
  maintained snapshot (+ write-time capture) between ON and OFF, applies the
  contrast only as a commit-time overlay, and matches compute by construction
  (measured ratio 1.001, work counters equal). Three subtle integration bugs
  were found and fixed (healthy-edge stripping, divergent maintenance input,
  aliased cached state). **Final result: PRIMARY on−off +0.215 at ratio 1.000
  with equal work counters (real replay gain); suite still FAIL** because SEALED
  criterion rate 0.35 (<0.60) and SEALED retention saturates 7/7 in both arms
  (ceiling → on−off 0). See `docs/LEARN5_CLI_POSTFIX.md`.
- **WFA/PDR/RSI** remain G6-saturated; unsaturated redesigns are specified but
  not yet implemented.
- **Web loop** root cause is corpus poverty + read-path rarity (all five prior
  tweaks tuned the read schedule); a curiosity-gated single-fact encoding
  mechanism is designed (default-OFF) but not yet A/B tested.

**Bottom line unchanged in substance:** three passes are still saturated mechanism
checks, and the suite requires both pools on all five components to PASS. The
work above repairs located defects and prevents recurrence; it does not yet
establish open-ended learning.

## Artifacts

`scripts/measure_learn5_{car,wfa,pdr,cli,rsi}.py`,
`tests/unit/test_learn5_*.py`, `docs/LEARN5_*_MEASURED.md`, and the design
sources `docs/BRAIN_LEARN5_DESIGN.md`, `docs/LEARN5_CODEMAP.md`.
