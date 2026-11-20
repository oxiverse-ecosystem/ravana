# RAVANA-LEARN-5 / CLI — measured result

**Status: FAIL.** The required sleep-on benefit is not present: sleep-on A
retention equals sleep-off retention on both pools (`on − off = 0.00`). A
separate, real retroactive-interference effect *was* measured and isolated
(A-only 0.90 vs A→B 0.70 on PRIMARY), and concurrent A+B interleaving
protected A (0.875), but the consolidation replay ablation did not. Per the
task's pass criteria, `on − off < 0.10` is an explicit FAIL; no threshold,
rubric or trust gate was relaxed to try to remove it.

**Measurement commit:** `c47ac09` (PDR commit; RSI `222cdd2` landed while the
run was in progress — CLI does not import PDR/RSI code). Host Windows,
`uv run`, `PYTHONHASHSEED=0`, `RAVANA_OFFLINE=1`. 12 cold boots in one
process, one isolated scratch `data_dir` per arm, **no `eng.save()` call at
all** — `weights/ravana_weights.{pkl,db}` were never a save target. Elapsed
**1496.4 s** (~25 min, dominated by the engine's per-turn cost on a growing
graph, as in CAR).

## 1. What was measured

CLI (continual learning under interference) is the A→B→A component of
RAVANA-LEARN-5. The mechanism is grounded in the standard retroactive-
interference paradigm (Wixted, 2004, *Annual Review of Psychology*) and the
catastrophic-interference result that sequential A→B training disrupts A while
**concurrent** A+B training does not (McCloskey & Cohen, 1989, *Psychology of
Learning and Motivation*). Sleep's role is stabilisation/enhancement by
offline reactivation (Walker, Brakefield, Hobson & Stickgold, 2003, *Nature*;
Diekelmann & Born, 2010, *Nature Reviews Neuroscience*); Walker & van der Helm
(2009, *Psychological Bulletin*) is a review/hypothesis about emotional memory
and is cited for the "sleep-dependent reprocessing" framing only. "Sleep
on/off" in RAVANA is its **offline replay/consolidation routine** (`SleepConsolidation.run_cycle` + `HippocampalReplay.sleep_cycle`)
enabled with A-replay vs ablated — an operational analogy, not biological
sleep. Full design: `docs/BRAIN_LEARN5_DESIGN.md` §6; reuse map:
`docs/LEARN5_CODEMAP.md` §2.4, §3.4; pre-registration:
`docs/BENCHMARKING_SYSTEMS_2026.md` §6.4.

**Observable criterion.** `ret_A = acc_A(after B + sleep) / acc_A(at criterion)`,
where `acc_A` is the fraction of A items with a **provenance-trusted 2-hop
causal chain** on the structured `ExplanationChain` object (the CAR headline
endpoint), never a prose regex. Transfer is structural correctness and
lexical-copy rate on a structurally identical but lexically new analogue.

## 2. Instrument and reuse (what was not reinvented)

- Reused verbatim from `scripts/measure_learn5_car.py`: `do_exposure`,
  `trusted_chain_detected`, `gate_census`, `trusted_chain_summary`,
  `canonical_hash`, `wilson`/`fisher_p` helpers, the `TRAVERSAL_LIMIT=64`
  detector. Reused verbatim from `scripts/measure_learning_curve.py`:
  `prepare_data_dir`, `label_map`, `node_ids_for`, `ask`, `answer_verdict`,
  `install_trace`, `run_detector_controls`. The trust gate functions
  (`grade_chain`, `assertable_chain`) were not modified.
- Sleep ablation uses the real `ravana/core/sleep.py:SleepConsolidation.run_cycle`
  (NREM replay, homeostatic downscaling, provenance-aware prune, REM sabotage)
  and `ravana/learn/consolidation.py:HippocampalReplay.sleep_cycle`, both
  unmodified. `time.time` inside `sleep.py` is pinned during the ablation
  (harness-only, restored in `finally`) so the REM RNG is identical in both
  arms; the only difference is whether the replayed triples name real A edges.
- **Compute matched.** ON and OFF process exactly 10 replay triples × 3
  cycles and the same `HippocampalReplay` counts. Measured wall clock:
  PRIMARY ON 0.313 s / OFF 0.309 s; SEALED ON 0.240 s / OFF 0.243 s (ratio
  1.01 and 0.99). OFF pads with out-of-range edge ids so `run_cycle` performs
  the same loop iterations but strengthens nothing.
- **Hash split frozen before boot.** `sha256("RAVANA-LEARN-5|CLI|<role>|<difficulty>|<item_id>")`,
  sorted within (role, difficulty), first 50% PRIMARY / rest SEALED; 20 A +
  18 B + 10 T items in total banks, 20 A per pool. Manifest hashes (canonical
  JSON, sorted keys):
  - PRIMARY (A) `83986e6b1cf52761e8b7511c3a06c11e8eab15617c399d3bc710e9b656de6e82`
  - SEALED (A) `0ba7c979a8452f0cf5e601cb0d0535f08cbd0a2ddb5bf8f91a075b643770cc5d`
  - B PRIMARY `b045ecf0d7a66aa78396b7b159d2eb08fcae8bb3d284c6c3a321a1bea888158b`,
    B SEALED `41a2aca4f5c0c137455c851ff0fc45c73a5e0ac2f8d5d631d3856e80fb248d49`
  - T PRIMARY `6caa887f3b062d9caa4d15b6e22e6db08fa3216357de1d33f0a513cba3bfb7c7`,
    T SEALED `c35a7b3dbc492346b758fddea57d6f7c40a3a364859455a5bc90989492d52bf2`
- **Arms.** `on` (A→B→sleep+replay), `off` (A→B→sleep, replay ablated),
  `a_only` (A + matched neutral chatter, no B), `concurrent` (interleaved A+B,
  positive control), `overlap_b` (B reuses A's a/b tokens, manipulation
  check), `fresh_f` (never taught A/B, matched transfer teaching baseline).
- **Instrument controls passed before any score:** detector positive/negative/
  direction all true; replay-strengthening control real edge 0.20→0.22 vs
  no-op 0.20→0.20; corpus audit below.

## 3. Measured numbers — primary and sealed side by side

Each cell is a fraction of A items **that reached criterion** (PRIMARY n=10,
SEALED n=6). Full per-item input/output is in
`scratch/learn5_cli_measured.json` → `arm_records` and the 1,410-row
`scratch/learn5_cli_probes.jsonl`.

| Metric | PRIMARY | SEALED | Criterion | Result |
|---|---|---|---|---|
| A reached criterion | 0.50 (10/20) | 0.30 (6/20) | ≥0.60 gate | **FAIL** |
| `ret_A` sleep-on | 0.700 (7/10) | 0.833 (5/6) | ≥0.85 | **FAIL** |
| `ret_A` sleep-off | 0.700 (7/10) | 0.833 (5/6) | — | — |
| on − off | **0.000** | **0.000** | ≥0.15 (fail <0.10) | **FAIL** |
| A-only, no B (time control) | 0.900 | 0.833 | — | — |
| Concurrent A+B (positive) | 0.875 | 1.000 | — | — |
| Overlap-B (manipulation) | 0.800 | 0.833 | — | — |
| B acquisition on / off | 0.611 / 0.611 | 0.222 / 0.222 | degrade ≤0.10 | PASS (0.000) |
| Transfer structural (trained) | 1.000 | 1.000 | ≥0.60 | PASS* |
| Transfer structural (fresh F) | 1.000 | 1.000 | — | saturation flag |
| Forward transfer (trained − F) | 0.000 | 0.000 | — | no benefit |
| Lexical copy | 0.000 | 0.000 | ≤0.10 (fail >0.15) | PASS |
| A items in B training corpus | 0 | 0 | must be 0 | PASS |
| Compute (wall clock on/off) | 0.313/0.309 s | 0.240/0.243 s | matched | PASS |

\* Transfer PASS is **not a capability result**: the fresh engine that never
taught A or B scores 1.000 on the same transfer items, so the front is
saturated (G6) and the 0.60 floor is met by prior curated knowledge, not by
learning. Flagged `transfer_saturated_invalid_g6: true`; forward transfer is
0.000. No rubric was changed to "fix" this.

**Overall: FAIL on both pools** (on−off, sleep-on fraction, criterion gate).
Hard-fail bounds were NOT tripped for retention floor (0.700 primary meets
the ≥0.70 floor exactly; sealed 0.833), lexical copy (0.000 ≤ 0.15), B
degradation (0.000 ≤ 0.15), or corpus audit (0 A items in B).

## 4. What the controls actually show

- **Interference is real but is not the sleep mechanism.** PRIMARY: A-only
  retention 0.900 → sequential A→B 0.700 (a 0.200 drop), while interleaved
  A+B is 0.875. This is the qualitative McCloskey & Cohen pattern: sequential
  teaching disrupts A, concurrent teaching largely protects it. The REM/
  prune replay ablation, however, moves retention by exactly zero on both
  pools: the replay-strengthening control proves replay *can* move an edge
  (0.20→0.22), but the trusted-chain verdict is a function of provenance and
  causal typing, so a +0.02 weight bump does not change it. The located
  bottleneck is therefore the **metric/mechanism mismatch**: replay strengthens
  weights, while the retention endpoint is a binary trust-and-typing gate.
- **Hand-checkable example (PRIMARY `frost`).** A1 chain
  `['chill','deposition','frost']` present; after A→B + sleep, the answer to
  `why does frost occur? explain the mechanism.` was *"honestly, that is a bit
  outside what i know right now…"* and the trusted chain was absent; in the
  A-only arm the same item retained the chain. `shadow` was lost after A→B
  but retained under concurrent A+B — the interleaving protection.
- **Transfer saturation example.** Fresh-engine item `flute` returned the
  trusted chain `['airflow','pitch','flute']` despite never being taught:
  prior curated knowledge, which is why trained-minus-fresh is 0.000.

## 5. Instrument repairs made after a pilot (documented, not tuned)

- A first pilot taught A with early stopping at the first successful round.
  A focused probe (`scratch/cli_reps_probe`) showed the trusted-chain census
  **fluctuates** as other items are taught (9/10 → 4/10 → 9/10 → 10/10 over
  rounds 1–4): stopping early left items that later lost their chain. The
  harness was changed to teach all A items on every round (CAR's full-round
  protocol), and the criterion gate still verifies A reached criterion before
  B. No pass threshold was changed.

## 6. Limitations

1. **Power.** n=20 A items per pool (10 / 6 reaching criterion), versus the
   design's n≥40. With only 6 SEALED criterion items the retention estimate is
   a hypothesis, not a stable result; the binding FAIL (on−off = 0) is
   nevertheless reported plainly.
2. **Low criterion rate** (0.50 / 0.30): most A items never acquire a trusted
   chain. The web teaching channel writes 0 trusted causal facts under the
   intake gate (diagnosed in CAR), so "A" is largely pre-existing curated
   knowledge. There is little that B can erase for those items.
3. **No cold reload.** Arms were measured in-process. The design asks for a
   cold reload before retest; that requires an `eng.save()` into per-arm
   scratch dirs. It was omitted to keep 12 arms within budget and to honour
   the "never save on weights" constraint conservatively. A gain that does
   not survive a cold reload cannot be excluded here — but since no gain was
   observed at all, this does not affect the FAIL direction.
4. **G6 saturation.** Transfer structural correctness is 1.000 in both the
   trained and fresh arms. The 0.60 criterion is met by prior knowledge; the
   front is flagged INVALID as a progress signal and forward transfer is
   0.000.
5. **Single seed, one run.** `PYTHONHASHSEED=0`; no second full replication.
6. **Secular graph growth / shared nodes.** Teaching all items in one engine
   grows ~hundreds of untrusted chains per subject; the detector needs the
   CAR `TRAVERSAL_LIMIT=64`. A/B token vocabularies were disjoint by
   construction, but generic graph effects (global prune/REM) are shared
   across arms by design.
7. **Commit moved during the run.** Measured at `c47ac09`; RSI (`222cdd2`)
   landed mid-run. CLI imports neither PDR nor RSI modules.

## 7. Citations (from `docs/BRAIN_LEARN5_DESIGN.md` §11)

- Wixted (2004), *Annual Review of Psychology* — retroactive interference as
  the A→B→A paradigm; consolidation reduces interference.
- McCloskey & Cohen (1989), *Psychology of Learning and Motivation* —
  catastrophic interference severe under sequential but not concurrent
  training; motivates the interleaved positive control.
- Walker, Brakefield, Hobson & Stickgold (2003), *Nature* — wake stabilises,
  sleep enhances; reactivation returns a memory to a labile state.
- Diekelmann & Born (2010), *Nature Reviews Neuroscience 11*(2):114–126,
  doi:10.1038/nrn2762 — slow-wave reactivation, system consolidation,
  generalisation.
- Walker & van der Helm (2009), *Psychological Bulletin 135*(5):649–674 —
  sleep-dependent (re)processing framing; emotional-memory review, cited for
  framing only.

## 8. Reproduction

```
cd C:/Users/Likhith/Documents/projects/ravana
PYTHONHASHSEED=0 uv run python scripts/measure_learn5_cli.py
PYTHONPATH=scripts uv run python -m pytest tests/unit/test_learn5_cli.py -q
```

Artifacts: `scratch/learn5_cli_measured.json` (manifest, controls, criteria,
12 arm records with every per-item chain and answer),
`scratch/learn5_cli_probes.jsonl` (1,410 per-item rows),
`scratch/learn5_cli/<arm>/` (isolated data dirs). Exit code 1 = FAIL.
