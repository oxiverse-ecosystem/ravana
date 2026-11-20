# LEARN-5 CLI: making sleep replay restore earned A knowledge

Scope: the CLI continual-learning component (`scripts/measure_learn5_cli.py`), its
sleep implementation (`ravana/src/ravana/core/sleep.py`) and unit tests. This
document records the defect, the mechanism of the fix, the controls, and the
measured outcome. Thresholds, gates and the headline metric are unchanged.

## 1. Defect (measured baseline, commit `c47ac09`)

`scratch/learn5_cli_measured.json` on the pre-fix code:

| pool | sleep ON | sleep OFF | ON−OFF | A-only | concurrent |
|---|---|---|---|---|---|
| PRIMARY | 0.700 (7/10) | 0.700 (7/10) | 0.000 | 0.900 | 0.875 |
| SEALED  | 0.833 (5/6)  | 0.833 (5/6)  | 0.000 | 0.833 | 1.000 |

The interference effect was real (A-only / interleaved exceed sequential), so the
instrument was not blind. Replay was inert. `docs/CLI_REPLAY_CODEMAP.md` located
the cause:

1. Sleep replay only did `weight += 0.02`, `confidence += 0.01` on edges that
   already existed (`sleep.py:118-122`). It never wrote the fields the trust gate
   reads: `edge.source_metadata.edge_kind`, `edge.provenance` and
   `edge.relation_type` (`learn/explanation.py:209-210, 268-316, 331-366`).
2. replay content was derived from the **post-B** graph, so an A edge that B
   deleted or REM-flipped was absent from the replay list and could never be
   restored;
3. the `HippocampalReplay` half of the manipulation mutated only a private
   dict, and the harness used throwaway instances, so the engine's
   gate-preserving Phase 3b drain (`engine_generation.py:1305-1339`) never ran.

A second implementation wrinkle: the chat engine's `sleep_engine` is the
unrelated `ravana_grace.core.sleep.SleepConsolidation`, not
`ravana.core.sleep.SleepConsolidation`. The harness historically called the
latter directly, outside the engine.

## 2. Brain-grounded mechanism of the fix

A replayed memory is a **reactivation**, not a fresh perception. Two findings
drive the design:

- **Reconsolidation**: a reactivated trace becomes labile and is restabilised
  with its original content only when the reactivation carries the original
  context; the update is prediction-error/schema-gated, not a blanket overwrite
  (Nader, Schafe & LeDoux, 2000, *Nature* 406, 722-726; Kumaran, Hassabis &
  McClelland, 2016, *Neuron* 92, 36-49). Therefore replay must carry the
  experience's **own** write-time provenance, and the write must pass the same
  consistency gate as ingestion. Restoring a fabricated tag would be a false
  memory, not consolidation.
- **Complementary learning systems**: hippocampal replay reactivates a
  pattern-separated index and the neocortex slowly integrates it (McClelland,
  McNaughton & O'Reilly, 1995, *Psychological Review* 102, 429-457). The engine's
  Phase 3b drain is exactly that transfer; it is why the harness now drives the
  engine's sleep path rather than a detached object.

REM sabotage is left intact (counterfactual edges, emotional decay, creative
recombination, relation-type flips). Consolidation must survive interference,
not be exempted from it: only the specific edges reactivated this cycle are
shaded from the flip/prune passes for that cycle, which is the physiological
analogue of a recently reactivated trace being transiently stabilised.

## 3. What changed

### 3.1 `ravana/src/ravana/core/sleep.py`

- Replay triples may carry a 5th element: the experience's own provenance
  captured at write time (`source_metadata`, `relation_type`, `weight`,
  `confidence`). 4-tuples remain valid and behave as before.
- STAGE 1 accepts that record **only** when its `edge_kind` is one the trust gate
  already accepts (`web_fact`, `definition`, `curriculum`, `curated`). Anything
  else (e.g. `inferred`, `co_occurrence`) is refused: no edge is forged, no
  blanket tag is stamped. `EARNED_EDGE_KINDS` mirrors the gate; it does not loosen
  it.
- For an accepted record STAGE 1 restores the recorded metadata and
  `relation_type` onto the live edge (repairing a REM-flipped causal hop),
  recreates the edge if B deleted it, and restores recorded confidence when it
  drifted below the earned value. The classic `+0.02` / `+0.01` strengthening
  is preserved verbatim.
- Restored `(src, tgt)` keys are marked for the cycle and passed to the weight
  prune and the REM flip, which skip them for that cycle; the mark is cleared at
  end of cycle. No other edge is exempted.
- `BeliefStore.users` access is made defensive so this implementation can be
  driven from the real engine (`_sleep_consolidate` passes a real BeliefStore,
  which has no `.users` attribute).

### 3.2 `scripts/measure_learn5_cli.py`

- `capture_replay_memory` snapshots each A item's two causal hops **after A
  teaching and before B**, with a deep copy of the edge's own `source_metadata`.
  Replay content therefore comes from the write-time episodic store, not from
  the post-B graph.
- `order_replay_experiences` puts fading traces first (items whose chain is
  currently absent); `replay_window` supplies exactly 10 triples per cycle and
  rotates across 3 cycles. Selection depends only on current graph state, which
  is identical in both arms.
- `run_sleep` drives `eng._sleep_consolidate()` in **both** arms, so the
  gate-preserving Phase 3b drain runs in both. The engine's `sleep_engine` is
  harness-swapped to the `ravana.core.sleep` implementation for the call and
  restored immediately; no engine source file is edited. `time.time` is pinned
  during the calls (restored immediately) so the REM RNG seed is identical.
- OFF replays the **same** 10 experiences at the same 3 cycles with deliberately
  wrong (`inferred`/`semantic`) tags: identical processing and step count, no
  earned knowledge restored.
- The matched wall-clock band is tightened from ±30% to the required ±10%
  (`COMPUTE_RATIO_LO/HI = 0.90/1.10`).
- New `replay_restore_control` proves on a synthetic damaged chain that earned
  provenance restores an assertable chain while wrong tags restore neither the
  chain nor a forged deleted edge.

### 3.3 `tests/unit/test_learn5_cli.py`

Added tests for write-time capture, wrong-tag retention of ids, 10-step window
rotation, earned-vs-wrong restoration, unchanged gate thresholds/metric, and
pickle (cold-reload) survival of restored provenance. Existing tests and the
fixed thresholds are untouched.

## 4. Controls

| control | implementation | expectation |
|---|---|---|
| A-in-B-corpus audit | `audit_corpus` | 0 A tokens in B training text |
| detector positive/negative/direction | `mlc.run_detector_controls` | pass |
| replay strengthening | `replay_strengthening_control` | real edge 0.20→0.22, noop unchanged |
| earned restore | `replay_restore_control` | damaged chain becomes assertable |
| wrong tags | `replay_restore_control` + main OFF arm | no gain, no forged edge |
| compute | wall clock ON/OFF in [0.90, 1.10], steps and cycles equal | pass |
| gate integrity | `THETA_ASSERT == 0.70`, `grade_chain`/`assertable_chain` source unchanged | pass |
| metric integrity | `ret_on/ret_off/on_minus_off` from `chain_result` only | pass |
| cold reload | restored edge survives `pickle` round-trip (unit test) | pass |

Explicitly **not** done: no threshold lowered, no gate relaxed, no
`web_fact`/`curated` tag stamped onto an `inferred` hop, no change to the
retention metric, no extra compute given to one arm, no engine source file
modified.

## 5. Measured result (full run, commit `527f401`, PYTHONHASHSEED=0)

`scratch/learn5_cli_fix_full.log`, `scratch/learn5_cli_measured.json`,
elapsed 1410 s, `instrument_valid: true`, all five instrument controls pass.

| pool | sleep ON | sleep OFF | ON−OFF | criterion | A-only | compute ON/OFF |
|---|---|---|---|---|---|---|
| PRIMARY | **0.800** (8/10) | 0.700 (7/10) | **+0.100** | 10/20 = 0.500 | 0.900 | 8.398 / 7.487 s (ratio 1.122) |
| SEALED  | 0.833 (5/6)  | 0.833 (5/6)  | 0.000 | 6/20 = 0.300 | 0.833 | 7.523 / 7.601 s (ratio 0.990) |

Against the pre-fix baseline (PRIMARY 0.700/0.700, SEALED 0.833/0.833) replay
now recovers one PRIMARY item: `A_000/fog` goes from no assertable chain to the
**structurally correct** taught chain `chill → condensation → fog`, both hops
`causal`/`curated`, grade `assert`, with `edges_strengthened = 10` per cycle in
ON and OFF (matched budget). So the mechanism is real: the write-time
provenance is restored onto a REM-flipped/deleted causal hop and survives the
same REM/prune passes that previously destroyed it. The wrong-tag OFF arm,
having received the identical experiences with `inferred`/`semantic` tags,
does not recover that item.

**Overall verdict: FAIL** (harness exit 1). Honest reasons, none of which was
hidden by relaxing a gate:

- `on_minus_off` is +0.100 in PRIMARY (below the 0.15 pass bound; it does clear
  the 0.10 hard-fail floor) and 0.000 in SEALED.
- `ret_on` is 0.800 PRIMARY / 0.833 SEALED (below the 0.85 pass bound).
- `criterion_rate` is 0.500 / 0.300 (below the 0.60 gate). This is an A-teaching
  property measured *before* sleep and is unchanged by this fix.
- PRIMARY wall-clock ratio 1.122 slightly exceeds the required 1.10 band. Steps
  and cycles are exactly matched (10 / 3 in both arms); the difference comes
  from ON restoring a handful of causal edges that then survive the graph's
  prune loops while OFF's wrong-tag edges do not.

Remaining lost items are diagnosed, not glossed over: `A_015/frost` still has a
trusted two-hop chain but `graded_assert = 0` (a low-confidence hop fails the
grade), and `A_030/erosion` plus SEALED `A_025/lightning` have `raw_chains = 0`
(no causal chain survives at all). Earned-provenance restoration alone cannot
repair a grade-low confidence or recreate a hop whose node is gone; that is the
next work item, not a reason to lower `THETA_ASSERT` or stamp a tag the
experience never earned.

No threshold, gate, or metric was changed to obtain the measured +0.100.

---

## Parent decision: REVERTED

The change was measured honestly and then reverted. Primary sleep-on retention
rose from 0.700 to 0.800 (delta +0.100 from 0.000), and the wrong-tag OFF arm
restored nothing — useful evidence that restoring earned write-time
provenance, not extra processing, is the active ingredient. But the
pre-registered release rule requires on-minus-off >= 0.15 on both pools and a
wall-clock ratio within [0.9, 1.1]. Sealed stayed at 0.000, primary ratio was
1.122, and the criterion-rate gate (0.50/0.30) was unmet. Overall FAIL.

Shipping a change that violates its own compute-parity control would repeat
the exact failure mode this benchmark was built to catch. Code is reverted;
this document retains the measurement and the diagnosis of the remaining
items (A_015 grade-fail, A_030/A_025 raw_chains=0) for a properly matched
follow-up.
