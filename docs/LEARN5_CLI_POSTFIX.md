# RAVANA-LEARN-5 / CLI — post-fix measurement (P1)

**Status: still FAIL, but a real replay gain is now demonstrated on PRIMARY at
exact compute parity.** Measured 2026-11-21 on branch `fix/weight-store-durability`,
repaired weights, `PYTHONHASHSEED=0`, `RAVANA_OFFLINE=1`, one process, isolated
scratch data_dirs. `scratch/learn5_cli_measured.json`, `scratch/cli_full_postfix6.log`,
elapsed 4195 s (12 arms). Pre-fix baseline preserved as
`scratch/learn5_cli_measured_PREFIX_baseline.json`; invalid intermediate runs as
`scratch/learn5_cli_measured_INVALID_overlaybug.json`.

## Fix applied (docs/investigations/CLI_ROOT_CAUSE.md)
P1 = **snapshot maintenance + commit-time metadata overlay**:
- `capture_replay_memory` snapshots A's hops AFTER A-teaching and BEFORE B with
  the edge's own write-time provenance (reconsolidation carries original context;
  Nader, Schafe & LeDoux 2000; CLS index reactivation, McClelland et al. 1995).
- ALL cycles' maintenance runs on ONE frozen snapshot, IDENTICALLY in ON and OFF
  (same noop triples, same neutral hippocampal buffer, same pinned REM RNG).
  The sequential ON and OFF arms **share** this maintained snapshot AND the
  write-time capture via a cache keyed by teaching config; the cached state and
  capture are deep-copied so ON's overlay cannot mutate what OFF reuses.
- Restoration is a metadata-only overlay at COMMIT: ON restores earned provenance,
  OFF applies the SAME operations with an unearned `inferred` tag the gate refuses.
  Healthy trusted hops are skipped in both arms; only damaged hops are repaired.
- Compute parity holds **by construction**: identical maintenance work, identical
  overlay counts, and each arm reports the shared maintenance cost + its own
  overlay. Pre-registered ±10% wall band and ALL thresholds are unchanged.
- Brain basis: synaptic tagging & capture (Frey & Morris 1997, Nature 385:533–536)
  + sleep downscaling with protection of reactivated synapses (Tononi & Cirelli
  2014, Neuron 81:12–34).

Three integration bugs were found and fixed by measurement, not guessed:
1. OFF overlay stripped healthy edges → ret_off collapsed to 0 (invalid control).
2. Real A triples passed into ON's `run_cycle` → 30 edges strengthened → cascade
   divergence (ratio 1.693).
3. `G_snap.__getstate__()` shares dicts by reference → ON's overlay mutated the
   cached state OFF reused → OFF overlay 0 → on−off 0 (aliasing).
Regression test: `test_shared_maintenance_cache_is_isolated_from_live_overlay`.

## Measured result

| pool | crit rate | ret_on | ret_off | **on−off** | b_degrade | transfer | lex copy | wall ratio | work equal |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | **0.70** | **0.929** (13/14) | 0.714 (10/14) | **+0.215** | 0.111 | 0.75 | 0.00 | **1.000** | **yes** |
| SEALED | **0.35** | 1.00 (7/7) | 1.00 (7/7) | 0.000 | 0.00 | 1.00 | 0.00 | **1.000** | **yes** |

Compute detail: PRIMARY overlay 5/5 edges, work counters identical; SEALED
overlay 9/9, work identical. The 1.122 wall-clock ratio that killed the earlier
provenance-restore attempt is **eliminated**.

## Failure analysis (no gate was tuned)

| criterion | PRIMARY | SEALED | why |
|---|---|---|---|
| A reached criterion ≥0.60 | PASS 0.70 | **FAIL 0.35** | write-time acquisition bottleneck (P2, below); only 7 of 20 SEALED A items form an assertable chain before sleep |
| ret_on ≥0.85 | PASS 0.929 | PASS 1.00 | — |
| on−off ≥0.15 | **PASS 0.215** | **FAIL 0.000** | SEALED criterion items are 7/7 in BOTH arms — a ceiling effect: with all retained, no replay gain can be demonstrated |
| b_degrade ≤0.10 | **FAIL 0.111** | PASS 0.00 | 0.011 over; separate-boot teaching noise, not related to the overlay (b_post is measured BEFORE sleep) |
| transfer ≥0.60, lex ≤0.10 | PASS | PASS | transfer is G6-saturated on SEALED (fresh=1.0) |
| compute matched | PASS | PASS | exact |

**Overall: FAIL.** Two independent root causes remain, both documented before
this run and neither addressable by P1:
- **P2 — write-time criterion repair** (CLI_ROOT_CAUSE.md §2): Class-1 and
  Class-2 items never form an assertable chain during A teaching (intake
  routing / typing / confidence; junk-node crowding; one-hop provenance). This
  caps SEALED at 7 criterion items and is the dominant release blocker. Needs
  execution-owner probes of the write path.
- **SEALED ceiling**: even with P2 raising the criterion set, an on−off contrast
  requires the wrong-tag control to actually lose some items; if all survive
  maintenance, the endpoint is saturated and the component is G6-invalid on
  that pool. Harder/damaging items or a lower-retention criterion set is needed,
  without relaxing the rubric.
- **b_degrade 0.111** is a 0.011 borderline miss on teaching noise; replicate
  with a fixed-seed paired design before treating it as a real B interference.

## Reproduction
```
cd C:/Users/Likhith/Documents/projects/ravana
PYTHONPATH=scripts PYTHONHASHSEED=0 RAVANA_OFFLINE=1 uv run python scripts/measure_learn5_cli.py
PYTHONHASHSEED=0 uv run python -m pytest tests/unit/test_learn5_cli.py -q   # 17 passed
```
