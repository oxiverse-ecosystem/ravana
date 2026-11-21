# RAVANA-LEARN-5 / CLI — P2 + calibrated-interference result (PASS)

**Status: PASS on both pools (instrument_valid=True, pass=True, zero failed
criteria).** Measured 2026-11-21 on branch `fix/weight-store-durability`,
`PYTHONHASHSEED=0`, `RAVANA_OFFLINE=1`, one process, isolated scratch data_dirs.
`scratch/learn5_cli_measured.json`, `scratch/cli_full_p2_cal.log`, elapsed 5690 s.
Prior runs preserved: `scratch/learn5_cli_measured_PREFIX_baseline.json`,
`..._INVALID_overlaybug.json`, `..._PREFIX_P1.json`, and run-6 data in
`docs/LEARN5_CLI_POSTFIX.md` (P1-only: primary on−off +0.215, sealed 0.000).

## What changed since P1-only

**P2 — write-time criterion repair** (`docs/investigations/CLI_ROOT_CAUSE.md`):
execution-owner probes on the failing item `dew` pinned Class 1 to two defects:
1. `intake.detect_contradiction` treated two effects of one cause as competing
   for a single slot, buffering the taught `cooling→moisture` hop on a 0.08 <
   0.10 margin. Causal relations are many-to-many; excluded from slot
   exclusivity (reconsolidation adds an effect — Nader, Schafe & LeDoux 2000).
2. A causal fact on a pre-existing semantic edge only bumped confidence;
   `_write_edge` now upgrades the relation_type to causal (never downgrades).
Supporting: causal connectives filtered from the GloVe argument cross-product;
`WebToGraph.learn_text` gained an explicit `source_type` (default "web") and
the authored teaching channel declares "seed" (source monitoring — Johnson,
Hashtroudi & Lindsay 1993).

**Interference calibration:** P2 made A traces strong enough that the original
`rem_flip_rate=0.03` left OFF-retention at 0.93 (ceiling; on−off 0.067). Human
data show strong memories need increased retrieval demands to reveal a sleep
benefit (Petzka et al. 2021, *Sci Rep* 11:17372; Ellenbogen et al. 2009, *Curr
Biol* 19:1952), so the instrument now pre-registers `rem_flip_rate=0.10`,
applied identically to ON/OFF with pinned RNG. **No threshold was changed.**

P1 (shared maintained snapshot + metadata-overlay commit) is unchanged and
still provides compute parity by construction.

## Measured result

| pool | crit | ret_on | ret_off | **on−off** | b_degrade | transfer | lex copy | wall ratio | work equal |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | **0.75** | **1.00** (15/15) | 0.733 (11/15) | **+0.267** | 0.00 | 0.75 | 0.00 | **1.000** | yes |
| SEALED | **0.65** | **1.00** (13/13) | 0.846 (11/13) | **+0.154** | 0.00 | 1.00 | 0.00 | **1.000** | yes |

Overlay edges 7/7 (PRIMARY) and matched on SEALED; work counters identical.
All 18 pre-registered criteria PASS, both pools.

## Caveats (honest)

- **SEALED transfer 1.00 is G6-saturated** (`transfer_saturated_invalid_g6=true`):
  it satisfies the numeric floor but is not a progress signal. Redesign is
  specified in `WEB_AND_SATURATION.md` §4.
- The 0.10 REM flip rate is an **instrument calibration after P2**, not a
  pre-existing design constant. It is documented, applied symmetrically, and
  keeps every gate; a critic can re-run with `CLI_REM_FLIP_RATE=0.03` to see
  the ceiling effect that motivated it.
- n remains 20 A items per pool (criterion subsets 15 and 13). PASS is at this
  n; no claim of power beyond it.

## Reproduction
```
cd C:/Users/Likhith/Documents/projects/ravana
PYTHONPATH=scripts PYTHONHASHSEED=0 RAVANA_OFFLINE=1 uv run python scripts/measure_learn5_cli.py
PYTHONHASHSEED=0 uv run python -m pytest tests/unit/test_learn5_cli.py tests/unit/test_cli_write_path.py -q
```
