# Tier-0 instrument gates

Implements `docs/BENCHMARKING_SURVEY.md` §5.1 for both front harnesses.
A front score is a claim about the system **and** about the probe; these gates
test the probe before the score is printed. A blocking gate FAIL suppresses
the whole score block and exits nonzero. Everything is deterministic, logged,
and rerunnable with `--gates-only`.

Implementation: `scripts/front_gates.py` (shared, no engine import),
`scripts/measure_fronts.py`, `scripts/measure_fronts_web.py`.
Gate records go to `scratch/front_probes_{n30,web}_gates.jsonl`. The per-probe
JSONL and the aggregate print format are unchanged; gates only add audit
records and a gate table.

## What each gate checks

| Gate | Check | Pass rule | Blocking? |
|---|---|---|---|
| **G1** paired rubric | Feeds one hand-labelled good and one bad string to **every** prose judge (conversation, grounded, opinion, thinking, speaking). Records both verdicts and both reasons. Thinking bad string is `"a device that stores chemical energy"`; good is a 2-clause causal answer. | `bad < good` (0 < 1) for all 5 judges | yes |
| **G2** selectivity | 10 control outputs (deletions, degenerate association template, bare definitions, unrelated text) through the same judges; compares their pass rate with the real-probe pass rate from the per-probe JSONL. | `real − control >= 0.30` **and** control rate `<= 0.05` | yes |
| **G3** symmetric guard | Thinking rubric: 3 decline-only answers and 3 assert-without-chain answers must all FAIL, 3 real-chain answers must all PASS. Code front: replay honest-decline arithmetic — "answer everything" scores 80/100, "decline everything" 20/100. | decline=0, assert=0, good=1, both code policies `< 100` | yes |
| **GC** controls | Existing hand-labelled `run_controls()` set (incl. CRITERION-FIX cases) must have zero failures. | failures = 0 | yes |
| **G4** mode parity | (a) Scoring regions of the two harnesses must be identical after stripping comments/docstrings; (b) no `RAVANA_OFFLINE` token or environment-conditional credit in scoring code; (c) every reported score must equal the score recomputed from its per-probe records; (d) OFF−ON language-learning delta must be `< 10` (the removed +20 signature). | all sub-checks pass | yes |
| **G5** census + cold reload | Per-front census from the JSONL: exactly 27 records, indices `1..27`, non-empty subjects. Source scan for the `for x in g.nodes` dict-iteration footgun and `eng.save()`. Two sequential cold engine boots: never-taught symbol `llnov01_inc` absent in both, `ConceptGraph.nodes` is a dict, all 740 values have non-empty labels, `get_node(id)` resolves. | census ok, footguns=0, both boots ok | yes |
| **G6** saturation | Flags any front `>= 95` or `<= 5` as non-informative. Non-blocking: it marks *those front rows* INVALID; the other fronts still report. | flag, not failure | no |

`--gates-only` runs the full gate set on the cached JSONL plus the two cold
boots (~15 s here) and prints no front scores. Normal runs execute G1/GC's
static parts before the engine boot and all measured gates between scoring and
printing, so no ungated number ever reaches the console or summary file.

## Measured result on `7e55cda` (n=27 — SUPERSEDED)

> **Staleness note (2026-11-19).** This section records the n=27 gate run at
> commit `7e55cda`. That is **no longer current main** (main is `6a2b5fa`,
> 68 PRs merged). Every gate measurement number below — including the 27-record
> census, the selectivity/control vectors and the front scores — is from that
> n=27 run and is retained as history. The current gated seven-front result is
> the honest **n=60** offline measurement in
> [docs/BENCHMARK_60_ITEMS.md](BENCHMARK_60_ITEMS.md) §4–§5 (G1–G6 PASS/FLAG,
> 420 records, indices 1..60 per front). The web arm at n=60 **timed out**
> (2400 s, 207 of 434 probes, no valid summary), so **no n=60 web score exists**;
> do not reuse the n=27 web figures below as current. The gate mechanics
> (G1–G6) themselves are unchanged.

Base commit `7e55cda` (historical); measured after this change. Fresh offline
run, cached web run (web scoring code is unchanged by this work). Exact numbers
from `scratch/gate_only_{offline,web}.log` and the gate JSONL.

### RAVANA_OFFLINE=1 (offline)

| Gate | Result | Measured |
|---|---|---|
| G1 | PASS | bad<good 5/5; good=5/5, bad=0/5 |
| G2 | PASS | real=0.5778, control=0.0000, selectivity=**+0.5778**, 10 controls rejected 10/10 |
| G3 | PASS | thinking decline 0/3, assert 0/3, good 3/3; code answer-all 80/100, decline-all 20/100 |
| GC | PASS | 0 control failures |
| G4 | PASS | identical scoring code, 0 env-conditional tokens, 0 bonus tokens, earned−reported = 0 on all 7 fronts, OFF−ON LL delta = 0 |
| G5 | PASS | 189 records, footguns=0, 2/2 cold boots: symbol absent, 740/740 labelled nodes, `get_node`→`psychology` |
| G6 | **FLAG** | `code generation=100` INVALID as progress signal |
| **Overall** | **PARTIAL** | 6 PASS / 0 FAIL / 1 FLAG, exit 0 |

Scores reported after gates at n=27 (SUPERSEDED): conversation 85, grounded
78, opinion 70, thinking 19, speaking 37, **code 100 (INVALID: saturated)**,
language learning 80; MEAN 67.
**Current honest n=60 offline (2026-11-19, BENCHMARK_60_ITEMS.md §4):**
conversation 93, grounded 17, opinions 87, thinking 8, speaking 7,
code 100 (saturated), language learning 80; **MEAN 56**. No rubric was relaxed
to recover the n=27 numbers.

### Web on

| Gate | Result | Measured |
|---|---|---|
| G1 | PASS | bad<good 5/5; good=5/5, bad=0/5 |
| G2 | PASS | real=0.7704, control=0.0000, selectivity=**+0.7704** |
| G3 | PASS | same vectors as offline |
| GC | PASS | 0 control failures |
| G4 | PASS | earned−reported = 0 on all 7 fronts, OFF−ON LL delta = 0 |
| G5 | PASS | same cold-reload numbers as offline |
| G6 | **FLAG** | `code generation=100` |
| **Overall** | **PARTIAL** | 6 PASS / 0 FAIL / 1 FLAG, exit 0 |

## Finding: the gate caught a stale, bonus-credited cache

Before refreshing the offline cache, `--gates-only` reported **G4 FAIL**:
`front_probes_n30_summary.json` claimed language learning = **100**, while the
per-probe records summed to **80** (`taught=40 + recall=40`, no honest bonus),
and OFF−ON LL delta was **+20**. That is exactly the removed synthetic
offline bonus, preserved in a stale summary from before commit `1753dfa`.
The live offline harness was rerun with the gated code; it now reports 80 and
the parity gate passes. No score function was changed.

## Limitations (honest scope)

- G1 exercises the instrument's **current prose rubrics**. The thinking judge
  is still a regex over generated text, not the structured causal-chain
  object the survey prefers; when that judge is replaced, G1's vectors should
  be swapped with it — the gate itself needs no change.
- G2 gates on the aggregate across five query fronts. Per-front selectivity
  in the offline run is conversation 0.85, grounded 0.78, opinion 0.70,
  thinking 0.19, speaking 0.37; thinking is below 0.30 and speaking is only
  marginally above, so those two front scores are weak evidence even though
  the aggregate passes. No rubric was loosened to raise them.
- G3 is scoped to the thinking and code fronts, where the decline/assert
  symmetry is actually measured. The conversation judge cannot credit an
  honest decline (it passes a fluent "I don't know"); this is a rubric
  limitation, recorded here rather than hidden by weakening the gate.
- G6 flags saturation; it does not fix it. The code front at 100 needs new
  held-out items before it can be used as a progress signal.

## Usage

```bash
uv run python scripts/measure_fronts.py --gates-only
uv run python scripts/measure_fronts_web.py --gates-only
uv run python scripts/measure_fronts.py          # full gated run
uv run pytest -q tests/test_front_gates.py      # 9 engine-free tests
```

Exit 0 = reportable (saturated fronts flagged INVALID); exit 1 = instrument
INVALID, no scores printed or written. "SKIP" is treated as blocking so a
gate can never pass by not running.
