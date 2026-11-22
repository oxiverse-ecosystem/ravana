# LIVE OBSERVATION WIRING — making `--live` grade the engine, not the fake oracle

Scope: STATIC / code work. No engine process was started and no pytest suite was
run here (the parent owns the live runs and pytest). Only four owned scripts/tests
and this document changed. **No gate, threshold, item bank, sealing rule,
`decide()`, `instrument_controls()` or `gate_saturation()` was touched.**

Evidence driving this change: `docs/investigations/G6_FLAGS_ANALYSIS.md` (read in
full). Decisive finding: both `backend=fake` runs are deterministic fake-oracle
results, and the two existing `--live` paths were *also* disconnected from the
graded input:

* PDR — `RealBackend` inherited `FakeBackend.query()`, which returns
  `deepcopy(self.facts)` (synthetic). The real `Provenance` / `intake` /
  `eng.save()` / cold-reload calls were ungraded side effects.
* RSI — `RealBackend.present()` called `eng.process_turn(s)`, discarded the
  return value, and returned `super().present()` (the input demo triples).

So `--live` still graded synthetic facts and all endpoints stayed 1.00/0.00 by
construction. This revision rewires the two observation seams; it does not
change what a "pass" means.

## Exact changed functions

`scripts/measure_learn5_pdr.py`
* `RealBackend.__init__` — additive counters `engine_observation_missing`;
  `api_calls` gains a real `boot` count (keys: boot/intake/save/cold_reload/
  process_turn).
* `RealBackend._boot` — increments `api_calls["boot"]` (was untracked).
* `RealBackend.reset_item` (new) — clears the census roster pointer.
* `RealBackend.inject_poison` — records `_active_p` so the census roster is the
  authored facts for the p actually injected.
* `RealBackend._norm`, `RealBackend._roster`, `RealBackend._edge_active`,
  `RealBackend._census` (new) — project real graph edges into the existing fact
  schema: `relation_type`, `source_metadata` (`edge_kind`, `source_url` /
  `source`, `source_type`), `weight`, `confidence` and active/dormant status,
  for originals, plain poison, provenance-forged poison and sleeper
  (before/after cue).
* `RealBackend.query` (NEW OVERRIDE) — censuses `self._engine.graph`. No engine
  or unrecoverable fact => emit **no** observation and increment
  `engine_observation_missing`; **never** falls back to `FakeBackend.facts`.
* `run_durable_sweep` — additive per-row `engine_census` and
  `engine_observation_missing` fields.
* `summarize` — additive `engine_observation_missing`, `engine_response_unparsed`.
* `main` — passes the backend census counter into `summarize`.
  teach / inject / `eng.save()` / cold reload / cue are unchanged.

`scripts/measure_learn5_rsi.py`
* `_parse_engine_answer` (new module function) — parses raw `process_turn` text
  into the existing `{relation, target, verdict}` answer shape; explicit
  `relation=X target=Y verdict=Z` form or a known relation phrase + candidate
  from natural text. Returns `None` when nothing structured is recoverable.
* `FakeBackend` — additive marker fields (`engine_graded=False`,
  `last_engine_answer=None`, `last_unparsed=False`); `present()` unchanged.
* `RealBackend.__init__` / class marker — `engine_graded=True`, additive
  `engine_response_unparsed`, `engine_observation_missing` and full `api_calls`
  keys.
* `RealBackend.present` (REWIRED) — collects each `process_turn` emission, parses
  it, and stores the graded answer on `last_engine_answer`; returns the raw
  engine emissions, **not** the demo triples. Unparseable => increment
  `engine_response_unparsed` and score NO ANSWER (`verdict="no_answer"`).
* `RSIComponent.run_item` — for the **structural** arm under a live backend,
  uses the engine's graded answer directly (no world-model shortcut, no demo
  triple). Confound arms (naive/chance/counterexample) keep their deterministic
  manipulations so the controls still discriminate. Additive per-row
  `engine_graded`, `engine_responses`, `engine_answer`, `engine_unparsed`.
* `summarize` / `main` — additive `engine_observation_missing`,
  `engine_response_unparsed` in the written summary.

`tests/unit/test_learn5_pdr.py`, `tests/unit/test_learn5_rsi.py` — fake-engine
store/emission stubs injected via `backend._engine` (bypassing `_boot()`), so
these tests never import or boot `CognitiveChatEngine`. No pytest run here.

## Before → after (graded observation source)

| harness | before (graded input) | after (graded input) |
|---|---|---|
| PDR | `FakeBackend.query()` -> `deepcopy(self.facts)` synthetic list, even under `--live` | `RealBackend.query()` -> census of the cold-reloaded `engine.graph` edges, projected into the fact schema |
| RSI | `RealBackend.present()` -> `super().present()` = the input demo triples | parsed `eng.process_turn()` emissions -> `{relation, target, verdict}`; no parse -> NO ANSWER |

Both still emit the **same schema** to the unchanged grader. Additive summary
provenance fields in both: `backend`, `live`, `api_calls` (real
boot/intake/save/cold_reload/process_turn counts), `backend_error`,
`engine_observation_missing`, `engine_response_unparsed`.

## Measurable prediction (falsifiable, no gate moved)

With `backend="live"`, `live=true`, non-empty `api_calls`, `backend_error=null`:
1. At least one flagged endpoint leaves the saturated band. PDR: a real intake
   that routes the contradictory poison to the episodic buffer / revise path
   leaves no edge to census, so `forged_reject` or `sleeper_rejected` drops
   below 1.00 (a missing fact is NOT replaced by the synthetic one), or
   `focus_intr > 0.00`. RSI: near(k=1) < 1.00 < near(k=8) so
   `growth_headroom > 0`, because real emissions vary with k instead of being
   a dictionary lookup.
2. Primary and sealed diverge on at least one front, supplying the overfitting
   estimate that is currently identically zero.
3. If the live engine cannot be parsed into a structured answer, RSI rows score
   NO ANSWER and `engine_response_unparsed > 0` — an honest failure, not a
   synthetic fallback. G6 then flags/blocks the score instead of reporting a
   fake 1.00.

## Falsifier and disambiguation

If, with the live provenance fields above populated and no `backend_error`,
every endpoint is still exactly 1.00/0.00 **and** the per-probe rows are
byte-identical to the fake rows on identical state, the rewiring did not take
effect (or the capability is genuinely saturated). Do **not** report a
saturated score from that state.

Disambiguate with the **mutation control** (implemented as fake-based unit
tests here): plant one deliberately wrong engine fact/emission in a known row
and require a scored metric to move.
* PDR: omit the trusted original edge -> `retained` flips False; plant one
  poison edge as a trusted `curated` fact -> `contaminated` flips True. Both
  are asserted in `test_mutation_control_engine_fact_moves_score`.
* RSI: one wrong target in the engine emission flips `flags["correct"]` from
  True to False (`test_engine_emission_mutation_moves_grade`).
* Threshold neutrality is frozen by exact `TH` snapshots and decision-rule key
  sets in both `test_threshold_neutrality_all_old_criteria_identical` tests.

Only after the mutation controls pass does a flat 1.00 count as capability
evidence; otherwise the instrument, not the system, is the finding.

## Verification performed here (no engine, no pytest)

* `python -m py_compile` on both scripts and both test files: clean.
* Direct-call, fixture-aware runner over all 58 tests in the two files: 58/58
  pass; `ravana.chat.engine` is **not** imported (no engine boot).
* Fake oracle paths still produce the pre-rewiring endpoints: PDR focus
  `orig=1.0 / intr=0.0 / forged=1.0`; RSI `near=1.0 / far=1.0 /
  growth_headroom=0.0`.
* NOT verified here (parent owns): actual `--live` engine runs, real endpoint
  movements, primary/sealed divergence, and the real pytest invocation.
