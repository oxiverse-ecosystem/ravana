# Online epistemic calibration — RAVANA learns how well-calibrated it is

**Capability:** RAVANA pairs the confidence it *predicted* for a turn with the
quality it *actually produced*, keeps the pairs in an online ledger, and lets
the signed error in that pairing move its own assert-gate. An overconfident
RAVANA raises its own bar and hedges more. An underconfident one lowers it. The
gate is **learned from experience**, not authored.

This is the capability the `2026-09-25T1155Z` round's feature card added
(`6ef50c49`, wired in `a7458a4d`, limitation pinned in `e3dbab62`; all three
landed on `main` via the round-0748Z/2044Z merge `4287916a`, PR #94). The
source is `ravana/src/ravana/chat/calibration.py` (317 lines, class
`EpistemicCalibrator` opens at `calibration.py:124`).

Every number below is copied from real executed output on this repo at
`RAVANA_OFFLINE=1` with the `.venv-real` interpreter, against the tree at
`95c85690`. Nothing here is inferred.

---

## 1. What was missing

RAVANA computed **both halves** of a calibration pair every single turn and then
discarded them separately:

| Half | What it is | Where |
|------|-----------|-------|
| the **prediction** | a per-turn confidence scalar, set once and used to route the turn | `engine.py:9721` (`confidence`), consumed by `dual_process.decide_route` |
| the **outcome** | a per-turn realized response-quality score | `engine.py`, `_assess_response_quality`, scored just before the wiring point |

A prediction you never score is not a prediction. The code said so itself, in a
comment that stood in the engine until this change:

> No `theta_withhold` modulation is wired because no prediction-vs-quality
> pair exists in the codebase to drive it; this makes the signal observable
> for future calibration work instead of leaving it dead.

`calibration.py` is that missing pair.

## 2. The three artefacts it maintains

All are pure numeric state. `grep -oE '"[a-z][^"]{45,}"'` over
`calibration.py` returns **0 hits** — no reply text, no topic table, no
authored string of any kind.

### 2.1 Signed bias — *am I systematically over- or under-confident?*

`mean(predicted − observed)` over the window (`calibration.py:184`). Positive
means overconfident.

**Measured, 60 turns predicting 0.9 and delivering 0.3:**

```
{"n": 60, "mean_bias": 0.6, "bias": 0.5143, "brier": 0.36,
 "theta_withhold": 0.8143, "bands_visited": 1, "bands_total": 5,
 "calibrated": false}
```

**The mirror case**, 60 turns predicting 0.2 and delivering 0.8:

```
bias = -0.5143    theta 0.3 -> 0.0500
```

### 2.2 Reliability curve — *am I well-calibrated on EASY and bad on HARD?*

Observed hit-rate per confidence band (`calibration.py:228`). Bands start
**empty** and accumulate as RAVANA meets confidence levels it has not seen
before. A band with no observations returns `None`, not `0.0` — an unvisited
band is *absent knowledge*, and collapsing the two is how a reliability diagram
turns into a lie.

**Measured from the live engine, 20 conversational turns** (`what is gravity`,
`tell me about the sea`, `who was ada lovelace`, …):

```
after 20 live turns: {"n": 10, "total_observed": 10, "mean_bias": 0.0695,
 "bias": 0.0348, "brier": 0.0401, "theta_withhold": 0.3348,
 "bands_visited": 3, "bands_total": 5, "calibrated": true}

reliability: [null,
              {"lo":0.2,"hi":0.4,"n":1,"observed":0.680},
              {"lo":0.4,"hi":0.6,"n":3,"observed":0.545},
              {"lo":0.6,"hi":0.8,"n":6,"observed":0.514},
              null]
```

That curve is a real, and unflattering, finding: RAVANA's confidence runs
**above** what it delivers in every band it has visited (0.68 / 0.55 / 0.51
observed against 0.2–0.8 nominal). Two bands are still `null` because 20 turns
of general conversation did not visit them.

### 2.3 Adaptive assert-gate — the payoff

`theta_withhold` (`calibration.py:215`) is the threshold the ACC uses to decide
whether a claim may be stated flatly or must be hedged. The module default is
`THETA_WITHDHOLD = 0.30` (`metacognition.py:29`); the ledger shifts it by the
learned bias, clamped to `[THETA_MIN 0.05, THETA_MAX 0.85]`
(`calibration.py:92-93`).

**Measured through the real `Metacognition` object**, 60 overconfident turns
(`learned theta = 0.8143`):

| support count | confidence | cold assert | overconfident assert | underconfident assert |
|---------------|-----------|-------------|----------------------|------------------------|
| 1.0 | 0.721 | True | **False** | True |
| 1.5 | 0.813 | True | **False** | True |
| 2.0 | 0.893 | True | True | True |
| 2.5 | 0.965 | True | True | True |

The gate bites in the mid-confidence band and correctly stands aside when
support saturates `fok_confidence` to 1.0 — a claim RAVANA is *certain* of is
still asserted by an overconfident engine, because a miscalibrated gate should
change hedging, not manufacture doubt. That is the intended behaviour, and it
is worth stating plainly rather than hiding behind the headline number.

**Convergence, overconfident stream, `BIAS_GAIN = 1.0`:**

```
n= 1  raw=0.600  shrunk=0.0545  theta=0.3545
n= 6  raw=0.600  shrunk=0.2250  theta=0.5250
n=16  raw=0.600  shrunk=0.3692  theta=0.6692
n=31  raw=0.600  shrunk=0.4537  theta=0.7537
n=56  raw=0.600  shrunk=0.5091  theta=0.8091
```

The raw bias is flat at 0.600 for every `n`. The **shrunk** bias climbs and
saturates — see §3.

## 3. Shrinkage, not a raw mean

`bias()` (`calibration.py:193`) multiplies the raw mean by `n / (n + prior_n)`.
A single turn cannot swing the gate; that is why it is stable turn-to-turn
instead of jittery. `prior_n` is a named, overridable constructor argument — a
modelling choice, not a value tuned against any test.

**Measured: one maximally-wrong turn** (predict 1.0, deliver 0.0):

```
raw mean_bias = 1.000   shrunk bias = 0.0909   theta 0.3 -> 0.3909
```

A raw implementation would have slammed the gate to its 0.85 ceiling on turn
one. The shrunk one moves it by **+0.0909**.

## 4. Where it is wired

| Site | Line | What it does |
|------|------|--------------|
| `engine.py:1818` | construction | `self.calibrator = EpistemicCalibrator()`, one per engine |
| `engine.py:10163` | the pair | `self.calibrator.observe(float(confidence), float(quality_score))` — wrapped in `try/except` so calibration can never break a turn |
| `response_gen.py:3641` | the payoff | the metacognitive realizer reads `self.calibrator.theta_withhold()` instead of the module default |
| `engine_web_search.py:546` | the payoff | the numeric-claim honesty gate refreshes `_mc.theta_withhold` from the ledger each turn |
| `engine.py:10796` | persistence | `epistemic_calibration` in the save payload |
| `engine.py:11491` | persistence | restored via `load_state` on load |

`confidence` is deliberately **not** recomputed at the observe site — the
recorded prediction is exactly the number the dual-process router acted on.

**Measured end-to-end round trip through the engine's own state:**

```
engine-state round trip ok: True | n: 10 | theta equal: True
```

`load_state` **fails closed** on unusable input rather than partially
restoring: a half-restored curve would shift the assert-gate on evidence that
was never observed. Measured: `load_state({"pairs": "junk"})` → `False`.

**NaN is dropped, not absorbed** (`calibration.py:163`) — one NaN must not
freeze the gate at a wrong value forever. Measured: `n` stays `0` after
`observe(0.5, nan)`.

## 5. Design properties, verified

- **No retraining, no LLM, no rebuild.** Every update is an incremental fold
  over one new observation. The capability is live on the very next utterance.
- **Cold start is the identity.** With no observations `bias() == 0.0`, so
  `theta_withhold(base) == base`. Measured: a fresh calibrator reports
  `theta_withhold 0.3 == theta_base 0.3`, all-None curve, `calibrated: false`.
  Wiring this in is **behaviour-preserving** until real experience arrives.
  `test_cold_start_is_not_claimed_calibrated` pins that it will not claim
  calibration off a thin sample either — `is_calibrated` requires `n >= 5`
  (`calibration.py:256`).
- **Fails safe.** No calibrator ⇒ the module default. A raised exception at
  the observe site ⇒ the ledger is unchanged ⇒ the gate stays at base.
- **No fabrication.** RAVANA never invents a prediction for a turn it did not
  make one on. `test_no_observation_is_invented_for_a_turn_with_no_prediction`
  pins that the count cannot drift on its own.

## 6. Coverage and honest limits

**Test coverage: 43 tests, green.** `pytest tests/unit/test_epistemic_calibration.py
tests/test_epistemic_calibration_wiring.py` → **43 passed in 125.19s**.

**A coverage gap this docs round found and closed.** Grading
`docs/ACCEPTANCE_LEDGER.md` surfaced that `chat/metacognition.py` — the module
whose gate the calibrator moves — had **no dedicated test file**:
`tests/test_epistemic_calibration_wiring.py` was its only importer, and that
suite was written for the *calibrator*. So `fok_confidence` and
`modality_from_support`, the two functions that decide how confident RAVANA
claims to be and how it hedges, were untested in their own right. Added
`tests/unit/test_metacognition_acc.py` (**20 tests, green in 17.84s**) and
upgraded the ledger row YELLOW → GREEN.

Those 20 tests are **sabotage-verified** — a gate that has never failed is not a
gate. Five real behavioural breaks were applied to `metacognition.py` in turn,
each confirmed to have landed on disk before the run, and each turned the suite
red:

| Sabotage | Test that caught it |
|----------|---------------------|
| gate comparison inverted (`>=` → `<`) | `test_should_assert_is_inclusive_at_the_gate` |
| retrieval-success credit removed (0.5 → 0.0) | `test_retrieval_success_raises_confidence_at_every_support_count` |
| FOK curve inverted (support lowers confidence) | `test_fok_confidence_is_monotonic_in_support` |
| `Metacognition.read` ignores its own `theta_withhold` | `test_metacognition_honours_a_learned_gate_end_to_end` |
| recent-verdict buffer no longer bounded | `test_metacognition_keeps_a_bounded_recent_verdict_buffer` |

The source file was restored byte-identical afterwards (asserted in the
sabotage script, which keeps a `shutil.copy2` backup and cleans it up).

### Measured limitations
- **Observation coverage is partial, by construction.** `process_turn` has
  **60 `return` statements before** the confidence prediction is assigned
  (measured by AST walk: 62 returns total, `confidence` at relative line 2864,
  `process_turn` opening at `engine.py:6858`, prediction at `engine.py:9721`).
  On those turns RAVANA makes no prediction, so there is nothing to score —
  and inventing one would fabricate the very signal calibration rests on. The
  measured consequence: 20 conversational turns yielded **10** observations.
  `test_observation_coverage_is_partial_because_predictions_are_partial` pins
  this so the number cannot silently drift.
- **This is response-quality calibration, not truth calibration.** The outcome
  side is RAVANA's own realized response quality, not external ground truth.
  What the module learns is *"does my confidence track the quality of what I
  actually produce?"* — a real and useful turn-level signal, and strictly
  narrower than per-claim truth calibration. The module docstring keeps that
  distinction explicit rather than letting the name overclaim.
- **The ledger is not yet exposed on any CLI or status surface.** `grep` over
  `scripts/` and `engine_status.py` finds no reader, so `get_status()` is
  currently reachable only from in-process code. The wiring that persists it
  is tested; the observability surface is not built. Filed, not bundled.
- **`BIAS_GAIN = 1.0` is a modelling choice, not a fitted value.** 1.0 is the
  identity correction: shift the gate by exactly the miscalibration you
  measured. Any other value would apply a partial correction with no
  principled reason for the fraction. The constants are named and overridable
  so the trade-off is visible rather than buried in an expression.
- **The bounds are structural, not fitted.** `THETA_MIN`/`THETA_MAX` exist so
  the gate cannot degenerate into asserting everything or abstaining
  everything — that would not be calibration, it would be a policy.

## 7. Source map

| What | Where |
|------|-------|
| the module | `ravana/src/ravana/chat/calibration.py` |
| class + constructor | `calibration.py:124` |
| the online update | `calibration.py:152` (`observe`) |
| signed bias | `calibration.py:184` (`mean_bias`), `calibration.py:193` (`bias`) |
| Brier score | `calibration.py:204` |
| the adaptive gate | `calibration.py:215` (`theta_withhold`) |
| reliability curve | `calibration.py:228` (`reliability`) |
| observability snapshot | `calibration.py:260` (`get_status`) |
| persistence | `calibration.py:277` / `calibration.py:289` |
| the ACC it moves | `ravana/src/ravana/chat/metacognition.py:29`, `:65`, `:85` |
| engine construction | `engine.py:1818` |
| the prediction/outcome pair | `engine.py:10163` |
| realizer payoff | `response_gen.py:3641` |
| web-honesty payoff | `engine_web_search.py:546` |
| save / load | `engine.py:10796` / `engine.py:11491` |
| tests | `tests/unit/test_epistemic_calibration.py` (28), `tests/test_epistemic_calibration_wiring.py` (15) |

## 8. Brain basis

Confidence is not a reading off a scale; it is a *prediction* that a claim
will survive contact with the world, and predictions can be scored. Two
well-replicated findings motivate the design:

- **Calibration is measurable and it varies.** Elicited-confidence subjects
  produce a genuine reliability curve — P(correct) against stated confidence —
  not a constant (Fischhoff & Brewer 1995; Kerste 2014). The **hard–easy
  effect** (Kornell 2009) is the sharpest demonstration: the same person is
  well calibrated on easy items and badly overconfident on hard ones. That is
  why this is a per-band curve and not one global number.
- **Feedback moves the curve**, and the improvement is largest exactly where
  the curve was worst — which is what the assert-gate shift is for.
