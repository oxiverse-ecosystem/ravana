# G6 FLAGS ANALYSIS — why PDR and RSI still pass at 1.00

Scope: STATIC / evidence analysis only. No engine process was started, no pytest
was run, no source or threshold was changed. Sources inspected: the two on-disk
summaries/logs from the 2026-11-22 01:17 IST runs, all per-probe rows, the two
harness scripts (PR #462, commit `53c6979`), `scripts/front_gates.py`,
`docs/investigations/WEB_AND_SATURATION.md` §4,
`docs/investigations/PDR_RSI_UNSATURATED.md`, and
`docs/investigations/UNSATURATED_COMPONENTS_REAL_RUN.md` (#463).

## Verdict (up front)

Both PASSes are **deterministic fake-oracle results**, not live-engine results.
The live engine was **not** booted in either run; `eng.save()`/cold reload and
conversational teaching were **simulated in memory**, and — more seriously —
the graders in both harnesses never consume engine output even in `--live`
mode. The 1.00/0.00 endpoints are **algebraically pinned by construction**:
the hard cases are constructed so the deterministic grader's own rules always
fire. Nothing in these artifacts is evidence about whether the real RAVANA
system has zero headroom on poison resistance or rule induction.

G6 is doing exactly its job: it flags a non-informative front. It is
non-blocking by design (`front_gates.gate_saturation`, `blocking=False`), which
is why PASS + FLAG coexist.

---

## 1. What the artifacts say about backend and real-path execution

| fact | PDR | RSI |
|---|---|---|
| summary `backend` | `"fake"` (`learn5_pdr_summary.json:99`) | `"fake"` (`learn5_rsi_summary.json:255`) |
| summary `live` | `false` (:167) | `false` (:256) |
| `api_calls` | `{}` (:168) | `{}` (:257) |
| `elapsed_s` | **0.2** (:201) | **0.5** (:262) |
| log banner | `backend=fake` (log :2) | `backend=fake` (log :2) |
| probe rows with `backend="fake"` | **984 / 984** | **11520 / 11520** |
| per-row `reloaded`/`save_count` | integer counters 1..N, from `FakeBackend.save_and_reload()` | n/a |
| live scratch store `scratch/pdr_live` / `scratch/rsi_live` | **does not exist** | **does not exist** |
| WFA comparison run (same evening) | — | `learn5_wfa_summary.json` `elapsed_s=225.2`, consistent with a real engine run |

Selection is explicit in `main()`: `live = "--live" in argv`;
`backend = RealBackend() if live else FakeBackend()`
(`measure_learn5_pdr.py:989,993`; `measure_learn5_rsi.py:800,808`). No `--live`
token appears in either run's log or summary. A 0.2 s / 0.5 s elapsed time for
328 / 11520 graded rows is also incompatible with engine boots.

`PDR_RSI_UNSATURATED.md` §3 says this in plain language: with the deterministic
oracle every endpoint is 1.00; "The `--live` runs are the actual engine
measurement ... Until the live runs are executed, the 1.00 numbers are the
deterministic oracle result, explicitly labelled as such in each summary
(`backend: "fake"`)."

**Documentation discrepancy.** `UNSATURATED_COMPONENTS_REAL_RUN.md` (#463) is
titled "Real-engine runs of the unsaturated WFA / PDR / RSI" and concludes that
"the current system has zero headroom" on PDR. For WFA the 225 s runtime is
consistent with a real run; for PDR/RSI the cited artifacts carry
`backend="fake"`, `live=false`. The PDR/RSI sections of that doc therefore
report oracle results under a "real-engine" heading. Its "zero headroom" claim
is an **inference**, not a measurement.

### 1a. PDR — why each endpoint is exactly 1.00/0.00

`FakeBackend` (`:387-467`) writes every fact into a Python list:
original as `kind="curated"`, `trusted=True`, `obj=claim`; poison snippets as
`kind ∈ {co_occurrence, web_fact, definition}`, `trusted=False`, with a
guaranteed `wrong_token` and a `detail` string that is **never equal** to the
trusted claim. `save_and_reload()` is `copy.deepcopy(self.facts)`; the sleeper
cue flips `active=False → True`.

The grader (`PDRGrader.believed`, `:594-617`) is the whole "component under
test":

* **plain poison** has `kind="co_occurrence"`, which is not in
  `FakeBackend.VERIFIABLE` → rejected at `:603-604` on the tag alone.
* **forged_web_fact** and **reactivated sleeper** have verifiable kind and a
  real URL, but the slot-exclusivity rule at `:609-615` rejects any
  non-trusted fact with the same `(subject, relation)` and a different `obj`.
  Because `make_variant()` (`:286-329`) *guarantees* `detail != claim`, the
  contradiction always fires. Hence `forged_reject = 1.00` and
  `sleeper_rejected = 1.00` **for every item**; `focus_intr = 0.00` follows.
* **sleeper_dormant = 1.00**: write-stage facts carry `active=False`, skipped
  at `:598`.
* **orig_retention / orig_0 / orig_5 / durable = 1.00**: the trusted original
  is admitted unconditionally at `:600-601` and survives the deep-copy.
* **src = 1.00**: `attribute_poison()` (`:634-663`) scores the target detail's
  own token overlap against its own snippet; it is its own maximum. The
  label-shuffled control yields 0.125/0.1667 rather than chance-from-engine.
* **sense = 1.00**: `validate_items()` (`:708-715`) rejects any item whose
  context does not select the correct sense by margin ≥ 2, so
  `select_sense()` cannot fail.
* **Sealed ≈ primary**: the hash split only partitions the same deterministic
  construction; there is nothing to overfit to.

### 1b. RSI — why every condition and every k is exactly 1.00/0.00

`FakeBackend.present()` (`:386-388`) returns the **input demo triples
unchanged**. `RSIComponent.induce()` (`:471-487`) canonicalises those surface
strings and `run_item()` (`:489-543`) does a lookup in the authored
`world_model`, then `grade()` (`:562-576`) compares the resulting structured
answer to the authored gold. No engine emission enters this path.

* **near / far = 1.00 at every k**: one positive demo already yields the
  canonical relation; `world_model[(rel, subject)]` contains exactly one gold
  target. `far` uses a different domain but the same authored table. k = 1, 2,
  4, 8 differ only in how many demos are fed, so **all k are 1.00** and
  `growth_headroom = near(k=8) − near(k=1) = 0.000` **by construction**.
* **foil = 1.00 reject**: foil subject is reserved and its candidate list has
  no world-model target for the induced relation → zero targets → `"reject"`.
* **negative_reject = 1.00**: negative demos use `"not " + forward phrase`
  (`:457-462`). That string is not in the relation vocabulary, so
  `canonicalise()` returns `None`, `induce()` returns `None`, zero targets →
  `"reject"`. The structural arm never has to decide *not* to generalise from
  a counterexample; it simply cannot parse the negation. The counterexample
  confound arm sets `strip_negation=True`, recovers the phrase, induces the
  rule and asserts it → `counterexample_reject = 0.00`. That control proves
  only that stripping the literal prefix "not " changes the parser's output —
  it does not test the engine's negative-rule behaviour.
* **chance ≈ 0.1667/0.20**: a hash-shuffled relation label cannot match gold.
* **C1–C8 all pass** because they are assertions over the same deterministic
  construction (including hand-built grader vectors in C8), not over engine
  behaviour.

### 1c. Even the `--live` paths would not change the graded input

Static code reading, not a run:

* PDR `RealBackend.teach_original/inject_poison/save_and_reload`
  (`:528-573`) call `super()` first — i.e. they maintain the fake fact list —
  and *also* write to the engine. But `query()` is inherited unchanged from
  `FakeBackend` (`:466-467`) and returns `deepcopy(self.facts)`. The grader
  therefore always grades the **synthetic** fact list; the real
  `Provenance`/`intake`/`save`/cold-reload calls are ungraded side effects.
* RSI `RealBackend.present()` (`:422-431`) calls `eng.process_turn(s)` and
  discards its return value, then `return super().present(...)`, i.e. the
  input triples. Engine output cannot affect any endpoint.

So "run --live" alone is necessary to establish that the engine was exercised,
but on this code it is **not sufficient** to create headroom.

---

## 2. Genuine saturation vs instrument that never reached the hard cases

It is the latter. The evidence supports three nested claims:

1. **No live measurement exists for PDR/RSI.** Backend fields, empty
   `api_calls`, sub-second runtimes, all-fake probe rows, and absent live
   scratch stores agree.
2. **The oracle's "hard" cases are trivially decidable by its own rules.**
   PDR poison is guaranteed to contradict the trusted claim on the same slot;
   RSI's negations are out of vocabulary and its answers are dictionary
   lookups independent of k. The instruments cannot produce a wrong answer on
   their current item banks.
3. **The real system's capability is therefore UNKNOWN, not "saturated."**
   There is no observation in which the engine's poison rejection, sleeper
   reactivation, save/reload durability, or rule induction from demonstrations
   was measured and found perfect.

Examples of the specific "hard cases too easy" pattern: forged poison is
admitted by provenance in the fake model but is trivially detected because its
`detail` string differs from the claim; the sleeper is never truly active in
the engine — a boolean is flipped in a Python dict; negative-rule
counterexamples are rejected because the parser does not recognise the phrase,
not because a learner declined to generalise.

---

## 3. Minimal change to give headroom — no threshold or gate moved

**Step 0 (zero code change, do this first):** run the two existing
`--live` commands on isolated scratch state and record `backend`, `live`,
`api_calls`, `backend_error`. This settles whether the engine can be exercised
at all. Predicted from static reading: endpoints stay 1.00 because the grader
still reads the synthetic observation (§1c).

**Minimal change that actually creates headroom:** reconnect the *graded
observation* to the engine, using the already-present `--live` backend. One
seam per harness; **no change to `TH`, `decide()`, `instrument_controls()`,
`gate_saturation()`, item banks, sealing, or any threshold/gate**.

* PDR: give `RealBackend` its own `query()` that censuses the reloaded
  engine's actual store (nodes/edges with real `edge_kind`, `source_url`,
  provenance and active/dormant record) and projects it into the existing fact
  schema. Stop using the inherited synthetic `FakeBackend.facts` for graded
  observations. Keep teach / inject / `eng.save()` / cold reload / cue exactly
  as they are.
* RSI: make `RealBackend.present()` grade the engine's response to each
  demonstration instead of returning the input triples. Parse the emitted
  text into the existing `{relation, target, verdict}` answer shape; when no
  structured answer is recoverable, record `engine_response_unparsed` and
  score that row as no-answer (a legitimate failure), never as the demo
  triple.

**Measurable prediction:** with `backend="live"`, `live=true`, `api_calls`
non-empty and `backend_error=null`, at least one flagged endpoint leaves the
saturated band — e.g. PDR `forged_reject < 1.00` or `focus_intr > 0.00` (real
intake routes `co_occurrence`/`web_fact` edges and may not persist an inactive
sleeper), or RSI `near(k=1) < 1.00 < near(k=8)` so `growth_headroom > 0`.
Primary/sealed scores should also diverge on at least one front, supplying the
overfitting estimate that is currently identically zero.

**Falsifier:** if, with the live provenance fields above and no engine errors,
every endpoint remains exactly 1.00/0.00 **and** per-probe rows are
byte-identical to the fake rows on identical state, then either (a) the
observation rewiring did not take effect — instrument it at `query()`/`present()`
input and output and discard the result — or (b) the capability is genuinely
saturated. Disambiguate with a **mutation control**: plant one deliberately
wrong engine fact/emission in a known row and require the score to move. If the
score does not move, the grader is still disconnected. Only after the mutation
control passes does a flat 1.00 count as capability evidence.

**Alternative, if rewiring is ruled out:** make the oracle's hard items
non-trivially decidable *before* looking at outcomes — e.g. a forged PDR fact
that does not share the exact `(subject, definition, different obj)`
contradiction signature, and RSI negation demos using a predicate the
structural arm would otherwise induce. Prediction: oracle endpoints drop below
1.00; falsifier: they remain 1.00, meaning the oracle still short-circuits.
This is smaller in code but changes the construct being measured; the
observation-rewiring option is preferred because it tests the system rather
than a harder puzzle for the oracle.

---

## 4. Evidence vs inference, and the settling probe

**Evidence (read directly from files):**
backend/live/api_calls/elapsed fields in both summaries; both log banners;
984/984 and 11520/11520 probe rows labelled `fake`; missing `scratch/pdr_live`
and `scratch/rsi_live`; the `FakeBackend`/`RealBackend` code paths quoted in
§1a–1c; `PDR_RSI_UNSATURATED.md` §3's explicit "until live runs are executed"
note; the 225.2 s WFA runtime contrasting with 0.2 s / 0.5 s here.

**Inference (not measured; do not cite as fact):**
that a live engine would score below 1.00 on any endpoint; that real intake
would route poison to the episodic buffer or reject it by consistency; that
RSI growth would become positive under conversational teaching; the #463 doc's
"zero headroom on the current system" wording; the claim that the system
"saturates" these capabilities.

**Probe that settles it (no source edits required for the first two):**
1. Run `uv run PYTHONHASHSEED=0 python scripts/measure_learn5_pdr.py --live`
   and the RSI equivalent on isolated scratch state. Verify in the written
   summaries: `backend=="live"`, `live==true`, `api_calls` showing
   PDR `intake≥1 / save≥1 / cold_reload≥1 / process_turn≥1` and RSI
   `boot≥1 / process_turn≥1`, `backend_error==null`, and creation of the live
   scratch directories. This settles "was the engine exercised."
2. A/B the fake and live runs on identical state and diff the per-probe rows.
   If they are byte-identical, the engine is decorative and §1c is confirmed.
3. Hand-audit 5–10 rows: print each query, the raw `process_turn` response,
   and the graded verdict; check by hand that a pass means the intended
   property (per `ravana-debugging` §1 / `probe-validation.md`).
4. After the §3 rewiring, run the mutation control and an ungated-positive
   control. If the metric still cannot move, the instrument — not the system —
   is the finding.

If all four come back clean and endpoints remain 1.00, then and only then is
"genuinely saturated" an evidence-backed statement, and the next step is a new
harder instrument (with its own pre-registered thresholds), not a change to
the existing gates.

## Files created / modified

* Created: `docs/investigations/G6_FLAGS_ANALYSIS.md` (this document).
* Modified: none. No threshold, gate, harness, or weight store was touched.
