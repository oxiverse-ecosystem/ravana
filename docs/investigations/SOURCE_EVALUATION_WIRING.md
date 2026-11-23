# Source evaluation — engine wiring (pre-articulation source-monitoring gate)

**Status:** implemented on `feat/source-evaluation-wiring` (worktree
`C:/Users/Likhith/Documents/projects/ravana_source_wire`). The pure evaluator
lives in `source_evaluation.py` (PR #476); this change wires it into the engine
as a **general, default-off, per-turn source-monitoring gate**. No benchmark
gate, threshold, item bank or hash seal was touched. No live WFA run was done
here — the parent runs the suite and the harness.

## 1. Why

`docs/investigations/WFA_ECHO_IMMUNE_RESULT.md` measured, with a clean
single-token forced-choice surface and echoed text stripped:

| signal | value | reading |
|---|---|---|
| `pd_unparseable_rate` | 1.00 | no own decisions survived the parser |
| `forced_choice_explicit_rate` | 0.00 | zero explicit Y/N/W decisions |
| forced-choice echo marker | 0.000 | echo cannot explain the zero |

That is the pre-registered falsifier: the gap is **upstream of articulation —
there is no decision routine**. `evaluate_source()` supplies the routine; this
change makes its output the engine's emitted text at the articulation boundary.

Brain grounding and the full citation/support-vs-extrapolation ledger are in
`docs/investigations/SOURCE_EVALUATION_DESIGN.md`; they are not repeated here.

## 2. Exact hook locations

All line numbers are on this branch's HEAD (verify with `grep -n` after edits).

| File | Line | What |
|---|---|---|
| `ravana/src/ravana/chat/models.py` | 112-114 | `CognitiveResponseContext.source_context` / `.source_evaluation_decision` / `.source_evaluation_reason` added (all additive, `None`/`""` defaults) |
| `ravana/src/ravana/chat/engine.py` | 1941-1953 | `__init__` state: `_pending_source_context`, `_source_gate_ctx`, `_last_source_evaluation` |
| `ravana/src/ravana/chat/engine.py` | 8062 | **public API** `set_source_context_for_turn(claim, context=None)` |
| `ravana/src/ravana/chat/engine.py` | 8080 | `_source_monitor_decision(ctx)` — pure static core (reads env + `ctx.source_context`, calls `evaluate_source`) |
| `ravana/src/ravana/chat/engine.py` | 8100 | `_apply_source_monitoring_gate(response, ctx)` — emission-boundary helper |
| `ravana/src/ravana/chat/engine.py` | 8185-8186 | `process_turn` calls the gate on the **final** response from `_process_turn_impl`, before dialogue-memory recording and before `return` |
| `ravana/src/ravana/chat/engine.py` | 8190-8192 | `finally` clears the staged context and the ctx reference (one-turn consumption) |
| `ravana/src/ravana/chat/engine.py` | 11786 | `ctx` construction copies `_pending_source_context` into `ctx.source_context` |
| `ravana/src/ravana/chat/engine.py` | 11791 | `_process_turn_impl` publishes the live ctx on `self._source_gate_ctx` |

**Why the gate is at the `process_turn` emission boundary and not immediately
after `_generate_response`:** `_process_turn_impl` has ~66 early returns and at
least three post-generation rewriters (the stale-salad fallback, the correction
ack, `_final_emit_guard`). A token substituted at `_generate_response` can be
overwritten by any of them. Applying it to the response that `process_turn` is
about to return guarantees the decision token is what the caller sees, and
covers every reply-producing path that produced a `CognitiveResponseContext`.
This is still "the response-generation path", and the helper is invoked from
`process_turn` exactly as intended.

## 3. Environment flag and default-off guarantee

- Flag: `RAVANA_SOURCE_EVAL=1` (read at call time, no module-level cache).
- **Anything other than the exact string `"1"` (including unset) disables the
  gate.** With the flag off, or with no staged context, `_source_monitor_decision`
  returns `None` and `_apply_source_monitoring_gate` returns the response object
  **unchanged** — byte-identical to pre-change behaviour. Importing the module
  or calling the setter cannot enable it.
- The evaluator module itself remains default-off (`integration_hook`) and was
  not modified.

## 4. API and data flow

```python
# Harness / caller, immediately before the turn it wants gated:
engine.set_source_context_for_turn(claim, context)
#   claim:   str  — the claim being evaluated (supplied by caller; no wording
#                   is hard-coded in engine code)
#   context: ClaimContext from ravana.chat.source_evaluation
#   also accepts a single (claim, context) 2-tuple

response = engine.process_turn(user_input)   # gate runs here if flag on
decision = engine._last_source_evaluation    # SourceDecision or None (audit)
# ctx fields exposed on the turn's CognitiveResponseContext:
#   ctx.source_evaluation_decision  -> the SourceDecision
#   ctx.source_evaluation_reason    -> decision.reason (auditable prose)
#   ctx.source_evaluation_decision.rule -> R_* rule id
```

- `set_source_context_for_turn` only stages a value; it never touches the
  graph, store or emission path. The value is consumed by exactly one
  `process_turn` and cleared in a `finally` (including on exception) — **no
  cross-turn leakage**. A harness that wants a decision every turn must set the
  context every turn.
- Decision surface: `ENDORSE -> Y`, `ABSTAIN -> N`, `WITHDRAW -> W`,
  `UNKNOWN -> ?` (from `DECISION_TOKENS`). Per the brief, on **ENDORSE the
  generated response (the claim) is emitted unchanged**; the gate substitutes a
  token only for the other three decisions. The harness must therefore treat a
  non-`N/W/?` response under flag-on as an endorsement, or emit `Y` itself from
  the exposed decision.
- Fail-open: a malformed context, a missing ctx, or any exception in the gate
  path leaves the response untouched. A bad context can never break a turn.
- No change to CAR-chain articulation: the gate only replaces text when a
  caller-supplied context yields a non-ENDORSE decision; otherwise the chain
  narration is emitted byte-for-byte.

## 5. Predicted WFA endpoints (not measured here)

Assumes the harness sets the context each forced-choice turn (including
`prior_decision` and `invalidated` from its own bookkeeping) and reads the
emitted response. These are **predictions**, from
`SOURCE_EVALUATION_DESIGN.md` §6; no engine or harness run was performed in
this change.

| Endpoint | Pre-registered target | Predicted with gate wired | Why |
|---|---|---|---|
| `forced_choice_explicit_rate` | > 0.80 | > 0.80 (≈1.00 on well-formed items) | Each valid `(claim, context)` yields a decision; ENDORSE emits the claim, the other three emit Y/N/W. |
| `R = I − E` (primary and sealed) | ≥ 0.25 | ≥ 0.25, likely near 1.0 | Credible claims → ENDORSE; familiar discredited decoys → ABSTAIN. |
| `invalidation_withdrawal_primary/sealed` | ≥ 0.50 | ≥ 0.50 (≈1.00 when `invalidated`/`prior_decision` supplied) | Rule 2 → `WITHDRAW`/W. |
| `pd_exclusion_credible_*` (parity) | ≥ 0.70 | ≥ 0.70 | Rule is source-selective, not blanket abstention. |
| `pd_unparseable_rate` | hard-fail ceiling | ≈ 0 | Decision is a typed token; echoed text never enters the surface. |

## 6. G6 CAVEAT — saturation is a mechanism implementation, NOT capability evidence

**This is explicit and must travel with any result produced with the flag on.**

The evaluator is a deterministic rule table over provenance **supplied by the
caller**. If wiring it drives a WFA front to a near-perfect score (R ≈ 1.00,
withdrawal ≈ 1.00, `explicit_rate` ≈ 1.00), that is a **G6 saturated PASS**.
Per `ravana-debugging` ("A deterministic decision table wired into a harness
measures the table, not the system"):

- It proves an explicit decision surface exists at the articulation gate.
- It does **not** prove RAVANA autonomously recollects or attributes its
  sources. The harness hands the mechanism the answer (source status, origin,
  cue, invalidation); the mechanism echoes it back as a token.
- It must be reported as **G6 FLAG — saturated / mechanism implementation**,
  not as autonomous source recollection. Report `0 < R < 0.999`
  informativeness; do **not** tune thresholds to manufacture a mid-range number.
- To claim genuine source memory, a later instrument must withhold the label
  and require the engine to produce the attribution itself (within-engine
  inclusion/exclusion, Jacoby 1991, with a source-invalidation withdrawal
  condition). This change does not do that.

## 7. Unverified / parent checklist

1. No live engine or harness run in this change. All §5 values are predictions.
2. pytest was **not** invoked here (parent runs). The gate logic was exercised
   directly against the real class methods with fakes (flag-off, ENDORSE,
   ABSTAIN, WITHDRAW, dormant/cued, audit fields, one-turn consumption) — all
   passed — but pytest collection/integration is unverified.
3. `prior_decision` / `invalidated` bookkeeping is the harness's responsibility;
   without it, withdrawal reads as ABSTAIN and the ≥0.50 prediction will not
   hold. Wiring dependency, not a module defect.
4. Early-return paths that occur **before** `CognitiveResponseContext` is built
   have no ctx and therefore are not gated (the spec conditions on
   `ctx.source_context`). The forced-choice path reaches `_generate_response`,
   so it is covered.
5. Provenance is supplied, not discovered; cross-lingual/paraphrase robustness
   of `credibility_token` is untested (closed token set by design).
6. No benchmark gate, threshold, item bank or hash seal was changed.

## 8. Falsifiers (pre-registered)

| # | Falsifier | If observed |
|---|---|---|
| F1 | flag-off turn differs from pre-change output | Default-off guarantee broken; revert the wiring |
| F2 | `explicit_rate ≤ 0.80` with echo ≈ 0, flag on | The staged context is not reaching `ctx.source_context` |
| F3 | `R < 0.25` while explicit rate > 0.80 | Rules not source-selective; check `source_status` → `ClaimContext` |
| F4 | withdrawal < 0.50 with `invalidated=True` | `prior_decision` not tracked by harness |
| F5 | credible items endorse < 0.70 | Old trusted-list defect reintroduced |
| F6 | familiarity=1.0 + unknown provenance → ENDORSE | No-shortcut property broken |
| F7 | R ≈ 1.00 reported without a G6 FLAG | Saturation hidden; result is not capability evidence |
| F8 | a decision leaks into a later turn with no staged context | `finally` clearing broken |
