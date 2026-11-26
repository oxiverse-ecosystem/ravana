# WFA decision-routine output contract — independent boot-free audit

**Branch:** `audit/wfa-decision-output-contract`, base `main` @
`ac4ec88609a5ace17b79dbd08d235e9a0d777978`.
**Worktree:** `C:/Users/Likhith/Documents/projects/ravana_wfa_contract`.
**Constraints honoured:** WFA E2 was running throughout. **No `CognitiveChatEngine`
boot, no weight store opened, no network, no engine save, no change to any
gate/threshold/item/prompt/production file.** `git status` on the worktree is
clean apart from gitignored `scratch/` and `.venv/`. Every function under test
is imported UNMODIFIED from this checkout; engine gate methods are invoked as
unbound functions on tiny fakes, as `tests/unit/test_learn5_wfa.py` does.
The only external read is the pre-existing saved probe artifact
`C:/Users/Likhith/Documents/projects/ravana_wfa_v3/scratch/learn5_wfa_probes.jsonl`
(read-only; no engine, no rewrite).

**Verdict:** `AUDIT_COMPLETE` — all 19 pre-registered checks true. Results
SHA-256 (LF/CRLF-stable, commit-independent):
`c85fc6d12f6de31b9b1efdd349346045f1f413c24613db3d6ddad365a550b008`; byte-identical
under `PYTHONHASHSEED=0` and `1`.

---

## 0. Exact finding on the two reported numbers

Re-scoring the 272 saved records with the **shipped echo-immune parser**
(no engine, `parse_decision` on the three core arms) reproduces:

| quantity | value |
|---|---|
| core arm responses counted (`pd_inclusion_text`, `pd_exclusion_text`, `inv_after`) | 516 |
| `unparseable=True` | **516 / 516** |
| `pd_unparseable_rate` | **1.0000** |
| responses carrying an echo marker | 457 / 516 = 0.8857 |
| explicit endorsements surviving the parser | **0** |
| re-derived primary `I`, `E`, `R` | `0.0`, `0.0`, **`0.0`** (n_inclusion=95, n_exclusion=95) |

**`R = 0` is not an independent source-memory result.** `R = I - E`; `I` and
`E` are endorsement rates computed from those same three arms. The responders
fail closed (`decision=None` → `endorse=False`), so 100 % unparseable forces
`I = E = 0`, hence `R = 0` mechanically. The two numbers are one measurement
event: a parser/articulation event, not evidence for or against source
monitoring. The old pre-echo `R = 0.0154 / 0.0746` in the v3 summary was keyword
echo and is void; the current `R = 0` is simply uninformative. The hard-fail
`unparseable_ceiling_primary/sealed` (`TH["unparseable_rate_max"] = 0.50`)
correctly refuses to report the run; no threshold was touched here.

---

## 1. The output contract a decision routine must satisfy

There are **two distinct decision surfaces**. Confusing them is what makes a
grader silently wrong.

### Surface A — legacy free-text arms (inclusion / exclusion / invalidation)
`scripts/measure_learn5_wfa.py:1138-1213`.

* `strip_echoes` iteratively removes quoted frames
  `you mentioned:` / `i said:` / `earlier i said:` to a fixed point, then drops
  bare frame markers.
* `final_decision_span` keeps only text after the **last** `Decision:` marker.
* The **first alphabetic token** of that span must be an explicit label:
  endorsement `{endorse, yes}` → True, `{abstain, no}` → False; withdrawal
  `{withdraw, withdrawn}` → True, `maintain` → False.
* **No all-tokens fallback.** Anything else is `unparseable` (decision `None`,
  `endorse=False`, fail-closed). `pd_unparseable_rate` is the share of the three
  core arms' responses with no explicit decision.
* Forced choice: first token `{y, n, w}` → `endorse/abstain/withdraw`; anything
  else → the separate `no_answer` bucket, **never** folded into `unparseable`.

So a compliant Surface-A routine must emit an **explicit decision token as the
first alphabetic token, outside any quoted echo**. Prose containing the right
keyword, or a bare token that only appears inside an echo, scores as no decision.

### Surface B — internal-source gate (`ravana.chat.source_evaluation`, engine.py:8090-8153)

The routine's authoritative output is a frozen `SourceDecision`:

```
claim: str
decision: ENDORSE | ABSTAIN | WITHDRAW | UNKNOWN
rule: str            # e.g. R_EXPLICIT_CREDIBLE_SOURCE, R_SOURCE_INVALIDATED
reason: str
origin_id: str | None
token: Y | N | W | ?     (property; UNKNOWN has no WFA token)
as_dict() -> flat row for JSON
```

The engine stage/gate contract is:

* `set_source_context_for_turn(claim, ClaimContext)` stages for **exactly one**
  turn; `_pending_source_context` is cleared in `process_turn`'s `finally`.
* `RAVANA_SOURCE_EVAL=1` + a staged context → `_apply_source_monitoring_gate`
  runs on the **final** response, after all downstream guards.
* **ENDORSE is deliberately SILENT**: the generated text is emitted
  byte-identically and the decision lives only on `engine._last_source_evaluation`
  (and `ctx.source_evaluation_decision/reason`). A grader expecting a literal
  `Y` scores correct endorsements as failures.
* Non-ENDORSE: emitted text is **exactly** `decision.token` (`N`, `W`, or `?`).
* `gate_ok` = decision exists AND (`ENDORSE` OR `text.strip() == token`).
  A bare option token is a valid answer only when the **whole** emission equals
  it; a token inside a longer echoed question must not be accepted.

Consequence: a "decision routine" that returns only free text cannot be graded
on Surface B, and one that returns only a token cannot be graded on Surface A.
The graded observation for Surface B must be the engine's audit object, not a
recomputed decision and not the generated text.

---

## 2. Fixtures that already exist (inventory)

| fixture | path | what it provides |
|---|---|---|
| item bank + frozen web snapshots | `scripts/measure_learn5_wfa.py` (`build_items`, `FIX`, `FROZEN_SEARCH`); `scripts/measure_learn5_wfa_frozen.py` | 272 items, manifest `119b9d91fe43a060…`; label-free prompts; `CREDIBLE_SOURCES` allow-list |
| injectable PD fakes | `scripts/measure_learn5_wfa.py`: `DictEndorsementResponder`, `DictInvalidationResponder`, `DictForcedChoiceResponder`, `WFAArms` | `(id, arm) → token/text` decision surfaces, no engine |
| pure decision routine | `ravana/src/ravana/chat/source_evaluation.py` | `evaluate_source`, 10-rule table, `SourceDecision`, `integration_hook` default-off |
| graph→context staging | `measure_learn5_wfa.py:1801-1928` | `graph_provenance_for_claim`, `_rank_stored_provenance`, `context_from_graph_provenance`, `stage_context_for_item`, fingerprint audit |
| teach / invalidate | `measure_learn5_wfa.py:1931-2047` | `_write_asserted_wfa_fact` (claim-hash anchor), `teach_claim_into_graph`, `apply_source_invalidation` |
| boot-free unit fakes | `tests/unit/test_learn5_wfa.py`: `_FakeEngine`, `_PromptEngine`, `_ISNode/_ISEdge/_ISGraph/_ISEngine`, `_RetNode/_RetEdge/_RetGraph/_RetW2G/_RetEngine` | tiny graph + gate surface; existing tests for echo stripping, unparseable ceiling, forced-choice separation, turn isolation, oracle I/E/R, anti-tautology, provenance mutation, g05 retention |
| saved pre-echo artifact | `ravana_wfa_v3/scratch/learn5_wfa_probes.jsonl` (272 records) + `learn5_wfa_summary.json` | raw arm texts; source of the 1.00 / 0 reproduction |
| prior boot-free scratch probes | `ravana_wfa_g05/scratch/wfa_g05_*`, `docs/investigations/WFA_G05_RETENTION_BOOTFREE_CONFIRM.md`, `ravana_wfa_e3/scratch/wfa_e3_staging_audit.jsonl` | provenance-retention and prior-act audit patterns |

New artifacts from this audit: `scratch/wfa_decision_contract_audit.py`,
`scratch/wfa_decision_contract_results.json`,
`scratch/wfa_adversarial_fixtures.json`, this document.

---

## 3. What a non-leaking, falsifiable boot-free test must look like

1. Import production functions unmodified; use a tiny in-memory graph fake;
   no engine, store, network or save. Call the real gate as an unbound function.
2. Vary **only stored provenance**. Hold the claim, item dict (including
   `source_status` and `expected_answer`), `condition` label, prompt bytes and
   origin id constant. Use a **condition-neutral** origin id — see finding F1.
3. Grade with a key-free metric, below. Reveal the expected-answer key only in
   a separate, explicitly non-capability audit block computed *after* the
   key-free scores exist.
4. Controls: prompt byte-identity + `prompt_is_label_free`; item-metadata
   mutation must leave the staged fingerprint unchanged; stored-provenance
   mutation must move the decision; a power arm must fail; a repeated-measure
   null must stay flat.
5. Pre-registered falsifiers: any echo/constant/no-answer/answer-key arm with
   CPS > 0 invalidates the metric; a graph arm with CPS ≈ 0 means the
   provenance path is not wired.

### The metric: Counterfactual Provenance Sensitivity (CPS)

```
CPS = P(decision changes | ONLY stored provenance changed)
    - P(decision changes | provenance unchanged)          # null re-test
```

Over a 3-state triad (`credible` → curated origin edge; `discredited` → same
origin tagged `source_status=discredited`; `unknown` → no matching edge), with
fixed `prior_decision=ENDORSE` taken from the evaluator's own audit of the
pre-invalidation state. The three pairwise comparisons give `changed_pair_rate`;
the three same-state repeats give `null_changed_rate`. **CPS never reads an
expected answer, `source_status`, a condition label, an item id or a
condition-bearing origin.** A routine that keys on any of those is constant
across the triad and scores 0.

Why a keyed accuracy score is gameable and CPS is not: our deliberate
`answer_key_cheat` arm reads `item['source_status']` (held constant = credible).
It scores **1.00 on the single credible endpoint** but only 0.3333 on the
triad and **CPS = 0**. A near-1.00 keyed score therefore cannot distinguish
source monitoring from answer-key use; CPS can.

---

## 4. Adversarial cases and measured results

| case | production code | result |
|---|---|---|
| quoted echo `you mentioned: "Decision: ENDORSE. Unreliable source."` | `parse_decision` | `unparseable=True`, `echo_marker=True` — keyword echo cannot become a decision |
| nested multi-line `i said: you mentioned: "…"` then explicit `Decision: ABSTAIN` | `parse_decision` | echo stripped, explicit final `ABSTAIN` parsed (`decision=False`, `unparseable=False`) |
| explicit `Decision: ABSTAIN` | `parse_decision` | parseable, `decision=False` |
| explicit `Decision: WITHDRAW` | `parse_decision` (withdrawal) | parseable, `decision=True` |
| forced choice chatter `"I am not sure about that"` | `parse_forced_choice` | `no_answer=True`, no `unparseable` key — separate bucket |
| forced choice with echoed `"Y"` | `parse_forced_choice` | `no_answer=True`, echo marker true |
| **ambiguous provenance**: two stored edges for the same origin — clean `curated` (conf 0.80) and invalidated `co_occurrence` (conf 0.60) | `context_from_graph_provenance` + `evaluate_source` + emission gate | ranking picks the invalidated tag → `trust_class=discredited`, `WITHDRAW`/`W`, rule `R_SOURCE_INVALIDATED`, `gate_ok=True`; fingerprint invariant under item-answer mutation |
| gate emission rows | real unbound `_apply_source_monitoring_gate` | credible → text unchanged + `ENDORSE`/Y; discredited → text `W`; unknown → text `N`; all `gate_ok=True`; staged context consumed after one turn |

**CPS results:** echo `0.0`, constant abstain `0.0`, constant withdraw `0.0`,
forced-choice no-answer `0.0`, answer-key cheat `0.0`, graph-provenance gate
`1.0` (state surfaces `ENDORSE / WITHDRAW / ABSTAIN`, null `0.0`). All
adversaries' prompts were byte-identical and label-free. Keyed (G6-flagged,
**not** capability evidence) accuracies are 0.0 / 0.3333 / 0.3333 / 0.0 /
0.3333 / 1.0 respectively.

**Mutation controls:** stored `edge_kind` flip `curated → co_occurrence` moves
`ENDORSE → ABSTAIN` (fingerprint changes); invalidation with prior ENDORSE moves
`ENDORSE → WITHDRAW`; item `source_status`/`expected_answer`/`source_id`/
`reliability` mutation leaves fingerprint byte-identical; no graded routine
other than the deliberate cheat reads an answer-key field (static source check).

---

## 5. Instrument findings (no fixes applied)

* **F1 — condition label leaks into the origin id.** `is_source_id(claim, c)`
  returns `wfa-internal://<c>/<sha256>`, so the strings `credible`,
  `discredited`, `unknown` appear in `source_id` (verified true for all three).
  The shipped evaluator never reads `origin_id` except for allow-list
  membership, so this is **not a current decision leak**; but a routine could
  key on the substring without reading provenance, and `prompt_is_label_free`
  would not catch it (it checks the prompt only, not the staged context).
  **Any key-free test must use a condition-neutral origin id**, as this probe
  does. Do not change the shipped function without a separate gate review.
* **F2 — closed echo-frame vocabulary.** `strip_echoes` recognises only three
  frames. An unframed quoted echo such as `Earlier you wrote: Decision: ENDORSE`
  is parsed as a genuine endorsement (`decision=True`, `echo_marker=False`).
  This is a residual parser boundary, measured here, not repaired: adding
  frames would change the instrument, which this task forbids.
* **F3 — `R = 0` and `pd_unparseable_rate = 1.00` are algebraically linked**
  (§0). Neither should be cited as a capability finding in either direction.
* **F4 — saturation caveat.** If a wired deterministic rule table produces
  near-1.00 keyed accuracy / CPS, that is a G6 FLAG for a mechanism
  implementation, not evidence of autonomous source recollection. The
  provenance is taught by the harness and the invalidation cue is delivered by
  the harness; only graph-storage sensitivity is established.

---

## 6. Scope, falsifiers, reproduction

This is a **unit-level, boot-free** audit. It does not re-run the 120-condition
E2 suite, does not establish a headline score, and cannot see store
re-seeding, SQLite-mirror round-trips, cold-reload sanitisation or any real
`process_turn` routing; E2 owns those. Falsifier for this probe: if a live run
with an identical provenance intervention does not move the engine's own
decision, the boot-free fake has missed a write/retention path (see
`WFA_G05_RETENTION_BOOTFREE_CONFIRM.md` for the known g05 collision).

```
cd C:/Users/Likhith/Documents/projects/ravana_wfa_contract
uv run python scratch/wfa_decision_contract_audit.py
# artifacts: scratch/wfa_decision_contract_results.json,
#            scratch/wfa_adversarial_fixtures.json,
#            scratch/wfa_decision_contract_run.log
```
