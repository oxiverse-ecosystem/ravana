# WFA echo-immune source-monitoring measurement — changes, criteria, predictions

**Status:** code + fake-based unit tests only. No engine process and no pytest
run by this change's author; the parent validates the real-engine run. All
claims below are either code facts (file:line in `scripts/measure_learn5_wfa.py`)
or static re-scores of already-surviving artifacts.

**Evidence base:** `docs/investigations/WFA_FAILURE_ANALYSIS.md` (full analysis),
the 272 surviving records in `C:/Users/Likhith/Documents/projects/ravana_wfa_v3/
scratch/learn5_wfa_probes.jsonl`, and the unchanged item bank / hash seal.

---

## 0. VOID statement — previous numbers must not be cited

The earlier report that `R = 0.0154 / 0.0746` (and the accompanying `F`,
`exclusion_margin`, `I`, `E`, `I_discredited`, `E_credible` and `withdrawal =
0.1385 / 0.0448`) "proves RAVANA lacks source monitoring" was **wrong** and is
**withdrawn**. Those numbers were keyword-hit rates on the engine's quoted
reproductions of the harness's OWN prompts (`ABSTAIN`, `unreliable`, `Endorse`,
`yes`), not engine decisions. **Every R / F / exclusion-margin / withdrawal
figure produced before this change is void** — algebraically well-formed,
evidentially empty. No new R/F value is asserted here.

No threshold was tuned to produce any of the changes below.

---

## 1. What changed (four fixes, additive and gate-preserving)

All four fixes are in `scripts/measure_learn5_wfa.py`; the item bank, hash split,
old thresholds, old gates, `process_dissociation` math and every JSON field name
are preserved.

### Fix 1 — echo-immune, fail-closed parser (`strip_echoes` :1152,
`final_decision_span` :1169, `parse_decision` :1180, `parse_forced_choice` :1197,
`pd_quality_counters` :1564, hard-fail :1671)

- `strip_echoes` removes quoted frames `you mentioned:` / `i said:` /
  `earlier i said:` iteratively to a fixed point (handles nested frames and
  quoted prior prompts containing newlines), then drops bare frame markers.
- `final_decision_span` keeps only the text after the LAST `Decision:` marker.
- `parse_decision` recognises a decision ONLY when the first alphabetic token
  of the post-echo, post-`Decision:` span is an explicit label
  (`endorse`/`yes` → True, `abstain`/`no` → False; `withdraw`/`withdrawn` →
  True, `maintain` → False). **No all-tokens content fallback.** Anything else
  is `unparseable` (decision `None`, `endorse=False`, fail-closed).
- The legacy `parse_endorsement` / `parse_withdrawal` remain only for backwards
  compatibility and are no longer called by any arm.
- New additive counters (top-level AND per-pool in the summary):
  `echo_marker_rate` (+ `echo_marker_rate_inclusion/exclusion/invalidation/
  forced_choice`), `unparseable_rate`, `own_decision_rate`,
  `pd_unparseable_rate`, `forced_choice_explicit_rate`,
  `forced_choice_no_answer_rate`, `n_quality_observations`.
- **New hard-fail:** `unparseable_ceiling_primary` /
  `unparseable_ceiling_sealed` require `pd_unparseable_rate <= 0.50`
  (pre-registered `TH["unparseable_rate_max"] = 0.50`, :124). It is a stricter
  gate: no existing gate was relaxed or removed.
- Per-item `pd_inclusion_text`, `pd_exclusion_text`, `inv_after`, the parsed
  decisions, `*_unparseable` and `*_echo_marker` flags, and the forced-choice
  text/decision are persisted in both the probe JSONL and `summary["pd_audit"]`
  (:1850) for audit.

### Fix 2 — turn isolation (`EngineEndorsementResponder` :1264,
`EngineInvalidationResponder` :1305, `ForcedChoiceResponder` :1345)

- Every engine responder accepts either a zero-arg **engine factory** (one
  fresh engine per turn) or a `reset_episodic_state` **isolate callback**
  invoked once per item before its turn. `CognitiveChatEngine.reset_episodic_state`
  is documented in-engine as "a fresh engine would have these empty anyway"
  and clears the transcript / episodic index / hippocampal buffer that produced
  the `you mentioned:` echoes — so isolation is cheap, not a 500-boot penalty.
- Invalidation resets once **before teach**, never between teach and cue, so the
  within-item pairing survives while cross-turn contamination cannot.
- `main()` modes (additive, pre-registered for the falsifier):
  - default `cleared_episodic_context_per_turn`;
  - `WFA_FRESH_ENGINE=1` → `fresh_engine_per_arm_turn` (strongest form);
  - `WFA_SHARED_ENGINE=1` → `shared_engine_unisolated_AB` (old behaviour, A/B
    falsifier). Mode is recorded as `summary["pd_mode"]`.
- `echo_marker_rate` is kept **per arm** so the isolation check is auditable.

### Fix 3 — self-consistent trusted set (`trusted_sources_for` :1209,
`endorsement_prompt` :1225)

- For a credible gap the displayed allow-list starts with **that claim's own
  `source_id`**, followed by seven other real fixture URLs. For discredited /
  unknown items the own id is **never** included.
- The exclusion criterion ("endorse ONLY if provenance is in the trusted set")
  is unchanged; G2 and the fabrication/instrument controls still hold because
  `src_discredited_*` ids remain outside every allow-list.

### Fix 4 — forced-choice arm (`forced_choice_prompt` :1245,
`ForcedChoiceResponder` :1345, `WFAArms` :1458, `run_pd_arms` :1481)

- New **additive** arm: single-token prompt (Y = endorse, N = abstain,
  W = withdraw); `parse_forced_choice` reads the first token only; anything
  else is a **separate `no_answer` category**, never merged into `unparseable`.
- It is A/B-able on identical state: both response surfaces use the same
  engine/factory, same items and same self-consistent trusted set (`summary[
  "forced_choice_ab"]["same_state"] = True`); only the surface changes.
- Reported, **not gated**: it adds no new threshold and does not change PD math.
- A missing forced-choice arm is a hard fail (`arm_present_forced_choice`) in
  the same way as the four pre-existing arms.

---

## 2. Exact new criteria and counters

| Name | Definition | Where | Gate? |
|---|---|---|---|
| `echo_marker_rate[_arm]` | share of persisted arm texts containing a quoted-frame marker | `pd_quality_counters` | no (additive) |
| `unparseable_rate` | share of the three core PD arms' responses with no explicit decision | `pd_quality_counters` | no |
| `own_decision_rate` | `1 - unparseable_rate` | `pd_quality_counters` | no |
| `pd_unparseable_rate` | same as `unparseable_rate`, per pool and overall | `pd_quality_counters` | **yes** |
| `unparseable_ceiling_primary/sealed` | `pd_unparseable_rate <= 0.50` | `decide` :1671 | **yes (new, stricter)** |
| `forced_choice_explicit_rate` | share of forced-choice responses in {endorse, abstain, withdraw} | `pd_quality_counters` | no |
| `forced_choice_no_answer_rate` | share in the separate `no_answer` bucket | `pd_quality_counters` | no |
| `arm_present_forced_choice` | forced-choice responder injected | `decide` | **yes (new, stricter)** |

Unchanged: `wfa_valid_min 0.70`, `abstain_min 0.90`, `delta_min 0.20`,
`decoy_rej_min 0.85`, `internal_no_tag_min 0.90`, `sealed_drop_max 0.10`,
`PD_R_min 0.25`, `PD_exclusion_margin_min 0.10`,
`PD_exclusion_credible_min 0.70`, `PD_R_lt_one_max 0.999`,
`invalidation_withdrawal_min 0.50`, all structural minima, and
`R = I - E`, `F = E/(1-R)`, `margin = I_discredited - E`.

---

## 3. Predictions and falsifiers (from `WFA_FAILURE_ANALYSIS.md` §4)

| Fix | Pre-registered prediction | Falsifier / next step |
|---|---|---|
| 1 | Echo-driven I/E/margin collapse to 0 with `unparseable_rate > 0.90`; all 12 withdrawal flags and all 3 endorsement flags become unparseable. | If ≥10% of responses have an explicit decision outside a quoted echo, the parser (not the engine route) was the main loss and this fix restores signal. |
| 2 | `echo_marker_rate` falls from ~0.92 to <0.10 and explicit decisions appear in >0.50 of turns; R becomes a real I−E. | If echo rate stays >0.50 on a fresh/isolated engine, the mirror is the default router response to the `Claim:…/Decision:` shape, not context carryover → proceed to fix 4. |
| 3 | Under a compliant responder `E_credible` rises from ~0.015 to ≥0.70 while `E` on decoys stays low; capability parity becomes testable. | With echoes eliminated, if `E_credible` stays <0.70 the engine cannot perform the provenance lookup even with a satisfiable list: instruction-following, not list construction. |
| 4 | Explicit-response rate >0.80; withdrawal on a correctly cued item rises above 0.50 without borrowing any keyword from the cue. | If the token arm still yields <0.50 explicit decisions, the failure is upstream of articulation (no source-evaluation routine) — a mechanism work item, not a measurement item. |

---

## 4. Static validation already performed (no engine)

Re-running the new parser over the 272 surviving v3 probe records (516 arm
responses) gives: **457 with an echo marker**, **0 with an engine-authored
explicit decision**, **516/516 unparseable** (`unparseable_rate = 1.00`).
This independently reproduces fix 1's prediction and confirms the old flags
were echoes. A self-consistent-list check confirms the credible claim's own
URL is inside the trusted segment while the decoy's id is not.

Fake-based unit tests added to `tests/unit/test_learn5_wfa.py`: echo stripping
(nested, multiline), explicit-decision requirement and no-content-fallback,
unparseable counting + hard-fail ceiling, forced-choice first-token parsing and
`no_answer` separation, trusted-list consistency, turn isolation (reset once
per item, not between teach/cue; factory freshness), **oracle compliance
control** (a fake responder returning known `Decision: ENDORSE/ABSTAIN` texts
reproduces known `I=1.0, E=0.25, I_dis=0.5, E_cred=0.75, R=0.75, F=1.0,
margin=0.25` through parser + `process_dissociation` + counters), and
missing-arm hard-fail including the new forced-choice arm. All 50 test
functions in the module were invoked directly with fakes (no pytest run by
this change's author, no engine boot): 50 passed, 0 failed.

---

## 5. Unverified / parent-owned

- No live-engine run was performed by this change; the four §3 predictions
  about real `R`, `E_credible`, `echo_marker_rate` and forced-choice explicit
  rate remain for the parent to measure.
- The A/B pairs the parent should run on identical items/seed:
  `WFA_SHARED_ENGINE=1` vs default isolation (fix 2 falsifier), and the
  forced-choice vs free-text surfaces within one run (fix 4).
- Route instrumentation (analysis §5.1) is still needed to distinguish the
  mirror route from parametric completion if `echo_marker_rate` stays high.
