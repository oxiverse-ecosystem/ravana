# WFA internal-source arm — testing provenance the engine stored itself

**Status:** code shipped, fakes exercised, **no live engine run and no pytest
run in this change** (parent runs both). Every number below is a
*pre-registered prediction*, not a measurement.

## 1. Problem this arm fixes

The echo-immune WFA run (`WFA_ECHO_IMMUNE_RESULT.md`) found
`forced_choice_explicit_rate = 0.00`: RAVANA had no routine for turning a
claim + provenance into an explicit source decision. PR #481 added that
routine behind `RAVANA_SOURCE_EVAL=1`: `engine.set_source_context_for_turn(
claim, ClaimContext)` stages one decision for one turn, and the emission gate
replaces the response with Y/N/W/? on any non-ENDORSE decision.

That wiring has a **tautology hazard the module author flagged explicitly**
(`SOURCE_EVALUATION_WIRING.md` §6). If the harness constructs the
`ClaimContext` from the item's own metadata — `source_status`,
`source_id`, expected answer — then the harness already knows the answer and
hands it to a deterministic rule table. A near-1.00 score measures the rule
table, not the system: G6 saturation. The same hazard applies if the source
question itself states the credibility.

This arm removes both shortcuts:

1. The claim is **first taught into the engine's real graph** through
   `WebToGraph.learn_text(text, source_url=..., source_type='seed')`. The
   written edge carries `source_metadata` (`source`, `edge_kind`,
   `source_type`) — the engine's own stored provenance.
2. The decision-turn prompt is **the same label-free question in all three
   conditions**: "Did you learn the following claim from a credible source?
   Reply with one token: Y / N / W." It asks *about* credibility; it never
   states the claim's credibility, origin or expected answer.
3. The staged `ClaimContext` is built **only by reading graph edges back**:
   `context_from_graph_provenance()` takes the graph, the claim and the origin
   we taught from — never an item's expected answer, `source_status` or
   `source_id`. `source_id` is `is_source_id(claim, condition)`, a hash of the
   claim text and condition, so it is an experimental independent variable,
   not answer metadata.
4. The graded decision is the **engine's own audit object**
   (`engine._last_source_evaluation`, a `SourceDecision` written by the real
   gate during `process_turn`), not a decision recomputed by the harness. For
   non-ENDORSE the emitted text must equal the engine's own token.

## 2. Mechanism map (brain → engine primitive)

| Brain mechanism (evidence) | RAVANA primitive in this arm | What is deliberately NOT done |
|---|---|---|
| Source monitoring is an explicit evaluation, dissociable from item familiarity [1] | `evaluate_source(claim, context)` at the emission boundary (`engine.py` `_apply_source_monitoring_gate`), audit rule id recorded | No keyword scoring of free text; no all-tokens fallback |
| Noetic "know" vs autonoetic "remember" [2] | Claim presence in graph (item) is separate from `edge.source_metadata` origin/`edge_kind` (source). Only the latter drives trust | A bare parametric answer is never accepted as an attribution |
| Recollection vs familiarity are separable processes [3,5] | Credibility token + origin are the controlled signal; edge `confidence` is passed as `reliability`/`familiarity` and **never read** by the evaluator | No numeric strength may substitute for an explicit source |
| Source invalidation: reevaluate a just-learned mapping when the speaker proves unreliable [4] | `apply_source_invalidation()` tags the stored origin edge `source_status='discredited'`; prior ENDORSE is read from the engine's last emitted decision → rule `R_SOURCE_INVALIDATED` → W | The cue is never placed in the question prompt; it lives in stored provenance |
| Epistemic vigilance calibrates trust on the communicator [7] | `origin_id` is the stored `source`; no allow-list is supplied by the harness | Trust is not a property of claim wording |
| Left-anterior PFC monitors specific source features at retrieval [8] | A distinct evaluation step runs at articulation, after generation | Not claimed as a neural homologue (extrapolation) |

Three conditions, defined by **stored state**, not by item labels:

| condition | what the graph holds at decision time | engine decision | token |
|---|---|---|---|
| credible | verifiable origin edge (`web_fact`/`curated` + `source`) | ENDORSE | Y |
| discredited | same origin after invalidation tag, after a prior ENDORSE from the engine's own audit | WITHDRAW | W |
| unknown | no provenanced edge for that origin (source amnesia) | ABSTAIN | N |

## 3. Verified citations

Each was resolved against the publisher record, DOI record, PubMed, or the
author's publication list during this change. DOIs and page ranges are as
published, not as remembered.

1. **Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993).** Source
   monitoring. *Psychological Bulletin*, **114**(1), 3–28.
   doi:**10.1037/0033-2909.114.1.3**. Verified: APA PsycNET record
   1993-40167-001 + DOI landing + OpenAlex `W4234477066`.
2. **Tulving, E. (1985).** Memory and consciousness. *Canadian Psychology /
   Psychologie canadienne*, **26**(1), 1–12. doi:**10.1037/h0080017**.
   Verified: APA PsycNET record 1985-27196-001 + Crossref metadata
   (volume 26, issue 1, pages 1–12).
3. **Jacoby, L. L. (1991).** A process dissociation framework: separating
   automatic from intentional uses of memory. *Journal of Memory and
   Language*, **30**(5), 513–541. doi:**10.1016/0749-596X(91)90025-F**.
   Verified: APA PsycNET record 1992-07943-001 + OpenAlex `W2027380211`.
4. **Dautriche, I., Goupil, L., Smith, K., & Rabagliati, H. (2021).** Knowing
   how you know: toddlers reevaluate words learned from an unreliable
   speaker. *Open Mind*, **5**, 1–19. doi:**10.1162/opmi_a_00038**,
   PMID **34485794**. Verified: Edinburgh Research Explorer publisher PDF,
   PMC full text and PubMed. (Earlier notes calling this "Koenig et al."
   were wrong; corrected here and in `SOURCE_EVALUATION_DESIGN.md`.)
5. **Yonelinas, A. P. (2002).** The nature of recollection and familiarity: a
   review of 30 years of research. *Journal of Memory and Language*,
   **46**(3), 441–517. doi:**10.1006/jmla.2002.2864**. Verified: APA
   PsycNET record 2002-02957-001 + UC Davis author publication list.
   **Correction:** this review is in *JML*, not *Psychological Bulletin*
   128:459–493; cite the verified record.
6. **Hasher, L., Goldstein, D., & Toppino, T. (1977).** Frequency and the
   conference of referential validity. *Journal of Verbal Learning and
   Verbal Behavior*, **16**(1), 107–112.
   doi:**10.1016/S0022-5371(77)80012-1**. Verified in
   `SOURCE_EVALUATION_DESIGN.md` §3 (title is "referential **validity**").
7. **Sperber, D., Clément, F., Heintz, C., Mascaro, O., Mercier, H., Origgi,
   G., & Wilson, D. (2010).** Epistemic vigilance. *Mind & Language*,
   **25**(4), 359–393. doi:**10.1111/j.1468-0017.2010.01394.x**. Verified
   in `SOURCE_EVALUATION_DESIGN.md` §3.
8. **Ranganath, C., Johnson, M. K., & D'Esposito, M. (2000).** Left anterior
   prefrontal activation increases with demands to recall specific
   perceptual information. *Journal of Neuroscience*, **20**(22), RC108,
   1–5. doi:**10.1523/JNEUROSCI.20-22-j0005.2000**, PMID **11069977**.
   Verified in `SOURCE_EVALUATION_DESIGN.md` §3.

## 4. Support vs extrapolation ledger

| Claim made here | Status | Boundary |
|---|---|---|
| An explicit source-evaluation step is required; familiarity cannot endorse | **SUPPORTED** (direct analogue) | [1], [2], [3], [5]. The evaluator ignores `familiarity`/`reliability` by construction. |
| The engine must read its OWN stored origin, not a caller-supplied label | **SUPPORTED as method, not as brain fact** | [1], [7] motivate source-specific attribution; the specific graph-read implementation is an engineering instantiation, not a neural claim. |
| A discredited origin should retract a prior endorsement | **SUPPORTED with extrapolation** | [4] shows behavioural reevaluation in 30-month-olds, not a verbal `WITHDRAW` token. The token is a proxy; the invalidation cue is experimenter-delivered and written into graph metadata. |
| Familiar/unprovenanced claims fail exclusion | **SUPPORTED** (predictive) | [5], [6], [3]: repetition and familiarity drive false endorsement when no explicit source check exists. |
| A distinct retrieval-time evaluation step | **EXTRAPOLATION** | [8] is fMRI evidence for left-anterior-PFC monitoring; a symbolic gate is an analogy, not a homologue. |
| This arm demonstrates human-like autonomous source recollection | **NOT CLAIMED** | The rule table is deterministic and the harness teaches the facts and delivers the invalidation cue. See §6. |

## 5. Exact criteria, predictions and falsifiers

Pre-registered in `TH` (new keys only; **no existing gate or threshold was
changed**). The arm's rules live in its own additive JSON subtree
`summary['internal_source']`; the main `decide()` verdict is byte-unchanged.

| # | Criterion | Pre-registered bound | Falsifier / reading if it fails |
|---|---|---|---|
| C1 | credible → ENDORSE / Y | `correct >= 0.90` per pool | The stored provenance is not being read, or the real teach path wrote no provenanced edge. Inspect `staged_*` audit + `teached` count, do not lower the bound. |
| C2 | discredited → WITHDRAW / W | `correct >= 0.90` per pool | `prior_decision` was not taken from the engine's audit, or the invalidation tag missed the stored edges (`invalidated == 0`). |
| C3 | unknown → ABSTAIN / N | `correct >= 0.90` per pool | A claim with no stored origin is being endorsed from familiarity — the core source-amnesia defect. |
| C4 | gate invariant | `gate_ok >= 0.95` per pool | Non-ENDORSE text is not the engine's token: a downstream guard rewrote the decision, or the graded observation did not come from the engine. |
| C5 | prompt label-free | `>= 1.00` | Credibility/source leaked into the question → the harness is supplying the answer. |
| C6 | anti-tautology | `anti_tautology_ok` true per pool | Mutating item `source_status`/`expected_answer`/`source_id` changed the staged context or decision → the context came from item metadata, not the graph. |
| C7 | sample size | `n >= 20` per pool | Below this the arm is a hypothesis, not a result. |
| C8 | provenance mutation control | ENDORSE → WITHDRAW when only stored metadata flips | A decision that cannot move is measuring the harness, not the engine (`ravana-debugging` §1 mutation rule). |

Additional structural controls: the three source-question prompts are
byte-identical for a given claim; `prior_decision` for C2 is the engine's own
`_last_source_evaluation.decision`; the `RAVANA_SOURCE_EVAL` flag is restored
after the arm; a boot failure leaves the arm missing (reported, never faked).

**Predictions (untested here):** with a live engine and the flag on, C1–C6
and C8 should hold because the stored provenance is clean in all three
conditions; C7 holds for the full item bank. These are predictions about a
mechanism implementation, not capability evidence (§6).

## 6. What would be G6 saturation — and what this arm does NOT prove

If the live run gives ~1.00 on C1/C2/C3 (and `explicit_rate` ~1.00), that is
a **G6 FLAG: saturated / mechanism implementation**, not evidence that RAVANA
autonomously recollects sources. Reasons:

- `evaluate_source` is a deterministic rule table. Clean stored provenance
  fed into a clean rule table yields near-perfect agreement **by
  construction**; a near-1.00 result cannot distinguish "the engine uses its
  provenance" from "the harness wired a truth table".
- The harness still chooses what to teach, which origin to teach from, and
  delivers the invalidation cue. The human analogue is being told the source
  later; this is a proxy (ledger above), not unaided source monitoring.
- Per `ravana-debugging` ("A deterministic decision table wired into a
  harness measures the table, not the system"), report the near-1.00 with a
  G6 flag, report informativeness `0 < R < 0.999`, and **do not tune
  thresholds** to manufacture a mid-range number.

What the arm *does* establish, if C1–C8 hold: the shipped emission gate can
consume **provenance the engine itself stored**, and the decision follows
changes to that stored provenance (C8) while being invariant to item answer
metadata (C6). That is strictly stronger than PR #481's caller-supplied-label
arm, but strictly weaker than an engine that answers a source question with
no staged context at all. That stronger endpoint remains a falsifier for a
future instrument: *if a no-staged-context source question still yields ~0
explicit decisions, the missing mechanism is autonomous source attribution,
and this arm must not be cited as closing that gap.*

## 7. Unverified / parent checklist

1. No live engine/harness run in this change; all §5 predictions are
   predictions. Parent runs `uv run python scripts/measure_learn5_wfa.py`
   with `RAVANA_SOURCE_EVAL=1` set by the arm itself.
2. pytest was **not** run here (parent runs). All 59 pure test functions in
   `tests/unit/test_learn5_wfa.py` were invoked directly with fakes in one
   plain-Python process (no pytest runner, no engine) and all passed; that is
   not a pytest result.
3. The fake graph exercises the read/grade path and the REAL unbound gate
   methods; it does not exercise the real `WebToGraph.learn_text` write path
   or a cold engine boot. If `learn_text` writes no provenanced edge, C1/C2
   fail honestly rather than silently passing.
4. Real-engine runtime of ~3 source-turns per gap item may be substantial;
   `WFA_IS_MAX_PER_POOL` bounds the arm deterministically (default 0 = all).
5. Full texts of [6]–[8] were not re-read here; their verification is
   recorded in `SOURCE_EVALUATION_DESIGN.md` §3 and reused. [5]'s journal
   correction was verified via publisher/author records this change.

## 8. Reproduction

```
uv run python scripts/measure_learn5_wfa.py            # flag set/restored by the arm
uv run python scripts/measure_learn5_wfa.py --validate-only
# optional bound: WFA_IS_MAX_PER_POOL=40
```

Additive JSON only: `summary['internal_source']` and per-record
`is_<condition>_*` keys. Existing fields, gates and thresholds unchanged.
