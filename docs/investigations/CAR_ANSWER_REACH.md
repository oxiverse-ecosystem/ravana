# CAR answer reach — why a learned trusted chain is built but not said

**Status:** final. **Method:** static source reading + per-item audit + verified
literature. No engine run, no pytest, no weight/store mutation. The only repo
artifact owned by this task is this file.

**Evidence base (measured, not inferred):**
- `scratch/learn5_car_measured.json` — commit `131ddf5`, 2026-11-21T15:07:51Z,
  120 records, `pass: true`, all 10 criteria PASS. Primary Δ curve
  `[0, 0.9, 0.9, 1.0, 1.0]`, sealed `[0, 0.9, 0.9, 0.9, 0.9]`.
  Answer endpoints at k=8: primary `immediate_answer_reach = 0.30`,
  `identical_surface_echo = 0.50`; sealed `0.10` / `0.00`.
- `scratch/car_full_stratified.log` — trace of that same run (manifest commit
  `131ddf5`): `POST Q: 'how is cough produced?' -> A: "i don't really have a
  solid grasp on cough so far ..." | chain=True` (`:12777`).
- Current source at HEAD `812b072` (branch `docs/learn5-final-verdict`); note
  the context brief said `main`, but the checked-out branch is not `main`.
  Commit `527f401` (`mechanism_intent`, limit 64) and the `915ef7b` root-cause
  fixes are in this HEAD. No source line cited below was changed by this task.

---

## 1. The decisive measurement: the chain *is* assembled, then thrown away

`measure_learning_curve.py:203-220` wraps `ResponseGenMixin._why_how_causal_chain`;
`:222-231` resets the hit slot per question; `:233-244` computes

```
texts     = [hit.chain[0] for hit in why_hits if hit.chain]
generated = bool(texts)
reached   = any(t in answer for t in texts)
```

So `generated=True` is **not** a regex guess: it is the literal string returned
by the chain assembler. Auditing every k=8 subject item in the measured JSON:

| pool | item | immediate | `chain_present` | `generated` | assembler's `chain_texts[0]` | emitted answer |
|---|---|---|---|---|---|---|
| primary | sneeze | yes | true | true | `irritation leads to reflex, and reflex leads to sneeze` | same string (reach) |
| primary | cough | yes | true | true | `tickling leads to expulsion, and expulsion leads to cough` | `i don't really have a solid grasp on cough so far ...` |
| primary | ivory | yes | true | true | `material leads to tusk, and tusk leads to ivory` | `i don't really have a solid grasp on ivory so far ...` |
| primary | hide | yes | true | true | `tanning leads to leather, and leather leads to hide` | same string (reach) |
| primary | coral | yes | true | true | `secretion leads to reef, and reef leads to coral` | same string (reach) |
| primary | fossil | yes | true | true | `burial leads to mineralization, and mineralization leads to fossil` | `honestly, Fossil is a bit outside what i know ...` |
| sealed | eclipse | yes | true | true | `darkness leads to shadow, and shadow leads to eclipse` | same string (reach) |
| sealed | fog / yawn / snore / leaf / boil / basalt / flood / erosion / quartz | yes | true | true | correct `a→b→x` narration | metacognitive uncertainty |

**All 10 primary and all 10 sealed k=8 subject items have `chain_present=true`
and `generated=true`.** The graph detector and the answer-time assembler both
find and narrate the learned chain. The emitted answer differs from the
assembler's own return value on 7/10 primary and 9/10 sealed items. The loss is
therefore **post-assembly**, not acquisition, not enumeration, and not the trust
grade. This supersedes the earlier codemap claim (§1 of
`docs/CAR_ANSWER_REACH_CODEMAP.md`) that `reached` is necessarily true whenever
the assembler fires — that was true at the older commit, not at `131ddf5`.

## 2. Located drop point, with file:line evidence

Question → emitted answer, in order:

1. `response_gen.py:233-250` `mechanism_intent()` admits `why|how|what causes|
   what makes|explain|mechanism|effects?` and returns `backward` for
   `how is X produced?` (the k=8 immediate form) and `what is the mechanism
   behind X?` (the cold form). No intent rejection: the assembler fires on
   every item.
2. `response_gen.py:8508-8603` `_why_how_causal_chain()` resolves seeds
   (`:8560-8578`), enumerates at `limit=64` (`:8579-8582`), grades with the
   **unchanged** `grade_chain` / `assertable_chain` predicate (`:8583-8585`),
   builds the narration `a leads to b, and b leads to x` (`:8589-8592`), and
   returns the tuple `(narr, "gated_explanation_chain")` (`:8603`). The measured
   `chain_texts` above come from exactly this line.
3. `response_gen.py:8661-8664` `_generate_response()` calls that assembler at
   the top and **returns it verbatim** on success. `engine.py:11733` receives
   `response, strategy = self._generate_response(ctx)`.
4. `engine.py:12086` `register_controller.compose(...)`; `engine.py:12091`
   `response = self._forward_model_check(response, ctx, strategy)` — this runs
   on the chain text with `strategy == "gated_explanation_chain"`.
5. `response_gen.py:10111-10290` `_forward_model_check()` **has no exemption for
   `gated_explanation_chain`**. Its explicit strategy exemptions are
   `counterfactual_simulation` (`:10134-10135`), `seeded_relation`
   (`:10141-10142`), `emotional_empathy` (`:10150-10151`), and
   `web_direct_answer`/`web_unverified` plus web-prefixed text (`:10211-10216`).
   A chain narration falls through to the degeneracy strip at `:10233`.
6. `response_gen.py:9997-10031` `_strip_degenerate_clauses()` sees one clause
   (the narration has a comma but no `.!?`), evaluates it with
   `_is_word_salad_any_sentence(..., grain="clause")` (`:10024`) and
   `_sm_response_grounded(ctx, c, skip_step1=True)` (`:10051-10052`). If either
   fails it reports `_dropped=True`; with nothing kept, `_forward_model_check`
   returns `self._human_like_uncertainty(ctx)[0]` at **`:10245`**.
7. `response_gen.py:4770-5076` `_sm_response_grounded()`: with `skip_step1=True`
   the graph-anchor requirement is waived (`:4838-4839`), but the per-sentence
   REFERENCE check still demands that the clause touch `assoc_set`, built only
   from `ctx.associated_concepts[:8]` after a content-word filter
   (`:4949-4952`) and `defn_words` from the stored definition (`:4953-4956`).
   Failing both returns `False` at `:5076`. **The graph chain that the assembler
   just verified is never offered as evidence to this check** — the monitor
   asks "does a known lexical associate or the definition appear in the text?",
   not "is this text the retrieved trusted chain?".
8. `response_gen.py:7350-7543` `_human_like_uncertainty()` emits the exact
   observed strings at **`:7434-7436`**:
   `i don't really have a solid grasp on {X} so far.` /
   `hmm, i'm not totally sure about {X} yet, and that's okay.` /
   `honestly, {X} is a bit outside what i know right now.` (with the trailing
   curiosity clauses appended). For the cold paraphrase the grounded subject is
   the wh-cue phrase (`mechanism behind X`), which `_human_like_uncertainty`
   reduces to `mechanism` at `:7391-7401` — hence the measured cold answers
   that say "i don't have a solid grasp on **mechanism**" while the assembler
   already produced the correct chain about X.
9. `engine.py:12903-12926` `_final_emit_guard()` *does* exempt this strategy
   (`:12924-12925`: `if strategy == "gated_explanation_chain" ...: return text`).
   It never gets the chance: step 6 already replaced the text. **The two
   articulation monitors disagree about what counts as retrieved, validated
   knowledge.**

The item-level pattern supports this reading: the four subjects that reached
the answer at k=8 (sneeze→*reflex*, hide→*leather*, coral→*reef*,
eclipse→*shadow*) are exactly the ones whose middle concept is a common stored
associate/definition word of the subject; the failures (cough→*expulsion*,
ivory→*tusk*, fog→*condensation*, reverb→*repetition*, fossil→*mineralization*,
…) are chains whose verified middle hop is not in the lexical-associate/
definition evidence set the reference check consults. This is an inference
from the item split, not a direct observation of `assoc_set`; the falsifier in
F1 tests it directly.

## 3. Discriminating the five proposed causes

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 1 | Intent classification still rejects held-out paraphrases | **Not supported** at the measured endpoint | `mechanism_intent` at `response_gen.py:245-250` admits both `how is X produced?` and `what is the mechanism behind X?`; every k=8 item has `generated=true`, i.e. the assembler ran and returned a narration. (The wh-cue phrase still damages the *grounded subject* used to name the uncertainty — step 8 — but it does not stop the chain.) |
| 2 | Enumeration finds the chain but answer construction doesn't use it | **Partly right, mislocated** | The consumer of `enumerate_chains` is `_why_how_causal_chain` (`response_gen.py:8579-8603`), and it *does* use the chain — it builds and returns the string. `_generate_response` returns it verbatim (`:8662-8664`). The constructed answer is discarded **downstream** by `_forward_model_check` (`response_gen.py:10245`). Making it emit = make the articulation monitor accept already-verified retrieved structure (F1/F2). |
| 3 | Confidence/grade gates reject at answer time while accepting at graph time | **Not supported** | `grade_chain` and `assertable_chain` are called once in the assembler (`response_gen.py:8583-8585`) with unchanged thresholds (`explanation.py:47 THETA_ASSERT=0.70`); they accepted — the narration was built. The rejecting stage is a lexical grounding monitor (`_sm_response_grounded`), not a chain confidence/adequacy gate. No threshold moved between the graph endpoint and the answer endpoint. |
| 4 | Generic parametric answer path wins because the learned candidate scores lower | **Not supported** | `_ok[0]` is the graded learned chain (`:8588`); there is no competing parametric candidate in `_generate_response` after it returns at `:8664`. The uncertainty string is not an answer that out-scored the chain — it is a post-hoc replacement produced at `engine.py:12091` → `response_gen.py:10245`, *after* the chain was the response. |
| 5 | `identical_surface_echo = 0.50` shows parroting instead of composing | **Supported as a separate measurement artifact, not as the cause of reach** | The echo metric is `a in answer and b in answer` on the IDENTICAL question (`measure_learn5_car.py:495-501,540-543`); that question contains the taught surface, so the assertion mirror returns the user's words. It never measures chain composition and never feeds `immediate_answer_reach`. The primary 0.50 / sealed 0.00 split says only that half the primary items echoed in that form. |

**Net root cause:** `gated_explanation_chain` is trusted at the graph gate *and*
at the assembler *and* at the final emit guard, but not at the intermediate
pre-articulation monitor, which re-litigates the composed utterance against
lexical associates and the definition store instead of against the verified
retrieval that produced it. The system can retrieve the relation and can
assemble the sentence; the monitor refuses to let the sentence be articulated.

## 4. Brain mechanism — knowledge in memory but not expressed in language

All citations below were resolved to publisher / PubMed / PsycNET records
(author, year, journal, volume/pages, DOI). Theory/review status is flagged.

### 4.1 Semantic retrieval into a working-memory message before speaking

**Levelt, W. J. M., Roelofs, A., & Meyer, A. S. (1999).** A theory of lexical
access in speech production. *Behavioral and Brain Sciences, 22*(1), 1–38.
doi:10.1017/S0140525X99001776 — verified (PsycNET 1999-13199-001; MPG copy).
Speaking begins with conceptual preparation — activating and selecting a
lexical concept from semantic memory — then lemma selection, then phonological
encoding. A correct conceptual message can still fail to be verbalized if it
is not selected as the message to formulate.

**Baddeley, A. (2000).** The episodic buffer: a new component of working memory?
*Trends in Cognitive Sciences, 4*(11), 417–423.
doi:10.1016/S1364-6613(00)01538-2 — verified (Cell Press full text;
PsycNET 2006-00305-004). Review/theory: a limited-capacity buffer binds
information from subsidiary systems **and from long-term memory** into a
temporary unitary representation available for utterance planning.

**RAVANA mapping:** `_why_how_causal_chain` successfully binds the retrieved
`a→b→x` relation into a message (`response_gen.py:8589-8592`), but the
pre-articulation monitor (`_forward_model_check`) judges that message against
`ctx.associated_concepts`/definitions rather than against the bound retrieval.
This is a failure at the "retrieved structure → selected message" step, not a
failure of semantic storage. F1 makes the retrieved chain part of the monitor's
evidence so the buffer's content is admissible to production.

### 4.2 Hippocampal–neocortical replay makes new relations retrievable by the slow system

**McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995).** Why there are
complementary learning systems in the hippocampus and neocortex: Insights from
the successes and failures of connectionist models of learning and memory.
*Psychological Review, 102*(3), 419–457. doi:10.1037/0033-295X.102.3.419 —
verified (PsycNET doiLanding; PubMed 7624455). Theory/model: fast hippocampal
one-shot storage plus slow neocortical statistical learning; **reinstatement /
replay** is what makes a new trace progressively available to neocortical
retrieval without catastrophic interference.

**Kumaran, D., & McClelland, J. L. (2012).** Generalization through the recurrent
interaction of episodic memories: A model of the hippocampal system.
*Psychological Review, 119*(3), 573–616. doi:10.1037/a0028681 — verified (MPI
Pure record; publisher PDF). Model: recurrent interaction between episodic
traces and neocortical structure supports generalization; replay is not just
copying but recurrent re-activation that makes relations usable.

**Kumaran, D., Hassabis, D., & McClelland, J. L. (2016).** What learning systems
do intelligent agents need? Complementary learning systems theory updated.
*Trends in Cognitive Sciences, 20*(7), 512–534. doi:10.1016/j.tics.2016.05.004 —
verified (PubMed 27315762; Cell Press). Review/update: replay gives
goal-dependent weighting to experience statistics; schema-consistent material
can be acquired rapidly by the slow system.

**RAVANA mapping:** the taught `edge_kind='curated'` chain is the fast one-shot
trace; the PMI/co-occurrence graph and `ctx.associated_concepts` are the slow
statistical store. The graph endpoint (Δ=1.0) shows the fast trace survives and
can be traversed. The emission monitor, however, validates the utterance only
against slow-store lexical associations/definitions — so a newly integrated
fast-store relation is systematically inadmissible to language until it happens
to resemble an existing associate. That is the CLS prediction for exactly this
gap: presence in the fast store ≠ availability to the slow retrieval/expression
path without replay-mediated integration.

### 4.3 Schema assimilation

**Tse, D., Langston, R. F., Kakeyama, M., Bethus, I., Spooner, P. A., Wood, E. R.,
Witter, M. P., & Morris, R. G. M. (2007).** Schemas and memory consolidation.
*Science, 316*(5821), 76–82. doi:10.1126/science.1135935 — verified (Science
publisher page; PubMed 17412951). Once a neocortical schema exists, a
schema-consistent paired associate is assimilated in **one trial** and becomes
rapidly hippocampal-independent; schemas play a causal role in lasting
associative memory.

**RAVANA mapping:** the authored curriculum scaffold is the schema. It should
let a new `a→b→x` relation be assimilated and *expressed* on the first taught
occurrence. Instead the monitor demands that the new relation already overlap
the lexical associate/definition set — the opposite of one-trial assimilation.
Schema congruence here must supply evidence for the emission path, not become
an extra precondition it fails. (Schema congruence is also a source of false
memory, so the fix must not let the schema fabricate a link — the
`grade_chain`/`assertable_chain` gate remains the only admissions path.)

### 4.4 Knowing versus saying, and competitor suppression

**Brown, R., & McNeill, D. (1966).** The "tip of the tongue" phenomenon.
*Journal of Verbal Learning and Verbal Behavior, 5*(4), 325–337.
doi:10.1016/S0022-5371(66)80040-3 — verified (PsycNET 1966-12901-001).
Empirical: participants can know the target word (partial phonological/semantic
knowledge) yet fail to produce it — the knowing/saying dissociation this task
exhibits in machine form.

**Anderson, M. C., & Spellman, B. A. (1995).** On the status of inhibitory
mechanisms in cognition: Memory retrieval as a model case. *Psychological
Review, 102*(1), 68–100. doi:10.1037/0033-295X.102.1.68 — verified
(PsycNET 1995-16174-001; PMID 7878163). Theory + experiment: retrieving a
target suppresses competitors that share cues; retrieval is conceptually
focused selection, and classical associative interference is insufficient.
Boundary conditions (cue independence) are debated, so treat this as the
selective account, not an unconditional law.

**RAVANA mapping:** the failure is a tip-of-the-tongue state with an inverted
cause — the chain is retrieved and selected (`_ok[0]`) but withheld at
articulation, and the answer reports uncertainty about `mechanism`/X. The
analogue of competitor suppression is the monitor's evidence selection: the
lexical associates (`ctx.associated_concepts`) and definition win the reference
check while the just-retrieved chain is not represented at all. The
brain-faithful move is to make the retrieved trace a legitimate competitor in
that selection (F1), **not** to weaken the trust gate or to suppress the taught
edge. Note the direction: RAVANA's measured "interference" elsewhere is
overwrite-at-write, which is a different phenomenon and must not be conflated
with retrieval-induced suppression.

## 5. Fix proposals (no gate, threshold, or rubric is changed)

All three are hypotheses with falsifiers, not results. Execution belongs to the
execution-owning agent: identical copied state, one process, `PYTHONHASHSEED=0`,
`RAVANA_OFFLINE=1`, A/B on the same k=8 items, plus the k=0/definition-only
controls and at least two unrelated fronts.

### F1 — Bind the verified chain into the monitor's evidence set (rank 1)

- **Targets:** `response_gen.py:8597-8603` (`_why_how_causal_chain`, at the
  point it already holds `_best`, `_labels`, `_g`) and
  `response_gen.py:4949-4956` (`_sm_response_grounded`, construction of
  `assoc_set` / `defn_words`).
- **Primitive change:** on `verdict=='assert' AND assertable_chain`, attach the
  graded chain's labels and hop provenance to the response context
  (`ctx.retrieved_chain_labels`, `ctx.retrieved_chain_hops`) alongside the
  returned narration. In `_sm_response_grounded`, add those labels — after the
  existing `_is_content_word` filter — to `assoc_set` used by the REFERENCE
  check. This supplies the monitor with the retrieved, source-monitored
  evidence it is missing; it does **not** lower `THETA_ASSERT`, change
  `grade_chain`/`assertable_chain`, or add a new trust path.
- **Measurable prediction:** primary `immediate_answer_reach(k=8)` rises from
  0.30 toward the assembler-fire rate (10/10 = 1.00) and sealed from 0.10
  toward 1.00; the emitted string equals `chain_texts[0]` byte-for-byte;
  `identical_surface_echo` is unchanged; k=0 and definition-only stay 0.00.
- **Falsifier:** instrument `assoc_set` for the failing items. If the taught
  `a`/`b` labels are *already* in `assoc_set`, the reference check was not the
  veto → F1 is refuted; go to F2. If adding them makes an unrelated front emit
  an untrusted chain, revert (the gate admitted nothing new by construction,
  so this would indicate a wiring error).

### F2 — One source-monitoring rule at both articulation monitors (rank 2)

- **Targets:** `response_gen.py:10111-10290` (`_forward_model_check`, exemption
  block at `:10134-10216`) and `engine.py:12903-12926` (`_final_emit_guard`,
  exemption at `:12924-12925`).
- **Primitive change:** define one predicate, e.g.
  `is_retrieved_constructed(strategy, ctx)`, true for a strategy whose text was
  assembled solely from gate-verified retrieved hops, and use it at *both*
  monitors. Today `_final_emit_guard` says `gated_explanation_chain` is
  retrieved structure while `_forward_model_check` treats it as free
  association — the same utterance gets two contradictory source-monitoring
  verdicts. Unifying them is a consistency fix between existing gates, not a
  relaxation: the chain must still clear `grade_chain` + `assertable_chain` at
  `response_gen.py:8583-8585` before it can carry the strategy.
- **Measurable prediction:** same reach/echo/control predictions as F1;
  additionally, the count of `_log_monitor_fire` records with reason
  `all-clauses-degenerate` on `gated_explanation_chain` drops to zero while the
  count for known-bad salad inputs is unchanged.
- **Falsifier:** if F1 already clears the gap, do **not** stack F2 (it is
  redundant). If bypassing the monitor lets a chain that failed
  `assertable_chain` reach the user on any known-bad probe, revert immediately.

### F3 — Cold-paraphrase cue→trace addressability (rank 3, narrower than the reverted experiment)

- **Targets:** `response_gen.py:8560-8578` (seed resolution in
  `_why_how_causal_chain`) and the unused helper
  `longest_label_seed()` at `response_gen.py:253-261`; subject used by
  `_human_like_uncertainty` at `:7357-7358`.
- **Primitive change:** when the grounded `ctx.subject` is a wh-cue phrase
  (`mechanism behind X`, `X comes`) that resolves to no node, use the longest
  taught label actually present in the raw query as a **replacement** seed
  (not an additional one). Carry that resolved concept into `ctx.subject` so
  any fallback utterance names X rather than `mechanism`. The earlier
  unconditional seed experiment was measured and reverted (see
  `docs/investigations/CAR_ROOT_CAUSE.md` RC-2.1); this is the narrow,
  untested variant.
- **Measurable prediction:** cold-probe answers name X and, once F1/F2 land,
  reach the chain; cold `answer_verdict.chain_texts` still match the taught
  `a→b→x`; k=0 and definition-only stay 0.00.
- **Falsifier:** if unrelated-chain emissions rise on non-mechanism fronts,
  revert this half and keep only F1/F2 (as the first experiment was reverted).

## 6. What the evidence does NOT support

- **Not an acquisition failure on these items.** Every k=8 subject item has
  `chain_present=true` at both immediate and cold stages; the graph endpoint is
  saturated (primary 10/10, sealed 9–10/10 cold). The +0.80/+0.90 headline is a
  graph-object endpoint and is not in dispute.
- **Not intent rejection or enumeration truncation on the measured form.**
  `mechanism_intent` admits the forms and the assembler enumerates at 64
  (`response_gen.py:8579-8582`); `generated=true` on all 20 k=8 subject items.
- **Not a trust-grade rejection at answer time.** The same `grade_chain` /
  `assertable_chain` accepted the chain that was narrated; no threshold or
  rubric was changed anywhere in this investigation.
- **Not a competing parametric answer beating the learned one.** The learned
  narration is the response at `engine.py:11733`; uncertainty is substituted
  later at `engine.py:12091`.
- **Not the final emit guard discarding the chain.** `engine.py:12924-12925`
  exempts `gated_explanation_chain`; the chain never reaches it because
  `_forward_model_check` already replaced the text.
- **Not the 0.50 echo as evidence of mechanism reach.** It is the assertion
  mirror copying the taught surface (`measure_learn5_car.py:495-501`); it is a
  separate endpoint and does not enter `immediate_answer_reach`.
- **Not a claim that the monitor is "wrong to be cautious".** It correctly
  withholds free-association salad. The defect is that verified retrieved
  structure is not represented in the evidence it checks — a source-monitoring
  wiring gap, not a bad threshold.
- **Not a claim that the item-level associate explanation is observed.** It is
  inferred from the four reach items vs the sixteen failures; F1's
  instrumentation is the test that would confirm or refute it.
- **No gates, thresholds, rubrics, weights, source files, or existing docs were
  modified.** This document is the sole artifact.
