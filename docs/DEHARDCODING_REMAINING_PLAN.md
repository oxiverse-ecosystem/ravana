# RAVANA — De-Hardcoding: Remaining Work Plan (Stages 3, 5, 6, 7 + inline gate constants)

Follow-up to `DEHARDCODING_BRAIN_PLAN.md`. This document (a) **confirms** what is
genuinely still hardcoded after the recent targeted fixes, and (b) gives a full,
brain-grounded, non-implemented plan to finish it. Nothing here is implemented.

---

## Part 0 — Confirmation of current state (verified in code)

Your read of the situation is **correct**. Verified against `engine.py` (11,498 lines):

### The Q5 / break-a-promise fix is targeted, not the M-A router
The new machinery added is a small, honest **prediction-error stack** on the snippet
path — good, brain-faithful, but it is *coverage + grounding*, not the M-A centroid
router:
- `_answer_type_mismatch` (`engine.py:7630`) — **still uses inline keyword sets**:
  `_bare_moral`, `_ambiguous`, `_normative`, plus `_proc`/`_procedural` regexes
  (`engine.py:7647-7702`).
- `_subject_head` (`engine.py:7704`) — **inline** `_generic` framing set
  (`engine.py:7713-7714`).
- `_topic_coverage_pe` (`engine.py:7729`) — **inline `_cov_thr = 0.6`**
  (`engine.py:7780`), returns hardcoded surprise `0.7` (`engine.py:7783`).
- `_polarity_mismatch` (`engine.py:7589`) — inline `_INC/_DEC/_REM` lexicons
  (`engine.py:7601-7605`).
- Malformed-grounding guard `_FRAMING` set (`engine.py:10645-10646`) — **inline**, a
  second copy of the framing words, feeding `_subject_head`.

### The ~15 routing lists still exist (Stage 3 / M-A not built)
`QUERY_PATTERNS` (`engine.py:9105`), the classical-paradox list (`engine.py:1382-1393`),
`_is_conditional/_is_yesno/_is_informational` regex, `_self_pat`, etc. — all present.
The full **Semantic Prototype Router (M-A)** replacing them with learned centroids +
fit margins is **substantial remaining work**.

### Stages 5 / 6 / 7 not started (confirmed)
- **Stage 5** — no `data/*.json` fit file exists for POS or for these gate thresholds;
  `_cov_thr=0.6`, coverage-surprise `0.7`, `_answer_type_mismatch` returns `0.6`, and
  the lexical/POS tables (`constants.json` + inline sets) are still typed constants.
- **Stage 6** — the canned assertion template `f"yeah, {topic}."` is still at
  `response_gen.py:2415` (plus its sibling leads/follows).
- **Stage 7** — `INAPPROPRIATE_WORDS` still hardcoded in `constants.py` (2 refs) and
  used in `web_learning.py`.

**So the specific failures are fixed, but the mechanisms are inline constants, not
data-driven learned parameters.** That is exactly the gap this plan closes.

---

## Part 1 — Neuroscience grounding for the remaining work

The recent additions already embody the right principle (prediction error / N400-style
surprise). What's missing is that **the criterion itself is hand-set**, whereas in the
brain the decision criterion is **learned and adaptively regulated**:

- **Decision thresholds are not fixed.** Perceptual/semantic decisions fire when
  accumulated evidence crosses a **criterion that the brain tunes to context** — the
  ACC adjusts the distance-to-threshold in proportion to predictive information, DLPFC
  accumulates evidence (Domenech & Dreher 2010, *J. Neurosci.* "Decision Threshold
  Modulation in the Human Brain"; Herz 2016, STN threshold regulation; LC-NE speed/
  accuracy control). ⇒ a hand-typed `0.6`/`0.7` is the un-brain-like part; it should be
  a **fit criterion** (Signal-Detection-Theory criterion `c`, pinned at Equal-Error-Rate
  — the calibration-neutral operating point), refittable and, ideally, context-modulated.
- **Criterion learning is trial-by-trial and reinforcement-shaped** (Computational Brain
  & Behavior 2024, SDT criterion learning ↔ matching law; homeostatic learners adapt
  criteria under concept shift, arXiv 2205.08645). ⇒ the seed→decay discipline from the
  first plan applies: ship the constant as a *seed*, then let a fit harness move it.
- **Categorization is prototype/centroid-based**, not rule-list membership (Rosch;
  distributed semantic codes, Zhang 2020 *Nat. Commun.*). ⇒ the M-A router (Stage 3)
  and the lexical/POS classifier (Stage 5) are the same mechanism: nearest-centroid in
  embedding space with a fit margin — the pattern `salad_classifier.py` /
  `SpeechActClassifier` / `brain_regions._CAUSE_SEEDS` already use.

**The template already exists in-repo** (`salad_classifier.py` + `measure_salad_classifier.py`):
a logistic/centroid model with `load()`/`save()` to `data/*.json`, threshold pinned at
EER by a `experiments/measure_*.py` harness, legacy constant reported for auditable no-
regression. **Every item below reuses that exact template.**

---

## Part 2 — Stage 5a (do FIRST): externalize the inline gate constants

Lowest-risk, highest-alignment-with-your-note. Convert the snippet-PE gate's typed
numbers into a single fit file, without changing behavior on day one (seed = current
values).

### Target file: `data/snippet_pe.json`
```jsonc
{
  "coverage_threshold": 0.6,        // _cov_thr  (engine.py:7780)
  "coverage_surprise": 0.7,         // return value (engine.py:7783)
  "answer_type_surprise": 0.6,      // _answer_type_mismatch returns (7687/7701)
  "veto_midpoint": 0.5, "veto_slope": 8.0,  // continuous veto (synaptic_dynamics style)
  "polarity_surprise": 1.0
}
```

### Loader (mirror `SaladClassifier.load`)
Add a tiny `SnippetPEConfig.load()` in a new `ravana/chat/snippet_pe_config.py`
(or extend `snippet_quality.py`) that reads `data/snippet_pe.json`, falling back to the
current constants if the file is absent (fail-open, no regression). Replace the inline
`_cov_thr = 0.6` etc. with `self._pe_cfg.coverage_threshold`.

### Fit harness: `experiments/measure_snippet_pe.py`
Following `measure_salad_classifier.py` exactly:
1. **Labeled corpus** of `(query, snippet, label)` where label = relevant / off-topic.
   Seed it with the 8 golden failures + the passing set + curated web pairs (the
   "no contact ex", "world without gravity", "language list", "govt secret" negatives
   are the anchor negatives).
2. Compute each PE component (`_topic_coverage_pe`, `_polarity_mismatch`,
   `_answer_type_mismatch`, belief PE) as **features**.
3. Fit the **combine weights** (replace the current `max()` at `engine.py:7587` with a
   learned weighted sum) and **pin each threshold at EER** via `roc_curve`.
4. Emit `data/snippet_pe.json` + a dashboard to `experiments/_snippet_pe_calib.json`,
   and **report how the legacy `0.6/0.7/max` would score the same corpus** (auditable).

**Brain basis:** the criterion becomes the SDT `c` at EER — the calibration-neutral
operating point — instead of a typed guess (Domenech 2010; SDT).

**Deliverable:** `_topic_coverage_pe` / `_answer_type_mismatch` / `_polarity_mismatch`
read all numbers from `data/snippet_pe.json`; `0.6`/`0.7`/`0.6` deleted from source.

---

## Part 3 — Stage 5b: learned lexical/POS + de-duplicate framing/lexicon sets

Two sub-parts.

### 5b-i. Distributional POS model → `data/pos_model.json`
Replace `classify_word_pos` (`constants.py:68`) + `KNOWN_VERBS/ADJS/FUNCTION_WORDS`
(`constants.json`) with a **learned distributional POS classifier**:
- Feature: a word's GloVe vector + suffix n-grams; label: POS.
- Model: nearest-centroid or logistic over embeddings (the `--learned-pos` flag already
  anticipates this — `engine.py` `use_learned_pos`).
- Fit harness `experiments/measure_pos_model.py`: train on a POS-tagged word list
  (seed from the current `known_verbs/known_adjs/function_words`), pin any ambiguity
  threshold at EER, save `data/pos_model.json`; keep the hardcoded sets as cold-start
  seed that decays. Report legacy accuracy for no-regression.
- **Brain basis:** POS/grammatical class is distributional (words in similar syntactic
  slots share embedding neighborhoods); the brain does not carry a function-word list.

### 5b-ii. Collapse the duplicated framing/generic/lexicon sets
The `_generic` (`engine.py:7713`), `_FRAMING` (`engine.py:10646`), `_bare_moral`,
`_INC/_DEC/_REM` sets are small **closed-class functional** lexicons (legitimate to keep
as *functional* primitives, per the code's own comments) — but they are **duplicated**
and **inline**. Move them to `data/functional_lexicon.json` with one loader, so there is
a single source of truth. These are functional (quantity/negation/framing) not topical,
so they are the one category where a curated seed is defensible — but it must be a data
file, not three inline copies.

---

## Part 4 — Stage 3: Semantic Prototype Router (M-A) — the big one

Replace the ~15 routing lists with **one** learned centroid router. This is the largest
remaining piece; stage it internally.

### Mechanism (one class, `ravana/chat/intent_router.py`)
- Each route = a **centroid** in embedding space (query mean-pooled GloVe), learned from
  labeled example queries: `definition_seeking`, `philosophical_abstract`, `self_directed`,
  `self_disclosure`, `episodic_recall`, `moral_advice`, `factual_yesno`, `conditional`,
  `procedural`, `humor`, `chitchat`, `remember_store` (the new store intent).
- Decision = **z-scored nearest centroid + fit margin** (exactly `SpeechActClassifier`).
  Below margin → "uncertain" → fall back to the current regex (kept as seed/backstop).
- **Fit file** `data/intent_router.json` (centroids + per-class margins at EER);
  **harness** `experiments/measure_intent_router.py`.

### Migration order (so nothing regresses)
1. Build the router; **seed centroids from the existing keyword/regex lists** (generate
   synthetic positive examples from each list) so day-one behavior matches.
2. Wire it **in parallel**, behind a flag (e.g. `--intent-router`), logging router-vs-
   regex disagreements. Fit on the golden set + collected queries.
3. Route one decision at a time to the router (start with the already-centroid-friendly
   `_is_conditional`, `moral/advice`, `philosophical`), verifying each on the regression
   suite before promotion.
4. Once all routes pass, flip default and demote `QUERY_PATTERNS` / paradox list / the
   `_is_*` regex to decaying seeds.

**Brain basis:** categorization by distributed prototype similarity, not rule lists
(Rosch; Zhang 2020 distributed semantic code). Also fixes the class of bugs where a
literal list mislabels an edge case (the "right now" false-positive the code already
had to patch by hand in `_answer_type_mismatch`).

---

## Part 5 — Stage 6: Compositional realizer (retire "yeah, {topic}")

Replace the canned assertion leads/follows (`response_gen.py:2403-2424`), self-model
stances (`response_gen.py:1996-2059`), and paradox replies (`engine.py:1575-1593`).

### Approach (staged, creative-risk managed)
- **Immediate (already partly done):** ensure the assertion mirror never fires on a
  misrouted question (the speech-act override fix from the first plan) so the malformed
  "yeah, ever tired" cannot recur even before the realizer lands.
- **Target:** realize backchannel/acknowledgement/self-model replies from the
  **discourse plan + concept graph** via `surface_realizer.py` /
  `prefrontal_workspace.py`, selecting lexical realizations by fit rather than
  `random.choice` over a typed list.
- Demote the current template strings to **exemplars** that seed the realizer's
  candidate pool; a learned scorer (reuse `salad_classifier` for fluency + a coherence
  score) picks/ranks. Fit file `data/realizer_lexicon.json`.
- **Brain basis:** production is compositional from a message + situation model, not
  slot-filling a fixed frame (DMN "internal narrative", Menon 2023). Do this **last** —
  highest creative risk, lowest correctness impact.

---

## Part 6 — Stage 7: Learned safety (retire INAPPROPRIATE_WORDS)

Replace `INAPPROPRIATE_WORDS` (`constants.py:41`) — the file's own docstring already
states the intent — with **learned emotional-valence / OFC-style reality filtering**:
- Score a candidate word/definition by **learned affective valence** (VAD engine already
  exists) + social-feedback correlation, gating on a **fit threshold**, not a word list.
- Fit file `data/safety_valence.json`; harness `experiments/measure_safety.py` pinned at
  EER on a labeled appropriate/inappropriate corpus.
- **Keep a minimal hard-override** for a tiny high-severity set (the docstring's "last-
  resort safety override") — this is the one place a short list is ethically defensible;
  everything else becomes learned.
- **Brain basis:** the OFC learns what is socially inappropriate through valence/feedback
  (social reward learning), not an innate blocklist.

---

## Part 7 — Execution order & guardrails

Recommended order (risk-ascending, value-first):
1. **Stage 5a** — externalize the inline gate constants to `data/snippet_pe.json` +
   `measure_snippet_pe.py`. (small, immediate, directly addresses your `0.6/0.7` note)
2. **Stage 5b** — POS fit file + collapse duplicated functional lexicons.
3. **Stage 3** — M-A intent router (large; internal staging as in Part 4).
4. **Stage 7** — learned safety valence.
5. **Stage 6** — compositional realizer (last).

Guardrails (unchanged from the first plan, restated because they are the contract):
- **Golden regression suite is the gate.** `tests/test_dehardcode_plan.py` already
  covers `_topic_coverage_pe` etc. — extend it to lock every migrated decision before a
  default flip.
- **Seed-then-decay, never delete cold.** Each constant/list becomes the cold-start seed
  of its fit file, weighted to decay as evidence accrues. Day-one behavior identical.
- **Flag-gated rollout, verified to beat the backstop on the regression set**, then
  promoted (the repo's established Track-B discipline).
- **Every threshold is an EER-fit criterion in `data/*.json`, refittable — never inline.**
  This is the concrete, testable definition of "brain-faithful" for the criterion.

---

## Appendix — remaining-item → target → brain basis

| Item (file:line) | Target | Brain basis |
|---|---|---|
| `_cov_thr=0.6` (`engine.py:7780`), surprise `0.7` (7783) | `data/snippet_pe.json`, EER-fit | SDT criterion `c`; adaptive decision threshold (Domenech 2010) |
| `_answer_type_mismatch` returns `0.6` (7687/7701) + keyword sets | fit weight + intent-router prototypes | speech-act congruence; prototype categorization |
| `_polarity_mismatch` `_INC/_DEC/_REM` (7601-7605) | `data/functional_lexicon.json` | closed-class quantity/negation primitives |
| `_subject_head` `_generic` (7713) / `_FRAMING` (10646) | single `data/functional_lexicon.json` | dedup functional lexicon |
| QUERY_PATTERNS / paradox list / `_is_*` regex (~15 lists) | `data/intent_router.json` centroids + margins | distributed prototype categorization (Rosch; Zhang 2020) |
| `classify_word_pos` + `known_verbs/adjs/function_words` | `data/pos_model.json` | distributional POS |
| `f"yeah, {topic}."` + self-model/paradox templates | compositional realizer + `data/realizer_lexicon.json` | compositional production; DMN narrative |
| `INAPPROPRIATE_WORDS` (`constants.py:41`) | `data/safety_valence.json` learned valence | OFC social-reward valence learning |
