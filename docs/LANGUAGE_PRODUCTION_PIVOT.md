# RAVANA — Language Production Pivot

**Status:** research document (backlog task 1, the FOUNDATION task).
**Date:** 2026-10-21 · round `auto/round-2026-10-21T0009Z`
**Constraint honoured throughout:** no LLM, no retraining, no hardcoded reply strings.
Every mechanism described below is learnable online, from conversation, in one session.

---

## 0. What this document is

The pivot in `ravana.txt` changes the target from *"remove hardcoded `if`s from the
reply path"* to *"the text generation itself must be produced from learned
structure"*. This document is the foundation the pivot's other tasks depend on.

It does three things:

1. States, with citations, how the human brain produces language.
2. Maps each mechanism onto RAVANA's **real** emit path, naming the file and line,
   and marking **ADOPT / REJECT / DEFER** with a reason.
3. Names the **structural gap** — not in adjectives, but as a function and a line.

Every gap claim below was verified against the checked-out tree at
`auto/round-2026-10-21T0009Z` (merge-base `ceaa6071`). Line numbers are real.

---

## 1. How the brain produces language

### A1 — Hierarchical predictive processing
Rao & Ballard (1999) showed cortex is a hierarchical predictive coder: each level
predicts the level below and passes only the *prediction error* up. Friston
(2005, 2010) generalised this to the Free Energy Principle — action and perception
both minimise variational free energy. Clark (2013) applies it to language: a
high-precision (confident) top-down prediction suppresses bottom-up error, so the
system asserts; low precision lets error drive hedging and questioning.

**Key point for RAVANA:** prediction is not a post-hoc confidence label. It is the
*generative* mechanism — the surface form is the argmin of an energy function over
candidate forms, and the confidence is a by-product of that minimisation.

### A2 — Production / comprehension asymmetry
Dell & Reich (1981) and Levelt (1989) established that speaking and understanding
are not the same system run in reverse. Comprehension is a *search* over possible
parses; production is a *competitive selection* among lemmas and structures under
a forward plan. The two share a lexicon but not a mechanism.

**Key point:** a system that only ever *parses* its input has no production
mechanism at all. RAVANA must have a forward generator, not a matcher.

### A3 — Incremental left-to-right composition
Levelt (1989, 2001) and Ferreira & Tanenhaus (1996): the speaker commits to a
fragment and articulates it before the whole sentence is planned. The plan is
revised on the fly as the fragment's own prediction error comes back. This is why
self-correction and restart exist.

**Key point:** production is single-pass *with revision*, not a single
non-refundable choice. A system that picks a frame and never revisits it cannot
self-correct.

### A4 — Lexicon vs the combinatorial mechanism
Marcus (1995) and Dehaene et al. (2004) separate the **lexicon** (stored
word-forms, arbitrary, item-specific) from the **combinatorial mechanism**
(rule-governed, productive, generalises to novel combinations). Pulvermüller (1995)
grounds the distinction neurophysiologically: cell assemblies for words vs
syntactic circuits that bind them.

**Key point:** a closed word list is a lexicon with no growth path. A frozen
grammar is a combinator with no learning. Both must be open.

### A5 — How a ~20 W system avoids brute-force search
Olshausen & Field (1997) — sparse coding: represent with few active units.
Hopfield (1982) — energy-based attractor retrieval: the answer is the minimum of
an energy landscape, found by relaxation, not enumeration. Schultz (1997) —
dopamine as a reward-prediction-error signal that *teaches* the selector.

**Key point:** the brain never enumerates all sentences. It relaxes to a minimum
under a learned energy function, and the energy function is *learned from feedback*.

---

## 2. Mapping onto RAVANA's real emit path

RAVANA's emit path is a **router** (`engine.py:7290 process_turn`) with ~55
early-return strategy branches, each returning a string. The branches that
*generate* rather than *retrieve* are:

| Stage | File:line | What it does |
|---|---|---|
| Router | `chat/engine.py:7290` | `if/elif` strategy dispatch, ~55 early returns |
| Support retrieval | `chat/response_gen.py:3495` `_retrieve_support` | counts graph edges / definitions into a bundle |
| Gap selection | `chat/response_gen.py:3600` `_select_probe_dimension` | deterministic information-gain over the bundle |
| Realization | `chat/response_gen.py:3626` `_realize_metacognitive` | composes the reply from the bundle |
| Cue loop | `chat/response_gen.py:3703` `_generative_cue_loop` | cue → retrieve → realize |
| Frame binding | `language/syntactic_cell_assembly.py:200` `bind_to_sentence` | Hebbian role binding from POS priors |
| Verb selection | `language/syntactic_cell_assembly.py:272` `_pick_verb_for_relation` | Hebbian verb_role argmax |
| Realization | `language/surface_realizer.py:348` `realize` | free-energy-driven surface composition |
| Variant choice | `language/surface_realizer.py:696` `_select_variant` | weighted pick from `MERGE_VARIANTS` |
| Hedge | `language/surface_realizer.py:233` `_compose_hedge` | `random.choice(HEDGE_MORPHEMES)` |
| Epistemic frame | `language/surface_realizer.py:246` `_generate_epistemic_frame` | hash-pick from `EPISTEMIC_FRAMES` |
| Reflective clause | `language/surface_realizer.py:269` `_generate_reflective_clause` | `random.choice` over magnitude pool |
| Question | `language/surface_realizer.py:305` `_generate_question` | `random.choice` over FE pool |
| Discourse marker | `language/surface_realizer.py:1210` `_select_discourse_marker` | `random.choice` over type pool |
| Hedge frame | `chat/hedges.py:108` `hedge_frame` | composes from measured modality + slots |
| Marker lexicon | `chat/hedges.py:72` `_MARKERS` + `:87 register_marker` | runtime-extensible modal tags |
| Reward | `chat/engine.py:10682` | `user_understood = quality_score >= 0.55` |
| Learning | `chat/engine.py:10684,10687` | `learn_from_feedback(...)` on both modules |

### A1 — Hierarchical predictive processing
**RAVANA analogue:** `surface_realizer.py:348 realize` is explicitly
free-energy-driven, and `_get_confidence_level:219` maps FE → confidence band.
`hedges.py:161 modality_from_support` maps a support score to a modality.

**What is missing — the gap.** The hierarchy is **two levels, hand-plumbed**:
`SyntacticFrame` (`syntactic_cell_assembly.py:24`) is *given* as a triple
(subject, verb, object) by the caller; `realize` never predicts downward, it
*formats* a finished triple. Precision is a **scalar with hardcoded cuts**
(`_get_confidence_level:219` → `<0.3 / <0.6 / else`; `modality_from_support:161`
→ `>0.55 / >0.30 / else`). There is no per-level prediction error, so there is
nothing for a higher level to minimise.

**Verdict: PARTIAL ADOPT.** Keep the free-energy framing and the FE→confidence
mapping (they are real and grounded). Replace the scalar cuts with a *learned*
precision, and add the missing downward-prediction step.

### A2 — Production / comprehension asymmetry
**RAVANA analogue:** comprehension is `process_turn`'s router + the miners in
`user_model.py`. Production is `realize`.

**What is missing — the gap.** The two sides **never meet**. `bind_to_sentence:200`
builds a frame and `realize` renders it, but **nothing re-encodes what RAVANA just
produced** into its own stores. The only forward check is
`_utterance_complete:663`, backed by a hand-authored verb list
`_COMPLETE_VERBS` (`surface_realizer.py:619-628`). So RAVANA cannot learn its own
output — it has no production model to learn from.

**Verdict: REJECT the current shape; ADOPT the asymmetry.** RAVANA must have a
forward generator whose output is fed back as training signal. Today the
generator's output is discarded.

### A3 — Incremental left-to-right composition
**RAVANA analogue:** `realize` *is* left-to-right (`:407` subject → `:419` object
→ `:431` agreement → `:576` punctuate).

**What is missing — the gap.** It is **single-pass with no revision**. The variant
is chosen *after* all constituents exist (`:481 _select_variant`) and applied as
one of ten f-string templates (`_apply_variant:778`, templates at `:808-840`).
There is no point at which RAVANA can say "that fragment's prediction error is too
high, back up, re-plan". Self-correction is structurally impossible.

**Verdict: DEFER (partially).** Left-to-right is correct and worth keeping. The
revision loop is a real gap but is a *second* mechanism; fix A1/A2 first.

### A4 — Lexicon vs the combinatorial mechanism
**RAVANA analogue:** the lexicon is the ConceptGraph + GloVe; the combinator is
`SyntacticCellAssembly` (Hebbian role matrices, `POS_ROLE_PRIORS:59`) +
`CerebellarNgram` (sparse bigram/trigram, `cerebellar_ngram.py:45`).

**What is missing — the gap.** Two separate failures:

1. **The content lexicon is closed.** `_uncountable_nouns:125`,
   `_abstract_nouns:160`, `_pronoun_map:145` are hand-authored English sets.
   Removing one entry is not graceful — it is a code edit. There is no runtime
   growth path for *content* words (contrast `hedges.register_marker:87`, which
   *does* have one).
2. **The combinator is dead.** `_pick_verb_for_relation:272` **ignores its
   `relation` argument**: it takes the global argmax of `verb_role` (`:285-289`)
   and only consults the relation map at `:293` as a *fallback when no verb is
   found*. So the relation-conditioned verb choice — the entire point of a
   combinatorial mechanism — never fires when any verb exists.

**Verdict: ADOPT the separation; REJECT the current implementations.** The
Hebbian role matrices are the right shape and should be kept. The closed word
sets and the relation-ignoring verb picker must go.

### A5 — How a ~20 W system avoids brute-force search
**RAVANA analogue:** `CerebellarNgram` is *designed* for this — sparse storage
only (`bigram`/`trigram` dicts, `cerebellar_ngram.py:45-52`), explicitly to
avoid a dense 10 000×10 000 matrix.

**What is missing — the gap — a verified defect.** The cerebellum is
**disconnected from the output path**:

* `learn_chain:73` writes **concept labels** into `bigram`/`trigram`
  (`_learn_label_sequence:125`, `_learn_hops:100`).
* `_select_verb_phrase:1098` reads with key `f"phrase:{relation}"` (`:1108`).
* The string `"phrase:"` appears **only at that read site** — grep over
  `ravana/src/ravana/language/` and `ravana/src/ravana/chat/` finds **no write
  site**. `predict_next("phrase:semantic")` therefore always returns `{}`, the
  `if ngram_result:` guard at `:1110` is never true, and control always falls
  through to `VerbLexicon.select_verb`.

So the one live online sequential learner in the architecture **contributes
nothing to any reply**. It is dead code wearing a learner's clothing.

Two further defects in the same module:

* **Unconditional decay.** `_apply_decay:156` runs on *every* `learn_chain` call
  (`:98`) and subtracts `decay_rate` from **every** entry, with no usage-based
  protection. `learn_chain` is called on every turn
  (`engine_generation.py:415,625`; `interface.py:1078,1772`). A transition not
  re-learned *this turn* is erased. That is the opposite of memory.
* **Self-graded reward.** `engine.py:10682` sets
  `user_understood = (quality_score >= 0.55)` where `quality_score` comes from
  `_assess_response_quality` (`engine_generation.py:627`) — a heuristic over
  *RAVANA's own output* (strategy label, length, unique-noun count). RAVANA is
  grading itself. The real signal — did the *user* understand — is available one
  turn later and is not used.

**Verdict: ADOPT the sparse design; REJECT the wiring.** Fix the key mismatch
(makes the learner live), make decay usage-protected, and replace the self-graded
reward with a user-derived signal.

---

## 3. The structural gap, stated precisely

Not in adjectives — as a function and a line:

> **RAVANA has a formatter, not a generator.** `surface_realizer.py:348 realize`
> takes a finished `SyntacticFrame` triple and applies one of ten authored
> f-string templates (`_apply_variant:778`, `:808-840`). The *content* of the
> reply comes from cognition; the *form* does not — it comes from
> `MERGE_VARIANTS:154` and a `random.choices` draw (`_weighted_variant_select:756`).
>
> The brain's form is the **argmin of a learned energy function over candidate
> forms** (A1, A5). RAVANA's form is a **weighted random pick from a fixed list**.
> That is the whole gap. Everything else in this document is a consequence of it.

Three concrete consequences, each verified:

1. **No two replies are the same string unless the state is the same** — the
   pivot's acceptance criterion — is *false today*, because the variant draw is
   `random.choices` (`:756`) and the hedge/reflective/question draws are
   `random.choice` (`:243`, `:303`, `:346`). Same state, different string.
2. **The online sequential learner is dead** (A5) — so no amount of conversation
   teaches RAVANA a new transition.
3. **The relation-conditioned combinator never fires** (A4) — so the relation
   type that *should* determine the verb phrase is discarded.

---

## 4. ADOPT / REJECT / DEFER summary

| Mechanism | Verdict | Reason |
|---|---|---|
| A1 free-energy framing + FE→confidence | **ADOPT** | Real, grounded, already implemented |
| A1 per-level prediction error | **ADOPT (new)** | The missing generative step |
| A1 scalar hardcoded cuts | **REJECT** | Replace with learned precision |
| A2 production/comprehension asymmetry | **ADOPT** | RAVANA has no production model |
| A2 re-encode own output | **ADOPT (new)** | The feedback channel |
| A3 left-to-right | **ADOPT** | Correct, keep |
| A3 revision loop | **DEFER** | Second mechanism; fix A1/A2 first |
| A4 Hebbian role matrices | **ADOPT** | Right shape, keep |
| A4 closed content word sets | **REJECT** | No runtime growth path |
| A4 relation-ignoring verb picker | **REJECT** | Verified defect, combinator dead |
| A5 sparse n-gram design | **ADOPT** | Correct, keep |
| A5 cerebellar key mismatch | **REJECT** | Verified defect, learner dead |
| A5 unconditional decay | **REJECT** | Erases unreinforced memory |
| A5 self-graded reward | **REJECT** | Grades itself, not the user |

---

## 5. Incremental online design (the adopted shape)

**Name: OCG — Online Construction Grammar.** Five components, each learnable from
one conversation, none requiring a retrain or an LLM.

### C1 — Construction schema induction (learns *structure*, not strings)
A new module `language/construction_schemas.py` induces **slot schemas** from
contrastive utterance pairs (pattern grammar, Nakazawa 1995). Given two user
utterances that differ in exactly one position, the differing span is a **slot**
and the shared frame is a **schema**. Stored as data (a schema store), not as
`if/elif`. RAVANA can add, revise, and overwrite schemas at runtime; removing one
degrades gracefully to the unschematised path.

*Seed:* 6 slots (`SUBJ/VERB/OBJ/MOD/CLAUSE/ADJUNCT`), 5 relation types (already
at `syntactic_cell_assembly.py:89-96`), 2 contrastive pairs (the minimum for a
slot to generalise).

### C2 — Energy-argmin realization (replaces every `random.choice`)
`realize` generates a **candidate set** from the instantiated schema and selects
by argmin of a learned energy function (Friston 2005) instead of
`random.choices` (`:756`). The energy has real, learned terms: schema fit,
cerebellar transition strength (once C4 is live), and free energy. Same state →
same argmin → same string. This is what makes the pivot's acceptance criterion
(true) achievable.

### C3 — Grounded learning signal: alignment + repair from the user's next turn
Replace `user_understood = quality_score >= 0.55` (`engine.py:10682`) with a
signal computed from **the user's next utterance**:

* **alignment** — Jaccard overlap of content lemmas between RAVANA's utterance
  `u_t` and the user's reply `u_{t+1}`, plus whether `u_{t+1}` introduces a new
  fact about RAVANA's subject (positive evidence the utterance landed).
* **repair** — detection of correction markers over RAVANA's claim. The detector
  **already exists**: `user_model.py:6695 _detect_correction` (called from
  `user_model.py:6655`), which sets the supersession field on the fact
  (`user_model.py:1839` references `detected_correction_fact`). The signal is
  available; the wiring into the realizer's learning is not.

### C4 — Make the cerebellum live (fix the key mismatch)
Write the same key the reader uses. Either `learn_chain` writes
`f"phrase:{relation}"` entries, or `_select_verb_phrase:1108` reads concept-label
keys. One line, and the dead learner becomes a live one. Also make
`_apply_decay:156` usage-protected (decay only entries not reinforced within a
window) so memory survives a conversation that moves on.

### C5 — Open content lexicon + relation-conditioned predicate binding
Replace the closed sets (`_uncountable_nouns:125`, `_abstract_nouns:160`) with a
**growable** store: a word learned tonight is usable tomorrow, participates in
composition, and is revisable when corrected. And fix
`_pick_verb_for_relation:272` to actually use its `relation` argument — the
relation-conditioned choice is the combinator's whole job.

### Rejected alternatives
* **Any retrain** — rejected on principle (pivot constraint 3).
* **Any LLM** — rejected on principle (pivot constraint 3).
* **More authored strings** — rejected (hardcoding rule).
* **Re-weighting the existing RNG pools** — rejected: that is tuning the
  formatter, not building a generator.

---

## 6. Minimal seed structure (not answers)

| Seed | Where | Why it is seed, not hardcode |
|---|---|---|
| 6 slots | `construction_schemas.py` | RAVANA induces new slots from contrastive pairs at runtime |
| 5 relation types | already `syntactic_cell_assembly.py:89-96` | Extensible; the verb picker reads them |
| Minimal closed class + 4 modal tags | already `hedges.py:47 MODALITY` | `register_marker:87` extends at runtime |
| 2 contrastive utterance pairs | `construction_schemas.py` | The minimum for a slot to generalise |
| Morph rule set | `surface_realizer.py` (existing) | Rules, not word forms |
| 1 alignment/repair channel | `engine.py` end-of-turn | Load-bearing; the only grounded reward |

**Deleted, not extended:** `response_gen.py:3676-3698` (five authored leads +
five authored probes) and the `random.choice` pools at
`surface_realizer.py:134-175, 284-295, 329-345` and the closed sets at
`syntactic_cell_assembly.py:125-174`. Their function is re-derived at runtime
from the schema store and the energy function.

`hedges.py:12-38` already states the diagnosis in its own comments: *"the content
of the reply did not come from RAVANA's state, only its nouns did."* This
document is the mechanism that makes the form come from state too.

---

## 7. Verification of every gap claim

Each claim in §2-§3 was checked against the tree at `auto/round-2026-10-21T0009Z`:

| Claim | Check | Result |
|---|---|---|
| `_pick_verb_for_relation` ignores `relation` | read `:272-300` | **confirmed** — global argmax at `:285-289`, relation map fallback-only at `:293` |
| Cerebellar key mismatch | grep `"phrase:"` in `language/` + `chat/` | **confirmed** — read site `:1108` only, no write site |
| Unconditional decay | read `_apply_decay:156`, call at `:98` | **confirmed** — subtracts from every entry, every call |
| Self-graded reward | read `engine.py:10682`, `engine_generation.py:627` | **confirmed** — grades RAVANA's own output |
| Closed content lexicon | read `syntactic_cell_assembly.py:125-174` | **confirmed** — hand-authored sets, no growth path |
| Correction detector exists | grep `_detect_correction` | **confirmed** — `user_model.py:6454`, `engine_persistence.py:577` |
| `random.choices` in variant select | read `_weighted_variant_select:756` | **confirmed** — `:776` |
| `random.choice` in hedge/reflective/question | read `:243, :303, :346` | **confirmed** |

---

## 8. Residual risk

* **C2 changes the emit path.** Any change to `realize` risks the existing
  routing tests. Mitigation: the energy-argmin must be a *drop-in* for the
  current weighted pick — same signature, same return type — and the seed energy
  must reproduce today's distribution so day-one behaviour is unchanged.
* **C3 needs a turn of latency.** The reward for turn *t* arrives at turn *t+1*.
  Mitigation: buffer the pending reward (the engine already has
  `_calib_buf` at `engine.py:10629` to model this).
* **C1 schema induction can over-fit.** Two pairs is the minimum; more pairs
  give more slots but also more noise. Mitigation: a schema must be supported by
  at least two *distinct* pairs before it is used for realization.
