# RAVANA — Brain-Faithful De-Hardcoding Plan

**Goal:** replace *every* hardcoded table / regex list / hand-set threshold / canned
template in the chat stack with a real, learned, brain-grounded mechanism. No
allowlists, no keyword gates, no magic numbers, no fixed reply strings. The system
should decide the same way a brain does: by *prediction error*, *distributed semantic
representations*, *learned control*, and *self-referential memory encoding*.

This document is a **plan only** — nothing here has been implemented. It records the
diagnosis of the 8 reported failures, the neuroscience that motivates each fix, a full
census of what is hardcoded today, and a staged roadmap to remove all of it.

---

## Part 0 — The failures (verified in code)

| # | Symptom | Root cause (file:line) | Class |
|---|---------|------------------------|-------|
| Q11 | "perpetual motion" → conspiracy snippet | only semantic gate is GloVe cosine `_snippet_plausibility` (`engine.py:7386`); veto only fires `plaus<0.12` or `plaus<0.38 & trust<0.5` (`engine.py:7735-7756`). No claim-type / truth check. | relevance gate under-filters |
| Q16 | "why does my code crash" → language list | `_best_answer_snippet` (`engine.py:6733`) has no enumeration/list-junk detector; `_SNIPPET_REJECT_SHAPES` (`engine.py:6064`) too narrow; `SnippetStructureModel` that would catch it is **off by default** (`engine.py:369`). | relevance gate under-filters |
| Q15 | "gravity doubled" → "world WITHOUT gravity" | **no negation/contradiction modeling anywhere**; conditional scoring (`engine.py:6957-6967`) boosts *any* gravity-counterfactual regardless of premise polarity. | plausibility gate under-filters |
| Q4 | "square a circle" → POS system | topic extraction (`interface.py:705-811`) has no sense/acronym disambiguation; `_sense_biasing_framing` knows only 4 domains (`web_learning.py:315-326`). | sense disambiguation missing |
| — | "is it okay to break a promise" → "no contact ex" | no query-intent (moral/advice vs factual) → answer-type match; snippet matched on subject token "promise" only. | question-type routing missing |
| — | "meaning of life" → raw "life" definition | §5 pre-pass `_consult_internal_knowledge` (`engine.py:3317`) grounds subject="life" → `definitions["life"]` (`brain_regions.py:273`); paradox list (`engine.py:1382-1393`) lacks "meaning of life"; the exclusion that exists lives only in *downstream* gates the pre-pass bypasses. | definitional literalness |
| — | "do you ever get tired" → "yeah, ever tired" | (1) `_handle_self_model` predicate whitelist (`response_gen.py:2019-2021`) omits "tired" → bails; (2) hybrid speech-act classifier semantic override (`prefrontal_workspace.py:277`) mislabels the question as a *statement* because "tired" sits near the statement prototype; (3) assertion mirror echoes malformed subject "ever tired" behind hardcoded `"yeah, "` (`response_gen.py:2408`). | self-model gate + speech-act override + echo template |
| — | "remember i love stargazing" → not stored | `_is_self_disclosure_stmt` **hard-excludes any "remember"** (`engine.py:3142-3145`), so the fact never reaches `observe_user_query`; `_episodic_remember` then treats it as recall (`engine.py:2034-2045`) and finds nothing. There is **no "remember X = store X" intent** anywhere. | episodic encoding of directives |

**Two structural themes unify all eight:**
1. **Gating by lists instead of by prediction error / evidence.** Every filter is a
   fixed set membership test, not a learned "does this surprise my forward model?"
2. **No model of *communicative intent* or *claim truth*.** The system matches topics
   (bag-of-words / GloVe cosine) but never asks *what kind of answer is wanted* or
   *is this claim consistent with what I believe*.

---

## Part 1 — Neuroscience grounding (what "the brain way" actually is)

Each fix maps to a documented brain mechanism (citations gathered during research):

- **Relevance / plausibility (Q11, Q16, Q15, break-a-promise).** The brain does *not*
  keep a blocklist of bad snippets. It runs a **predictive model of meaning** and
  reacts to **semantic surprise** — the **N400** is literally the amplitude of
  prediction error against a probabilistic representation of meaning (Rabovsky/
  McClelland 2018 *Nat. Hum. Behav.*; Lindborg 2023 "Semantic surprise predicts the
  N400"). Plausibility and predictability are **dissociable** contributions to that
  signal (Nieuwland 2019). **Negation is processed incrementally by the same
  predictive machinery** — "not X" is integrated against expectation, producing a
  distinct post-N400 positivity for implausible continuations (Farshchi & Paradis 2026,
  *Brain & Language*). ⇒ RAVANA needs a *forward model that produces a surprise/PE
  signal per candidate answer, sensitive to negation and claim-type*, not a regex list.

- **Definitional literalness ("meaning of life").** Abstract vs concrete concepts are
  processed by **different sub-networks** (Hoffman & Bair 2025 meta-analysis, 72
  studies): abstract concepts recruit **social + semantic-control** networks; concrete
  concepts recruit **action/situation** networks; the **Default Mode Network** handles
  **abstraction, self-reference, and "internal narrative"** (Menon 2023, *Neuron*, "20
  years of the DMN"; Zhang 2020 *Nat. Commun.* — concreteness→abstractness is a DMN
  gradient). ⇒ A philosophical question should route to a *reflective/abstraction*
  process, not a dictionary lookup, and that routing should be driven by a **learned
  concreteness/abstractness signal**, not a phrase list.

- **Self-model ("do you ever get tired").** Introspective questions about the agent are
  self-referential; the correct handler is a *stance* about the agent's own nature,
  produced compositionally — not a keyword-gated template.

- **Episodic encoding ("remember i love stargazing").** The **self-reference effect**:
  information linked to the self is encoded far more richly (mPFC + hippocampus;
  Piolino 2015; Symons/Johnson SRE). An **explicit "remember" directive is intentional
  declarative encoding** — the strongest possible encode signal (Squire; hippocampal
  episode-specific neurons bind *what happened*, Nature Hum. Behav. 2023). ⇒ "remember
  X" must be treated as a **high-salience store command**, the opposite of what the code
  does today (it discards it).

---

## Part 2 — Full census of hardcoded elements (the removal targets)

Grouped by category. Full file:line detail lives in the investigation notes; summary
counts here define the work.

| Category | Approx items | Worst offenders |
|---|---|---|
| 1. Intent/routing keyword+regex lists | ~15 lists, ~25 regex | `engine.py:9105` QUERY_PATTERNS; `engine.py:1355-1404` paradox list; `engine.py:6041/6064` snippet domain/shape; `interface.py:48-92` dup lists |
| 2. Response templates / canned strings | ~30 | `engine.py:1575-1593` paradox replies; `response_gen.py:1996-2059` self-model; `response_gen.py:2403-2424` assertion leads ("yeah, {topic}"); `hedges.py:30-79` |
| 3. Snippet/web filters | ~8 lists, ~17 regex | `constants.json:web_garbage` (158); `_JUNK_SNIPPET_DOMAINS`; `_SNIPPET_REJECT_SHAPES`; `web_learning.py:1024-1059` code regexes |
| 4. Lexical/POS tables | ~20 sets + `constants.json` 5 lists (~480 words) | `engine.py:8992-9084` subject-context words; `_GRAMMATICAL_CONCEPTS`; glue/vague sets |
| 5. Thresholds / numeric constants | ~45 named + inline | `constants.py:20-21` salad; `engine.py:7372` plausibility floor; `provenance.py:59`; `synaptic_dynamics.py` sigmoid params |
| 6. Category/ontology tables | ~12 | `engine.py:2261-2301` category affordances; `engine.py:3478-3517` sensory dims; `response_gen.py:4406` relation predicates |
| 7. Safety/content lists | ~5 | `constants.py:41` INAPPROPRIATE_WORDS; `engine.py:177` protected concepts |

**Already-learned substrate to imitate** (these modules were deliberately built as the
"brain answer" and only retain small *seed* tables): `salad_classifier.py`,
`snippet_quality.py`, `provenance.py`, `self_model_router.py`, `case_distribution.py`,
`synaptic_dynamics.py`, `brain_regions.py`. **The pattern they establish is the target
pattern for everything else:** a small vector/statistical model + a fit file in `data/`,
with the old list demoted to a cold-start seed that decays as evidence accrues.

---

## Part 3 — The replacement mechanisms (design)

Five reusable learned primitives replace nearly all the hardcoded tables. Build these
once; wire them everywhere.

### M-A. Semantic Prototype Router (replaces intent/routing keyword lists — cat. 1, 6)
A single mechanism: every "is this a X-kind-of-utterance?" decision becomes
**nearest-prototype in embedding space with a learned, per-class margin**, exactly like
`SpeechActClassifier` / `brain_regions._CAUSE_SEEDS` already do.
- Represent each route (definition-seeking, philosophical/abstract, self-directed,
  disclosure, recall, moral/advice, factual-yesno, conditional, humor…) as a **centroid
  learned from examples**, not a regex.
- Decisions use **z-scored distance + confidence**, and the winning margin is **fit to
  labeled data** (equal-error-rate, the method already used for `salad_classifier.json`).
- Cold start: seed centroids from the *current* keyword lists (so behavior doesn't
  regress), then let online evidence + a fit corpus move them. Lists become seeds, not
  gates.
- **Kills:** QUERY_PATTERNS, paradox list, `_is_conditional/_is_yesno/_is_informational`
  regex, `_self_pat`, `is_likes_query` literal lists, category dicts.

### M-B. Concreteness/Abstractness signal (fixes "meaning of life" — cat. 6)
A learned scalar per subject: **abstract ↔ concrete**, computed from embedding
neighborhood statistics (abstract words cluster with social/evaluative terms; concrete
with sensorimotor terms — the DMN social-spatial axis). 
- High-abstract + question-form ⇒ route to a **reflective/abstraction** generator (DMN
  "internal narrative"), *not* the dictionary path.
- This replaces the need to special-case "meaning of life": *any* abstract-philosophical
  query routes reflectively because the **signal is continuous and learned**, not a
  phrase in a list.
- Fixes the architectural bug too: the reflective check must run **in the §5 pre-pass**
  (`engine.py:3317`), before `consult_internal` can dump a definition.

### M-C. Answer-Plausibility Forward Model (fixes Q11/Q15/Q16/break-a-promise — cat. 3, 5)
The core N400 analog. For each candidate snippet, produce a **prediction-error /
plausibility score** that is *learned* and sensitive to more than cosine:
1. **Structural PE** — already exists (`SnippetStructureModel`); **turn it on** and let
   its threshold stay data-fit. Catches Q16 list-junk and boilerplate.
2. **Claim/answer-type match** — score whether the snippet's *speech act* matches the
   query's *requested* answer type (moral question → normative statement, not advice;
   factual → assertion). Reuse M-A prototypes on the snippet. Fixes break-a-promise.
3. **Negation/premise-polarity consistency** — detect the query's premise polarity
   ("gravity **doubled**" = increase) and the snippet's ("**without** gravity" =
   removal); penalize polarity mismatch. This is the missing negation model (Farshchi &
   Paradis). Fixes Q15. Implement as a small learned scorer over
   negation/quantity-modifier features (the repo already has `quantity_modifier.py` and
   `implicature_detector.py` to build on).
4. **Belief-consistency PE** — score the claim against the belief store / graph; a
   fringe claim ("perpetual motion = govt secret") has **low convergence** and should
   raise PE. Provenance already tracks convergence (`provenance.py:59`); wire it into
   the answer veto. Fixes Q11.
- **Combine** these as a learned weighted PE (weights fit, not hand-set), and make the
  veto a **function of PE**, retiring the `0.12/0.38/0.5` magic floors
  (`engine.py:7372-7756`). The `synaptic_dynamics.py` sigmoid pattern is the template:
  continuous gate, midpoint/slope fit from the PE distribution of known-good vs
  known-bad answers.

### M-D. Sense/Acronym Disambiguator (fixes "square a circle" — cat. 1)
Word-sense disambiguation via **context-conditioned embedding**: pick the sense whose
neighborhood best fits the *rest of the query*, not the globally most frequent sense.
- "square"+"circle" co-context ⇒ geometry sense wins over payments-company sense.
- Feeds M-C's plausibility and the query-variant builder, replacing the 4-domain
  `_sense_biasing_framing` list. Standard brain account: lexical access is
  context-modulated (the N400 literature — context sharpens the predicted sense).

### M-E. Self-Reference Episodic Encoder (fixes "remember i love stargazing" — cat. 1)
Two coupled fixes, both brain-grounded (self-reference effect + intentional encoding):
1. **"remember X" is a STORE directive, not a recall trigger.** Detect via M-A whether a
   "remember"-framed utterance *contains a new self-disclosure* (a proposition with an
   `I`-subject and an attitude/preference predicate). If so, it is the **highest-salience
   encode event** — route it into `observe_user_query` / `preferences`, tagging it with
   a boosted encoding weight (the self-reference effect). Remove the blanket "remember"
   exclusion at `engine.py:3142-3145`.
2. **Disclosure detection becomes learned**, not regex: a proposition-level check
   (subject = user, predicate expresses stance/possession/identity) using the existing
   `proposition_parser.py`. This generalizes past "i love/like/hate" to any phrasing.
- Recall side: same-turn stores must be visible to recall (the `transcript[:-1]` bug at
  `engine.py:2045` excludes the current turn); an explicit store makes the fact
  retrievable immediately.

### M-F. Compositional Reply Realizer (replaces canned strings — cat. 2)
The templates ("yeah, {topic}.", self-model stances, paradox replies, empathy frames,
hedges) are the hardest to fully de-hardcode and the **lowest-risk to defer**. Target
state: replies are realized from the **discourse plan + concept graph** via
`surface_realizer.py` / `prefrontal_workspace.py`, with the current template strings
demoted to fallback exemplars that seed a learned realizer. The immediate win is
**stopping the malformed echo** ("ever tired"): fix the self-model gate (M-A predicate
is learned, so "tired" routes correctly) and the speech-act override
(`prefrontal_workspace.py:277`) so a syntactic question is never overridden into a
statement — then the assertion-mirror template never fires on a question at all.

---

## Part 4 — Staged roadmap

Each stage: build behind a flag, seed from current lists (no regression), fit on a
labeled set, verify it **beats the hardcoded backstop on a regression suite**, then flip
default and demote the list to a decaying seed. This mirrors the repo's existing
Track-B discipline (`--snippet-pe`, `--source-trust`, `--learned-pos`).

**Stage 1 — Turn on what already exists + fix the 3 pure routing bugs (low risk, high value)**
- Enable `SnippetStructureModel` by default (wire the missing `--snippet-pe`/default),
  after confirming it beats regex on a regression set. (Q16)
- Fix §5 pre-pass ordering so a reflective/abstract check runs before `consult_internal`.
  (meaning-of-life architectural bug)
- Remove the blanket "remember" exclusion; add "remember X = store" routing. (stargazing)
- Fix the speech-act semantic override so syntactic questions aren't relabeled
  statements; widen self-model routing to be prototype-based. (do-you-get-tired)
- Fix `transcript[:-1]` same-turn recall exclusion.
- Build the **regression harness** first: freeze all 8 failures + a passing set as
  golden tests (`tests/`), run via `scripts/ravana_chat.py --chat "...|..."`.

**Stage 2 — Answer-Plausibility Forward Model (M-C)** — the biggest correctness win.
- Add negation/premise-polarity scorer (Q15), claim-type match (break-a-promise),
  belief-convergence PE (Q11). Combine into one learned PE; retire the magic veto floors.
- Fit weights + gate on a labeled snippet set (good/bad per query-type).

**Stage 3 — Semantic Prototype Router (M-A)** — collapse cat.-1 & cat.-6 lists into
learned centroids + fit margins. One mechanism replaces ~15 lists + ~25 regex.

**Stage 4 — Concreteness signal (M-B) + Sense disambiguator (M-D)** — finish
meaning-of-life generalization and fix "square a circle" and future sense errors.

**Stage 5 — Lexical/POS + thresholds (cat. 4, 5)** — externalize remaining word tables
to fit files; replace remaining hand-set thresholds with `synaptic_dynamics`-style
continuous gates whose params are fit, not typed.

**Stage 6 — Compositional realizer (M-F)** — retire canned reply strings; realize from
plan + graph. Longest, most creative-risk; do last.

**Stage 7 — Safety (cat. 7)** — replace `INAPPROPRIATE_WORDS` with learned emotional
valence / OFC-style reality filtering (the file's own docstring already states this is
the intent), keeping a minimal last-resort override.

---

## Part 5 — Guardrails (so we don't regress while de-hardcoding)

1. **Golden regression suite first.** The 8 failures + a broad "known-good" set become
   automated tests before any mechanism is swapped. Nothing flips default until it
   passes both.
2. **Seed-then-decay, never delete cold.** Every list becomes the *cold-start seed* of
   its learned replacement, weighted to decay as real evidence/fit data accrues — so
   day-one behavior never gets worse.
3. **Flag-gated rollout.** Each learned model ships off-by-default with a CLI flag,
   verified to beat its backstop on the regression set, then promoted. (Existing
   `--snippet-pe` / `--source-trust` / `--learned-pos` show the pattern.)
4. **Fit files in `data/`.** Learned params live next to `salad_classifier.json` /
   `attribute_theta.json` / `case_dist.json`, versioned and refittable — never inline
   constants.
5. **Prediction-error everywhere.** The unifying acceptance test: a gate is "brain-
   faithful" only if its decision is a *continuous surprise/evidence signal* with
   *learned* parameters, not a set-membership or a typed threshold.

---

## Appendix — one-line mapping: failure → mechanism → brain basis

- Q11 conspiracy → M-C belief-convergence PE → N400 semantic surprise + provenance
- Q16 language list → Stage-1 enable SnippetStructureModel → structural prediction error
- Q15 without-gravity → M-C negation/premise-polarity → incremental negation processing
- Q4 POS system → M-D sense disambiguation → context-modulated lexical access
- break-a-promise → M-C claim-type match + M-A intent router → speech-act/answer-type fit
- meaning of life → M-B concreteness signal + §5 ordering fix → abstract-concept DMN routing
- do-you-get-tired → M-A self-model router + speech-act override fix → self-reference stance
- remember stargazing → M-E self-reference encoder → self-reference effect + intentional encoding
