# Stage 3 Promotion Plan — Richer (Syntactic-Shape) Features for the First-Person Cluster

**Status of the gap (verified 2026-07-19 by running `experiments/measure_intent_router.py`):**

The `IntentRouter` (Stage 3 / M-A centroid router) is **built and externalized**
(`data/intent_router.json`) but **NOT promoted**. It is flag-gated `use_intent_router=False`
and the engine falls back to the ~15 legacy regex/lists. It is safe today (0 misroutes at
the conservative 0.06 margin — 12/27 corpus queries return `None` → regex) but does not
replace the hardcoded router.

The blocker is the **first-person cluster** — these routes collapse in mean-pooled GloVe
space because their *content words are nearly identical* ("I", "my", "favorite", "remember"):

```
self_disclosure  min_gap = +0.001   (collides with self_directed / remember_store)
episodic_recall  min_gap = +0.009   (collides with remember_store)
remember_store   min_gap = +0.030   (collides with episodic_recall)
self_directed    min_gap = +0.037   (collides with self_disclosure)
```

Promoting at these gaps would regress Q5-remember-stargazing (currently fixed by
`_topic_coverage_pe` + `_subject_head` `_generic` in `engine.py`), the recall routes, and
"my favorite color is blue" (currently `self_disclosure`, not `self_directed`).

Per the plan's guardrail — *"verified to beat the backstop before promotion"* — I did NOT flip
the flag. This document is the plan to make it beat the backstop, then promote it
route-by-route. No code is changed here.

---

## 1. Brain basis: why mean-pooled GloVe fails here, and why shape fixes it

### The failure mode is exactly a known brain-faithfulness gap
Mean-pooled GloVe encodes **lexical semantics** (what words mean). The four colliding routes
share almost all their *content* words, so their semantic centroids are ~indistinguishable
(gaps 0.001–0.037). But the brain does **not** route intent by lexical semantics alone:

- **Self-reference** → medial PFC / ventromedial PFC (mPFC/vmPFC), the "self model"
  (Philippi et al. 2012, *J Cogn Neurosci*; Rudrauf; Wikipedia "Self-reference effect").
- **Episodic recall** ("do you remember…") → hippocampus + posterior cingulate + angular
  gyrus, the autonoetic/recollection system (Spreng/Buckner; D'Argembeau & Salmon 2011).
- **Semantic self-knowledge / disclosure** ("my favorite color is…", "I am a teen") →
  cortical midline, *semantic* (not autonoetic) autobiographical channel.
- **Imperative / command shape** ("remember that…", "keep in mind…") → a distinct syntactic
  structure (2nd-person imperative + matrix-verb-initial), processed via the lateral PFC /
  IFG syntactic pipeline, not the mPFC self-channel.

The four routes are **neurally and structurally separable**. The missing dimension is
**syntactic shape** — the *grammatical form* of the utterance, not its topic words. This is
the "syntactic-shape dimensions" option floated in `docs/DEHARDCODING_REMAINING_PLAN.md`.
Adding it is itself more brain-faithful: the brain routes on **both** semantic content
(ventral) **and** syntactic structure (dorsal stream), not on a word-bag.

### Why a "richer feature" is legitimate, not a hack
Prototype categorization (Rosch; Zhang 2020) routes on the **full stimulus representation**.
Our current router uses only the semantic sub-vector. Extending the prototype vector to
include a learned syntactic-shape sub-vector is the principled fix — it makes the prototype
*complete*, the same way `pos_model.py` already adds a distributional POS sub-code and
`salad_classifier` adds a structural sub-code. We are not adding rules; we are widening the
learned representation.

---

## 2. Design: fused (semantic ⊕ syntactic-shape) prototype vector

Keep the existing nearest-centroid mechanism. Change only the **prototype vector** and the
**builder**:

```
prototype(route) = [ α · semantic_centroid(route) ;   (mean-pooled GloVe, as today)
                      β · shape_centroid(route)   ]    (NEW: syntactic-shape dims)
```

- `semantic_centroid` built exactly as today (`_mean_pool` over content words, stop-weighted).
- `shape_centroid` built from a small set of **learned shape dimensions** (below), each a
  normalized scalar or low-dim vector, pooled per route.
- `α`, `β` are **fit weights** (not hardcoded constants) chosen to maximize per-route
  separation on the calibration corpus — i.e. the fused gap must beat the legacy regex's
  accuracy. Persist `alpha`/`beta` in `data/intent_router.json`.

### 2a. The syntactic-shape dimensions (all learned / distributional, no new regex)

1. **Person/agent marker (self vs other vs command)** — a learned 2-D code:
   - Build from the *existing* `pos_model.py` POS centroids + a "first-person pronoun"
     prototype. We already have `_detect_emotional_disclosure`'s first-person signal
     (`response_gen.py:3432-3622`) and `syntactic_assembly`. Seed a `person_centroid`
     = mean GloVe of `{"i","me","my","we","our","myself"}`; and a `command_centroid`
     = mean GloVe of imperative cue verbs `{"remember","keep","tell","recall","note"}`.
     Project each query onto these → 2 scalars. This separates `remember_store`
     (high command) from `episodic_recall` (high person, ~0 command) and from
     `self_disclosure` (high person, low command).
2. **Clause/utterance shape** — count + position features the engine already computes
   cheaply at tokenization (no GloVe needed, but we *learn* their weight, not hardcode a rule):
   - `imperative_head`: 1 if the first content token is a `pos_model` "verb" centroid match
     AND a 2nd-person/imperative frame (reuses `syntactic_assembly.bind_to_sentence`).
   - `has_remember_verb`: projection on `command_centroid` (above) — learned, not a string test.
   - `be_copula_shape`: presence of copula "am/is/are" + predicate adjective/noun
     ("i am a teen", "my favorite color is blue") → flags `self_disclosure` semantic form
     vs `self_directed` ("do you think…", interrogative).
   - `interrogative_shape`: auxiliary-initial ("do/are/can/is …") → flags `self_directed`
     and `factual_yesno` and `episodic_recall` ("do you remember…") as *questions*,
     separated from declarative `self_disclosure` / `remember_store`.
3. **Tense/aspect marker** — `past` ("did","told","said","was") vs `present` vs `future`
   (`will`,`would`). `episodic_recall` is past/interrogative; `remember_store` is
   imperative-present. This is the cleanest single separator of recall vs store.

> None of 2a is a hardcoded routing rule. Each is a **learned projection** (centroid dot
> product) or a **learned-weight** structural feature. The *only* new hardcoded seeds are the
> tiny pronoun/verb cue **seed word lists used to BUILD centroids** — exactly the seed
> discipline already used in `intent_router.py` (`_SEED_QUERIES`) and `pos_model.py`
> (`_seed_from_constants`). Day-one behavior is preserved; the model then refines.

### 2b. Builder changes (`intent_router.py`)
- Add `_shape_features(query, glove_fn, pos_fn) -> np.ndarray` returning the concatenated
  shape sub-vector (normalized).
- `from_seed` builds both `semantic_centroids` and `shape_centroids`; stores `alpha`,`beta`.
- `classify` concatenates `α·sem` and `β·shape`, then nearest-centroid as today.
- Persist schema: `{semantic_centroids, shape_centroids, alpha, beta, margins}`.

### 2c. Fit the fusion weights + margins (`experiments/measure_intent_router.py`)
- Extend the harness to (a) compute fused gaps per route, (b) grid-search `α,β` and
  per-route `margin` to **maximize agreement vs legacy regex** and, critically, to **maximize
  the min fused gap on the 4 colliding routes** until each clears its legacy backstop.
- Report a new dashboard `_intent_router_calib.json` with `fused_min_gap_per_route` and a
  `promotable_routes` list = routes whose fused gap now beats the regex on the golden corpus.

---

## 3. Promotion protocol (route-by-route, guardrail-preserved)

Promotion = the engine consults `IntentRouter` for route **R** *before* the legacy regex, but
only if R is in the verified set. Implement as a per-route allow-list
(`_PROMOTED_ROUTES` loaded from `data/intent_router.json["promoted"]`), not a single flag flip.

Order (lowest-risk first, each gated by the golden regression suite
`tests/test_dehardcode_plan.py`):

1. **definition_seeking, philosophical_abstract, moral_advice, conditional, procedural,
   humor, chitchat, factual_yesno** — these already have fused gaps ≥0.03 and the corpus shows
   them separable; promote once the harness confirms ≥ legacy accuracy.
2. **self_directed** (gap 0.037 → should widen via `interrogative_shape`).
3. **remember_store** (gap 0.030 → widen via `command_centroid` + `imperative_head`).
4. **episodic_recall** (gap 0.009 → widen via `past` tense + `person_centroid`).
5. **self_disclosure** (gap 0.001 → widen via `be_copula_shape`; this is the hardest, promote last).

Each promotion step: run `tests/test_dehardcode_plan.py`; if any assertion on
`_topic_coverage_pe` / recall / disclosure regresses, revert that route to regex and note why.

### What gets deleted when a route is promoted
The corresponding legacy list/regex in `engine.py` is **removed** (e.g. the `_self_pat`,
`QUERY_PATTERNS` self/recall/disclosure entries, the paradox-list self entries). This is the
actual de-hardcoding payoff. Removed code must be covered by a regression assertion first.

---

## 4. Option (b) fallback — if fused features still can't clear the gap

If, after fusion, `self_disclosure` vs `self_directed` remain within 0.03 (possible: "I am
bored" is both interrogative-adjacent and disclosure-shaped), we accept the router as the
**safe prototype layer** it already is and keep the legacy regex as the operational router for
that residual cluster only. This is explicitly permitted by `docs/DEHARDCODING_REMAINING_PLAN.md`
option (b). The router still earns its place as a *confidence prior* feeding the existing
`_topic_coverage_pe` gate, rather than as the sole router. Either way, no regression.

---

## 5. Files touched (plan only — not yet edited)

| File | Change |
|------|--------|
| `ravana/src/ravana/chat/intent_router.py` | add `_shape_features`, fused prototype, `alpha/beta`, `shape_centroids`; schema v2 |
| `data/intent_router.json` | new schema: `semantic_centroids`, `shape_centroids`, `alpha`, `beta`, `margins`, `promoted` |
| `experiments/measure_intent_router.py` | fused-gap grid search, `promotable_routes`, new dashboard |
| `ravana/src/ravana/chat/engine.py` | per-route allow-list gating; remove promoted legacy lists/regex |
| `tests/test_dehardcode_plan.py` | assertions guarding each promoted route (recall, disclosure, remember_store) |

---

## 6. Summary

- **Confirmed**: Stage 3 router built but not promoted; the only blocker is the first-person
  cluster (gaps 0.001–0.037), exactly as reported.
- **Root cause**: router uses only lexical-semantic prototypes; the colliding routes differ by
  **syntactic shape**, not word meaning — and the brain routes on shape too (mPFC self-model vs
  hippocampus recollection vs IFG imperative), so this is a brain-faithfulness gap, not just a
  metric gap.
- **Fix**: widen the prototype to a **fused semantic ⊕ learned syntactic-shape vector**
  (person/command centroids, interrogative/copula/tense shape, all learned — no new rules),
  fit `α,β` + per-route margins at EER on the calibration corpus, then **promote route-by-route**
  behind the golden regression suite, deleting the legacy regex as each route clears the backstop.
- **Guardrail**: single-flag flip is forbidden; promotion is per-route and regression-gated;
  if fusion can't separate `self_disclosure`, option (b) keeps the regex as operational router
  for that residual cluster (no regression either way).
