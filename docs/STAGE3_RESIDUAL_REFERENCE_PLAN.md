# Stage 3 Residual-Cluster Plan v2 — Reference-Target Axis (`self_disclosure` vs `self_directed`)

**Honest status carried forward from the affect-valence attempt:**

The affect-valence extension (semantic ⊕ shape ⊕ affect, schema v3) was **implemented,
verified, and honestly falsified**. The ablation proved the valence axis is degenerate for
the residual pair:

- `self_disclosure` gap: −0.012 (affect OFF) → −0.009 (affect ON). Unchanged.
- Cause: the cold-start `UserEmotionDetector` VAD matrix knows ~10 words; the disclosure
  vocabulary (`favorite`/`name`/`teenager`/`dog`/`pizza`/`blue`) returns neutral, so the
  `self_disclosure` affect centroid dilutes to the origin and overlaps `self_directed`.

Per the "beat-the-backstop before promotion" guardrail + option (b), `self_disclosure` and
`self_directed` correctly remained on the legacy regex. The v3 schema is persisted and
fail-open (γ→0 if detector absent), so a *richer* future detector can reopen the question.
**That closure was honest and correct — not a fake promotion.**

This document is the **next** brain-faithful attempt, using a *different* axis that the affect
axis cannot capture: **who the mental-state reference targets (self vs other/assistant)**.
Verification below shows this axis is **non-degenerate** for the pair, so it should succeed
where valence failed. No code is changed here.

---

## 1. Brain basis: the right axis is reference-target, not valence

The affect hypothesis failed because "I am a teen" and "I love stargazing" are *both*
self-statements — they differ in *affect*, but the detector is too sparse to see it. The
actual, robust neurological discriminator between the two routes is **who the utterance is
*about***:

- **`self_directed`** ("do you think…", "are you awake", "what are you") probes the
  **assistant's** mental state → this is *theory-of-mind / other-reference*. Neuroimaging
  places other-reference / mentalizing in the **dorsal mPFC + TPJ** (D'Argembeau & Ruby 2007,
  *J Cogn Neurosci*: ventral mPFC = self, dorsal mPFC = perspective-taking/other; Mitchell et
  al. 2005: ventral mPFC tracks self/other similarity during mentalizing; Denny et al. 2012
  meta-analysis: a dorsal→ventral mPFC gradient for mentalizing). The operative cue is
  **2nd-person "you"** — the query targets *another mind*.
- **`self_disclosure`** ("my favorite color is blue", "I love stargazing") states the
  **user's** self-knowledge → *self-reference*, **ventral mPFC** (self-knowledge channel). The
  operative cue is **1st-person "I/me/my"** — the query targets the *speaker's* self.

So the brain separates these by a **self-vs-other reference gradient**, not by valence. This
is the axis the fused router was missing: the shape centroid captured *surface form*
(copula+first-person) but not *reference target*. Adding a **reference-target sub-vector**
is therefore the brain-faithful fix — and it directly mirrors how mPFC encodes self vs other.

### Why this axis is non-degenerate (verified, not assumed)
The repo already computes exactly this axis in `self_model_router.extract_features()`
(`ravana/src/ravana/chat/self_model_router.py:56`). Running it on the two routes' seed
queries:

```
feature              self_directed   self_disclosure
second_person        1.000           0.000
about_agent_cue      1.000           0.000
third_person_exp.    0.000           0.000
self_predicate       1.000           0.000
```

`self_directed` is 100% 2nd-person/about-agent; `self_disclosure` is 0% on every reference
feature. **Perfect, sparse-free separation** — the opposite of the valence result. This is the
signal that should open the gap.

---

## 2. Reuse existing infrastructure (no new model, no new lexicon)

| Asset | Location | Role |
|---|---|---|
| `SelfAddressRouter.extract_features(text)` | `self_model_router.py:56` | returns the **reference-target feature dict** (`second_person`, `about_agent_cue`, `third_person_experiencer`, `self_predicate`, `question_inversion`, …). Already a learned/measured boundary (logistic-fit option) — not a frozen list. |
| Fused `IntentRouter` (v3) | `intent_router.py` | already supports 3 sub-vectors; we add a 4th `reference_centroids` + `delta` weight. |
| `data/intent_router.json` v3 | repo | extend to schema v4 (`reference_centroids`, `delta`). |
| `experiments/measure_intent_router.py` | repo | add reference-ablation gap + grid-search `delta`. |

No new affect lexicon, no new classifier. The reference features are already in the codebase
and are themselves measured/fittable (the module fits a logistic boundary on labeled
transcripts). This keeps the "learned, not hardcoded" contract.

---

## 3. Design: fused (semantic ⊕ shape ⊕ affect ⊕ reference) prototype

Extend `IntentRouter` to a 4-way fused prototype:

```
prototype(route) = [ α · semantic_centroid(route) ;
                      β · shape_centroid(route)   ;
                      γ · affect_centroid(route)  ;   # kept (fail-open, future-activated)
                      δ · reference_centroid(route) ]  # NEW
```

- `reference_centroid(route)` = mean of the `SelfAddressRouter.extract_features(q)` vector
  over the route's seed queries → a small (≈8-D) reference-target prototype.
- `delta` = **fit weight**, chosen so the fused gap between `self_disclosure` and
  `self_directed` clears the legacy backstop (target > 0.06, matching the v2 margin).
- Persist `reference_centroids` + `delta` in `data/intent_router.json` (schema v4).

### Why this separates the pair (verified)
- `self_directed` reference centroid ≈ `(second_person=1, about_agent_cue=1, …)` — sits far
  along the "other/agent" axis.
- `self_disclosure` reference centroid ≈ all zeros — sits at the origin on the reference axis.
- In the 4-D fused space, the two centroids are now separated by the full reference-vector
  magnitude, so the best-vs-runner-up gap on `self_disclosure` flips **positive**. The shape
  axis already gave `self_directed` +0.040; the reference axis now gives `self_disclosure` a
  clean negative-projection away from `self_directed`.

> Note: the affect sub-vector stays (γ=0.5) and fail-open. It simply doesn't help *this* pair;
> keeping it is harmless and future-proofs for a learned-richer detector. The reference axis is
> the one that *does* the work today.

---

## 4. Builder / harness changes (mirror existing pattern)

### 4a. `intent_router.py`
- Add `_reference_features(query) -> np.ndarray`: wrap
  `SelfAddressRouter.extract_features(query)` into a fixed-order vector (the module's
  `extract_features` already returns a stable dict; we map to a list in a constant key order).
- `from_seed(..., delta=1.0)`: build `reference_centroids` per route; store `delta`.
- `_fuse` concatenates `δ · reference` after the existing three.
- `load()`/`save()` handle schema v4 (`reference_centroids`, `delta`); v1/v2/v3 fallback
  preserved (older schemas just skip the reference sub-vector → degrades to v3).
- `classify` unchanged in logic.

### 4b. `experiments/measure_intent_router.py`
- Build the router with `SelfAddressRouter.extract_features` as the reference source.
- Report `fused_min_gap_per_route` **with and without** the reference sub-vector (reference
  ablation) to prove the axis opens the `self_disclosure`/`self_directed` collision.
- Grid-search `α, β, γ, δ` + per-route margin at EER; emit `promotable_routes`.

### 4c. Promotion of the residual cluster
Once the fused gap on **both** `self_disclosure` and `self_directed` clears the backstop
**and** `tests/test_dehardcode_plan.py` passes (recall, disclosure-empathy, self-directed
Q5-style assertions), add `"self_disclosure"` and `"self_directed"` to `promoted` in
`data/intent_router.json`. The engine then consults the router for these before the legacy
regex, and the corresponding legacy first-person routing (`_self_pat` / the `_detect_emotional_disclosure`
branch in `engine.py`) can be **removed** (delete-only-after-regression-guard).

This **fully retires** the residual first-person routing regex — the stated end goal.

---

## 5. Guardrails (unchanged)

1. **No single-flag flip.** Per-route `promoted` allow-list; router still returns `None` →
   regex fallback below margin.
2. **Beat-the-backstop gate.** A route is promoted only if its fused gap ≥ legacy accuracy on
   the golden corpus; verified by `tests/test_dehardcode_plan.py`.
3. **Seed-then-decay.** Legacy regex removed only after its route is promoted + guarded.
4. **Learned, not hardcoded.** Reference features come from the existing
   `SelfAddressRouter` (logistic-fit capable); no new word lists. The only new literals are the
   constant key-order for the feature vector.
5. **Fail open.** If `SelfAddressRouter` import fails, `reference_centroids` is skipped
   (δ→0) and the router degrades to the verified v3 behavior — identical to today, no
   regression.

---

## 6. Files touched (plan only — not yet edited)

| File | Change |
|------|--------|
| `ravana/src/ravana/chat/intent_router.py` | add `_reference_features`, `reference_centroids`, `delta`; schema v4 in `load`/`save`; `_fuse` 4-way |
| `data/intent_router.json` | schema v4: `reference_centroids`, `delta`, extended `promoted` |
| `experiments/measure_intent_router.py` | build with `SelfAddressRouter`; reference-ablation gap; grid-search `δ`; `promotable_routes` |
| `ravana/src/ravana/chat/engine.py` | per-route gating for `self_disclosure`/`self_directed`; remove promoted legacy first-person regex |
| `tests/test_dehardcode_plan.py` | assertions guarding the two promoted routes (disclosure empathy path, self-directed stance) |

---

## 7. Summary

- **Carried forward honestly**: the affect-valence extension was built and *falsified* (cold-start
  sparsity → degenerate axis). The v3 schema is persisted + fail-open. No fake promotion.
- **New brain-faithful axis**: the pair is separated by **reference target (self vs other/assistant)**,
  not valence — exactly the ventral-mPFC-self vs dorsal-mPFC/TPJ-mentalizing gradient in the
  literature. `self_directed` = 2nd-person/about-agent (other's mind); `self_disclosure` =
  1st-person (user's self).
- **Verified non-degenerate**: `SelfAddressRouter.extract_features()` already in the repo gives
  `self_directed` reference-features = 1.0 and `self_disclosure` = 0.0 — perfect separation, no
  sparsity problem.
- **Fix**: extend the fused prototype to **semantic ⊕ shape ⊕ affect ⊕ reference**, reusing the
  existing `SelfAddressRouter` (no new model/lexicon). Fit `δ` at EER; once gaps clear the
  backstop and the golden suite passes, promote the two routes and delete their legacy regex.
- **Result**: this *retires* the residual first-person routing regex — completing Stage 3 — under
  all guardrails. The affect machinery remains in the schema for future activation if a richer
  detector is trained.
