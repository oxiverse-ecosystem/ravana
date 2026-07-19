# Stage 3 Residual-Cluster Plan — Affect-Valence Projection for `self_disclosure` vs `self_directed`

**Status of the residual gap (as reported, option b):**

After the Stage 3 fused router (semantic ⊕ syntactic-shape, `data/intent_router.json` v2
schema with `semantic_centroids` + `shape_centroids` + `alpha`/`beta` + `promoted`), the
first-person cluster is separable *except*:

- **`self_disclosure` vs `self_directed` cannot be separated even with fusion** (reported gap
  ≈ −0.012). Both share the **copula + first-person** shape ("I am …", "do you …"), so the
  shape dimension collides rather than separates them.
- Consequence: the legacy regex remains the **operational router** for this residual
  first-person cluster. No regression either way (the router already defers to regex when
  uncertain, and `self_disclosure`/`self_directed` are simply not in `promoted`).

The suggested further extension (per `docs/STAGE3_PROMOTION_PLAN.md` §4): a **richer signal —
an affect-valence projection** that distinguishes "I am sad" (affect) from "I am a teen"
(semantic self-knowledge). This document is the full brain-faithful plan for that extension.
No code is changed here.

---

## 1. Brain basis: why valence is the missing axis

The colliding pair is genuinely ambiguous on semantics *and* syntax, but they are **not**
ambiguous on affect:

- **"I am sad" / "I love stargazing"** = *affective* self-statement. Encoded by the
  **OFC / limbic / vmPFC affective channel** — the brain appraises the *valence* of
  self-related material (De Pisapia et al. 2019, *Brain Imaging & Behavior*, TMS/fMRI:
  mPFC codes the **affective dimension** of self-referential info, not just self-vs-other;
  Iravani et al. 2024, *J Neurosci*, iEEG: OFC populations separate valenced self-judgments
  from neutral autobiographical/trait retrieval; Benoit et al. 2014: OFC assesses the
  *emotional value* of self-related stimuli).
- **"I am a teenager" / "my name is Sam"** = *semantic* self-knowledge (trait/identity),
  coded by the **vmPFC→oPFC trait-judgment stream** with near-neutral valence (Iravani 2024:
  autobiographical-memory vs self-trait neurons are *non-overlapping*; trait judgments are
  cognitively, not affectively, loaded).

So the brain's **valence signal is the natural discriminator** between these two self-channels.
Adding a learned valence projection to the prototype vector is therefore the brain-faithful
fix — it mirrors how OFC tags self-statements on a valence axis, not a rule list. This is the
same dimensional-affect principle already used elsewhere in the repo (VAD emotion engine,
`safety_valence.py`).

Critically: **the repo already contains the exact learned signal we need** — no new model
required, just reuse.

---

## 2. Reuse existing infrastructure (no new model to train)

| Existing asset | Location | What it gives us |
|---|---|---|
| `UserEmotionDetector.detect(text)` | `ravana/core/mirror.py:327` | returns **(valence, arousal, dominance)**, *learned* via Hebbian VAD matrix (cold-starts from a ~10-word seed, grows by interaction). No hardcoded affect lists. |
| `_detect_emotional_disclosure(text)` | `response_gen.py:3432` | already detects first-person + affect; confirms the affect signal is the right discriminator for disclosure. |
| `data/safety_valence.json` + `safety_valence.py` | repo | precedent for persisting a learned valence model to `data/*.json` with EER-fit margins — exact template to mirror. |
| Fused `IntentRouter` schema | `intent_router.py` (v2) | already supports `semantic_centroids` ⊕ `shape_centroids`; we add a third `affect_centroids` sub-vector + `gamma` weight. |

`UserEmotionDetector.detect()` is the affect-valence projection. We do **not** build a new
classifier; we project each query through this already-learned detector to get a 3-D VAD
vector and append it (scaled by a fit `gamma`) to the prototype. This keeps the "learned,
not hardcoded" contract — the valence values come from the Hebbian matrix, the *only* new
seed is the tiny VAD seed already present in the detector.

---

## 3. Design: fused (semantic ⊕ shape ⊕ affect) prototype vector

Extend `IntentRouter` from 2 sub-vectors to 3:

```
prototype(route) = [ α · semantic_centroid(route) ;
                      β · shape_centroid(route)   ;
                      γ · affect_centroid(route)  ]     # NEW
```

- `affect_centroid(route)` = mean of `UserEmotionDetector.detect(q)` over the route's seed
  queries → a 3-D (valence, arousal, dominance) prototype.
- `gamma` = **fit weight** (not hardcoded), chosen so the fused gap between `self_disclosure`
  and `self_directed` clears the legacy backstop.
- Persist `affect_centroids` + `gamma` in `data/intent_router.json` (schema v3).

### Why this separates the pair
- `self_disclosure` seed ("i love stargazing", "i'm happy", "my dog is called rex" — pet
  attachment is positive-valence) → affect prototype has **nonzero valence** (positive or
  negative depending on examples).
- `self_directed` seed ("do you ever get tired", "what do you think", "are you awake") →
  these are *questions about the assistant*, not user self-statements → `detect()` returns
  **near-neutral (0.0, 0.3, 0.5)** because the affect words belong to the assistant, not the
  user. The VAD projection on the *user's* utterance is flat.
- Net: `self_disclosure` sits off-origin on the valence axis; `self_directed` sits at origin
  → the two centroids separate in the 3-D affect subspace even though they collide on
  semantic+shape.

> Edge note: "I am bored" is both copula-shaped AND mildly negative-valence, so it naturally
> lands in `self_disclosure` (affect) — which matches the engine's existing
> `_detect_emotional_disclosure` behavior (`response_gen.py:3619` routes first-person affect
> to the empathic responder). Consistent, not a regression.

---

## 4. Builder / harness changes (mirror existing pattern)

### 4a. `intent_router.py`
- Add `_affect_features(query, detector) -> np.ndarray(3)`:
  `np.array(detector.detect(query), dtype=float)` (already normalized to [−1,1]×[0,1]×[0,1]).
- `from_seed(glove_fn, detector, ..., gamma=1.0)`: build `affect_centroids` per route;
  store `gamma`.
- `_fuse` concatenates `γ · affect` after `α·sem` and `β·shape`.
- `load()`/`save()` handle schema v3 (`affect_centroids`, `gamma`); v1/v2 fallback preserved.
- `classify` unchanged in logic — just routes in the wider fused space.

### 4b. `experiments/measure_intent_router.py`
- Build the router with the live `UserEmotionDetector` (the engine already constructs one;
  reuse `CognitiveChatEngine`'s detector or a fresh `UserEmotionDetector()`).
- Report `fused_min_gap_per_route` **with and without** the affect sub-vector, proving the
  affect axis is what opens the `self_disclosure`/`self_directed` gap (target: gap > legacy
  backstop margin, e.g. > 0.06).
- Grid-search `α, β, γ` + per-route margin at EER; emit `promotable_routes`.

### 4c. Promotion of the residual cluster
Once the fused gap on `self_disclosure` clears the backstop **and** `tests/test_dehardcode_plan.py`
passes (recall, disclosure, remember_store assertions), add `"self_disclosure"` and
`"self_directed"` to `promoted` in `data/intent_router.json`. The engine then consults the
router for these before the legacy regex, and the corresponding legacy first-person regex
(`_self_pat` / `_detect_emotional_disclosure` routing branch in `engine.py`) can be **removed**
(delete-only-after-regression-guard).

This **fully retires** the residual first-person regex — the stated goal of the extension.

---

## 5. Guardrails (unchanged from prior plans)

1. **No single-flag flip.** Promotion is per-route via the `promoted` allow-list; the router
   still returns `None` → regex fallback when below margin.
2. **Beat-the-backstop gate.** A route is promoted only if its fused gap ≥ legacy accuracy on
   the golden corpus; verified by `tests/test_dehardcode_plan.py`.
3. **Seed-then-decay, never delete cold.** Legacy regex removed only after its route is
   promoted and regression-guarded.
4. **Learned, not hardcoded.** The affect values come from the Hebbian `UserEmotionDetector`;
   the only new literal is the reuse of its existing VAD seed. No new affect word lists.
5. **Fail open.** If `UserEmotionDetector` is unavailable, `affect_centroids` is skipped
   (γ→0) and the router degrades to the already-verified semantic⊕shape behavior — identical
   to today, no regression.

---

## 6. Files touched (plan only — not yet edited)

| File | Change |
|------|--------|
| `ravana/src/ravana/chat/intent_router.py` | add `_affect_features`, `affect_centroids`, `gamma`; schema v3 in `load`/`save`; `_fuse` 3-way |
| `data/intent_router.json` | schema v3: `affect_centroids`, `gamma`, extended `promoted` |
| `experiments/measure_intent_router.py` | build with `UserEmotionDetector`; report affect-ablation gap; grid-search `γ`; `promotable_routes` |
| `ravana/src/ravana/chat/engine.py` | per-route gating for `self_disclosure`/`self_directed`; remove promoted legacy first-person regex |
| `tests/test_dehardcode_plan.py` | assertions guarding the two promoted routes (disclosure empathy path, self-directed Q5-style) |

---

## 7. Summary

- **Confirmed**: the only remaining un-promoted routes are `self_disclosure` and
  `self_directed`; they collide on semantic+shape (copula+first-person), gap ≈ −0.012, so the
  legacy regex stays operational for them. No regression.
- **Brain basis**: OFC/vmPFC encodes self-statements on a **valence axis** — affective
  self-statements ("I am sad") are valenced, semantic self-knowledge ("I am a teen") is
  near-neutral. Valence is the brain's natural discriminator for exactly this pair.
- **Fix**: extend the fused prototype to **semantic ⊕ shape ⊕ affect**, using the repo's
  already-learned `UserEmotionDetector.detect()` (VAD) as the affect projection — no new
  model, no new hardcoded lists. Fit `γ` at EER; once the fused gap clears the backstop and
  the golden suite passes, promote the two routes and delete their legacy regex.
- **Result**: this *fully retires* the residual first-person routing regex — the stated end
  goal of option (b)'s extension — while preserving all guardrails (per-route promotion,
  regression-gated, fail-open).
