# Stage 3 Final Remaining-Items Plan — `self_directed` Clean Promotion + Optional Axis Boosts

**Carried-forward honest state (verified 2026-07-19 from `data/intent_router.json` v4 + `_intent_router_calib.json`):**

The 4-way fused router (semantic ⊕ shape ⊕ affect ⊕ reference, schema v4) is built, externalized,
and empirically validated. Reference ablation proved the axis works: `self_disclosure`
+0.015→+0.031, `self_directed` +0.012→+0.044. Promotion is per-route, regression-gated.

Current `promoted = [conditional, factual_yesno, self_disclosure]`. The residual first-person
**disclosure** regex (`_self_pat`) is **retired** — the concrete de-hardcoding win.

Remaining items (all correctly still on the legacy regex today, because their fused gap did
not clear the 0.02 backstop — the guardrail working as designed):

| Route             | v4 fused gap | Why on regex | Brain-faithful next step |
|-------------------|-------------|--------------|--------------------------|
| `self_directed`   | +0.044      | router *separates* it but the agent-address block is a **compositional self-answering cluster** (`_route_self_query`), not a single boolean gate; wiring the router in as a *replacement* caused empty responses → reverted (no fragile promotion) | **Clean promotion path** (this plan §1): router as a *pre-admit* to the existing block, never a replacement |
| `definition_seeking` | <0.02   | overlaps `factual_yesno` (both WH-questions about facts); semantic centroids too close | Optional axis boost §2 (predicate-shape / wh-word force) |
| `episodic_recall` | +0.023     | just below 0.02; collides with `remember_store` (both user-memory) | Optional axis boost §3 (retrieval-vs-encoding tense/force) |
| `remember_store`  | +0.005     | essentially unresolved from `episodic_recall` | Optional axis boost §3 |

This document gives (1) the concrete, safe promotion path for `self_directed` — the only
remaining route the router *already separates* — and (2) brain-grounded optional extensions for
the other three, with the explicit understanding that they stay on regex until their gap clears
the backstop. No code is changed here.

---

## 1. `self_directed` — clean promotion (the real remaining win)

### 1a. The regression cause (diagnosed, not guessed)
`self_directed` maps to `_route_self_query` (`engine.py:3372`), which is a **compositional
self-answering block**: it matches specific sub-regex (`name`, `who are you`, `what are you`,
`what can you do`, bare self-subject) and *composes* a distinct answer for each. The prior
attempt wired the router in as a **replacement** of this block's admission logic, so when the
router said `self_directed` but the block's specific sub-regex didn't fire, no answer was
produced → empty response. That is the "no fragile promotion" guardrail firing correctly.

### 1b. Brain basis — the router should be a *gate*, not a *replacement*
The self/other boundary is the **TPJ / mirror-neuron self-other distinction** (the block's own
docstring cites it). The brain does not *replace* its self-model retrieval with a single
categorical decision; the categorical "this is about me" signal **admits** the query to the
self-model system, which then retrieves the specific content. So the router's job is to
**confirm admission** ("this utterance is about the agent's mind"), exactly like
`_router_says("self_disclosure", ...)` already does for `_is_self_disclosure_stmt`. The block's
internal compositional logic stays intact.

### 1c. The fix (mirrors the `self_disclosure` promotion exactly)
Add ONE line to the §4 self/other gate (`engine.py:4166-4179`), *before* the existing
`_route_self_query` call, that **pre-admits** via the router without removing the regex:

```python
# Router-driven pre-admit (promoted self_directed): the reference axis confirms
# the utterance is about the agent's mind (2nd-person/about-agent). This ADDS
# coverage — it never replaces _route_self_query's compositional answering, so it
# cannot regress. Falls through to the regex block when the router is silent/unpromoted.
if (self.use_intent_router
        and self._router_says("self_directed", user_input)):
    _self_ans = self._route_self_query(user_input)
    if _self_ans is not None:
        self._last_strategy = "self_model"
        self._last_responses.append(_self_ans)
        ...
        return _self_ans
# (existing _route_self_query call follows unchanged)
```

Key safety properties:
- The router only *calls* `_route_self_query`; the block's regex still does the actual answering
  (name/identity/favorite branches). **No empty-response regression possible** — if the block
  returns None, we fall through to the existing legacy path exactly as today.
- `_router_says` returns True **only for promoted routes**, so this is inert unless
  `self_directed` is in `promoted`.
- At `use_intent_router=False` (default), the whole block is skipped → zero behavior change.

### 1d. Promotion step
1. Add the pre-admit gate (1c).
2. Add `"self_directed"` to `promoted` in `data/intent_router.json`.
3. Run `tests/test_dehardcode_plan.py` — must pass, including the "do you ever get tired" →
   self-model-stance assertion (the exact regression from before, now fixed).
4. Run the live 8-query smoke from the prior report; confirm `do-you-get-tired` returns a
   self-model stance (not empty, not a web def).

This **retires the operational need for the agent-address regex as the *admission* mechanism**
(the router now admits agent-mind queries), while the block's answering regex stays as the
content retriever. That is the brain-faithful, non-fragile completion of `self_directed`.

---

## 2. `definition_seeking` (optional, not promotable today) — predicate-shape axis

### Brain basis
`definition_seeking` ("what is gravity") and `factual_yesno` ("is a whale a mammal") are
semantically near-identical (both WH-questions about facts) — their centroids collide. The
brain distinguishes **query intent** by **illocutionary force / predicate type**: a *copula
definition* ("what is X") seeks a concept's meaning; an *auxiliary verification* ("is X a Y")
seeks a truth value. This is a **wh-word + predicate-shape** signal — already partially in the
shape sub-vector but diluted at current `beta`.

### Proposed extension (only if gap must be closed)
- Strengthen the shape sub-vector's **predicate dimension**: add a `copula_def` feature
  (query begins with "what is/are/does X mean" vs "is/are/can X") and a `wh_word` feature
  (what/who=definition vs is/can/do=verification). These are **learned projections**, not new
  regex — seed from the existing `_SEED_QUERIES` definition vs yesno lists.
- Re-grid `beta` + margins. If `definition_seeking` gap clears 0.02 *and* golden suite passes,
  promote it. Until then: stays on regex (correct).

---

## 3. `episodic_recall` & `remember_store` (optional) — retrieval-vs-encoding force

### Brain basis
Both are **user-memory** routes (hippocampal / medial-temporal system; CLS theory, McClelland —
hippocampus = fast episode encoder/retriever; neocortex = slow semantic consolidation). The
brain distinguishes **retrieval vs encoding** by **tense + illocutionary force**:
- `episodic_recall` ("do you remember…", "what did i tell you") = **past + interrogative**
  (retrieval intent, autonoetic).
- `remember_store` ("remember that…", "keep in mind…") = **imperative + present** (encoding/
  compliance intent, driven by IFG lateral-PFC instruction-following).

The shape sub-vector already has a `tense`/`imperative_head` dimension, but `beta` is too low
to separate them (recall +0.023, store +0.005).

### Proposed extension (only if gaps must be closed)
- **Boost the force/tense dimensions** of the shape sub-vector: explicitly seed
  `remember_store` with imperative-cue projection (`command_anchor`, already built in v2) and
  `episodic_recall` with past-tense + 2nd-person-interrogative projection. The `command_anchor`
  already exists in `intent_router.py` (`_COMMAND_WORDS`); it just needs to be weighted more
  heavily for these two routes (raise `beta` or add a route-specific shape gain).
- Optionally add a **retrieval recency/frequency signal** from the actual memory store
  (`belief_store`): a query that references a previously stored user fact ("what did i tell you
  about my cat") gets a retrieval-force boost — brain-faithful (retrieval cued by reactivation).
- Re-grid. If both gaps clear 0.02 *and* golden suite passes, promote. Until then: stay on
  regex (correct — promoting would regress Q5/break-promise fixes).

---

## 4. Guardrails (unchanged, all preserved)

1. **No single-flag flip.** Per-route `promoted` allow-list; router returns None → regex
   fallback below margin.
2. **Beat-the-backstop gate.** A route is promoted only if its fused gap ≥ legacy accuracy on
   the golden corpus; verified by `tests/test_dehardcode_plan.py`.
3. **No fragile promotion.** `self_directed` is promoted as a *pre-admit gate*, never as a
   replacement of `_route_self_query` (the prior regression's lesson, now encoded).
4. **Learned, not hardcoded.** All new shape/predicate features are learned projections seeded
   from existing seed lists; no new frozen regex.
5. **Fail open.** `use_intent_router=False` (default) → entire router path inert, behavior
   identical to legacy.

---

## 5. Files touched (plan only — not yet edited)

| File | Change |
|------|--------|
| `ravana/src/ravana/chat/engine.py` | §4 gate: add router-driven `self_directed` **pre-admit** to `_route_self_query` (mirrors `self_disclosure` pattern); never replaces the block |
| `data/intent_router.json` | add `"self_directed"` to `promoted` (v4 stays) |
| `experiments/measure_intent_router.py` | (optional §2/§3) boost shape predicate/force dims; re-grid `beta`; report gaps for def/recall/store |
| `tests/test_dehardcode_plan.py` | assertion: "do you ever get tired" → self-model stance (regression guard for the new gate) |

---

## 6. Summary

- **Confirmed**: v4 router built + validated; `self_disclosure` promoted + `_self_pat` retired.
  `self_directed` is *separated* (+0.044) but not promoted because its agent-address block is
  compositional; `definition_seeking`/`episodic_recall`/`remember_store` are below the 0.02
  backstop and correctly stay on regex. The prior "no fragile promotion" revert was the
  guardrail working.
- **Real remaining win (§1)**: promote `self_directed` as a **pre-admit gate** to the existing
  `_route_self_query` — the router confirms "about the agent's mind" (TPJ self-other boundary)
  and the block's compositional logic still answers. This cannot empty-response-regress and
  completes `self_directed` cleanly.
- **Optional extensions (§2/§3)**: `definition_seeking` needs a predicate-shape boost
  (copula-def vs auxiliary-verify); `episodic_recall`/`remember_store` need a retrieval-vs-
  encoding force boost (past-interrogative vs imperative-present, CLS/hippocampus basis). Both
  are brain-grounded but stay on regex until their gap clears 0.02 — no forced promotion.
- **End state after §1**: every route the router can separate is promoted; the only routes on
  regex are those whose prototype gap genuinely doesn't clear the backstop — exactly the
  guardrail's intended final state. No fake closure.
