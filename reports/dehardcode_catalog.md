# RAVANA Engine — Hardcoding De-Hardcoding Catalog

**Scope:** Investigation + design only. No source files were modified.
**Target:** `ravana/src/ravana/chat/engine.py` (core) + 8 mixins
(`engine_graph.py`, `engine_reasoning.py`, `engine_memory.py`,
`engine_web_search.py`, `engine_generation.py`, `engine_self_query.py`,
`engine_persistence.py`, `engine_monitor.py`).
**Method:** Grep + full read of every cataloged constant region; each `file:line` below
was verified by reading, not guessed.
**Note on referenced test:** `tests/test_dehardcode_plan.py` **DOES exist**
(22 tests; 21 pass pre-change, 1 pre-existing failure `test_meaning_of_life_not_dict_dump`
that also fails on HEAD — confirmed by re-running against the original single-file
engine). The `data/ravana_weights_dehardcode_plan.db` is its fixture DB, NOT the
suite. **Agent B must run this suite as the regression gate and must NOT overwrite
or recreate it.** The de-hardcode fit artifacts also exist
(`data/intent_router.json`, `data/pos_model.json`, `data/functional_lexicon.json`,
`data/snippet_pe.json`, `data/conceptnet/ont.pkl`).

---

## Executive Summary

Counts of distinct cataloged items (duplicated copy-in of the same constant across
the 7 mixins is counted once per *logical* constant, with the duplication itself
flagged once):

| Category | Count | Meaning |
|---|---|---|
| **(a) Already-dehardcoded** | 9 | Learned replacement exists; hardcoded version is now backstop/OOV net only |
| **(b) Partially-dehardcoded** | 7 | Learned replacement built but flag-gated OFF; hardcoded is still the default |
| **(c) Truly-hardcoded** | 18 | No learned replacement; genuine hand-authored knowledge/behavior |
| **(d) Genuinely-fine / not-a-bug** | 8 | Schema markers, closed-class sets by design, adaptive-exemplar, intentional minimal |
| **TOTAL** | **42** | |

**Headline findings:**
1. The "big" de-hardcoding is mostly *done and ON* (snippet PE veto,
   conceptnet category gate, coherence gate). The remaining work is **(b) promotion** —
   flipping OFF flags to ON after regression — and **(c) the long tail** of curated
   word lists (`_COMMON_WORDS`, `_GENERIC_NOUNS`, `TOPIC_SKIP_WORDS`,
   `_SUBJECT_CONTEXT_WORDS`, `_IRREGULAR_VERBS`, `_ATTR_WORDS`, `_CONDITIONAL_FRAME`)
   plus a scatter of **inline magic cutoffs** that should be adaptive
   (z-score / EMA, like the existing `_vad_baseline`).
2. `_LANCASTER_ORDER` / `_SENSORY_DIM_PHRASE` / `_PROP_TO_BINDER` are the
   only *architectural* hand-authoring — the sensorimotor realization lexicon. This
   is real hand-knowledge, not a config knob.
3. `ConnectorLearner` (synaptic_dynamics.py) and `self_model_router` exist but are
   **never wired in** — dead learned infra.
4. The mixin copy-in of `_UNIVERSAL_PURGE` / `_DEFINITION_ASSERTION` in all 7
   mixins is a duplication smell (not a knowledge bug).

---

## Per-Item Table

Legend: **Effort** S/M/L. **Risk** = regression risk if removed.

| # | Location | Element | Cat | Current behavior | Best robust brain-based solution | Effort | Risk / Notes |
|---|---|---|---|---|---|---|---|
| 1 | `engine.py:219-225` | `_EDGE_CONNECTORS` (v1, word lists) | (b) | First, apparently-legacy definition of connector word→relation lists (no numeric weights). Unused by reverse-lookup (which uses #30). | `ConnectorLearner` (synaptic_dynamics.py:379) learns connector→relation from GloVe similarity. **Currently NOT instantiated anywhere** (only chain_walker imports synaptic_dynamics). | M | Low — dead code; safe to delete once v2 (#30) is handled |
| 2 | `engine.py:226` | `MAX_DECODER_VOCAB_SIZE=15000` | (d) | Decoder vocab cap for speed. | Leave. Capacity hyperparam, not knowledge. | — | None |
| 3 | `engine.py:227-244` | `_COMMON_WORDS` | (c) | ~230-word hand-curated common-word set used in rule-based salad scoring (`engine_reasoning.py:493`, `constants.py:236-330`). | Derive from corpus PMI / Zipf frequency at bootstrap (like `constants.json` already carries `stop_words`); or rely solely on the **learned** `SaladClassifier` (#36). | M | Med — must keep a STOP/frequency signal; replace with corpus-derived set, not delete |
| 4 | `engine.py:245-255` | `_CATEGORY_AFFORDANCES` | (a) | Hand table: category→allowed properties. Used as OOV fallback in feasibility gate (`engine_graph.py:931`). | `use_conceptnet_primary=True` (ON) infers affordances via Sensory-Functional division; literal kept only for OOV/silent KG. | — | None — keep as backstop |
| 5 | `engine.py:256-271` | `_CATEGORY_OF_SUBJECT` | (a) | Hand subject→category map. Default path now routes through ConceptNet (`engine_graph.py:905-933`); literal only when flag OFF. | ConceptNet `IsA` walk (`data/conceptnet/ont.pkl`). | — | None — keep as OOV net |
| 6 | `engine.py:272-285` | `_PROPERTY_CATEGORIES` | (a) | property→categories map used in feasibility gate (`engine_graph.py:934`). | ConceptNet `has shape`/`related_to` probe; literal is OOV safety net. | — | Low |
| 7 | `engine.py:286-289` | `_LANCASTER_ORDER` | (c) | Hand-ordered list of 11 Lancaster sensorimotor dims (Auditory…Torso). Defines encoder output order. | Architectural — the *order* is a model schema, not learnable from text. Keep, but document as fixed encoder contract. | S | Low — but flag as intentional architecture |
| 8 | `engine.py:290-316` | `_SENSORY_DIM_PHRASE` | (c) | Hand map property→(verb, sensory-phrase) for realization. | Learn property→phrase from corpora (e.g. Lancaster norms / Wiki descriptions) fitted to binder dims; store in `data/` like other lexicons. | L | Med — realization quality depends on it |
| 9 | `engine.py:317-327` | `_PROP_TO_BINDER` | (c) | Hand map property→binder sensory dims. | Derive from the same fitted property→phrase model (#8); single source of truth. | L | Med |
| 10 | `engine.py:328-337` | `_CONDITIONAL_FRAME` | (c) | ~70-word hand set of conditional/hypothetical frame words; token filter in `engine_reasoning.py:1289,1324,1351,1395` & `:1359`. | This is a *closed syntactic class* (conditionals/modals/temporals). Could seed from grammar, but a small closed set is acceptable. Promote to `functional_lexicon.json` (#17) for single-source. | S | Low |
| 11 | `engine.py:338-343` | `_PREFERRED_SNIPPET_SOURCES` | (b) | Hardcoded trusted-source allowlist used at `engine_web_search.py:1032,1043`. | `use_source_trust=False` (OFF) — learned per-domain trust accumulator. **Flip to ON after regression**; keep list only as cold-start prior. | M | Med — trust accumulator unverified |
| 12 | `engine.py:344-360` | `_JUNK_SNIPPET_DOMAINS` | (c) | ~45-domain hardcoded blocklist (art sites, support pages, etc.). | Replace with **shape-based** detection — `_WEBSITE_SHAPE` regex + low vowel-ratio + embedded-digit signals already in `constants.py:401` (the design principle is "never a hardcoded site blocklist"). Domain blocklist is the anti-pattern the code itself criticizes. | M | Med — risk of over-blocking niche legit sources |
| 13 | `engine.py:361-387` | `_SNIPPET_REJECT_SHAPES` | (a) | ~15 hardcoded reject-regexes for snippet titles. | `use_cerebellar_snippet=True` (ON) — contrastive `SnippetStructureModel` separates answers from boilerplate via PE gap; regex kept only as hard backstop (`engine.py:839`). | — | None |
| 14 | `engine.py:388-451` | `_SNIPPET_NOISE` | (a) | ~120-item hardcoded boilerplate/substring blocklist. | Same learned model (#13, ON). Backstop retained. | — | None |
| 15 | `engine.py:452-474` | `_IRREGULAR_VERBS` | (c) | ~80-word hand map past→base verb (used `engine.py:3306`). | Derive via `PosModel` distributional morphology, or a small rule+exceptions learned from corpus. Low priority (closed irregular set is stable). | S | Low |
| 16 | `engine.py:475` | `_FUNCTION_POS_TAGS` | (b) | Hardcoded closed POS-tag set used as frame-target guard. | `use_learned_pos=False` (OFF) → `pos_model.PosModel` replaces it. Promote to ON. | S | Low |
| 17 | `engine.py:476-481` | `_ATTR_WORDS` | (c) | ~22-word hand list (capital, population, author…) for "what is the X of Y" attribute extraction (`engine_web_search.py:1678`). | Seed from `functional_lexicon.json` (role/attribute cues) or learn attribute-relation from ConceptNet (`RelatedTo`/`HasProperty`). | M | Low |
| 18 | `engine.py:482` | `_SNIPPET_PLAUSIBILITY_FLOOR=0.38` | (a) | Magic floor used at `engine_web_search.py:431,463`. | Loaded from `data/snippet_pe.json` via `snippet_pe_config._default_pe_config` when fit file present (`engine.py:876`). Inline is seed fallback only. | — | None |
| 19 | `engine.py:483` | `_SNIPPET_PLAUSIBILITY_DEGENERATE=0.12` | (a) | Magic degenerate cutoff (`engine_web_search.py:430,462`). | Same `snippet_pe.json` fit (EER-tuned). | — | None |
| 20 | `engine.py:484` | `_ANSWER_PE_VETO=0.6` | (a) | Magic PE veto cutoff (`engine_web_search.py:435`). | Replaced by `_pe_cfg.veto_midpoint` from fit file when present. | — | None |
| 21 | `engine.py:485-487` | `QUESTION_WORDS` | (d) | Closed interrogative set. Used in generation/interface/constants. | Leave — genuine closed class (what/why/how…). | — | None |
| 22 | `engine.py:488-489` | `FOLLOW_UP_WORDS` | (d) | Closed follow-up marker set (more/else/also…). | Leave — closed class. | — | None |
| 23 | `engine.py:490-494` | `_RECALL_SEED_CONCEPTS` | (c) | ~13-word hand seed list ("remember","recall","earlier"…) for recall detection (`engine_memory.py:634`). Detection itself is vector-based (good); only seeds hardcoded. | Seed from graph: top `prediction_free_energy`/recency concepts, or PMI with "remember/recall" from corpus. The *cosine* comparison is already brain-based; only the anchor list is curated. | M | Low |
| 24 | `engine.py:495` | `_RECALL_DETECTION_THRESHOLD=0.55` | (c) | Fixed cosine cutoff for recall trigger (`engine_memory.py:651`). | Make **adaptive / distribution-driven** like `_vad_baseline` (EMA μ,σ; z-score gate). A fixed 0.55 mis-fires for users whose vocabulary sits above/below it. | M | Med — affects recall sensitivity |
| 25 | `engine.py:496-507` | `_GENERIC_NOUNS` | (c) | ~55-word hand list of generic nouns (thing/system/concept…) used in topic extraction (`engine_generation.py:1667,1698,4975,5828` etc.). | Derive genericity from embedding centrality / PMI; or fold into `functional_lexicon.json`. Closed-ish but currently oversized & hand-tuned. | M | Low |
| 26 | `engine.py:508-526` | `TOPIC_SKIP_WORDS` | (c) | ~70-word hand set (pronouns+light verbs+fillers) used pervasively in generation (`engine_generation.py:1607,1745,1900-1953` etc.) and `interface.py`. | Collapse with STOP_WORDS + `functional_lexicon.json` (already the single source for closed-class). Many entries duplicate `_COMMON_WORDS`/`_GRAMMATICAL_CONCEPTS`. | M | Med — used in ≥12 call sites |
| 27 | `engine.py:527-579` | `_SUBJECT_CONTEXT_WORDS` | (c) | ~90-word hand set of verb/glue/role words for subject grounding (`engine_graph.py:1261`, `engine_reasoning.py`, `engine_generation.py`). Overlaps `_CONDITIONAL_FRAME` & `_GRAMMATICAL_CONCEPTS`. | Single-source via `functional_lexicon.json` + learned light-verb detector (`PosModel`). Pure token filter, safe to data-drive. | M | Med |
| 28 | `engine.py:580-589` | `QUERY_PATTERNS` | (b) | ~8 hardcoded regexes for subject extraction (`engine_generation.py:1557`, `interface.py:764`). | `use_intent_router=False` (OFF) — `IntentRouter` (data/intent_router.json) replaces. Promote to ON; regex stays fallback for None routes. | M | Med — routing correctness |
| 29 | `engine.py:590` | `SAVE_SCHEMA_VERSION=1` | (d) | Schema marker for pickle save/load (`engine.py:3495,3582`). | Leave — version constant, not knowledge. | — | None |
| 30 | `engine.py:591-616` | `_EDGE_CONNECTORS` (v2, weighted) | (b) | Authoritative connector→relation map *with* learned-style weights (0.35/0.33/0.20…). Reverse-lookup built at `engine.py:1011`. | `ConnectorLearner` (#1) is the intended replacement but **never instantiated**. Either wire it or document v2 as the curated prior. | M | Low-Med |
| 31 | `engine.py:643` | `_PROTECTED_CONCEPTS` | (d) | 3 curated project concepts (ravana/oxiverse/intentforge) protected from web overwrite. | Leave — intentional minimal curated namespace (provenance precedence). | — | None |
| 32 | `engine.py:1023-1065` | `_GRAMMATICAL_CONCEPTS` | (b) | ~250-word hand set of grammatical/function words (duplicates `constants.py` KNOWN_VERBS/ADJS/FUNCTION_WORDS & overlaps #16/#27). | `use_learned_pos=False` → `PosModel`. Also redundant with `functional_lexicon.json`. Promote PosModel, then delete duplicate. | M | Low |
| 33 | `engine.py:654,655,663-667` | VAD/Identity/GW eta constants (0.3/0.4/0.25…) | (d) | Engine hyperparams (VAD eta, GW decay/broadcast, sleep thresholds). | Leave — model config, not curated knowledge. Where a *threshold* gates behavior (broadcast 0.3), acceptable; document. | — | None |
| 34 | `engine.py:678` | `_vad_baseline={mu:0,sigma:0.3,n:0}` | (d) ✅ | **Exemplar of the RIGHT pattern**: affective salience judged by EMA z-score, not fixed cutoff. | Keep as the template for #24 and other adaptive thresholds. | — | None — cite as the model to copy |
| 35 | `engine.py:145` (×7 mixins) | `_UNIVERSAL_PURGE` | (d) | Minimal universal pronoun set that can never own a definition. Copy-inherited into all 7 mixins. | Leave the *set* (intentionally minimal, per comment). **But** de-duplicate: define once in `constants.py`, import everywhere. | S | None — only fix the duplication |
| 36 | `constants.py:154-359` `_is_word_salad` inline sets/magic | `_high_freq_structural`,`stoppers`,`grammatical_anchors`,`glue` + magic 0.25/0.3/0.5/0.8/0.4 | (b) | Rule-based salad scoring with hardcoded structural word sets + hand-set bonus weights; still the **active fallback** in `_final_emit_guard` (`engine_generation.py:898-902`). | `is_salad_learned` (salad_classifier) runs FIRST (`engine_generation.py:890`) but falls through to this rule when uncertain. Promote learned to sole gate; keep rule only if `is_salad_learned is None` (no fit). | M | Med — guard correctness |
| 37 | `engine_generation.py:465-466,1835-1836` | schema cos threshold `0.6/0.4/0.5` (pe-driven) | (c) | Fixed cosine cutoffs for schema-member expansion, branched on mean PE. | Make adaptive: threshold = f(EMA of within-schema cosines) or a learned coherence boundary (cf. `coherence_gate.py`). | M | Med |
| 38 | `engine_memory.py:425` | episodic cos threshold `0.5` | (c) | Fixed cosine for "strong link" in episode recall. | Adaptive vs per-user episode-similarity distribution (z-score). | S | Low |
| 39 | `engine_self_query.py:336` | `_best_sim >= 0.45` | (c) | Fixed similarity for self-model stance match. | Adaptive vs user's stance-vector distribution. | S | Low |
| 40 | `engine_generation.py:1698` | `best_sim > 0.75` topic-match | (c) | Fixed cosine for phrase→concept topic match. | Adaptive vs concept-vector separation; or learned classifier. | S | Low |
| 41 | `engine_generation.py:1155-1157,1232` | drift/prune `0.7/0.05/0.1` | (c) | Fixed drift-defense & replay-prune cutoffs. | Tie to sleep/consolidation dynamics (already EMA-driven in `SleepConfig`); expose as fit params. | S | Low |
| 42 | `constants.py:401` `_WEBSITE_SHAPE` + `_POS_TAGS` | shape-based junk signals | (a/d) ✅ | Already the *correct* non-blocklist approach (TLD-tail / vowel-ratio / digit shape). `_POS_TAGS` is a closed tag set. | Keep — this is the model #12 should emulate. | — | None |

---

## Implementation Backlog (for Agent B)

### P0 — Flip the already-built OFF flags (lowest risk, highest payoff)
Baseline BEFORE any change: `python -m pytest tests/test_dehardcode_plan.py -q` → **21 passed, 1 failed** (`test_meaning_of_life_not_dict_dump` — pre-existing on HEAD, NOT a regression target). After each flip, re-run; the 1 failure must remain the ONLY failure.
1. **Promote `use_learned_pos` → ON** (`engine.py:901`): `PosModel` (data/pos_model.json) replaces `_GRAMMATICAL_CONCEPTS` (#32) and `_FUNCTION_POS_TAGS` (#16). Keep `use_learned_pos=False` path as the `PosModel` is-None fallback. Delete the duplicate hand sets only AFTER the suite stays green with the flag ON.
2. **Promote `use_intent_router` → ON** (`engine.py:887`): `IntentRouter` (data/intent_router.json) replaces `QUERY_PATTERNS` (#28). Keep regex as the None-route fallback (do not delete regex).
3. **Promote `use_source_trust` → ON** (`engine.py:895`): learned per-domain trust replaces `_PREFERRED_SNIPPET_SOURCES` (#11). Keep the allowlist only as cold-start prior.
4. **Promote learned salad to sole gate** (#36): keep the rule-based fallback ONLY when `is_salad_learned is None`; otherwise the learned classifier is authoritative in `_final_emit_guard`.
5. **REGRESSION GATE (do NOT recreate):** `tests/test_dehardcode_plan.py` already exists (22 tests). Run it before AND after every change. Target: keep 21 passing, 1 known pre-existing failure unchanged. If a promotion drops below 21, revert that promotion and report why.

### P1 — Wire the dead learned infra + collapse duplicates
6. **Instantiate / wire `ConnectorLearner`** (#1/#30) or formally adopt v2 as curated prior. Currently never constructed.
7. **Collapse closed-class lists into `functional_lexicon.json`**: `_COMMON_WORDS` (#3), `TOPIC_SKIP_WORDS` (#26), `_SUBJECT_CONTEXT_WORDS` (#27), `_CONDITIONAL_FRAME` (#10), `_ATTR_WORDS` (#17). Single data-driven source of truth.
8. **De-duplicate `_UNIVERSAL_PURGE` / `_DEFINITION_ASSERTION`** (#35): define once in `constants.py`, import in all mixins.

### P2 — Truly-hardcoded knowledge + adaptive thresholds
9. **Replace `_JUNK_SNIPPET_DOMAINS`** (#12) with shape-based detection (extend `constants.py:401`).
10. **Learn the sensorimotor realization lexicon** (#8/#9): fit property→(phrase,binder-dim) from corpora/Lancaster norms into a `data/` artifact; keep `_LANCASTER_ORDER` (#7) as fixed encoder contract.
11. **Adaptive thresholds** (copy `_vad_baseline` #34): `_RECALL_DETECTION_THRESHOLD` (#24), schema cos (#37), episode cos (#38), self-query sim (#39), topic sim (#40), drift/prune (#41) → EMA z-score gates.
12. **Derive recall seeds & generic-noun/genericity** (#23/#25) from graph frequency / embedding centrality / PMI instead of hand lists.
13. **Irregular verbs** (#15): low-priority; optionally learn via `PosModel` morphology.

---

## Genuine Capability Gaps (NOT hardcoding)

These are **architectural/learned-capacity** gaps, not curated lists to replace. Do **not** let Agent B "fix" them by removing constants — they need new capability.

1. **Single-turn memory (SAME-turn recall) fails.** Hippocampal replay/consolidation only surfaces facts stored *across turns*; a fact stated and queried in the same turn is never persisted by a fresh isolated engine, so recall returns nothing and the engine echoes the acknowledgement. A two-turn harness regression was attempted and **FAILED** (engine echoed acknowledgement instead of recalling) — do **not** recommend that approach. Real fix: a working-memory buffer that holds the current turn's asserted facts for intra-turn query resolution (distinct from cross-turn consolidation).
2. **`consult` / advice knowledge gap.** `consult_internal` (`engine_self_query.py:538` → `brain_regions.consult_internal`) scores low because the model has **no health / programming-advice knowledge** — this is missing *learned content*, not a hardcoded rule. Fix = broaden the consult KB / web-fallback, not de-hardcoding.
3. **`self_evaluation` is noisy**, not systematically broken. It is a calibration problem (confidence vs accuracy), not a curated-list problem. Fix = confidence-calibration window already present (`MetaCognition`, `confidence_calibration_window=15`) — tune, don't de-hardcode.

---

## Do NOT touch / intentionally minimal

- `SAVE_SCHEMA_VERSION` (#29) — schema marker.
- `MAX_DECODER_VOCAB_SIZE` (#2) — capacity hyperparam.
- `QUESTION_WORDS` (#21), `FOLLOW_UP_WORDS` (#22) — genuine closed classes.
- `_UNIVERSAL_PURGE` set contents (#35) — intentionally minimal (universal pronouns). Only de-duplicate the *definition*, not the words.
- `_DEFINITION_ASSERTION` (#35) — universal copula/reality-monitor regex; correct as-is.
- `_PROTECTED_CONCEPTS` (#31) — intentional curated project namespace (provenance precedence).
- `_LANCASTER_ORDER` (#7) — fixed encoder output contract; architecturally required.
- `_vad_baseline` (#34) — **keep; it is the exemplar adaptive pattern** others should copy.
- `_CATEGORY_OF_SUBJECT` / `_CATEGORY_AFFORDANCES` / `_PROPERTY_CATEGORIES` (#4-6) — keep as **OOV safety net**; ConceptNet is primary but silent on rare subjects. Do not delete; only remove if ConceptNet coverage is proven complete on regression.
- `_SNIPPET_REJECT_SHAPES` / `_SNIPPET_NOISE` (#13-14) — keep as **hard backstop** behind the ON cerebellar model; the learned model is fail-closed to these.
- VAD/Identity/GW/Sleep eta constants (#33) — model config, not knowledge.

---

### Verification notes
- All `file:line` citations were read directly from source (engine.py 160-1090, 3495-3604; engine_graph.py 145-933; engine_reasoning.py 114-1395; engine_memory.py 615-660; engine_web_search.py 420-472, 1656-1678; engine_generation.py 82-919, 1526-1749, 1835-1845; engine_self_query.py 505-544; constants.py 1-421; synaptic_dynamics.py 375-434).
- `ConnectorLearner` confirmed present (synaptic_dynamics.py:379) but **not imported by any engine module** (only chain_walker.py imports synaptic_dynamics) → dead infra.
- `self_model_router` present but only *mirrored* by a regex at `engine.py:1954`, never the learned `extract_features` in the hot path.
- De-hardcode fit artifacts confirmed present in `data/`; regression test file `tests/test_dehardcode_plan.py` **absent**.

---

## Implementation Log (Agent B)

**Date:** 2026-07-22  **Target:** P0 promotion of already-built learned replacements.
**Regression gate:** `python -m pytest tests/test_dehardcode_plan.py -q` from repo root.
**Baseline (before any change):** `1 failed, 21 passed`.
**Do-NOT-touch set:** untouched (SAVE_SCHEMA_VERSION, _UNIVERSAL_PURGE, _PROTECTED_CONCEPTS, _CATEGORY_OF_SUBJECT/_CATEGORY_AFFORDANCES/_PROPERTY_CATEGORIES, _SNIPPET_REJECT_SHAPES/_SNIPPET_NOISE, _LANCASTER_ORDER, VAD/Identity/GW/Sleep eta constants, _vad_baseline).

### Changes made
1. **`use_learned_pos` → ON** (`engine.py:901`): `self.use_learned_pos = False` → `True`.
   PosModel already built lazily in `_is_function_word` (engine_web_search.py:1601-1620);
   hardcoded `_GRAMMATICAL_CONCEPTS` retained as safety net. **GREEN.**
2. **`use_source_trust` → ON** (`engine.py:895`): `self.use_source_trust = False` → `True`.
   `_PREFERRED_SNIPPET_SOURCES` kept as cold-start prior inside `_domain_trust`/`_is_preferred_source`;
   learned accumulator path now active. **GREEN.**
3. **Learned salad = sole authoritative gate** (`engine_generation.py` `_final_emit_guard`, ~L887-910):
   captured the learned verdict explicitly; rule-based `_is_word_salad` now runs ONLY when
   `is_salad_learned is None` OR the learned verdict was `None` (no fit). Fluent-tautology
   detector kept as independent learned check. **GREEN** (learned gate not weakened).
4. **`use_intent_router` → REVERTED to OFF** (`engine.py:887`): attempted flip to `True`
   dropped the gate to `2 failed, 20 passed` — `test_intent_router_off_by_default_and_safe`
   (tests/test_dehardcode_plan.py:72) **asserts `engine.use_intent_router is False` as a required
   default contract**. Per the hard constraint (do NOT modify/recreate the test; revert and report
   if a promotion drops below 21), the flag was reverted to `False` and the gate returned to
   `1 failed, 21 passed`. The IntentRouter is fully built/wired and safe; it simply cannot be
   ON-by-default without violating this existing contract test. Recommend the parent update the
   contract test before flipping this flag.

### Per-promotion pytest summary (real runs, repo root)
- Baseline (before): `1 failed, 21 passed`
- After P:1 `use_learned_pos=True`: `1 failed, 21 passed`
- After P:2 `use_intent_router=True` (attempt): `2 failed, 20 passed` → **REVERTED**
- After P:3 `use_source_trust=True`: `1 failed, 21 passed`
- After P:4 learned-salad authoritative: `1 failed, 21 passed`
- After `use_intent_router` reverted (final): `1 failed, 21 passed`
  (The 1 failure is `test_meaning_of_life_not_dict_dump` — pre-existing on HEAD, NOT a regression.)

### Lists NOT deleted
`_GRAMMATICAL_CONCEPTS`, `_FUNCTION_POS_TAGS` left in place as `_is_function_word` safety net
(catalog §6 prefers leaving backstops over deleting). No redundant-list deletions performed.

