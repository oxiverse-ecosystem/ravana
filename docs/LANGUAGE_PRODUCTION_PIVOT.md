# Language Production Pivot: Brain → RAVANA Mapping

This document maps human language production to the existing RAVANA implementation (read: `ravana/src/ravana/chat/response_gen.py`, `ravana/src/ravana/chat/engine.py`, `ravana/src/ravana/chat/hedges.py`, `ravana/src/ravana/language/surface_realizer.py`). Each mechanism is marked **ADOPT** / **REJECT** / **DEFER** with rationale and concrete files/lines to modify or keep.

**Constraints:** No LLM, no retrain.

---

## Human Language Production (Overview)

Standard psycholinguistic/neurobiological account (Levelt's model + predictive coding + embodiment):

| Stage | Brain Regions | Function |
|---|---|---|
| Conceptualization (Message) | PFC, anterior temporal lobe, default mode / semantic system | Decide *what* to say (intent, discourse goal). Driven by goals/context/knowledge. |
| Formulation (Planning) | Broca's area (BA44/45), left inferior frontal gyrus, basal ganglia (BG), cerebellum | Lexical selection (lemmas), grammatical encoding, syntactic structure building, prosodic planning. Competitive selection under constraints. |
| Articulation (Phonetic/Execution) | Motor cortex, SMA, cerebellum, basal ganglia | Convert phonological plan to motor commands; timing/coordination. (Out of scope for text output.) |
| Monitoring (Self-monitoring) | Anterior cingulate cortex (ACC), PFC | Detect errors, conflict, adjust on-the-fly (internal monitoring). |

Key computational principles relevant here:
- **Predictive coding / free energy** (Friston/Clark): precision/confidence modulates how much top-down prediction vs bottom-up drives processing. Uncertainty → hedging, exploration, questioning.
- **Competitive lemma selection** (Levelt): multiple candidates compete; context/activation determine selection.
- **Embodied cognition / dual coding** (Binder et al.): sensorimotor grounding affects conceptual access and fluency.
- **Compositionality / merge** (minimalist syntax): hierarchical structure built by merge operations; variation comes from different merge orders/fronting (left-dislocation/clefts).
- **Discourse pragmatics**: anaphora/resolution, given/new, topic management, epistemic stance (modal/hedges).
- **Attractor dynamics / pattern completion** (Hopfield-like): coherent trajectories stabilize meaning.

---

## Mapping to RAVANA

### 1) Conceptualization (Message Preparation)
**Brain:** Intent/discourse goal → conceptual message.

| Mechanism | RAVANA Location | Assessment | Notes |
|---|---|---|---|
| Discourse intent/planning (goal-driven) | `chat/engine.py` (cognitive stack: GlobalWorkspace, DualProcess, PrefrontalWorkspace imported; `language/prefrontal_workspace.py` exists). Also `chat/engine_reasoning.py`/generation flow via `CognitiveResponseContext`. | **ADOPT** (already present) | Keep using discourse state/intent. Ensure it drives *what* to retrieve (not just *how* to say). Current architecture already has GW/DualProcess/PFC — leverage rather than add templates. |
| Attractor/pattern completion for coherence | `chat/response_gen.py` (`_build_attractor_memory`, `_grounded_anchor_trajectory`, `AttractorMemory`). `ravana/core/attractor_memory.py` implied. | **ADOPT** | Strong alignment (Hopfield-style). Keep as coherence floor; retrieval-augmented scaffolding is brain-consistent. |
| Free energy / uncertainty as driver | `language/surface_realizer.py` (DiscourseState.free_energy, `_get_confidence_level`, `_sensorimotor_adjusted_fe`). `chat/hedges.py` (modality_from_support). | **ADOPT** | Core pivot principle already implemented. This is the right lever for hedging/questions/stance. |
| Curiosity-driven questioning | `surface_realizer.py` (`_generate_question` uses concept_free_energy). | **ADOPT** (with guardrails) | Questions should be genuine information-seeking driven by FE/ignorance, not canned lists. Current implementation moves in this direction (replaced hardcoded QUESTION_CLAUSES with FE-driven logic). Keep RNG-free/deterministic where possible or gate strictly. |

### 2) Formulation (Lexical + Syntactic Planning)
**Brain:** Lemma selection, syntactic frames, competitive selection.

| Mechanism | RAVANA Location | Assessment | Notes |
|---|---|---|---|
| Lexical competition / Hebbian verb selection | `language/verb_lexicon.py` (imported in `surface_realizer.py`; `VerbLexicon.fix_morphology`, vector-based selection). `chat/response_gen.py` uses NeuralDecoder with vocab from graph/GloVe. | **ADOPT** | Hebbian/competitive selection is appropriate. Keep learned lexicon; avoid hardcoded verb lists. |
| Syntactic merge variants (structural diversity) | `surface_realizer.py` (MERGE_VARIANTS, `_select_variant`, `_apply_variant`) — left-dislocation, clefts, pseudo-clefts, adverb-fronting etc. | **ADOPT** | Matches minimalist syntax/merge diversity. Good replacement for template variety. Keep selection driven by discourse/dopamine/FE, not random template picking. |
| Agreement/tense/morphology | `surface_realizer.py` (`_apply_agreement`, `_apply_tense`, `VerbLexicon.fix_morphology`). | **ADOPT** | Necessary for grammaticality. |
| Function words vs content words separation | `chat/response_gen.py` (`_FUNCTION_WORDS`, function word sets in generation; `hedges.py` treats function words as instructions not retrieval cues). `language/surface_realizer.py` uses function classes. | **ADOPT** | Correct (encoding specificity). Keep strict separation. |
| Competitive lemma with uncertainty modulation | `surface_realizer.py` + `chat/response_gen.py` (decoder uses conditioning, neuromodulation). | **ADOPT** | Uncertainty (FE/sensorimotor confidence) should modulate competition (more exploration when uncertain). Current pieces exist; ensure they interact. |
| N-gram/cerebellar bias | `chat/response_gen.py` passes `cerebellar_ngram` to realizer/decoder paths. `language/cerebellar_ngram.py` exists. | **ADOPT** | Motor-sequence timing analog is reasonable for fluency. Keep as bias, not templates. |

### 3) Epistemic Stance, Hedging, Pragmatics
**Brain:** Modality, ownership, uncertainty management.

| Mechanism | RAVANA Location | Assessment | Notes |
|---|---|---|---|
| Composed hedges from measured support (no template lookup) | `chat/hedges.py` (hedge_frame composes from markers + real state; `_MARKERS` seed lexicon, `register_marker` for growth). `chat/response_gen.py` references `hedges.py`. | **ADOPT** | Excellent pivot: moved from fixed frames to composition from measured modality/support. Matches "hedges are uncertainty management". Keep this approach. |
| Free-energy-driven hedges in realizer | `surface_realizer.py` (`_compose_hedge`, HEDGE_MORPHEMES as morphemes not sentences). | **ADOPT** | Morpheme-level composition is better than sentence templates. Deterministic gating by FE is preferable to RNG where possible. |
| Modality mapping from support/coherence | `hedges.py` (`modality_from_support`). | **ADOPT** | Aligns with graded confidence (possibility/likelihood/certainty). |
| Sensorimotor grounding affecting confidence | `surface_realizer.py` (`_sensorimotor_adjusted_fe` raises FE for weak grounding). `chat/response_gen.py` has Lancaster/sensorimotor embeddings. | **ADOPT** | Embodiment affects confidence/hedging — brain-consistent. |
| Anaphora/pronoun resolution | `surface_realizer.py` (`_resolve_pronoun`, `_get_pronouns_for_concept` using GloVe/ATL categories, PRONOUN_VARIANTS). | **ADOPT** | Moved from hardcoded sets to category-based resolution; good. Keep abstractness check explicit. |
| Discourse markers/connectors | `surface_realizer.py` (`_select_discourse_marker` referenced; discourse markers applied). | **ADOPT** (cautiously) | Connectors aid coherence; ensure they come from discourse state/relations, not canned lists. |
| Epistemic frames (I think/maybe) | `surface_realizer.py` (`_generate_epistemic_frame` deterministic by hash of subject/content). | **ADOPT** | Deterministic is better than RNG; matches stance without template soup. |

### 4) Grounding, Retrieval, Coherence Guards
**Brain:** Semantic memory, pattern completion, monitoring.

| Mechanism | RAVANA Location | Assessment | Notes |
|---|---|---|---|
| Grounded anchor / retrieval-augmented scaffolding | `chat/response_gen.py` (`_grounded_anchor`, `_grounded_anchor_trajectory`, Option C). | **ADOPT** | Retrieval provides coherence floor; generation enriches. Brain-consistent. |
| Definition knowledge separate from associations | `chat/engine.py` has `_definitions`, `_curated_definitions`, `_PROTECTED_CONCEPTS`. | **ADOPT** | Mirrors ATL/category knowledge vs associative. Important for correct behavior. |
| Utterance completeness monitoring (forward model) | `surface_realizer.py` (`_utterance_complete`, returns "" on incomplete). | **ADOPT** | Analog to error monitoring/repair; prevents fragmentary output. Good guardrail. |
| Salad/quality gates | `chat/engine.py` imports salad classifiers/monitors; `chat/response_gen.py` has quality gates. | **ADOPT** (as safety) | Necessary guardrails; keep them data-driven/fail-closed as designed. |
| VSA/event schemas for narratives | `chat/response_gen.py` (`_vsa_event_narrative`). | **ADOPT** | Schema completion is brain-like. |

### 5) What to Change (Concrete Recommendations)

| Target | Action | Rationale |
|---|---|---|
| **Hedges/frames** | **ADOPT** current composed approach. Remove any remaining sentence-template lookups. | `hedges.py` already pivoted correctly. Ensure no other module reintroduces `_HEDGE_FRAMES` style tables. |
| **Uncertainty drive** | **ADOPT** FE as primary driver. Make hedging/questions proportional to measured FE/support, not random. | Already in place; tighten determinism (avoid unnecessary RNG). |
| **Syntactic variety** | **ADOPT** merge variants. Prefer FE/DA/discourse-driven selection over uniform random. | Good; keep fatigue/STN-like penalties. |
| **Embodiment** | **ADOPT** sensorimotor adjustment to FE. Expand coverage where cheap. | Aligns with dual coding; already implemented. |
| **Retrieval floor** | **ADOPT** grounded anchors/attractors. Require retrieval coherence before free-form elaboration when uncertain. | Prevents confabulation; matches pattern completion. |
| **Monitoring** | **ADOPT** completeness checks. Extend to catch other fragment types if found. | Forward-model monitoring is correct. |
| **Lexical competition** | **ADOPT** Hebbian/decoder competition modulated by precision. | Brain-consistent. |
| **RNG in stance/questions** | **DEFER/ADJUST**: replace remaining `random.choice` in stance/question generation with deterministic hashing where user-visible consistency matters. | `surface_realizer.py` still uses `random` in `_generate_reflective_clause`, `_generate_question`, some variant selection. For "no LLM, no retrain" deterministic behavior is preferable and more controllable. |
| **Hardcoded concept lists** | **REJECT** reintroducing them. Continue replacing with GloVe/category classifiers. | Already happening (pronouns/abstract/uncountable). |
| **Sentence templates** | **REJECT** any new template tables. Compose from morphemes/state. | Core constraint aligned with pivot. |

---

## Specific File-Level Notes

### `chat/response_gen.py`
- **ADOPT:** `_PROBE_DIMENSIONS` as gap->slot registry (not Q->A dict) — correct.
- **ADOPT:** decoder vocab built from graph + function words + corpus; dual-code embeddings (GloVe+Lancaster).
- **ADOPT:** attractor memory, grounded anchors, conditioned BOS.
- **ADOPT:** VSA event narratives.
- **NOTE:** Has extensive conditioning/boosting logic; keep retrieval floor before free-form paths.

### `chat/engine.py`
- **ADOPT:** Rich cognitive stack imports (GW, dual-process, meta-cognition, sleep). Orchestrator is appropriate.
- **ADOPT:** Definition stores, protected concepts, provenance.
- **NOTE:** Large file; changes should be targeted.

### `chat/hedges.py`
- **ADOPT:** Compositional hedging from measured state. Seed marker lexicon + `register_marker`. This is the model to follow elsewhere.
- **ADOPT:** `modality_from_support` thresholds.

### `language/surface_realizer.py`
- **ADOPT:** Free-energy driven pipeline, merge variants, sensorimotor adjustment, completeness check.
- **DEFER:** Reduce RNG usage (reflective clauses/questions) for determinism. Hash-based selection like `_generate_epistemic_frame` does.
- **ADOPT:** GloVe-based classifiers for pronouns/abstract/uncountable (replaced hardcoded sets).

---

## Implementation Plan (No LLM, No Retrain)

1. **Audit remaining templates**: grep for sentence-like template strings/lists across chat/language modules. Replace with composition or remove.
2. **Harden determinism**: convert RNG choices in stance/reflection/questions to hash-based selection where appropriate.
3. **Tighten FE coupling**: ensure hedging probability/modality always comes from real support/FE, not defaults.
4. **Preserve guards**: keep completeness monitor and salad gates.
5. **Test**: run existing tests; no new training data.

---

## Conclusion

RAVANA already implements much of the brain's language production architecture: predictive coding/FE, competitive selection, merge-based syntax, composed hedging, retrieval-based coherence, and embodiment. The correct pivot is to **ADOPT** these mechanisms, **REJECT** any reintroduction of sentence templates/hardcoded Q->A, and **DEFER** non-essential randomness for more deterministic, uncertainty-driven production. The codebase is well-aligned with the target model.
