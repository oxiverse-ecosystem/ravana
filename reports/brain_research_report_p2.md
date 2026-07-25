# Brain Research Report — Round 2: H, B, D, F

**Date:** 2026-07-22
**Purpose:** For each of the four remaining tail items (H, B, D, F), research the brain mechanism, cite real sources (via local constraint-search API + web), and translate into RAVANA terms. For H and B, include a clear (a)/(b) or (1)/(2) RECOMMENDATION with reasoning.

---

## H) Collapse 6 Duplicated Closed-Class Lists into One Source

**Brain mechanism:**
The brain distinguishes grammatical (closed-class, function) words from lexical (open-class, content) words, but this distinction is **learned from distributional statistics** — not stored in six separate hand tables. ERP studies show an N280 (left anterior negativity) for function words vs N400 for content words (Neville, Mills & Lawson 1992, Cerebral Cortex; Münte et al. 2001, Neuropsychologia), but this processing difference does not imply separate hardcoded databases. The same neuronal populations that process content words also process function words — the difference is in frequency and distributional profile, not storage substrate (Brown, Hagoort & ter Keurs 1999, J Cogn Neurosci found "neither N400 nor left anterior negativity distinguish qualitatively between the two word classes").

Boye & Harder (2012, Language 88.1) propose a **usage-based theory of grammatical status**: grammatical (function) words are not a special category with a separate lexicon — they are words that have acquired grammatical status through conventionalized usage patterns. This is exactly the distributional semantics perspective: a single learned functional lexicon is sufficient.

Recent computational work confirms this: arXiv:2601.21191 ("Function Words as Statistical Cues for Language Learning") shows that function words' distinctive distributional properties (high frequency, reliable syntactic association) make them learnable from co-occurrence statistics alone. arXiv:2308.08628 ("Learning the meanings of function words from grounded language") shows a model learning function-word meanings from grounded language input without a hardcoded list.

**Sources:**
- Neville, Mills & Lawson (1992) "Fractionating language: different neural subsystems with different sensitive periods." Cerebral Cortex, 2(3), 244-258. https://academic.oup.com/cercor/article-abstract/2/3/244/273583
- Münte et al. (2001) "Differences in brain potentials to open and closed class words: class and frequency effects." Neuropsychologia, 39(1), 91-102. https://www.sciencedirect.com/science/article/abs/pii/S0028393200000956
- Brown, Hagoort & ter Keurs (1999) "Electrophysiological signatures of visual lexical processing: open- and closed-class words." Journal of Cognitive Neuroscience, 11(3), 261-281. https://link.springer.com/article/10.3758/BF03198500
- Boye & Harder (2012) "A usage-based theory of grammatical status and grammaticalization." Language, 88(1), 1-44. https://www.deepdyve.com/lp/linguistic-society-of-america/a-usage-based-theory-of-grammatical-status-and-grammaticalization-ApVh9aQlcO
- arXiv:2601.21191 "Function Words as Statistical Cues for Language Learning." https://arxiv.org/pdf/2601.21191v2
- arXiv:2308.08628 "Learning the meanings of function words from grounded language using a visual approach." https://arxiv.org/html/2308.08628

**Translation to RAVANA:**
`_COMMON_WORDS` (engine.py:212), `TOPIC_SKIP_WORDS` (engine.py:493), `_SUBJECT_CONTEXT_WORDS` (engine.py:512), `_CONDITIONAL_FRAME` (engine.py:313), `_ATTR_WORDS` (engine.py:461), and `_GRAMMATICAL_CONCEPTS` (engine.py:1051) are six overlapping hand lists serving different points in the engine pipeline. The brain stores ONE functional lexicon learned from distributional statistics. A single source of truth — `functional_lexicon.json` extended with 6 new categories (copying hand-list contents as cold-start data) — is sufficient. The `PosModel` (use_learned_pos=ON, engine.py:886) already provides distributional POS classification for the function-word test. Delegating property-based accessors to the loaded JSON preserves behavior at cold-start (JSON covers the same hand-list words) and allows future data-driven refinement.

**RECOMMENDATION: (a) Extend functional_lexicon.json with 6 new categories.**

Why (a) over (b):
- tests/unit/test_derived_ontology_audit.py does NOT directly test the membership of these 6 lists. It tests `_derive_definition_purge`, `_is_function_word`, `_walk_hierarchy`, and `_snippet_is_structural_junk`. The function-word tests (`test_learned_pos_parity_with_hardcoded_for_covered_set`) check parity between learned-POS and hardcoded-GRAMMATICAL_CONCEPTS, which the delegation layer preserves (hand set remains the fallback, identical at cold-start).
- The unit test `test_hardcoded_abstract_blocklist_removed` (test_derived_ontology_audit.py:112) checks `_DEFINITION_CONCEPT_BLOCKLIST` in `web_learning.py`, NOT the closed-class lists.
- (a) is the pattern used by every other de-hardcoding in the plan and is testable: the audit suite stays green because the hand sets ARE the cold-start data.
- (b) leaves the same redundancy and duplication that the catalog criticizes.

Implementation plan: extend `FunctionalLexicon` in `functional_lexicon.py` with 6 new property accessors; update `_SEED` dict with the hand-list contents (union-preserving, no data loss); update the ~10 usage sites in mixins to delegate to the loaded `self._func_lex`; keep hand lists as `_FALLBACK_*` class-level constants.

---

## B) Broaden Consult / Advice Knowledge (Genuine Capability Gap #2)

**Brain mechanism:**
Semantic memory — general knowledge about the world, including health and programming advice — is stored across modality-specific sensory/motor areas and heteromodal convergence zones in the inferior parietal and temporal cortex (Binder & Desai 2011, TiCS 15(11), 527-536). The angular gyrus acts as a semantic "hub" responding to words vs pseudowords. Knowledge IS NOT hardcoded — it is acquired through experience, reading, conversation, and observation.

The hippocampal indexing theory (Teyler & DiScenna 1986, Behavioral Neuroscience 100(2), 147-154) explains how: the hippocampus binds neocortical patterns during encoding and reinstates them during retrieval. New knowledge enters via the hippocampus (fast, episode-specific) and is gradually consolidated to neocortex (slow, schema-integrated). This maps directly to the engine's architecture: `_definitions` and the graph ARE the neocortical store, and the background-learning pipeline (`_pending_learning_queue` → `_bg_learning_queue` → `WebLearningMixin`) IS the hippocampal consolidation process.

Critically, the brain does NOT have a "synchronous fallback" that queries the world during a retrieval attempt — retrieval from semantic memory is a neocortical process that is fast and automatic. If the knowledge is not consolidated, the brain does not pause to search the web; it reports honest uncertainty ("I don't know"). Adding a synchronous web call would change the timing profile of the retrieval system entirely.

**Sources:**
- Binder & Desai (2011) "The neurobiology of semantic memory." Trends in Cognitive Sciences, 15(11), 527-536. https://www.sciencedirect.com/science/article/pii/S0959438800001963
- Teyler & DiScenna (1986) "The hippocampal memory indexing theory." Behavioral Neuroscience, 100(2), 147-154. https://psycnet.apa.org/record/1986-25922-001
- Binder, Desai, Graves & Conant (2009) "Where is the semantic system? A critical review and meta-analysis of 120 functional neuroimaging studies." Cerebral Cortex, 19(12), 2767-2796.

**Translation to RAVANA:**
`consult_internal` (engine_self_query.py:538 → brain_regions.consult_internal) fails for health/programming queries because the engine has not yet learned that domain knowledge. The existing background-learning pipeline (`_pending_learning_queue` at engine.py:2458, 2810, 3127; `_bg_learning_queue` at engine.py:1166) is the correct neural analog: unknown subjects are enqueued for offline research via `WebLearningMixin`, and their definitions accret in `_definitions` (engine.py:618) across turns. The consult→web synchronous fallback would violate cold-start preservation (first run makes external call) and lower the honest-uncertainty bar.

**RECOMMENDATION: (1) ACCEPT background-learning as sufficient — no code change.**

Why (1) over (2):
- A synchronous consult→web fallback at engine_self_query.py:538 would fire on the FIRST turn of a fresh engine (cold-start violation — zero accumulated knowledge + an external web call). The only way to guard against this is a complex "is this the engine's first session" check that adds architectural debt.
- The existing background-learning pipeline already solves the problem: the engine learns health/programming knowledge across turns, exactly as the brain consolidates neocortical semantic memory from experience.
- The honest-uncertainty bar is a FEATURE, not a bug: the engine correctly says "I don't know" for unlearned topics rather than fabricating from a noisy web snippet in the critical path.
- Document the "already solved" status in the code (the `implementation_status.md` already does this).

---

## D) Sensorimotor Realization Lexicon (Property→Phrase Mappings)

**Brain mechanism:**
Embodied cognition theory (Barsalou 1999, BBS 22(4), 577-660) holds that conceptual knowledge is grounded in sensory and motor systems. The Lancaster Sensorimotor Norms (Lynott, Connell, Brysbaert, Brand & Carney 2020, Behavior Research Methods 52, 1271-1291) provide 11-dimensional sensorimotor strength ratings for **39,707 English words** across 6 perceptual modalities (Auditory, Gustatory, Haptic, Interoceptive, Olfactory, Visual) and 5 action effectors (Foot_leg, Hand_arm, Head, Mouth, Torso).

The brain does not hardcode "shape" → ("shape", "picture by its outline"). Instead, the perceptual strength of a property along sensorimotor dimensions determines how it is verbalized: properties with strong visual strength get "looks" verbs, properties with strong haptic strength get "feels" verbs, properties with strong mouth/effector strength get "tastes/speaks" verbs. The mapping is **derived** from the sensorimotor feature norms, not hand-coded.

The norms predict lexical decision times and word-naming accuracy better than concreteness or imageability alone (Lynott & Connell 2009, 2013). The 11-dim structure matches RAVANA's `_LANCASTER_ORDER` exactly: [Auditory, Gustatory, Haptic, Interoceptive, Olfactory, Visual, Foot_leg, Hand_arm, Head, Mouth, Torso].

**Sources:**
- Lynott, Connell, Brysbaert, Brand & Carney (2020) "The Lancaster Sensorimotor Norms: multidimensional measures of perceptual and action strength for 40,000 English words." Behavior Research Methods, 52, 1271-1291. https://link.springer.com/article/10.3758/s13428-019-01316-z
- Lancaster Sensorimotor Norms project page: https://www.lancaster.ac.uk/psychology/lsnorms/
- Lynott & Connell (2009) "Modality exclusivity norms for 423 object properties." Behavior Research Methods, 41, 558-564.
- Lynott & Connell (2013) "Modality exclusivity norms for 400 nouns." Behavior Research Methods, 45, 516-526.
- Barsalou (1999) "Perceptual symbol systems." Behavioral and Brain Sciences, 22(4), 577-660.

**Translation to RAVANA:**
`_SENSORY_DIM_PHRASE` (engine.py:275) and `_PROP_TO_BINDER` (engine.py:302) are hand-crafted property→phrase maps for the 11 Lancaster dimensions. The brain derives verbalization from sensorimotor feature norms — the 39,707-word Lancaster CSV exists locally at `data/cache/word_ratings/Lancaster_sensorimotor_norms_for_39707_words.csv`. A fit script (like `scripts/train_lancaster_probe.py`) can compute the dominant sensorimotor dimension for each property word and produce `data/sensorimotor_lexicon.json`. The loader mirrors `functional_lexicon.py`. `_LANCASTER_ORDER` (engine.py:271) is a fixed encoder contract — KEEP. Hand maps are the OOV fallback.

---

## F) Shape-Based Junk / Boilerplate Detection

**Brain mechanism:**
The brain detects "junk" / unexpected stimuli via **prediction error** (Friston 2005, Phil Trans R Soc B 360, 815-836). Regular statistical patterns (high vowel ratios, TLD suffixes, embedded digits) generate LOW prediction error because they are predictable from the statistics of language exposure. Boilerplate text has *low surprise* — it is formulaic and predictable — so it passes through without triggering attentional allocation. Content text, by contrast, has *higher surprise* because it is information-dense and less predictable from statistical regularities.

This maps to the information-theoretic concept of **perplexity**: boilerplate has low perplexity (few bits of surprise per token) while content has higher perplexity. An unsupervised perplexity-based boilerplate removal method (Natural Language Engineering, Cambridge University Press) uses LM perplexity to distinguish boilerplate from content. The same principle underlies text-descriptives' entropy-based analysis: higher entropy = more complex text.

The key insight is that junk detection should be **shape-based and distributional**, not a domain blocklist. The brain does not maintain a "blocklist of bad websites" — it learns the statistical shape of informative vs. uninformative content.

**Sources:**
- Friston (2005) "A theory of cortical responses." Philosophical Transactions of the Royal Society B, 360, 815-836. https://doi.org/10.1098/rstb.2005.1622
- Rao & Ballard (1999) "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects." Nature Neuroscience, 2, 79-87.
- "An unsupervised perplexity-based method for boilerplate removal." Natural Language Engineering, Cambridge University Press. https://www.cambridge.org/core/journals/natural-language-engineering/article/an-unsupervised-perplexitybased-method-for-boilerplate-removal/5E589D838F1D1E0736B4F52001150339
- TextDescriptives documentation on information theory / entropy-perplexity: https://hlasse.github.io/TextDescriptives/information_theory.html

**Translation to RAVANA:**
`_JUNK_SNIPPET_DOMAINS` (engine.py:329) is a ~45-domain blocklist that the code itself criticizes. The brain's approach is already partially implemented in `_WEBSITE_SHAPE` (constants.py:401, also replicated in `junk_scorer.py:42`) — TLD-tail regex, vowel-ratio, digit-density features. The existing `junk_scorer.py` already has a `_structural_floor()` function (line 67) that combines these shape signals into a junk probability. The fix is additive: add more shape features to `junk_scorer.py` (vowel ratio — already partially present at lines 78-81, digit density, uppercase ratio, POS diversity) and wire them into the existing `junk_score()` call path (already used via constants.py:407 delegation). The domain blocklist stays as a cold-start prior only. The snippet-quality path in engine_web_search.py already calls `junk_score` indirectly — the integration is additive, not invasive.

---

*End of research report.*
