# Brain Research Report — Mechanisms for RAVANA Cognitive Architecture

**Date:** 2026-07-22
**Purpose:** For each of 9 items (A–I), summarize the brain mechanism found via web research, cite real sources, and translate into RAVANA terms.

---

## A) Working Memory / Same-Turn Buffer

**Brain mechanism:**
Baddeley & Hitch (1974) proposed a multicomponent working memory model with a central executive, phonological loop (verbal buffer), and visuospatial sketchpad. The phonological loop stores ~2s of speech-based information via an "inner ear" (phonological store) refreshed by an "inner voice" (articulatory rehearsal) (Baddeley, Thomson & Buchanan 1975). Baddeley (2000) added the *episodic buffer* — a limited-capacity store that binds information from multiple modalities into integrated, multimodal representations accessible to conscious awareness. The buffer bridges rapid perceptual streams with slower long-term representations and is critical for immediately remembered facts during ongoing cognition.

Critically, working memory is *not* the same as long-term consolidation. The hippocampus replays information offline for cross-turn consolidation (Carr, Jadhav & Frank 2011) but the prefrontal / parietal phonological + episodic buffer holds information *online* within a single cognitive episode. Without an online buffer, a fact stated and queried in the same turn is lost — hippocampal consolidation operates across turns, not within them.

**Sources:**
- Baddeley & Hitch (1974) "Working Memory." In Bower (Ed.), The Psychology of Learning and Motivation. DOI: 10.1016/S0079-7421(08)60452-1
- Baddeley (2000) "The episodic buffer: a new component of working memory?" Trends in Cognitive Sciences, 4(11), 417-423. DOI: 10.1016/S1364-6613(00)01538-2
- Baddeley (2007) *Working Memory, Thought, and Action.* Oxford University Press.
- Carr, Jadhav & Frank (2011) "Hippocampal replay in the awake state: a potential substrate for memory consolidation and retrieval." Nature Neuroscience, 14, 147-153. DOI: 10.1038/nn.2732

**Translation to RAVANA:**
The current `_recent_user_turns` ring buffer (engine.py:1462) captures verbatim turns but is not *propositionally indexed* for intra-turn query resolution. The hippocampal buffer (HippocampalBuffer, engine.py:939) and `working_memory` (WorkingMemory, engine.py:960) exist but are not wired to serve intra-turn facts. The brain's solution is a dedicated WM buffer that extracts asserted propositions from the current turn and makes them queryable *before* any cross-turn consolidation path. This is distinct from the existing `_consult_internal_knowledge` (engine.py:1443) which reads cross-turn definitions.

---

## B) Semantic Knowledge / Consult Domain Gap

**Brain mechanism:**
Semantic memory is stored across modality-specific sensory/motor areas and heteromodal convergence zones in the inferior parietal and temporal cortex (Binder & Desai 2011, TiCS). The angular gyrus acts as a semantic "hub" — it responds more to words than pseudowords, more to high-frequency than low-frequency words, and more to concrete than abstract words. Knowledge is acquired through learning *from the world* (reading, conversation, observation), not from hardcoded fact tables. The hippocampal index theory (Teyler & DiScenna 1986) posits that the hippocampus binds neocortical patterns during encoding and later reinstates them during retrieval.

Critically, advice / health / programming knowledge is DOMAIN SEMANTIC knowledge — it is not a separate "advice module" in the brain. It is acquired by the same semantic learning mechanisms: repeated exposure, schema formation, and generalization from examples. The brain does not have hardcoded medical advice lists; it learns from experience.

**Sources:**
- Binder & Desai (2011) "The neurobiology of semantic memory." Trends in Cognitive Sciences, 15(11), 527-536. DOI: 10.1016/j.tics.2011.10.001
- Binder, Desai, Graves & Conant (2009) "Where is the semantic system? A critical review and meta-analysis of 120 functional neuroimaging studies." Cerebral Cortex, 19(12), 2767-2796. DOI: 10.1093/cercor/bhp055
- Teyler & DiScenna (1986) "The hippocampal memory indexing theory." Behavioral Neuroscience, 100(2), 147-154. DOI: 10.1037/0735-7044.100.2.147

**Translation to RAVANA:**
`consult_internal` (engine_self_query.py:538 → brain_regions.consult_internal) fails because the engine has no stored domain knowledge about health/programming/etc. The existing web-learning pipeline (WebLearningMixin, engine_web_search.py) already accumulates knowledge from the web across turns. The fix is to broaden the consult KB / web-fallback so consult retrieves from this accumulated graph + definition store, not a hardcoded advice table. The existing `_definitions` dict + graph edges are the right substrate; they just need to cover more domains (which comes from usage, not code).

---

## C) Self-Evaluation / Confidence Calibration

**Brain mechanism:**
Metacognitive confidence involves the anterior cingulate cortex (ACC) monitoring conflict, the dorsomedial PFC tracking uncertainty, and the striatum encoding positive confidence signals (Fleming et al. 2012, Neuron; Molenberghs et al. 2016, Human Brain Mapping). Koriat's Accessibility Model (1993, Psychological Review) argues that the Feeling of Knowing (FOK) is computed from the accessibility of partial information during retrieval attempts — it is a *distributional* phenomenon, not a fixed threshold. Calibration (the match between confidence and accuracy) is learned through experience: repeated feedback adjusts the mapping from internal cues to confidence reports.

Critically, noisy self-evaluation is a *calibration* problem, not a hardcoded-value problem. The brain does not use a fixed 0.55 threshold for "I know this." It uses precision-weighted prediction errors (Friston 2010, Nature Reviews Neuroscience) and an expected vs. actual error signal (the ERN / Pe complex in ACC) to iteratively calibrate.

**Sources:**
- Koriat (1993) "How do we know that we know? The accessibility model of the feeling of knowing." Psychological Review, 100(4), 609-639. DOI: 10.1037/0033-295X.100.4.609
- Fleming & Dolan (2012) "The neural basis of metacognitive ability." Philosophical Transactions of the Royal Society B, 367, 1338-1349. DOI: 10.1098/rstb.2011.0417
- Molenberghs et al. (2016) "Neural correlates of metacognitive ability and of feeling confident: a large-scale fMRI study." Social Cognitive and Affective Neuroscience, 11(12), 1956-1966. DOI: 10.1093/scan/nsw075
- Friston (2010) "The free-energy principle: a unified brain theory?" Nature Reviews Neuroscience, 11, 127-138. DOI: 10.1038/nrn2787

**Translation to RAVANA:**
The existing `MetaCognition` (metacognition.py) with `confidence_calibration_window=15` (engine.py:1071) is already the right infrastructure. The `_calibration_error` (engine.py:1161) tracks the gap. The fix is not a new module — it is tuning the existing calibration: make the window adaptive (longer when stable, shorter when volatile) and use the accumulated error history to adjust the theta_withhold threshold distributionally rather than keeping it at fixed 0.30.

---

## D) Sensorimotor Realization Lexicon (Property→Phrase Mappings)

**Brain mechanism:**
Embodied cognition theory (Barsalou 1999, BBS; Lynott & Connell 2009, 2013, BRM) holds that conceptual knowledge is grounded in sensory and motor systems. The Lancaster Sensorimotor Norms (Lynott, Connell, Brysbaert, Brand & Carney 2020, Behavior Research Methods) provide 11-dimensional sensorimotor strength ratings for ~40,000 English words across 6 perceptual modalities (Auditory, Gustatory, Haptic, Interoceptive, Olfactory, Visual) and 5 action effectors (Foot_leg, Hand_arm, Head, Mouth, Torso). These norms predict lexical decision times and word-naming accuracy better than concreteness or imageability.

The brain does not hardcode "shape" → ("shape", "picture by its outline"). Instead, the perceptual strength of a property along sensorimotor dimensions determines how it is verbalized — properties with strong visual strength get "looks" verbs, properties with strong haptic strength get "feels" verbs, etc. The mapping can be *derived* from norm data.

**Sources:**
- Lynott, Connell, Brysbaert, Brand & Carney (2020) "The Lancaster Sensorimotor Norms: multidimensional measures of perceptual and action strength for 40,000 English words." Behavior Research Methods, 52, 1271-1291. DOI: 10.3758/s13428-019-01316-z
- Lynott & Connell (2009) "Modality exclusivity norms for 423 object properties." Behavior Research Methods, 41, 558-564. DOI: 10.3758/BRM.41.2.558
- Lynott & Connell (2013) "Modality exclusivity norms for 400 nouns." Behavior Research Methods, 45, 516-526. DOI: 10.3758/s13428-012-0267-0
- Barsalou (1999) "Perceptual symbol systems." Behavioral and Brain Sciences, 22(4), 577-660. DOI: 10.1017/S0140525X99002149

**Translation to RAVANA:**
`_SENSORY_DIM_PHRASE` (engine.py:290) and `_PROP_TO_BINDER` (engine.py:317) are hand-crafted property→phrase maps for the 11 Lancaster dimensions. The brain does not store such lookup tables — it derives the verbalization from sensorimotor feature norms. The fix is to fit property→(verb, binder-dim) from the Lancaster 40k norms into a `data/` artifact (mirroring how `pos_model.json` / `functional_lexicon.json` are stored + loaded). Keep `_LANCASTER_ORDER` as a fixed encoder contract.

---

## E) Adaptive Thresholds (Replace Fixed Cutoffs)

**Brain mechanism:**
The brain uses prediction-error-driven adaptive gating, not fixed thresholds. Friston's Free Energy Principle (2010, Nat Rev Neurosci) formalizes this: precision (inverse variance) is estimated from the statistics of prediction errors and encoded by neuromodulatory gain. Sensory evidence is precision-weighted — the influence of a prediction error scales with its estimated reliability. The brain tracks running statistics (mean and variance) of activation and gates by z-score relative to this dynamic distribution, not by a hard 0.55 cutoff.

Feldman & Friston (2010, Frontiers in Human Neuroscience) demonstrate that attention is precision-weighting: the post-synaptic gain of prediction-error units is modulated based on estimated uncertainty. This is mathematically identical to an EMA z-score gate — the brain learns what is "surprising" relative to its own history.

**Sources:**
- Friston (2010) "The free-energy principle: a unified brain theory?" Nature Reviews Neuroscience, 11, 127-138. DOI: 10.1038/nrn2787
- Feldman & Friston (2010) "Attention, uncertainty, and free-energy." Frontiers in Human Neuroscience, 4, 215. DOI: 10.3389/fnhum.2010.00215
- Friston (2009) "The free-energy principle: a rough guide to the brain?" Trends in Cognitive Sciences, 13(7), 293-301. DOI: 10.1016/j.tics.2009.04.005

**Translation to RAVANA:**
The engine already has the EXEMPLAR: `_vad_baseline` (engine.py:678) — an EMA z-score gate with `{"mu": 0.0, "sigma": 0.3, "n": 0}`. Every fixed threshold (`_RECALL_DETECTION_THRESHOLD=0.55`, schema cos 0.6/0.4/0.5, episodic cos 0.5, self-query sim 0.45, topic sim 0.75, drift/prune 0.7/0.05/0.1) should be replaced by this pattern. Each gets a cold-start prior equal to the current value and decays toward the learned distribution.

---

## F) Shape-Based Junk / Boilerplate Detection

**Brain mechanism:**
The brain detects "junk" / unexpected stimuli via prediction error (Friston 2005, Phil Trans R Soc B). Regular statistical patterns (high vowel ratios, TLD suffixes, embedded digits) generate low prediction error because they are predictable from the statistics of language exposure. Boilerplate text has *low surprise* (it is formulaic), not high surprise — the MMN (mismatch negativity) in auditory cortex fires for unexpected stimuli, while expected boilerplate passes through without triggering attention. The brain does not maintain a hardcoded "blocklist of bad websites" — it learns the *statistical shape* of informative vs. uninformative content.

_WEBSITE_SHAPE (constants.py:401) already implements this principle: regex detection of TLD tails, vowel ratios, and embedded digits. This is exactly the right approach — shape-based, not domain-list-based.

**Sources:**
- Friston (2005) "A theory of cortical responses." Philosophical Transactions of the Royal Society B, 360, 815-836. DOI: 10.1098/rstb.2005.1622
- Rao & Ballard (1999) "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects." Nature Neuroscience, 2, 79-87. DOI: 10.1038/4580

**Translation to RAVANA:**
`_JUNK_SNIPPET_DOMAINS` (engine.py:344) is a ~45-domain blocklist that the code itself criticizes. The fix is to extend the existing `_WEBSITE_SHAPE` shape signals into the snippet-quality path (`snippet_quality.py` / `engine_web_search.py`). Junk should be caught by shape (vowel ratio, digit density, POS profile, TLD shape), not a domain list. The old list stays only as an optional cold-start prior.

---

## G) Connector Learning (Relational Structure Learning)

**Brain mechanism:**
The brain learns relational structure through Hebbian and spike-timing-dependent plasticity (STDP). Temporal contiguity is encoded in synaptic weights: synapses that fire together wire together, and the *order* of firing encodes directional relations (Gerstner et al. 1993, Biol Cybern). The hippocampus encodes sequential structure via temporally asymmetric Hebbian learning (Levy 1996, Hippocampus). Synaptic weights converge to transition probabilities — the probability that event B follows event A is encoded in the strength of the A→B synapse (Surace et al. 2013, Frontiers in Computational Neuroscience).

Critically, connector words like "because", "so", "but" are *learned* from co-occurrence statistics, not hardcoded. Their mapping to semantic relations (causal, contrastive, temporal) can be derived from vector similarity to prototypes.

**Sources:**
- Gerstner, Ritz & van Hemmen (1993) "Why spikes? Hebbian learning and retrieval of time-resolved excitation patterns." Biological Cybernetics, 69, 503-515.
- Surace, Pfister, Gerstner & Teuscher (2013) "Synaptic encoding of temporal contiguity." Frontiers in Computational Neuroscience, 7, 32. DOI: 10.3389/fncom.2013.00032
- Montague, Dayan & Sejnowski (1996) "A framework for mesencephalic dopamine systems based on predictive Hebbian learning." Journal of Neuroscience, 16(5), 1936-1947.

**Translation to RAVANA:**
`ConnectorLearner` (synaptic_dynamics.py:379) already exists and learns connector→relation from GloVe similarity. It is built, tested, but **never imported** by any engine module — only `chain_walker.py` imports `synaptic_dynamics`. The engine uses a hardcoded `_EDGE_CONNECTORS` (v2, weighted, engine.py:591) instead. The fix is to wire `ConnectorLearner` into the engine's edge-building path (engine_graph.py), replacing/augmenting `_EDGE_CONNECTORS` with the learned connector→relation, keeping the hand map as a cold-start / OOV prior.

---

## H) Closed-Class / Function-Word Collapse

**Brain mechanism:**
The brain distinguishes grammatical (closed-class, function) words from lexical (open-class, content) words (Neville et al. 1992, J Cogn Neurosci; Friederici 2002, Trends in Cognitive Sciences). However, this distinction is learned from distributional statistics — function words have high frequency, low imageability, and specific syntactic roles — not from a hardcoded list. ERP studies show an N280 for function words vs. N400 for content words, but this is a *processing* difference, not evidence of separate hardcoded databases. The brain's lexicon is a single distributed network; function words emerge as a natural cluster from frequency × distributional statistics.

**Sources:**
- Neville, Mills & Lawson (1992) "Fractionating language: different neural subsystems with different sensitive periods." Cerebral Cortex, 2(3), 244-258. DOI: 10.1093/cercor/2.3.244
- Brown, Hagoort & ter Keurs (1999) "Electrophysiological signatures of visual lexical processing: open- and closed-class words." Journal of Cognitive Neuroscience, 11(3), 261-281.
- Münte et al. (2001) "Differences in brain potentials to open and closed class words: class and frequency effects." Neuropsychologia, 39(1), 91-102.
- Boye & Harder (2012) "A usage-based theory of grammatical status and grammaticalization." Language, 88(1), 1-44.

**Translation to RAVANA:**
`_COMMON_WORDS`, `TOPIC_SKIP_WORDS`, `_SUBJECT_CONTEXT_WORDS`, `_CONDITIONAL_FRAME`, `_ATTR_WORDS`, and `_GRAMMATICAL_CONCEPTS` are multiple overlapping hand lists. The brain does not store six separate lists — it has one functional lexicon learned from statistics. A single source of truth (`functional_lexicon.json`) + `PosModel` (use_learned_pos, already ON) is sufficient.

---

## I) `_UNIVERSAL_PURGE` Deduplication

**Brain mechanism:**
This is a pure software-engineering concern, not a brain mechanism. The brain does not copy-paste the same pronoun set. However: the brain *does* have an innate bias that function words / pronouns cannot have learned definitions (you cannot "define" the word "you" in the same way you define "quokka"). This is consistent with keeping the set contents as-is but defining it once.

**Source:**
- Not applicable (software architecture, not brain science).

**Translation to RAVANA:**
Define `_UNIVERSAL_PURGE` and `_DEFINITION_ASSERTION` once in `constants.py`, import in all 9 modules (core + 8 mixins). Contents stay — only the duplication is fixed.

---

*End of research report.*
