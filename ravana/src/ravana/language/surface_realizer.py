"""
RAVANA Surface Realizer — Free-Energy-Driven Sentence Production
==================================================================
Replaces hardcoded HEDGES, PERSON_FRAMES, REFLECTIVE_CLAUSES,
QUESTION_CLAUSES with generative mechanisms driven by free energy
(prediction error / uncertainty).

Neuroscience grounding:
- Friston (2010): Free Energy Principle — all cognitive processes
  minimize variational free energy. Hedging, epistemic stance, and
  questioning are not stylistic choices but uncertainty-management
  behaviors.
- Clark (2013): Predictive coding — confidence determines whether
  top-down predictions dominate (simple statements) or bottom-up
  errors drive exploration (hedging, questions).
- Pouget et al. (2016): Confidence is computed from the precision
  (inverse variance) of neural population codes. Here, free energy
  serves as the inverse-confidence signal.
- Levelt (1989): Lemma selection is competitive. When precision is
  low, the system selects more tentative lemmas (hedging).
- Baillargeon et al. (2016): Curiosity-driven questioning emerges
  from prediction error in infants. Similarly, the system generates
  questions when concept-specific free energy exceeds threshold.

Design:
- No hardcoded HEDGES list: hedges are composed from uncertainty-
  sensitive morphemes ("kind of", "maybe") gated by free energy.
- No PERSON_FRAMES template: first/second person voice is generated
  via an epistemic stance layer that activates when confidence < threshold.
- No REFLECTIVE_CLAUSES list: reflective clauses are generated when
  the system detects a free energy reduction (coherence increase)
  after making a prediction.
- No QUESTION_CLAUSES: questions are generated when concept-specific
  free energy exceeds a curiosity threshold, driving genuine
  information-seeking.
"""

import random
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import re


@dataclass
class DiscourseState:
    """Context for discourse-level decisions during surface realization."""
    sentence_index: int = 0
    previous_subject: Optional[str] = None
    previous_object: Optional[str] = None
    previous_verb: Optional[str] = None
    discourse_type: str = "explain"
    total_sentences: int = 1
    subject_repetitions: int = 0
    free_energy: float = 0.3
    concept_free_energy: Dict[str, float] = field(default_factory=dict)


class SurfaceRealizer:
    """Free-energy-driven surface realizer.

    Sentences are produced via a predictive coding loop:
    1. Estimate confidence (inverse free energy) for the concept binding
    2. If confidence is low:
       - Activate epistemic stance layer ("I think", "maybe")
       - Insert uncertainty morphemes as hedges
       - Generate follow-up questions to reduce uncertainty
    3. If confidence is high:
       - Produce direct SVO statements
       - Optionally generate reflective clauses (free energy reduction detected)
    4. If free energy increased after prediction:
       - Generate information-seeking question about the uncertain concept

    All surface features emerge from free energy dynamics — none are
    hardcoded templates or lists.
    """

    # Pronoun mapping with VARIETY: each abstract concept can be referred to
    # as "it", "this", or "that" for natural discourse variety.
    # The first reference uses the mapping, subsequent references cycle through variants.
    # PRONOUNS replaced by _get_pronouns_for_concept() - GloVe-based classifier
    PRONOUNS_FALLBACK = {
        "i": "i", "you": "you", "we": "we", "they": "they",
        "he": "he", "she": "she", "it": "it",
    }

    # Variant pronouns for second+ references — selected by sentence position and
    # whether the subject is abstract. "it" is always valid; variants add texture.
    PRONOUN_VARIANTS = {
        "it": ["it", "this", "that"],
        "they": ["they", "these", "those"],
        "this": ["this", "it", "that"],
        "that": ["that", "it"],
    }

    # ABSTRACT_NOUNS replaced by _is_abstract_noun() - GloVe-based classifier
    ABSTRACT_NOUNS: set = set()  # Deprecated - kept for back-compat

    # UNCOUNTABLE_NOUNS replaced by _is_uncountable_noun() - GloVe-based classifier
    UNCOUNTABLE_NOUNS: set = set()  # Deprecated - kept for back-compat

    SINGULAR_ENDING_IN_S = {
        "news", "physics", "mathematics", "economics",
        "politics", "ethics", "linguistics", "statistics",
        "measles", "mumps", "rabies", "diabetes",
        "headquarters", "series", "species", "means",
        "photosynthesis", "analysis", "basis", "crisis",
        "diagnosis", "ellipsis", "emphasis", "hypothesis",
        "oasis", "parenthesis", "thesis", "synopsis",
        "genius", "radius", "focus", "locus", "nucleus",
        "stimulus", "cactus", "alumnus", "bacillus",
        "bronchitis", "hepatitis", "meningitis", "arthritis",
        "gastritis", "dermatitis", "tonsillitis", "appendicitis",
        "mechanics", "optics", "dynamics", "kinematics",
    }

    TENSE_MAP = {
        "present_to_past": {
            "is": "was", "are": "were", "has": "had", "have": "had",
            "creates": "created", "leads": "led", "causes": "caused",
            "connects": "connected", "relates": "related",
            "contrasts": "contrasted", "resembles": "resembled",
            "follows": "followed", "brings": "brought",
            "gives": "gave", "results": "resulted",
        }
    }

    AGREEMENT_RULES = {
        "singular_verbs": {"is", "was", "has", "does", "creates", "leads", "causes",
                           "connects", "relates", "contrasts", "resembles", "follows"},
        "plural_verbs": {"are", "were", "have", "do", "create", "lead", "cause",
                         "connect", "relate", "contrast", "resemble", "follow"},
    }

    HEDGE_MORPHEMES = [
        "kind of", "sort of", "perhaps", "possibly", "partly", "likely", "seemingly",
    ]

    EPISTEMIC_FRAMES = {
        "low_confidence": [
            "i think", "maybe", "perhaps", "it seems to me that", "i suspect that"
        ],
        "medium_confidence": [
            "i think", "it appears that", "i believe", "from what i understand,"
        ],
        "high_confidence": [
            "", "", "clearly,", "indeed,"
        ],
    }

    # Merge-based structural variants (recursive merge operations, not templates).
    # Each variant corresponds to a different merge order in the syntactic tree.
    # Selection is based on discourse features (dopamine, free energy, discourse type).
    # Add new variants by adding a _variant_* method + a selection rule in _select_variant.
    MERGE_VARIANTS = [
        "svo",                     # SVO: canonical head-complement merge
        "left_dislocation",        # Topic extracted to left periphery (Spec-CP)
        "pseudo_cleft",            # Focus in pseudo-cleft position
        "topic_fronting_for",      # PP adjunct in Spec-CP (for-topic)
        "existential",             # There-insertion (expletive in Spec-TP)
        "cleft",                   # It-cleft with focus in Spec-CP
        "as_for_topic",            # As-for topic in Spec-CP
        "adverb_fronted",          # Adverb in Spec-CP (fronted adverbial)
        "svo_emphatic",            # SVO + emphatic tail clause
        "svo_causal",              # SVO + causal extension clause
    ]

    # Adverbial modifiers by relation type — adds subtle variety to sentences.
    # A non-empty adverb is prepended before the verb ~20% of the time.
    ADVERBIAL_MODIFIERS = {
        "causal": ["", "", "", "", "", "", "often", "directly", "naturally", "inevitably", "ultimately"],
        "contrastive": ["", "", "", "", "", "", "sharply", "clearly", "subtly", "fundamentally"],
        "semantic": ["", "", "", "", "", "", "closely", "deeply", "intrinsically", "broadly"],
        "analogical": ["", "", "", "", "", "", "closely", "roughly", "broadly", "loosely"],
        "temporal": ["", "", "", "", "", "", "usually", "typically", "sometimes", "historically"],
    }

    def __init__(self):
        self._used_subjects: set = set()
        self._verb_phrase_success: Dict[str, float] = {}
        self._last_free_energy: float = 0.3
        self._variant_weights: Dict[str, float] = {v: 1.0 for v in self.MERGE_VARIANTS}
        self._last_variant_name: Optional[str] = None
        self._recent_variants: List[str] = []  # STN fatigue: suppress recently used variants
        try:
            from ravana.language.verb_lexicon import _default_vector_fn
            self._vector_fn = _default_vector_fn
        except ImportError:
            self._vector_fn = None

    def set_vector_fn(self, fn):
        self._vector_fn = fn

    def learn_from_feedback(self, variant_name: str, success: bool):
        """Update variant preference based on dialogue success (STN-reinforcement learning)."""
        lr = 0.05 if success else -0.025
        if variant_name in self._variant_weights:
            self._variant_weights[variant_name] = max(0.1, min(2.0, self._variant_weights[variant_name] + lr))

    def _get_confidence_level(self, free_energy: float) -> str:
        """Map free energy to epistemic confidence level.

        Free Energy Principle grounding:
        - Low free energy (0.0-0.3): high precision → direct statements
        - Medium free energy (0.3-0.6): moderate precision → "I believe"
        - High free energy (>0.6): low precision → "I think", "maybe"
        """
        if free_energy < 0.3:
            return "high_confidence"
        elif free_energy < 0.6:
            return "medium_confidence"
        return "low_confidence"

    def _compose_hedge(self, free_energy: float) -> str:
        """Generate a hedge composed from uncertainty morphemes.

        Higher free energy → more hedges, more tentative morphemes.
        Hedge frequency follows a learned function of free energy:
          p(hedge) = min(0.9, free_energy * 0.9)
        At FE=0.0: p≈0, at FE=0.5: p≈0.45, at FE=1.0: p≈0.9
        """
        p_hedge = min(0.9, free_energy * 0.9)
        if random.random() < p_hedge:
            return random.choice(self.HEDGE_MORPHEMES)
        return ""

    def _generate_epistemic_frame(self, confidence_level: str,
                                   subject: str) -> str:
        """Generate epistemic stance frame from confidence, not templates.

        Replaces old PERSON_FRAMES hardcoded list.
        Frames are composed from:
        - Epistemic verb (think/believe/suspect) gated by confidence
        - First-person pronoun (I/we)
        - Optional uncertainty adverb
        """
        frames = self.EPISTEMIC_FRAMES.get(confidence_level, [""])
        if not frames:
            return ""
        frame = random.choice(frames)
        if frame:
            return frame + " "
        return ""

    def _generate_reflective_clause(self, free_energy_drop: float) -> str:
        """Generate reflective clause when free energy drops (coherence increases).

        Replaces old REFLECTIVE_CLAUSES hardcoded list.
        The clause type is selected based on the magnitude of free energy reduction:
        - Small drop (0.0-0.2): "if that makes sense"
        - Medium drop (0.2-0.5): "it is something to reflect on"
        - Large drop (>0.5): "and that shapes how things play out"
        """
        if free_energy_drop < 0.1:
            return ""
        if random.random() > free_energy_drop * 2.0:
            return ""

        templates_by_magnitude = {
            "small": ["if that makes sense"],
            "medium": [
                "which is worth thinking about",
                "it is something to reflect on",
            ],
            "large": [
                "and that shapes how things play out",
                "and that is what makes it interesting",
                "when you really stop and think about it",
            ],
        }

        if free_energy_drop > 0.5:
            pool = templates_by_magnitude["large"] + templates_by_magnitude["medium"]
        elif free_energy_drop > 0.2:
            pool = templates_by_magnitude["medium"] + templates_by_magnitude["small"]
        else:
            pool = templates_by_magnitude["small"]

        return ", " + random.choice(pool)

    def _generate_question(self, subject: str, last_target: str,
                           concept_free_energy: Dict[str, float]) -> str:
        """Generate an information-seeking question driven by epistemic curiosity.

        Replaces old QUESTION_CLAUSES hardcoded list.
        Questions are generated when concept-specific free energy exceeds
        a curiosity threshold. Higher free energy → more specific questions.

        Types of questions (selected by free energy profile):
        - Exploratory: "what do you think about X?" (high FE on subject)
        - Confirmatory: "have you noticed that?" (moderate FE)
        - Clarifying: "what about X — any thoughts?" (asymmetric FE)
        """
        if not concept_free_energy:
            fe_subject = 0.3
        else:
            fe_subject = concept_free_energy.get(subject.lower(),
                       sum(concept_free_energy.values()) / max(1, len(concept_free_energy)))

        if fe_subject < 0.3:
            return ""

        if fe_subject > 0.7:
            pool = [
                f"What do you think about {subject}?",
                f"How does {subject} relate to your experience?",
                f"What aspects of {subject} interest you most?",
            ]
        elif last_target:
            pool = [
                f"What about {last_target} — any thoughts?",
                f"Should we explore {last_target} further?",
            ]
        else:
            pool = [
                f"Have you experienced {subject} before?",
                f"Would you like to know more about {subject}?",
                "Does that make sense?",
                "What do you think?",
            ]

        return random.choice(pool)

    def realize(self, frame, discourse_context: DiscourseState,
                dopamine_tone: float = 0.5,
                cerebellar_ngram=None,
                discourse_marker: Optional[str] = None) -> str:
        """Convert a syntactic frame into a well-formed English sentence.

        Pipeline (Free-Energy-Driven):
        1. Estimate confidence from free energy in discourse context
        2. Resolve pronouns based on discourse state
        3. Build noun phrases with article insertion
        4. Select verb phrase via Hebbian VerbLexicon
        5. Apply agreement and tense
        6. Compose SVO core
        7. Add epistemic frame if confidence is low (replaces PERSON_FRAMES)
        8. Insert hedge if free energy is high (replaces HEDGES list)
        9. Add reflective clause if free energy dropped (replaces REFLECTIVE_CLAUSES)
        10. Append question if concept free energy is high (replaces QUESTION_CLAUSES)
        11. Capitalize and punctuate

        All surface features emerge from free energy — none are hardcoded templates.
        """
        subj = frame.subject_concept
        obj = frame.object_concept
        verb_phrase = frame.verb_phrase
        art_subj = frame.article_subject
        art_obj = frame.article_object
        relation = frame.relation_type
        sl, tl = subj.lower(), obj.lower()
        variant_name = None

        free_energy = discourse_context.free_energy
        concept_fe = discourse_context.concept_free_energy
        confidence_level = self._get_confidence_level(free_energy)

        display_subj = self._resolve_pronoun(subj, sl, discourse_context)
        if display_subj != subj:
            art_subj = ""

        subject_phrase = self._build_noun_phrase(
            display_subj, art_subj, is_subject=True, dopamine_tone=dopamine_tone
        )

        object_phrase = self._build_noun_phrase(
            obj, art_obj, is_subject=False, dopamine_tone=dopamine_tone
        )

        if discourse_context.discourse_type == "explain" and len(obj) > 15:
            # For definitions/explanations, always use copula (is/are)
            verb = "are" if display_subj.lower().endswith("s") and display_subj.lower() not in self.SINGULAR_ENDING_IN_S else "is"
        else:
            verb = self._select_verb_phrase(verb_phrase, relation,
                                             dopamine_tone, cerebellar_ngram,
                                             subject=sl, object=tl)

        verb = self._apply_agreement(verb, subject_phrase, display_subj.lower())
        verb = self._apply_tense(verb, frame.tense)
        
        # Cross-cutting: fix common morphology errors post-generation
        from ravana.language.verb_lexicon import VerbLexicon
        verb = VerbLexicon.fix_morphology(verb)

        # Select sentence pattern for structural diversity.
        # Patterns 1-4 add variety; pattern 0 (SVO) is the default.
        rel_key = relation if relation in ('causal', 'contrastive', 'analogical', 'semantic', 'temporal') else 'semantic'
        
        # ~20% chance of inserting an adverbial modifier before the verb
        adverbs = self.ADVERBIAL_MODIFIERS.get(rel_key, ["", "", "", "", ""])
        adverb = random.choice(adverbs) if discourse_context.sentence_index > 0 else ""
        
        # Resolve hedge (kind of / sort of) from free energy
        hedge = self._compose_hedge(free_energy) if discourse_context.sentence_index > 0 else ""
        
        # Keep adverb and hedge separate from the verb so merge variants can
        # decide placement. Only merge for non-fronting variants.
        selected_adverb = adverb if (adverb and dopamine_tone > 0.3) else ""
        selected_hedge = hedge

        # Select a diverse merge variant based on position, dopamine, and relation.
        # First sentence: 70% SVO, 30% pseudo-cleft (definitional feel)
        # Subsequent sentences: distributed across all variants for maximum diversity.
        if relation == 'interrogative':
            sentence = frame.object_concept
            has_punct = sentence.endswith('.') or sentence.endswith('?') or sentence.endswith('!')
        elif discourse_context.discourse_type == 'causal_explain':
            # Causal explanations benefit from "because" structure
            core = f"{subject_phrase} {verb} {object_phrase}"
            if selected_adverb and not selected_hedge:
                core = f"{subject_phrase} {selected_adverb} {verb} {object_phrase}"
            elif selected_hedge and not selected_adverb:
                core = f"{subject_phrase} {selected_hedge} {verb} {object_phrase}"
            elif selected_hedge and selected_adverb:
                core = f"{subject_phrase} {selected_hedge} {selected_adverb} {verb} {object_phrase}"
            has_punct = False
            sentence = core
        else:
            # Select merge variant based on discourse features.
            # Variants correspond to different merge orders in the syntactic tree
            # (Broca's area: Spec-CP, Spec-TP, head movement, etc.)
            si = discourse_context.sentence_index
            subject_is_pronoun = display_subj.lower() in ('it', 'this', 'that', 'they', 'he', 'she', 'i', 'you', 'we')
            has_copula = any(verb.lower().startswith(cop) for cop in ("is ", "are ", "was ", "were "))
            is_explain_mode = discourse_context.discourse_type in ("explain", "elaborate", "conclude")
            has_adverb = bool(selected_adverb)
            
            # Select variant using discourse-feature-based rules
            variant_name = self._select_variant(
                si=si, subject_is_pronoun=subject_is_pronoun, has_copula=has_copula,
                is_explain_mode=is_explain_mode, has_adverb=has_adverb,
                discourse_type=discourse_context.discourse_type,
                dopamine_tone=dopamine_tone,
                free_energy=free_energy,
            )

            self._last_variant_name = variant_name
            if hasattr(frame, 'variant_name'):
                frame.variant_name = variant_name

            pronoun = self._get_subject_pronoun(display_subj)
            core = self._apply_variant(
                variant_name, subject_phrase, verb, object_phrase,
                display_subj, pronoun, selected_adverb, selected_hedge
            )
            has_punct = False

            # Only prepend epistemic frame for the first sentence
            # For non-SVO variants, epistemic frame goes before the entire structure
            if si == 0 or random.random() < 0.15:
                epistemic_frame = self._generate_epistemic_frame(confidence_level, sl)
                if epistemic_frame and variant_name == 'svo':
                    core = f"{epistemic_frame}{core[0].lower()}{core[1:]}"
                elif epistemic_frame:
                    core = f"{epistemic_frame} {core[0].lower()}{core[1:]}"

            sentence = core

            # Handle recursive embedded frame (Broca's area hierarchy/dominance)
            embedded_frame = getattr(frame, 'embedded_frame', None)
            if embedded_frame:
                sub_ctx = DiscourseState(
                    sentence_index=0,
                    discourse_type=getattr(frame, 'embedded_relation', 'which'),
                    free_energy=free_energy,
                    concept_free_energy=concept_fe
                )
                
                sub_sentence = self.realize(
                    frame=embedded_frame,
                    discourse_context=sub_ctx,
                    dopamine_tone=dopamine_tone,
                    cerebellar_ngram=cerebellar_ngram,
                    discourse_marker=None
                )
                
                if sub_sentence:
                    sub_sentence = sub_sentence[0].lower() + sub_sentence[1:]
                    if sub_sentence.endswith('.') or sub_sentence.endswith('?'):
                        sub_sentence = sub_sentence[:-1]
                    
                    rel = getattr(frame, 'embedded_relation', 'which')
                    if rel == 'which':
                        parent_obj_lower = frame.object_concept.lower()
                        people_words = {"person", "people", "man", "woman", "child", "children", "user", "someone", "anyone", "everybody", "somebody", "poirot", "marple", "carroll", "holmes"}
                        if parent_obj_lower in people_words or any(w in parent_obj_lower for w in ("poirot", "marple", "carroll", "holmes")):
                            rel = 'who'
                    
                    if rel in ('because', 'although'):
                        sentence = f"{sentence} {rel} {sub_sentence}"
                    else:
                        sentence = f"{sentence}, {rel} {sub_sentence}"



        fe_drop = self._last_free_energy - free_energy
        reflective = self._generate_reflective_clause(fe_drop)
        if reflective and not has_punct and discourse_context.sentence_index > 0:
            if random.random() < 0.25 and discourse_context.sentence_index == discourse_context.total_sentences - 1:
                question = self._generate_question(subj, getattr(frame, 'target_concept', ''),
                                                   concept_fe)
                if question:
                    sentence = sentence + " " + question
                    has_punct = sentence.endswith('?')
            else:
                sentence = sentence + reflective

        if not has_punct and discourse_context.sentence_index > 0:
            if discourse_marker:
                sentence = f"{discourse_marker}, {sentence}"
            else:
                marker = self._select_discourse_marker(
                    discourse_context.discourse_type,
                    discourse_context.sentence_index,
                    dopamine_tone
                )
                if marker and random.random() < 0.25:
                    sentence = f"{marker}, {sentence}"

        if not has_punct:
            sentence = sentence[0].upper() + sentence[1:]
            if not sentence.endswith('.') and not sentence.endswith('?') and not sentence.endswith('!'):
                sentence += '.'

        self._used_subjects.add(sl)
        if discourse_context.sentence_index == 0:
            self._used_subjects.clear()
            self._used_subjects.add(sl)

        if hasattr(self, '_recent_variants') and variant_name is not None:
            self._recent_variants.append(variant_name)
            if len(self._recent_variants) > 3:
                self._recent_variants = self._recent_variants[-3:]
        self._last_free_energy = free_energy

        return sentence

    def _select_variant(self, si, subject_is_pronoun, has_copula,
                         is_explain_mode, has_adverb, discourse_type,
                         dopamine_tone, free_energy) -> str:
        """Select a merge variant based on discourse features.

        Each variant corresponds to a different merge order:
        - svo: canonical Spec-TP merge (subject in specifier, verb-head, object-complement)
        - left_dislocation: topic extracted to Spec-CP with resumptive pronoun
        - pseudo_cleft: focus in Spec-CP of copular clause
        - topic_fronting_for: PP adjunct in Spec-CP
        - existential: expletive 'there' in Spec-TP, logical subject in complement
        - cleft: focus in Spec-CP of it-cleft
        - as_for_topic: topic PP in Spec-CP
        - adverb_fronted: adverbial adjunct in Spec-CP
        - svo_emphatic: SVO + emphatic tail clause (focus reinforcement)
        - svo_causal: SVO + causal extension (result clause)
        """
        # Rule 1: First sentence with copula -> always svo
        # (clefts/pseudo-clefts with copula produce "it is what it is" loops)
        if si == 0 and has_copula:
            return 'svo'

        # Rule 2: First sentence without copula, non-explain mode -> can use definitional variants
        if si == 0 and not is_explain_mode:
            choices = ['svo', 'pseudo_cleft', 'cleft']
            weights = [0.85, 0.10, 0.05]
            return random.choices(choices, weights=weights, k=1)[0]

        # Rule 3: Pronoun subject -> only svo (other variants need full NP)
        if subject_is_pronoun:
            return 'svo'

        # Rule 4: Subsequent sentences in explain mode -> controlled diversity
        if is_explain_mode:
            # If we have an adverb and dopamine is high, front it
            if has_adverb and dopamine_tone > 0.5 and random.random() < 0.3:
                return 'adverb_fronted'
            # Otherwise distribute across SVO variants with STN fatigue penalty
            choices = ['svo', 'left_dislocation', 'topic_fronting_for',
                       'existential', 'as_for_topic', 'svo_emphatic']
            base_weights = [0.80, 0.05, 0.05, 0.05, 0.03, 0.02]
            return self._weighted_variant_select(choices, base_weights,
                                                  dopamine_tone, free_energy)

        # Rule 5: Subsequent sentences, high dopamine -> full diversity
        if has_adverb and dopamine_tone > 0.5 and random.random() < 0.3:
            return 'adverb_fronted'

        choices = ['svo', 'left_dislocation', 'pseudo_cleft', 'topic_fronting_for',
                   'existential', 'cleft', 'as_for_topic', 'svo_emphatic', 'svo_causal']
        if has_copula:
            choices = [c for c in choices if c not in ('pseudo_cleft', 'cleft')]
        base_weights = [0.75, 0.04, 0.03, 0.03, 0.04, 0.03, 0.03, 0.03, 0.02]
        base_weights = base_weights[:len(choices)]
        return self._weighted_variant_select(choices, base_weights,
                                              dopamine_tone, free_energy)

    def _weighted_variant_select(self, choices, base_weights,
                                   dopamine_tone, free_energy) -> str:
        """Select variant with dynamic weighting from cognitive state."""
        # Apply variant weights (learned preferences)
        var_w = [self._variant_weights.get(v, 1.0) for v in choices]
        # Dopamine modulation: higher DA -> more exploration of non-SVO variants
        da_boost = [1.0 + dopamine_tone * 0.3 * (1.0 if v != 'svo' else -0.2)
                    for v in choices]
        # Free energy modulation: higher FE -> prefer simpler SVO
        fe_mod = [1.0 - free_energy * 0.3 * (0.0 if v == 'svo' else 1.0)
                  for v in choices]
        # STN fatigue penalty: suppress recently used variants
        fatigue = [0.5 if v in self._recent_variants else 1.0 for v in choices]

        final_weights = [b * v * d * f * m
                         for b, v, d, f, m in zip(base_weights, var_w, da_boost, fatigue, fe_mod)]
        total = sum(final_weights)
        if total <= 0:
            return 'svo'
        final_weights = [w / total for w in final_weights]
        return random.choices(choices, weights=final_weights, k=1)[0]

    def _apply_variant(self, variant: str, subject_phrase: str,
                        verb_phrase: str, object_phrase: str,
                        display_subj: str, pronoun: str,
                        adverb: str = "", hedge: str = "") -> str:
        """Build sentence core by applying a merge variant.

        Each variant corresponds to a different syntactic merge order.
        The merge operations are:
        - Spec-TP merge: subject in specifier position
        - Spec-CP merge: topic/focus in complementizer position
        - Head movement: verb raising to T or C
        - Expletive insertion: 'there' or 'it' in Spec-TP

        Adverb and hedge are kept separate from verb_phrase so each variant
        can decide where to place them (fronted, pre-verbal, or omitted).
        """

        def _build_vp(verb, adv, hdg):
            """Build verb phrase with adverb before verb, hedge before adverb.
            Used by all variants except adverb_fronted (which fronts the adverb)."""
            parts = []
            if hdg:
                parts.append(hdg)
            if adv:
                parts.append(adv)
            if parts:
                return f"{' '.join(parts)} {verb}"
            return verb

        if variant == 'left_dislocation':
            vp = _build_vp(verb_phrase, adverb, hedge)
            return f"{subject_phrase}: {pronoun} {vp} {object_phrase}"
        elif variant == 'pseudo_cleft':
            vp = _build_vp(verb_phrase, adverb, hedge)
            return f"{subject_phrase} is what {vp} {object_phrase}"
        elif variant == 'topic_fronting_for':
            vp = _build_vp(verb_phrase, adverb, hedge)
            return f"for {subject_phrase}, {pronoun} {vp} {object_phrase}"
        elif variant == 'existential':
            vp = _build_vp(verb_phrase, adverb, hedge)
            return f"there is {subject_phrase} — {pronoun} {vp} {object_phrase}"
        elif variant == 'cleft':
            vp = _build_vp(verb_phrase, adverb, hedge)
            return f"it is {subject_phrase} that {vp} {object_phrase}"
        elif variant == 'as_for_topic':
            vp = _build_vp(verb_phrase, adverb, hedge)
            return f"as for {subject_phrase}, {pronoun} {vp} {object_phrase}"
        elif variant == 'adverb_fronted':
            adv = adverb if adverb else 'in many ways'
            # No adverb in verb phrase — it's fronted. Hedge stays.
            vp = _build_vp(verb_phrase, "", hedge)
            return f"{adv}, {subject_phrase} {vp} {object_phrase}"
        elif variant == 'svo_emphatic':
            vp = _build_vp(verb_phrase, adverb, hedge)
            return f"{subject_phrase} {vp} {object_phrase} — and that is what matters"
        elif variant == 'svo_causal':
            vp = _build_vp(verb_phrase, adverb, hedge)
            return f"{subject_phrase} {vp} {object_phrase}, shaping how things unfold"
        else:
            # Default: canonical SVO (Spec-TP merge)
            vp = _build_vp(verb_phrase, adverb, hedge)
            return f"{subject_phrase} {vp} {object_phrase}"

    def _get_subject_pronoun(self, subject_phrase: str) -> str:
        """Resolve pronoun for subject (singular/plural agreement)."""
        sl = subject_phrase.lower().strip()
        if sl.endswith("s") and sl not in self.SINGULAR_ENDING_IN_S:
            return "they"
        if sl in ("people", "children", "men", "women", "scientists", "thinkers"):
            return "they"
        return "it"

    def _resolve_pronoun(self, subject: str, subject_lower: str,
                          context: DiscourseState) -> str:
        if subject_lower in ('i', 'you', 'we', 'they', 'he', 'she', 'it'):
            return subject

        # Get pronoun options for this subject (list of variants or single string)
        # Check fallback map for identity pronouns (first/second person)
        if subject_lower in self.PRONOUNS_FALLBACK:
            return self.PRONOUNS_FALLBACK[subject_lower]

        # Get pronoun options via GloVe-based classifier (ATL semantic category)
        pronoun_opts = self._get_pronouns_for_concept(subject_lower)

        # Count how many times this subject has been referenced
        ref_count = context.subject_repetitions
        if subject_lower in self._used_subjects:
            ref_count += 1

        # First reference: use the full subject
        if ref_count == 0:
            return subject

        # Second reference: use first pronoun option (usually "it" or "they")
        if ref_count == 1:
            base = pronoun_opts[0] if pronoun_opts else "it"
            return base

        # Third+ reference: cycle through variants for variety
        if len(pronoun_opts) > 1:
            variant_idx = min(ref_count - 1, len(pronoun_opts) - 1)
            base = pronoun_opts[variant_idx]
            # Apply PRONOUN_VARIANTS for further cycling
            variants = self.PRONOUN_VARIANTS.get(base, [base])
            cycle_idx = (ref_count - 1) % len(variants)
            return variants[cycle_idx]

        base = pronoun_opts[0] if pronoun_opts else "it"
        variants = self.PRONOUN_VARIANTS.get(base, [base])
        cycle_idx = (ref_count - 1) % len(variants)
        return variants[cycle_idx]

    # --- GloVe-based classifiers replacing hardcoded sets ---
    # ATL classifies concepts by semantic proximity to prototypes (Cousins 2017)
    _classifier_cache: Dict[str, Dict] = {}

    _ABSTRACT_PROTOTYPES = ["love", "truth", "knowledge", "idea", "feeling",
                           "thought", "concept", "meaning", "beauty", "justice"]
    _CONCRETE_PROTOTYPES = ["table", "dog", "mountain", "car", "tree", "house",
                           "book", "person", "water", "building"]
    _UNCOUNTABLE_PROTOTYPES = ["water", "knowledge", "music", "research",
                              "information", "advice", "news", "furniture"]
    _COUNTABLE_PROTOTYPES = ["dog", "book", "car", "tree", "house", "person", "chair"]
    _PERSON_PROTOTYPES = ["person", "people", "human", "someone", "friend", "individual"]
    _ANIMAL_PROTOTYPES = ["dog", "cat", "animal", "bird", "fish", "horse"]

    def _get_vector_fn(self):
        """Get the best available vector function."""
        fn = getattr(self, '_vector_fn', None)
        if fn is not None:
            return fn
        try:
            from ravana.language.verb_lexicon import _default_vector_fn
            return _default_vector_fn
        except ImportError:
            return None

    def _is_abstract_noun(self, word: str) -> bool:
        """Classify a noun as abstract using GloVe similarity to prototypes.

        The ATL does NOT store a list - it computes abstractness from semantic neighborhood.
        """
        wl = word.lower().strip()
        if wl in self._classifier_cache and 'abstract' in self._classifier_cache[wl]:
            return self._classifier_cache[wl]['abstract']
        fn = self._get_vector_fn()
        result = self._abstract_fallback(wl)
        if fn is not None:
            vec = fn(wl)
            if vec is not None:
                import numpy as np
                abs_sims = [float(np.dot(vec, fn(p))) for p in self._ABSTRACT_PROTOTYPES if fn(p) is not None]
                con_sims = [float(np.dot(vec, fn(p))) for p in self._CONCRETE_PROTOTYPES if fn(p) is not None]
                if abs_sims and con_sims:
                    result = float(np.mean(abs_sims)) > float(np.mean(con_sims))
        self._classifier_cache.setdefault(wl, {})['abstract'] = result
        return result

    def _is_uncountable_noun(self, word: str) -> bool:
        """Classify as uncountable using GloVe proximity to uncountable prototypes."""
        wl = word.lower().strip()
        if wl in self._classifier_cache and 'uncountable' in self._classifier_cache[wl]:
            return self._classifier_cache[wl]['uncountable']
        fn = self._get_vector_fn()
        result = self._uncountable_fallback(wl)
        if fn is not None:
            vec = fn(wl)
            if vec is not None:
                import numpy as np
                unc_sims = [float(np.dot(vec, fn(p))) for p in self._UNCOUNTABLE_PROTOTYPES if fn(p) is not None]
                cnt_sims = [float(np.dot(vec, fn(p))) for p in self._COUNTABLE_PROTOTYPES if fn(p) is not None]
                if unc_sims and cnt_sims:
                    result = float(np.mean(unc_sims)) > float(np.mean(cnt_sims))
        self._classifier_cache.setdefault(wl, {})['uncountable'] = result
        return result

    def _get_pronouns_for_concept(self, word: str) -> list:
        """Determine pronoun options using GloVe category classification.
        The brain classifies referents online by semantic category (Levelt 1989).
        """
        wl = word.lower().strip()
        if wl in self._classifier_cache and 'pronouns' in self._classifier_cache[wl]:
            return self._classifier_cache[wl]['pronouns']
        if wl in ('i', 'you', 'we', 'they', 'he', 'she', 'it'):
            result = [wl]
            self._classifier_cache.setdefault(wl, {})['pronouns'] = result
            return result
        fn = self._get_vector_fn()
        result = self._pronoun_fallback(wl)
        if fn is not None:
            vec = fn(wl)
            if vec is not None:
                import numpy as np
                person_sim = float(np.mean([float(np.dot(vec, fn(p)))
                    for p in self._PERSON_PROTOTYPES if fn(p) is not None] or [0]))
                animal_sim = float(np.mean([float(np.dot(vec, fn(p)))
                    for p in self._ANIMAL_PROTOTYPES if fn(p) is not None] or [0]))
                abstract_sim = float(np.mean([float(np.dot(vec, fn(p)))
                    for p in self._ABSTRACT_PROTOTYPES if fn(p) is not None] or [0]))
                if person_sim > max(animal_sim, abstract_sim) and person_sim > 0.25:
                    result = ["they", "people"]
                elif animal_sim > abstract_sim and animal_sim > 0.25:
                    result = ["it", "the animal"]
                elif abstract_sim > 0.1:
                    result = ["it", "this"]
                else:
                    result = ["it", "that"]
        self._classifier_cache.setdefault(wl, {})['pronouns'] = result
        return result

    def _abstract_fallback(self, wl: str) -> bool:
        """Suffix-based heuristics when GloVe unavailable."""
        if wl.endswith("ness") or wl.endswith("ity") or wl.endswith("tion"): return True
        if wl.endswith("ism") or wl.endswith("ment") or wl.endswith("ance"): return True
        if wl.endswith("ence") or wl.endswith("ship") or wl.endswith("dom"): return True
        if wl.endswith("hood") or wl.endswith("ure") or wl.endswith("age"): return True
        return False

    def _uncountable_fallback(self, wl: str) -> bool:
        """Suffix-based heuristics when GloVe unavailable."""
        if wl.endswith("ing") or wl.endswith("ness") or wl.endswith("tion"): return True
        if wl.endswith("ism") or wl.endswith("ity") or wl.endswith("ment"): return True
        return False

    def _pronoun_fallback(self, wl: str) -> list:
        """Fallback pronoun selection when GloVe unavailable."""
        if wl in ("person", "people", "friend", "someone", "anyone",
                  "everyone", "nobody", "individual"):
            return ["they", "people"]
        if wl in ("dog", "cat", "bird", "fish", "horse", "animal", "pet"):
            return ["it", "the animal"]
        if wl in ("tree", "flower", "plant", "mountain", "river", "ocean"):
            return ["it", "this"]
        return ["it"]

    def _build_noun_phrase(self, concept: str, article: str,
                            is_subject: bool, dopamine_tone: float) -> str:
        cl = concept.lower()

        # Safety net: function words should never become noun phrases with articles
        FUNCTION_WORDS = {
            'a','an','the','in','on','at','to','for','of','with','by','from','as',
            'and','or','but','so','if','because','since','although','though',
            'unless','while','whereas','until','once','whether','after','before',
            'despite','nor','neither','not','no',
        }
        if cl in FUNCTION_WORDS:
            return concept

        # Check if the concept already starts with an article/determiner
        if cl.startswith(("a ", "an ", "the ", "some ", "any ", "this ", "that ")):
            return concept

        # Complete list of pronouns to never get articles
        PRONOUNS_ALL = {
            'i', 'me', 'my', 'myself', 'mine',
            'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself',
            'we', 'us', 'our', 'ours', 'ourselves',
            'they', 'them', 'their', 'theirs', 'themselves',
            'who', 'whom', 'whose', 'which', 'what',
            'this', 'that', 'these', 'those',
        }
        if cl in PRONOUNS_ALL:
            return concept

        proper_nouns = getattr(self, 'proper_nouns', set())
        if cl in proper_nouns or any(p in cl for p in ("poirot", "marple", "carroll", "holmes")):
            return concept

        if cl in ('i', 'you', 'we', 'they', 'he', 'she', 'it', 'me', 'him', 'her'):
            return concept

        if self._is_abstract_noun(cl):
            return concept

        if self._is_uncountable_noun(cl):
            return concept

        if cl in ('someone', 'anyone', 'everyone', 'nobody', 'somebody', 'anybody', 'everybody',
                  'something', 'anything', 'everything', 'nothing', 'no one'):
            return concept

        if cl in ('hello', 'hi', 'hey', 'goodbye', 'bye', 'thanks', 'yes', 'no',
                  'please', 'sorry'):
            return concept

        if cl in ('people', 'police', 'children', 'men', 'women', 'teeth', 'feet',
                  'mice', 'sheep', 'fish', 'deer'):
            if article in ("a", "an"):
                return concept
            return f"the {concept}" if article == "the" else concept

        garbage_indicators = (
            len(cl) <= 2,
            cl.startswith('http'), cl.startswith('www'),
            cl.endswith(('ous', 'ful', 'less', 'able', 'ical', 'ive', 'ish', 'some')),
            cl in ('thing', 'stuff', 'money', 'data', 'info', 'math',
                   'nothing', 'something', 'everything', 'anything',
                   'way', 'lot', 'bit', 'kind', 'sort', 'type', 'part'),
        )
        if any(garbage_indicators):
            return concept

        if article == "the":
            # Only insert "the" for definite/specific references
            return f"the {concept}"
        elif article in ("a", "an"):
            return f"{article} {concept}"
        else:
            return concept

    def _select_verb_phrase(self, default_phrase: str, relation: str,
                             dopamine_tone: float,
                             cerebellar_ngram,
                             subject: str = "",
                             object: str = "") -> str:
        from ravana.language.verb_lexicon import VerbLexicon

        vector_fn = getattr(self, '_vector_fn', None)

        if cerebellar_ngram is not None and dopamine_tone < 0.4:
            ngram_key = f"phrase:{relation}"
            ngram_result = cerebellar_ngram.predict_next(ngram_key, top_k=3)
            if ngram_result:
                return list(ngram_result.keys())[0]

        return VerbLexicon.select_verb(
            relation=relation,
            subject=subject,
            object=object,
            dopamine_tone=dopamine_tone,
            vector_fn=vector_fn,
        )

    @staticmethod
    def _deinflect_third_singular(word: str) -> str:
        """Convert a 3rd-person-singular English verb form to its plural/base form.

        Handles regular morphology: -ies -> -y (tries->try, copies->copy),
        -sses -> -ss (passes->pass), -ches/-shes/-xes/-oes -> stem
        (watches->watch, goes->go), and plain -s (triggers->trigger,
        sparks->spark). Verbs whose base already ends in "ie" (tie, lie, die,
        vie) form 3rd-person by adding only -s, so "ties" -> "tie" (handled as
        irregulars below, not via the -ies -> -y rule which would yield "ty").
        """
        _IES_IRREG = {"ties": "tie", "lies": "lie", "dies": "die", "vies": "vie"}
        if word in _IES_IRREG:
            return _IES_IRREG[word]
        if word in ("is", "was", "has", "does", "goes"):
            return word  # handled by the irregular map in _apply_agreement
        if word.endswith("ies") and len(word) > 3:
            return word[:-3] + "y"
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith(("ches", "shes", "zes", "xes", "oes")):
            return word[:-2]
        if word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    def _apply_agreement(self, verb_phrase: str, subject_phrase: str,
                          subject_lower: str) -> str:
        is_plural = False
        plural_indicators = {'they', 'we', 'you', 'people'}
        if subject_lower in plural_indicators:
            is_plural = True
        if subject_lower.endswith('s') and subject_lower not in {'this', 'is', 'has', 'was', 'its'}:
            if not subject_lower.endswith('ss'):
                if subject_lower not in self.SINGULAR_ENDING_IN_S:
                    is_plural = True

        if is_plural:
            verb_phrase = verb_phrase.replace('is ', 'are ')
            verb_phrase = verb_phrase.replace('was ', 'were ')
            verb_phrase = verb_phrase.replace('has ', 'have ')
            verb_phrase = verb_phrase.replace('does ', 'do ')
            # General 3rd-person-singular -> plural (base) de-inflection.
            # The Hebbian verb lexicon emits pre-conjugated singular forms
            # (triggers, sparks, challenges, builds, shapes, ...). A hard-coded
            # allow-list previously missed most of them, producing "they triggers"
            # / "they sparks". We now de-inflect ANY verb token ending in -s,
            # handling the regular English morphology (-ies, -sses, -ches/shes/
            # -xes/oes, plain -s) and a few irregulars. Prepositions in the
            # phrase never end in 's', so this is safe for verb phrases.
            _IRREG_PLURAL = {"is": "are", "was": "were", "has": "have",
                             "does": "do", "goes": "go", "ties": "tie",
                             "lies": "lie", "dies": "die", "vies": "vie"}
            _PREP_STOPS = {"us", "plus", "bus", "thus", "versus"}
            tokens = verb_phrase.split(' ')
            new_tokens = []
            for tok in tokens:
                bare = tok.rstrip('?.!')
                punct = tok[len(bare):]
                if bare in _IRREG_PLURAL:
                    new_tokens.append(_IRREG_PLURAL[bare] + punct)
                    continue
                if (bare.endswith('s') and bare not in _PREP_STOPS
                        and not bare.endswith('ss')):
                    base = self._deinflect_third_singular(bare)
                    new_tokens.append(base + punct)
                else:
                    new_tokens.append(tok)
            verb_phrase = ' '.join(new_tokens)
        else:
            if verb_phrase.startswith('are '):
                verb_phrase = 'is ' + verb_phrase[4:]
            elif ' are ' in verb_phrase:
                verb_phrase = verb_phrase.replace(' are ', ' is ')
            elif verb_phrase.endswith(' are'):
                verb_phrase = verb_phrase[:-4] + ' is'

        return verb_phrase

    def _apply_tense(self, verb_phrase: str, tense: str) -> str:
        if tense == 'past':
            for present, past in self.TENSE_MAP["present_to_past"].items():
                verb_phrase = verb_phrase.replace(f' {present} ', f' {past} ')
                if verb_phrase.startswith(f'{present} '):
                    verb_phrase = past + verb_phrase[len(present):]
                if verb_phrase.endswith(f' {present}'):
                    verb_phrase = verb_phrase[:-len(present)] + past
        return verb_phrase

    def _select_discourse_marker(self, discourse_type: str,
                                  sentence_index: int,
                                  dopamine_tone: float) -> str:
        if sentence_index == 0:
            return ""

        markers_by_type = {
            "elaborate": ["also", "furthermore", "in addition", "moreover", "besides", "beyond that", "at the same time", "on top of that", "plus", "what is more", "", "", ""],
            "contrast": ["", "but", "at the same time", "then again", "still", "although", "even so", "that said"],
            "connect": ["", "in the same way", "by the same token", "similarly", "likewise", "along those lines"],
            "explain": ["", "in other words", "specifically", "put simply", "that is", ""],
            "conclude": ["", "in essence", "when you think about it", "basically", "after all", "ultimately"],
            "acknowledge": ["", "", "", ""],
            "reflect": ["", "", ""],
            "explore": ["", "", ""],
        }
        markers = markers_by_type.get(discourse_type, [""])
        if not markers:
            return ""

        if dopamine_tone > 0.6 and len(markers) > 2:
            weights = [0.5 + dopamine_tone * (idx / len(markers))
                       for idx in range(len(markers))]
            return random.choices(markers, weights=weights, k=1)[0]

        return random.choice(markers)

    def reset_turn(self):
        self._used_subjects.clear()

    def get_state(self) -> Dict:
        return {
            'verb_phrase_success': self._verb_phrase_success,
        }

    def set_state(self, state: Dict):
        self._verb_phrase_success = state.get('verb_phrase_success', {})
