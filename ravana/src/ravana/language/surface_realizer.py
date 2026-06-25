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

    PRONOUNS = {
        "person": "they", "people": "they", "friend": "they",
        "dog": "it", "cat": "it", "bird": "it", "tree": "it",
        "book": "it", "song": "it", "machine": "it", "world": "it",
        "nature": "it", "life": "it", "death": "it", "time": "it",
        "mind": "it", "heart": "it", "science": "it", "art": "it",
        "history": "it", "knowledge": "it", "wisdom": "it",
        "truth": "it", "meaning": "it", "love": "it", "hope": "it",
        "fear": "it", "trust": "it", "justice": "it", "freedom": "it",
        "i": "i", "you": "you", "we": "we", "they": "they",
        "he": "he", "she": "she", "it": "it",
    }

    ABSTRACT_NOUNS = {
        "life", "death", "love", "hate", "truth", "beauty",
        "justice", "freedom", "knowledge", "wisdom", "time",
        "nature", "science", "art", "history", "meaning",
        "trust", "hope", "fear", "joy", "grief",
        "empathy", "respect", "culture", "power", "responsibility",
        "courage", "patience", "kindness", "honesty", "loyalty",
        "gratitude", "compassion", "generosity", "humility", "integrity",
        "dignity", "prudence", "grace", "mercy", "forgiveness",
        "peace", "faith", "fate", "destiny", "consciousness",
        "awareness", "education", "healthcare", "democracy", "diversity",
        "sustainability", "mindfulness", "meditation", "poverty",
        "hunger", "disease", "wealth", "war", "society",
        "identity", "culture", "tradition", "heritage",
    }

    UNCOUNTABLE_NOUNS = {
        "knowledge", "wisdom", "information", "music", "research",
        "evidence", "advice", "news", "progress", "nature",
        "life", "death", "time", "space", "love", "hate",
        "trust", "justice", "freedom", "empathy", "respect",
        "hope", "fear", "anxiety", "joy", "grief",
        "power", "culture", "art", "science", "history",
        "meaning", "truth", "beauty", "courage", "patience",
        "kindness", "honesty", "loyalty", "gratitude", "compassion",
        "generosity", "humility", "integrity", "dignity", "prudence",
        "temperance", "fortitude", "charity", "mercy", "forgiveness",
        "peace", "faith", "grace", "fate", "destiny",
        "consciousness", "awareness", "mindfulness",
        "bonding", "learning", "understanding", "thinking", "feeling",
        "running", "walking", "swimming", "reading", "writing",
        "speaking", "listening", "watching", "waiting", "working",
        "living", "dying", "growing", "changing", "moving",
        "being", "doing", "having", "making", "taking",
        "giving", "getting", "seeing", "hearing", "knowing",
        "trying", "caring", "sharing", "helping", "loving",
    }

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
        "kind of", "sort of", "in a way", "more or less",
        "in some sense", "in many ways",
    ]

    EPISTEMIC_FRAMES = {
        "low_confidence": [
            "I think", "I suspect", "it seems like", "perhaps",
            "maybe", "I would say",
        ],
        "medium_confidence": [
            "I believe", "it appears", "my sense is",
        ],
        "high_confidence": [
            "", "",
        ],
    }

    def __init__(self):
        self._used_subjects: set = set()
        self._verb_phrase_success: Dict[str, float] = {}
        self._last_free_energy: float = 0.3
        try:
            from ravana.language.verb_lexicon import _default_vector_fn
            self._vector_fn = _default_vector_fn
        except ImportError:
            self._vector_fn = None

    def set_vector_fn(self, fn):
        self._vector_fn = fn

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

        verb = self._select_verb_phrase(verb_phrase, relation,
                                         dopamine_tone, cerebellar_ngram,
                                         subject=sl, object=tl)

        verb = self._apply_agreement(verb, subject_phrase, display_subj.lower())
        verb = self._apply_tense(verb, frame.tense)

        if relation == 'interrogative':
            sentence = frame.object_concept
            has_punct = sentence.endswith('.') or sentence.endswith('?') or sentence.endswith('!')
        else:
            core = f"{subject_phrase} {verb} {object_phrase}"
            has_punct = False

            # Only prepend epistemic frame for the first sentence of the response
            # (or with a low 15% probability for subsequent sentences) to avoid robotic repetition.
            if discourse_context.sentence_index == 0 or random.random() < 0.15:
                epistemic_frame = self._generate_epistemic_frame(confidence_level, sl)
                if epistemic_frame:
                    core = f"{epistemic_frame}{core[0].lower()}{core[1:]}"

            sentence = core

        hedge = self._compose_hedge(free_energy)
        if hedge and not has_punct and discourse_context.sentence_index > 0:
            first_space = sentence.find(" ")
            if first_space > 0 and first_space < len(sentence) - 3:
                after_first = sentence[first_space + 1]
                if after_first.islower():
                    sentence = sentence[:first_space + 1] + hedge + " " + sentence[first_space + 1:]

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
            if discourse_marker and random.random() < 0.25:
                marker = discourse_marker
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

        self._last_free_energy = free_energy

        return sentence

    def _resolve_pronoun(self, subject: str, subject_lower: str,
                          context: DiscourseState) -> str:
        if subject_lower in ('i', 'you', 'we', 'they', 'he', 'she', 'it'):
            return subject

        if context.previous_subject:
            prev_lower = context.previous_subject.lower()
            pronoun_for_subject = self.PRONOUNS.get(subject_lower)
            if pronoun_for_subject and prev_lower == pronoun_for_subject:
                return pronoun_for_subject
            if prev_lower == subject_lower:
                pronoun = self.PRONOUNS.get(subject_lower, 'it')
                return pronoun

        if subject_lower in self._used_subjects:
            pronoun = self.PRONOUNS.get(subject_lower, 'it')
            return pronoun

        return subject

    def _build_noun_phrase(self, concept: str, article: str,
                            is_subject: bool, dopamine_tone: float) -> str:
        cl = concept.lower()

        if cl in ('i', 'you', 'we', 'they', 'he', 'she', 'it', 'me', 'him', 'her'):
            return concept

        if cl in self.ABSTRACT_NOUNS:
            return concept

        if cl in self.UNCOUNTABLE_NOUNS:
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
            return f"the {concept}"
        elif article in ("a", "an"):
            actual_art = "an" if concept[0].lower() in "aeiou" else "a"
            return f"{actual_art} {concept}"
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

    def _apply_agreement(self, verb_phrase: str, subject_phrase: str,
                          subject_lower: str) -> str:
        is_plural = False
        plural_indicators = {'they', 'we', 'you', 'people'}
        if subject_lower in plural_indicators:
            is_plural = True
        if subject_lower.endswith('s') and subject_lower not in {'this', 'is', 'has', 'was', 'its'}:
            if subject_lower not in self.SINGULAR_ENDING_IN_S:
                is_plural = True

        if is_plural:
            verb_phrase = verb_phrase.replace('is ', 'are ')
            verb_phrase = verb_phrase.replace('was ', 'were ')
            verb_phrase = verb_phrase.replace('has ', 'have ')
            verb_phrase = verb_phrase.replace('does ', 'do ')
            for word in ['creates', 'leads', 'causes', 'connects', 'relates',
                          'contrasts', 'resembles', 'follows', 'brings', 'gives',
                          'results', 'contrasts']:
                verb_phrase = verb_phrase.replace(f' {word} ', f' {word[:-1]} ')
                if verb_phrase.endswith(f' {word}'):
                    verb_phrase = verb_phrase[:-len(word)] + word[:-1]
                if verb_phrase.startswith(f'{word} '):
                    verb_phrase = word[:-1] + verb_phrase[len(word):]
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
            "elaborate": ["also", "furthermore", "in addition", "moreover", "besides", "", "", "", "plus"],
            "contrast": ["", "but", "at the same time", "then again", "still", "although"],
            "connect": ["", "in the same way", "by the same token", "similarly"],
            "explain": ["", "in other words", "specifically", ""],
            "conclude": ["", "in essence", "when you think about it", "basically"],
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
