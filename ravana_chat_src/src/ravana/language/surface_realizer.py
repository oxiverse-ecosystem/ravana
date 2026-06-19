"""
RAVANA Surface Realizer — Morphology, Articles, Tense, Pronoun Resolution
==========================================================================
Converts syntactic frames into well-formed English sentences using
rule-governed (not template) production. Rules are weighted by the
cerebellar n-gram model and dopamine tone modulates variety.

Grammar Rules (seeded, not learned):
- subject_verb_agreement: singular vs plural verb forms
- article_insertion: a/an/the vs none for abstract/countable nouns
- pronoun_substitution: replace repeated subjects with pronouns
- tense: present/past verb forms

Verb phrase selection: cerebellar-weighted (not random), with dopamine tone
modulating exploration of less-used phrases.

Replaces _format_sentence's random template approach.
"""

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
    subject_repetitions: int = 0  # how many times subject has been used


# Forward reference — SyntacticFrame is defined in the assembly module
# We'll import it at the wiring site


class SurfaceRealizer:
    """Rule-governed system that produces English sentences from syntactic frames.

    Rules are NOT hardcoded arbitrary choices — they're weighted by the
    cerebellar n-gram model when available.

    Key design decisions:
    - Grammar rules are rule-based (not learned), providing baseline competence
    - Verb phrase selection is cerebellar-weighted (not random)
    - Pronoun resolution prevents repetition
    - Dopamine tone modulates variety: high DA -> more variety, low DA -> conservative
    - Article insertion follows English rules (a/an/the, abstract noun exceptions)
    """

    # Grammatical agreement rules
    AGREEMENT_RULES = {
        "singular_verbs": {"is", "was", "has", "does", "creates", "leads", "causes",
                           "connects", "relates", "contrasts", "resembles", "follows"},
        "plural_verbs": {"are", "were", "have", "do", "create", "lead", "cause",
                         "connect", "relate", "contrast", "resemble", "follow"},
    }

    # Tense transformation map
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

    # Pronouns for substitution (subject form)
    PRONOUNS = {
        # Common nouns -> pronoun
        "person": "they", "people": "they", "friend": "they",
        "dog": "it", "cat": "it", "bird": "it", "tree": "it",
        "book": "it", "song": "it", "machine": "it", "world": "it",
        "nature": "it", "life": "it", "death": "it", "time": "it",
        "mind": "it", "heart": "it", "science": "it", "art": "it",
        "history": "it", "knowledge": "it", "wisdom": "it",
        "truth": "it", "meaning": "it", "love": "it", "hope": "it",
        "fear": "it", "trust": "it", "justice": "it", "freedom": "it",
        # Pronouns -> themselves
        "i": "i", "you": "you", "we": "we", "they": "they",
        "he": "he", "she": "she", "it": "it",
    }

    # Abstract nouns that don't take articles
    ABSTRACT_NOUNS = {
        "life", "death", "love", "hate", "truth", "beauty",
        "justice", "freedom", "knowledge", "wisdom", "time",
        "nature", "science", "art", "history", "meaning",
        "trust", "hope", "fear", "joy", "grief",
        "empathy", "respect", "culture", "power", "responsibility",
    }

    # Uncountable nouns (no "a"/"an")
    UNCOUNTABLE_NOUNS = {
        "knowledge", "wisdom", "information", "music", "research",
        "evidence", "advice", "news", "progress", "nature",
        "life", "death", "time", "space", "love", "hate",
        "trust", "justice", "freedom", "empathy", "respect",
        "hope", "fear", "anxiety", "joy", "grief",
        "power", "culture", "art", "science", "history",
    }

    # Singular nouns ending in 's' (fields of study, diseases, etc.)
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

    # Discourse markers (mapped from intent type)
    DISCOURSE_MARKERS = {
        "elaborate": ["furthermore", "in addition", "also", "moreover", "besides"],
        "contrast": ["however", "on the other hand", "yet", "but", "at the same time"],
        "connect": ["similarly", "likewise", "in the same way", "correspondingly"],
        "explain": ["", "", "in fact", "indeed"],
        "conclude": ["ultimately", "in essence", "at its core", "fundamentally"],
    }

    def __init__(self):
        # Track which subjects we've already used (for pronoun resolution)
        self._used_subjects: set = set()
        # Track verb phrase success rates (learned from feedback)
        self._verb_phrase_success: Dict[str, float] = {}

    def realize(self, frame, discourse_context: DiscourseState,
                dopamine_tone: float = 0.5,
                cerebellar_ngram=None,
                discourse_marker: Optional[str] = None) -> str:
        """Convert a syntactic frame into a well-formed English sentence.

        Pipeline:
        1. Determine subject-verb agreement from POS tags + countability
        2. Insert articles where needed (a/an/the)
        3. Choose best verb phrase (cerebellar-weighted, not random)
        4. Handle pronoun resolution (don't repeat subject)
        5. Apply discourse marker (if not first sentence)
        6. Capitalize and punctuate

        Args:
            frame: SyntacticFrame (from SyntacticCellAssembly.bind_to_sentence)
            discourse_context: Context for discourse-level decisions
            dopamine_tone: 0-1, higher = more variety, lower = conservative
            cerebellar_ngram: Optional CerebellarNgram instance for weighted phrases

        Returns:
            Well-formed English sentence string
        """
        import random
        subj = frame.subject_concept
        obj = frame.object_concept
        verb_phrase = frame.verb_phrase
        art_subj = frame.article_subject
        art_obj = frame.article_object
        relation = frame.relation_type
        sl, tl = subj.lower(), obj.lower()

        # Step 1: Pronoun resolution
        display_subj = self._resolve_pronoun(subj, sl, discourse_context)
        if display_subj != subj:
            art_subj = ""  # no article before pronoun

        # Step 2: Article insertion for subject
        subject_phrase = self._build_noun_phrase(
            display_subj, art_subj, is_subject=True, dopamine_tone=dopamine_tone
        )

        # Step 3: Article insertion for object
        object_phrase = self._build_noun_phrase(
            obj, art_obj, is_subject=False, dopamine_tone=dopamine_tone
        )

        # Step 4: Get verb phrase (cerebellar-weighted)
        verb = self._select_verb_phrase(verb_phrase, relation,
                                         dopamine_tone, cerebellar_ngram)

        # Step 5: Apply subject-verb agreement
        verb = self._apply_agreement(verb, subject_phrase, display_subj.lower())

        # Step 6: Apply tense
        verb = self._apply_tense(verb, frame.tense)

        # Step 7: Assemble core sentence
        if relation == 'causal':
            sentence = f"{subject_phrase} {verb} {object_phrase}"
        elif relation == 'contrastive':
            sentence = f"{subject_phrase} {verb} {object_phrase}"
        else:
            sentence = f"{subject_phrase} {verb} {object_phrase}"

        # Step 8: Add discourse marker
        if discourse_marker:
            marker = discourse_marker
        else:
            marker = self._select_discourse_marker(
                discourse_context.discourse_type,
                discourse_context.sentence_index,
                dopamine_tone
            )
        if marker:
            sentence = f"{marker}, {sentence[0].lower() + sentence[1:]}"

        # Step 9: Capitalize and punctuate
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith('.'):
            sentence += '.'

        # Track subject usage
        self._used_subjects.add(sl)
        if discourse_context.sentence_index == 0:
            self._used_subjects.clear()
            self._used_subjects.add(sl)

        return sentence

    def _resolve_pronoun(self, subject: str, subject_lower: str,
                          context: DiscourseState) -> str:
        """Replace subject with pronoun if it was used recently."""
        if subject_lower in ('i', 'you', 'we', 'they', 'he', 'she', 'it'):
            return subject  # Already a pronoun

        # Check if this subject was used in the PREVIOUS sentence
        if context.previous_subject:
            prev_lower = context.previous_subject.lower()
            # If previous subject was a pronoun, check if it maps to this subject
            pronoun_for_subject = self.PRONOUNS.get(subject_lower)
            if pronoun_for_subject and prev_lower == pronoun_for_subject:
                return pronoun_for_subject
            # Also check direct match (for explicit repetition)
            if prev_lower == subject_lower:
                pronoun = self.PRONOUNS.get(subject_lower, 'it')
                return pronoun

        # Check if this subject was used 2+ times total (not just previous)
        if subject_lower in self._used_subjects:
            pronoun = self.PRONOUNS.get(subject_lower, 'it')
            return pronoun

        return subject

    def _build_noun_phrase(self, concept: str, article: str,
                            is_subject: bool, dopamine_tone: float) -> str:
        """Build a noun phrase with proper article insertion."""
        cl = concept.lower()

        # No article for pronouns
        if cl in ('i', 'you', 'we', 'they', 'he', 'she', 'it', 'me', 'him', 'her'):
            return concept

        # No article for abstract nouns
        if cl in self.ABSTRACT_NOUNS:
            return concept

        # No article for uncountable
        if cl in self.UNCOUNTABLE_NOUNS:
            return concept

        # Apply article
        if article == "the":
            return f"the {concept}"
        elif article in ("a", "an"):
            actual_art = "an" if concept[0].lower() in "aeiou" else "a"
            return f"{actual_art} {concept}"
        else:
            return concept

    def _select_verb_phrase(self, default_phrase: str, relation: str,
                             dopamine_tone: float,
                             cerebellar_ngram) -> str:
        """Select verb phrase with cerebellar-weighted selection."""
        import random

        # If we have cerebellar n-gram, try to use it for phrase selection
        if cerebellar_ngram is not None:
            # Check if ngram has a preferred phrase for this relation
            ngram_key = f"phrase:{relation}"
            ngram_result = cerebellar_ngram.predict_next(
                ngram_key, top_k=3
            )
            if ngram_result:
                # Use the top ngram prediction if dopamine is low (conservative)
                if dopamine_tone < 0.4:
                    return list(ngram_result.keys())[0]

        # Default: return the frame's verb phrase
        # Dopamine modulation: high DA = random exploration
        if dopamine_tone > 0.7 and random.random() < (dopamine_tone - 0.5):
            # Try alternate phrasing
            from ravana.language.syntactic_cell_assembly import SyntacticCellAssembly
            phrases = SyntacticCellAssembly.VERB_PHRASES.get(relation,
                        SyntacticCellAssembly.VERB_PHRASES['semantic'])
            return random.choice(phrases)

        return default_phrase

    def _apply_agreement(self, verb_phrase: str, subject_phrase: str,
                          subject_lower: str) -> str:
        """Apply subject-verb agreement.

        Plural subjects -> plural verb forms
        Singular subjects (default) -> singular verb forms
        """
        # Check if subject is plural
        is_plural = False
        plural_indicators = {'they', 'we', 'you', 'people'}
        if subject_lower in plural_indicators:
            is_plural = True
        # Words ending in 's' that are not pronouns
        if subject_lower.endswith('s') and subject_lower not in {'this', 'is', 'has', 'was', 'its'}:
            # Check if it's a known plural-form concept (actually singular)
            if subject_lower not in self.SINGULAR_ENDING_IN_S:
                is_plural = True

        if is_plural:
            # Replace singular verbs with plural
            verb_phrase = verb_phrase.replace('is ', 'are ')
            verb_phrase = verb_phrase.replace('was ', 'were ')
            verb_phrase = verb_phrase.replace('has ', 'have ')
            verb_phrase = verb_phrase.replace('does ', 'do ')
            # Remove trailing 's' from verbs
            for word in ['creates', 'leads', 'causes', 'connects', 'relates',
                          'contrasts', 'resembles', 'follows', 'brings', 'gives',
                          'results', 'contrasts']:
                verb_phrase = verb_phrase.replace(f' {word} ', f' {word[:-1]} ')
                if verb_phrase.endswith(f' {word}'):
                    verb_phrase = verb_phrase[:-len(word)] + word[:-1]
                if verb_phrase.startswith(f'{word} '):
                    verb_phrase = word[:-1] + verb_phrase[len(word):]
        else:
            # Singular: ensure "are" becomes "is"
            # Handle "are" at start, middle, or end
            if verb_phrase.startswith('are '):
                verb_phrase = 'is ' + verb_phrase[4:]
            elif ' are ' in verb_phrase:
                verb_phrase = verb_phrase.replace(' are ', ' is ')
            elif verb_phrase.endswith(' are'):
                verb_phrase = verb_phrase[:-4] + ' is'

        return verb_phrase

    def _apply_tense(self, verb_phrase: str, tense: str) -> str:
        """Apply tense to verb phrase."""
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
        """Select a discourse marker for the current sentence."""
        import random
        if sentence_index == 0:
            return ""

        markers = self.DISCOURSE_MARKERS.get(discourse_type, [""])
        if not markers:
            return ""

        # Dopamine modulation: high = pick less common markers
        if dopamine_tone > 0.6 and len(markers) > 2:
            # Weight toward less common markers
            weights = [0.5 + dopamine_tone * (idx / len(markers))
                       for idx in range(len(markers))]
            return random.choices(markers, weights=weights, k=1)[0]

        return random.choice(markers)

    def reset_turn(self):
        """Reset per-turn state (used_subjects)."""
        self._used_subjects.clear()

    def get_state(self) -> Dict:
        return {
            'used_subjects': list(self._used_subjects),
            'verb_phrase_success': self._verb_phrase_success,
        }