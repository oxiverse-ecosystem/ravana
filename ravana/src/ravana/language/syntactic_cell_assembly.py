"""
RAVANA Syntactic Cell Assembly — Pulvermüller's Neural Syntax (2010)
====================================================================
Learns which concepts fill which grammatical roles through Hebbian
co-activation. Each assembly is a distributed circuit:

    [Subject_role, concept_A] → [Verb_role, concept_B] → [Object_role, concept_C]

Co-activation strengthens the whole circuit.

KEY DESIGN DECISIONS:
- Seeded with baseline English rules (Noun→Verb→Noun high initial weight)
- Hebbian plasticity refines and overrides based on actual usage
- Role matrices: subject_role, verb_role, object_role (concept → weight)
- Sequence patterns for sequential binding P(role | previous_role, concept)
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import re


@dataclass
class SyntacticFrame:
    """A grammatical frame from binding concepts to roles."""
    subject_concept: str = ""
    verb_concept: str = ""
    object_concept: str = ""
    relation_type: str = "semantic"
    subject_pos: str = "n"       # POS tag of subject
    object_pos: str = "n"        # POS tag of object
    verb_phrase: str = ""        # The actual verb phrase chosen
    article_subject: str = ""    # "the", "a", "an", or ""
    article_object: str = ""     # "the", "a", "an", or ""
    pronoun_subject: str = ""    # "it", "they", "he", "she", or ""
    is_contrastive: bool = False
    is_causal: bool = False
    tense: str = "present"       # "present" or "past"
    depth: int = 0               # How many hops in the chain


class SyntacticCellAssembly:
    """Learns which concepts fill which grammatical roles through Hebbian co-activation.

    Seeded with baseline English syntactic priors:
    - Nouns (n) → high subject_role and object_role weight
    - Verbs (v) → high verb_role weight
    - Pronouns (pron) → low verb_role weight
    - Adjectives (adj) → low subject/verb weight (likely modifiers)

    Hebbian plasticity then refines these priors based on actual usage.
    """

    # Baseline syntactic priors — seeded from English grammar rules
    # These prevent the "caveman cold start" by giving initial structure
    POS_ROLE_PRIORS = {
        'n':     {'subject_role': 0.7, 'verb_role': 0.05, 'object_role': 0.7},
        'v':     {'subject_role': 0.05, 'verb_role': 0.8, 'object_role': 0.1},
        'adj':   {'subject_role': 0.15, 'verb_role': 0.0, 'object_role': 0.15},
        'adv':   {'subject_role': 0.05, 'verb_role': 0.1, 'object_role': 0.05},
        'pron':  {'subject_role': 0.6, 'verb_role': 0.0, 'object_role': 0.3},
        'interj':{'subject_role': 0.0, 'verb_role': 0.0, 'object_role': 0.0},
        'det':   {'subject_role': 0.0, 'verb_role': 0.0, 'object_role': 0.0},
        'conj':  {'subject_role': 0.0, 'verb_role': 0.0, 'object_role': 0.0},
        'prep':  {'subject_role': 0.0, 'verb_role': 0.0, 'object_role': 0.0},
        'num':   {'subject_role': 0.3, 'verb_role': 0.0, 'object_role': 0.3},
    }

    # Baseline sequence patterning: P(next_role | previous_role, relation_type)
    # Noun → Verb → Noun is high initial weight (prevents caveman speech)
    SEQUENCE_PRIORS = {
        # Normal semantic: subject → verb → object
        ('subject_role', 'semantic'):  {'verb_role': 0.85, 'object_role': 0.05, 'subject_role': 0.05},
        ('verb_role', 'semantic'):     {'object_role': 0.7, 'subject_role': 0.1, 'verb_role': 0.1},
        ('object_role', 'semantic'):   {'verb_role': 0.5, 'subject_role': 0.1, 'object_role': 0.1},
        # Causal: subject → cause-verb → effect
        ('subject_role', 'causal'):    {'verb_role': 0.9, 'object_role': 0.02, 'subject_role': 0.02},
        ('verb_role', 'causal'):       {'object_role': 0.8, 'subject_role': 0.05, 'verb_role': 0.05},
        # Contrastive: subject → contrast-verb → opposite
        ('subject_role', 'contrastive'): {'verb_role': 0.85, 'object_role': 0.05, 'subject_role': 0.05},
        ('verb_role', 'contrastive'):    {'object_role': 0.7, 'subject_role': 0.1, 'verb_role': 0.1},
    }

    # Verb phrases per relation type — delegated to VerbLexicon (Phase 6)
    # The VerbLexicon provides semantically-driven verb selection based on
    # Levelt's lemma retrieval model (1989). Kept as a class variable for
    # backward compatibility with tests, but actual selection uses VerbLexicon.
    # Phase 6: VERB_PHRASES is a deprecated alias; use VerbLexicon instead.
    VERB_PHRASES = {
        'semantic': [
            'ties into', 'is part of', 'plays a role in',
            'feeds into', 'goes hand in hand with', 'is bound up with',
            'is deeply connected with', 'is tied to',
            'has a relationship with', 'has a lot to do with',
        ],
        'causal': [
            'leads to', 'creates', 'causes', 'brings about',
            'influences', 'gives rise to', 'results in',
            'sparks', 'triggers', 'fuels', 'contributes to',
            'drives', 'prompts',
        ],
        'contrastive': [
            'contrasts with', 'differs from', 'stands against',
            'challenges', 'is the opposite of',
            'clashes with', 'pulls against', 'runs counter to',
            'is at odds with', 'diverges from', 'pushes back against',
        ],
        'analogical': [
            'is like', 'resembles', 'mirrors', 'echoes', 'is similar to',
            'can be compared to', 'is akin to', 'parallels',
            'reflects', 'brings to mind', 'reminds us of',
        ],
        'temporal': [
            'comes before', 'follows', 'leads into', 'precedes',
            'happens before', 'occurs after',
            'ushers in', 'paves the way for', 'sets the stage for',
            'traces back to',
        ],
        'episodic': [
            'brings up', 'recalls', 'reminds us of',
            'is linked with', 'ties into', 'feeds into',
        ],
    }

    # Discourse type mapping — maps edge relation types to surface discourse types
    # Used by the SurfaceRealizer to pick natural clause templates.
    RELATION_TO_DISCOURSE = {
        'semantic': 'explain',
        'causal': 'causal',
        'contrastive': 'contrast',
        'analogical': 'connect',
        'temporal': 'elaborate',
        'episodic': 'elaborate',
    }

    def __init__(self, learning_rate: float = 0.05):
        self.learning_rate = learning_rate

        # Role activation matrices: concept → weight for each role
        # Seeded from POS priors when seed_from_pos is called
        self.subject_role: Dict[str, float] = {}
        self.verb_role: Dict[str, float] = {}
        self.object_role: Dict[str, float] = {}

        # Sequential binding patterns: (previous_role, relation_type) → {next_role → weight}
        self.sequence_patterns: Dict[Tuple[str, str], Dict[str, float]] = {}
        for (prev_role, rel_type), next_roles in self.SEQUENCE_PRIORS.items():
            self.sequence_patterns[(prev_role, rel_type)] = dict(next_roles)

        # Track verb phrase usage for cerebellar-weighted selection
        self.verb_phrase_counts: Dict[str, int] = {}
        from ravana.language.verb_lexicon import VerbLexicon
        for rel_type in VerbLexicon.VERB_PATTERNS:
            for phrase in VerbLexicon.get_phrases(rel_type):
                self.verb_phrase_counts[phrase] = 1  # base count
        # Also include legacy VERB_PHRASES for backward compatibility
        for rel_type, phrases in self.VERB_PHRASES.items():
            for phrase in phrases:
                if phrase not in self.verb_phrase_counts:
                    self.verb_phrase_counts[phrase] = 1

        # Countability tracking for article insertion
        # Seeded with common uncountable/abstract nouns
        # EXPANDED to cover more abstract concepts the graph might contain
        self._uncountable_nouns: Set[str] = {
            'knowledge', 'wisdom', 'information', 'music', 'research',
            'evidence', 'advice', 'news', 'progress', 'nature',
            'life', 'death', 'time', 'space', 'love', 'hate',
            'trust', 'justice', 'freedom', 'empathy', 'respect',
            'hope', 'fear', 'anxiety', 'joy', 'grief',
            'power', 'responsibility', 'culture', 'art',
            'science', 'history', 'meaning', 'truth', 'beauty',
            'courage', 'patience', 'kindness', 'honesty', 'loyalty',
            'gratitude', 'compassion', 'generosity', 'humility', 'integrity',
            'dignity', 'prudence', 'temperance', 'fortitude', 'charity',
            'wisdom', 'faith', 'grace', 'mercy', 'forgiveness',
            'peace', 'justice', 'equality', 'diversity', 'sustainability',
            'consciousness', 'awareness', 'attention', 'mindfulness',
            # Gerunds (-ing forms used as uncountable nouns)
            'bonding', 'learning', 'understanding', 'thinking', 'feeling',
            'running', 'walking', 'swimming', 'reading', 'writing',
            'speaking', 'listening', 'watching', 'waiting', 'working',
            'living', 'dying', 'growing', 'changing', 'moving',
            'being', 'doing', 'having', 'making', 'taking',
            'giving', 'getting', 'seeing', 'hearing', 'knowing',
            'trying', 'caring', 'sharing', 'helping', 'loving',
        }

        # Pronouns for substitution
        self._pronoun_map: Dict[str, str] = {
            'i': 'i', 'you': 'you', 'we': 'we', 'they': 'they',
            'he': 'he', 'she': 'she', 'it': 'it',
            'person': 'they', 'people': 'they', 'friend': 'they',
            'dog': 'it', 'cat': 'it', 'bird': 'it', 'tree': 'it',
            'book': 'it', 'song': 'it', 'machine': 'it',
        }

        # Abstract concepts that don't take articles
        # EXPANDED to cover graph concepts that shouldn't have "a"/"an"
        self._abstract_nouns: Set[str] = {
            'life', 'death', 'love', 'hate', 'truth', 'beauty',
            'justice', 'freedom', 'knowledge', 'wisdom', 'time',
            'nature', 'science', 'art', 'history', 'meaning',
            'trust', 'hope', 'faith', 'grace', 'luck', 'fate',
            'destiny', 'karma', 'dharma', 'nirvana', 'heaven', 'hell',
            'god', 'spirit', 'soul', 'consciousness', 'awareness',
            'peace', 'war', 'wealth', 'poverty', 'hunger', 'disease',
            'education', 'healthcare', 'democracy', 'tyranny',
            'courage', 'patience', 'kindness', 'honesty', 'gratitude',
            'compassion', 'generosity', 'humility', 'integrity',
            'dignity', 'prudence', 'temperance', 'fortitude', 'charity',
            'mercy', 'forgiveness', 'equality', 'diversity',
            'sustainability', 'mindfulness', 'meditation',
        }

    def seed_from_pos(self, concept_pos: Dict[str, str]):
        """Seed role matrices from POS tags — prevents caveman cold start.

        Each concept gets initial role weights based on its POS tag.
        Hebbian plasticity will refine these over time.
        """
        # Map full POS names to short codes used by POS_ROLE_PRIORS
        pos_map = {
            'noun': 'n', 'verb': 'v', 'adj': 'adj', 'adverb': 'adv',
            'pron': 'pron', 'interj': 'interj', 'conj': 'conj', 'prep': 'prep', 'det': 'det', 'num': 'num'
        }
        for concept, pos in concept_pos.items():
            short_pos = pos_map.get(pos, pos)  # fallback to original if not in map
            priors = self.POS_ROLE_PRIORS.get(short_pos, {})
            cl = concept.lower()
            if priors.get('subject_role', 0) > 0:
                self.subject_role[cl] = priors['subject_role']
            if priors.get('verb_role', 0) > 0:
                self.verb_role[cl] = priors['verb_role']
            if priors.get('object_role', 0) > 0:
                self.object_role[cl] = priors['object_role']

    # ─── Binding ───

    def bind_to_sentence(self, subject: str, relation: str,
                         target: str, pos_map: Dict[str, str],
                         chain_concepts: Optional[List[str]] = None,
                         chain_connectors: Optional[List[str]] = None,
                         depth: int = 0) -> SyntacticFrame:
        """Build a grammatical frame from a (subject, relation, target) triple.

        Uses Hebbian-weighted role assignments:
        - If "trust" has high subject_role weight → subject role
        - If "causes" has high verb_role weight → verb/relation role
        - If "knowledge" has high object_role weight → object role

        Args:
            subject: The main topic concept
            relation: The edge relation type (semantic, causal, contrastive, etc.)
            target: The object/target concept
            pos_map: Concept → POS tag mapping
            chain_concepts: Additional concepts from the chain (for multi-hop)
            chain_connectors: Connector words from the chain
            depth: Number of hops in the chain

        Returns:
            SyntacticFrame with roles assigned and verb phrase chosen
        """
        sl, tl = subject.lower(), target.lower()

        # Determine roles using Hebbian-weighted assignment
        subj_role_weight = self.subject_role.get(sl, 0.5)
        obj_role_weight = self.object_role.get(tl, 0.5)

        # Pick the best verb concept for this relation
        verb_concept = self._pick_verb_for_relation(relation, sl, tl, pos_map)

        # Pick the verb phrase (cerebellar-weighted, not random)
        verb_phrase = self._pick_verb_phrase(relation)

        # Determine article for subject
        subj_pos = pos_map.get(sl, 'noun')
        obj_pos = pos_map.get(tl, 'noun')
        # Convert to short POS codes for article determination
        pos_short = {'noun': 'n', 'verb': 'v', 'adj': 'adj', 'adverb': 'adv',
                     'pron': 'pron', 'interj': 'interj', 'conj': 'conj', 'prep': 'prep', 'det': 'det', 'num': 'num'}
        subj_pos_short = pos_short.get(subj_pos, subj_pos)
        obj_pos_short = pos_short.get(obj_pos, obj_pos)
        art_subj = self._determine_article(subject, subj_pos_short, is_subject=True)
        art_obj = self._determine_article(target, obj_pos_short, is_subject=False)

        # Pronoun for subject (used in subsequent sentences)
        pronoun = self._pronoun_map.get(sl, '')

        # Tense
        tense = self._determine_tense(relation)

        return SyntacticFrame(
            subject_concept=subject,
            verb_concept=verb_concept,
            object_concept=target,
            relation_type=relation,
            subject_pos=subj_pos,
            object_pos=obj_pos,
            verb_phrase=verb_phrase,
            article_subject=art_subj,
            article_object=art_obj,
            pronoun_subject=pronoun,
            is_contrastive=(relation == 'contrastive'),
            is_causal=(relation == 'causal'),
            tense=tense,
            depth=depth,
        )

    def _pick_verb_for_relation(self, relation: str,
                                 subject: str, target: str,
                                 pos_map: Dict[str, str]) -> str:
        """Pick the best verb concept for this relation type.

        Uses Hebbian-weighted verb_role assignments:
        - Prefer concepts with high verb_role weight
        - Prefer verbs that match the relation type
        - Fallback: the relation name itself ("leads to" for causal)
        """
        # Find concepts with high verb_role weight that are verbs
        best_verb = ""
        best_weight = 0.0
        for concept, weight in self.verb_role.items():
            pos = pos_map.get(concept, '')
            if pos == 'v' and weight > best_weight and len(concept) >= 3:
                best_verb = concept
                best_weight = weight

        if not best_verb:
            # Map relation types to default verb concepts
            rel_to_verb = {
                'causal': 'cause', 'semantic': 'shape',
                'contrastive': 'contrast', 'analogical': 'resemble',
                'temporal': 'follow', 'episodic': 'tie',
            }
            best_verb = rel_to_verb.get(relation, 'shape')

        return best_verb

    def _pick_verb_phrase(self, relation: str) -> str:
        """Pick a verb phrase for this relation type.

        Phase 6: Delegates to VerbLexicon (semantic-vector-driven selection).
        Keeps verb_phrase_counts for backward compatibility.

        Uses cerebellar-weighted selection when available.
        Higher dopamine → more variety (exploration).
        Lower dopamine → exploits best-matching phrase.
        """
        import random
        from ravana.language.verb_lexicon import VerbLexicon

        # Use VerbLexicon for semantic-vector-driven selection
        # Without vector_fn, it falls back to complexity-similarity algorithm
        return VerbLexicon.select_verb(
            relation=relation,
            subject="",
            object="",
            dopamine_tone=0.5,
            vector_fn=None,
        )

    def _determine_article(self, concept: str, pos: str,
                            is_subject: bool) -> str:
        """Determine the appropriate article for a concept.

        Rules:
        - Abstract nouns → no article
        - Uncountable nouns → no article
        - Proper nouns → no article (capitalized already)
        - Pronouns → no article
        - Adjectives and verbs → no article (don't function as NPs alone)
        - Countable singular → "a"/"an" or "the"
        - First mention: "a/an", subsequent: "the"
        - Web-garbage concepts (no GloVe vector) → no article
        - Concepts ending with common adjective suffixes → no article
        """
        cl = concept.lower()
        # Non-noun POS → no article
        if pos in ('pron', 'interj', 'conj', 'prep', 'det', 'adj', 'verb', 'v'):
            return ""
        # Abstract nouns → no article (check FIRST to catch words like 'gas', 'atlas')
        if cl in self._abstract_nouns:
            return ""
        # Uncountable nouns → no article
        if cl in self._uncountable_nouns:
            return ""
        # Short words → no article (likely abbreviations, garbage)
        if len(cl) <= 2:
            return ""
        # Plural-looking nouns that aren't known singular exceptions → no article
        # Check AFTER abstract/uncountable to avoid false positives on words like 'gas', 'grass', 'lens'
        if cl.endswith('s') and cl not in {
            'news', 'physics', 'mathematics', 'economics', 'politics', 'ethics',
            'linguistics', 'statistics', 'genetics', 'dynamics', 'kinematics',
            'acoustics', 'optics', 'mechanics', 'thermodynamics',
            'gas', 'atlas', 'campus', 'virus', 'bus', 'lens', 'bonus',
            'genius', 'cactus', 'alumnus', 'focus', 'corpus', 'status',
        }:
            return ""
        # Common adjective suffixes → likely not a noun, no article
        adj_suffixes = ('ous', 'ful', 'less', 'able', 'ible', 'ical', 'ive',
                        'like', 'ish', 'some', 'ward', 'fold', 'most')
        if cl.endswith(adj_suffixes):
            return ""
        # Web-garbage / non-English indicators: no article
        # These patterns suggest the word isn't a real English noun
        garbage_indicators = (
            cl.startswith('http'), cl.startswith('www'),
            cl.startswith('font'), cl.startswith('class'),
            cl.startswith('div'), cl.startswith('span'),
            cl.startswith('var'), cl.startswith('func'),
            cl.startswith('btn'), cl.startswith('img'),
            cl.startswith('href'), cl.startswith('src'),
            cl.startswith('data'), cl.startswith('meta'),
            'gform' in cl, 'https' in cl, 'http' in cl,
            'javascript' in cl, 'stylesheet' in cl,
        )
        if any(garbage_indicators):
            return ""
        # Subject gets "the" for specificity
        if is_subject:
            return "the"
        # For objects: use a/an based on vowel start
        if cl[0] in 'aeiou':
            return "an"
        return "a"

    def _determine_tense(self, relation: str) -> str:
        """Determine tense from relation type.
        Most relations are present tense (general truths).
        Temporal relations might be past.
        """
        return "present"  # Default present for general semantic truths

    # ─── Learning ───

    def learn_from_feedback(self, frame: SyntacticFrame,
                             user_understood: bool):
        """Hebbian update: reinforce the role assignments that worked.

        user_understood = True → strengthen all co-occurrences
        user_understood = False → weaken them
        """
        lr = self.learning_rate if user_understood else -self.learning_rate * 0.5
        sl = frame.subject_concept.lower()
        tl = frame.object_concept.lower()
        vl = frame.verb_concept.lower()

        # Strengthen/weaken role assignments
        if sl in self.subject_role:
            self.subject_role[sl] = max(0.0, min(1.0, self.subject_role[sl] + lr))
        if tl in self.object_role:
            self.object_role[tl] = max(0.0, min(1.0, self.object_role[tl] + lr))
        if vl in self.verb_role:
            self.verb_role[vl] = max(0.0, min(1.0, self.verb_role[vl] + lr))

        # Strengthen/weaken sequence patterns
        seq_key = ('subject_role', frame.relation_type)
        if seq_key in self.sequence_patterns:
            for next_role in ['verb_role', 'object_role', 'subject_role']:
                cur = self.sequence_patterns[seq_key].get(next_role, 0.0)
                boost = lr * 0.5 if next_role == 'verb_role' else -lr * 0.2
                self.sequence_patterns[seq_key][next_role] = max(0.0, min(1.0, cur + boost))

        # Track verb phrase usage
        if frame.verb_phrase and user_understood:
            self.verb_phrase_counts[frame.verb_phrase] = \
                self.verb_phrase_counts.get(frame.verb_phrase, 1) + 1

        # Track abstract/uncountable status
        if user_understood and frame.article_subject == '' and sl not in self._abstract_nouns:
            self._abstract_nouns.add(sl)

    # ─── Surface Helpers ───

    def compose_sentence(self, frame: SyntacticFrame,
                         discourse_marker: str = "",
                         use_pronoun: bool = False,
                         dopamine_tone: float = 0.5) -> str:
        """Compose a full English sentence from a syntactic frame.

        Applies:
        1. Article insertion
        2. Subject-verb agreement
        3. Verb phrase insertion
        4. Object article
        5. Pronoun substitution (optional)
        6. Discourse marker prefix
        7. Capitalization and punctuation

        Args:
            frame: The syntactic frame
            discourse_marker: Optional discourse marker ("furthermore", "however")
            use_pronoun: Whether to substitute the subject with a pronoun
            dopamine_tone: 0-1, modulates variety in phrasing

        Returns:
            A well-formed English sentence string
        """
        subj = frame.subject_concept
        obj = frame.object_concept
        verb_phrase = frame.verb_phrase
        art_subj = frame.article_subject
        art_obj = frame.article_object

        # Pronoun substitution
        if use_pronoun and frame.pronoun_subject:
            display_subj = frame.pronoun_subject
            art_subj = ""  # no article before pronoun
        else:
            display_subj = subj

        # Build subject phrase
        if art_subj and display_subj[0].islower():
            subject_phrase = f"{art_subj} {display_subj}"
        else:
            subject_phrase = display_subj

        # Build object phrase
        if art_obj:
            # Check vowel for a/an
            if art_obj in ('a', 'an'):
                actual_art = 'an' if obj[0].lower() in 'aeiou' else 'a'
                object_phrase = f"{actual_art} {obj}"
            else:
                object_phrase = f"{art_obj} {obj}"
        else:
            object_phrase = obj

        # Subject-verb agreement
        verb = self._apply_agreement(verb_phrase, subject_phrase, frame)

        # Tense application
        verb = self._apply_tense(verb, frame.tense)

        # Assemble sentence
        if frame.is_causal:
            sentence = f"{subject_phrase} {verb} {object_phrase}"
        elif frame.is_contrastive:
            sentence = f"{subject_phrase} {verb} {object_phrase}"
        else:
            sentence = f"{subject_phrase} {verb} {object_phrase}"

        # Discourse marker prefix
        if discourse_marker:
            sentence = f"{discourse_marker}, {sentence[0].lower() + sentence[1:]}"

        # Capitalize and punctuate
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith('.'):
            sentence += '.'

        return sentence

    def _apply_agreement(self, verb_phrase: str, subject_phrase: str,
                          frame: SyntacticFrame) -> str:
        """Apply subject-verb agreement.

        If the subject is plural or "they"/"we"/"you", use plural form.
        Default: singular (is, has, etc.)
        """
        sp = subject_phrase.lower()
        # Words that take plural agreement
        plural_subjects = {'they', 'we', 'you', 'people', 'i'}
        if sp in plural_subjects or sp.endswith('s'):
            # Replace "is" → "are", "was" → "were", "has" → "have"
            verb_phrase = verb_phrase.replace('is ', 'are ')
            verb_phrase = verb_phrase.replace('has ', 'have ')
            verb_phrase = verb_phrase.replace('does ', 'do ')
        return verb_phrase

    def _apply_tense(self, verb_phrase: str, tense: str) -> str:
        """Apply tense to the verb phrase.
        Currently only present tense is fully supported.
        """
        if tense == 'past':
            # Simple past: replace is→was, are→were
            verb_phrase = verb_phrase.replace('is ', 'was ')
            verb_phrase = verb_phrase.replace('are ', 'were ')
            verb_phrase = verb_phrase.replace('creates ', 'created ')
            verb_phrase = verb_phrase.replace('leads ', 'led ')
        return verb_phrase

    # ─── State ───

    def get_state(self) -> Dict:
        return {
            'subject_role': self.subject_role,
            'verb_role': self.verb_role,
            'object_role': self.object_role,
            'sequence_patterns': {str(k): v for k, v in self.sequence_patterns.items()},
            'verb_phrase_counts': self.verb_phrase_counts,
        }

    def set_state(self, state: Dict):
        self.subject_role = state.get('subject_role', {})
        self.verb_role = state.get('verb_role', {})
        self.object_role = state.get('object_role', {})
        seq_raw = state.get('sequence_patterns', {})
        self.sequence_patterns = {}
        for k_str, v in seq_raw.items():
            parts = k_str.strip("()").split(', ')
            if len(parts) == 2:
                self.sequence_patterns[(parts[0].strip("'"), parts[1].strip("'"))] = v
        self.verb_phrase_counts = state.get('verb_phrase_counts', {})
