"""
RAVANA Question Decomposition Engine
======================================
Inspired by how the human brain decomposes complex problems:

1. FRONTOPOLAR CORTEX (BA 10): Cognitive branching — holds the main question in
   "pending" state while managing sub-questions. The most anterior PFC region is
   uniquely engaged when a primary goal must be maintained while sub-goals are 
   actively processed (Braver & Bongiolatti, 2002; Koechlin & Hyafil, 2007).

2. PFC HIERARCHY (Rostro-Caudal axis): Anterior regions represent abstract goals
   (the overall question), posterior regions manage concrete sub-questions and
   answer steps (Badre & D'Esposito, 2009; Christoff & Gabrieli, 2000).

3. WORKING MEMORY CHUNKING: Sub-answers are compressed into "chunks" that reduce
   cognitive load, allowing deeper deliberation (Miller, 1956; Gobet et al., 2001).

4. SOAR-STYLE UNIVERSAL SUBGOALING: When a question cannot be answered directly,
   an "impasse" triggers recursive decomposition into sub-states (Newell, 1990).

5. HIPPOCAMPAL SCENE CONSTRUCTION: The hippocampus recombines memory fragments
   to simulate novel causal sequences and alternative scenarios (Schacter & Addis,
   2007; Buckner, 2010).

ARCHITECTURE:
    ┌──────────────────────────────────────────────────────────┐
    │  QuestionDecompositionEngine (BA 10 / rostral PFC)       │
    │                                                           │
    │  ┌──────────────────────────────────────────────────┐    │
    │  │  Decomposition Strategies:                       │    │
    │  │  • Definitional (what_is) → category + props     │    │
    │  │  • Causal (why) → cause + mechanism + effect     │    │
    │  │  • Procedural (how) → parts + roles + interact   │    │
    │  │  • Comparative → analyze_A + analyze_B + diff    │    │
    │  │  • Hypothetical → role + removal + cascade       │    │
    │  │  • Abstract (meaning/love/truth) → multi-persp   │    │
    │  └──────────────────────────────────────────────────┘    │
    │                                                           │
    │  ┌──────────────────────────────────────────────────┐    │
    │  │  Goal Stack (frontopolar pending set)            │    │
    │  │  Main: "what is quantum entanglement?"           │    │
    │  │  Sub:  ├─ "what is quantum in physics?"          │    │
    │  │  │      ├─ "what is entanglement?"               │    │
    │  │  │      └─ "how do they combine?"                │    │
    │  └──────────────────────────────────────────────────┘    │
    │                                                           │
    │  ┌──────────────────────────────────────────────────┐    │
    │  │  Answer Synthesis                                │    │
    │  │  Combines sub-answers into coherent narrative    │    │
    │  │  using discourse markers and logical flow        │    │
    │  └──────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────┘
"""

from __future__ import annotations
import re
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum


# ─── Question Types ───

class QuestionCategory(Enum):
    """High-level categories of questions the engine handles."""
    WHAT_IS = "what_is"               # Definitional: "what is gravity?"
    WHY = "why"                        # Causal: "why is the sky blue?"
    HOW = "how"                        # Procedural: "how does a computer work?"
    COMPARE = "compare"                # Comparative: "compare love and fear"
    HYPOTHETICAL = "hypothetical"      # Counterfactual: "what if the sun disappeared?"
    TELL_ME = "tell_me"                # Elaboration: "tell me about X"
    ANALOGY = "analogy"                # Analogical: "X is to Y as Z is to ?"
    ABSTRACT = "abstract"              # Abstract: "what is the meaning of life?"
    GENERAL = "general"                # General query / statement
    SOCIAL = "social"                  # Greeting, wellbeing, farewell
    COMPLEX = "complex"                # Multi-faceted question needing deep decomposition
    IMPOSSIBLE = "impossible"          # Unanswerable / paradoxical


@dataclass
class SubQuestion:
    """An atomic sub-question that can be answered independently.
    
    Each sub-question represents one step in the decomposition tree.
    The answer field is populated by the engine after processing.
    """
    id: int                           # Position in decomposition order
    text: str                         # The sub-question text (e.g., "what is quantum?")
    category: QuestionCategory         # Type of sub-question
    target_concept: str               # The main concept this sub-question addresses
    relation_type: str = "semantic"   # Graph relation to prioritize
    depth: int = 0                    # Decomposition depth (0 = root)
    is_answered: bool = False         # Whether this has been answered
    answer: str = ""                  # The answer text (populated later)
    confidence: float = 0.0           # How confident we are in the answer
    sub_questions: List[SubQuestion] = field(default_factory=list)  # Nested sub-questions
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "category": self.category.value,
            "target_concept": self.target_concept,
            "relation_type": self.relation_type,
            "depth": self.depth,
            "is_answered": self.is_answered,
            "answer": self.answer,
            "confidence": self.confidence,
            "sub_questions": [sq.to_dict() for sq in self.sub_questions],
        }


@dataclass
class DecompositionResult:
    """The complete decomposition of a user's question."""
    original_query: str                # The user's original text
    main_subject: str                  # The extracted main subject
    category: QuestionCategory         # The question category
    complexity: float                  # 0.0 (simple) to 1.0 (very complex)
    sub_questions: List[SubQuestion]   # Ordered list of sub-questions
    goal_stack: List[SubQuestion]      # Frontopolar goal stack (pending sub-questions)
    is_complete: bool = False          # Whether all sub-questions are answered
    has_ungrounded_aspects: bool = False  # Whether some aspects need web search
    synthesis_plan: List[str] = field(default_factory=list)  # How to combine answers
    
    def to_dict(self) -> Dict:
        return {
            "original_query": self.original_query,
            "main_subject": self.main_subject,
            "category": self.category.value,
            "complexity": self.complexity,
            "sub_questions": [sq.to_dict() for sq in self.sub_questions],
            "is_complete": self.is_complete,
            "has_ungrounded_aspects": self.has_ungrounded_aspects,
        }


# ─── Decomposition Strategy: Abstract Concept Detection ───

# Words whose GloVe similarity to abstract prototypes > concrete prototypes
# indicate the concept is abstract. Used instead of hardcoded lists.
_ABSTRACT_PROTOTYPES = ["love", "truth", "knowledge", "meaning", "beauty",
                        "justice", "freedom", "consciousness", "purpose"]
_CONCRETE_PROTOTYPES = ["table", "dog", "mountain", "car", "tree", "house",
                        "water", "rock", "bird", "star"]


def _is_abstract(concept: str, vector_fn=None) -> bool:
    """Determine if a concept is abstract using GloVe-based classification.
    
    ATL (anterior temporal lobe) computes abstractness from semantic neighborhood
    (Cousins et al., 2017). Falls back to suffix heuristics when vectors unavailable.
    """
    if not concept:
        return False
    cl = concept.lower().strip()
    
    # Try vector-based classification first
    if vector_fn is not None:
        try:
            vec = vector_fn(cl)
            if vec is not None:
                abs_sims = [float(np.dot(vec, vector_fn(p)))
                           for p in _ABSTRACT_PROTOTYPES if vector_fn(p) is not None]
                con_sims = [float(np.dot(vec, vector_fn(p)))
                           for p in _CONCRETE_PROTOTYPES if vector_fn(p) is not None]
                if abs_sims and con_sims:
                    return float(np.mean(abs_sims)) > float(np.mean(con_sims))
        except Exception:
            pass
    
    # Suffix heuristics as fallback
    if cl.endswith('ness') or cl.endswith('ity') or cl.endswith('tion'):
        return True
    if cl.endswith('ism') or cl.endswith('ment') or cl.endswith('ance'):
        return True
    if cl.endswith('ence') or cl.endswith('ship') or cl.endswith('dom'):
        return True
    
    # Known abstract concepts
    _ABSTRACT_SET = {
        "love", "hate", "fear", "joy", "truth", "beauty", "justice",
        "freedom", "meaning", "purpose", "life", "death", "time",
        "consciousness", "mind", "soul", "knowledge", "wisdom",
        "power", "hope", "dream", "fate", "destiny", "reality",
        "existence", "god", "infinity", "eternity", "morality",
        "ethics", "good", "evil", "sin", "virtue", "honor",
        "courage", "compassion", "empathy", "trust", "faith",
        "imagination", "creativity", "intelligence", "memory",
    }
    return cl in _ABSTRACT_SET


# ─── Question Analyzer ───

class QuestionAnalyzer:
    """Analyzes a user's question to determine its type, complexity, and decomposition needs.
    
    Inspired by the PFC's role in task-set identification: before any reasoning
    begins, the brain classifies what KIND of problem it faces (Badre, 2008).
    """
    
    # Question type detection patterns
    WHAT_IS_PATTERNS = [
        re.compile(r"what\s+is\s+(an?|the)?\s*(.+)", re.IGNORECASE),
        re.compile(r"what\s+are\s+(an?|the)?\s*(.+)", re.IGNORECASE),
        re.compile(r"what's\s+(an?|the)?\s*(.+)", re.IGNORECASE),
        re.compile(r"what\s+does\s+(.+?)\s+mean", re.IGNORECASE),
        re.compile(r"define\s+(.+)", re.IGNORECASE),
        re.compile(r"explain\s+(.+)", re.IGNORECASE),
        re.compile(r"describe\s+(.+)", re.IGNORECASE),
    ]
    
    WHY_PATTERNS = [
        re.compile(r"why\s+(?:is|are|does|do|can|did|would)\s+(.+)", re.IGNORECASE),
        re.compile(r"why\s+(.+)", re.IGNORECASE),
        re.compile(r"what\s+causes?\s+(.+)", re.IGNORECASE),
        re.compile(r"what's?\s+the\s+(?:reason|cause)\s+(?:for|of)\s+(.+)", re.IGNORECASE),
    ]
    
    HOW_PATTERNS = [
        re.compile(r"how\s+does\s+(.+?)\s+work(?:\?|$)", re.IGNORECASE),
        re.compile(r"how\s+does\s+(.+?)\s+(?:function|operate|happen|form|occur)(?:\?|$)", re.IGNORECASE),
        re.compile(r"how\s+(?:does|do|is|are|can|did|would)\s+(.+)", re.IGNORECASE),
        re.compile(r"what\s+is\s+the\s+(?:process|mechanism)\s+(?:of|for)\s+(.+)", re.IGNORECASE),
    ]
    
    COMPARE_PATTERNS = [
        re.compile(r"(?:compare|difference|contrast|versus|vs)\s+(?:the\s+)?(?:between\s+)?(.+?)\s+(?:and|vs|versus|with|to)\s+(.+)", re.IGNORECASE),
        re.compile(r"what\s*(?:is\s+|'s\s+)?the\s+difference\s+between\s+(.+?)\s+and\s+(.+)", re.IGNORECASE),
        re.compile(r"how\s+(?:is|are)\s+(.+?)\s+(?:different|similar)\s+(?:to|from|than)\s+(.+)", re.IGNORECASE),
        re.compile(r"what\s+do\s+(.+?)\s+and\s+(.+?)\s+have\s+in\s+common", re.IGNORECASE),
    ]
    
    HYPOTHETICAL_PATTERNS = [
        re.compile(r"what\s+if\s+(.+)", re.IGNORECASE),
        re.compile(r"what\s+happens?\s+if\s+(.+)", re.IGNORECASE),
        re.compile(r"what\s+would\s+happen\s+if\s+(.+)", re.IGNORECASE),
        re.compile(r"what\s+happens?\s+when\s+(.+)", re.IGNORECASE),
        re.compile(r"what\s+would\s+happen\s+when\s+(.+)", re.IGNORECASE),
        re.compile(r"suppose\s+(.+?)(?:\?|$)", re.IGNORECASE),
        re.compile(r"imagine\s+(.+?)(?:\?|$)", re.IGNORECASE),
        re.compile(r"if\s+(.+?)(?:,\s*what|,what|what)\s+(?:would|will|does|happens?)", re.IGNORECASE),
    ]
    
    TELL_ME_PATTERNS = [
        re.compile(r"tell\s+me\s+(?:about|more\s+about)\s+(.+)", re.IGNORECASE),
        re.compile(r"tell\s+me\s+(.+)", re.IGNORECASE),
    ]
    
    ANALOGY_PATTERNS = [
        re.compile(r"(.+?)\s*:\s*(.+?)\s*::\s*(.+?)\s*:\s*(.+)", re.IGNORECASE),
        re.compile(r"(.+?)\s+is\s+to\s+(.+?)\s+as\s+(.+?)\s+is\s+to\s+(.+)", re.IGNORECASE),
        re.compile(r"what's?\s+the\s+analogy\s+(?:for|of)\s+(.+?)\s+(?:and|to)\s+(.+)", re.IGNORECASE),
    ]
    
    # Words/phrases that mark abstract concepts needing multi-perspective treatment
    _ABSTRACT_MARKERS = {
        "meaning of life", "purpose of", "nature of", "essence of",
        "what is truth", "what is love", "what is beauty", "what is justice",
        "what is consciousness", "what is reality", "what is time",
        "why do we exist", "why are we here", "what happens after",
    }
    
    # Words indicating multi-part questions
    _MULTI_PART_CONJUNCTIONS = {"and", "or", "also", "additionally", "furthermore"}
    
    # Words indicating impossible/paradoxical questions
    _IMPOSSIBLE_MARKERS = {
        "can god create", "unstoppable force", "irresistible force",
        "immovable object", "what is the sound", "one hand clapping",
        "can you prove", "can we know", "is reality real",
        "exist instead of nothing", "why is there something instead of nothing",
        "why does everything exist", "why does anything exist",
    }

    @classmethod
    def analyze(cls, query: str, vector_fn=None) -> Tuple[QuestionCategory, str, float]:
        """Analyze the question to determine its type, main subject, and complexity.
        
        Returns:
            (category, main_subject, complexity_score)
        """
        query_lower = query.lower().strip()
        query_clean = query_lower.strip(" ?!.,;:")
        
        # Step 1: Check for abstract markers (overrides other patterns)
        for marker in cls._ABSTRACT_MARKERS:
            if marker in query_clean:
                # Extract the abstract concept after the marker
                for pat in [r"(?:meaning|purpose|nature|essence)\s+of\s+(.+)",
                            r"what\s+is\s+(.+?)$",
                            r"what\s+is\s+(an?|the)\s+(.+?)$"]:
                    m = re.search(pat, query_clean)
                    if m:
                        concept = m.group(1) if m.lastindex == 1 else m.group(2)
                        # Check if target concept is abstract
                        if _is_abstract(concept.strip(), vector_fn):
                            return (QuestionCategory.ABSTRACT, concept.strip(), 0.8)
                return (QuestionCategory.ABSTRACT, query_clean, 0.7)
        
        # Step 2: Check for social patterns
        social_patterns = {
            "hello": ["hi", "hello", "hey", "greetings", "howdy", "sup", "yo"],
            "goodbye": ["bye", "goodbye", "farewell", "see you", "good night"],
            "how_are_you": ["how are you", "how's it", "how is it", "how are you doing"],
            "who_are_you": ["who are you", "what can you do", "what are you", "tell me about yourself"],
        }
        for social_type, patterns in social_patterns.items():
            if any(p in query_clean for p in patterns):
                return (QuestionCategory.SOCIAL, query_clean, 0.1)
        
        # Step 3: Check analogy patterns
        for pat in cls.ANALOGY_PATTERNS:
            if pat.search(query_clean):
                return (QuestionCategory.ANALOGY, query_clean, 0.7)
        
        # Step 4: Check compare patterns
        for pat in cls.COMPARE_PATTERNS:
            m = pat.search(query_clean)
            if m:
                # Extract both concepts being compared
                groups = [g.strip() for g in m.groups() if g]
                if len(groups) >= 2:
                    return (QuestionCategory.COMPARE, groups[0], 0.6)
        
        # Step 5: Check for impossible/paradoxical markers FIRST.
        # A paradox ("unstoppable force meets immovable object") can
        # superficially match hypothetical/why patterns, but the brain's
        # paradox network (rIFG/BA47 identification of contradiction +
        # ACC conflict detection) flags it before simulation. Mirror that by
        # testing impossibility before hypothetical, so the contradiction is
        # routed to the paradox handler rather than a literal simulation.
        for marker in cls._IMPOSSIBLE_MARKERS:
            if marker in query_clean:
                return (QuestionCategory.IMPOSSIBLE, query_clean, 0.9)

        # Step 6: Check hypothetical patterns
        for pat in cls.HYPOTHETICAL_PATTERNS:
            m = pat.search(query_clean)
            if m:
                subject = m.group(1).strip() if m.groups() else query_clean
                return (QuestionCategory.HYPOTHETICAL, subject, 0.7)
        
        # Step 6: Check "why" patterns
        for pat in cls.WHY_PATTERNS:
            m = pat.match(query_clean)
            if m:
                subject = m.group(1).strip() if m.lastindex >= 1 else query_clean
                return (QuestionCategory.WHY, cls._clean_subject(subject), 0.5)

        # Step 7: Check "how" patterns
        for pat in cls.HOW_PATTERNS:
            m = pat.match(query_clean)
            if m:
                subject = m.group(1).strip() if m.lastindex >= 1 else query_clean
                return (QuestionCategory.HOW, cls._clean_subject(subject), 0.6)

        # Step 8: Check "tell me" patterns
        for pat in cls.TELL_ME_PATTERNS:
            m = pat.match(query_clean)
            if m:
                subject = m.group(1).strip()
                if _is_abstract(subject, vector_fn):
                    return (QuestionCategory.ABSTRACT, cls._clean_subject(subject), 0.7)
                return (QuestionCategory.TELL_ME, cls._clean_subject(subject), 0.4)
        
        # Step 9: Check "what is" patterns
        for pat in cls.WHAT_IS_PATTERNS:
            m = pat.match(query_clean)
            if m:
                subject = m.group(m.lastindex).strip() if m.groups() else query_clean
                if _is_abstract(subject, vector_fn):
                    return (QuestionCategory.ABSTRACT, subject, 0.7)
                return (QuestionCategory.WHAT_IS, subject, 0.3)
        
        # Step 11: Check for multi-part questions (multiple "?" or conjunctions between clauses)
        num_questions = query.count("?")
        if num_questions >= 2:
            return (QuestionCategory.COMPLEX, query_clean, 0.8)
        
        # Check for conjunctive multi-part (e.g., "what is X and how does Y work")
        if any(conj in query_clean.split() for conj in cls._MULTI_PART_CONJUNCTIONS):
            # Count how many distinct question-like phrases
            question_words = ["what", "why", "how", "who", "where", "when"]
            qw_count = sum(1 for w in question_words if w in query_clean)
            if qw_count >= 2:
                return (QuestionCategory.COMPLEX, query_clean, 0.8)
        
        # Step 12: Default - treat as general
        return (QuestionCategory.GENERAL, query_clean, 0.2)

    @staticmethod
    def _clean_verb_phrases(subject: str) -> str:
        """Remove trailing light verbs from HOW-extracted subjects.
        
        E.g. 'a computer work' -> 'a computer', 'black holes form' -> 'black holes'.
        Also removes determiners for cleaner sub-question templates.
        """
        if not subject:
            return subject
        # Strip trailing light verbs
        light_verb_suffixes = [
            r'\s+work(?:s|ing|ed)?$', r'\s+function(?:s|ing|ed)?$',
            r'\s+operate(?:s|d)?$', r'\s+happen(?:s|ed|ing)?$',
            r'\s+form(?:s|ed|ing)?$', r'\s+occur(?:s|red|ring)?$',
            r'\s+behave?(?:s|d)?$', r'\s+exist(?:s|ed|ing)?$',
        ]
        result = subject
        for suffix in light_verb_suffixes:
            result = re.sub(suffix, '', result)
        
        # Also strip leading determiners that make templates awkward
        result = re.sub(r'^(an?|the)\s+', '', result, flags=re.IGNORECASE).strip()
        
        return result if result else subject

    # ─── Subject cleaning (dependency-aware head recovery) ───
    # The brain's left inferior frontal gyrus / Broca's area performs *dependency*
    # parsing: it binds each verb to its argument (who did what to whom) and
    # recovers the head noun phrase, rather than taking the raw surface residual
    # as "the subject". RAVANA's regex patterns are greedy and frequently
    # capture relational framing ("similar to time", "the causes of WW1",
    # "the sun to rise") as the subject. _clean_subject implements the
    # dependency-aware head-recovery analog: strip relational modifiers,
    # "of"-framing heads, and infinitive/light-verb complements so the
    # graph walk and web search target the true concept, not the surface string.
    _SUBJECT_OF_HEADS = {
        "cause", "causes", "reason", "reasons", "meaning", "definition",
        "nature", "purpose", "result", "results", "effect", "effects",
        "significance", "difference", "differences", "relationship", "relation",
        "relations", "origin", "origins", "importance", "role", "roles",
        "source", "value", "problem", "issue",
    }
    _SUBJECT_LEAD_MODS = (
        r"^(?:similar|related|opposite|different|contrasted|analogous|comparable)\s+(?:to|from|than|with)\s+",
        r"^instead\s+of\s+",
        r"^versus\s+",
    )
    _SUBJECT_TRAIL_VERBS = {
        "rise", "fall", "go", "come", "happen", "happens", "occur", "occurs",
        "exist", "exists", "work", "works", "function", "operate", "form",
        "forms", "appear", "disappear", "change", "move", "grow", "develop",
        "emerge", "remain", "stay", "become", "rotate", "orbit", "expand",
        "shrink", "flow", "burn", "shine", "erupt", "freeze", "melt", "boil",
        "react", "evolve", "survive", "thrive", "fail", "win", "lose", "die",
        "live", "fly", "run", "eat", "see", "know", "make", "take", "give",
        "get", "find", "use", "create", "build", "play", "show", "tell",
        "feel", "think", "learn", "help", "lift", "meet", "do", "does",
        "did", "is", "are", "was", "were", "be", "been", "being",
    }

    @classmethod
    def _clean_subject(cls, subject: str) -> str:
        """Recover the true head noun phrase from a raw extracted subject.

        Mirrors IFG dependency parsing: remove relational wrappers and verb
        complements so the subject is the concept, not the surface residual.
        """
        if not subject:
            return subject
        original = subject
        s = subject.lower().strip(" ?!.,;:()\"'")
        if not s:
            return original

        # 1. Strip leading relational modifiers ("similar to X" -> "X").
        for pat in cls._SUBJECT_LEAD_MODS:
            s = re.sub(pat, "", s, flags=re.IGNORECASE).strip()

        # 1b. Strip a leading dummy pronoun "it " ("it rain" -> "rain")
        s = re.sub(r"^it\s+", "", s, flags=re.IGNORECASE).strip()

        # 2. Strip a leading determiner + relational "of"-head
        #    ("the causes of WW1" -> "WW1"). Deliberately excludes valid
        #    compounds like "speed of light", "law of X", "capital of X".
        m = re.match(r"^(?:the|an?)\s+([a-z]+(?:s)?)\s+of\s+(.+)$", s)
        if m and m.group(1) in cls._SUBJECT_OF_HEADS:
            s = m.group(2).strip()

        # 3. Strip a trailing infinitive complement ("the sun to rise" -> "the sun").
        s = re.sub(r"\s+to\s+[a-z]+(?:s|ed|ing)?$", "", s, flags=re.IGNORECASE).strip()

        # 4. (Removed) A bare trailing-verb strip destroyed valid noun
        #    compounds: "sun rise" -> "sun", "stock market" -> "stock". The
        #    infinitive case ("the sun TO rise") is already handled by step 3.
        #    Genuine verb-complement subjects are rare in practice and the
        #    over-trim caused web search to target the wrong (too-short) head.

        # 5. Strip a leading determiner for a clean head.
        s = re.sub(r"^(?:the|an?)\s+", "", s, flags=re.IGNORECASE).strip()

        # 6. Guard against over-cleaning -> fall back to the original.
        if not s:
            return original
        return s

    @classmethod
    def extract_subject(cls, query: str) -> str:
        """Extract the main subject/concept from a question.
        
        Uses heuristics rather than full NLP parsing. Returns the most likely
        subject noun phrase the question is about.
        """
        query_lower = query.lower().strip(" ?!.,;:")
        
        # Try to extract from known question patterns
        all_patterns = [
            (cls.WHAT_IS_PATTERNS, lambda m: m.group(m.lastindex).strip()),
            (cls.WHY_PATTERNS, lambda m: m.group(1).strip()),
            (cls.HOW_PATTERNS, lambda m: m.group(1).strip()),
            (cls.TELL_ME_PATTERNS, lambda m: m.group(1).strip()),
            (cls.COMPARE_PATTERNS, lambda m: m.group(1).strip()),
            (cls.HYPOTHETICAL_PATTERNS, lambda m: m.group(1).strip()),
        ]
        
        for patterns, extractor in all_patterns:
            for pat in patterns:
                m = pat.search(query_lower)
                if m:
                    return extractor(m)
        
        # Fallback: extract the first noun-like word that's not a question word
        words = query_lower.split()
        question_words = {"what", "why", "how", "who", "where", "when", "which",
                          "is", "are", "was", "were", "do", "does", "did", "can",
                          "could", "would", "should", "will", "shall", "may", "might"}
        for w in words:
            wc = w.strip(".,!?\"'()")
            if wc not in question_words and len(wc) >= 3:
                return wc
        
        return query_lower


# ─── Decomposition Strategies ───

class DecompositionStrategies:
    """Decomposition strategies for each question type.
    
    Each strategy defines:
    - What sub-questions to ask
    - In what order
    - What relation type to prioritize for each
    - How to synthesize the sub-answers
    
    Inspired by:
    - SOAR's Universal Subgoaling (Newell, 1990): when direct answer fails, 
      generate subgoals
    - Frontopolar cortex cognitive branching (Koechlin & Hyafil, 2007): 
      hold main goal while processing sub-goals
    - Mental model construction (Johnson-Laird, 1983): build answer by 
      combining multiple mental models
    """

    @staticmethod
    def definitional(subject: str, vector_fn=None) -> DecompositionResult:
        """'What is X?' decomposition.
        
        Human cognitive process:
        1. CATEGORIZE: "What kind of thing is X?" (ATL semantic hub)
        2. DEFINE: "What are X's essential properties?" (PFC retrieval)
        3. EXPLAIN: "How does X work / what does X do?" (temporal-parietal)
        4. EXAMPLE: "What are examples of X?" (hippocampal exemplar retrieval)
        5. RELATE: "What is X connected to?" (semantic network spread)
        """
        subj_lower = subject.lower() if subject else ""
        
        is_abstract_concept = _is_abstract(subject, vector_fn) if subject else False
        
        sub_questions = []
        
        # SQ1: Category - "What kind of thing is X?"
        sq1 = SubQuestion(
            id=1,
            text=f"what category does {subject} belong to" if subject else "what is this about",
            category=QuestionCategory.WHAT_IS,
            target_concept=subj_lower,
            relation_type="is_a",
            depth=1,
        )
        sub_questions.append(sq1)
        
        # SQ2: Properties/Essence - "What are X's defining features?"
        sq2 = SubQuestion(
            id=2,
            text=f"what are the key properties of {subject}" if subject else "what are its properties",
            category=QuestionCategory.WHAT_IS,
            target_concept=subj_lower,
            relation_type="property",
            depth=1,
        )
        sub_questions.append(sq2)
        
        if not is_abstract_concept:
            # SQ3: How it works / what it does (for concrete concepts)
            sq3 = SubQuestion(
                id=3,
                text=f"how does {subject} work" if subject else "how does it work",
                category=QuestionCategory.HOW,
                target_concept=subj_lower,
                relation_type="causal",
                depth=1,
            )
            sub_questions.append(sq3)
        else:
            # SQ3: Examples/manifestations (for abstract concepts)
            sq3 = SubQuestion(
                id=3,
                text=f"what are examples of {subject}" if subject else "what are examples",
                category=QuestionCategory.WHAT_IS,
                target_concept=subj_lower,
                relation_type="semantic",
                depth=1,
            )
            sub_questions.append(sq3)
        
        # SQ4: Relation/connection to other concepts
        sq4 = SubQuestion(
            id=4,
            text=f"what is {subject} related to" if subject else "what is it related to",
            category=QuestionCategory.GENERAL,
            target_concept=subj_lower,
            relation_type="semantic",
            depth=1,
        )
        sub_questions.append(sq4)
        
        synthesis_plan = [
            "category",     # Start with what kind of thing it is
            "properties",   # Then describe its features
            "process",      # Then how it works / manifests
            "relation",     # Finally relate to broader context
        ]
        
        return DecompositionResult(
            original_query=f"what is {subject}" if subject else "general question",
            main_subject=subject or "",
            category=QuestionCategory.WHAT_IS,
            complexity=0.5 if not is_abstract_concept else 0.7,
            sub_questions=sub_questions,
            goal_stack=list(sub_questions),
            synthesis_plan=synthesis_plan,
        )

    @staticmethod
    def causal(subject: str, query: str = "") -> DecompositionResult:
        """'Why X?' decomposition.
        
        Human cognitive process:
        1. IDENTIFY CAUSE: "What directly causes X?" (causal reasoning network)
        2. TRACE MECHANISM: "Through what mechanism?" (dorsal stream simulation)
        3. CONDITIONS: "Under what conditions?" (counterfactual thinking)
        4. ALTERNATIVES: "Are there other causes?" (exploratory search)
        5. SIGNIFICANCE: "Why does it matter?" (valuation/meaning)
        """
        subj_lower = subject.lower() if subject else ""
        
        sub_questions = []
        
        # SQ1: Direct cause
        sq1 = SubQuestion(
            id=1,
            text=f"what causes {subject}" if subject else "what causes this",
            category=QuestionCategory.WHY,
            target_concept=subj_lower,
            relation_type="causal",
            depth=1,
        )
        sub_questions.append(sq1)
        
        # SQ2: Mechanism explanation
        sq2 = SubQuestion(
            id=2,
            text=f"how does {subject} happen" if subject else "how does it happen",
            category=QuestionCategory.HOW,
            target_concept=subj_lower,
            relation_type="causal",
            depth=1,
        )
        sub_questions.append(sq2)
        
        # SQ3: Conditions/context
        sq3 = SubQuestion(
            id=3,
            text=f"under what conditions does {subject} occur" if subject else "under what conditions",
            category=QuestionCategory.HYPOTHETICAL,
            target_concept=subj_lower,
            relation_type="causal",
            depth=1,
        )
        sub_questions.append(sq3)
        
        # SQ4: Significance
        sq4 = SubQuestion(
            id=4,
            text=f"why does {subject} matter" if subject else "why does it matter",
            category=QuestionCategory.ABSTRACT,
            target_concept=subj_lower,
            relation_type="causal",
            depth=1,
        )
        sub_questions.append(sq4)
        
        synthesis_plan = [
            "cause",        # Start with the direct cause
            "mechanism",    # Explain the mechanism
            "conditions",   # Note the conditions
            "significance", # Why it matters
        ]
        
        return DecompositionResult(
            original_query=query or f"why {subject}",
            main_subject=subject or "",
            category=QuestionCategory.WHY,
            complexity=0.6,
            sub_questions=sub_questions,
            goal_stack=list(sub_questions),
            synthesis_plan=synthesis_plan,
        )

    @staticmethod
    def procedural(subject: str, query: str = "") -> DecompositionResult:
        """'How does X work?' decomposition.
        
        Human cognitive process:
        1. COMPONENTS: "What parts make up X?" (part-whole decomposition)
        2. FUNCTION: "What does each part do?" (functional attribution)
        3. INTERACTION: "How do the parts interact?" (causal chain simulation)
        4. SEQUENCE: "In what order does it happen?" (temporal sequencing)
        """
        # Clean subject of trailing light verbs and determiners
        clean_subject = QuestionAnalyzer._clean_verb_phrases(subject)
        subj_lower = clean_subject.lower() if clean_subject else ""
        
        sub_questions = []
        subj_display = clean_subject or subject or ""
        
        # SQ1: Components
        sq1 = SubQuestion(
            id=1,
            text=f"what are the parts of {subj_display}" if subj_display else "what are its parts",
            category=QuestionCategory.WHAT_IS,
            target_concept=subj_lower,
            relation_type="semantic",
            depth=1,
        )
        sub_questions.append(sq1)
        
        # SQ2: Function of each part
        sq2 = SubQuestion(
            id=2,
            text=f"what does each part of {subj_display} do" if subj_display else "what does each part do",
            category=QuestionCategory.HOW,
            target_concept=subj_lower,
            relation_type="causal",
            depth=1,
        )
        sub_questions.append(sq2)
        
        # SQ3: Interaction between parts
        sq3 = SubQuestion(
            id=3,
            text=f"how do the parts of {subj_display} work together" if subj_display else "how do they work together",
            category=QuestionCategory.HOW,
            target_concept=subj_lower,
            relation_type="causal",
            depth=1,
        )
        sub_questions.append(sq3)
        
        # SQ4: Sequential process
        sq4 = SubQuestion(
            id=4,
            text=f"what is the step by step process of {subj_display}" if subj_display else "what is the process",
            category=QuestionCategory.HOW,
            target_concept=subj_lower,
            relation_type="temporal",
            depth=1,
        )
        sub_questions.append(sq4)
        
        synthesis_plan = [
            "components",    # Start with what parts
            "functions",     # What each does
            "interaction",   # How they interact
            "process",       # The sequential process
        ]
        
        return DecompositionResult(
            original_query=query or f"how does {clean_subject or subject} work",
            main_subject=clean_subject or subject or "",
            category=QuestionCategory.HOW,
            complexity=0.6,
            sub_questions=sub_questions,
            goal_stack=list(sub_questions),
            synthesis_plan=synthesis_plan,
        )

    @staticmethod
    def comparative(concept_a: str, concept_b: str, query: str = "") -> DecompositionResult:
        """'Compare X and Y' decomposition.
        
        Human cognitive process:
        1. ANALYZE A: "What is X?" (independent analysis)
        2. ANALYZE B: "What is Y?" (independent analysis)
        3. DIFFERENCES: "How are they different?" (contrastive retrieval)
        4. SIMILARITIES: "How are they similar?" (analogical mapping)
        5. SYNTHESIS: "What can we learn from comparing them?" (integration)
        """
        sub_questions = []
        
        # SQ1: Analyze concept A
        sq1 = SubQuestion(
            id=1,
            text=f"what is {concept_a}",
            category=QuestionCategory.WHAT_IS,
            target_concept=concept_a.lower(),
            relation_type="is_a",
            depth=1,
        )
        sub_questions.append(sq1)
        
        # SQ2: Analyze concept B
        sq2 = SubQuestion(
            id=2,
            text=f"what is {concept_b}",
            category=QuestionCategory.WHAT_IS,
            target_concept=concept_b.lower(),
            relation_type="is_a",
            depth=1,
        )
        sub_questions.append(sq2)
        
        # SQ3: Differences
        sq3 = SubQuestion(
            id=3,
            text=f"how are {concept_a} and {concept_b} different",
            category=QuestionCategory.COMPARE,
            target_concept=f"{concept_a.lower()}|{concept_b.lower()}",
            relation_type="contrastive",
            depth=1,
        )
        sub_questions.append(sq3)
        
        # SQ4: Similarities
        sq4 = SubQuestion(
            id=4,
            text=f"how are {concept_a} and {concept_b} similar",
            category=QuestionCategory.COMPARE,
            target_concept=f"{concept_a.lower()}|{concept_b.lower()}",
            relation_type="analogical",
            depth=1,
        )
        sub_questions.append(sq4)
        
        synthesis_plan = [
            "a",            # First, describe A
            "b",            # Then describe B
            "differences",  # Then highlight differences
            "similarities", # Then note similarities
        ]
        
        return DecompositionResult(
            original_query=query or f"compare {concept_a} and {concept_b}",
            main_subject=concept_a,
            category=QuestionCategory.COMPARE,
            complexity=0.6,
            sub_questions=sub_questions,
            goal_stack=list(sub_questions),
            synthesis_plan=synthesis_plan,
        )

    @staticmethod
    def hypothetical(scenario: str, query: str = "") -> DecompositionResult:
        """'What if X?' decomposition.
        
        Human cognitive process (counterfactual thinking):
        1. BASELINE: "What is X's normal role/function?" (establish baseline)
        2. DISRUPTION: "What happens if X is changed/removed?" (intervention)
        3. CASCADE: "What cascade of effects follows?" (mental simulation)
        4. MAGNITUDE: "How significant are these effects?" (valuation)
        5. ALTERNATIVES: "What other outcomes are possible?" (counterfactual diversity)
        """
        subj_lower = scenario.lower() if scenario else ""
        
        # Extract the key concept from the scenario
        words = subj_lower.split()
        # Find the first content noun in the scenario
        stop_words = {"the", "a", "an", "if", "when", "what", "would", "happen",
                      "is", "are", "was", "were", "we", "you", "they", "it"}
        key_concept = ""
        for w in words:
            if w not in stop_words and len(w) >= 3:
                key_concept = w
                break
        
        sub_questions = []
        
        # SQ1: Establish baseline - what is this thing's normal role?
        sq1 = SubQuestion(
            id=1,
            text=f"what is the normal role of {key_concept}" if key_concept else "what is the normal situation",
            category=QuestionCategory.WHAT_IS,
            target_concept=key_concept,
            relation_type="semantic",
            depth=1,
        )
        sub_questions.append(sq1)
        
        # SQ2: Immediate effect of the change
        sq2 = SubQuestion(
            id=2,
            text=f"what happens immediately when {scenario}" if scenario else "what happens immediately",
            category=QuestionCategory.WHY,
            target_concept=key_concept,
            relation_type="causal",
            depth=1,
        )
        sub_questions.append(sq2)
        
        # SQ3: Cascading effects (second-order consequences)
        sq3 = SubQuestion(
            id=3,
            text=f"what are the cascading effects of {scenario}" if scenario else "what are the cascading effects",
            category=QuestionCategory.HYPOTHETICAL,
            target_concept=key_concept,
            relation_type="causal",
            depth=1,
        )
        sub_questions.append(sq3)
        
        # SQ4: Long-term consequences
        sq4 = SubQuestion(
            id=4,
            text=f"what would be the long term consequences of {scenario}" if scenario else "what would be the long term consequences",
            category=QuestionCategory.GENERAL,
            target_concept=key_concept,
            relation_type="temporal",
            depth=1,
        )
        sub_questions.append(sq4)
        
        synthesis_plan = [
            "baseline",     # Explain normal situation
            "immediate",    # What happens right away
            "cascade",      # Chain of effects
            "longterm",     # Long-term outcomes
        ]
        
        return DecompositionResult(
            original_query=query or f"what if {scenario}",
            main_subject=key_concept or scenario,
            category=QuestionCategory.HYPOTHETICAL,
            complexity=0.7,
            sub_questions=sub_questions,
            goal_stack=list(sub_questions),
            synthesis_plan=synthesis_plan,
        )

    @staticmethod
    def abstract_analysis(subject: str, query: str = "") -> DecompositionResult:
        """Abstract concept analysis (meaning of life, truth, love, etc.).
        
        Human cognitive process (multi-perspective):
        1. EXPERIENTIAL: "What does this feel like / how is it experienced?"
        2. DEFINITIONAL: "How do people define this?"
        3. PHILOSOPHICAL: "What are different perspectives on this?"
        4. PERSONAL: "What does this mean to individuals?"
        5. CULTURAL: "How do different cultures see this?"
        """
        subj_lower = subject.lower() if subject else ""
        
        sub_questions = []
        
        # SQ1: Definitional - how it's commonly defined
        sq1 = SubQuestion(
            id=1,
            text=f"what is the common definition of {subject}" if subject else "what is the common definition",
            category=QuestionCategory.WHAT_IS,
            target_concept=subj_lower,
            relation_type="semantic",
            depth=1,
        )
        sub_questions.append(sq1)
        
        # SQ2: Perspectives - different ways of understanding it
        sq2 = SubQuestion(
            id=2,
            text=f"what are different perspectives on {subject}" if subject else "what are different perspectives",
            category=QuestionCategory.ABSTRACT,
            target_concept=subj_lower,
            relation_type="contrastive",
            depth=1,
        )
        sub_questions.append(sq2)
        
        # SQ3: Why it matters / significance
        sq3 = SubQuestion(
            id=3,
            text=f"why does {subject} matter" if subject else "why does it matter",
            category=QuestionCategory.ABSTRACT,
            target_concept=subj_lower,
            relation_type="causal",
            depth=1,
        )
        sub_questions.append(sq3)
        
        # SQ4: Personal/human connection
        sq4 = SubQuestion(
            id=4,
            text=f"how do people relate to {subject}" if subject else "how do people relate to it",
            category=QuestionCategory.GENERAL,
            target_concept=subj_lower,
            relation_type="semantic",
            depth=1,
        )
        sub_questions.append(sq4)
        
        synthesis_plan = [
            "definition",    # Start with what it means
            "perspectives",  # Different viewpoints
            "significance",  # Why it matters
            "connection",    # How it relates to us
        ]
        
        return DecompositionResult(
            original_query=query or f"what is {subject}",
            main_subject=subject or "",
            category=QuestionCategory.ABSTRACT,
            complexity=0.8,
            sub_questions=sub_questions,
            goal_stack=list(sub_questions),
            synthesis_plan=synthesis_plan,
        )

    @staticmethod
    def general(query: str) -> DecompositionResult:
        """Generic decomposition for unclassified queries."""
        return DecompositionResult(
            original_query=query,
            main_subject=query,
            category=QuestionCategory.GENERAL,
            complexity=0.3,
            sub_questions=[],  # No sub-questions for general queries
            goal_stack=[],
            synthesis_plan=["response"],
        )

    @staticmethod
    def impossible(query: str) -> DecompositionResult:
        """Decomposition for impossible/paradoxical questions.
        
        Human response to paradoxes:
        1. Identify the paradox
        2. Explain why it's paradoxical
        3. Discuss what makes it interesting
        4. Explore what it reveals about thinking
        """
        sub_questions = []
        
        sq1 = SubQuestion(
            id=1,
            text="what makes this question paradoxical",
            category=QuestionCategory.WHY,
            target_concept="paradox",
            relation_type="causal",
            depth=1,
        )
        sub_questions.append(sq1)
        
        sq2 = SubQuestion(
            id=2,
            text="why do humans find this question interesting",
            category=QuestionCategory.ABSTRACT,
            target_concept="paradox",
            relation_type="semantic",
            depth=1,
        )
        sub_questions.append(sq2)
        
        sq3 = SubQuestion(
            id=3,
            text="what does this question teach us about thinking",
            category=QuestionCategory.ABSTRACT,
            target_concept="paradox",
            relation_type="causal",
            depth=1,
        )
        sub_questions.append(sq3)
        
        return DecompositionResult(
            original_query=query,
            main_subject="paradox",
            category=QuestionCategory.IMPOSSIBLE,
            complexity=0.9,
            sub_questions=sub_questions,
            goal_stack=list(sub_questions),
            has_ungrounded_aspects=True,
            synthesis_plan=["paradox", "interest", "insight"],
        )


# ─── The Core Decomposition Engine ───

class QuestionDecompositionEngine:
    """Main decomposition engine — the brain's "question analysis" center.
    
    Inspired by:
    - Rostral PFC (BA 10): holds the main goal while managing sub-goals
    - Caudal PFC: executes concrete steps
    - DLPFC: maintains the goal stack in working memory
    
    Usage:
        engine = QuestionDecompositionEngine()
        result = engine.decompose("what is the meaning of life?")
        # result.sub_questions = [...]
        # Iterate through sub-questions, answer each, then synthesize
    """
    
    def __init__(self, vector_fn=None):
        self.vector_fn = vector_fn  # GloVe vector function for semantic analysis
        self._current_decomposition: Optional[DecompositionResult] = None
        
        # Goal stack (frontopolar pending set analog)
        self._pending_sub_questions: List[SubQuestion] = []
        self._completed_sub_questions: List[SubQuestion] = []
        self._current_goal_index: int = 0
    
    def decompose(self, query: str) -> DecompositionResult:
        """Decompose a question into its sub-questions.
        
        This is the main entry point. It:
        1. Analyzes the question type
        2. Applies the appropriate decomposition strategy
        3. Returns a structured decomposition result
        
        The caller should then iterate through sub_questions, answer each,
        and call synthesize() to combine them.
        """
        # Step 1: Analyze the question
        category, subject, complexity = QuestionAnalyzer.analyze(query, self.vector_fn)

        # IFG-style dependency cleanup: recover the true head noun phrase from
        # the greedily-captured surface subject ("similar to time" -> "time",
        # "the causes of WW1" -> "world war 1", "the sun to rise" -> "sun").
        subject = QuestionAnalyzer._clean_subject(subject)

        # If no subject was extracted, try harder
        if not subject or subject == query.lower().strip(" ?!.,;:"):
            subject = QuestionAnalyzer._clean_subject(QuestionAnalyzer.extract_subject(query))
        
        result = None
        
        # Step 2: Apply decomposition strategy based on category
        if category == QuestionCategory.WHAT_IS:
            result = DecompositionStrategies.definitional(subject, self.vector_fn)
        elif category == QuestionCategory.WHY:
            result = DecompositionStrategies.causal(subject, query)
        elif category == QuestionCategory.HOW:
            # Clean the subject before decomposing
            how_subject = QuestionAnalyzer._clean_verb_phrases(subject)
            result = DecompositionStrategies.procedural(how_subject, query)
        elif category == QuestionCategory.COMPARE:
            # For compare, extract BOTH concepts from the two capture groups.
            # (Previously used re.sub(..., r"\1", ...) which KEPT the first
            # concept as concept_b, and then extract_subject dropped
            # multi-word tails like "world war 2" -> "world". Capture
            # group(2) directly and clean it with the IFG-style helper.)
            _cmp = re.search(
                r"(?:compare|difference|contrast|versus|vs)\s+(?:the\s+)?(?:between\s+)?(.+?)\s+"
                r"(?:and|vs|versus|with|to)\s+(.+)",
                query.lower(), flags=re.IGNORECASE)
            concept_a_raw = _cmp.group(1) if _cmp else subject
            concept_b_raw = _cmp.group(2) if _cmp else ""
            concept_a = QuestionAnalyzer._clean_subject(concept_a_raw)
            concept_b = QuestionAnalyzer._clean_subject(concept_b_raw) if concept_b_raw else ""
            # Keep the cleaned subject consistent with concept_a.
            subject = concept_a or subject
            if not concept_b or concept_b == subject:
                concept_b = "other"
            result = DecompositionStrategies.comparative(subject, concept_b, query)
        elif category == QuestionCategory.HYPOTHETICAL:
            result = DecompositionStrategies.hypothetical(subject, query)
        elif category == QuestionCategory.ABSTRACT:
            result = DecompositionStrategies.abstract_analysis(subject, query)
        elif category == QuestionCategory.IMPOSSIBLE:
            result = DecompositionStrategies.impossible(query)
        elif category == QuestionCategory.COMPLEX:
            # Multi-part question: decompose each part
            result = self._decompose_complex(query)
        else:
            result = DecompositionStrategies.general(query)
        
        # Override category with the actual detected one
        if result:
            result.category = category
            result.complexity = complexity
        
        # Store the result for later synthesis
        self._current_decomposition = result
        self._pending_sub_questions = list(result.sub_questions) if result else []
        self._completed_sub_questions = []
        self._current_goal_index = 0
        
        return result
    
    def _decompose_complex(self, query: str) -> DecompositionResult:
        """Decompose complex multi-part questions by splitting into individual questions.
        
        Handles cases like "what is gravity and how does it affect time?"
        by splitting into "what is gravity" and "how does gravity affect time".
        """
        query_lower = query.lower().strip(" ?!.,;:")
        
        # Split on conjunctions that link independent questions
        parts = re.split(r"\s+(?:and|or|also|additionally)\s+", query_lower)
        
        if len(parts) < 2:
            return DecompositionStrategies.general(query)
        
        sub_questions = []
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            
            # Determine what kind of question this part is
            part_cat, part_subj, _ = QuestionAnalyzer.analyze(part, self.vector_fn)
            if not part_subj:
                part_subj = QuestionAnalyzer.extract_subject(part)
            part_subj = QuestionAnalyzer._clean_subject(part_subj)
            
            sq = SubQuestion(
                id=i + 1,
                text=part,
                category=part_cat,
                target_concept=part_subj,
                relation_type=QuestionCategory_RelationMap.get(part_cat, "semantic"),
                depth=1,
            )
            sub_questions.append(sq)
        
        return DecompositionResult(
            original_query=query,
            main_subject=sub_questions[0].target_concept if sub_questions else query,
            category=QuestionCategory.COMPLEX,
            complexity=0.8,
            sub_questions=sub_questions,
            goal_stack=list(sub_questions),
            synthesis_plan=[f"part_{i+1}" for i in range(len(parts))],
        )
    
    def next_sub_question(self) -> Optional[SubQuestion]:
        """Get the next pending sub-question from the goal stack.
        
        Frontopolar analog: pop the next sub-goal from the pending set
        while maintaining the main goal in active memory.
        """
        if not self._pending_sub_questions:
            return None
        
        sq = self._pending_sub_questions.pop(0)
        self._current_goal_index += 1
        return sq
    
    def complete_sub_question(self, sq: SubQuestion, answer: str, confidence: float = 0.0):
        """Mark a sub-question as answered and store its result.
        
        Called after each sub-question has been answered by the engine.
        """
        sq.answer = answer
        sq.confidence = confidence
        sq.is_answered = True
        self._completed_sub_questions.append(sq)
    
    def get_progress(self) -> Tuple[int, int]:
        """Return (completed, total) sub-questions."""
        total = len(self._completed_sub_questions) + len(self._pending_sub_questions)
        return (len(self._completed_sub_questions), total)
    
    def all_answered(self) -> bool:
        """Check if all sub-questions have been answered."""
        return len(self._pending_sub_questions) == 0 and len(self._completed_sub_questions) > 0
    
    def synthesize(self) -> Optional[str]:
        """Synthesize sub-answers into a coherent response.
        
        Combines the answered sub-questions into a natural-sounding narrative
        using the synthesis plan. This is the final output of the decomposition
        pipeline — a coherent, multi-sentence answer that addresses the original
        question from multiple angles.
        
        Returns the synthesized text, or None if no sub-questions were answered.
        """
        if not self._completed_sub_questions:
            return None
        
        if not self._current_decomposition:
            return None
        
        # Sort by ID to ensure correct order
        answered = sorted(self._completed_sub_questions, key=lambda sq: sq.id)
        
        synthesis_plan = self._current_decomposition.synthesis_plan
        category = self._current_decomposition.category
        subject = self._current_decomposition.main_subject
        
        return self._synthesize_from_plan(answered, synthesis_plan, category, subject)
    
    def _synthesize_from_plan(self, 
                              answered: List[SubQuestion],
                              plan: List[str],
                              category: QuestionCategory,
                              subject: str) -> str:
        """Execute the synthesis plan to combine sub-answers."""
        
        utterances = []
        
        for i, plan_item in enumerate(plan):
            if i < len(answered):
                answer = answered[i].answer
                if answer and len(answer) > 5:
                    utterances.append(answer)
        
        if not utterances:
            # Fallback: just use whatever answers we have
            for sq in answered:
                if sq.answer and len(sq.answer) > 5:
                    utterances.append(sq.answer)
        
        if not utterances:
            return None
        
        # Join with appropriate flow
        if len(utterances) == 1:
            return utterances[0]
        
        # Build a coherent paragraph
        result = " " .join(utterances)
        
        # Clean up
        result = result[0].upper() + result[1:] if result else result
        if not result.endswith((".", "?", "!")):
            result += "."
        result = re.sub(r'\s+', ' ', result)
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get full status of current decomposition."""
        if not self._current_decomposition:
            return {"status": "idle"}
        
        completed, total = self.get_progress()
        return {
            "status": "complete" if self.all_answered() else "in_progress",
            "original_query": self._current_decomposition.original_query,
            "category": self._current_decomposition.category.value,
            "complexity": self._current_decomposition.complexity,
            "sub_questions_completed": completed,
            "sub_questions_total": total,
            "completed_answers": [
                {"text": sq.text, "answer": sq.answer[:80] + "..." if len(sq.answer) > 80 else sq.answer}
                for sq in self._completed_sub_questions
            ],
        }


# ─── Utility Map ───

QuestionCategory_RelationMap = {
    QuestionCategory.WHAT_IS: "semantic",
    QuestionCategory.WHY: "causal",
    QuestionCategory.HOW: "causal",
    QuestionCategory.COMPARE: "contrastive",
    QuestionCategory.HYPOTHETICAL: "causal",
    QuestionCategory.TELL_ME: "semantic",
    QuestionCategory.ANALOGY: "analogical",
    QuestionCategory.ABSTRACT: "semantic",
    QuestionCategory.GENERAL: "semantic",
    QuestionCategory.SOCIAL: "social",
    QuestionCategory.COMPLEX: "semantic",
    QuestionCategory.IMPOSSIBLE: "semantic",
}
