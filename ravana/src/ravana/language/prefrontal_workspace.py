"""
RAVANA Prefrontal Workspace — Discourse Planner
=================================================
Plans multi-sentence discourse BEFORE generating. Inspired by Hagoort's
MUC (Memory, Unification, Control) framework (2005).

KEY DESIGN DECISIONS:
- Plans before generating: discourse intents are determined before any chain walk
- Cross-sentence coherence: each sentence's intent knows what came before
- Replaces independent "per-sentence seen sets" with structured discourse state
- Capacity: 7±2 slots (standard working memory), teens = 5
- Strategy selection is deterministic based on question type, not random
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import re


class DiscourseType:
    """Discourse intent types for sentence planning."""
    EXPLAIN = "explain"              # subject → semantic relation → object
    CAUSAL_EXPLAIN = "causal_explain"  # subject → causal relation → effect
    ELABORATE = "elaborate"          # go deeper on the previous concept
    CONTRAST = "contrast"            # subject vs opposite perspective
    CONNECT = "connect"             # link to broader context
    ASK_BACK = "ask_back"           # end with a question to the user
    CONTINUE = "continue"           # continue previous topic
    SELF_REFERENCE = "self_reference"  # "I think", "I feel" — epistemic stance


@dataclass
class DiscourseIntent:
    """A single sentence's discourse intent."""
    type: str                        # DiscourseType value
    subject: str                     # Main subject of this sentence
    primary_relation: str = "semantic"  # Graph edge type to traverse
    target_concept: str = ""         # Main object of discussion
    secondary_concept: str = ""      # For compare/contrast
    use_epistemic_hedge: bool = False  # "I think", "maybe", etc.
    end_with_question: bool = False  # "what do you think?"
    discourse_marker: str = ""       # "furthermore", "however", "also"
    seen_so_far: set = field(default_factory=set)  # concepts already used


@dataclass
class DiscoursePlan:
    """Full discourse plan for one response turn."""
    intents: List[DiscourseIntent] = field(default_factory=list)
    original_subject: str = ""
    question_type: str = "unknown"


class PrefrontalWorkspace:
    """Structured buffer that plans discourse BEFORE chain generation.

    Capacity: 5 (teen mode) or 7 (adult mode).
    For each turn, builds a discourse plan with 3 sentence intents
    that form a coherent narrative arc.
    """

    def __init__(self, capacity: int = 5):
        self.capacity = capacity  # teen capacity (adults = 7)
        self.last_plan: Optional[DiscoursePlan] = None
        self.topic_history: List[str] = []  # last 10 topics discussed

    # ─── Question Type Detection ───

    QUESTION_PATTERNS = {
        "what_is": [
            re.compile(r"what\s+is\s+(.+)", re.IGNORECASE),
            re.compile(r"what\s+are\s+(.+)", re.IGNORECASE),
            re.compile(r"what's\s+(.+)", re.IGNORECASE),
        ],
        "why": [
            re.compile(r"why\s+(?:is|are|does|do|can)\s+(.+)", re.IGNORECASE),
            re.compile(r"why\s+(.+)", re.IGNORECASE),
        ],
        "how": [
            re.compile(r"how\s+(?:is|are|does|do|can)\s+(.+)", re.IGNORECASE),
        ],
        "tell_me": [
            re.compile(r"tell\s+me\s+about\s+(.+)", re.IGNORECASE),
            re.compile(r"tell\s+me\s+more\s+about\s+(.+)", re.IGNORECASE),
        ],
        "compare": [
            re.compile(r"(?:compare|difference|versus|vs)\s+(.+)\s+(?:and|vs|versus|with)\s+(.+)", re.IGNORECASE),
            re.compile(r"what'?s?\s+the\s+difference\s+between\s+(.+)\s+and\s+(.+)", re.IGNORECASE),
        ],
        "do_you_know": [
            re.compile(r"do\s+you\s+know\s+(.+)", re.IGNORECASE),
            re.compile(r"have\s+you\s+heard\s+of\s+(.+)", re.IGNORECASE),
        ],
        "follow_up": [
            re.compile(r"(?:more|else|another|also|further|tell me more)", re.IGNORECASE),
        ],
    }

    # Discourse markers with empty-string padding for non-use.
    # These are minimal transition cues, not templates.
    DISCOURSE_MARKERS = {
        "elaborate": ["also", "furthermore", "in addition", "moreover", "besides"],
        "contrast": ["however", "but", "on the other hand", "yet", "although"],
        "connect": ["similarly", "likewise", "in the same way", "correspondingly"],
        "conclude": ["ultimately", "in essence", "at its core", "fundamentally"],
    }

    @classmethod
    def detect_question_type(cls, text: str) -> Tuple[str, List[str]]:
        """Detect question type from user input.

        Returns:
            (question_type, extracted_parts)
        """
        text_lower = text.lower().strip(" ?!.")

        for qtype, patterns in cls.QUESTION_PATTERNS.items():
            for pattern in patterns:
                m = pattern.match(text_lower)
                if m:
                    groups = [g.strip() for g in m.groups() if g]
                    return (qtype, groups)

        # Default: treat as general statement/query
        return ("general", [text_lower])

    # ─── Discourse Planning ───

    def plan_discourse(self, user_input: str, subject: str,
                       concept_pos: Dict[str, str],
                       associations: List[Tuple[str, float]],
                       past_topics: Optional[List[str]] = None,
                       is_follow_up: bool = False) -> DiscoursePlan:
        """Plan a multi-sentence discourse response.

        Analyzes the user's question, then builds a coherent 3-sentence plan.

        Args:
            user_input: Raw user text
            subject: Extracted topic/subject
            concept_pos: Part-of-speech map for concepts
            associations: Spread activation results from graph
            past_topics: Previously discussed topics
            is_follow_up: Whether this is a follow-up query

        Returns:
            DiscoursePlan with 3 DiscourseIntents
        """
        qtype, parts = self.detect_question_type(user_input)
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)

        # Build seen set from subject and top associations
        seen = {subject.lower()}
        for label, _ in associations[:3]:
            seen.add(label.lower())

        # Plan based on question type
        if qtype == "what_is":
            plan = self._plan_explain(subject, associations, seen, qtype)
        elif qtype == "why":
            plan = self._plan_causal_explain(subject, associations, seen, qtype)
        elif qtype == "tell_me":
            plan = self._plan_elaborate(subject, associations, seen, qtype)
        elif qtype == "compare":
            plan = self._plan_compare(subject, parts, associations, seen, qtype)
        elif qtype == "follow_up":
            plan = self._plan_continue(subject, associations, seen, qtype)
        elif qtype == "do_you_know":
            plan = self._plan_explain(subject, associations, seen, qtype)
        else:
            plan = self._plan_general(subject, associations, seen, qtype)

        # Ensure we have exactly 3 intents (pad if needed)
        while len(plan.intents) < 3:
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.ELABORATE,
                subject=subject,
                primary_relation="semantic",
                seen_so_far=seen.copy(),
            ))

        # Trim to capacity
        plan.intents = plan.intents[:self.capacity]

        # Apply discourse markers based on intent type
        for i, intent in enumerate(plan.intents):
            if i == 0:
                intent.discourse_marker = ""  # No marker for first sentence
            elif intent.type == DiscourseType.ELABORATE:
                intent.discourse_marker = self._pick_marker("elaborate")
            elif intent.type == DiscourseType.CONTRAST:
                intent.discourse_marker = self._pick_marker("contrast")
            elif intent.type == DiscourseType.CONNECT:
                intent.discourse_marker = self._pick_marker("connect")

        self.last_plan = plan
        if subject and subject not in self.topic_history:
            self.topic_history.append(subject.lower())
            if len(self.topic_history) > 10:
                self.topic_history.pop(0)

        return plan

    # ─── Planning Strategies ───
    # Enhanced with more diverse explanatory patterns (Bug E fix)
    # Neuroscience basis: prefrontal cortex builds causal models, not just association lists
    
    def _plan_explain(self, subject: str, associations: List[Tuple[str, float]],
                      seen: set, qtype: str) -> DiscoursePlan:
        """Plan: [EXPLAIN → ELABORATE → CONTRAST/CONNECT]
        
        Produces multi-perspective responses instead of flat association lists.
        """
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)
        subj_lower = subject.lower()

        # Sentence 1: EXPLAIN - what the subject is, its core nature
        target1 = self._pick_best_association(associations, seen, exclude_verbs=True)
        # Try to get a more specific explanation target (not just top association)
        deeper = self._pick_best_association(associations, seen, exclude_subject=target1 if target1 else subject)
        
        # Use deeper association for more explanatory intent
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.EXPLAIN,
            subject=subject,
            primary_relation="semantic",
            target_concept=target1,
            secondary_concept=deeper if deeper else "",
            seen_so_far=seen.copy(),
        ))
        if target1:
            seen.add(target1.lower())
        if deeper:
            seen.add(deeper.lower())

        # Sentence 2: ELABORATE — go deeper with causal or analogical reasoning
        causal_target = self._pick_best_association(associations, seen, prefer_causal=True)
        if causal_target:
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.CAUSAL_EXPLAIN,
                subject=subject,
                primary_relation="causal",
                target_concept=causal_target,
                seen_so_far=seen.copy(),
            ))
            seen.add(causal_target.lower())
        else:
            target2 = self._pick_best_association(associations, seen, exclude_subject=target1 if target1 else subject)
            if not target2:
                target2 = self._pick_random_relation(associations, seen)
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.ELABORATE,
                subject=target1 if target1 else subject,
                primary_relation="semantic",
                target_concept=target2,
                seen_so_far=seen.copy(),
            ))
            if target2:
                seen.add(target2.lower())

        # Sentence 3: CONTRAST or CONNECT back to broader context
        target3 = self._pick_best_association(associations, seen, prefer_contrast=True)
        intent3_type = DiscourseType.CONTRAST if target3 else DiscourseType.CONNECT
        plan.intents.append(DiscourseIntent(
            type=intent3_type,
            subject=subject,
            primary_relation="contrastive" if intent3_type == DiscourseType.CONTRAST else "semantic",
            target_concept=target3 or "",
            end_with_question=(target3 is None),  # ask back if nothing left to say
            seen_so_far=seen.copy(),
        ))

        return plan

    def _plan_causal_explain(self, subject: str, associations: List[Tuple[str, float]],
                              seen: set, qtype: str) -> DiscoursePlan:
        """Plan: [CAUSAL_EXPLAIN → ELABORATE → CONNECT]
        
        For 'why' and 'how' questions - produces deeper causal explanations
        rather than just listing associations.
        """
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)

        # Sentence 1: CAUSAL — what causes/creates the subject
        target1 = self._pick_best_association(associations, seen, prefer_causal=True)
        if not target1:
            # Fall back to any semantic association for explanatory power
            target1 = self._pick_best_association(associations, seen)
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.CAUSAL_EXPLAIN,
            subject=subject,
            primary_relation="causal",
            target_concept=target1,
            seen_so_far=seen.copy(),
        ))
        if target1:
            seen.add(target1.lower())

        # Sentence 2: ELABORATE — what the subject causes/leads to (effect)
        target2 = self._pick_best_association(associations, seen, prefer_causal=True, exclude_subject=target1 if target1 else subject)
        if not target2:
            target2 = self._pick_best_association(associations, seen, exclude_subject=target1 if target1 else subject)
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.ELABORATE,
            subject=target1 if target1 else subject,
            primary_relation="causal",
            target_concept=target2,
            use_epistemic_hedge=True,
            seen_so_far=seen.copy(),
        ))
        if target2:
            seen.add(target2.lower())

        # Sentence 3: CONNECT — link to a broader concept or practical implication
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.CONNECT,
            subject=subject,
            primary_relation="semantic",
            target_concept=self._pick_best_association(associations, seen) or "",
            end_with_question=True,
            seen_so_far=seen.copy(),
        ))

        return plan

    def _plan_elaborate(self, subject: str, associations: List[Tuple[str, float]],
                         seen: set, qtype: str) -> DiscoursePlan:
        """Plan: [EXPLAIN → CONTRAST → ASK_BACK] — for 'tell me about X'"""
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)

        target1 = self._pick_best_association(associations, seen)
        target2 = self._pick_best_association(associations, seen, prefer_contrast=True, exclude_subject=target1 if target1 else subject)

        plan.intents.append(DiscourseIntent(
            type=DiscourseType.EXPLAIN,
            subject=subject,
            target_concept=target1,
            seen_so_far=seen.copy(),
        ))
        if target1:
            seen.add(target1.lower())

        plan.intents.append(DiscourseIntent(
            type=DiscourseType.ELABORATE,
            subject=target1 if target1 else subject,
            target_concept=target2,
            seen_so_far=seen.copy(),
        ))
        if target2:
            seen.add(target2.lower())

        # ASK_BACK: generate an actual question to the user
        question = self._generate_follow_up_question(subject, target1)
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.ASK_BACK,
            subject=subject,
            target_concept=question,
            end_with_question=True,
            primary_relation="interrogative",
            seen_so_far=seen.copy(),
        ))

        return plan

    def _plan_compare(self, subject: str, parts: List[str],
                       associations: List[Tuple[str, float]],
                       seen: set, qtype: str) -> DiscoursePlan:
        """Plan: [EXPLAIN_A → EXPLAIN_B → CONTRAST] — for compare questions"""
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)
        concept_a = parts[0] if len(parts) > 0 else subject
        concept_b = parts[1] if len(parts) > 1 else ""

        plan.intents.append(DiscourseIntent(
            type=DiscourseType.EXPLAIN,
            subject=concept_a,
            target_concept=self._pick_best_association(associations, seen),
            seen_so_far=seen.copy(),
        ))
        seen.add(concept_a.lower())

        plan.intents.append(DiscourseIntent(
            type=DiscourseType.EXPLAIN,
            subject=concept_b if concept_b else concept_a,
            target_concept=self._pick_best_association(associations, seen),
            seen_so_far=seen.copy(),
        ))
        if concept_b:
            seen.add(concept_b.lower())

        plan.intents.append(DiscourseIntent(
            type=DiscourseType.CONTRAST,
            subject=concept_a,
            primary_relation="contrastive",
            secondary_concept=concept_b,
            seen_so_far=seen.copy(),
        ))

        return plan

    def _plan_continue(self, subject: str, associations: List[Tuple[str, float]],
                        seen: set, qtype: str) -> DiscoursePlan:
        """Plan: [CONTINUE → ELABORATE → CONNECT] — for follow-ups"""
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)

        target1 = self._pick_best_association(associations, seen)

        plan.intents.append(DiscourseIntent(
            type=DiscourseType.CONTINUE,
            subject=subject,
            target_concept=target1,
            seen_so_far=seen.copy(),
        ))

        if target1:
            seen.add(target1.lower())

        target2 = self._pick_best_association(associations, seen, exclude_subject=target1)

        plan.intents.append(DiscourseIntent(
            type=DiscourseType.ELABORATE,
            subject=subject,
            target_concept=target2,
            seen_so_far=seen.copy(),
        ))

        plan.intents.append(DiscourseIntent(
            type=DiscourseType.CONNECT,
            subject=subject,
            target_concept=self._pick_best_association(associations, seen, exclude_subject=target2 or "") or "people",
            seen_so_far=seen.copy(),
        ))

        return plan

    def _plan_general(self, subject: str, associations: List[Tuple[str, float]],
                       seen: set, qtype: str) -> DiscoursePlan:
        """Default plan for general statements."""
        return self._plan_explain(subject, associations, seen, qtype)

    # ─── Helpers ───

    def _pick_best_association(self, associations: List[Tuple[str, float]],
                                seen: set,
                                exclude_verbs: bool = False,
                                prefer_causal: bool = False,
                                prefer_contrast: bool = False,
                                exclude_subject: str = "") -> str:
        """Pick the best association not already seen."""
        # Minimal closed-class function words that should never be discourse targets.
        # Based on linguistic universals (closed-class items across languages):
        # pronouns, determiners, prepositions, conjunctions, auxiliaries.
        GRAMMATICAL_CONCEPTS = {
            "a", "an", "the", "this", "that", "these", "those",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "us", "them", "my", "your", "his", "its", "our", "their",
            "in", "on", "at", "to", "for", "with", "by", "from", "of", "as",
            "and", "or", "but", "so", "if", "not", "no",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
            "can", "could", "will", "would", "shall", "should", "may", "might", "must",
            "here", "there", "where", "when", "why", "how", "what", "who",
            "very", "too", "also", "just", "only", "still", "yet", "already",
            "up", "down", "out", "off", "on", "over",
        }
        for label, score in associations:
            ll = label.lower()
            if ll in seen:
                continue
            if ll in GRAMMATICAL_CONCEPTS:
                continue
            if exclude_subject and ll == exclude_subject.lower():
                continue
            if exclude_verbs:
                from ravana.language.verb_lexicon import VerbLexicon
                verb_roots = set(VerbLexicon.MORPHEMIC_SEEDS["roots"])
                if ll in verb_roots or any(ll.endswith(s) for s in ("ing", "ed", "es", "ion", "ment")):
                    continue
            return label
        return ""

    def _pick_random_relation(self, associations: List[Tuple[str, float]],
                               seen: set) -> str:
        """Pick any unseen association."""
        for label, _ in associations:
            if label.lower() not in seen:
                return label
        return ""

    def _pick_marker(self, marker_type: str) -> str:
        """Pick a discourse marker for the given type."""
        import random
        markers = self.DISCOURSE_MARKERS.get(marker_type, ["also"])
        if markers:
            return random.choice(markers)
        return ""

    def _generate_follow_up_question(self, subject: str, last_target: str) -> str:
        """Generate a follow-up question from epistemic curiosity.

        Replaces hardcoded templates with a generative approach:
        Questions are composed from:
        - Question type (exploratory, clarifying, connecting) selected
          by how much is known about the subject
        - Question frame (what/how/why) determined by discourse depth
        - Subject or last_target as the focus

        When subject is novel → exploratory questions
        When subject is familiar → clarifying/connecting questions
        When last_target exists → targeted follow-up
        """
        import random

        is_novel = subject.lower() not in [t.lower() for t in self.topic_history[-3:]]

        if is_novel:
            frames = [
                f"What do you think about {subject}?",
                f"Have you experienced {subject} before?",
                f"What aspects of {subject} interest you most?",
                f"How does {subject} relate to your experience?",
            ]
        elif last_target:
            frames = [
                f"What about {last_target} — any thoughts?",
                f"Should we explore {last_target} further?",
                f"How does {last_target} connect to what we were discussing?",
            ]
        else:
            frames = [
                f"Would you like to know more about {subject}?",
                f"Does that perspective on {subject} make sense?",
                f"What else comes to mind about {subject}?",
            ]

        return random.choice(frames)

    # ─── State ───

    def get_state(self) -> Dict:
        return {
            'capacity': self.capacity,
            'topic_history': self.topic_history,
        }

    def set_state(self, state: Dict):
        self.capacity = state.get('capacity', 5)
        self.topic_history = state.get('topic_history', [])
