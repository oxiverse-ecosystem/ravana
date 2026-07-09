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

    def __init__(self, capacity: int = 5, analogy_engine=None, abstraction_engine=None):
        self.capacity = capacity  # teen capacity (adults = 7)
        self.last_plan: Optional[DiscoursePlan] = None
        self.topic_history: List[str] = []  # last 10 topics discussed
        # Optional cognitive engines for advanced reasoning
        self.analogy_engine = analogy_engine
        self.abstraction_engine = abstraction_engine

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
        "hypothetical": [
            re.compile(r"what\s+happens\s+if\s+(.+)", re.IGNORECASE),
            re.compile(r"what\s+would\s+happen\s+if\s+(.+)", re.IGNORECASE),
            re.compile(r"what\s+happens\s+when\s+(.+)", re.IGNORECASE),
        ],
        "do_you_know": [
            re.compile(r"do\s+you\s+know\s+(.+)", re.IGNORECASE),
            re.compile(r"have\s+you\s+heard\s+of\s+(.+)", re.IGNORECASE),
        ],
        "follow_up": [
            re.compile(r"(?:more|else|another|also|further|tell me more)", re.IGNORECASE),
        ],
        "greeting": [
            re.compile(r"\b(hi|hello|hey|yo|sup|greetings|whats\s*up|howdy|good\s*morning|good\s*afternoon|good\s*evening)\b", re.IGNORECASE),
        ],
        "wellbeing": [
            re.compile(r"\b(how\s*are\s*you|how\s*is\s*it\s*going|how\s*are\s*you\s*doing|how\s*have\s*you\s*been|hows\s*it\s*going|hows\s*life)\b", re.IGNORECASE),
        ],
        "capability": [
            re.compile(r"\b(what\s*can\s*you\s*do|what\s*do\s*you\s*do|how\s*do\s*you\s*work|tell\s*me\s*about\s*yourself|who\s*are\s*you|what\s*is\s*your\s*name)\b", re.IGNORECASE),
        ],
        "introduction": [
            re.compile(r"\bmy\s+name\s+is\s+(.+)", re.IGNORECASE),
            re.compile(r"\bi\s+am\s+called\s+(.+)", re.IGNORECASE),
            re.compile(r"\bi\s+am\s+(.+)", re.IGNORECASE),
            re.compile(r"\bi'm\s+(.+)", re.IGNORECASE),
            re.compile(r"\bcall\s+me\s+(.+)", re.IGNORECASE),
        ],
        "farewell": [
            re.compile(r"\b(bye|goodbye|see\s*you|good\s*night|farewell)\b", re.IGNORECASE),
        ],
        "analogy": [
            re.compile(r"(.+?)\s*:\s*(.+?)\s*::\s*(.+?)\s*:\s*(.+)", re.IGNORECASE),
            re.compile(r"(.+?)\s+is\s+to\s+(.+?)\s+as\s+(.+?)\s+is\s+to\s+(.+)", re.IGNORECASE),
            re.compile(r"(.+?)\s+relates?\s+to\s+(.+?)\s+like\s+(.+?)\s+relates?\s+to\s+(.+)", re.IGNORECASE),
            re.compile(r"what\s+is\s+the\s+analogy\s+(?:of|for)\s+(.+?)\s+(?:and|to)\s+(.+)", re.IGNORECASE),
            re.compile(r"(.+?)\s*:\s*(.+?)\s*::\s*(.+?)\s*:\s*(.+)", re.IGNORECASE),
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

    # Maps each detected question type to the primary relation the PFC will
    # use for task-set biasing during activation spread. Mirrors the dispatch
    # logic in plan_discourse — the PFC's architectural decision about what
    # kind of reasoning the current question requires.
    QTYPE_PRIMARY_RELATION = {
        "what_is": "semantic",
        "why": "causal",
        "how": "causal",
        "tell_me": "semantic",
        "compare": "contrastive",
        "hypothetical": "causal",
        "do_you_know": "semantic",
        "follow_up": "semantic",
        "general": "semantic",
        "analogy": "analogical",
    }

    @classmethod
    def get_primary_relation_for_qtype(cls, qtype: str) -> str:
        """Return the PFC's task-set relation for a given question type."""
        return cls.QTYPE_PRIMARY_RELATION.get(qtype, "semantic")

    @classmethod
    def detect_question_type(cls, text: str, concept_pos: Optional[Dict[str, str]] = None) -> Tuple[str, List[str]]:
        """Detect question type from user input.

        Returns:
            (question_type, extracted_parts)
        """
        text_lower = text.lower().strip(" ?!.")

        # Check social/chitchat types first to prevent general pattern hijacking
        social_types = ["greeting", "wellbeing", "capability", "introduction", "farewell"]
        for qtype in social_types:
            if qtype in cls.QUESTION_PATTERNS:
                for pattern in cls.QUESTION_PATTERNS[qtype]:
                    m = pattern.match(text_lower)
                    if m:
                        groups = [g.strip() for g in m.groups() if g]
                        # Extra validation for introduction to avoid state adjectives
                        if qtype == "introduction":
                            name_candidate = groups[0].lower() if groups else ""
                            state_words = {
                                "happy", "sad", "tired", "thinking", "learning", "busy", "ready", "sure", 
                                "fine", "good", "well", "hungry", "sick", "bored", "excited", "doing", 
                                "going", "coming", "trying", "working", "studying", "making", "having"
                            }
                            is_state_or_action = False
                            if name_candidate and concept_pos:
                                pos = concept_pos.get(name_candidate)
                                if pos in ("adj", "verb", "adverb"):
                                    is_state_or_action = True
                            if name_candidate in state_words or is_state_or_action or not name_candidate:
                                continue
                        return (qtype, groups)

        for qtype, patterns in cls.QUESTION_PATTERNS.items():
            if qtype in social_types:
                continue
            for pattern in patterns:
                m = pattern.match(text_lower)
                if m:
                    groups = [g.strip() for g in m.groups() if g]
                    return (qtype, groups)

        # Default: treat as general statement/query
        return ("general", [text_lower])

    @classmethod
    def detect_concept_drift(cls, current_topic: str, next_hop: str,
                              vector_fn=None) -> float:
        """Detect if a graph walk has drifted to an unrelated concept.
        
        Returns a drift score 0.0 (on-topic) to 1.0 (completely unrelated).
        When drift > 0.6, the PFC should intervene (step back or insert transition).
        
        Uses vector similarity between consecutive hops.
        """
        if not vector_fn or not current_topic or not next_hop:
            return 0.0
        try:
            v1 = vector_fn(current_topic)
            v2 = vector_fn(next_hop)
            if v1 is not None and v2 is not None:
                import numpy as np
                sim = float(np.dot(v1, v2))
                # sim ranges from -1 to 1. Drift = 1 - normalized_similarity
                drift = 1.0 - max(0.0, (sim + 1.0) / 2.0)
                return drift
        except Exception:
            pass
        return 0.0

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
        qtype, parts = self.detect_question_type(user_input, concept_pos=concept_pos)
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)

        # Build seen set from subject only. Do NOT pre-populate with top
        # associations — that would consume the most causally-relevant concepts
        # (e.g. "explosion" for "what happens if lamp?") before the
        # task-set-aware selection can evaluate them.
        seen = {subject.lower()}

        # Plan based on question type
        if qtype == "what_is":
            # Check if subject is abstract → use multi-perspective planning
            if self._is_abstract_concept(subject):
                plan = self._plan_abstract(subject, associations, seen, qtype)
            else:
                plan = self._plan_explain(subject, associations, seen, qtype)
        elif qtype == "why":
            plan = self._plan_causal_explain(subject, associations, seen, qtype)
        elif qtype == "tell_me":
            if self._is_abstract_concept(subject):
                plan = self._plan_abstract(subject, associations, seen, qtype)
            else:
                plan = self._plan_elaborate(subject, associations, seen, qtype)
        elif qtype == "compare":
            plan = self._plan_compare(subject, parts, associations, seen, qtype)
        elif qtype == "follow_up":
            plan = self._plan_continue(subject, associations, seen, qtype)
        elif qtype == "hypothetical":
            plan = self._plan_causal_explain(subject, associations, seen, qtype)
        elif qtype == "analogy":
            plan = self._plan_analogy(subject, parts, associations, seen, qtype)
        elif qtype == "do_you_know":
            plan = self._plan_explain(subject, associations, seen, qtype)
        elif qtype in ("greeting", "wellbeing", "capability", "introduction", "farewell"):
            plan = self._plan_social(subject, qtype)
        else:
            if self._is_abstract_concept(subject):
                plan = self._plan_abstract(subject, associations, seen, qtype)
            else:
                plan = self._plan_general(subject, associations, seen, qtype)
 
        # Ensure we have exactly 3 intents (pad if needed, skip for social intents)
        if qtype not in ("greeting", "wellbeing", "capability", "introduction", "farewell"):
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
    
    def _plan_social(self, subject: str, qtype: str) -> DiscoursePlan:
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)
        plan.intents.append(DiscourseIntent(
            type="social",
            subject=qtype,
            primary_relation="social",
        ))
        return plan

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
        """Plan: [CAUSAL_EXPLAIN → CAUSAL_EXPLAIN → CONNECT]
        
        For 'why' and 'how' questions — produces deeper causal explanations
        with "because" structures rather than just listing associations.
        
        Enhanced with causal chain extraction: walks the causal path from
        seed concept to target concept, generating multi-sentence explanations
        that follow the causal mechanism.
        """
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)

        # Sentence 1: CAUSAL — what causes/creates the subject (cause)
        target1 = self._pick_best_association(associations, seen, prefer_causal=True)
        if not target1:
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

        # Sentence 2: CAUSAL_EXPLAIN — what the subject causes/leads to (effect)
        # Using CAUSAL_EXPLAIN type for "BECAUSE" structures instead of ELABORATE
        target2 = self._pick_best_association(associations, seen, prefer_causal=True, exclude_subject=target1 if target1 else subject)
        if not target2:
            target2 = self._pick_best_association(associations, seen, exclude_subject=target1 if target1 else subject)
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.CAUSAL_EXPLAIN,
            subject=target1 if target1 else subject,
            primary_relation="causal",
            target_concept=target2,
            use_epistemic_hedge=False,
            discourse_marker="because",  # Signal "because" structure
            seen_so_far=seen.copy(),
        ))
        if target2:
            seen.add(target2.lower())

        # Sentence 3: CONNECT — only use CAUSAL edges, not semantic
        # For why/hypothetical, restrict to causal relations
        causal_assocs = [(l, s) for l, s in associations 
                         if self._is_causal_association(l, associations)]
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.CAUSAL_EXPLAIN,
            subject=subject,
            primary_relation="causal",
            target_concept=self._pick_best_association(causal_assocs or associations, seen) or "",
            end_with_question=True,
            seen_so_far=seen.copy(),
        ))

        return plan

    def _is_causal_association(self, label: str, associations: List[Tuple[str, float]]) -> bool:
        """Check if an association is causal rather than semantic."""
        ll = label.lower()
        causal_indicators = ["cause", "effect", "result", "lead", "because", "since",
                            "trigger", "create", "produce", "influence"]
        return any(ind in ll for ind in causal_indicators)

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
        """Plan: [CONTRASTIVE_DIFF → EXPLAIN_A → EXPLAIN_B] — for compare questions
        Uses contrastive parallel activation to compute difference sets."""
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)
        concept_a = parts[0] if len(parts) > 0 else subject
        concept_b = parts[1] if len(parts) > 1 else ""

        # Compute difference sets: unique associations for A, unique for B
        a_unique = []
        b_unique = []
        common = []
        a_lower = concept_a.lower()
        b_lower = concept_b.lower() if concept_b else ""

        if concept_b:
            for label, score in associations:
                ll = label.lower()
                if ll == a_lower or ll == b_lower:
                    continue
                # Simple heuristic: check graph edge types to find what's unique
                # Associations closer to A than B are "unique to A"
                # For a real implementation, we would use bidirectional spread
                if any(hint in label.lower() for hint in [a_lower[:3], ""]) and not any(hint in label.lower() for hint in [b_lower[:3]]):
                    a_unique.append((label, score))
                elif any(hint in label.lower() for hint in [b_lower[:3]]):
                    b_unique.append((label, score))
                else:
                    common.append((label, score))

        # Sentence 1: CONTRAST — highlight the key difference
        diff_target = a_unique[0][0] if a_unique else (b_unique[0][0] if b_unique else common[0][0] if common else "")
        if diff_target:
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.CONTRAST,
                subject=concept_a,
                primary_relation="contrastive",
                target_concept=diff_target,
                secondary_concept=concept_b,
                seen_so_far=seen.copy(),
            ))
            seen.add(diff_target.lower())
        else:
            # Fallback: original behavior
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.EXPLAIN,
                subject=concept_a,
                target_concept=self._pick_best_association(associations, seen),
                seen_so_far=seen.copy(),
            ))

        # Sentence 2: EXPLAIN_A
        target_a = a_unique[0][0] if a_unique else self._pick_best_association(associations, seen)
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.EXPLAIN,
            subject=concept_a,
            target_concept=target_a,
            seen_so_far=seen.copy(),
        ))
        if target_a and target_a.lower() not in seen:
            seen.add(target_a.lower())
        seen.add(concept_a.lower())

        # Sentence 3: EXPLAIN_B or CONNECT
        if concept_b:
            target_b = b_unique[0][0] if b_unique else self._pick_best_association(associations, seen)
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.EXPLAIN,
                subject=concept_b,
                target_concept=target_b,
                seen_so_far=seen.copy(),
            ))
            if concept_b:
                seen.add(concept_b.lower())
        else:
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.CONNECT,
                subject=concept_a,
                target_concept=self._pick_best_association(associations, seen) or "",
                seen_so_far=seen.copy(),
            ))

        return plan

    def _plan_analogy(self, subject: str, parts: List[str],
                        associations: List[Tuple[str, float]],
                        seen: set, qtype: str,
                        vector_fn=None) -> DiscoursePlan:
        """Plan: [EXPLAIN_RELATION → CANDIDATE → CONNECT] — for A:B::C:___ analogies.

        Uses the AnalogyEngine for structure mapping:
        1. Extract relation between A and B from graph edges
        2. Find concepts that have the SAME relation with C
        3. Score candidates by relation vector similarity
        4. Generate explanation of the analogy
        """
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)

        # Parse parts: A from parts[0], B from parts[1], C from parts[2]
        concept_a = parts[0].strip() if len(parts) > 0 else subject
        concept_b = parts[1].strip() if len(parts) > 1 else ""
        concept_c = parts[2].strip() if len(parts) > 2 else ""

        # Use AnalogyEngine if available
        candidate = ""
        if self.analogy_engine and concept_a and concept_b and concept_c:
            try:
                best_d = self.analogy_engine.get_best_completion(concept_a, concept_b, concept_c)
                if best_d:
                    candidate = best_d
            except Exception:
                pass

        # Fallback to heuristic
        if not candidate and concept_c:
            for label, score in associations:
                if label.lower() != concept_c.lower() and label.lower() not in seen:
                    candidate = label
                    break

        # Sentence 1: Explain the A:B relation
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.EXPLAIN,
            subject=concept_a,
            primary_relation="analogical",
            target_concept=concept_b,
            seen_so_far=seen.copy(),
        ))
        if concept_b:
            seen.add(concept_b.lower())

        # Sentence 2: Propose the C:D relation as analogy
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.CAUSAL_EXPLAIN,
            subject=concept_c if concept_c else concept_a,
            primary_relation="analogical",
            target_concept=candidate,
            discourse_marker="similarly",
            seen_so_far=seen.copy(),
        ))
        if candidate:
            seen.add(candidate.lower())

        # Sentence 3: CONNECT with question
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.CONNECT,
            subject=concept_a,
            primary_relation="semantic",
            target_concept=candidate or "",
            end_with_question=True,
            seen_so_far=seen.copy(),
        ))

        return plan

    def _plan_abstract(self, subject: str, associations: List[Tuple[str, float]],
                       seen: set, qtype: str) -> DiscoursePlan:
        """Plan: [EXPERIENTIAL → SOCIAL → REFLECTIVE] — for abstract concepts.

        Uses the AbstractionEngine for multi-perspective reflection:
        1. Experiential: what the concept involves/feels like
        2. Social: what it means in society/relationships
        3. Reflective: personal/epistemic reflection
        """
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)

        # Use AbstractionEngine if available
        if self.abstraction_engine:
            try:
                result = self.abstraction_engine.analyze_abstract_concept(subject)
                # Convert discourse intents to plan intents
                for i, intent_data in enumerate(result.discourse_intents):
                    plan.intents.append(DiscourseIntent(
                        type=intent_data.get("type", DiscourseType.EXPLAIN),
                        subject=intent_data.get("subject", subject),
                        primary_relation=intent_data.get("primary_relation", "semantic"),
                        target_concept=intent_data.get("target_concept", ""),
                        secondary_concept=intent_data.get("secondary_concept", ""),
                        use_epistemic_hedge=intent_data.get("use_epistemic_hedge", False),
                        end_with_question=intent_data.get("end_with_question", False),
                        discourse_marker=intent_data.get("discourse_marker", ""),
                        seen_so_far=seen.copy(),
                    ))
                    if intent_data.get("target_concept"):
                        seen.add(intent_data["target_concept"].lower())
                    if intent_data.get("secondary_concept"):
                        seen.add(intent_data["secondary_concept"].lower())
                return plan
            except Exception:
                pass  # Fall back to heuristic

        # Fallback heuristic
        # Sentence 1: Experiential — what the concept involves
        target1 = self._pick_best_association(associations, seen, exclude_verbs=True)
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.EXPLAIN,
            subject=subject,
            primary_relation="semantic",
            target_concept=target1,
            use_epistemic_hedge=True,  # "It seems like..."
            seen_so_far=seen.copy(),
        ))
        if target1:
            seen.add(target1.lower())

        # Sentence 2: Social — broader context/perspective
        target2 = self._pick_best_association(associations, seen)
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.ELABORATE,
            subject=target1 if target1 else subject,
            primary_relation="semantic",
            target_concept=target2,
            discourse_marker="in society",
            seen_so_far=seen.copy(),
        ))
        if target2:
            seen.add(target2.lower())

        # Sentence 3: Reflective — personal/epistemic reflection
        plan.intents.append(DiscourseIntent(
            type=DiscourseType.SELF_REFERENCE,
            subject=subject,
            primary_relation="semantic",
            target_concept=self._pick_best_association(associations, seen) or "",
            use_epistemic_hedge=True,
            end_with_question=True,
            seen_so_far=seen.copy(),
        ))

        return plan

    def _plan_continue(self, subject: str, associations: List[Tuple[str, float]],
                        seen: set, qtype: str) -> DiscoursePlan:
        """Plan: [CONTINUE → ELABORATE → CONNECT] — for follow-ups.

        Three-layer fallback for target selection:
        1. Try _pick_best_association (highest-scoring unseen)
        2. If all seen, try _pick_random_relation (random exploration)
        3. If even that fails, generate a fresh question to the user
        """
        plan = DiscoursePlan(original_subject=subject, question_type=qtype)

        # --- Sentence 1: CONTINUE ---
        target1 = self._pick_best_association(associations, seen)
        if not target1:
            target1 = self._pick_random_relation(associations, seen)

        if target1:
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.CONTINUE,
                subject=subject,
                target_concept=target1,
                seen_so_far=seen.copy(),
            ))
            seen.add(target1.lower())
        else:
            # All associations exhausted — generate a question instead
            question = self._generate_follow_up_question(subject, target_concept=None)
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.ASK_BACK,
                subject=subject,
                target_concept=question,
                end_with_question=True,
                primary_relation="interrogative",
                seen_so_far=seen.copy(),
            ))

        # --- Sentence 2: ELABORATE ---
        if target1:
            target2 = self._pick_best_association(associations, seen, exclude_subject=target1)
            if not target2:
                target2 = self._pick_random_relation(associations, seen)

            if target2:
                plan.intents.append(DiscourseIntent(
                    type=DiscourseType.ELABORATE,
                    subject=subject,
                    target_concept=target2,
                    seen_so_far=seen.copy(),
                ))
                seen.add(target2.lower())

        # --- Sentence 3: CONNECT or ASK_BACK ---
        if target1 and target2:
            target3 = self._pick_best_association(associations, seen, exclude_subject=target2)
            if not target3:
                target3 = self._pick_random_relation(associations, seen)
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.CONNECT,
                subject=subject,
                target_concept=target3 or "people",
                end_with_question=(not target3),
                seen_so_far=seen.copy(),
            ))
        elif target1 and not target2:
            # Only have one topic — ask about it
            question = self._generate_follow_up_question(subject, target1)
            plan.intents.append(DiscourseIntent(
                type=DiscourseType.ASK_BACK,
                subject=target1,
                target_concept=question,
                end_with_question=True,
                primary_relation="interrogative",
                seen_so_far=seen.copy(),
            ))

        return plan

    def _plan_general(self, subject: str, associations: List[Tuple[str, float]],
                       seen: set, qtype: str) -> DiscoursePlan:
        """Default plan for general statements."""
        return self._plan_explain(subject, associations, seen, qtype)

    # ─── Helpers ───

    # ABSTRACT_NOUNS replaced by GloVe-based _is_abstract_concept
    ABSTRACT_NOUNS: set = set()  # Deprecated - kept for back-compat

    def _is_abstract_concept(self, subject: str) -> bool:
        """Check if a subject is abstract using GloVe-based classifier.

        ATL computes abstractness from semantic neighborhood (Cousins 2017),
        not a stored list. Falls back to suffix heuristics when GloVe unavailable.
        """
        if not subject:
            return False
        sl = subject.lower().strip()
        try:
            from ravana.language.verb_lexicon import _default_vector_fn
            fn = _default_vector_fn
            vec = fn(sl)
            if vec is not None:
                import numpy as np
                abstract_protos = ["love", "truth", "knowledge", "idea", "meaning", "beauty"]
                concrete_protos = ["table", "dog", "mountain", "car", "tree", "house"]
                abs_sims = [float(np.dot(vec, fn(p))) for p in abstract_protos if fn(p) is not None]
                con_sims = [float(np.dot(vec, fn(p))) for p in concrete_protos if fn(p) is not None]
                if abs_sims and con_sims:
                    return float(np.mean(abs_sims)) > float(np.mean(con_sims))
        except Exception:
            pass
        if sl.endswith('ness') or sl.endswith('ity') or sl.endswith('tion') or sl.endswith('ism'):
            return True
        if sl.endswith('ment') or sl.endswith('ance') or sl.endswith('ence'):
            return True
        if sl.endswith('ship') or sl.endswith('dom') or sl.endswith('hood'):
            return True
        return False

    def _pick_best_association(self, associations: List[Tuple[str, float]],
                                   seen: set,
                                   exclude_verbs: bool = False,
                                   exclude_subject: Optional[str] = None,
                                   prefer_causal: bool = False,
                                   prefer_contrast: bool = False) -> Optional[str]:
        """Pick the best association from a scored list, respecting constraints.

        Selects the highest-scoring item from associations that:
        - Is not already in the seen set
        - Is not a verb (if exclude_verbs=True)
        - Is not the exclude_subject
        - Prioritizes causal/contrastive relations if preferred

        Returns the label string, or None if no valid association found.
        """
        if not associations:
            return None

        # Filter by constraints
        candidates = []
        for label, score in associations:
            ll = label.lower().strip()
            if ll in seen:
                continue
            if exclude_verbs and self._is_verb_label(ll):
                continue
            if exclude_subject and ll == exclude_subject.lower().strip():
                continue
            candidates.append((label, score))

        if not candidates:
            return None

        # Apply preference boosts
        boosted = []
        for label, score in candidates:
            ll = label.lower()
            boost = 1.0
            if prefer_causal or prefer_contrast:
                if prefer_causal and self._has_causal_hint(ll):
                    boost *= 1.5
                if prefer_contrast and self._has_contrast_hint(ll):
                    boost *= 1.5
            boosted.append((label, score * boost))

        # Sort by boosted score descending and return best
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted[0][0]

    def _is_verb_label(self, label: str) -> bool:
        """Check if a label is likely a verb."""
        ll = label.lower()
        verb_suffixes = ("ing", "ed", "en", "ify", "ize", "ate", "ish")
        verb_forms = {
            "be", "do", "have", "make", "take", "give", "get", "go", "come",
            "see", "know", "think", "feel", "say", "tell", "ask", "use",
            "find", "want", "seem", "need", "help", "work", "call", "try",
            "leave", "keep", "let", "begin", "show", "hear", "play", "run",
            "move", "live", "believe", "hold", "bring", "happen", "write",
            "provide", "sit", "stand", "lose", "pay", "meet", "include",
            "continue", "set", "learn", "change", "lead", "understand",
            "watch", "follow", "stop", "create", "cause", "let", "mean",
            "exist", "form", "act", "result", "produce", "connect", "relate",
        }
        if ll in verb_forms:
            return True
        if any(ll.endswith(s) for s in verb_suffixes):
            return True
        return False

    def _has_causal_hint(self, label: str) -> bool:
        """Check if a label has causal semantics."""
        ll = label.lower()
        causal_words = {
            "cause", "effect", "result", "consequence", "impact", "influence",
            "lead", "lead", "trigger", "produce", "create", "make", "generate",
            "force", "drive", "push", "enable", "allow", "prevent", "block",
            "because", "since", "hence", "therefore", "reaction", "response",
        }
        return ll in causal_words or any(ll.startswith(w) for w in causal_words)

    def _has_contrast_hint(self, label: str) -> bool:
        """Check if a label has contrastive semantics."""
        ll = label.lower()
        contrast_words = {
            "but", "however", "although", "though", "yet", "nevertheless",
            "contrast", "opposite", "different", "vs", "versus", "unlike",
            "instead", "rather", "still", "while", "whereas", "conversely",
            "on the other hand", "difference", "conflict", "against",
        }
        return ll in contrast_words

    def _pick_random_relation(self, associations: List[Tuple[str, float]],
                               seen: set) -> Optional[str]:
        """Pick a random unseen relation from associations.

        Basal ganglia analog: when no directed selection is possible,
        a random exploration step is used (Go/NoGo pathway).
        """
        if not associations:
            return None
        import random
        unseen = [(l, s) for l, s in associations if l.lower().strip() not in seen]
        if not unseen:
            return None
        return random.choice(unseen)[0]

    def _generate_follow_up_question(self, subject: str,
                                      target_concept: Optional[str] = None) -> str:
        """Generate a follow-up question to engage the user.

        Composes a context-appropriate question from primitives.
        Inspired by the DMN's social-cognitive questioning reflex.
        """
        import random
        if not target_concept:
            questions = [
                f"what do you think about {subject}?",
                f"have you experienced {subject} yourself?",
                f"what aspects of {subject} interest you?",
                f"would you like to explore more about {subject}?",
            ]
        else:
            questions = [
                f"does that match your understanding of {target_concept}?",
                f"have you noticed this about {target_concept}?",
                f"what is your perspective on {target_concept}?",
                f"how does {target_concept} relate to your experience?",
                f"would you like to know more about {target_concept}?",
            ]
        return random.choice(questions)

    def _pick_marker(self, marker_type: str) -> str:
        """Pick a discourse marker for the given type.

        Selects from the DISCOURSE_MARKERS dict. If the type has markers,
        returns the first one and rotates the list for next time.
        Falls back to empty string.
        """
        import random
        markers = self.DISCOURSE_MARKERS.get(marker_type, [])
        if not markers:
            return ""
        return random.choice(markers)

    def get_state(self) -> Dict:
        return {
            'capacity': self.capacity,
            'topic_history': self.topic_history,
        }

    def set_state(self, state: Dict):
        self.capacity = state.get('capacity', 5)
        self.topic_history = state.get('topic_history', [])
