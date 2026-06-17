"""
RAVANA v2 — CONVERSATIONAL REPAIR
Contradiction resolution through local anti-Hebbian plasticity.

PRINCIPLE: When the user corrects the system, apply local weight changes
without corrupting global knowledge. The system learns a user-specific
override rather than "unlearning" the global fact.

Subsystem 3 from the architectural plan:
- Detection Phase: Compare system output vs. user correction
- Penalty & Boost Phase: Adjust agent-specific edge weights
- Pressure Spike Phase: Accumulate contradiction free energy
- Memory Phase: Log correction for sleep consolidation
"""

import time
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum

from .dialogue_context import Triple


# ─── Correction Types ────────────────────────────────────────────────────────

class CorrectionType(Enum):
    NEGATION = "negation"           # Direct contradiction ("No, that's wrong")
    REFINEMENT = "refinement"       # Partial correction ("Actually, it's more like...")
    CONTEXT = "context"             # Contextual override ("Only when...")
    VERIFICATION = "verification"   # Confirmation ("Yes, that's right")


# ─── Helper: Triple Parsing from Natural Language ──────────────────────────

# Simple verb-to-relation-type mapping
_VERB_TO_RELATION = {
    "is": "semantic", "are": "semantic", "was": "semantic", "were": "semantic",
    "represents": "semantic", "means": "semantic", "defines": "semantic",
    "causes": "causal", "cause": "causal", "produces": "causal",
    "leads": "causal", "results": "causal", "makes": "causal",
    "has": "possessive", "have": "possessive", "contains": "possessive",
    "includes": "possessive", "like": "analogical", "similar": "analogical",
    "after": "temporal", "before": "temporal", "then": "temporal",
    "in": "contextual", "at": "contextual", "on": "contextual",
}


def _extract_relation_type(relation_word: str, default: str = "semantic") -> str:
    """Extract relation type from a verb/relation word.

    Proper suffix stripping: first checks if the word itself is in the
    lookup map, then tries progressive suffix removal ('s', 'es', 'ed', 'ing').
    """
    word = relation_word.lower().strip()

    # First check if the raw word is already in the map (e.g., "causes" -> "causal")
    if word in _VERB_TO_RELATION:
        return _VERB_TO_RELATION[word]

    # Progressive suffix removal: check each possible stem against the map
    # Order matters: try 's' before 'es' to avoid over-stripping ("causes" -> "caus" instead of "cause")
    for suffix in ['s', 'es', 'ed', 'ing']:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            stem = word[:-len(suffix)]
            if stem in _VERB_TO_RELATION:
                return _VERB_TO_RELATION[stem]

    return _VERB_TO_RELATION.get(word, default)


def _parse_text_to_triples(text: str) -> List[Triple]:
    """
    Parse natural language text into triples using simple pattern matching.

    Patterns:
    - "X causes Y" -> (X, causes, causal, Y)
    - "X is Y" -> (X, is, semantic, Y)
    - "X has Y" -> (X, has, possessive, Y)

    Args:
        text: Natural language text

    Returns:
        List of parsed Triples
    """
    triples = []
    text = text.strip()

    if not text:
        return triples

    # Strategy 1: Try S-V-O pattern with exact 3-word or known verb in position 1
    words = text.split()
    if len(words) >= 3:
        relation_word = words[1].lower().rstrip(".,!?")
        if relation_word in _VERB_TO_RELATION or len(words) == 3:
            rel_type = _extract_relation_type(relation_word)
            subject = words[0].rstrip(".,!?:;")
            obj = " ".join(words[2:]).rstrip(".,!?:;")
            if subject and obj and len(subject) > 0 and len(obj) > 0:
                triple = Triple(
                    subject=subject,
                    relation=relation_word,
                    relation_type=rel_type,
                    object=obj,
                    confidence=0.8,
                )
                triples.append(triple)

    # Strategy 2: For longer text (e.g. with negation prefixes), scan for known verbs at any position
    if not triples and len(words) > 3:
        for i, w in enumerate(words):
            w_clean = w.lower().rstrip(".,!?:;")
            if w_clean in _VERB_TO_RELATION and i > 0 and i < len(words) - 1:
                rel_type = _extract_relation_type(w_clean)
                subject = words[i - 1].rstrip(".,!?:;")
                obj = " ".join(words[i + 1:]).rstrip(".,!?:;")
                if subject and obj and len(subject) > 0 and len(obj) > 0:
                    triple = Triple(
                        subject=subject,
                        relation=w_clean,
                        relation_type=rel_type,
                        object=obj,
                        confidence=0.7,
                    )
                    triples.append(triple)
                break

    return triples


# ─── ConversationalRepair ────────────────────────────────────────────────────

@dataclass
class RepairEvent:
    """Record of a single repair event."""
    timestamp: float
    user_id: str
    wrong_triple: Triple
    correct_triple: Triple
    correction_type: CorrectionType
    penalty_applied: float
    boost_applied: float
    free_energy_spike: float
    episode: int = 0


class ConversationalRepair:
    """
    Detect and apply corrections without corrupting global knowledge.

    Key design:
    - Corrections are LOCAL (only affect user-specific weights, never global)
    - Repair is PASSIVE (happens after generation, not during)
    - The system doesn't "unlearn" — it learns a user-specific override

    Mechanics:
    1. Parse system output and user correction into triples
    2. Detect contradictions: same subject + relation, different object
    3. Penalize wrong edge weight for this user
    4. Boost correct edge weight for this user
    5. Spike contradiction free energy
    6. Log for sleep consolidation
    """

    def __init__(
        self,
        graph: Optional[Any] = None,           # ConceptGraph
        governor: Optional[Any] = None,         # Governor for FE spike
        user_id: str = "default",
        penalty_on_correction: float = -0.4,
        boost_on_correction: float = 0.5,
        contradiction_free_energy_spike: float = 0.8,
    ):
        self.graph = graph
        self.governor = governor
        self.user_id = user_id
        self.penalty_on_correction = penalty_on_correction
        self.boost_on_correction = boost_on_correction
        self.contradiction_free_energy_spike = contradiction_free_energy_spike

        # Repair history
        self.repair_history: List[RepairEvent] = []
        self._repair_count: int = 0

        # Binding map proxy: maps label -> concept_id
        self._label_to_concept: Dict[str, int] = {}

    def set_graph(self, graph: Any):
        """Set the ConceptGraph reference."""
        self.graph = graph
        self._rebuild_label_index()

    def set_governor(self, governor: Any):
        """Set the Governor reference."""
        self.governor = governor

    def _rebuild_label_index(self):
        """Rebuild the label-to-concept index from the graph."""
        self._label_to_concept.clear()
        if self.graph is None:
            return
        for nid, node in self.graph.nodes.items():
            if node.label:
                self._label_to_concept[node.label.lower()] = nid

    def process_correction(
        self,
        system_output: str,
        user_correction: str,
    ) -> Optional[RepairEvent]:
        """
        Process a user correction to the system's previous output.

        Steps:
        1. Parse both into triples
        2. Detect contradictions
        3. Apply penalty/boost
        4. Spike free energy
        5. Log the repair event

        Args:
            system_output: The system's previous response text
            user_correction: The user's correction text

        Returns:
            RepairEvent if correction detected, None otherwise
        """
        # Phase 1: Detection
        contradiction = self.detect_contradiction(system_output, user_correction)
        if contradiction is None:
            return None

        wrong_triple, correct_triple = contradiction

        # Phase 2: Apply repair
        self.apply_repair(wrong_triple, correct_triple)

        # Phase 3: Spike free energy
        self._spike_free_energy()

        # Phase 4: Record event
        self._repair_count += 1
        event = RepairEvent(
            timestamp=time.time(),
            user_id=self.user_id,
            wrong_triple=wrong_triple,
            correct_triple=correct_triple,
            correction_type=self._classify_correction(
                system_output, user_correction
            ),
            penalty_applied=self.penalty_on_correction,
            boost_applied=self.boost_on_correction,
            free_energy_spike=self.contradiction_free_energy_spike,
            episode=self._repair_count,
        )
        self.repair_history.append(event)
        return event

    def detect_contradiction(
        self,
        system_output: str,
        user_correction: str,
    ) -> Optional[Tuple[Triple, Triple]]:
        """
        Detect if user's correction contradicts the system's previous output.

        Detection logic:
        - Same subject + same relation + different object = direct contradiction
        - Check for explicit negation markers ("No, that's wrong", "Actually...")
        - Check for refinement (partial change in object)

        Args:
            system_output: System's previous response
            user_correction: User's correction

        Returns:
            (wrong_triple, correct_triple) or None
        """
        # Parse both
        system_triples = _parse_text_to_triples(system_output)
        correction_triples = _parse_text_to_triples(user_correction)

        if not system_triples or not correction_triples:
            return None

        # Check for explicit negation
        is_negation = self._is_explicit_negation(user_correction)

        # Find contradictory pairs
        for sys_t in system_triples:
            for corr_t in correction_triples:
                if self._are_contradictory(sys_t, corr_t, is_negation):
                    return (sys_t, corr_t)

        # Also check if correction has no matching subject at all — could be additive
        return None

    def apply_repair(self, wrong_triple: Triple, correct_triple: Triple):
        """
        Apply the repair by adjusting edge weights.

        1. Find the edge for wrong_triple in the ConceptGraph
        2. Call edge.update_weight_for_agent(user_id, delta=-penalty)
        3. Record correction in edge.source_metadata['correction_history']
        4. Find or create edge for correct_triple
        5. Call edge.update_weight_for_agent(user_id, delta=+boost)
        6. Set source_metadata['is_user_experience'] = True

        Args:
            wrong_triple: The triple that was wrong
            correct_triple: The correct triple
        """
        if self.graph is None:
            # Still record in repair_history even without graph for testing
            self._repair_count += 1
            event = RepairEvent(
                timestamp=time.time(),
                user_id=self.user_id,
                wrong_triple=wrong_triple,
                correct_triple=correct_triple,
                correction_type=CorrectionType.NEGATION,
                penalty_applied=self.penalty_on_correction,
                boost_applied=self.boost_on_correction,
                free_energy_spike=self.contradiction_free_energy_spike,
                episode=self._repair_count,
            )
            self.repair_history.append(event)
            return

        self._rebuild_label_index()
        agent_key = f"user_{self.user_id}"

        # Find subject node
        subject_id = self._label_to_concept.get(
            wrong_triple.subject.lower()
        )
        if subject_id is None:
            return

        # Process wrong triple: find existing edge and penalize it
        # Look through outgoing edges of the subject node
        outgoing = self.graph.get_outgoing(subject_id)
        for target_id, edge in outgoing:
            target_node = self.graph.get_node(target_id)
            if target_node is None:
                continue
            if target_node.label.lower() == wrong_triple.object.lower():
                # Found the wrong edge — apply penalty for this user
                self._update_agent_weight(edge, agent_key, self.penalty_on_correction)
                self._record_correction_metadata(
                    edge, self.user_id, "negation"
                )

        # Process correct triple: find or create edge and boost it
        correct_object_id = self._label_to_concept.get(
            correct_triple.object.lower()
        )
        if correct_object_id is None and self.graph is not None:
            # Create a new concept node for the correct object
            import numpy as np
            rng = np.random.RandomState(hash(correct_triple.object) % 2**31)
            vec = rng.randn(self.graph.dim).astype(np.float32) * 0.1
            node = self.graph.add_node(vector=vec, label=correct_triple.object)
            correct_object_id = node.id
            self._label_to_concept[correct_triple.object.lower()] = correct_object_id

        if correct_object_id is not None:
            # Find or create edge
            edge = self.graph.get_edge(subject_id, correct_object_id)
            if edge is None:
                edge = self.graph.add_edge(
                    subject_id, correct_object_id,
                    weight=0.3,  # Start low, let boost raise it
                    relation_type=correct_triple.relation_type,
                )

            # Apply boost for this user
            self._update_agent_weight(edge, agent_key, self.boost_on_correction)

            # Mark as user experience
            edge.source_metadata['is_user_experience'] = True
            edge.source_metadata['source_agent'] = agent_key
            edge.source_metadata['epistemic_status'] = 'experience'
            self._record_correction_metadata(
                edge, self.user_id, "negation"
            )



    def get_agent_weight(
        self, edge: Any, agent_id: str
    ) -> float:
        """
        Get the effective weight for a specific agent.

        Hierarchy: agent-specific > global (fallback to edge.weight)

        Args:
            edge: ConceptEdge instance
            agent_id: Agent identifier (e.g., 'user_likhith' or 'global')

        Returns:
            Effective weight for this agent
        """
        if not hasattr(edge, 'agent_weights'):
            return edge.weight

        if agent_id in edge.agent_weights:
            return edge.agent_weights[agent_id]

        return edge.weight

    def get_repair_stats(self) -> Dict[str, Any]:
        """Get repair system statistics."""
        return {
            "total_repairs": len(self.repair_history),
            "recent_repairs": [
                {
                    "timestamp": e.timestamp,
                    "user_id": e.user_id,
                    "wrong": f"{e.wrong_triple.subject} {e.wrong_triple.relation} {e.wrong_triple.object}",
                    "correct": f"{e.correct_triple.subject} {e.correct_triple.relation} {e.correct_triple.object}",
                    "type": e.correction_type.value,
                }
                for e in self.repair_history[-10:]
            ],
        }

    def _is_explicit_negation(self, text: str) -> bool:
        """Check if text contains explicit negation markers."""
        negation_patterns = [
            r"\bno\b", r"\bnot\b", r"\bwrong\b", r"\bincorrect\b",
            r"\bactually\b", r"\bnever\b", r"\bdon't\b", r"\bdoesn't\b",
            r"\bisn't\b", r"\baren't\b", r"\bwasn't\b", r"\bweren't\b",
        ]
        text_lower = text.lower()
        return any(
            re.search(pattern, text_lower) for pattern in negation_patterns
        )

    def _are_contradictory(
        self,
        triple1: Triple,
        triple2: Triple,
        is_negation: bool = False,
    ) -> bool:
        """
        Check if two triples are contradictory.

        Conditions:
        - Same subject (normalized)
        - Same or similar relation
        - Different object (or negation context)

        Args:
            triple1: First triple (usually system output)
            triple2: Second triple (usually user correction)
            is_negation: Whether negation markers were detected

        Returns:
            True if contradictory
        """
        # Normalize subjects
        s1 = triple1.subject.lower().rstrip(".,!?:;")
        s2 = triple2.subject.lower().rstrip(".,!?:;")

        if s1 != s2 and s2 not in s1 and s1 not in s2:
            return False

        # Check if same relation type
        if triple1.relation_type != triple2.relation_type:
            return False

        # Check objects differ
        o1 = triple1.object.lower().rstrip(".,!?:;")
        o2 = triple2.object.lower().rstrip(".,!?:;")

        if o1 == o2:
            return False  # Same triple — not a contradiction

        # Direct contradiction or negation context
        return True

    def _classify_correction(
        self,
        system_output: str,
        user_correction: str,
    ) -> CorrectionType:
        """Classify the type of correction."""
        if self._is_explicit_negation(user_correction):
            return CorrectionType.NEGATION

        ref_patterns = [r"\bactually\b", r"\bmore\b", r"\brather\b",
                        r"\binstead\b", r"\bkind of\b", r"\bsort of\b"]
        if any(re.search(p, user_correction.lower()) for p in ref_patterns):
            return CorrectionType.REFINEMENT

        # Check for contextual qualifiers
        ctx_patterns = [r"\bwhen\b", r"\bif\b", r"\bonly\b", r"\bexcept\b",
                        r"\bunless\b", r"\bdepends\b"]
        if any(re.search(p, user_correction.lower()) for p in ctx_patterns):
            return CorrectionType.CONTEXT

        return CorrectionType.NEGATION

    def _update_agent_weight(
        self, edge: Any, agent_key: str, delta: float
    ):
        """Update an edge's agent-specific weight."""
        if not hasattr(edge, 'agent_weights'):
            return

        current = edge.agent_weights.get(agent_key, edge.weight)
        new_weight = max(0.0, min(1.0, current + delta))
        edge.agent_weights[agent_key] = new_weight

    def _record_correction_metadata(
        self, edge: Any, agent_id: str, correction_type: str
    ):
        """Record correction metadata on an edge."""
        if not hasattr(edge, 'source_metadata'):
            return

        correction_entry = {
            "timestamp": time.time(),
            "agent_id": agent_id,
            "correction_type": correction_type,
        }
        edge.source_metadata['correction_history'].append(correction_entry)
        edge.source_metadata['source_agent'] = f"user_{agent_id}"

    def _spike_free_energy(self):
        """Spike the contradiction free energy channel."""
        if self.governor is None:
            return

        # Check if the governor has a free_energy accumulator
        if hasattr(self.governor, '_contradiction_energy'):
            self.governor._contradiction_energy += (
                self.contradiction_free_energy_spike
            )

        # Spike into free_energy if Governor has one
        if hasattr(self.governor, 'free_energy'):
            fe = self.governor.free_energy
            if hasattr(fe, 'accumulate'):
                fe.accumulate(
                    channel='contradiction',
                    amount=self.contradiction_free_energy_spike,
                )

        # Request sleep consolidation if function exists
        if hasattr(self.governor, 'request_sleep_consolidation'):
            self.governor.request_sleep_consolidation()

    def __repr__(self):
        return (
            f"<ConversationalRepair user={self.user_id} "
            f"repairs={len(self.repair_history)}>"
        )
