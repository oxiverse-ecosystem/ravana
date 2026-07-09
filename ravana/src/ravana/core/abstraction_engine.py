"""
Abstraction Engine — Multi-Perspective Concept Reflection
==========================================================
Generates multi-sentence reflective responses for abstract concepts
by walking the concept hierarchy (IS_A, HYPERNYM, PART_OF edges).

Neuroscience grounding:
- Anterior temporal lobe (ATL) as hub for abstract concept representation
- Default Mode Network (DMN) for self-referential, reflective processing
- Prefrontal cortex for perspective-taking and meta-cognition

Design:
- Three perspectives: Experiential (what it involves), Social (societal meaning), Reflective (personal)
- Walks abstraction hierarchy upward to find superordinate categories
- Generates discourse intents for PrefrontalWorkspace
"""
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import random

try:
    from ravana_ml.graph import ConceptGraph, ConceptEdge, ConceptNode
except ImportError:
    ConceptGraph = None
    ConceptEdge = None
    ConceptNode = None


@dataclass
class AbstractionPerspective:
    """A single perspective on an abstract concept."""
    name: str  # "experiential", "social", "reflective"
    description: str  # Natural language description
    supporting_concepts: List[str]  # Concepts that support this perspective
    confidence: float


@dataclass
class AbstractionResult:
    """Result of abstract concept analysis."""
    concept: str
    perspectives: List[AbstractionPerspective]
    hierarchy_path: List[str]  # Path up the abstraction hierarchy
    discourse_intents: List[Dict]  # Ready for PrefrontalWorkspace


@dataclass
class AbstractionConfig:
    """Configuration for abstraction engine."""
    max_hierarchy_depth: int = 5
    min_concept_confidence: float = 0.3
    max_supporting_per_perspective: int = 3
    use_vector_similarity: bool = True


class AbstractionEngine:
    """Multi-perspective abstraction engine for reflective reasoning.

    For abstract concepts like "friendship", "justice", "meaning":
    1. Walk IS_A/HYPERNYM hierarchy upward to find superordinates
    2. Generate three perspectives:
       - Experiential: what the concept involves/feels like (semantic/episodic edges)
       - Social: societal/institutional meaning (social/cultural edges)
       - Reflective: personal/epistemic stance (meta-cognitive edges)
    3. Produce discourse intents for multi-sentence generation
    """

    # Abstract concept hierarchy seeds (IS_A relationships)
    ABSTRACTION_HIERARCHY = {
        "friendship": ["relationship", "social bond", "connection", "interpersonal", "human"],
        "love": ["emotion", "feeling", "affection", "attachment", "human experience"],
        "justice": ["principle", "fairness", "morality", "ethics", "social institution"],
        "freedom": ["right", "liberty", "autonomy", "principle", "political concept"],
        "truth": ["property", "fact", "reality", "epistemology", "philosophy"],
        "beauty": ["quality", "aesthetic", "experience", "value", "philosophy"],
        "knowledge": ["understanding", "belief", "epistemology", "cognition", "mental state"],
        "wisdom": ["quality", "judgment", "experience", "virtue", "mental state"],
        "trust": ["belief", "reliance", "social bond", "psychology", "interpersonal"],
        "hope": ["emotion", "expectation", "optimism", "feeling", "mental state"],
        "fear": ["emotion", "response", "survival", "feeling", "biological"],
        "time": ["dimension", "continuum", "physics", "metaphysics", "fundamental"],
        "meaning": ["property", "significance", "purpose", "semantics", "philosophy"],
        "life": ["state", "biology", "existence", "process", "fundamental"],
        "death": ["state", "end", "biology", "transition", "fundamental"],
        "consciousness": ["state", "awareness", "mind", "phenomenology", "fundamental"],
        "identity": ["property", "self", "continuity", "psychology", "philosophy"],
        "culture": ["system", "shared", "social", "anthropology", "human"],
        "power": ["capacity", "influence", "control", "politics", "social"],
        "responsibility": ["obligation", "duty", "ethics", "agency", "moral"],
        "morality": ["system", "principles", "ethics", "philosophy", "social"],
        "ethics": ["philosophy", "principles", "morality", "study", "academic"],
        "science": ["method", "knowledge", "system", "epistemology", "institution"],
        "art": ["expression", "creativity", "aesthetic", "culture", "human"],
        "history": ["record", "narrative", "past", "study", "academic"],
        "society": ["group", "organization", "social", "collective", "human"],
        "language": ["system", "communication", "symbolic", "cognitive", "human"],
        "memory": ["process", "storage", "cognition", "mental", "biological"],
        "learning": ["process", "acquisition", "change", "cognitive", "adaptive"],
        "understanding": ["state", "comprehension", "cognition", "mental", "epistemic"],
        "belief": ["state", "conviction", "cognitive", "mental", "epistemic"],
        "experience": ["event", "lived", "phenomenology", "subjective", "personal"],
    }

    # Perspective-defining edge types and keywords
    PERSPECTIVE_CONFIG = {
        "experiential": {
            "edge_types": ["semantic", "episodic", "causal", "temporal"],
            "keywords": ["feel", "involve", "experience", "sense", "moment", "when", "like"],
            "question_frames": ["what it feels like", "what it involves", "the experience of"],
        },
        "social": {
            "edge_types": ["semantic", "causal", "analogical"],
            "keywords": ["society", "people", "community", "culture", "institution", "shared", "between"],
            "question_frames": ["what it means in society", "how it shapes relationships", "its social role"],
        },
        "reflective": {
            "edge_types": ["semantic", "analogical", "pragmatic"],
            "keywords": ["think", "believe", "mean", "understand", "perspective", "view", "consider"],
            "question_frames": ["what it means to me", "how I understand it", "my perspective on"],
        },
    }

    def __init__(self, graph: ConceptGraph, config: Optional[AbstractionConfig] = None):
        self.graph = graph
        self.config = config or AbstractionConfig()

    def analyze_abstract_concept(self, concept: str) -> AbstractionResult:
        """Analyze an abstract concept and generate multi-perspective reflection."""
        # Step 1: Find concept node
        node = self._find_node(concept)
        if not node:
            return self._fallback_analysis(concept)

        # Step 2: Walk abstraction hierarchy
        hierarchy_path = self._walk_hierarchy(concept)

        # Step 3: Generate three perspectives
        perspectives = []
        for perspective_name in ["experiential", "social", "reflective"]:
            perspective = self._generate_perspective(concept, node, perspective_name, hierarchy_path)
            if perspective:
                perspectives.append(perspective)

        # Step 4: Create discourse intents
        discourse_intents = self._create_discourse_intents(concept, perspectives)

        return AbstractionResult(
            concept=concept,
            perspectives=perspectives,
            hierarchy_path=hierarchy_path,
            discourse_intents=discourse_intents
        )

    def _walk_hierarchy(self, concept: str) -> List[str]:
        """Walk IS_A/HYPERNYM edges upward to find superordinates."""
        path = [concept]
        current = concept.lower()

        # First check seeded hierarchy
        if current in self.ABSTRACTION_HIERARCHY:
            path.extend(self.ABSTRACTION_HIERARCHY[current][:self.config.max_hierarchy_depth])

        # Then check graph edges for IS_A/HYPERNYM relations
        node = self._find_node(concept)
        if node:
            for depth in range(self.config.max_hierarchy_depth):
                # Find outgoing edges that are hierarchical (semantic with superordinate labels)
                for (src, tgt), edge in list(self.graph.edges.items()):
                    if src == node.id:
                        tgt_node = self.graph.nodes.get(tgt)
                        if tgt_node and edge.relation_type in ("semantic", "inferred"):
                            # Check if target is more abstract (higher level or abstract type)
                            if tgt_node.level > node.level or tgt_node.label.lower() not in path:
                                if tgt_node.label.lower() not in path:
                                    path.append(tgt_node.label)
                                    node = tgt_node
                                    break

        return path[:self.config.max_hierarchy_depth + 1]

    def _generate_perspective(self, concept: str, node: ConceptNode,
                              perspective_name: str, hierarchy_path: List[str]) -> Optional[AbstractionPerspective]:
        """Generate a single perspective on the concept."""
        config = self.PERSPECTIVE_CONFIG[perspective_name]

        # Find supporting concepts via spreading activation on relevant edge types
        supporting = self._find_supporting_concepts(node, config["edge_types"])

        # Filter by relevance
        supporting = [s for s in supporting if s.lower() != concept.lower()][:self.config.max_supporting_per_perspective]

        if not supporting:
            # Fallback to hierarchy-based supporting concepts
            supporting = hierarchy_path[1:4] if len(hierarchy_path) > 1 else []

        # Generate description
        description = self._generate_perspective_description(concept, perspective_name, supporting, hierarchy_path)

        # Confidence based on number and quality of supporting concepts
        confidence = min(1.0, len(supporting) / self.config.max_supporting_per_perspective * 0.8 + 0.2)

        return AbstractionPerspective(
            name=perspective_name,
            description=description,
            supporting_concepts=supporting,
            confidence=confidence
        )

    def _find_supporting_concepts(self, node: ConceptNode, edge_types: List[str]) -> List[str]:
        """Find concepts connected via specified edge types."""
        supporting = []

        for (src, tgt), edge in list(self.graph.edges.items()):
            if src == node.id and edge.relation_type in edge_types:
                tgt_node = self.graph.nodes.get(tgt)
                if tgt_node and tgt_node.activation > 0.01:
                    supporting.append(tgt_node.label)

        # Also check incoming edges
        for (src, tgt), edge in list(self.graph.edges.items()):
            if tgt == node.id and edge.relation_type in edge_types:
                src_node = self.graph.nodes.get(src)
                if src_node and src_node.activation > 0.01:
                    supporting.append(src_node.label)

        return list(set(supporting))

    def _generate_perspective_description(self, concept: str, perspective_name: str,
                                          supporting: List[str], hierarchy_path: List[str]) -> str:
        """Generate natural language description for a perspective."""
        templates = {
            "experiential": [
                f"{concept} involves {', '.join(supporting[:2])} — it's {hierarchy_path[1] if len(hierarchy_path) > 1 else 'a fundamental experience'}.",
                f"When you experience {concept}, you {', '.join(supporting[:2])}.",
                f"{concept} feels like {', '.join(supporting[:2])} combined.",
            ],
            "social": [
                f"In society, {concept} means {', '.join(supporting[:2])} — it's how we {hierarchy_path[1] if len(hierarchy_path) > 1 else 'relate to each other'}.",
                f"{concept} shapes {', '.join(supporting[:2])} between people.",
                f"Socially, {concept} is about {', '.join(supporting[:2])}.",
            ],
            "reflective": [
                f"I think {concept} is {', '.join(supporting[:2])} — it {hierarchy_path[1] if len(hierarchy_path) > 1 else 'matters deeply'}.",
                f"To me, {concept} means {', '.join(supporting[:2])}.",
                f"Understanding {concept} means seeing {', '.join(supporting[:2])}.",
            ],
        }

        tmpl = random.choice(templates[perspective_name])
        return tmpl

    def _create_discourse_intents(self, concept: str, perspectives: List[AbstractionPerspective]) -> List[Dict]:
        """Create discourse intents for PrefrontalWorkspace."""
        intents = []

        for i, p in enumerate(perspectives):
            # Determine discourse type
            if p.name == "experiential":
                dtype = "explain"
                relation = "semantic"
            elif p.name == "social":
                dtype = "elaborate"
                relation = "semantic"
            else:  # reflective
                dtype = "self_reference"
                relation = "semantic"

            intent = {
                "type": dtype,
                "subject": concept,
                "primary_relation": relation,
                "target_concept": p.supporting_concepts[0] if p.supporting_concepts else "",
                "secondary_concept": p.supporting_concepts[1] if len(p.supporting_concepts) > 1 else "",
                "use_epistemic_hedge": (p.name == "reflective"),
                "end_with_question": (i == len(perspectives) - 1),
                "discourse_marker": self._get_marker(p.name, i),
                "perspective": p.name,
                "description": p.description,
            }
            intents.append(intent)

        return intents

    def _get_marker(self, perspective_name: str, index: int) -> str:
        """Get discourse marker for perspective."""
        markers = {
            "experiential": ["", "furthermore", "in essence"],
            "social": ["", "in society", "collectively"],
            "reflective": ["", "personally", "ultimately"],
        }
        return markers.get(perspective_name, [""])[min(index, 2)]

    def _find_node(self, label: str) -> Optional[ConceptNode]:
        """Find node by label."""
        for node in list(self.graph.nodes.values()):
            if node.label.lower() == label.lower():
                return node
        return None

    def _fallback_analysis(self, concept: str) -> AbstractionResult:
        """Fallback when concept not in graph."""
        perspectives = [
            AbstractionPerspective(
                name="experiential",
                description=f"{concept} is a profound human experience.",
                supporting_concepts=[],
                confidence=0.3
            ),
            AbstractionPerspective(
                name="social",
                description=f"In society, {concept} connects people.",
                supporting_concepts=[],
                confidence=0.3
            ),
            AbstractionPerspective(
                name="reflective",
                description=f"I think {concept} means different things to different people.",
                supporting_concepts=[],
                confidence=0.3
            ),
        ]
        return AbstractionResult(
            concept=concept,
            perspectives=perspectives,
            hierarchy_path=[concept],
            discourse_intents=self._create_discourse_intents(concept, perspectives)
        )


# Convenience function for chat interface
def analyze_abstract_concept(graph: ConceptGraph, concept: str) -> AbstractionResult:
    """Analyze an abstract concept and return multi-perspective result."""
    engine = AbstractionEngine(graph)
    return engine.analyze_abstract_concept(concept)