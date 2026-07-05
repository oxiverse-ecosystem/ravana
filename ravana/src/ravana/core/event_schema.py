"""
RAVANA Event Schema Module — Hippocampal Event Schemas for Narrative Flow
=========================================================================
Stores and retrieves procedural/sequential knowledge chains that describe
how concepts unfold as processes, not just static relationships.

Neuroscience grounding:
- Schank & Abelson (1977): Scripts are structured representations of
  stereotypical event sequences (e.g., "restaurant script": enter → order
  → eat → pay → leave).
- Zacks & Tversky (2001): Event Segmentation Theory — the brain
  segments continuous experience into discrete events at boundaries,
  then stores them as hierarchical event schemas.
- Baldassano et al. (2017): Event boundaries are tracked in the
  posterior medial (PM) network; event schemas are stored in the
  anterior hippocampus and replayed during consolidation.
- Hassabis & Maguire (2009): Scene construction — the hippocampus
  retrieves episodic details and recombines them into coherent
  mental scenes for imagination and future planning.

Architecture:
- EventSchema: A sequence of steps (concept, process_verb, description)
- ProcessChain: A linear sequence of concepts linked by causal/temporal
  relations, describing how a process unfolds
- Discovered from web text by identifying temporal/causal chains
- Used by the situation model to generate narrative responses that
  describe processes rather than just static relations

Example schemas:
  trust → build → strengthen → bond
  knowledge → gather → understand → apply → wisdom
  love → meet → connect → grow → commit
"""

import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class ProcessStep:
    """A single step in a process chain."""
    concept: str                # The concept involved at this step
    verb: str                   # What happens (builds, forms, creates)
    description: str = ""       # Optional elaboration
    duration: str = ""          # "gradually", "over time", "immediately"
    causal_precedents: List[str] = field(default_factory=list)  # What leads to this step
    confidence: float = 0.5     # How confident we are this step is correct


@dataclass
class EventSchema:
    """A complete event/process schema — a structured sequence of steps."""
    name: str                   # Schema name (matches concept label)
    steps: List[ProcessStep]    # Ordered sequence of steps
    source: str = "discovered"  # "seed", "web", "inference"
    context: str = ""           # When/how this schema applies
    confidence: float = 0.5     # Overall schema confidence


class EventSchemaLibrary:
    """Library of event/process schemas for narrative generation.

    Stores procedural knowledge as (concept → EventSchema) mappings.
    Schemas can be:
    - Seeded with universal process priors (trust builds over time)
    - Discovered from web text by identifying causal/temporal chains
    - Combined from existing schemas (schema blending)
    """

    def __init__(self):
        # concept.lower() → EventSchema
        self._schemas: Dict[str, EventSchema] = {}
        self._seeded = False

    def seed_default_schemas(self):
        """Seed with default universal process schemas.

        These represent common event sequences that are true across
        most languages and cultures — analogous to innate cognitive
        priors for event structure (Baldassano et al., 2017).
        """
        if self._seeded:
            return
        self._seeded = True

        default_schemas = {
            "trust": EventSchema(
                name="trust development",
                steps=[
                    ProcessStep("trust", "is earned", "through consistent actions over time", "gradually"),
                    ProcessStep("vulnerability", "requires", "both parties to be open", "naturally"),
                    ProcessStep("bond", "strengthens", "creating a deeper connection", "over time"),
                    ProcessStep("reliability", "builds", "as each person follows through on commitments", "gradually"),
                ],
                source="seed",
                confidence=0.8,
            ),
            "knowledge": EventSchema(
                name="knowledge acquisition",
                steps=[
                    ProcessStep("curiosity", "drives", "the initial desire to learn", "naturally"),
                    ProcessStep("information", "is gathered", "through observation, study, and experience", "gradually"),
                    ProcessStep("understanding", "develops", "as patterns emerge and connections form", "over time"),
                    ProcessStep("wisdom", "grows", "from applying understanding to real situations", "gradually"),
                ],
                source="seed",
                confidence=0.8,
            ),
            "love": EventSchema(
                name="love development",
                steps=[
                    ProcessStep("connection", "begins", "with a spark of recognition and interest", "naturally"),
                    ProcessStep("bonding", "deepens", "through shared experiences and vulnerability", "over time"),
                    ProcessStep("care", "grows", "as emotional investment increases", "gradually"),
                    ProcessStep("commitment", "forms", "creating a lasting attachment", "over time"),
                ],
                source="seed",
                confidence=0.8,
            ),
            "learning": EventSchema(
                name="learning process",
                steps=[
                    ProcessStep("curiosity", "sparks", "interest in a new subject", "naturally"),
                    ProcessStep("attention", "focuses", "on relevant information and patterns", "actively"),
                    ProcessStep("practice", "reinforces", "neural pathways through repetition", "over time"),
                    ProcessStep("mastery", "emerges", "as skills become second nature", "gradually"),
                ],
                source="seed",
                confidence=0.8,
            ),
            "change": EventSchema(
                name="change process",
                steps=[
                    ProcessStep("awareness", "grows", "of the need for something different", "naturally"),
                    ProcessStep("decision", "is made", "to pursue a new direction", "deliberately"),
                    ProcessStep("transition", "occurs", "through a period of adjustment", "over time"),
                    ProcessStep("growth", "follows", "as the new state becomes familiar", "gradually"),
                ],
                source="seed",
                confidence=0.8,
            ),
            "friendship": EventSchema(
                name="friendship development",
                steps=[
                    ProcessStep("meeting", "happens", "through shared context or chance", "naturally"),
                    ProcessStep("shared experience", "builds", "common ground and memories", "over time"),
                    ProcessStep("trust", "develops", "through consistency and reliability", "gradually"),
                    ProcessStep("support", "deepens", "the bond through mutual care", "over time"),
                ],
                source="seed",
                confidence=0.8,
            ),
            "growth": EventSchema(
                name="personal growth",
                steps=[
                    ProcessStep("challenge", "appears", "pushing beyond current limits", "suddenly or gradually"),
                    ProcessStep("struggle", "teaches", "resilience and new perspectives", "over time"),
                    ProcessStep("adaptation", "occurs", "as new skills and understanding develop", "gradually"),
                    ProcessStep("transformation", "emerges", "as a stronger, wiser state", "over time"),
                ],
                source="seed",
                confidence=0.8,
            ),
            "creativity": EventSchema(
                name="creative process",
                steps=[
                    ProcessStep("inspiration", "strikes", "from observation or experience", "spontaneously"),
                    ProcessStep("exploration", "experiments", "with possibilities and combinations", "actively"),
                    ProcessStep("refinement", "shapes", "rough ideas into polished forms", "over time"),
                    ProcessStep("expression", "shares", "the creation with others", "deliberately"),
                ],
                source="seed",
                confidence=0.8,
            ),
            "communication": EventSchema(
                name="communication process",
                steps=[
                    ProcessStep("thought", "forms", "as an idea takes shape in the mind", "naturally"),
                    ProcessStep("expression", "conveys", "the thought through words or actions", "deliberately"),
                    ProcessStep("reception", "occurs", "when the other person takes it in", "actively"),
                    ProcessStep("understanding", "emerges", "through shared meaning and context", "mutually"),
                ],
                source="seed",
                confidence=0.8,
            ),
            "respect": EventSchema(
                name="respect development",
                steps=[
                    ProcessStep("recognition", "grows", "of another's value or achievements", "naturally"),
                    ProcessStep("consideration", "follows", "treating them with care", "deliberately"),
                    ProcessStep("admiration", "deepens", "through consistent positive qualities", "over time"),
                    ProcessStep("trust", "builds", "creating a foundation for mutual respect", "gradually"),
                ],
                source="seed",
                confidence=0.8,
            ),
            "hope": EventSchema(
                name="hope cycle",
                steps=[
                    ProcessStep("desire", "emerges", "for something better or different", "naturally"),
                    ProcessStep("belief", "forms", "that change is possible", "gradually"),
                    ProcessStep("action", "follows", "driven by the belief in possibility", "actively"),
                    ProcessStep("resilience", "grows", "sustaining effort through challenges", "over time"),
                ],
                source="seed",
                confidence=0.8,
            ),
            "truth": EventSchema(
                name="truth seeking",
                steps=[
                    ProcessStep("question", "arises", "from curiosity or doubt", "naturally"),
                    ProcessStep("investigation", "explores", "evidence and perspectives", "actively"),
                    ProcessStep("clarity", "emerges", "as patterns and facts align", "gradually"),
                    ProcessStep("understanding", "solidifies", "into a grounded perspective", "over time"),
                ],
                source="seed",
                confidence=0.8,
            ),
            "life": EventSchema(
                name="life journey",
                steps=[
                    ProcessStep("birth", "begins", "a unique journey of experiences", "naturally"),
                    ProcessStep("growth", "unfolds", "through learning, challenges, and connections", "over time"),
                    ProcessStep("purpose", "emerges", "from values, passions, and contributions", "gradually"),
                    ProcessStep("legacy", "remains", "through impact on others and the world", "beyond"),
                ],
                source="seed",
                confidence=0.8,
            ),
        }

        for concept, schema in default_schemas.items():
            self._schemas[concept.lower()] = schema

    def get_schema(self, concept: str) -> Optional[EventSchema]:
        """Get the event schema for a concept, if one exists."""
        if not self._seeded:
            self.seed_default_schemas()
        return self._schemas.get(concept.lower().strip())

    def has_schema(self, concept: str) -> bool:
        """Check if a concept has a registered schema."""
        if not self._seeded:
            self.seed_default_schemas()
        return concept.lower().strip() in self._schemas

    def add_schema(self, concept: str, schema: EventSchema):
        """Register a new event schema."""
        self._schemas[concept.lower().strip()] = schema

    def discover_from_text(self, text: str, subject: str) -> Optional[EventSchema]:
        """Attempt to discover an event schema from web text.

        Looks for sequential patterns like:
        "First X happens, then Y, and finally Z"
        "X leads to Y which leads to Z"

        Returns an EventSchema if a clear process chain is found.
        """
        text_lower = text.lower()
        subject_lower = subject.lower()

        # Look for temporal markers that indicate process chains
        steps = []
        # Pattern: "X [verb] and [verb] ..."
        # Pattern: "First/Then/Finally/Next"

        # Simple split on sentences and look for subject-related chains
        sentences = re.split(r'[.!?]+', text_lower)
        found_steps = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            # Check if this sentence describes a step involving the subject
            if subject_lower in sent:
                # Try to extract verb + object
                # Simple heuristic: first verb after subject mention
                words = sent.split()
                for i, w in enumerate(words):
                    if w == subject_lower and i + 2 < len(words):
                        verb = words[i + 1]
                        obj_words = words[i + 2:i + 5]
                        if verb.endswith('s') or verb.endswith('ed') or verb.endswith('ing'):
                            obj = ' '.join(obj_words).strip('.,!?')
                            if obj and len(obj) > 2:
                                found_steps.append((verb, obj))

        if len(found_steps) >= 2:
            steps = []
            for i, (verb, obj) in enumerate(found_steps[:4]):
                steps.append(ProcessStep(
                    concept=subject if i == 0 else obj.split()[-1],
                    verb=verb,
                    description=obj,
                    confidence=0.4,
                ))
            return EventSchema(
                name=f"{subject} process",
                steps=steps,
                source="discovered",
                confidence=0.4,
            )
        return None

    def get_narrative_from_schema(self, concept: str) -> Optional[List[str]]:
        """Generate narrative sentence descriptions from a schema.

        Returns a list of 2-3 sentences describing the process.
        """
        schema = self.get_schema(concept)
        if not schema or len(schema.steps) < 2:
            return None

        sentences = []
        # Introduction: "Concept is about [overall process]"
        if schema.steps:
            first = schema.steps[0]
            intro = f"{concept.capitalize()} {first.verb} {first.description}"
            if first.duration:
                intro = intro.rstrip('.') + f", {first.duration}"
            sentences.append(intro + ".")

        # Middle steps: "It then [verb] through [process]"
        for i, step in enumerate(schema.steps[1:-1], 1):
            if i == 1:
                prefix = "it then"
            else:
                prefix = "over time, it"
            duration = f", {step.duration}" if step.duration else ""
            sentence = f"{prefix} {step.verb} {step.description}{duration}."
            sentences.append(sentence)

        # Final step: describing culmination/significance
        if len(schema.steps) >= 2:
            last = schema.steps[-1]
            significance = f"and this {last.verb} {last.description}"
            if last.duration:
                significance += f" {last.duration}"
            sentences.append(f"{significance}.")

        return sentences

    def blend_schemas(self, concept_a: str, concept_b: str) -> Optional[EventSchema]:
        """Blend two schemas into a combined narrative.

        Takes the first half of schema A and second half of schema B,
        creating a blended process description.
        """
        schema_a = self.get_schema(concept_a)
        schema_b = self.get_schema(concept_b)
        if not schema_a or not schema_b:
            return None

        mid_a = len(schema_a.steps) // 2
        mid_b = len(schema_b.steps) // 2

        blended_steps = schema_a.steps[:mid_a] + schema_b.steps[mid_b:]
        return EventSchema(
            name=f"{concept_a} and {concept_b}",
            steps=blended_steps,
            source="blended",
            confidence=min(schema_a.confidence, schema_b.confidence) * 0.8,
        )

    def get_state(self) -> Dict:
        """Serialize schemas for saving."""
        state = {}
        for concept, schema in self._schemas.items():
            state[concept] = {
                "name": schema.name,
                "steps": [
                    {
                        "concept": s.concept,
                        "verb": s.verb,
                        "description": s.description,
                        "duration": s.duration,
                        "confidence": s.confidence,
                    }
                    for s in schema.steps
                ],
                "source": schema.source,
                "confidence": schema.confidence,
            }
        return state

    def set_state(self, state: Dict):
        """Restore schemas from saved state."""
        for concept, data in state.items():
            steps = [
                ProcessStep(
                    concept=s["concept"],
                    verb=s["verb"],
                    description=s.get("description", ""),
                    duration=s.get("duration", ""),
                    confidence=s.get("confidence", 0.5),
                )
                for s in data.get("steps", [])
            ]
            self._schemas[concept] = EventSchema(
                name=data.get("name", concept),
                steps=steps,
                source=data.get("source", "restored"),
                confidence=data.get("confidence", 0.5),
            )
        self._seeded = True
