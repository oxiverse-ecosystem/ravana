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
        # G4 (sensorimotor schema selection): when no direct/GloVe match exists,
        # the hippocampus-like pattern-completion path retrieves a known schema
        # whose SUBJECT shares the query's sensorimotor fingerprint. This index
        # is built lazily from the lancaster_fn passed to the query.
        self._sensorimotor_schema_index: Dict[str, np.ndarray] = {}

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

    def get_schema_by_similarity(self, concept: str,
                                  vector_fn: Optional[Callable] = None,
                                  min_similarity: float = 0.25) -> Optional[Tuple[str, EventSchema, float]]:
        """Find the closest schema by semantic vector similarity.

        Conceptual blending mechanism: when a concept has no direct schema,
        find the semantically closest KNOWN concept that has a schema,
        and return its schema along with the similarity score.

        This mimics the brain's analogical reasoning: hippocampus binds
        known schemas onto novel concepts by structural similarity.

        Args:
            concept: The concept to find a schema for
            vector_fn: Function that returns a vector for a concept label.
                       E.g., lambda x: self._glove_vector(x)
            min_similarity: Minimum cosine similarity to accept a match

        Returns:
            Tuple (closest_concept, schema, similarity) or None if no match
        """
        if not self._seeded:
            self.seed_default_schemas()

        concept_lower = concept.lower().strip()

        # Direct match first
        if concept_lower in self._schemas:
            return (concept_lower, self._schemas[concept_lower], 1.0)

        # No vector function means no similarity search
        if vector_fn is None:
            return None

        # Get the query vector
        query_vec = vector_fn(concept_lower)
        if query_vec is None:
            return None
        qnorm = np.linalg.norm(query_vec)
        if qnorm < 1e-10:
            return None
        query_vec = query_vec / qnorm

        # Find the closest schema concept by cosine similarity
        best_sim = 0.0
        best_concept = None
        best_schema = None

        for schema_concept, schema in self._schemas.items():
            if schema_concept == concept_lower:
                continue
            schema_vec = vector_fn(schema_concept)
            if schema_vec is None:
                continue
            snorm = np.linalg.norm(schema_vec)
            if snorm < 1e-10:
                continue
            sim = float(np.dot(query_vec, schema_vec / snorm))
            # Clamp to [0, 1] — we only care about POSITIVE similarity
            sim = max(0.0, sim)
            if sim > best_sim:
                best_sim = sim
                best_concept = schema_concept
                best_schema = schema

        if best_schema is None or best_sim < min_similarity:
            return None

        return (best_concept, best_schema, best_sim)

    def get_schema_by_sensorimotor_similarity(
        self,
        concept: str,
        lancaster_fn: Optional[Callable] = None,
        glove_fn: Optional[Callable] = None,
        min_similarity: float = 0.20,
        alpha: float = 0.5,
    ) -> Optional[Tuple[str, EventSchema, float]]:
        """Hippocampal pattern-completion by sensorimotor fingerprint.

        When a concept has no direct schema AND no strong GloVe match, the
        hippocampus generalizes a known schema whose SUBJECT shares the
        query's embodied (Lancaster 11-D) profile — e.g. a query about
        "hand" (high Hand/Arm) retrieves the "creativity" schema (also high
        Hand/Arm) even if they are semantically distant. This is cross-domain
        transfer by relational/embodied structure (Goudar et al., 2023;
        Masís-Obando et al., 2022).

        Dual similarity blends the sensorimotor and semantic channels:
            sim = alpha * cos(lanc_q, lanc_schema)
                 + (1 - alpha) * cos(glove_q, glove_schema)
        `alpha` is the mPFC integration weight (modulated by arousal /
        confidence elsewhere); 0 = GloVe-only, 1 = sensorimotor-only.

        FAIL-CLOSED: if the best similarity is below `min_similarity`, return
        None (do NOT force a schema onto an unrelated concept).

        Requires at least one of lancaster_fn / glove_fn. If only one channel
        is available, that channel is used with weight 1.0.
        """
        if not self._seeded:
            self.seed_default_schemas()
        if lancaster_fn is None and glove_fn is None:
            return None

        concept_lower = concept.lower().strip()
        # Direct match short-circuits (highest confidence).
        if concept_lower in self._schemas:
            return (concept_lower, self._schemas[concept_lower], 1.0)

        # Query vectors.
        q_lanc = lancaster_fn(concept_lower) if lancaster_fn else None
        q_glove = glove_fn(concept_lower) if glove_fn else None
        if q_lanc is not None:
            q_lanc = np.asarray(q_lanc, dtype=np.float64)
            n = float(np.linalg.norm(q_lanc))
            if n > 1e-10:
                q_lanc = q_lanc / n
            else:
                q_lanc = None
        if q_glove is not None:
            q_glove = np.asarray(q_glove, dtype=np.float64)
            n = float(np.linalg.norm(q_glove))
            if n > 1e-10:
                q_glove = q_glove / n
            else:
                q_glove = None
        if q_lanc is None and q_glove is None:
            return None

        # Effective blend weights when only one channel is present.
        if q_lanc is None:
            a_eff, have_lanc = 0.0, False
        elif q_glove is None:
            a_eff, have_lanc = 1.0, True
        else:
            a_eff, have_lanc = float(alpha), True

        best_sim, best_concept, best_schema = 0.0, None, None
        for sc_name, schema in self._schemas.items():
            if sc_name == concept_lower:
                continue
            sc_lanc = self._sensorimotor_schema_index.get(sc_name)
            if sc_lanc is None and lancaster_fn is not None:
                try:
                    sc_lanc = lancaster_fn(sc_name)
                except Exception:
                    sc_lanc = None
                if sc_lanc is not None:
                    sc_lanc = np.asarray(sc_lanc, dtype=np.float64)
                    sn = float(np.linalg.norm(sc_lanc))
                    sc_lanc = sc_lanc / sn if sn > 1e-10 else None
                    if sc_lanc is not None:
                        self._sensorimotor_schema_index[sc_name] = sc_lanc
            sc_glove = glove_fn(sc_name) if glove_fn else None
            if sc_glove is not None:
                sc_glove = np.asarray(sc_glove, dtype=np.float64)
                sn = float(np.linalg.norm(sc_glove))
                sc_glove = sc_glove / sn if sn > 1e-10 else None

            sim_parts = []
            if have_lanc and q_lanc is not None and sc_lanc is not None:
                sim_parts.append(a_eff * float(np.dot(q_lanc, sc_lanc)))
            if q_glove is not None and sc_glove is not None:
                w = (1.0 - a_eff) if have_lanc else 1.0
                sim_parts.append(w * float(np.dot(q_glove, sc_glove)))
            if not sim_parts:
                continue
            sim = max(0.0, sum(sim_parts))
            if sim > best_sim:
                best_sim, best_concept, best_schema = sim, sc_name, schema

        if best_schema is None or best_sim < min_similarity:
            return None
        return (best_concept, best_schema, best_sim)

    def get_narrative_from_similar_schema(self, concept: str,
                                           vector_fn: Optional[Callable] = None,
                                           min_similarity: float = 0.25) -> Optional[Tuple[List[str], str, float]]:
        """Generate a narrative by blending concepts onto similar schemas.

        Like the brain's analogical encoding (Hofstadter 2001, Holyoak 2012):
        when we encounter an unknown concept, the hippocampus retrieves
        a structurally similar KNOWN schema and uses it as a scaffold,
        substituting the target concept into the schema roles.

        Returns:
            Tuple (narrative_sentences, source_concept, similarity) or None
        """
        result = self.get_schema_by_similarity(concept, vector_fn, min_similarity)
        if result is None:
            return None

        source_concept, schema, similarity = result
        concept_lower = concept.lower().strip()

        # Generate narrative sentences but SUBSTITUTE the query concept
        # into the schema, producing blended narratives like:
        #   Query: "infinity" -> Schema: "growth" (sim=0.31)
        #   Blended: "infinity begins naturally..."
        narrative = self.get_narrative_from_schema(source_concept)
        if narrative is None:
            return None

        # Substitute the concept name in sentences
        blended = []
        for sent in narrative:
            # Replace source concept references with query concept
            # Handle capitalization variants
            lower_sent = sent.lower()
            if source_concept.lower() in lower_sent:
                # Case-insensitive replacement preserving original case
                pattern = re.compile(re.escape(source_concept), re.IGNORECASE)
                new_sent = pattern.sub(concept_lower, sent)
                # Capitalize first letter
                new_sent = new_sent[0].upper() + new_sent[1:]
            else:
                new_sent = sent
            blended.append(new_sent)

        return (blended, source_concept, similarity)

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

        Returns a list of 2-4 sentences describing the process.
        """
        schema = self.get_schema(concept)
        if not schema or len(schema.steps) < 2:
            return None

        sentences = []
        concept_disp = concept.capitalize()

        # Step 1: Introduction
        first = schema.steps[0]
        if first.concept.lower() == concept.lower():
            intro = f"{concept_disp} {first.verb} {first.description}"
        else:
            intro = f"For {concept_disp.lower()}, the process begins when {first.concept} {first.verb} {first.description}"
        
        if first.duration:
            intro = intro.rstrip('.') + f", {first.duration}"
        sentences.append(intro + ".")

        # Step 2: Middle steps
        for i, step in enumerate(schema.steps[1:-1], 1):
            # Formulate transition/subject
            if i == 1:
                prefix = f"This process unfolds as {step.concept}"
            else:
                prefix = f"Through this development, {step.concept}"
                
            duration = f", {step.duration}" if step.duration else ""
            sentence = f"{prefix} {step.verb} {step.description}{duration}."
            sentences.append(sentence)

        # Step 3: Final step
        last = schema.steps[-1]
        if last.concept.lower() == concept.lower():
            significance = f"Ultimately, {concept_disp.lower()} {last.verb} {last.description}"
        else:
            significance = f"Ultimately, this leads to {last.concept} which {last.verb} {last.description}"
            
        if last.duration:
            significance += f", {last.duration}"
        sentences.append(significance + ".")

        return sentences

    def blend_schemas(self, concept_a: str, concept_b: str,
                      vector_fn: Optional[Callable] = None) -> Optional[EventSchema]:
        """Blend two schemas into a combined narrative.

        Takes the first half of schema A and second half of schema B,
        creating a blended process description.

        If a concept doesn't have a direct schema, tries similarity search.
        """
        schema_a = self.get_schema(concept_a)
        if schema_a is None and vector_fn is not None:
            res = self.get_schema_by_similarity(concept_a, vector_fn)
            if res:
                _, schema_a, _ = res

        schema_b = self.get_schema(concept_b)
        if schema_b is None and vector_fn is not None:
            res = self.get_schema_by_similarity(concept_b, vector_fn)
            if res:
                _, schema_b, _ = res

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
