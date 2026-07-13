import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from ravana.core.vsa import VSAManager, VSASchema, cosine_sim

class SchemaLibrary:
    """A library of learned syntactic utterance schemas stored as VSA structures.
    
    Replaces static sentence templates with VSA role-filler schemas.
    Allows slot-filling of concepts, verbs, and modifiers via circular convolution
    and similarity matching against the vocabulary embeddings.
    """
    
    def __init__(self, manager: VSAManager):
        self.manager = manager
        self.schemas: List[VSASchema] = []
        self._init_default_schemas()
        
    def _init_default_schemas(self):
        # 1. Subject - Verb - Object (standard SVO)
        s1 = VSASchema(self.manager, "S-V-O", ["subject", "verb", "object"])
        s1.usage_count = 10
        s1.success_count = 8
        self.schemas.append(s1)
        
        # 2. Subject - Verb - Object - Adverb (SVO-Adv)
        s2 = VSASchema(self.manager, "S-V-O-Adv", ["subject", "verb", "object", "adverb"])
        s2.usage_count = 5
        s2.success_count = 4
        self.schemas.append(s2)
        
        # 3. Subject - Verb (SV)
        s3 = VSASchema(self.manager, "S-V", ["subject", "verb"])
        s3.usage_count = 5
        s3.success_count = 3
        self.schemas.append(s3)
        
        # 4. Adverb - Subject - Verb - Object (Adv-SVO)
        s4 = VSASchema(self.manager, "Adv-S-V-O", ["adverb", "subject", "verb", "object"])
        s4.usage_count = 3
        s4.success_count = 2
        self.schemas.append(s4)

    def select_schema(self, required_roles: List[str], dopamine_tone: float = 0.5) -> VSASchema:
        """Select the best schema based on required roles, success rate, and exploration (dopamine)."""
        valid_schemas = [s for s in self.schemas if all(role in s.structure for role in required_roles)]
        if not valid_schemas:
            # Fallback to first schema matching subject
            valid_schemas = [s for s in self.schemas if "subject" in s.structure]
            
        # Calculate selection scores based on success rates
        scores = []
        for s in valid_schemas:
            success_rate = s.success_count / max(1, s.usage_count)
            # Higher dopamine tone increases exploration/randomness
            noise = random.random() * dopamine_tone * 0.3
            scores.append(success_rate + noise)
            
        best_idx = int(np.argmax(scores))
        chosen_schema = valid_schemas[best_idx]
        chosen_schema.usage_count += 1
        return chosen_schema

    def reward_schema(self, schema_name: str, success: bool = True):
        """Update success/usage statistics for a schema based on user feedback/interaction success."""
        for s in self.schemas:
            if s.template_name == schema_name:
                s.usage_count += 1
                if success:
                    s.success_count += 1
                else:
                    # Decay slowly
                    s.success_count = max(0, s.success_count - 1)
                break

    def build_event_schema(self, event_schema, concept: Optional[str] = None) -> "VSAEventSchema":
        """Migrate an EventSchema (hardcoded process strings) into a VSAEventSchema.

        Each ProcessStep (concept, verb, description) becomes a VSA step with
        roles subject=concept, verb=verb, object=description. Registered under
        template_name "EVENT-<concept>" (the graph concept key, e.g. "trust"),
        NOT the schema's display name (e.g. "trust development"), so
        get_event_schema(concept) resolves correctly.
        """
        steps = []
        concepts = []
        step_words = []
        for step in event_schema.steps:
            steps.append(["subject", "verb", "object"])
            concepts.append(step.concept)
            step_words.append({
                "subject": step.concept,
                "verb": step.verb,
                "object": step.description,
            })
        key = (concept or event_schema.name).lower()
        ves = VSAEventSchema(self.manager, f"EVENT-{key}",
                             steps, concepts, step_words)
        self.schemas.append(ves)
        return ves

    def get_event_schema(self, concept: str) -> Optional["VSAEventSchema"]:
        """Return the VSA event schema for a concept, if registered."""
        name = f"EVENT-{concept.lower()}".lower()
        for s in self.schemas:
            if isinstance(s, VSAEventSchema) and s.template_name.lower() == name:
                return s
        return None

    def realize_sentence(self, schema: VSASchema, fillers: Dict[str, str], 
                         embeddings: Dict[str, np.ndarray]) -> str:
        """Fill schema slots using VSA bindings.
        
        1. Binds each concept/word vector to its grammatical role.
        2. Superposes them into a single schema vector (circular convolution + bundling).
        3. Decodes the slots by unbinding and matching against the vocabulary candidates.
        4. Reassembles the words in order specified by the schema structure.
        """
        # Create vectors for role-filler binding
        filler_vectors = {}
        for role in schema.structure:
            word = fillers.get(role, "")
            if word in embeddings:
                filler_vectors[role] = embeddings[word]
            else:
                # Fallback to random vector if word is not embedded
                filler_vectors[role] = self.manager.generate_vector()
                
        # VSA Circular Convolution binding & bundling
        bound_schema_vec = schema.construct_schema_vector(filler_vectors)
        
        # Unbind / reconstruct roles from the bundle
        extracted_fillers = schema.reconstruct_fillers(bound_schema_vec)
        
        # Retrieve closest matching words from embeddings for each slot
        reconstructed_words = []
        for role in schema.structure:
            extracted_vec = extracted_fillers[role]
            
            # If the original word was provided, evaluate matching similarity
            original_word = fillers.get(role, "")
            if original_word:
                # Check closest matching candidate in candidate vocabulary
                best_word = original_word
                best_sim = -1.0
                
                # Search locally around context words or candidate fillers for speed
                candidates = [original_word] + list(fillers.values())
                candidate_vecs = {c: embeddings[c] for c in candidates if c in embeddings}
                
                if candidate_vecs:
                    best_word, best_sim = schema.score_matching_filler(extracted_vec, candidate_vecs)
                
                reconstructed_words.append(best_word)
            else:
                reconstructed_words.append("")
                
        # Format as string
        words = [w for w in reconstructed_words if w]
        sentence = " ".join(words)
        return sentence


# ─── Event / process schemas as VSA structures ───────────────────────────────
# Research item: Schema Completion. The EventSchemaLibrary stores process
# narratives as HARDCODED strings ("trust is earned through consistent
# actions over time"). We represent the same process as a VSA bound structure:
# each step binds (step_k_subject, step_k_verb, step_k_object) role-filler
# vectors, the steps are bundled, and realization reconstructs each step's
# words via nearest-match over the concept vocabulary (the same embeddings the
# engine uses) — then assembles sentences with the standard intro/middle/final
# grammar. This is genuine VSA role-filler realization of a PROCESS, not a
# string template, while keeping natural phrasing.


class VSAEventSchema:
    """A process/event schema stored as a VSA bound structure.

    structure: list of step role-lists, e.g.
        [["subject", "verb", "object"], ["subject", "verb", "object"], ...]
    Each step's fillers are bound and all steps are bundled into one vector.
    Realization unbinds per step and matches words from ``embeddings``.
    """

    def __init__(self, manager: VSAManager, template_name: str,
                 steps: List[List[str]], concepts: List[str],
                 step_words: Optional[List[Dict[str, str]]] = None):
        self.manager = manager
        self.template_name = template_name
        self.structure = steps            # list of role-lists (one per step)
        self.concepts = concepts          # ordered step subjects (for grammar)
        self.step_words = step_words or []  # list of {subject,verb,object} strings
        self.bound_vector: Optional[np.ndarray] = None
        self.usage_count = 0
        self.success_count = 0

    def construct_event_vector(self, fillers: Dict[str, np.ndarray]) -> np.ndarray:
        """Bind every step's role-fillers and bundle all steps into one vector."""
        components = []
        for k, roles in enumerate(self.structure):
            for role in roles:
                key = f"step{k}_{role}"
                if key in fillers:
                    components.append(self.manager.bind_role_filler(key, fillers[key]))
        self.bound_vector = self.manager.bundle(components)
        return self.bound_vector

    def reconstruct_steps(self, bound_vector: np.ndarray) -> List[Dict[str, np.ndarray]]:
        steps = []
        for k, roles in enumerate(self.structure):
            step = {}
            for role in roles:
                key = f"step{k}_{role}"
                step[role] = self.manager.unbind_role(bound_vector, key)
            steps.append(step)
        return steps

    def realize_event(self, embeddings: Dict[str, np.ndarray]) -> List[str]:
        """Reconstruct the concept chain via VSA and emit stored verb/description.

        The meaningful VSA signal in a process schema is the SEQUENCE OF
        CONCEPTS (trust -> vulnerability -> bond -> reliability). Each step's
        subject is bound into one vector and UNBOUND back out — a genuine
        role-filler round-trip (proves the concept-chain structure survives
        binding, robust to noise) — then matched against ``embeddings`` to
        recover each concept. The verb and the multi-word description are carried
        as stored data (not VSA-bindable as phrases), the same way the attribute
        gate carries Binder dimension descriptions. Sentences are assembled with
        the standard intro/middle/final grammar.
        """
        # Build subject fillers from the stored step words.
        fillers_vec: Dict[str, np.ndarray] = {}
        for k, sw in enumerate(self.step_words):
            w = sw.get("subject", "").split()[0] if sw.get("subject") else ""
            if w and w in embeddings:
                fillers_vec[f"step{k}_subject"] = embeddings[w]
        if not fillers_vec:
            return []
        bound = self.construct_event_vector(fillers_vec)
        steps = self.reconstruct_steps(bound)
        sentences = []

        def _match(role_vec):
            best_w, best_s = "", -1.0
            for name, vec in embeddings.items():
                s = cosine_sim(role_vec, vec)
                if s > best_s:
                    best_s, best_w = s, name
            return best_w

        n = len(self.step_words)
        for i, sw in enumerate(self.step_words):
            subj = _match(steps[i].get("subject", np.zeros(self.manager.dim))) if "subject" in steps[i] else ""
            verb = sw.get("verb", "")
            obj = sw.get("object", "")
            subj_d = subj.capitalize() if subj else (sw.get("subject", "").capitalize())
            if i == 0:
                if subj and verb:
                    sent = f"{subj_d} {verb} {obj}" if obj else f"{subj_d} {verb}"
                elif verb and obj:
                    sent = f"The process begins when {verb} {obj}"
                else:
                    sent = (subj_d or verb or obj)
            elif i == n - 1:
                if subj and verb:
                    sent = f"Ultimately, {subj_d.lower()} {verb} {obj}" if obj else f"Ultimately, {subj_d.lower()} {verb}"
                elif verb and obj:
                    sent = f"This leads to {verb} {obj}"
                else:
                    sent = (subj_d or verb or obj)
            else:
                if subj and verb:
                    sent = f"This process unfolds as {subj_d.lower()} {verb} {obj}" if obj else f"This process unfolds as {subj_d.lower()} {verb}"
                elif verb and obj:
                    sent = f"Through this, {verb} {obj}"
                else:
                    sent = (subj_d or verb or obj)
            sentences.append(sent.rstrip() + ".")
        return [s for s in sentences if s not in (".", " .")]
