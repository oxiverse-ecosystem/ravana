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
