import numpy as np
from typing import Dict, Tuple, List, Optional

def hrr_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind two vectors using circular convolution (Holographic Reduced Representation)."""
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=len(a))

def hrr_unbind(c: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unbind vector a from c = a * b using circular correlation."""
    return np.fft.irfft(np.fft.rfft(c) * np.conj(np.fft.rfft(b)), n=len(c))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class VSAManager:
    """Manages VSA (Vector Symbolic Architecture) operations and role-filler bindings."""
    
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.role_vectors: Dict[str, np.ndarray] = {}
        # Predefine standard roles
        self.get_role("subject")
        self.get_role("verb")
        self.get_role("object")
        self.get_role("adverb")
        
    def generate_vector(self) -> np.ndarray:
        """Generate a random unit-norm vector of dimension dim."""
        v = np.random.randn(self.dim)
        return v / np.linalg.norm(v)
        
    def get_role(self, role_name: str) -> np.ndarray:
        """Get or generate a role vector."""
        if role_name not in self.role_vectors:
            self.role_vectors[role_name] = self.generate_vector()
        return self.role_vectors[role_name]
        
    def bind_role_filler(self, role: str, filler_vector: np.ndarray) -> np.ndarray:
        """Bind a filler vector to a specific role vector."""
        role_vec = self.get_role(role)
        # Ensure correct dimension
        if len(filler_vector) != self.dim:
            filler_vector = self._resize_vector(filler_vector)
        return hrr_bind(role_vec, filler_vector)
        
    def unbind_role(self, bound_structure: np.ndarray, role: str) -> np.ndarray:
        """Unbind/extract the filler vector for a given role from a bound structure."""
        role_vec = self.get_role(role)
        return hrr_unbind(bound_structure, role_vec)
        
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Superposition (addition) of multiple vectors, normalized to unit-norm."""
        if not vectors:
            return np.zeros(self.dim)
        res = np.sum(vectors, axis=0)
        norm = np.linalg.norm(res)
        if norm == 0.0:
            return res
        return res / norm
        
    def _resize_vector(self, vec: np.ndarray) -> np.ndarray:
        if len(vec) == self.dim:
            return vec
        elif len(vec) < self.dim:
            resized = np.zeros(self.dim)
            resized[:len(vec)] = vec
            return resized
        else:
            return vec[:self.dim]


class VSASchema:
    """Utterance schema stored as a VSA bound structure."""
    
    def __init__(self, manager: VSAManager, template_name: str, structure: List[str]):
        self.manager = manager
        self.template_name = template_name  # e.g. "S-V-O", "S-V-O-Adv"
        self.structure = structure          # e.g. ["subject", "verb", "object"]
        self.bound_vector: Optional[np.ndarray] = None
        self.usage_count: int = 0
        self.success_count: int = 0
        
    def construct_schema_vector(self, fillers: Dict[str, np.ndarray]) -> np.ndarray:
        """Construct the bound VSA vector for the schema using the provided fillers."""
        components = []
        for role in self.structure:
            if role in fillers:
                components.append(self.manager.bind_role_filler(role, fillers[role]))
        self.bound_vector = self.manager.bundle(components)
        return self.bound_vector
        
    def reconstruct_fillers(self, bound_vector: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract fillers from the bound schema vector."""
        fillers = {}
        for role in self.structure:
            fillers[role] = self.manager.unbind_role(bound_vector, role)
        return fillers
        
    def score_matching_filler(self, extracted_filler: np.ndarray, candidates: Dict[str, np.ndarray]) -> Tuple[str, float]:
        """Find the closest matching candidate for an extracted filler."""
        best_candidate = ""
        best_score = -1.0
        
        for name, vec in candidates.items():
            sim = cosine_sim(extracted_filler, vec)
            if sim > best_score:
                best_score = sim
                best_candidate = name
                
        return best_candidate, best_score
