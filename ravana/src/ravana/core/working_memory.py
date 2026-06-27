import numpy as np
import time
from typing import List, Dict, Tuple, Optional

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


class WorkingMemorySlot:
    """Represents a single slot in the Prefrontal Working Memory."""
    
    def __init__(self, vector: np.ndarray, tag: str = "topic", context_vector: Optional[np.ndarray] = None):
        self.vector = vector.copy()
        self.tag = tag
        self.timestamp = time.time()
        
        # If context is provided, bind the vector to the context using VSA
        if context_vector is not None:
            self.bound_vector = hrr_bind(self.vector, context_vector)
        else:
            self.bound_vector = self.vector.copy()
            
    def unbind(self, context_vector: np.ndarray) -> np.ndarray:
        """Unbind and retrieve the original vector using the given context vector."""
        return hrr_unbind(self.bound_vector, context_vector)


class WorkingMemory:
    """Prefrontal Working Memory (PFC) buffer with limited capacity (~5 slots).
    
    Implements:
    - Content-addressable retrieval by concept similarity.
    - VSA role-filler context binding.
    - Activity-dependent exponential decay.
    - Adjacent-slot lateral interference/inhibition.
    """
    
    def __init__(self, capacity: int = 5, decay_rate: float = 0.1, interference_scale: float = 0.15):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.interference_scale = interference_scale
        self.slots: List[WorkingMemorySlot] = []
        self.current_context: Optional[np.ndarray] = None
        
    def push(self, vector: np.ndarray, tag: str = "topic"):
        """Push a vector into working memory. Removes oldest if capacity exceeded."""
        if len(self.slots) >= self.capacity:
            self.slots.pop(0)  # FIFO eviction
            
        slot = WorkingMemorySlot(vector, tag, self.current_context)
        self.slots.append(slot)
        self.apply_interference()
        
    def set_context(self, context_vector: np.ndarray):
        """Set the VSA context vector. If context changes significantly, decay all slots."""
        if self.current_context is not None:
            similarity = cosine_sim(self.current_context, context_vector)
            # If new context is very different (similarity < 0.5), trigger mass decay/forgetting
            if similarity < 0.5:
                self.decay(factor=2.0)
                
        self.current_context = context_vector.copy()
        
    def retrieve(self, query_vector: np.ndarray) -> Optional[WorkingMemorySlot]:
        """Retrieve the best matching working memory slot based on cosine similarity."""
        if not self.slots:
            return None
            
        best_slot = None
        best_sim = -2.0
        
        for slot in self.slots:
            # If we have a context, we decode/unbind first before comparing
            if self.current_context is not None:
                decoded_vector = slot.unbind(self.current_context)
            else:
                decoded_vector = slot.vector
                
            sim = cosine_sim(query_vector, decoded_vector)
            if sim > best_sim:
                best_sim = sim
                best_slot = slot
                
        # Accessing memory decays other slots slightly (activity-dependent decay)
        self.decay()
        return best_slot
        
    def decay(self, factor: float = 1.0):
        """Apply exponential decay to the vector magnitudes in all slots."""
        actual_decay = self.decay_rate * factor
        for slot in self.slots:
            slot.vector *= (1.0 - actual_decay)
            slot.bound_vector *= (1.0 - actual_decay)
            
    def apply_interference(self):
        """Inhibit similar items in adjacent slots (lateral inhibition)."""
        if len(self.slots) < 2:
            return
            
        # Compute adjacent similarities and apply mutual inhibition
        for i in range(len(self.slots) - 1):
            s1 = self.slots[i]
            s2 = self.slots[i+1]
            sim = max(0.0, cosine_sim(s1.vector, s2.vector))
            
            if sim > 0.3:
                # Inhibit both vectors proportional to their similarity
                inhibition = self.interference_scale * sim
                s1.vector *= (1.0 - inhibition)
                s1.bound_vector *= (1.0 - inhibition)
                s2.vector *= (1.0 - inhibition)
                s2.bound_vector *= (1.0 - inhibition)
                
    def clear(self):
        """Clear all slots from working memory."""
        self.slots.clear()
        self.current_context = None
        
    def get_state(self) -> List[Dict]:
        """Return serialized state of working memory."""
        return [
            {
                "tag": slot.tag,
                "timestamp": slot.timestamp,
                "vector_norm": float(np.linalg.norm(slot.vector))
            }
            for slot in self.slots
        ]
