import numpy as np
from typing import Dict, List, Tuple, Set, Optional

class CoherenceNetwork:
    """Coherence Network implementing Thagard's ECHO model for constraint satisfaction.
    
    Propositions are represented as nodes. Symmetric links represent:
    - Positive constraints (coherence: explanation, association, inference)
    - Negative constraints (incoherence: contradiction, competition)
    
    Evidence nodes are clamped to positive activation to ground the network.
    """
    
    def __init__(self, decay: float = 0.05, upper_bound: float = 1.0, lower_bound: float = -1.0):
        self.decay = decay
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        
        self.propositions: Dict[str, float] = {}          # pid -> current activation
        self.initial_activations: Dict[str, float] = {}   # pid -> initial activation
        self.evidence: Set[str] = set()                   # set of evidence pids
        
        # constraints: dict of (pid1, pid2) -> weight
        # We ensure pid1 < pid2 for canonical keys
        self.constraints: Dict[Tuple[str, str], float] = {}
        
    def add_proposition(self, pid: str, initial_activation: float = 0.01, is_evidence: bool = False):
        """Add a proposition to the network."""
        self.propositions[pid] = initial_activation
        self.initial_activations[pid] = initial_activation
        if is_evidence:
            self.evidence.add(pid)
            self.propositions[pid] = 1.0
            
    def add_constraint(self, pid1: str, pid2: str, weight: float):
        """Add a symmetric constraint between two propositions.
        
        Positive weight = Coherence (excitatory link)
        Negative weight = Contradiction/Incoherence (inhibitory link)
        """
        if pid1 not in self.propositions or pid2 not in self.propositions:
            raise ValueError("Both propositions must be added to the network first.")
        if pid1 == pid2:
            return # No self-links
            
        key = (min(pid1, pid2), max(pid1, pid2))
        self.constraints[key] = weight
        
    def add_coherence(self, pid1: str, pid2: str, weight: float = 0.1):
        """Helper to add positive (coherence) constraint."""
        self.add_constraint(pid1, pid2, abs(weight))
        
    def add_contradiction(self, pid1: str, pid2: str, weight: float = -0.2):
        """Helper to add negative (contradiction) constraint."""
        self.add_constraint(pid1, pid2, -abs(weight))
        
    def settle(self, max_iter: int = 100, tolerance: float = 1e-4, learning_rate: float = 0.1) -> Dict[str, float]:
        """Settle the network by updating activations iteratively.
        
        Returns a dictionary of final activations.
        """
        pids = list(self.propositions.keys())
        if not pids:
            return {}
            
        # Compile constraints into a fast lookup dict mapping node -> list of (neighbor, weight)
        adjacency: Dict[str, List[Tuple[str, float]]] = {pid: [] for pid in pids}
        for (p1, p2), w in self.constraints.items():
            adjacency[p1].append((p2, w))
            adjacency[p2].append((p1, w))
            
        for _ in range(max_iter):
            prev_activations = self.propositions.copy()
            max_change = 0.0
            
            for pid in pids:
                # Evidence nodes are clamped
                if pid in self.evidence:
                    self.propositions[pid] = 1.0
                    continue
                    
                # Compute net input to the node
                net_input = 0.0
                for neighbor, w in adjacency[pid]:
                    net_input += w * prev_activations[neighbor]
                    
                # Thagard's ECHO activation update rules
                current = prev_activations[pid]
                if net_input > 0:
                    delta = net_input * (self.upper_bound - current) - self.decay * current
                else:
                    delta = net_input * (current - self.lower_bound) - self.decay * current
                    
                new_val = np.clip(current + learning_rate * delta, self.lower_bound, self.upper_bound)
                self.propositions[pid] = float(new_val)
                
                max_change = max(max_change, abs(new_val - current))
                
            if max_change < tolerance:
                break
                
        return self.propositions.copy()
        
    def get_accepted(self, threshold: float = 0.5) -> List[str]:
        """Return the list of propositions with activation above threshold."""
        return [pid for pid, act in self.propositions.items() if act > threshold]
        
    def get_rejected(self, threshold: float = -0.5) -> List[str]:
        """Return the list of propositions with activation below threshold."""
        return [pid for pid, act in self.propositions.items() if act < threshold]
