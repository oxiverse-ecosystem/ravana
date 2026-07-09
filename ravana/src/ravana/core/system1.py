import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from ravana_ml.graph import ConceptGraph

class System1Attractor:
    """System 1 Attractor Dynamics for fast, intuitive pattern completion.
    
    Uses iterative activation settling across the ConceptGraph.
    Computes confidence from the entropy of the settled activation pattern.
    """
    
    def __init__(self, graph: ConceptGraph, decay: float = 0.1, threshold: float = 0.5):
        self.graph = graph
        self.decay = decay
        self.threshold = threshold  # Confidence threshold to avoid System 2 escalation
        
    def settle(self, seed_ids: List[int], max_iter: int = 50, tolerance: float = 1e-4) -> Tuple[Dict[int, float], float]:
        """Iteratively propagate activation until settling.
        
        Returns:
        - Dict of node_id -> settled activation
        - Confidence score (0.0 to 1.0, based on inverse entropy of activations)
        """
        if not seed_ids:
            return {}, 0.0
            
        # Initialize activations
        activations = {nid: 0.01 for nid in self.graph.nodes}
        for nid in seed_ids:
            if nid in activations:
                activations[nid] = 1.0
                
        seed_set = set(seed_ids)
        
        # Build adjacency for fast propagation
        adjacency: Dict[int, List[Tuple[int, float]]] = {nid: [] for nid in self.graph.nodes}
        for (s, t), edge in list(self.graph.edges.items()):
            # Excitatory edge weights propagate positive activation, inhibitory negative
            weight = edge.weight if getattr(edge, 'edge_type', 'excitatory') == 'excitatory' else -edge.weight
            adjacency[s].append((t, weight))
            adjacency[t].append((s, weight))  # Treat graph as undirected for spreading activation
            
        for _ in range(max_iter):
            prev_acts = activations.copy()
            max_change = 0.0
            
            for nid in self.graph.nodes:
                # Seed nodes can be kept clamped or partially clamped
                if nid in seed_set:
                    activations[nid] = 1.0
                    continue
                    
                net_input = 0.0
                for neighbor, w in adjacency[nid]:
                    net_input += w * prev_acts[neighbor]
                    
                # Update activation with decay
                new_act = prev_acts[nid] * (1.0 - self.decay) + net_input * 0.1
                new_act = np.clip(new_act, 0.0, 1.0)
                
                activations[nid] = float(new_act)
                max_change = max(max_change, abs(new_act - prev_acts[nid]))
                
            if max_change < tolerance:
                break
                
        # Filter out negligible activations
        settled = {nid: act for nid, act in activations.items() if act > 0.05}
        
        # Compute confidence based on normalized entropy of settled activations
        confidence = self.compute_confidence(settled)
        
        return settled, confidence
        
    def compute_confidence(self, settled_activations: Dict[int, float]) -> float:
        """Compute confidence score based on entropy.
        
        Sharp peak (few highly active nodes) -> low entropy -> high confidence.
        Dispersed/flat pattern -> high entropy -> low confidence.
        """
        if not settled_activations:
            return 0.0
            
        acts = np.array(list(settled_activations.values()))
        total = np.sum(acts)
        if total == 0.0:
            return 0.0
            
        probs = acts / total
        # Shannon Entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-15))
        
        # Normalize entropy by log2(N) where N is number of nodes
        max_entropy = np.log2(max(2, len(settled_activations)))
        norm_entropy = entropy / max_entropy
        
        # Confidence is inverse of normalized entropy
        confidence = 1.0 - norm_entropy
        return float(np.clip(confidence, 0.0, 1.0))
        
    def should_escalate(self, confidence: float) -> bool:
        """Determine if we should escalate to System 2 reasoning."""
        return confidence < self.threshold
