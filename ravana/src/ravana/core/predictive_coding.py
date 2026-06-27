import numpy as np
from typing import Dict, Tuple, List, Optional
from ravana_ml.graph import ConceptGraph, ConceptNode

class PredictiveCodingLearner:
    """Predictive Coding Learner for local backpropagation-free learning.
    
    Instead of global backpropagation or global prediction error minimization,
    each node learns local predictors that predict activity of target nodes
    based on context. Only prediction errors are propagated and learned from.
    """
    
    def __init__(self, graph: ConceptGraph, lr: float = 0.001):
        self.graph = graph
        self.lr = lr
        # Store predictor matrices: node_id -> (dim x dim) ndarray
        self.predictors: Dict[int, np.ndarray] = {}
        
    def get_predictor(self, node_id: int) -> np.ndarray:
        """Retrieve or initialize the predictor matrix for a specific node."""
        if node_id not in self.predictors:
            dim = self.graph.dim
            # Small random weights initialized to prevent symmetry
            self.predictors[node_id] = np.random.randn(dim, dim) * 0.01
        return self.predictors[node_id]
        
    def predict(self, node_id: int, context_vector: np.ndarray) -> np.ndarray:
        """Predict the node's vector given a context vector."""
        predictor = self.get_predictor(node_id)
        return predictor @ context_vector
        
    def learn_node(self, node_id: int, context_vector: np.ndarray, actual_vector: np.ndarray) -> Tuple[np.ndarray, float]:
        """Perform a local predictive coding update on a node.
        
        Updates the node's predictor matrix and slightly shifts the node's vector
        based on the local prediction error.
        
        Returns the error vector and its L2 norm.
        """
        node = self.graph.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in graph.")
            
        predictor = self.get_predictor(node_id)
        prediction = predictor @ context_vector
        error = actual_vector - prediction
        
        # Local learning rule (delta/Hebbian rule on the predictor weights)
        # d_predictor = lr * error x context_vector
        self.predictors[node_id] += self.lr * np.outer(error, context_vector)
        
        # Adjust the target node's vector using the prediction error,
        # scaled by the node's plasticity (1.0 - stability)
        plasticity = 1.0 - node.stability
        node.vector += self.lr * 0.1 * plasticity * error
        
        # Update node's prediction free energy
        error_norm = float(np.linalg.norm(error))
        node.prediction_free_energy = 0.9 * node.prediction_free_energy + error_norm
        
        # Log to history
        if not hasattr(node, 'free_energy_history') or node.free_energy_history is None:
            node.free_energy_history = []
        node.free_energy_history.append(error_norm)
        if len(node.free_energy_history) > 20:
            node.free_energy_history = node.free_energy_history[-20:]
            
        # Calculate free energy gradient
        if len(node.free_energy_history) >= 3:
            recent = np.mean(node.free_energy_history[-3:])
            older = np.mean(node.free_energy_history[:-3]) if len(node.free_energy_history) > 3 else recent
            node.free_energy_gradient = recent - older
            
        # Accumulate contradiction free energy
        if not hasattr(node, 'contradiction_free_energy'):
            node.contradiction_free_energy = 0.0
        node.contradiction_free_energy = 0.9 * node.contradiction_free_energy + error_norm
        
        if error_norm > 0.3:
            node.contradiction_count += 1
            
        return error, error_norm
        
    def propagate_errors(self, path: List[int], context_vector: np.ndarray) -> float:
        """Propagate prediction errors sequentially through a path of nodes (hierarchical/causal walk).
        
        Returns the total accumulated error norm along the path.
        """
        current_context = context_vector.copy()
        total_error = 0.0
        
        for i in range(len(path) - 1):
            target_id = path[i+1]
            target_node = self.graph.get_node(target_id)
            if target_node is None:
                continue
                
            error, error_norm = self.learn_node(target_id, current_context, target_node.vector)
            total_error += error_norm
            # The next context in the hierarchy is the error signal itself
            current_context = error
            
        return total_error
