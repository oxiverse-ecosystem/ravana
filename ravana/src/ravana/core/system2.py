import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from ravana_ml.graph import ConceptGraph
from ravana.core.causal_schema import CausalSchemaLearner, CausalSchema

class System2Simulator:
    """System 2 deliberative mental model simulation engine.
    
    Implements:
    - Causal subgraph extraction around a query topic.
    - Forward step-by-step mental simulation ("if A then B then C...").
    - Counterfactual simulation (flipping weights/relations to test alternatives).
    """
    
    def __init__(self, graph: ConceptGraph, schema_learner: CausalSchemaLearner):
        self.graph = graph
        self.schema_learner = schema_learner
        
    def extract_causal_subgraph(self, seed_concept: str, max_nodes: int = 30) -> Set[int]:
        """Extract a causal subgraph (up to max_nodes) starting from a seed concept."""
        subgraph_node_ids = set()
        seed_nids = self.graph.get_node_ids(seed_concept) if hasattr(self.graph, 'get_node_ids') else []
        if not seed_nids:
            # Fallback to keyword matching
            for nid, node in self.graph.nodes.items():
                if node.label.lower() == seed_concept.lower():
                    seed_nids = [nid]
                    break
                    
        if not seed_nids:
            return subgraph_node_ids
            
        from collections import deque
        queue = deque(seed_nids)
        subgraph_node_ids.update(seed_nids)
        
        while queue and len(subgraph_node_ids) < max_nodes:
            curr = queue.popleft()
            # Find outgoing causal/semantic neighbors
            for tid, edge in self.graph.get_outgoing(curr):
                if tid not in subgraph_node_ids:
                    # Prefer causal and semantic edges for mental models
                    if getattr(edge, 'relation_type', 'semantic') in ('causal', 'semantic'):
                        subgraph_node_ids.add(tid)
                        queue.append(tid)
                        if len(subgraph_node_ids) >= max_nodes:
                            break
                            
        return subgraph_node_ids
        
    def simulate_forward(self, start_state: str, steps: int = 5) -> List[Tuple[str, str, float]]:
        """Run a forward causal simulation.
        
        Given a starting state (e.g. 'heat'), predicts subsequent states step-by-step.
        Returns a list of (state, condition, confidence) transitions.
        """
        current_state = start_state.lower()
        simulation_trace = []
        visited = {current_state}
        
        for _ in range(steps):
            # Check for schemas that match current state
            best_next_state = None
            best_condition = None
            best_conf = 0.0
            
            for schema in self.schema_learner.schemas:
                matches, sim = schema.matches(current_state)
                if matches:
                    conf = schema.confidence * sim
                    # Prioritize higher confidence predictions
                    if conf > best_conf and schema.state_b not in visited:
                        best_conf = conf
                        best_next_state = schema.state_b
                        best_condition = schema.condition
                        
            if best_next_state:
                simulation_trace.append((current_state, best_condition, best_conf))
                current_state = best_next_state
                visited.add(current_state)
            else:
                break
                
        return simulation_trace

    def simulate_counterfactual(self, start_state: str, intervene_edge: Tuple[str, str], steps: int = 5) -> List[Tuple[str, str, float]]:
        """Simulate a counterfactual scenario where a specific causal link is suppressed or flipped.
        
        Intervene_edge is (cause_state, condition) to temporarily disable.
        """
        cause_state, condition = intervene_edge
        cause_state = cause_state.lower()
        condition = condition.lower()
        
        # Save original schemas
        original_schemas = []
        for s in self.schema_learner.schemas:
            original_schemas.append((s, s.confidence))
            
        # Apply intervention: temporarily set matching schema confidence to 0.0
        for s in self.schema_learner.schemas:
            if s.state_a.lower() == cause_state and s.condition.lower() == condition:
                s.confidence = 0.0
                
        try:
            # Run forward simulation under counterfactual conditions
            counterfactual_trace = self.simulate_forward(start_state, steps)
        finally:
            # Restore original confidences
            for s, orig_conf in original_schemas:
                s.confidence = orig_conf
                
        return counterfactual_trace
