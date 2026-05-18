import numpy as np
from typing import List, Optional, Dict, Tuple
from .graph import ConceptGraph, ConceptNode, ConceptEdge
from .tensor import StateTensor


class PropagationEngine:
    def __init__(self, graph: ConceptGraph):
        self.graph = graph
        self.propagation_count = 0
        self.energy_dissipated = 0.0

    def propagate(self, input_vector: np.ndarray, steps: int = 3,
                  k_active: int = 7, decay: float = 0.5) -> List[int]:
        self.graph.reset_activation()
        active_ids = self.graph.bind_input(input_vector, k=k_active)
        self.graph.spread_activation(steps=steps, k_active=k_active, decay=decay)
        self.propagation_count += 1
        final_active = [n.id for n in sorted(self.graph.nodes.values(),
                                              key=lambda n: n.activation, reverse=True)
                        if n.activation > 0.01][:k_active]
        return final_active

    def get_activation_vector(self, nids: List[int]) -> np.ndarray:
        vecs = []
        for nid in nids:
            node = self.graph.get_node(nid)
            if node and node.activation > 0.01:
                vecs.append(node.vector * node.activation)
        if not vecs:
            return np.zeros(self.graph.dim, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def measure_coherence(self, nids: List[int]) -> float:
        if len(nids) < 2:
            return 1.0
        activations = []
        for i, a_id in enumerate(nids):
            for b_id in nids[i + 1:]:
                edge = self.graph.get_edge(a_id, b_id)
                if edge:
                    activations.append(edge.weight * edge.confidence)
        if not activations:
            return 0.0
        return float(np.mean(activations))

    def get_prediction(self, nids: List[int], top_k: int = 3) -> List[int]:
        scores = {}
        for nid in nids:
            node = self.graph.get_node(nid)
            if node is None or node.activation < 0.01:
                continue
            outgoing = [(t, e) for (s, t), e in self.graph.edges.items()
                        if s == nid and t not in nids]
            for target_id, edge in outgoing:
                pred = node.activation * edge.weight
                scores[target_id] = scores.get(target_id, 0.0) + pred
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in ranked[:top_k]]
