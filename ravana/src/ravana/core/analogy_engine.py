"""
Analogy Engine — Structure Mapping for Analogical Reasoning
============================================================
Implements Gentner's Structure Mapping Theory for analogical completion:
A:B :: C:D — find D such that relation(A,B) ≈ relation(C,D)

Neuroscience grounding:
- RLPFC (rostral lateral PFC) performs relational integration
- Parietal cortex supports analogical mapping
- Hippocampus provides relational encoding for structure comparison

Design:
- Uses RLMv2 relation vectors for relational similarity
- Structure mapping: find maximal common substructure
- Candidate generation via graph traversal + vector similarity
- Supports cross-domain analogies (bird:sky :: fish:water)
"""
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import time

try:
    from ravana_ml.graph import ConceptGraph, ConceptEdge, ConceptNode
    from ravana_ml.nn.rlm_v2 import RelationPredictor
except ImportError:
    ConceptGraph = None
    ConceptEdge = None
    ConceptNode = None
    RelationPredictor = None


@dataclass
class AnalogicalMapping:
    """A candidate mapping in an analogy."""
    source_a: str
    source_b: str
    target_c: str
    candidate_d: str
    relation_similarity: float  # cosine similarity of relation vectors
    structural_consistency: float  # how well the mapping preserves structure
    confidence: float  # combined score


@dataclass
class AnalogyConfig:
    """Configuration for analogy engine."""
    max_candidates: int = 10
    min_relation_similarity: float = 0.5
    min_structural_consistency: float = 0.4
    max_hops: int = 2  # max graph distance for candidate search
    use_rlm_vectors: bool = True  # use RLMv2 relation vectors if available


class AnalogyEngine:
    """Structure mapping engine for analogical reasoning.

    Algorithm (Gentner's Structure Mapping):
    1. Extract relation between A and B from graph edges
    2. Find all concepts C that have outgoing edges of same relation type
    3. For each candidate D connected to C via same relation type:
       - Compute relation vector similarity: sim(rel_vec(A→B), rel_vec(C→D))
       - Compute structural consistency: check if A's other relations map to C's
    4. Return top candidates ranked by combined score
    """

    def __init__(self, graph: ConceptGraph, config: Optional[AnalogyConfig] = None,
                 rlm_predictor: Optional[RelationPredictor] = None):
        self.graph = graph
        self.config = config or AnalogyConfig()
        self.rlm_predictor = rlm_predictor

    def solve_analogy(self, a: str, b: str, c: str) -> List[AnalogicalMapping]:
        """Solve A:B :: C:___ analogy.

        Args:
            a: Source concept A (e.g., "bird")
            b: Source concept B (e.g., "sky")
            c: Target concept C (e.g., "fish")

        Returns:
            List of candidate mappings ranked by confidence
        """
        # Step 1: Find the relation between A and B
        rel_type, rel_vector = self._extract_relation(a, b)
        if not rel_type:
            # Fallback: use most common relation type
            rel_type = "analogical"
            rel_vector = self._get_default_relation_vector(rel_type)

        # Step 2: Find candidate D concepts for C with same relation type
        candidates = self._find_candidates(c, rel_type)

        # Step 3: Score each candidate
        mappings = []
        for d in candidates:
            score = self._score_mapping(a, b, c, d, rel_type, rel_vector)
            if score.confidence >= self.config.min_relation_similarity * self.config.min_structural_consistency:
                mappings.append(score)

        # Step 4: Rank and return top candidates
        mappings.sort(key=lambda m: m.confidence, reverse=True)
        return mappings[:self.config.max_candidates]

    def _extract_relation(self, a: str, b: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Extract the primary relation type and vector between A and B."""
        a_node = self._find_node(a)
        b_node = self._find_node(b)
        if not a_node or not b_node:
            return None, None

        # Check direct edge A→B
        edge = self.graph.get_edge(a_node.id, b_node.id)
        if edge and edge.relation_type in ("analogical", "semantic", "causal", "transitive"):
            return edge.relation_type, edge.relation_vector.copy()

        # Check reverse edge B→A
        edge = self.graph.get_edge(b_node.id, a_node.id)
        if edge and edge.relation_type in ("analogical", "semantic", "causal", "transitive"):
            return edge.relation_type, edge.relation_vector.copy()

        # Check spreading activation for indirect relation
        # Find paths up to 2 hops
        paths = self._find_paths(a_node.id, b_node.id, max_hops=2)
        if paths:
            # Use the relation type of the first edge in the best path
            best_path = max(paths, key=lambda p: p[1])  # (path, score)
            first_edge = self.graph.get_edge(best_path[0][0], best_path[0][1])
            if first_edge:
                return first_edge.relation_type, first_edge.relation_vector.copy()

        return None, None

    def _find_candidates(self, c: str, rel_type: str) -> List[str]:
        """Find all concepts D that have relation rel_type from C."""
        c_node = self._find_node(c)
        if not c_node:
            return []

        candidates = []
        for (src, tgt), edge in list(self.graph.edges.items()):
            if src == c_node.id and edge.relation_type == rel_type:
                tgt_node = self.graph.nodes.get(tgt)
                if tgt_node:
                    candidates.append(tgt_node.label)
            # Also check reverse direction for bidirectional relations
            elif tgt == c_node.id and edge.relation_type in ("analogical", "transitive"):
                src_node = self.graph.nodes.get(src)
                if src_node:
                    candidates.append(src_node.label)

        return list(set(candidates))

    def _score_mapping(self, a: str, b: str, c: str, d: str,
                       rel_type: str, rel_vector: np.ndarray) -> AnalogicalMapping:
        """Score a candidate mapping A:B :: C:D."""
        # Get relation vector for C→D
        c_node = self._find_node(c)
        d_node = self._find_node(d)
        cd_rel_vector = None

        if c_node and d_node:
            edge = self.graph.get_edge(c_node.id, d_node.id)
            if edge:
                cd_rel_vector = edge.relation_vector
            else:
                edge = self.graph.get_edge(d_node.id, c_node.id)
                if edge:
                    cd_rel_vector = edge.relation_vector

        # Relation vector similarity
        rel_similarity = 0.0
        if cd_rel_vector is not None:
            norm1 = np.linalg.norm(rel_vector)
            norm2 = np.linalg.norm(cd_rel_vector)
            if norm1 > 0 and norm2 > 0:
                rel_similarity = float(np.dot(rel_vector, cd_rel_vector) / (norm1 * norm2))

        # Structural consistency: check if A's neighbors map to C's neighbors
        structural_consistency = self._compute_structural_consistency(a, b, c, d)

        # Combined confidence
        confidence = (rel_similarity * 0.6 + structural_consistency * 0.4)

        return AnalogicalMapping(
            source_a=a, source_b=b, target_c=c, candidate_d=d,
            relation_similarity=rel_similarity,
            structural_consistency=structural_consistency,
            confidence=confidence
        )

    def _compute_structural_consistency(self, a: str, b: str, c: str, d: str) -> float:
        """Check if the relational structure around A maps to structure around C."""
        a_node = self._find_node(a)
        c_node = self._find_node(c)
        b_node = self._find_node(b)
        d_node = self._find_node(d)

        if not all([a_node, b_node, c_node, d_node]):
            return 0.0

        # Get neighbors of A and C (excluding B and D respectively)
        a_neighbors = self._get_neighbors(a_node.id, exclude={b_node.id})
        c_neighbors = self._get_neighbors(c_node.id, exclude={d_node.id})

        if not a_neighbors or not c_neighbors:
            return 0.5  # Neutral if no structure to compare

        # Check if A's relation types have counterparts in C
        a_rels = set()
        for nbr_id, edge in a_neighbors:
            a_rels.add(edge.relation_type)

        c_rels = set()
        for nbr_id, edge in c_neighbors:
            c_rels.add(edge.relation_type)

        # Overlap of relation types
        overlap = len(a_rels & c_rels)
        union = len(a_rels | c_rels)

        if union == 0:
            return 0.5
        return overlap / union

    def _get_neighbors(self, node_id: int, exclude: Set[int] = None) -> List[Tuple[int, ConceptEdge]]:
        """Get outgoing edges from a node."""
        exclude = exclude or set()
        neighbors = []
        for (src, tgt), edge in list(self.graph.edges.items()):
            if src == node_id and tgt not in exclude:
                neighbors.append((tgt, edge))
        return neighbors

    def _find_paths(self, src_id: int, tgt_id: int, max_hops: int = 2) -> List[Tuple[List[Tuple[int, int]], float]]:
        """Find paths between two nodes up to max_hops."""
        # Simple BFS
        paths = []
        queue = [([src_id], 0, 1.0)]  # (path, hops, score)

        while queue:
            path, hops, score = queue.pop(0)
            current = path[-1]

            if current == tgt_id and hops > 0:
                paths.append(([(path[i], path[i+1]) for i in range(len(path)-1)], score))
                continue

            if hops >= max_hops:
                continue

            for (s, t), edge in list(self.graph.edges.items()):
                if s == current and t not in path:
                    new_score = score * edge.weight * edge.confidence
                    queue.append((path + [t], hops + 1, new_score))

        return paths

    def _find_node(self, label: str) -> Optional[ConceptNode]:
        """Find node by label."""
        for node in list(self.graph.nodes.values()):
            if node.label.lower() == label.lower():
                return node
        return None

    def _get_default_relation_vector(self, rel_type: str) -> np.ndarray:
        """Get default relation vector for a type."""
        from ravana_ml.graph import ConceptEdge
        return ConceptEdge._init_relation_vector(rel_type, 16)

    def get_best_completion(self, a: str, b: str, c: str) -> Optional[str]:
        """Get the single best analogy completion."""
        mappings = self.solve_analogy(a, b, c)
        if mappings:
            return mappings[0].candidate_d
        return None


# Convenience function for chat interface
def solve_analogy_query(graph: ConceptGraph, a: str, b: str, c: str,
                        rlm_predictor: Optional[RelationPredictor] = None) -> Optional[str]:
    """Solve A:B::C:D analogy and return best D."""
    engine = AnalogyEngine(graph, rlm_predictor=rlm_predictor)
    return engine.get_best_completion(a, b, c)