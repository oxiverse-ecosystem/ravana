"""
Propagation Engine - Spreading activation over the concept graph.
Handles: multi-phase spreading activation, edge scoring, N-hop BFS traversal.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import deque

from ravana_ml.graph import ConceptGraph, ConceptEdge


class PropagationEngine:
    """Spreading activation inference over typed concept graph."""

    def __init__(self, graph: ConceptGraph):
        self.graph = graph

    def spread_activation(self, steps: int = 3, k_active: int = 10, decay: float = 0.3):
        """General spreading activation (Phase 1)."""
        for _ in range(steps):
            # Get currently active nodes
            active_nodes = [(nid, node.activation) for nid, node in self.graph.nodes.items()
                          if node.activation > 0.005]
            if not active_nodes:
                break

            to_activate = []
            for nid, act in active_nodes:
                # Spread to outgoing edges
                for tgt_id, edge in self.graph.get_outgoing(nid):
                    tgt_node = self.graph.get_node(tgt_id)
                    if tgt_node is not None:
                        score = act * edge.weight * edge.confidence * decay
                        if score > 0.01:
                            to_activate.append((tgt_id, score))
                # Spread to incoming edges (undirected semantics)
                for src_id, edge in self.graph.get_incoming(nid):
                    src_node = self.graph.get_node(src_id)
                    if src_node is not None:
                        score = act * edge.weight * edge.confidence * decay
                        if score > 0.01:
                            to_activate.append((src_id, score))

            for nid, amount in to_activate:
                self.graph.activate(nid, amount=amount)

    def relation_aware_spreading(self, subject_cid: int, rel_type_name: str,
                                  steps: int = 2, decay: float = 0.5,
                                  query_verb_word: str = "",
                                  decode_token_fn=None) -> Dict[int, float]:
        """Phase 2: Relation-aware spreading along matching edges."""
        scores: Dict[int, float] = {}

        for _ in range(steps):
            to_activate = []
            for nid, node in self.graph.nodes.items():
                if node.activation < 0.005:
                    continue
                for tgt_id, edge in self.graph.get_outgoing(nid):
                    if edge.relation_type != rel_type_name:
                        continue
                    tgt_node = self.graph.get_node(tgt_id)
                    if tgt_node is not None:
                        score = node.activation * decay * edge.weight * edge.confidence
                        # Predicate matching
                        pred_mult = 1.0
                        if query_verb_word and hasattr(edge, 'predicate_token_id') and edge.predicate_token_id != -1:
                            try:
                                edge_verb = decode_token_fn(edge.predicate_token_id).lower().strip()
                                if edge_verb:
                                    w1 = query_verb_word.rstrip('s').rstrip('d')
                                    w2 = edge_verb.rstrip('s').rstrip('d')
                                    if w1 == w2 or w1 in w2 or w2 in w1:
                                        pred_mult = 2.5
                                    else:
                                        pred_mult = 0.4
                            except Exception:
                                pass
                        score *= pred_mult
                        if score > 0.01:
                            scores[tgt_id] = scores.get(tgt_id, 0.0) + score
            for nid, amount in scores.items():
                self.graph.activate(nid, amount=amount)
        return scores

    def direct_edge_boost(self, subject_cid: int, rel_type_name: str,
                           query_verb_word: str = "",
                           decode_token_fn=None) -> Dict[int, float]:
        """Phase 2b: Direct edge boost from subject."""
        scores: Dict[int, float] = {}
        subject_node = self.graph.get_node(subject_cid)
        if subject_node is not None and subject_node.activation > 0.01:
            for tgt_id, edge in self.graph.get_outgoing(subject_cid):
                if edge.relation_type == rel_type_name and tgt_id != subject_cid:
                    tgt_node = self.graph.get_node(tgt_id)
                    if tgt_node is not None:
                        boost = subject_node.activation * 2.0 * edge.weight * edge.confidence
                        pred_mult = 1.0
                        if query_verb_word and hasattr(edge, 'predicate_token_id') and edge.predicate_token_id != -1:
                            try:
                                edge_verb = decode_token_fn(edge.predicate_token_id).lower().strip()
                                if edge_verb:
                                    w1 = query_verb_word.rstrip('s').rstrip('d')
                                    w2 = edge_verb.rstrip('s').rstrip('d')
                                    if w1 == w2 or w1 in w2 or w2 in w1:
                                        pred_mult = 2.5
                                    else:
                                        pred_mult = 0.4
                            except Exception:
                                pass
                        boost *= pred_mult
                        scores[tgt_id] = scores.get(tgt_id, 0.0) + boost
        return scores

    def n_hop_bfs(self, subject_cid: int, rel_type_name: str,
                   max_hops: int = 3, hop_base_boost: float = 8.0,
                   hop_decay: float = 0.7,
                   query_verb_word: str = "",
                   decode_token_fn=None) -> Dict[int, float]:
        """N-Hop BFS traversal for compositionality."""
        scores: Dict[int, float] = {}
        bfs_queue = deque()

        # Initialize with direct edges from subject
        for mid_cid, mid_edge in self.graph.get_outgoing(subject_cid):
            if mid_cid == subject_cid:
                continue
            mid_node = self.graph.get_node(mid_cid)
            if mid_node is None:
                continue
            hop_score = mid_edge.weight * mid_edge.confidence
            pred_mult = 1.0
            if query_verb_word and hasattr(mid_edge, 'predicate_token_id') and mid_edge.predicate_token_id != -1:
                try:
                    edge_verb = decode_token_fn(mid_edge.predicate_token_id).lower().strip()
                    if edge_verb:
                        w1 = query_verb_word.rstrip('s').rstrip('d')
                        w2 = edge_verb.rstrip('s').rstrip('d')
                        if w1 == w2 or w1 in w2 or w2 in w1:
                            pred_mult = 2.5
                        else:
                            pred_mult = 0.4
                except Exception:
                    pass
            hop_score *= pred_mult
            bfs_queue.append((mid_cid, hop_score, 1, {subject_cid}))

        while bfs_queue:
            nid, cum_score, depth, visited = bfs_queue.popleft()
            if depth > max_hops:
                continue
            for tgt_cid, tgt_edge in self.graph.get_outgoing(nid):
                if tgt_cid in visited or tgt_cid == subject_cid:
                    continue
                if depth >= 1 and tgt_edge.relation_type != rel_type_name:
                    continue

                pred_mult = 1.0
                if query_verb_word and hasattr(tgt_edge, 'predicate_token_id') and tgt_edge.predicate_token_id != -1:
                    try:
                        edge_verb = decode_token_fn(tgt_edge.predicate_token_id).lower().strip()
                        if edge_verb:
                            w1 = query_verb_word.rstrip('s').rstrip('d')
                            w2 = edge_verb.rstrip('s').rstrip('d')
                            if w1 == w2 or w1 in w2 or w2 in w1:
                                pred_mult = 2.5
                            else:
                                pred_mult = 0.4
                    except Exception:
                        pass

                edge_score = cum_score * tgt_edge.weight * tgt_edge.confidence * pred_mult
                final_score = edge_score * hop_base_boost * (hop_decay ** (depth - 1))
                scores[tgt_cid] = max(scores.get(tgt_cid, 0.0), final_score)

                if depth < max_hops:
                    new_visited = visited | {nid}
                    bfs_queue.append((tgt_cid, edge_score, depth + 1, new_visited))

        return scores

    def collect_active_concepts(self, subject_cid: int, relation_cid: int,
                                 rel_type_name: str,
                                 disable_spreading: bool = False) -> Dict[int, float]:
        """Collect and score all active concepts."""
        concept_scores: Dict[int, float] = {}

        if disable_spreading:
            for node in self.graph.nodes.values():
                node.activation = 0.0
            self.graph.activate(subject_cid, amount=1.0)
        else:
            # Already spread - just collect
            pass

        active_nodes = [(nid, node) for nid, node in self.graph.nodes.items()
                       if node.activation > 0.01 and nid != subject_cid and nid != relation_cid]

        matching_targets: Dict[int, float] = {}
        for nid, node in active_nodes:
            for tgt_id, edge in self.graph.get_outgoing(nid):
                type_match = (edge.relation_type == rel_type_name)
                base_score = node.activation * edge.weight * edge.confidence
                if type_match:
                    base_score *= 3.0
                else:
                    base_score *= 0.1
                if tgt_id in matching_targets:
                    matching_targets[tgt_id] = max(matching_targets[tgt_id], base_score)
                else:
                    matching_targets[tgt_id] = base_score

        # Cross-domain transfer from activated nodes
        if rel_type_name != "semantic":
            for nid, node in active_nodes:
                if node.activation < 0.01:
                    continue
                for tgt_id, edge in self.graph.get_outgoing(nid):
                    if edge.relation_type != rel_type_name or tgt_id == subject_cid:
                        continue
                    cross_score = node.activation * edge.weight * edge.confidence * 2.0
                    if tgt_id in matching_targets:
                        matching_targets[tgt_id] = max(matching_targets[tgt_id], cross_score)
                    else:
                        matching_targets[tgt_id] = cross_score

        return matching_targets

    def score_tokens_from_concepts(self, matching_targets: Dict[int, float],
                                    concept_to_embed, token_embed,
                                    token_embed_norms: np.ndarray,
                                    subject_tid: int,
                                    concept_dim: int, embed_dim: int) -> np.ndarray:
        """Batch concept-to-token scoring via cosine similarity."""
        vocab_size = token_embed.weight.data.shape[0]
        concept_scores = np.zeros(vocab_size, dtype=np.float32)

        batch_targets = []
        for tgt_cid, score in matching_targets.items():
            tgt_node = self.graph.get_node(tgt_cid)
            if tgt_node is None:
                continue
            # Check binding map first (exact match)
            # Note: binding map lookup would be done externally
            batch_targets.append((tgt_cid, score, tgt_node.vector))

        if batch_targets:
            tgt_vecs = np.stack([tv[2] for tv in batch_targets])
            # Project to embed_dim
            if concept_dim == embed_dim:
                tgt_embeds = tgt_vecs
            else:
                tgt_embeds = concept_to_embed(tgt_vecs).data
            tgt_norms = np.linalg.norm(tgt_embeds, axis=1)
            valid_tgt = tgt_norms > 0
            valid_tok = token_embed_norms > 0

            if np.any(valid_tgt) and np.any(valid_tok):
                normed_tgt = tgt_embeds.copy()
                normed_tgt[valid_tgt] /= tgt_norms[valid_tgt, np.newaxis]
                normed_tok = token_embed.weight.data.copy()
                normed_tok[valid_tok] /= token_embed_norms[valid_tok, np.newaxis]
                sim_matrix = normed_tgt @ normed_tok.T

                for i, (tgt_cid, score, _) in enumerate(batch_targets):
                    concept_scores += sim_matrix[i] * score * 0.3

        # Suppress self-prediction
        if 0 <= subject_tid < vocab_size:
            concept_scores[subject_tid] *= 0.1

        return concept_scores

    def get_prediction(self, active_concepts: List[int], top_k: int = 5) -> List[int]:
        """Get top-k predicted concept IDs from active nodes."""
        scored = [(cid, self.graph.nodes[cid].activation) for cid in active_concepts
                 if cid in self.graph.nodes]
        scored.sort(key=lambda x: -x[1])
        return [cid for cid, _ in scored[:top_k]]