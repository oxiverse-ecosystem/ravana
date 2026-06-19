"""
Plasticity - Hebbian, Anti-Hebbian, and Structural plasticity mechanisms.
Handles: edge weight updates, relation vector updates, concept vector updates.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

from ravana_ml.graph import ConceptGraph, ConceptEdge
from ravana_ml.plasticity import HebbianPlasticity, AntiHebbianPlasticity, StructuralPlasticity


class Plasticity:
    """Unified plasticity mechanisms for the concept graph."""

    def __init__(self, graph: ConceptGraph, base_lr: float = 0.005):
        self.graph = graph
        self.base_lr = base_lr
        self.hebbian = HebbianPlasticity(graph, lr=0.01)
        self.anti_hebbian = AntiHebbianPlasticity(graph, lr=0.01)
        self.structural = StructuralPlasticity(graph)

        # Cognitive state
        self._step_counter = 0
        self._sleep_pressure = 0.0
        self._sleep_interval = 100

        # Norm cache
        self._norm_cache: Dict[str, float] = {}
        self._token_embed_norms: Optional[np.ndarray] = None

        # Concept creation gating
        self._concept_similarity_threshold = 0.7
        self._max_concepts = 10000

        # Episodic memory
        self._episodic_triples: List[Tuple[int, int, int, float]] = []
        self._max_episodic = 500

        # Performance tracking
        self._train_correct = 0
        self._train_total = 0
        self._seen_predicates: Set[str] = set()
        self._last_loss = 0.0

        # Forward pass caches
        self._node_matrix_cache = None
        self._node_matrix_version = -1
        self._rel_vector_cache: Dict[str, np.ndarray] = {}
        self._rel_vector_version = -1

        # Cognitive currencies
        from ravana_ml.currencies import CognitiveCurrencies
        from ravana_ml.currency import create_rlm_currency
        self.currencies = CognitiveCurrencies()
        self.currency = create_rlm_currency()

        # Edge weight convergence
        self._edge_weight_ema = 0.0
        self._edge_weight_prev = 0.0
        self._token_hit_ema = 0.5

        self.sleep_cycles_completed = 0
        self.conceptual_accuracy = 0.0

        self._last_predicted_concepts: List[int] = []
        self._last_edge_pred: List[int] = []
        self._last_hidden_state: Optional[np.ndarray] = None

        # Replay buffers
        self._replay_buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        self._replay_buffer_max = 500
        self._replay_n_samples = 20
        self._domain_memories: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

        # Verb-stem offset predictor
        self._verb_offsets: Dict[str, np.ndarray] = {}
        self._verb_offset_count: Dict[str, int] = {}
        self._verb_accum_buffer: List[Tuple[str, np.ndarray]] = []
        self.use_verb_offset = False

        # Cross-domain edges
        self._cross_domain_edges_injected: Set[Tuple[int, int]] = set()

        # Predictive coding
        self.predictive_coding_enabled = True

    def learn(self, token_ids: np.ndarray, target_ids: np.ndarray,
              token_embed, subject_proj, concept_to_embed,
              binding_map, decode_token_fn,
              relation_predictor,
              sleep_consolidation) -> Dict[str, float]:
        """Full Hebbian learning from (context, target) pair."""
        from ravana_ml.tensor import tensor as make_tensor

        # Clear norm cache, pre-compute token embed norms
        self._norm_cache.clear()
        if self._token_embed_norms is None:
            self._token_embed_norms = np.linalg.norm(token_embed.weight.data, axis=1)

        # Flatten inputs
        if token_ids.ndim > 1:
            token_ids = token_ids.flatten()
        if target_ids.ndim > 1:
            target_ids = target_ids.flatten()

        input_ids = token_ids.tolist()
        target_id = int(target_ids.flatten()[0])

        # Forward pass with spreading
        self._training_mode = True
        # Note: actual forward would be called externally
        self._training_mode = False

        # Prediction error
        target_onehot = np.zeros(token_embed.weight.data.shape[0], dtype=np.float32)
        target_onehot[target_id] = 1.0

        # Decompose triple
        full_triple_ids = input_ids + [target_id]
        subject_ids, relation_ids, object_ids = self._decompose_triple(full_triple_ids)
        object_tid = target_id

        if not subject_ids:
            self._step_counter += 1
            return {"loss": 0.0, "accuracy": self._train_correct / max(1, self._train_total)}

        subject_tid = subject_ids[0]

        # Track seen predicates
        if relation_ids:
            try:
                pred_word = decode_token_fn(relation_ids[0]).lower().strip()
                if pred_word:
                    self._seen_predicates.add(pred_word)
                    relation_predictor.accumulate_verb_offset(subject_tid, object_tid, pred_word, token_embed)
            except Exception:
                pass

        # Get/create concept nodes
        subject_cid = self._get_or_create_concept(subject_tid, token_embed.weight.data[subject_tid],
                                                   token_embed, subject_proj, binding_map)
        object_cid = self._get_or_create_concept(object_tid, token_embed.weight.data[object_tid],
                                                  token_embed, subject_proj, binding_map)

        # Classify relation
        rel_type_idx, rel_type_embed = relation_predictor.classify_relation_learned(
            relation_ids, decode_token_fn, token_embed, subject_proj)
        rel_type_name = RELATION_TYPES[rel_type_idx]

        # Create relation concept
        if relation_ids:
            relation_tid = relation_ids[0]
            relation_embed = token_embed.weight.data[relation_tid]
            relation_cid = self._get_or_create_concept(relation_tid, relation_embed,
                                                        token_embed, subject_proj, binding_map)
        else:
            relation_cid = object_cid

        # Create relation-object concept
        if relation_ids:
            rel_obj_tid = 10000 + relation_ids[0] * 256 + object_tid
            rel_obj_embed = 0.5 * (relation_embed + token_embed.weight.data[object_tid])
            rel_obj_cid = self._get_or_create_concept(rel_obj_tid, rel_obj_embed,
                                                       token_embed, subject_proj, binding_map)
        else:
            rel_obj_cid = object_cid

        # Create/update edges
        edge_direct = self.graph.get_edge(subject_cid, object_cid)
        if edge_direct is None:
            edge_direct = self.graph.add_edge(source=subject_cid, target=object_cid,
                                              weight=0.3, relation_type=rel_type_name)

        edge_sr = self.graph.get_edge(subject_cid, rel_obj_cid)
        if edge_sr is None:
            edge_sr = self.graph.add_edge(source=subject_cid, target=rel_obj_cid,
                                          weight=0.3, relation_type=rel_type_name)

        edge_ro = self.graph.get_edge(rel_obj_cid, object_cid)
        if edge_ro is None:
            edge_ro = self.graph.add_edge(source=rel_obj_cid, target=object_cid,
                                          weight=0.3, relation_type=rel_type_name)

        # Reverse edge
        reverse_direct = self.graph.get_edge(object_cid, subject_cid)
        if reverse_direct is None:
            reverse_direct = self.graph.add_edge(source=object_cid, target=subject_cid,
                                                  weight=0.1, relation_type=rel_type_name)

        if relation_ids:
            edge_direct.predicate_token_id = relation_ids[0]
            edge_sr.predicate_token_id = relation_ids[0]
            edge_ro.predicate_token_id = relation_ids[0]

        # Validate edge bindings
        self._validate_edge_bindings(subject_cid, object_cid, rel_type_name, relation_ids)

        # Inject direct edges if needed
        self._inject_direct_edges_if_needed(subject_cid, object_cid, rel_type_name)

        # Hebbian edge updates
        src_node = self.graph.get_node(subject_cid)
        rel_obj_node = self.graph.get_node(rel_obj_cid)
        tgt_node = self.graph.get_node(object_cid)

        if src_node and rel_obj_node and tgt_node:
            rel_obj_node.activation = max(rel_obj_node.activation, 0.7)
            tgt_node.activation = max(tgt_node.activation, 0.8)

            # Direct edge
            delta = self.base_lr * src_node.activation * tgt_node.activation
            edge_direct.weight = np.clip(edge_direct.weight + delta, 0.0, 1.0)
            edge_direct.confidence = edge_direct.weight
            edge_direct.stability = min(1.0, edge_direct.stability + 0.01)

            # Subject -> rel_obj
            delta = self.base_lr * src_node.activation * rel_obj_node.activation
            edge_sr.weight = np.clip(edge_sr.weight + delta, 0.0, 1.0)
            edge_sr.confidence = edge_sr.weight
            edge_sr.stability = min(1.0, edge_sr.stability + 0.01)

            # rel_obj -> object
            delta = self.base_lr * rel_obj_node.activation * tgt_node.activation
            edge_ro.weight = np.clip(edge_ro.weight + delta, 0.0, 1.0)
            edge_ro.confidence = edge_ro.weight
            edge_ro.stability = min(1.0, edge_ro.stability + 0.01)

            # Relation vector updates
            tgt_vec = tgt_node.vector
            tgt_norm = np.linalg.norm(tgt_vec)
            if tgt_norm > 0:
                tgt_signal = tgt_vec / tgt_norm
                type_seed = ConceptEdge._init_relation_vector(rel_type_name, len(edge_ro.relation_vector))
                for e in [edge_sr, edge_ro]:
                    e.relation_vector = (0.70 * e.relation_vector +
                                         0.20 * tgt_signal[:len(e.relation_vector)] +
                                         0.10 * type_seed)
                    rv_norm = np.linalg.norm(e.relation_vector)
                    if rv_norm > 0:
                        e.relation_vector /= rv_norm
                    e._rv_norm_cache = None

            # Concept vector updates
            pull_lr = 0.02
            subject_concept_vec = subject_proj(token_embed.weight.data[subject_tid].reshape(1, -1)).data.flatten()
            src_delta = pull_lr * (subject_concept_vec - src_node.vector)
            src_node.vector += src_delta
            src_norm = np.linalg.norm(src_node.vector)
            if src_norm > 0:
                src_node.vector /= src_norm

            if relation_ids:
                rel_obj_concept_vec = subject_proj(rel_obj_embed.reshape(1, -1)).data.flatten()
                rel_obj_delta = pull_lr * (rel_obj_concept_vec - rel_obj_node.vector)
                rel_obj_node.vector += rel_obj_delta
                rel_obj_norm = np.linalg.norm(rel_obj_node.vector)
                if rel_obj_norm > 0:
                    rel_obj_node.vector /= rel_obj_norm

            object_concept_vec = subject_proj(token_embed.weight.data[object_tid].reshape(1, -1)).data.flatten()
            tgt_delta = pull_lr * (object_concept_vec - tgt_node.vector)
            tgt_node.vector += tgt_delta
            tgt_norm = np.linalg.norm(tgt_node.vector)
            if tgt_norm > 0:
                tgt_node.vector /= tgt_norm

            # Path-aware concept vector update
            path_lr = 0.005
            src_to_tgt = tgt_node.vector - src_node.vector
            path_delta_src = path_lr * src_to_tgt
            path_delta_tgt = -path_lr * src_to_tgt * 0.3
            src_node.vector += path_delta_src
            tgt_node.vector += path_delta_tgt
            for n in [src_node, tgt_node]:
                n_norm = np.linalg.norm(n.vector)
                if n_norm > 0:
                    n.vector /= n_norm

        # Update relation classifier
        relation_predictor._update_relation_classifier(relation_ids, rel_type_idx,
                                                        token_embed, subject_proj)

        # Predictive coding: update relevant edges only
        if self.predictive_coding_enabled:
            pc_lr = self.base_lr * 0.15
            obj_embed_norm = np.linalg.norm(token_embed.weight.data[object_tid])
            obj_signal = token_embed.weight.data[object_tid] / obj_embed_norm if obj_embed_norm > 0 else None
            relevant_cids = {subject_cid, object_cid}
            if relation_ids:
                relevant_cids.add(relation_cid)
                relevant_cids.add(rel_obj_cid)

            for (src_id, tgt_id), edge in list(self.graph.edges.items()):
                if src_id not in relevant_cids and tgt_id not in relevant_cids:
                    continue
                src_node = self.graph.nodes.get(src_id)
                tgt_node = self.graph.nodes.get(tgt_id)
                if src_node is None or tgt_node is None:
                    continue
                predicted = src_node.activation * edge.weight
                actual = tgt_node.activation
                if tgt_id == object_cid:
                    actual = max(actual, 0.5)
                if src_id == subject_cid:
                    actual_src = max(src_node.activation, 0.3)
                    predicted = actual_src * edge.weight
                error = actual - predicted
                w_delta = pc_lr * abs(error) * max(src_node.activation, 0.01)
                if error > 0:
                    edge.weight = min(1.0, edge.weight + w_delta * 0.3)
                else:
                    edge.weight = max(0.0, edge.weight - w_delta * 0.1)
                if abs(error) < 0.3:
                    edge.confidence = min(1.0, edge.confidence + 0.005)
                if tgt_id == object_cid and obj_signal is not None:
                    edge.relation_vector += pc_lr * 0.3 * obj_signal[:len(edge.relation_vector)]
                    rv_n = np.linalg.norm(edge.relation_vector)
                    if rv_n > 0:
                        edge.relation_vector /= rv_n
                    edge._rv_norm_cache = None

        # Store episodic triple
        self._episodic_triples.append((subject_cid, rel_type_idx, object_cid, time.time()))
        if len(self._episodic_triples) > self._max_episodic:
            self._episodic_triples = self._episodic_triples[-self._max_episodic:]

        # Auto-sleep check
        self._step_counter += 1
        if self._sleep_pressure > 0.7 and self._step_counter - getattr(self, '_last_sleep_step', 0) > 200:
            sleep_consolidation.run_cycle(self.graph, [], self._episodic_triples, None, [], None, [], {}, 0.7, 0.05)
            self._last_sleep_step = self._step_counter
        if self._sleep_interval > 0 and self._step_counter % self._sleep_interval == 0:
            sleep_consolidation.run_cycle(self.graph, [], self._episodic_triples, None, [], None, [], {}, 0.7, 0.05)

        # Train relation predictor MLP
        relation_predictor.rp_forward(subject_tid, rel_type_idx, token_embed)
        relation_predictor.rp_backward(target_id)

        return {"loss": self._last_loss, "accuracy": self._train_correct / max(1, self._train_total)}

    def _decompose_triple(self, token_ids: List[int]) -> Tuple[List[int], List[int], List[int]]:
        n = len(token_ids)
        if n >= 3:
            return [token_ids[0]], token_ids[1:-1], [token_ids[-1]]
        elif n == 2:
            return [token_ids[0]], [], [token_ids[1]]
        elif n == 1:
            return [token_ids[0]], [], []
        return [], [], []

    def _get_or_create_concept(self, token_id: int, embed_vec: np.ndarray,
                                token_embed, subject_proj, binding_map) -> int:
        bindings = binding_map.get_concepts(token_id, min_confidence=0.1)
        if bindings:
            cid = bindings[0].concept_id
            if self.graph.get_node(cid) is not None:
                return cid
        concept_vec = subject_proj(embed_vec.reshape(1, -1)).data.flatten()
        if len(self.graph.nodes) < self._max_concepts:
            node = self.graph.add_node(vector=concept_vec, label=f"tok_{token_id}")
            return node.id
        else:
            results = self.graph.find_similar(concept_vec, k=1)
            if results:
                return results[0][0]
            node = self.graph.add_node(vector=concept_vec, label=f"tok_{token_id}")
            return node.id

    def _validate_edge_bindings(self, subject_cid: int, object_cid: int,
                                 rel_type_name: str, relation_ids: List[int]):
        subj_bindings = self.binding_map.get_tokens(subject_cid, min_confidence=0.1)
        obj_bindings = self.binding_map.get_tokens(object_cid, min_confidence=0.1)
        if not subj_bindings or not obj_bindings:
            return
        direct_edge = self.graph.get_edge(subject_cid, object_cid)
        if direct_edge and relation_ids and direct_edge.predicate_token_id != relation_ids[0]:
            direct_edge.predicate_token_id = relation_ids[0]
            direct_edge.confidence = max(0.1, direct_edge.confidence * 0.5)

    def _inject_direct_edges_if_needed(self, subject_cid: int, object_cid: int,
                                        rel_type_name: str, threshold: float = 0.5):
        direct_edge = self.graph.get_edge(subject_cid, object_cid)
        if direct_edge is None:
            self.graph.add_edge(source=subject_cid, target=object_cid,
                                weight=0.7, relation_type=rel_type_name)
        elif direct_edge.weight < threshold:
            direct_edge.weight = threshold
            direct_edge.confidence = max(direct_edge.confidence, 0.6)

    def anti_hebbian_prune(self, mismatch_threshold: float = 0.3,
                            min_prediction_count: int = 5) -> int:
        pruned = 0
        for (src_id, tgt_id), edge in list(self.graph.edges.items()):
            if edge.prediction_count >= min_prediction_count:
                pred_ratio = edge.forward_pred_count / edge.prediction_count
                if pred_ratio < (1.0 - mismatch_threshold):
                    self.graph.anti_hebbian_update(src_id, tgt_id, lr=0.02)
                    pruned += 1
                    if edge.weight < 0.1 and edge.confidence < 0.1:
                        self.graph.remove_edge(src_id, tgt_id)
        return pruned

    # Properties for CognitiveCurrencies compatibility
    @property
    def identity_strength(self): return self.currencies.identity_strength
    @identity_strength.setter
    def identity_strength(self, val): self.currencies.identity_strength = val

    @property
    def sleep_pressure(self): return self.currencies.sleep_pressure
    @sleep_pressure.setter
    def sleep_pressure(self, val): self.currencies.sleep_pressure = val

    def _regulate_cognitive_state(self):
        self.currencies.regulate()

    def save(self, path: str):
        import pickle
        state = self.state_dict()
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def state_dict(self) -> dict:
        return {
            "graph_nodes": {nid: {"vector": n.vector.copy(), "core_vector": n.core_vector.copy(),
                                  "genesis_vector": getattr(n, "genesis_vector", n.vector).copy(),
                                  "label": n.label, "activation": n.activation,
                                  "confidence": n.confidence, "stability": n.stability}
                           for nid, n in self.graph.nodes.items()},
            "graph_edges": {(s, t): {"weight": e.weight, "confidence": e.confidence,
                                     "relation_type": e.relation_type,
                                     "relation_vector": e.relation_vector.copy(),
                                     "predicate_token_id": e.predicate_token_id,
                                     "prediction_count": e.prediction_count}
                           for (s, t), e in self.graph.edges.items()},
            "_episodic_triples": self._episodic_triples.copy(),
            "_step_counter": self._step_counter,
            "_train_correct": self._train_correct,
            "_train_total": self._train_total,
            "_seen_predicates": sorted(self._seen_predicates),
        }

import time