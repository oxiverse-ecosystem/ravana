import numpy as np
import time
from typing import Optional, List, Tuple, Dict, Set
from ..tensor import StateTensor, RawTensor, tensor, Parameter
from ..graph import ConceptGraph
from ..propagation import PropagationEngine
from ..pressure import PressureAccumulator
from ..plasticity import HebbianPlasticity, AntiHebbianPlasticity, StructuralPlasticity
from . import functional as F
from .module import Module, Linear, Embedding


class RLM(Module):
    """
    Recursive Learning Model (RLM)

    An alternative to the traditional LLM. Instead of transformer attention
    and backprop, RLM uses concept graphs, Hebbian plasticity, competitive
    inhibition, and pressure-driven sleep cycles. Maps input sequences to
    conceptual trajectories in a ConceptGraph.
    """
    def __init__(self, vocab_size: int, embed_dim: int, concept_dim: int,
                 n_concepts: int, n_hidden: int, n_layers: int = 1,
                 max_seq_len: int = 128, pressure_threshold: float = 8.0,
                 sleep_interval: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.concept_dim = concept_dim
        self.n_concepts = n_concepts
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.pressure_threshold = pressure_threshold
        self.sleep_interval = sleep_interval

        # Core layers
        self.token_embed = Embedding(vocab_size, embed_dim)
        self._init_structured_embeddings()

        self.recurrent_cell = Linear(embed_dim + n_hidden, n_hidden)
        self.hidden_layers = []
        for i in range(n_layers - 1):
            layer = Linear(n_hidden, n_hidden)
            self.hidden_layers.append(layer)
            self.register_module(f'hidden_{i}', layer)
        
        # Prediction heads
        self.concept_predictor = Linear(n_hidden, concept_dim)
        self.context_logits = Linear(n_hidden, vocab_size, bias=True)

        # Concept graph: more concepts than tokens for clustering
        actual_n = max(n_concepts, vocab_size * 2)
        self.graph = ConceptGraph(dim=concept_dim, max_nodes=actual_n * 2)
        self._init_structured_concepts()

        self.propagation = PropagationEngine(self.graph)
        self.pressure = PressureAccumulator(self.graph)
        self.hebbian = HebbianPlasticity(self.graph, lr=0.03)
        self.anti_hebbian = AntiHebbianPlasticity(self.graph, lr=0.02)
        self.structural = StructuralPlasticity(self.graph,
                                                prune_threshold=0.005,
                                                form_threshold=0.3)

        self._last_predicted_concepts: List[int] = []
        self._last_input_concepts: List[int] = []
        self._last_edge_pred: List[int] = []
        self._last_hidden_state: Optional[np.ndarray] = None
        self._last_ctx_logits: Optional[np.ndarray] = None
        self.sleep_cycles_completed = 0
        self.total_pressure = 0.0
        self.conceptual_accuracy = 0.0
        self.n_predictions = 0
        self._step_counter = 0

        self._edges_learned = 0

        # Cached token→concept mapping (updated during sleep)
        self._token_concept_map: List[int] = [-1] * vocab_size
        self._update_token_concept_map()

        # Context modulation strength
        self.context_bias = 0.5
        self.context_scale = 0.0  # disabled until trained

    def _init_structured_embeddings(self):
        n, d = self.vocab_size, self.embed_dim
        vectors = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            angle = 2.0 * np.pi * i / n
            vectors[i, 0] = np.cos(angle)
            vectors[i, 1] = np.sin(angle)
            if d > 2:
                vectors[i, 2:] = np.random.randn(d - 2).astype(np.float32) * 0.02
        self.token_embed.weight.data = vectors

    def _init_structured_concepts(self):
        token_vecs = self.token_embed.weight.data.copy()  # (n, d)
        n_token = min(self.vocab_size, self.n_concepts)

        # Phase 1: one concept per token, seeded from token embedding
        for i in range(n_token):
            vec = token_vecs[i] / (np.linalg.norm(token_vecs[i]) + 1e-15)
            self.graph.add_node(vec, label=f"tok_{i}")

        # Phase 2: extra concepts as interpolations between adjacent token pairs
        remaining = self.n_concepts - n_token
        if remaining > 0:
            for j in range(remaining):
                t1 = j % n_token
                t2 = (t1 + 1) % n_token
                alpha = np.random.uniform(0.3, 0.7)
                vec = alpha * token_vecs[t1] + (1 - alpha) * token_vecs[t2]
                vec = vec / (np.linalg.norm(vec) + 1e-15)
                self.graph.add_node(vec, label=f"c{n_token + j}")

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _nearest_concept(self, embed_vec: np.ndarray) -> int:
        best_id, best_sim = -1, -1.0
        for nid, node in self.graph.nodes.items():
            sim = float(np.dot(node.vector, embed_vec) / (np.linalg.norm(node.vector) * np.linalg.norm(embed_vec) + 1e-15))
            if sim > best_sim:
                best_sim = sim
                best_id = nid
        return best_id

    def _nearest_concepts(self, embed_vec: np.ndarray, k: int = 3) -> List[int]:
        sims = []
        for nid, node in self.graph.nodes.items():
            sim = float(np.dot(node.vector, embed_vec) / (np.linalg.norm(node.vector) * np.linalg.norm(embed_vec) + 1e-15))
            sims.append((nid, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in sims[:k]]

    # ──────────────────────────────────────────────────────────────
    # Recurrence & Forward
    # ──────────────────────────────────────────────────────────────

    def forward(self, token_ids: np.ndarray) -> StateTensor:
        """
        token_ids: (batch=1, seq_len)
        returns: (vocab_size) prediction logits
        """
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]
        T = token_ids.shape[1]
        h = np.zeros(self.n_hidden, dtype=np.float32)
        
        context_concepts = []

        for t in range(T):
            tid = int(token_ids[0, t])
            x = self.token_embed(StateTensor(np.array([tid]))).data[0]
            
            # Update nearest concept for propagation
            nid = self._nearest_concept(x)
            if nid >= 0:
                context_concepts.append(nid)
            
            # Recurrent step
            combined = np.concatenate([x, h])
            h = self.recurrent_cell(StateTensor(combined[np.newaxis, :])).data[0]
            h = np.tanh(h)
            for layer in self.hidden_layers:
                h = layer(StateTensor(h[np.newaxis, :])).data[0]
                h = np.tanh(h)

        self._last_hidden_state = h
        
        # Concept prediction from hidden state
        z = self.concept_predictor(StateTensor(h[np.newaxis, :])).data[0]
        z_norm = z / (np.linalg.norm(z) + 1e-15)
        
        # Activation based on conceptual similarity
        self.graph.reset_activation()
        edge_pred_list = []
        for nid, node in self.graph.nodes.items():
            sim = np.dot(node.vector, z_norm)
            if sim > 0.4:
                self.graph.activate(nid, sim)
                edge_pred_list.append(nid)
        
        self.graph.spread_activation(steps=2, k_active=7, decay=0.5)
        
        # Scoring vocab based on active concepts (using vector similarity for better resolution)
        token_vecs = self.token_embed.weight.data
        token_norms = token_vecs / (np.linalg.norm(token_vecs, axis=1, keepdims=True) + 1e-15)
        
        concept_scores = -np.ones(self.vocab_size, dtype=np.float32) * 1e9
        
        all_active = [n for n in sorted(self.graph.nodes.values(),
                                         key=lambda n: n.activation, reverse=True)
                      if n.activation > 0.01][:7]
        
        for node in all_active:
            if node.id >= self.vocab_size:
                continue
            norm = np.linalg.norm(node.vector) + 1e-15
            vec_norm = node.vector / norm
            local = (token_norms @ vec_norm) * node.activation
            concept_scores = np.maximum(concept_scores, local)

        self._last_predicted_concepts = [n.id for n in all_active][:5]
        self._last_edge_pred = self.propagation.get_prediction(self._last_predicted_concepts, top_k=5)

        # ── Context Priming with Temporal Decay ──
        if T > 1:
            for tok_id in range(self.vocab_size):
                if tok_id == int(token_ids[0, -1]):
                    continue
                
                tok_concept = self._token_concept_map[tok_id]
                if tok_concept < 0:
                    x_tok = self.token_embed(StateTensor(np.array([tok_id]))).data[0]
                    tok_concept = self._nearest_concept(x_tok)
                
                for i, ctx_nid in enumerate(context_concepts):
                    ce = self.graph.get_edge(ctx_nid, tok_concept)
                    if ce is not None and ce.weight > 0.01:
                        # Temporal decay: more recent context is stronger
                        # i ranges from 0 to T-1. Context tokens are from 0 to T-1.
                        dist = T - 1 - i
                        decay = 0.8 ** dist
                        
                        boost = ce.weight * decay
                        if concept_scores[tok_id] < -1e8:
                            concept_scores[tok_id] = boost * 0.5
                        else:
                            concept_scores[tok_id] *= (1.0 + boost)
        

        concept_scores = np.maximum(concept_scores, -1e8)
        concept_logits = concept_scores * 15.0

        # Context path: hidden state predicts token logits
        ctx_logits_raw = self.context_logits(StateTensor(h[np.newaxis, :]))
        ctx_logits = ctx_logits_raw.data.flatten()

        logits = concept_logits + ctx_logits * self.context_scale
        self._last_ctx_logits = ctx_logits
        return StateTensor(logits[np.newaxis, :])[0]

    def learn(self, token_ids: np.ndarray, next_token_ids: np.ndarray):
        self._step_counter += 1
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]

        logits_tensor = self.forward(token_ids)
        
        next_id = int(next_token_ids[0]) if next_token_ids.ndim == 1 else int(next_token_ids[0, 0])
        last_input_id = int(token_ids[0, -1])

        next_embed = self.token_embed(StateTensor(np.array([next_id]))).data[0]

        input_concept = self._nearest_concept(
            self.token_embed(StateTensor(np.array([last_input_id]))).data[0])
        output_concept = self._nearest_concept(next_embed)

        if input_concept >= 0 and output_concept >= 0 and input_concept != output_concept:
            edge = self.graph.get_or_create_edge(input_concept, output_concept, weight=0.3)
            edge.weight = min(1.0, edge.weight + 0.05)
            edge.confidence = min(1.0, edge.confidence + 0.03)
            edge.prediction_count += 1
            self._edges_learned += 1
            self._competitive_inhibition(input_concept, output_concept, 0.05)

        # Shortcut edges
        T = token_ids.shape[1]
        if T > 1:
            for t in range(T - 1):
                ctx_id = int(token_ids[0, t])
                ctx_concept = self._nearest_concept(
                    self.token_embed(StateTensor(np.array([ctx_id]))).data[0])
                if ctx_concept >= 0 and ctx_concept != output_concept and ctx_concept != input_concept:
                    cedge = self.graph.get_or_create_edge(ctx_concept, output_concept, weight=0.1, shortcut=True)
                    cedge.weight = min(0.8, cedge.weight + 0.03)
                    cedge.confidence = min(0.8, cedge.confidence + 0.02)
                    cedge.prediction_count += 1

        predicted_set = set(self._last_predicted_concepts)
        actual_set = set(self._nearest_concepts(next_embed, k=5))
        n_overlap = len(predicted_set & actual_set)
        n_union = max(1, len(predicted_set | actual_set))
        overlap_ratio = n_overlap / n_union
        conceptual_error = 1.0 - overlap_ratio
        self.pressure.accumulate_semantic(conceptual_error * 1.5, salience=0.5)

        edge_pred_set = set(self._last_edge_pred)
        single_correct = output_concept in edge_pred_set

        if not single_correct and input_concept >= 0:
            inode = self.graph.get_node(input_concept)
            if inode:
                inode.contradiction_count += 1
                inode.pressure += 0.5

        if single_correct and input_concept >= 0:
            inode = self.graph.get_node(input_concept)
            if inode and inode.contradiction_count > 0:
                inode.contradiction_count = max(0, inode.contradiction_count - 1)
                inode.pressure = max(0.0, inode.pressure - 0.3)

        if T > 1:
            target_onehot = np.zeros(self.vocab_size, dtype=np.float32)
            target_onehot[next_id] = 1.0
            ctx_logits = self._last_ctx_logits
            ctx_dist = F.softmax(StateTensor(ctx_logits[np.newaxis, :]), dim=-1).data.flatten()
            ctx_error = target_onehot - ctx_dist
            err_tensor = StateTensor(ctx_error[np.newaxis, :])
            err_tensor._salience = 3.0
            self.context_logits.accumulate_pressure(err_tensor)

            h_error = self.context_logits.backprop(ctx_error[np.newaxis, :])
            for layer in reversed(self.hidden_layers):
                h_error = layer.backprop(h_error)
                layer.accumulate_pressure(StateTensor(h_error))

        logit_dist = F.softmax(logits_tensor, dim=-1).data.flatten()
        entropy = -np.sum(logit_dist * np.log(logit_dist + 1e-15))
        entropy /= np.log(self.vocab_size)
        self.pressure.accumulate_episodic(entropy * 0.3)

        self.total_pressure = self.pressure.total
        factor = 0.95 if single_correct else 0.05
        self.conceptual_accuracy = 0.9 * self.conceptual_accuracy + 0.1 * factor
        self.n_predictions += 1

        if self._step_counter % self.sleep_interval == 0:
            self.sleep_cycle()

        return conceptual_error

    def _update_token_concept_map(self):
        for tid in range(self.vocab_size):
            emb = self.token_embed(StateTensor(np.array([tid]))).data[0]
            best_id, best_sim = -1, -1.0
            for nid, node in self.graph.nodes.items():
                if nid < self.vocab_size:
                    sim = float(np.dot(emb, node.vector) / (np.linalg.norm(emb) * np.linalg.norm(node.vector) + 1e-15))
                    if sim > best_sim:
                        best_sim = sim
                        best_id = nid
            self._token_concept_map[tid] = best_id

    def _competitive_inhibition(self, source: int, target: int, amount: float):
        for (s, t), e in list(self.graph.edges.items()):
            if s == source and t != target and not e.shortcut:
                e.weight = max(0.0, e.weight - amount * 0.3 * e.weight)
                e.confidence = max(0.0, e.confidence - amount * 0.15)

    def _normalize_outgoing_weights(self, budget: float = 3.0):
        source_weights: Dict[int, float] = {}
        for (s, t), e in list(self.graph.edges.items()):
            source_weights[s] = source_weights.get(s, 0.0) + e.weight
        for s, total in source_weights.items():
            if total > budget:
                scale = budget / total
                for (s2, t), e in list(self.graph.edges.items()):
                    if s2 == s and not e.shortcut:
                        e.weight *= scale
                        if e.weight < 0.005:
                            self.graph.remove_edge(s, t)

    def sleep_cycle(self):
        self._normalize_outgoing_weights()
        self.graph.spread_activation(steps=3)
        self.structural.step()
        self.sleep_cycles_completed += 1

    def __repr__(self):
        return (f"RLM(vocab={self.vocab_size}, embed={self.embed_dim}, "
                f"concepts={len(self.graph.nodes)}, edges={len(self.graph.edges)}, "
                f"sleep={self.sleep_cycles_completed}, acc={self.conceptual_accuracy:.3f}, "
                f"learned_edges={self._edges_learned})")
