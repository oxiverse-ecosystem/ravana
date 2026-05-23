import numpy as np
import time
import pickle
import json
import zipfile
import os
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Set
from ..tensor import StateTensor, RawTensor, tensor, Parameter
from ..graph import ConceptGraph, ConceptBindingMap
from ..propagation import PropagationEngine
from ..free_energy import FreeEnergyAccumulator
from ..plasticity import HebbianPlasticity, AntiHebbianPlasticity, StructuralPlasticity
from . import functional as F
from .module import Module, Linear, Embedding, LayerNorm, GRUCell


class RLM(Module):
    """
    Recursive Learning Model (RLM)

    An alternative to the traditional LLM. Instead of transformer attention
    and backprop, RLM uses concept graphs, Hebbian plasticity, competitive
    inhibition, and free-energy-driven sleep cycles. Maps input sequences to
    conceptual trajectories in a ConceptGraph.
    """
    def __init__(self, vocab_size: int, embed_dim: int, concept_dim: int,
                 n_concepts: int, n_hidden: int, n_layers: int = 3,
                 max_seq_len: int = 128, free_energy_threshold: float = 8.0,
                 sleep_interval: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.concept_dim = concept_dim
        self.n_concepts = n_concepts
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.free_energy_threshold = free_energy_threshold
        self.sleep_interval = sleep_interval

        # Core layers
        self.token_embed = Embedding(vocab_size, embed_dim)
        self._init_structured_embeddings()

        # Sinusoidal positional encoding
        max_len = 512
        pe = np.zeros((max_len, embed_dim), dtype=np.float32)
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pe[:, 0::2] = np.sin(position * div_term[:pe[:, 0::2].shape[1]])
        pe[:, 1::2] = np.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self._positional_encoding = pe

        self.recurrent_cell = GRUCell(embed_dim, n_hidden)
        self.hidden_layers = []
        self.hidden_norms = []
        for i in range(n_layers - 1):
            layer = Linear(n_hidden, n_hidden)
            self.hidden_layers.append(layer)
            self.register_module(f'hidden_{i}', layer)
            norm = LayerNorm(n_hidden)
            self.hidden_norms.append(norm)
            self.register_module(f'hidden_norm_{i}', norm)

        # Prediction heads
        self.concept_predictor = Linear(n_hidden, concept_dim)
        self.context_logits = Linear(n_hidden, vocab_size, bias=True)

        # Concept attention: active concepts attend to each other
        self.attn_W_q = Linear(concept_dim, concept_dim)
        self.attn_W_k = Linear(concept_dim, concept_dim)
        self.attn_W_v = Linear(concept_dim, concept_dim)

        # Concept graph: more concepts than tokens for clustering
        actual_n = max(n_concepts, vocab_size * 2)
        self.graph = ConceptGraph(dim=concept_dim, max_nodes=actual_n * 2)
        self._init_structured_concepts()

        self.propagation = PropagationEngine(self.graph)
        self.free_energy_engine = FreeEnergyAccumulator(self.graph)
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
        self._last_node_sims: List[Tuple[int, float]] = []
        self.sleep_cycles_completed = 0
        self.total_free_energy = 0.0
        self.conceptual_accuracy = 0.0
        self.n_predictions = 0
        self._step_counter = 0
        self._seq_position = 0  # tracks position for positional encoding in forward/forward_step
        self._last_regulation: Dict[str, Any] = {}

        self._edges_learned = 0

        # Vector update rate limiting (prevents oscillation from noisy single-sample updates)
        self._vector_update_interval = 5  # update concept vectors every N steps

        # Learning rate scheduling (warmup + cosine decay)
        self._warmup_steps = 100
        self._base_lr = 0.001  # 10x increase (was 0.0001) — Hebbian needs faster local learning

        # Token → concept binding (probabilistic, multi-meaning)
        self.binding_map = ConceptBindingMap()

        # Cached token→concept mapping (updated during sleep)
        self._token_concept_map: List[int] = [-1] * vocab_size
        self._update_token_concept_map()

        # Inverted index: concept_id -> set of token_ids bound to it
        # Used for fast context priming (avoids O(V*T) scan)
        self._concept_to_tokens: Dict[int, Set[int]] = defaultdict(set)
        self._rebuild_concept_to_tokens()

        # Context modulation strength
        self.context_bias = 0.5
        self.context_scale = 1.0  # active — context_logits head is trained by settle loop

        # Predictive coding config
        self.settle_steps = 5       # inference settling iterations
        self.settle_lr = 0.05       # state update rate during settling
        self.settle_damping = 0.9   # prevents oscillation
        self.noise_sigma = 0.0      # noise injection (0 during learning, >0 during REM)
        self._running_avg_states = None  # for energy floor / anti-collapse

        # ═══════════════════════════════════════════════════════════
        # Cognitive State (native to RLM — no external module deps)
        # ═══════════════════════════════════════════════════════════

        # Identity (from IdentityEngine)
        self.identity_strength = 0.5        # [0,1] — self-concept coherence
        self.identity_momentum = 0.0        # directional inertia
        self.identity_history: List[float] = []  # last 100 values

        # Emotion (from VADEmotionEngine — 3D VAD via differential equations)
        self.valence = 0.0                   # [-1,1] — positive/negative affect
        self.arousal = 0.3                   # [0,1] — activation level (baseline=0.3)
        self.dominance = 0.5                 # [0,1] — sense of control

        # Meaning (from MeaningEngine)
        self.accumulated_meaning = 0.0       # running meaning total
        self.meaning_history: List[float] = []  # last 100 values

        # Sleep pressure (from SleepConsolidation + GlobalWorkspace)
        self.sleep_pressure = 0.0            # [0,1] — accumulates from free energy + contradictions
        self.sleep_pressure_threshold = 0.7  # when to trigger auto-sleep

        # Regulation (lightweight Governor)
        self.regulation_mode = "NORMAL"      # NORMAL, EXPLORATION, RESOLUTION, RECOVERY, PLATEAU
        self.dissonance_ema = 0.5            # exponential moving average of prediction error

        # Native Memory (lightweight episodic buffer + semantic consolidation)
        self._episodic_buffer: List[Dict] = []    # recent experiences
        self._episodic_buffer_max = 100
        self._semantic_memories: Dict[int, Dict] = {}  # consolidated: {concept_id: {strength, access_count, last_access}}
        self._semantic_memory_max = 1000

        # Concept emotion tags (from VADEmotionEngine._concept_tags)
        self._concept_vad: Dict[int, Tuple[float, float, float]] = {}  # {concept_id: (v, a, d)}

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
        d = self.concept_dim  # concept vectors live in concept_dim space (matches graph.dim and attn layers)
        n = self.n_concepts
        scale = np.sqrt(d)  # scale angle dims so they dominate after normalization

        # Phase 1: one concept per token, seeded from angular position in concept space
        for i in range(n):
            token_idx = int(i * self.vocab_size / n) if n > 0 else i
            angle = 2.0 * np.pi * i / n
            vec = np.zeros(d, dtype=np.float32)
            vec[0] = np.cos(angle) * scale
            vec[1] = np.sin(angle) * scale
            if d > 2:
                vec[2:] = np.random.randn(d - 2).astype(np.float32) * 0.02
            vec = vec / (np.linalg.norm(vec) + 1e-15)
            self.graph.add_node(vec, label=f"tok_{token_idx}")

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _project_to_concept(self, embed_vec: np.ndarray) -> np.ndarray:
        """Project an embed_dim vector to concept_dim space (truncate or zero-pad)."""
        cd = self.concept_dim
        if len(embed_vec) == cd:
            return embed_vec
        elif len(embed_vec) > cd:
            return embed_vec[:cd]
        else:
            out = np.zeros(cd, dtype=np.float32)
            out[:len(embed_vec)] = embed_vec
            return out

    def _project_to_embed(self, concept_vec: np.ndarray) -> np.ndarray:
        """Project a concept_dim vector to embed_dim space (truncate or zero-pad)."""
        ed = self.embed_dim
        if len(concept_vec) == ed:
            return concept_vec
        elif len(concept_vec) > ed:
            return concept_vec[:ed]
        else:
            out = np.zeros(ed, dtype=np.float32)
            out[:len(concept_vec)] = concept_vec
            return out

    def _nearest_concept(self, embed_vec: np.ndarray) -> int:
        cvec = self._project_to_concept(embed_vec)
        results = self.graph.find_similar(cvec, k=1)
        return results[0][0] if results else -1

    def _nearest_concepts(self, embed_vec: np.ndarray, k: int = 3) -> List[int]:
        cvec = self._project_to_concept(embed_vec)
        results = self.graph.find_similar(cvec, k=k)
        return [r[0] for r in results]

    def _get_lr_scale(self) -> float:
        """Learning rate scale: warmup then cosine decay."""
        if self._step_counter < self._warmup_steps:
            return self._step_counter / max(1, self._warmup_steps)
        progress = (self._step_counter - self._warmup_steps) / max(1, 10000 - self._warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    def concept_attention(self, active_nodes: list):
        """Active concepts attend to each other — Hebbian attention.

        QKV attention with graph-based mask: connected concepts get bonus,
        inhibitory edges get penalty. O(n_active^2 * d) — small since n_active <= 7.
        """
        if len(active_nodes) < 2:
            return
        vectors = np.array([n.vector for n in active_nodes], dtype=np.float32)
        Q = self.attn_W_q(StateTensor(vectors)).data
        K = self.attn_W_k(StateTensor(vectors)).data
        V = self.attn_W_v(StateTensor(vectors)).data
        # Ensure ndarray
        Q = np.asarray(Q) if not isinstance(Q, np.ndarray) else Q
        K = np.asarray(K) if not isinstance(K, np.ndarray) else K
        V = np.asarray(V) if not isinstance(V, np.ndarray) else V
        d_k = Q.shape[-1]
        scores = (Q @ K.T) / np.sqrt(d_k)
        # Graph-based mask: connected concepts get bonus, inhibitory get penalty
        for i, ni in enumerate(active_nodes):
            for j, nj in enumerate(active_nodes):
                if i == j:
                    continue
                edge = self.graph.get_edge(ni.id, nj.id)
                if edge:
                    scores[i, j] += edge.weight * 2.0
                    if edge.edge_type == "inhibitory":
                        scores[i, j] -= 3.0
        # Softmax along last axis
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-10)
        attended = weights @ V
        # Update concept vectors with attended representation
        for i, node in enumerate(active_nodes):
            delta = attended[i] - node.vector
            self.graph.adjust_vector(node.id, delta, lr=0.01)

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
        self._seq_position = 0  # reset position counter for this sequence
        
        context_concepts = []
        embed_raw = self.token_embed.embed_raw

        for t in range(T):
            tid = int(token_ids[0, t])
            x = embed_raw(tid)
            # Add positional encoding
            pos = self._seq_position % len(self._positional_encoding)
            x = x + self._positional_encoding[pos]
            self._seq_position += 1

            # Update nearest concept for propagation
            nid = self._nearest_concept(x)
            if nid >= 0:
                context_concepts.append(nid)
            
            # Recurrent step (GRU cell with gating)
            h = self.recurrent_cell(x, h)
            for i, layer in enumerate(self.hidden_layers):
                h_res = layer.forward_raw(h[np.newaxis, :])[0]
                h_res = self.hidden_norms[i].forward_raw(h_res)
                h_res = np.tanh(h_res)
                h = h + h_res  # residual connection

        self._last_hidden_state = h
        
        # Concept prediction from hidden state → concept_dim (matches graph node vectors)
        z = self.concept_predictor.forward_raw(h[np.newaxis, :])[0]
        z_norm = z / (np.linalg.norm(z) + 1e-15)

        # Activation based on conceptual similarity (from hidden state prediction)
        # Use vectorized matrix multiply instead of per-node Python loop
        self.graph.reset_activation()
        if self.graph._vectors_dirty or self.graph._vector_matrix_normed is None:
            self.graph._rebuild_vector_matrix()
        if self.graph._vector_matrix_normed is not None and len(self.graph._node_id_order) > 0:
            sims = self.graph._vector_matrix_normed @ z_norm.astype(np.float32)  # (N,)
            # Build node_sims list from vectorized result
            node_sims = [(self.graph._node_id_order[i], float(sims[i]))
                         for i in range(len(self.graph._node_id_order))]
            # Partial sort for top entries using argpartition (O(N) vs O(N log N))
            k_top = min(20, len(sims))
            top_idx = np.argpartition(sims, -k_top)[-k_top:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
            self._last_node_sims = [(self.graph._node_id_order[i], float(sims[i])) for i in top_idx]
        else:
            node_sims = []
            self._last_node_sims = []

        edge_pred_list = []
        for nid, sim in self._last_node_sims[:7]:
            self.graph.activate(nid, max(0.01, sim))
            edge_pred_list.append(nid)
        
        self.graph.spread_activation(steps=2, k_active=7, decay=0.5)

        # Concept attention: active concepts attend to each other (global context)
        active_concepts = [n for n in self.graph.nodes.values() if n.activation > 0.01]
        active_concepts.sort(key=lambda n: n.activation, reverse=True)
        # concept_attention is deferred to learn() — it modifies vectors and is a learning operation

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
            vec_norm = node.vector / (np.linalg.norm(node.vector) + 1e-15)
            # Project concept_dim vector to embed_dim for token comparison
            vec_embed_norm = self._project_to_embed(vec_norm)
            vec_embed_norm = vec_embed_norm / (np.linalg.norm(vec_embed_norm) + 1e-15)
            local = (token_norms @ vec_embed_norm) * node.activation
            concept_scores = np.maximum(concept_scores, local)

        self._last_predicted_concepts = [n.id for n in all_active][:5]
        self._last_edge_pred = self.propagation.get_prediction(self._last_predicted_concepts, top_k=5)

        # ── Multi-hop Edge Traversal (Issues 1+2: cross-domain & relational transfer) ──
        # For each active concept, follow outgoing edges to neighbor concepts,
        # then score tokens bound to those neighbors. Enables 2-hop inference:
        # e.g., "vexol" → warm (edge) → pleasant (bound token)
        hop_decay = 0.6  # each hop reduces signal by 40%
        for node in all_active:
            outgoing = self.graph._outgoing.get(node.id, [])
            for tgt_id, edge in outgoing:
                if edge.edge_type == "inhibitory" or edge.weight < 0.05:
                    continue
                tgt_node = self.graph.get_node(tgt_id)
                if tgt_node is None:
                    continue
                hop_score = node.activation * edge.weight * hop_decay
                bound_tokens = self._concept_to_tokens.get(tgt_id, set())
                for tok_id in bound_tokens:
                    if tok_id >= self.vocab_size:
                        continue
                    if concept_scores[tok_id] < -1e8:
                        concept_scores[tok_id] = hop_score * 0.3
                    else:
                        concept_scores[tok_id] += hop_score * 0.3
                if tgt_id < self.vocab_size:
                    tgt_vec_norm = tgt_node.vector / (np.linalg.norm(tgt_node.vector) + 1e-15)
                    tgt_embed_norm = self._project_to_embed(tgt_vec_norm)
                    tgt_embed_norm = tgt_embed_norm / (np.linalg.norm(tgt_embed_norm) + 1e-15)
                    tgt_local = (token_norms @ tgt_embed_norm) * hop_score
                    concept_scores = np.maximum(concept_scores, tgt_local)

        concept_scores = np.maximum(concept_scores, -1e8)
        # Proper softmax normalization: temperature-controlled probability distribution
        # Replaces the raw *15.0 scaling hack with mathematically sound softmax
        temperature = max(0.5, 1.0 + 2.0 * self.arousal)
        concept_scores_t = concept_scores / temperature
        concept_scores_t = concept_scores_t - np.max(concept_scores_t)  # numerical stability
        exp_scores = np.exp(concept_scores_t)
        concept_probs = exp_scores / (np.sum(exp_scores) + 1e-10)
        concept_logits = np.log(concept_probs + 1e-10)

        # Context path: hidden state predicts token logits
        ctx_logits = self.context_logits.forward_raw(h[np.newaxis, :]).flatten()

        # ── Cognitive modulation: emotion + identity shape logit blend ──
        # High arousal → exploration (boost concept path), positive valence → trust concepts
        emotion_scale = 1.0 + 0.3 * self.arousal - 0.1 * max(0.0, -self.valence)
        identity_scale = 0.5 + 0.5 * self.identity_strength
        logits = concept_logits * identity_scale * emotion_scale + ctx_logits * self.context_scale
        self._last_ctx_logits = ctx_logits
        return StateTensor(logits[np.newaxis, :])[0]

    def learn(self, token_ids: np.ndarray, next_token_ids: np.ndarray):
        self._step_counter += 1
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]

        logits_tensor = self.forward(token_ids)
        
        next_id = int(next_token_ids[0]) if next_token_ids.ndim == 1 else int(next_token_ids[0, 0])
        last_input_id = int(token_ids[0, -1])

        next_embed = self.token_embed.embed_raw(next_id)

        input_concept = self._nearest_concept(
            self.token_embed.embed_raw(last_input_id))
        output_concept = self._nearest_concept(next_embed)

        if input_concept >= 0 and output_concept >= 0 and input_concept != output_concept:
            edge = self.graph.get_or_create_edge(input_concept, output_concept, weight=0.3)
            edge.weight = min(1.0, edge.weight + 0.05)
            edge.confidence = min(1.0, edge.confidence + 0.03)
            edge.prediction_count += 1
            self._edges_learned += 1
            self._competitive_inhibition(input_concept, output_concept, 0.05)

            # ── Contrastive relation learning (Gap 1 fix) ──
            # Push relation vectors apart for edges with different targets from same source
            if self._step_counter % self._vector_update_interval == 0:
                src_edges = self.graph._outgoing.get(input_concept, [])
                for tgt_id, other_edge in src_edges:
                    if tgt_id == output_concept or other_edge.edge_type == "inhibitory":
                        continue
                    # Different target = potentially different relation type
                    # Push relation vectors apart
                    rv_diff = edge.relation_vector - other_edge.relation_vector
                    edge.relation_vector += 0.01 * rv_diff
                    other_edge.relation_vector -= 0.01 * rv_diff
                    # Renormalize
                    rv1_norm = np.linalg.norm(edge.relation_vector)
                    if rv1_norm > 0:
                        edge.relation_vector /= rv1_norm
                    rv2_norm = np.linalg.norm(other_edge.relation_vector)
                    if rv2_norm > 0:
                        other_edge.relation_vector /= rv2_norm

            # Update token-concept bindings
            # Bind each token to its nearest concept
            self.binding_map.bind(last_input_id, input_concept, confidence=0.5, source="learned")
            self.binding_map.bind(next_id, output_concept, confidence=0.5, source="learned")
            # Also bind input token to output concept — this creates ambiguity
            # when the same input maps to different outputs (fire->hot AND fire->cold)
            self.binding_map.bind(last_input_id, output_concept, confidence=0.3, source="inferred")
            # Update inverted index for fast context priming
            self._concept_to_tokens[input_concept].add(last_input_id)
            self._concept_to_tokens[output_concept].add(next_id)
            self._concept_to_tokens[output_concept].add(last_input_id)

            # ── Vector Updates (Gap 3 fix: concept vectors were frozen) ──
            # Pull: drift concept vectors toward their bound token embeddings
            # Rate-limited to prevent oscillation from noisy single-sample updates
            if self._step_counter % self._vector_update_interval == 0:
                input_embed = self.token_embed.embed_raw(last_input_id)
                # Pull input concept toward input token (project embed→concept first)
                input_concept_vec = self._project_to_concept(input_embed)
                delta_in = input_concept_vec - self.graph.nodes[input_concept].vector
                self.graph.adjust_vector(input_concept, delta_in, lr=0.02)
                # Pull output concept toward output token
                output_concept_vec = self._project_to_concept(next_embed)
                delta_out = output_concept_vec - self.graph.nodes[output_concept].vector
                self.graph.adjust_vector(output_concept, delta_out, lr=0.02)

                # Contrastive push: repel non-matching concepts that are too close
                # Prevents concept collapse (multiple concepts converging to same meaning)
                for nid, sim in self._last_node_sims:
                    if nid != input_concept and sim > 0.7:
                        push_delta = self.graph.nodes[nid].vector - input_concept_vec
                        self.graph.adjust_vector(nid, push_delta, lr=0.005)

                # ── Semantic Drift Defense (Gap 9 fix) ──
                # If a concept has drifted too far from its core_vector (identity anchor),
                # apply corrective pull back toward core. This prevents concepts from
                # losing their original meaning through accumulated updates.
                drift_threshold = 0.4  # trigger defense when drift exceeds this
                drift_correction_strength = 0.1  # pull strength toward core
                for nid in [input_concept, output_concept]:
                    node = self.graph.get_node(nid)
                    if node is None:
                        continue
                    drift = node.drift_magnitude
                    if drift > drift_threshold:
                        # Pull toward core_vector (identity anchor) proportional to excess drift
                        excess = drift - drift_threshold
                        correction = (node.core_vector - node.vector) * drift_correction_strength * min(1.0, excess * 2.0)
                        self.graph.adjust_vector(nid, correction, lr=1.0)  # lr=1.0 because correction is already scaled

            # ── Periodic full-graph drift scan (every _vector_update_interval steps) ──
            # Expands drift defense beyond just input/output concepts
            if self._step_counter % (self._vector_update_interval * 5) == 0:
                for nid, node in self.graph.nodes.items():
                    # Core_vector stability anchor: if core drifted from genesis, snap back
                    core_genesis_drift = np.linalg.norm(node.core_vector - node.genesis_vector)
                    if core_genesis_drift > 0.5:
                        anchor_pull = (node.genesis_vector - node.core_vector) * 0.05
                        node.core_vector += anchor_pull
                        core_norm = np.linalg.norm(node.core_vector)
                        if core_norm > 0:
                            node.core_vector /= core_norm
                    # Standard drift defense for all concepts
                    drift = node.drift_magnitude
                    if drift > drift_threshold:
                        excess = drift - drift_threshold
                        correction = (node.core_vector - node.vector) * drift_correction_strength * min(1.0, excess * 2.0)
                        self.graph.adjust_vector(nid, correction, lr=1.0)

            # InfoNCE-style contrastive concept learning
            # Anchor=input_concept, positive=output_concept, negatives=other active concepts
            if input_concept >= 0 and output_concept >= 0 and self._last_node_sims:
                anchor_vec = self.graph.nodes[input_concept].vector
                positive_vec = self.graph.nodes[output_concept].vector
                # Pull toward positive
                pull_delta = positive_vec - anchor_vec
                self.graph.adjust_vector(input_concept, pull_delta, lr=0.01)
                # Push from top-3 negatives (excluding anchor and positive)
                neg_count = 0
                for nid, sim in self._last_node_sims:
                    if neg_count >= 3:
                        break
                    if nid == input_concept or nid == output_concept:
                        continue
                    negative_vec = self.graph.nodes[nid].vector
                    push_delta = anchor_vec - negative_vec
                    self.graph.adjust_vector(input_concept, push_delta, lr=0.003)
                    neg_count += 1

        # Shortcut edges
        T = token_ids.shape[1]
        if T > 1:
            for t in range(T - 1):
                ctx_id = int(token_ids[0, t])
                ctx_concept = self._nearest_concept(
                    self.token_embed.embed_raw(ctx_id))
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
        self.free_energy_engine.accumulate_semantic(conceptual_error * 1.5, salience=0.5)

        edge_pred_set = set(self._last_edge_pred)
        single_correct = output_concept in edge_pred_set

        if not single_correct and input_concept >= 0:
            inode = self.graph.get_node(input_concept)
            if inode:
                inode.contradiction_count += 1
                inode.prediction_free_energy += 1.5  # increased from 0.5 to make splitting reachable
                # Trigger hotspot tracking when free energy exceeds threshold
                if inode.prediction_free_energy > 2.0:  # lowered from 5.0
                    self.graph.contradiction_hotspots.add(input_concept)

            # Also track contradictions on predicted concepts that missed
            # Use actual concept vector (not raw token embedding) to avoid cross-space comparison
            if len(edge_pred_set) > 0 and output_concept >= 0:
                actual_vec = self.graph.nodes[output_concept].vector
                self.graph.apply_prediction_error(
                    list(edge_pred_set), actual_vec
                )

        if single_correct and input_concept >= 0:
            inode = self.graph.get_node(input_concept)
            if inode and inode.contradiction_count > 0:
                inode.contradiction_count = max(0, inode.contradiction_count - 1)
                inode.prediction_free_energy = max(0.0, inode.prediction_free_energy - 0.3)
            # Record successful path for compression (input→output becomes shortcut)
            if output_concept >= 0:
                self.graph.record_path(input_concept, output_concept)

        if T > 1:
            target_onehot = np.zeros(self.vocab_size, dtype=np.float32)
            target_onehot[next_id] = 1.0

            # === Predictive Coding: Settle + Local Error ===
            # Collect hidden states at each layer
            h_states = [self._last_hidden_state]
            h_temp = self._last_hidden_state
            for i, layer in enumerate(self.hidden_layers):
                h_temp = layer.forward_raw(h_temp[np.newaxis, :])[0]
                h_temp = self.hidden_norms[i].forward_raw(h_temp)
                h_temp = np.tanh(h_temp)
                h_states.append(h_temp)

            # Settle loop: iteratively reduce local prediction errors
            # No backprop — each layer gets ITS OWN error
            settled_states, local_errors = self._settle_predictive(
                h_states, target_onehot
            )

            # Apply local errors — error-gated Hebbian (Δw ∝ e_i · x_j)
            # Context logits: local error between target and prediction
            # Scale error magnitude for stronger signal (salience stays in [0,1])
            ctx_err = local_errors[-1] * 3.0
            ctx_err_tensor = StateTensor(ctx_err[np.newaxis, :])
            ctx_err_tensor._salience = 1.0
            self.context_logits.accumulate_free_energy(ctx_err_tensor)

            # Direct weight update for ctx_logits — accumulate+sleep_cycle scaling
            # is too slow (0.001 per step). Apply local Hebbian update directly:
            # ΔW = lr * error ⊗ input  (no chain rule, just local signal)
            # Use raw softmax error (not settle-normalized) for stable updates
            ctx_logits_now = self.context_logits.forward_raw(
                self._last_hidden_state[np.newaxis, :]
            ).flatten()
            ctx_exp_now = np.exp(ctx_logits_now - np.max(ctx_logits_now))
            ctx_probs_now = ctx_exp_now / (np.sum(ctx_exp_now) + 1e-10)
            raw_error = target_onehot - ctx_probs_now
            h_2d = self._last_hidden_state.reshape(1, -1)
            e_2d = raw_error.reshape(1, -1)
            direct_lr = self._base_lr * self._get_lr_scale()
            direct_update = (e_2d.T @ h_2d) * direct_lr
            self.context_logits.weight.data += direct_update
            np.clip(self.context_logits.weight.data, -5.0, 5.0,
                    out=self.context_logits.weight.data)

            # Direct Hebbian update for GRU cell (was frozen — bug fix)
            # Project error back to hidden dim through context_logits weights
            recurrent_err = raw_error @ self.context_logits.weight.data  # (n_hidden,)
            # Use stored combined inputs from GRU forward pass for proper outer product
            gru = self.recurrent_cell
            if hasattr(gru, '_last_combined') and gru._last_combined is not None:
                combined_2d = gru._last_combined.reshape(1, -1)     # (1, embed+n_hidden)
                combined_r_2d = gru._last_combined_r.reshape(1, -1) # (1, embed+n_hidden)

                # Project error through each gate's sigmoid derivative
                # d(sigmoid)/dx = sigmoid * (1 - sigmoid)
                z_gate_err = recurrent_err * gru._last_z * (1.0 - gru._last_z)
                r_gate_err = recurrent_err * gru._last_r * (1.0 - gru._last_r)
                # h_candidate uses tanh: d(tanh)/dx = 1 - tanh^2
                h_gate_err = recurrent_err * gru._last_z * (1.0 - np.tanh(
                    gru.W_h.forward_raw(combined_r_2d)[0])**2)

                # Direct weight updates: ΔW = lr * gate_error ⊗ combined_input
                z_update = (z_gate_err.reshape(-1, 1) @ combined_2d) * direct_lr * 0.5
                r_update = (r_gate_err.reshape(-1, 1) @ combined_2d) * direct_lr * 0.5
                h_update = (h_gate_err.reshape(-1, 1) @ combined_r_2d) * direct_lr * 0.5

                gru.W_z.weight.data += z_update
                gru.W_r.weight.data += r_update
                gru.W_h.weight.data += h_update
                np.clip(gru.W_z.weight.data, -5.0, 5.0, out=gru.W_z.weight.data)
                np.clip(gru.W_r.weight.data, -5.0, 5.0, out=gru.W_r.weight.data)
                np.clip(gru.W_h.weight.data, -5.0, 5.0, out=gru.W_h.weight.data)

            # Also accumulate for sleep_cycle consolidation
            for gate in [gru.W_z, gru.W_r, gru.W_h]:
                gate_err_tensor = StateTensor(recurrent_err[np.newaxis, :])
                gate_err_tensor._salience = 0.3
                gate.accumulate_free_energy(gate_err_tensor)

            # Hidden layers: direct Hebbian update (was accumulate-only, too slow)
            # Each layer gets ITS OWN local error from the settle loop
            for i, layer in enumerate(self.hidden_layers):
                layer_err = local_errors[i] * 2.0
                # Direct weight update: ΔW = lr * error ⊗ input
                layer_input = h_states[i].reshape(1, -1)
                layer_err_2d = layer_err.reshape(1, -1)
                layer_update = (layer_err_2d.T @ layer_input) * direct_lr
                layer.weight.data += layer_update
                np.clip(layer.weight.data, -5.0, 5.0, out=layer.weight.data)
                # Also accumulate for sleep_cycle consolidation
                layer_err_tensor = StateTensor(layer_err[np.newaxis, :])
                layer_err_tensor._salience = 1.0
                layer.accumulate_free_energy(layer_err_tensor)

        logit_dist = F.softmax(logits_tensor, dim=-1).data.flatten()
        entropy = -np.sum(logit_dist * np.log(logit_dist + 1e-15))
        entropy /= np.log(self.vocab_size)
        self.free_energy_engine.accumulate_episodic(entropy * 0.3)

        self.total_free_energy = self.free_energy_engine.free_energy
        factor = 0.95 if single_correct else 0.05
        self.conceptual_accuracy = 0.9 * self.conceptual_accuracy + 0.1 * factor
        self.n_predictions += 1

        # ── Cognitive Updates ──
        is_correct = single_correct

        # 1. Dissonance EMA (for Governor)
        self.dissonance_ema = 0.9 * self.dissonance_ema + 0.1 * conceptual_error

        # 2. Identity update
        identity_delta = self._compute_identity_update(conceptual_error, is_correct)
        self.identity_strength = np.clip(self.identity_strength + identity_delta, 0.0, 1.0)
        self.identity_momentum = 0.6 * self.identity_momentum + 0.4 * identity_delta
        self.identity_history.append(self.identity_strength)
        if len(self.identity_history) > 100:
            self.identity_history.pop(0)

        # 3. Emotion update (VAD differential equations)
        valence_stimulus = -conceptual_error if is_correct else conceptual_error
        self._update_emotion(valence_stimulus, arousal_stimulus=conceptual_error)

        # 4. Meaning computation
        meaning_gain = self._compute_meaning(conceptual_error)
        self.accumulated_meaning += meaning_gain
        self.meaning_history.append(meaning_gain)
        if len(self.meaning_history) > 100:
            self.meaning_history.pop(0)

        # 5. Sleep pressure accumulation
        self.sleep_pressure = min(1.0, self.sleep_pressure
                                  + conceptual_error * 0.05
                                  + (0.02 if not is_correct else 0.0))

        # 6. Episodic memory storage
        self._store_episode(conceptual_error, is_correct)

        # 7. Emotion-tag active concepts
        for cid in self._last_predicted_concepts:
            self._concept_vad[cid] = (self.valence, self.arousal, self.dominance)

        # 8. Lightweight self-regulation
        # Only run full regulation every 100 steps (graph_diagnostics is expensive)
        if self._step_counter % 100 == 0:
            self._regulate_cognitive_state()

        # Auto-sleep when pressure exceeds threshold (in addition to step-based sleep)
        if self.sleep_pressure >= self.sleep_pressure_threshold:
            self.sleep_cycle()

        if self._step_counter % self.sleep_interval == 0:
            self.sleep_cycle()

        # Record geometry snapshot periodically for long-horizon tracking
        # Reduced from every 10 steps to every 1000 (expensive diagnostic)
        if self._step_counter % 1000 == 0:
            self.graph.record_geometry_snapshot(event="learn", lightweight=True)

        return conceptual_error

    def _update_token_concept_map(self):
        # Use vectorized find_similar for batch concept lookup
        if not self.graph.nodes:
            return
        # Pre-compute all token embeddings, projected to concept_dim
        token_concepts = np.zeros((self.vocab_size, self.concept_dim), dtype=np.float32)
        for tid in range(self.vocab_size):
            embed = self.token_embed(StateTensor(np.array([tid]))).data[0]
            token_concepts[tid] = self._project_to_concept(embed)
        # Batch nearest concept lookup via graph's vectorized find_similar
        for tid in range(self.vocab_size):
            results = self.graph.find_similar(token_concepts[tid], k=1)
            self._token_concept_map[tid] = results[0][0] if results else -1
        self._vectors_dirty = True

    def _rebuild_concept_to_tokens(self):
        """Rebuild inverted index: concept_id -> set of bound token_ids."""
        self._concept_to_tokens = defaultdict(set)
        # From binding map
        for binding in self.binding_map._index:
            self._concept_to_tokens[binding.concept_id].add(binding.token_id)
        # From token_concept_map
        for tid, cid in enumerate(self._token_concept_map):
            if cid >= 0:
                self._concept_to_tokens[cid].add(tid)

    def _competitive_inhibition(self, source: int, target: int, amount: float):
        for t, e in self.graph.get_outgoing(source):
            if t != target and not e.shortcut:
                e.weight = max(0.0, e.weight - amount * 0.3 * e.weight)
                e.confidence = max(0.0, e.confidence - amount * 0.15)

    # ──────────────────────────────────────────────────────────────
    # Predictive Coding: Settle Loop
    # ──────────────────────────────────────────────────────────────

    def _settle_predictive(self, h_states, target):
        """
        Predictive coding settle loop with three stabilizers.

        Each layer predicts the layer above. Error = actual - predicted.
        States adjust to minimize local prediction errors.
        No gradients, no chain rule — just local message passing.

        Stabilizers:
          A. Prediction residual normalization — prevents giant attractors
          B. Noise injection — preserves diversity, enables REM-style creativity
          C. Energy floor / anti-collapse — prevents static minima

        Args:
            h_states: list of hidden states [h0, h1, ..., hN]
            target: target token distribution (one-hot), or None for attractor mode

        Returns:
            (settled_states, local_errors)
        """
        states = [s.copy() for s in h_states]
        n_states = len(states)
        n_hidden = len(self.hidden_layers)
        # states[0] = recurrent output, states[1..n_hidden] = hidden layer outputs
        # states[n_hidden] = top hidden layer (predicts context_logits)
        eps = 1e-6

        # Cache raw references for hot path (avoids repeated attribute lookups)
        hidden_layers_raw = [layer.forward_raw for layer in self.hidden_layers]
        hidden_norms_raw = [norm.forward_raw for norm in self.hidden_norms]
        ctx_logits_raw = self.context_logits.forward_raw
        ctx_weight_data = self.context_logits.weight.data

        for step in range(self.settle_steps):
            errors = []

            for i in range(n_states):
                if i < n_hidden:
                    # Hidden layer i predicts layer i+1's current state
                    pred = hidden_layers_raw[i](states[i][np.newaxis, :])[0]
                    pred = hidden_norms_raw[i](pred)
                    pred = np.tanh(pred)

                    # A. Prediction residual normalization
                    # Prevents giant attractors from dominating error landscape
                    pred_norm = eps + np.linalg.norm(pred)
                    e = (states[i + 1] - pred) / pred_norm
                else:
                    # Top hidden layer predicts context logits
                    ctx = ctx_logits_raw(states[i][np.newaxis, :]).flatten()
                    # Inline softmax (avoid StateTensor wrapping)
                    ctx_exp = np.exp(ctx - np.max(ctx))
                    ctx_dist = ctx_exp / (np.sum(ctx_exp) + 1e-10)

                    if target is not None:
                        # Learning mode: error against target
                        pred_norm = eps + np.linalg.norm(ctx_dist)
                        e = (target - ctx_dist) / pred_norm
                    else:
                        # Attractor mode: self-consistency error
                        pred_norm = eps + np.linalg.norm(ctx_dist)
                        e = -ctx_dist / pred_norm
                errors.append(e)

            # State updates with all three stabilizers
            for i in range(n_states):
                if i < n_hidden:
                    top_down = errors[i]
                else:
                    # Top layer: error projected back through context weights
                    top_down = errors[i] @ ctx_weight_data

                # Bottom-up: evidence from layer below
                if i > 0:
                    bottom_up = errors[i - 1]
                else:
                    bottom_up = np.zeros_like(states[i])

                # C. Energy floor / anti-collapse
                # Prevents total convergence into static minima
                # Tracks running average and pushes away from repetitive states
                if self._running_avg_states is not None and i < len(self._running_avg_states):
                    novelty = 0.1 * (states[i] - self._running_avg_states[i])
                else:
                    novelty = np.zeros_like(states[i])

                # Combined update
                update = self.settle_lr * (top_down + bottom_up) + novelty
                states[i] = states[i] - update
                states[i] *= self.settle_damping

                # B. Noise injection — preserves state diversity
                # Brains are noisy for a reason: creativity, transfer, exploration
                if self.noise_sigma > 0:
                    states[i] += np.random.normal(0, self.noise_sigma, states[i].shape)

                states[i] = np.tanh(states[i])  # keep bounded

        # Update running average for anti-collapse
        if self._running_avg_states is None:
            self._running_avg_states = [s.copy() for s in states]
        else:
            alpha = 0.1  # EMA decay
            for i in range(n_states):
                if i < len(self._running_avg_states):
                    self._running_avg_states[i] = (
                        alpha * states[i] + (1 - alpha) * self._running_avg_states[i]
                    )

        # Final error computation on settled states
        final_errors = []
        for i in range(n_states):
            if i < n_hidden:
                pred = hidden_layers_raw[i](states[i][np.newaxis, :])[0]
                pred = hidden_norms_raw[i](pred)
                pred = np.tanh(pred)
                pred_norm = eps + np.linalg.norm(pred)
                e = (states[i + 1] - pred) / pred_norm
            else:
                ctx = ctx_logits_raw(states[i][np.newaxis, :]).flatten()
                ctx_exp = np.exp(ctx - np.max(ctx))
                ctx_dist = ctx_exp / (np.sum(ctx_exp) + 1e-10)
                if target is not None:
                    pred_norm = eps + np.linalg.norm(ctx_dist)
                    e = (target - ctx_dist) / pred_norm
                else:
                    pred_norm = eps + np.linalg.norm(ctx_dist)
                    e = -ctx_dist / pred_norm
            final_errors.append(e)

        return states, final_errors

    # ──────────────────────────────────────────────────────────────
    # Cognitive Processing (native to RLM)
    # ──────────────────────────────────────────────────────────────

    def _update_emotion(self, valence_stimulus: float, arousal_stimulus: float,
                        dominance_stimulus: float = 0.0):
        """VAD differential equations (from VADEmotionEngine).

        dV/dt = eta_v * (stimulus - V) - lambda_v * V
        dA/dt = eta_a * (stimulus + uncertainty - A) - lambda_a * (A - baseline)
        dD/dt = eta_d * (stimulus - D) - lambda_d * D
        """
        dv = 0.3 * (valence_stimulus - self.valence) - 0.1 * self.valence
        da = 0.4 * (arousal_stimulus + 0.3 * self.dissonance_ema - self.arousal) - 0.1 * (self.arousal - 0.3)
        dd = 0.25 * (dominance_stimulus - self.dominance) - 0.1 * self.dominance
        self.valence = np.clip(self.valence + dv, -1.0, 1.0)
        self.arousal = np.clip(self.arousal + da, 0.0, 1.0)
        self.dominance = np.clip(self.dominance + dd, 0.0, 1.0)

    def _compute_identity_update(self, error: float, is_correct: bool) -> float:
        """Compute identity delta from prediction outcome (from IdentityEngine)."""
        if is_correct:
            base = 0.02 * (1.0 - error)
            if self.identity_strength < 0.5:
                base *= 1.2  # recovery bias
            if len(self.identity_history) >= 3 and all(h > 0.5 for h in self.identity_history[-3:]):
                base *= 1.3  # streak bonus
        else:
            base = -0.05  # fixed failure penalty
            if self.dissonance_ema > 0.8:
                base *= 1.3  # dissonance coupling
        # Momentum
        base += 0.3 * self.identity_momentum
        # Damping when identity is high
        if self.identity_strength > 0.85:
            base *= 0.5
        return base

    def _compute_meaning(self, error: float) -> float:
        """Meaning from dissonance reduction + identity gain + predictive power (from MeaningEngine)."""
        dissonance_reduction = max(0, self.dissonance_ema - error)
        prev_identity = self.identity_history[-2] if len(self.identity_history) >= 2 else 0.5
        identity_gain = self.identity_strength - prev_identity
        predictive_power = max(0, 1.0 - error)
        return 0.4 * dissonance_reduction + 0.3 * max(0, identity_gain) + 0.3 * predictive_power

    def _store_episode(self, error: float, is_correct: bool):
        """Store experience in episodic buffer for memory consolidation."""
        episode = {
            'vector': self._last_hidden_state.copy() if self._last_hidden_state is not None else None,
            'concepts': list(self._last_predicted_concepts),
            'error': error,
            'correct': is_correct,
            'valence': self.valence,
            'arousal': self.arousal,
            'timestamp': time.time(),
        }
        self._episodic_buffer.append(episode)
        if len(self._episodic_buffer) > self._episodic_buffer_max:
            self._episodic_buffer.pop(0)

    def _regulate_cognitive_state(self):
        """Lightweight Governor — prevents runaway state, detects mode."""
        # Hard constraints
        self.identity_strength = np.clip(self.identity_strength, 0.1, 0.95)
        self.sleep_pressure = np.clip(self.sleep_pressure, 0.0, 1.0)

        # Mode detection
        if self.dissonance_ema > 0.8:
            self.regulation_mode = "RECOVERY"
        elif self.dissonance_ema > 0.5:
            self.regulation_mode = "RESOLUTION"
        elif self.dissonance_ema < 0.15:
            self.regulation_mode = "EXPLORATION"
        else:
            self.regulation_mode = "NORMAL"

        # Boundary pressure (sigmoid near limits)
        if self.identity_strength > 0.85:
            overshoot = (self.identity_strength - 0.85) / 0.15
            self.identity_strength -= 0.01 * overshoot
        if self.identity_strength < 0.2:
            recovery = (0.2 - self.identity_strength) / 0.2
            self.identity_strength += 0.02 * recovery

        # Dissonance dampening
        if self.dissonance_ema > 0.9:
            self.dissonance_ema *= 0.95

    # ──────────────────────────────────────────────────────────────
    # Native Memory System (episodic → semantic → graph weights)
    # ──────────────────────────────────────────────────────────────

    def _replay_memories_through_graph(self):
        """Hippocampal replay — re-activate memories through ConceptGraph."""
        if not self._episodic_buffer:
            return
        episodes = self._episodic_buffer[-20:]
        for ep in episodes:
            if ep['concepts']:
                for cid in ep['concepts'][:3]:
                    node = self.graph.get_node(cid)
                    if node:
                        node.activation = min(1.0, node.activation + 0.3)
                self.graph.spread_activation(steps=1, k_active=5, decay=0.3)
                active = [n for n in self.graph.nodes.values() if n.activation > 0.1]
                for i, n1 in enumerate(active):
                    for n2 in active[i+1:]:
                        coact = n1.activation * n2.activation
                        if coact > 0.05:
                            self.graph.hebbian_update(n1.id, n2.id, coactivation=coact, lr=0.01)
                self.graph.reset_activation()

    def _consolidate_episodic_to_semantic(self):
        """Promote frequently-accessed episodic memories to semantic."""
        for ep in self._episodic_buffer:
            if ep['correct'] and ep['error'] < 0.3:
                for cid in ep['concepts']:
                    if cid in self._semantic_memories:
                        self._semantic_memories[cid]['strength'] = min(1.0,
                            self._semantic_memories[cid]['strength'] + 0.05)
                        self._semantic_memories[cid]['access_count'] += 1
                    else:
                        if len(self._semantic_memories) < self._semantic_memory_max:
                            self._semantic_memories[cid] = {
                                'strength': 0.3,
                                'access_count': 1,
                                'last_access': time.time(),
                            }

    def _decay_semantic_memories(self):
        """Ebbinghaus decay — forget unused memories."""
        now = time.time()
        to_remove = []
        for cid, mem in self._semantic_memories.items():
            dt = now - mem['last_access']
            access_factor = 0.5 + mem['access_count'] * 0.1
            retention = mem['strength'] * np.exp(-0.001 * dt / access_factor)
            mem['strength'] = retention
            if retention < 0.05:
                to_remove.append(cid)
        for cid in to_remove:
            del self._semantic_memories[cid]

    def _bridge_memories_to_graph(self):
        """Memory-as-weights: consolidated memories reshape ConceptGraph edges."""
        for cid, mem in self._semantic_memories.items():
            node = self.graph.get_node(cid)
            if node and mem['strength'] > 0.2:
                for cid2, mem2 in self._semantic_memories.items():
                    if cid2 != cid and mem2['strength'] > 0.2:
                        edge = self.graph.get_edge(cid, cid2)
                        if edge:
                            edge.weight += 0.01 * mem['strength'] * mem2['strength']
                            edge.confidence = min(1.0, edge.confidence + 0.005)
                        elif mem['strength'] > 0.5 and mem2['strength'] > 0.5:
                            edge = self.graph.add_edge(cid, cid2, weight=0.1)
                            edge.confidence = 0.3

    def _consolidate_identity(self):
        """Identity consolidation during sleep."""
        if len(self.identity_history) >= 10:
            self.identity_strength = 0.9 * self.identity_strength + 0.1 * np.mean(self.identity_history[-10:])

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
        """Two-phase sleep: SWS (consolidation) then REM (creative exploration).

        SWS (Slow-Wave Sleep): structural consolidation, vector stabilization,
        path compression, homeostatic downscale. The "boring but essential" phase.

        REM (Rapid Eye Movement): noise injection, creative recombination,
        dream sabotage, exploratory perturbation. The "creative" phase that
        enables cross-domain transfer and novel associations.
        """
        # Phase 1: SWS — consolidation
        self._sleep_sws()

        # Phase 2: REM — creative exploration
        self._sleep_rem()

        self.sleep_cycles_completed += 1

        # Cognitive consolidation (shared between phases)
        self._sleep_cognitive_consolidation()

    def _sleep_sws(self):
        """Slow-Wave Sleep: structural consolidation and stabilization."""
        # Memory replay happens during SWS (not after REM) — matches neuroscience
        self._replay_memories_through_graph()

        self._normalize_outgoing_weights()
        self.graph.spread_activation(steps=3)
        self.structural.step()

        # Contradiction Resolution
        self.graph.form_inhibitory_edges()

        # Split concepts that have accumulated enough signal
        # Rate-limited: max 2 splits per sleep cycle to prevent runaway growth
        splits_this_cycle = 0
        max_splits_per_cycle = 2
        # Check hotspots first, then scan high-drift/high-contradiction nodes as fallback
        split_candidates = set(self.graph.contradiction_hotspots)
        for nid, node in self.graph.nodes.items():
            if node.drift_magnitude > 0.4 or node.contradiction_count >= 4:
                split_candidates.add(nid)
        for nid in split_candidates:
            if splits_this_cycle >= max_splits_per_cycle:
                break
            if self.graph.should_split(nid):
                self.graph.split_concept(nid, binding_map=self.binding_map)
                splits_this_cycle += 1

        # Global synaptic downscaling
        self.graph.homeostatic_downscale()

        # Neural weight consolidation: flush accumulated free_energy buffers
        # on all Linear/Embedding/GRU weights (was never called — bug fix)
        super().sleep_cycle()

        # Reconcile: reset contradiction counts, reduce free energy
        self.graph.reconcile_contradictions()

        # Binding maintenance
        self.binding_map.decay_all(rate=0.005)
        self.binding_map.prune(min_strength=0.05)

        # Vector Consolidation: fast-changing active vectors → stable core vectors
        self.graph.consolidate_vectors(rate=0.08)

        # Path Compression: successful inference chains → shortcut edges
        compressible = self.graph.get_compressible_paths(min_usage=2)
        if compressible:
            compressed = self.graph.compress_paths(compressible, min_chain_score=0.15)
            if compressed > 0:
                self.graph._vectors_dirty = True

    def _sleep_rem(self):
        """REM Sleep: creative exploration via noise injection and perturbation.

        Unlike SWS (which stabilizes), REM destabilizes slightly to enable:
        - Cross-domain transfer (noisy activation finds novel paths)
        - Creative recombination (unrelated concepts briefly co-activate)
        - Escape from local minima (noise prevents premature convergence)
        """
        saved_noise = self.noise_sigma
        self.noise_sigma = 0.1  # inject noise during REM

        # 1. Noisy spreading activation — activates unexpected concept combinations
        self.graph.spread_activation(steps=2)

        # 2. Dream perturbation: randomly boost weak edges to explore novel associations
        rng = np.random.RandomState(self.sleep_cycles_completed)
        edges = list(self.graph.edges.values())
        n_perturb = max(1, len(edges) // 20)  # perturb ~5% of edges
        for _ in range(n_perturb):
            if not edges:
                break
            edge = edges[rng.randint(len(edges))]
            if edge.confidence < 0.3:  # only perturb uncertain edges
                boost = rng.uniform(0.01, 0.05)
                edge.weight = min(1.0, edge.weight + boost)
                edge.confidence = min(1.0, edge.confidence + 0.01)

        # 3. Concept vector jitter — explore nearby vector space
        for node in list(self.graph.nodes.values())[:20]:  # limit to 20 nodes
            if node.stability < 0.5:  # only jitter unstable concepts
                jitter = np.random.randn(len(node.vector)).astype(np.float32) * 0.01
                self.graph.adjust_vector(node.id, jitter, lr=1.0)

        # 4. Cross-link: if two active concepts share a common neighbor but no
        # direct edge, form a tentative shortcut (enables cross-domain transfer)
        active_nodes = [n for n in self.graph.nodes.values() if n.activation > 0.1]
        for i, n1 in enumerate(active_nodes):
            for n2 in active_nodes[i+1:]:
                if self.graph.get_edge(n1.id, n2.id) is not None:
                    continue
                # Check shared neighbors
                out1 = {t for t, _ in self.graph._outgoing.get(n1.id, [])}
                out2 = {t for t, _ in self.graph._outgoing.get(n2.id, [])}
                shared = out1 & out2
                if len(shared) >= 1:
                    self.graph.add_edge(n1.id, n2.id, weight=0.1,
                                       edge_type="contextual", shortcut=True)

        # Apply noise-modulated free energy (REM explores, doesn't just consolidate)
        self.free_energy_engine.decay(rate=0.05)  # gentle decay during REM

        # Restore noise setting
        self.noise_sigma = saved_noise

    def _sleep_cognitive_consolidation(self):
        """Cognitive consolidation: emotion, identity, memory, regulation."""
        # Run cognitive regulation
        regulation = self.graph.regulate()
        self._last_regulation = regulation

        # Record geometry snapshot
        self.graph.record_geometry_snapshot(event="sleep", lightweight=True)

        # 1. Hippocampal replay
        self._replay_memories_through_graph()

        # 2. Episodic → semantic consolidation
        self._consolidate_episodic_to_semantic()

        # 3. Semantic memory decay
        self._decay_semantic_memories()

        # 4. Memory → weights bridge
        self._bridge_memories_to_graph()

        # 5. Emotion processing
        self.arousal = 0.3 + (self.arousal - 0.3) * 0.5
        self.valence *= 0.8
        self.dominance = 0.5 + (self.dominance - 0.5) * 0.7

        # 6. Identity consolidation
        self._consolidate_identity()

        # 7. Meaning integration
        self.accumulated_meaning *= 0.99

        # 8. Sleep pressure reset — fully release pressure after sleeping
        self.sleep_pressure = 0.0

        # 9. Final self-regulation
        self._regulate_cognitive_state()

    def __repr__(self):
        return (f"RLM(vocab={self.vocab_size}, embed={self.embed_dim}, "
                f"concepts={len(self.graph.nodes)}, edges={len(self.graph.edges)}, "
                f"sleep={self.sleep_cycles_completed}, acc={self.conceptual_accuracy:.3f}, "
                f"learned_edges={self._edges_learned})")

    # ──────────────────────────────────────────────────────────────
    # Stateful Step & Autoregressive Generation
    # ──────────────────────────────────────────────────────────────

    def forward_step(self, token_id: int, h_prev: np.ndarray,
                     persist_activation: bool = False, k_active_acf: int = 5,
                     context_concepts: Optional[List[int]] = None,
                     fatigue_accumulation_rate: float = 0.3, fatigue_decay_rate: float = 0.1,
                     steps: int = 2) -> Tuple[StateTensor, np.ndarray]:
        """
        Calculates next-token logits and updates the recurrent state in O(1) step execution.
        Also updates nonlinear saturating fatigue.
        """
        # Auto-reset position counter when starting a fresh sequence (h_prev is zeros)
        if not np.any(h_prev):
            self._seq_position = 0

        x = self.token_embed(StateTensor(np.array([token_id]))).data[0]
        # Add positional encoding (use _seq_position for consistent forward/forward_step behavior)
        pos = self._seq_position % len(self._positional_encoding)
        x = x + self._positional_encoding[pos]
        self._seq_position += 1

        # Recurrent step (GRU cell with gating)
        h = self.recurrent_cell(x, h_prev)
        for i, layer in enumerate(self.hidden_layers):
            h_res = layer(StateTensor(h[np.newaxis, :])).data[0]
            h_res = self.hidden_norms[i](StateTensor(h_res)).data
            h_res = np.tanh(h_res)
            h = h + h_res  # residual connection

        # Concept prediction from hidden state → concept_dim (matches graph node vectors)
        z = self.concept_predictor(StateTensor(h[np.newaxis, :])).data[0]
        z_norm = z / (np.linalg.norm(z) + 1e-15)

        # Update concept fatigue values globally
        for node in self.graph.nodes.values():
            node.fatigue = max(0.0, node.fatigue * (1.0 - fatigue_decay_rate))

        # Active Cognitive Frontier: persistent activation & selective decay/spreading
        if not persist_activation:
            self.graph.reset_activation()
        else:
            # Decay existing activations slightly before injecting new ones
            for node in self.graph.nodes.values():
                node.activation *= 0.8

        # Activate based on conceptual similarity, scaled by fatigue
        # Use vectorized matrix multiply instead of per-node Python loop
        if self.graph._vectors_dirty or self.graph._vector_matrix_normed is None:
            self.graph._rebuild_vector_matrix()
        if self.graph._vector_matrix_normed is not None and len(self.graph._node_id_order) > 0:
            sims_raw = self.graph._vector_matrix_normed @ z_norm.astype(np.float32)  # (N,)
            # Apply fatigue scaling vectorized
            fatigue_arr = np.array([getattr(self.graph.nodes[nid], 'fatigue', 0.0)
                                    for nid in self.graph._node_id_order], dtype=np.float32)
            sims = sims_raw * (1.0 - fatigue_arr)
            k_top = min(k_active_acf + 5, len(sims))
            top_idx = np.argpartition(sims, -k_top)[-k_top:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
            node_sims = [(self.graph._node_id_order[i], float(sims[i])) for i in top_idx]
        else:
            node_sims = []

        edge_pred_list = []
        for nid, sim in node_sims[:k_active_acf]:
            self.graph.activate(nid, max(0.01, sim))
            edge_pred_list.append(nid)

        # Spreading activation with soft lateral inhibition (using k_active_acf to enforce ACF)
        self.graph.spread_activation(steps=steps, k_active=k_active_acf, decay=0.5)

        # Accumulate fatigue for currently active nodes based on actual activation level
        for node in self.graph.nodes.values():
            if node.activation > 0.01:
                node.fatigue = min(1.0, node.fatigue + (1.0 - node.fatigue) * node.activation * fatigue_accumulation_rate)

        # Scoring vocab based on active concepts (using effective_activation)
        token_vecs = self.token_embed.weight.data
        token_norms = token_vecs / (np.linalg.norm(token_vecs, axis=1, keepdims=True) + 1e-15)

        concept_scores = -np.ones(self.vocab_size, dtype=np.float32) * 1e9

        all_active = [n for n in sorted(self.graph.nodes.values(),
                                         key=lambda n: n.effective_activation, reverse=True)
                      if n.effective_activation > 0.01][:k_active_acf]

        for node in all_active:
            if node.id >= self.vocab_size:
                continue
            vec_norm = node.vector / (np.linalg.norm(node.vector) + 1e-15)
            # Project concept_dim vector to embed_dim for token comparison
            vec_embed_norm = self._project_to_embed(vec_norm)
            vec_embed_norm = vec_embed_norm / (np.linalg.norm(vec_embed_norm) + 1e-15)
            local = (token_norms @ vec_embed_norm) * node.effective_activation
            concept_scores = np.maximum(concept_scores, local)

        # ── Context Priming with Temporal Decay (inverted index) ──
        if context_concepts is not None and len(context_concepts) > 0:
            T_len = len(context_concepts)
            for i, ctx_nid in enumerate(context_concepts):
                candidate_tokens = self._concept_to_tokens.get(ctx_nid, set())
                dist = T_len - 1 - i
                decay = 0.8 ** dist
                for tok_id in candidate_tokens:
                    if tok_id == token_id or tok_id >= self.vocab_size:
                        continue
                    tok_concept = self._token_concept_map[tok_id]
                    if tok_concept < 0:
                        continue
                    ce = self.graph.get_edge(ctx_nid, tok_concept)
                    if ce is not None and ce.weight > 0.01:
                        boost = ce.weight * decay
                        if concept_scores[tok_id] < -1e8:
                            concept_scores[tok_id] = boost * 0.5
                        else:
                            concept_scores[tok_id] *= (1.0 + boost)

        # ── Multi-hop Edge Traversal (Gap 1&2 fix: cross-domain & relational transfer) ──
        # For each active concept, follow outgoing edges to neighbor concepts,
        # then score tokens bound to those neighbors. This enables 2-hop inference:
        # e.g., "vexol" → warm (edge) → pleasant (bound token)
        # Decay factor per hop prevents distant inference from dominating
        hop_decay = 0.6  # each hop reduces signal by 40% (was 0.4, relaxed for better transfer)
        for node in all_active:
            outgoing = self.graph._outgoing.get(node.id, [])
            for tgt_id, edge in outgoing:
                if edge.edge_type == "inhibitory" or edge.weight < 0.05:
                    continue
                tgt_node = self.graph.get_node(tgt_id)
                if tgt_node is None:
                    continue
                # Score: active_node_activation * edge_weight * hop_decay
                hop_score = node.effective_activation * edge.weight * hop_decay
                # Tokens bound to the neighbor concept get a boost
                bound_tokens = self._concept_to_tokens.get(tgt_id, set())
                for tok_id in bound_tokens:
                    if tok_id == token_id or tok_id >= self.vocab_size:
                        continue
                    if concept_scores[tok_id] < -1e8:
                        concept_scores[tok_id] = hop_score * 0.3
                    else:
                        concept_scores[tok_id] += hop_score * 0.3
                # Also consider the neighbor concept's own vector similarity
                if tgt_id < self.vocab_size:
                    tgt_vec_norm = tgt_node.vector / (np.linalg.norm(tgt_node.vector) + 1e-15)
                    tgt_embed_norm = self._project_to_embed(tgt_vec_norm)
                    tgt_embed_norm = tgt_embed_norm / (np.linalg.norm(tgt_embed_norm) + 1e-15)
                    tgt_local = (token_norms @ tgt_embed_norm) * hop_score
                    concept_scores = np.maximum(concept_scores, tgt_local)

        concept_scores = np.maximum(concept_scores, -1e8)
        # Proper softmax normalization: temperature-controlled probability distribution
        # Replaces the raw *15.0 scaling hack with mathematically sound softmax
        temperature = max(0.5, 1.0 + 2.0 * self.arousal)
        concept_scores_t = concept_scores / temperature
        concept_scores_t = concept_scores_t - np.max(concept_scores_t)  # numerical stability
        exp_scores = np.exp(concept_scores_t)
        concept_probs = exp_scores / (np.sum(exp_scores) + 1e-10)
        concept_logits = np.log(concept_probs + 1e-10)

        # Context path: hidden state predicts token logits
        ctx_logits_raw = self.context_logits(StateTensor(h[np.newaxis, :]))
        ctx_logits = ctx_logits_raw.data.flatten()

        # Cognitive modulation: emotion + identity shape logit blend (same as forward)
        emotion_scale = 1.0 + 0.3 * self.arousal - 0.1 * max(0.0, -self.valence)
        identity_scale = 0.5 + 0.5 * self.identity_strength
        logits = concept_logits * identity_scale * emotion_scale + ctx_logits * self.context_scale

        # Enforce ACF: Zero out activations of concepts that are outside the top-K winners
        active_set = {n.id for n in all_active}
        for node in self.graph.nodes.values():
            if node.id not in active_set:
                node.activation = 0.0

        return StateTensor(logits[np.newaxis, :])[0], h

    def generate(self, prompt: str, tokenizer, max_new_tokens: int = 20,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 0.9,
                 stop_tokens: Optional[List[int]] = None, use_acf: bool = True,
                 k_active_acf: int = 5, repetition_penalty: float = 1.5,
                 repetition_window: int = 10, fatigue_accumulation_rate: float = 0.3,
                 fatigue_decay_rate: float = 0.1, entropy_threshold: float = 0.6,
                 trace_json_path: Optional[str] = "cognitive_trace.json",
                 trace_md_path: Optional[str] = "cognitive_trace.md") -> str:
        """
        Autoregressively generate text from a prompt, stabilized with repetition penalties,
        saturating fatigue, and exploratory drive controls. Outputs JSON and MD tracing logs.
        """
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            return ""

        # Prime the hidden state by feeding the prompt sequence
        token_ids = np.array([prompt_ids], dtype=np.int64)
        _ = self.forward(token_ids)
        h = self._last_hidden_state.copy()

        # Track context concepts for temporal context decay priming
        context_concepts = []
        for tid in prompt_ids:
            x = self.token_embed(StateTensor(np.array([tid]))).data[0]
            nid = self._nearest_concept(x)
            if nid >= 0:
                context_concepts.append(nid)

        generated = list(prompt_ids)
        last_token = prompt_ids[-1]

        stop_set = set(stop_tokens) if stop_tokens is not None else set()
        traces = []

        for step_idx in range(max_new_tokens):
            # Calculate repetition score in a sliding window of the last 15 tokens
            recent_tokens = generated[-15:]
            if len(recent_tokens) > 1:
                repetition_score = 1.0 - (len(set(recent_tokens)) / len(recent_tokens))
            else:
                repetition_score = 0.0

            free_energy = getattr(self, "total_free_energy", 0.0)
            free_energy_norm = (free_energy + 0.5) / (free_energy + 1.5)
            
            # Predict logits first
            logits_tensor, h = self.forward_step(
                last_token, h, persist_activation=use_acf, k_active_acf=k_active_acf,
                context_concepts=context_concepts,
                fatigue_accumulation_rate=fatigue_accumulation_rate,
                fatigue_decay_rate=fatigue_decay_rate
            )
            logits = logits_tensor.data.copy()

            # ── Multi-Hop Relational Inference (generate-only enhancement) ──
            all_active_gen = [n for n in sorted(self.graph.nodes.values(),
                                                 key=lambda n: n.effective_activation, reverse=True)
                              if n.effective_activation > 0.01][:3]
            if all_active_gen:
                token_vecs_gen = self.token_embed.weight.data
                token_norms_gen = token_vecs_gen / (np.linalg.norm(token_vecs_gen, axis=1, keepdims=True) + 1e-15)
                multi_hop_boost_gen = np.zeros(self.vocab_size, dtype=np.float32)
                for node in all_active_gen:
                    chains = self.graph.infer_chain(node.id, max_hops=4,
                                                    confidence_threshold=0.05, min_weight=0.01, k=10,
                                                    frontier_budget=15)
                    for target_concept, chain_score, path in chains:
                        if chain_score <= 0.005 or target_concept not in self.graph.nodes:
                            continue
                        concept_vec = self.graph.nodes[target_concept].vector
                        concept_norm = concept_vec / (np.linalg.norm(concept_vec) + 1e-15)
                        concept_embed_norm = self._project_to_embed(concept_norm)
                        concept_embed_norm = concept_embed_norm / (np.linalg.norm(concept_embed_norm) + 1e-15)
                        token_sims_gen = token_norms_gen @ concept_embed_norm
                        top5_gen = np.argpartition(token_sims_gen, -5)[-5:]
                        for tok_id in top5_gen:
                            if token_sims_gen[tok_id] > 0.2:
                                boost = chain_score * node.effective_activation * token_sims_gen[tok_id]
                                multi_hop_boost_gen[tok_id] = max(multi_hop_boost_gen[tok_id], boost)
                        self.graph.record_path(node.id, target_concept)
                if np.any(multi_hop_boost_gen > 0):
                    boost_mask_gen = multi_hop_boost_gen > 0
                    logits[boost_mask_gen] += multi_hop_boost_gen[boost_mask_gen] * 15.0

            # Compute prediction entropy of raw next-step logits (before repetition penalty/temperature)
            raw_probs = np.exp(logits - np.max(logits))
            raw_probs /= np.sum(raw_probs)
            raw_entropy = -np.sum(raw_probs * np.log(raw_probs + 1e-15))
            normalized_entropy = raw_entropy / np.log(self.vocab_size)
            low_entropy_score = max(0.0, 1.0 - normalized_entropy)

            exploration_drive = repetition_score * low_entropy_score * free_energy_norm
            
            curr_temp = temperature
            curr_k_acf = k_active_acf
            curr_steps = 2

            if exploration_drive > 0.15:
                curr_temp = temperature + 2.0 * (exploration_drive - 0.15)
                curr_k_acf = min(15, k_active_acf + int(repetition_score * 5))
                curr_steps = 3
                
                # Re-run forward step with dynamic exploration params
                logits_tensor, h = self.forward_step(
                    last_token, h, persist_activation=use_acf, k_active_acf=curr_k_acf,
                    context_concepts=context_concepts,
                    fatigue_accumulation_rate=fatigue_accumulation_rate,
                    fatigue_decay_rate=fatigue_decay_rate,
                    steps=curr_steps
                )
                logits = logits_tensor.data.copy()

            # Apply token repetition penalty subtraction
            if repetition_penalty > 0 and len(generated) > 0:
                pen_window = generated[-repetition_window:]
                for tok_id in set(pen_window):
                    logits[tok_id] -= repetition_penalty

            # Apply temperature scaling
            if curr_temp > 0:
                logits /= max(curr_temp, 1e-5)

                # Apply top-k / top-p filtering
                if top_k > 0:
                    indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
                    logits[indices_to_remove] = -1e9

                if top_p < 1.0:
                    sorted_indices = np.argsort(logits)[::-1]
                    sorted_logits = logits[sorted_indices]
                    probs = np.exp(sorted_logits - np.max(sorted_logits))
                    probs /= np.sum(probs)
                    cumulative_probs = np.cumsum(probs)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                    sorted_indices_to_remove[0] = False

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = -1e9

                # Softmax and sample
                probs = np.exp(logits - np.max(logits))
                probs /= np.sum(probs)
                next_token = int(np.random.choice(len(probs), p=probs))
            else:
                # Greedy decoding
                next_token = int(np.argmax(logits))

            # Record telemetry
            top_concepts = []
            all_graph_nodes = sorted(self.graph.nodes.values(), key=lambda n: n.effective_activation, reverse=True)
            for node in all_graph_nodes[:5]:
                if node.effective_activation > 0.0:
                    top_concepts.append((node.label, float(node.effective_activation), float(getattr(node, 'fatigue', 0.0))))

            token_str = tokenizer.decode([next_token])
            traces.append({
                "step": step_idx,
                "token": token_str,
                "token_id": int(next_token),
                "entropy": float(normalized_entropy),
                "repetition_score": float(repetition_score),
                "exploration_drive": float(exploration_drive),
                "temperature": float(curr_temp),
                "k_active_acf": int(curr_k_acf),
                "steps": int(curr_steps),
                "top_concepts": top_concepts,
                "free_energy": float(free_energy)
            })

            generated.append(next_token)
            if next_token in stop_set:
                break

            last_token = next_token
            # Append next concept to context history
            x = self.token_embed(StateTensor(np.array([next_token]))).data[0]
            nid = self._nearest_concept(x)
            if nid >= 0:
                context_concepts.append(nid)
                if len(context_concepts) > self.max_seq_len:
                    context_concepts.pop(0)

        # Write traces
        import json
        if trace_json_path:
            try:
                with open(trace_json_path, 'w', encoding='utf-8') as f:
                    json.dump(traces, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving JSON trace: {e}")

        if trace_md_path:
            try:
                with open(trace_md_path, 'w', encoding='utf-8') as f:
                    f.write("# RAVANA Cognitive Trace Log\n\n")
                    f.write(f"**Prompt:** `{prompt}`\n\n")
                    f.write(f"**Generated Text:** `{tokenizer.decode(generated)}`\n\n")
                    f.write("## Step-by-Step Cognitive Metrics\n\n")
                    f.write("| Step | Token | Entropy | Repetition Score | Exploration Drive | Temperature | ACF | Steps | Top Active Concepts (Label, Activation, Fatigue) |\n")
                    f.write("|------|-------|---------|------------------|-------------------|-------------|-----|-------|--------------------------------------------------|\n")
                    for t in traces:
                        concepts_str = ", ".join([f"{c[0]} (act={c[1]:.2f}, fat={c[2]:.2f})" for c in t["top_concepts"]])
                        f.write(f"| {t['step']} | `{t['token']}` | {t['entropy']:.3f} | {t['repetition_score']:.2f} | {t['exploration_drive']:.2f} | {t['temperature']:.2f} | {t['k_active_acf']} | {t['steps']} | {concepts_str} |\n")
            except Exception as e:
                print(f"Error saving Markdown trace: {e}")

        return tokenizer.decode(generated)


    # ──────────────────────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save complete model checkpoint.

        Persists: neural weights with cognitive metadata, concept graph
        (nodes, edges, vectors, topology), RLM scalar state, and
        free energy accumulator state.

        Use RLM.load(path) to restore.
        """
        checkpoint = {
            # Config (for reconstruction verification)
            "config": {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "concept_dim": self.concept_dim,
                "n_concepts": self.n_concepts,
                "n_hidden": self.n_hidden,
                "n_layers": self.n_layers,
                "max_seq_len": self.max_seq_len,
                "free_energy_threshold": self.free_energy_threshold,
                "sleep_interval": self.sleep_interval,
            },
            # Neural weights + cognitive metadata
            "state_dict": self.state_dict(),
            # ConceptGraph (the model's long-term memory)
            "graph": self.graph,
            # RLM scalars
            "scalars": {
                "step_counter": self._step_counter,
                "sleep_cycles_completed": self.sleep_cycles_completed,
                "total_free_energy": self.total_free_energy,
                "conceptual_accuracy": self.conceptual_accuracy,
                "n_predictions": self.n_predictions,
                "edges_learned": self._edges_learned,
                "context_bias": self.context_bias,
                "context_scale": self.context_scale,
                "settle_steps": self.settle_steps,
                "settle_lr": self.settle_lr,
                "settle_damping": self.settle_damping,
                "noise_sigma": self.noise_sigma,
            },
            # Token-concept mapping
            "token_concept_map": self._token_concept_map,
            # Concept binding map (token↔concept↔memory bindings)
            "binding_map": self.binding_map,
            # Concept-to-tokens inverted index
            "concept_to_tokens": dict(self._concept_to_tokens),
            # Settle loop anti-collapse state
            "running_avg_states": self._running_avg_states,
            # Free energy accumulator state
            "free_energy_state": {
                "semantic_free_energy": self.free_energy_engine.semantic_free_energy,
                "episodic_free_energy": self.free_energy_engine.episodic_free_energy,
                "contradiction_free_energy": self.free_energy_engine.contradiction_free_energy,
                "linguistic_free_energy": self.free_energy_engine.linguistic_free_energy,
                "abstraction_free_energy": self.free_energy_engine.abstraction_free_energy,
            },
            # ── Cognitive State ──
            "cognitive_state": {
                "identity_strength": self.identity_strength,
                "identity_momentum": self.identity_momentum,
                "identity_history": self.identity_history,
                "valence": self.valence,
                "arousal": self.arousal,
                "dominance": self.dominance,
                "accumulated_meaning": self.accumulated_meaning,
                "meaning_history": self.meaning_history,
                "sleep_pressure": self.sleep_pressure,
                "sleep_pressure_threshold": self.sleep_pressure_threshold,
                "regulation_mode": self.regulation_mode,
                "dissonance_ema": self.dissonance_ema,
                "episodic_buffer": self._episodic_buffer,
                "semantic_memories": self._semantic_memories,
                "concept_vad": {str(k): v for k, v in self._concept_vad.items()},
            },
            # Sub-engine config
            "engine_config": {
                "hebbian_lr": self.hebbian.lr,
                "anti_hebbian_lr": self.anti_hebbian.lr,
            },
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

    @classmethod
    def load(cls, path: str) -> 'RLM':
        """Load a complete model checkpoint.

        Returns a new RLM instance with all weights, graph, and state restored.
        """
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        # Reconstruct model with saved config
        cfg = checkpoint["config"]
        model = cls(**cfg)

        # Restore neural weights + cognitive metadata
        model.load_state_dict(checkpoint["state_dict"])

        # Restore concept graph (the model's long-term memory)
        model.graph = checkpoint["graph"]
        model.propagation = PropagationEngine(model.graph)
        model.free_energy_engine = FreeEnergyAccumulator(model.graph)
        model.hebbian = HebbianPlasticity(model.graph,
                                          lr=checkpoint.get("engine_config", {}).get("hebbian_lr", 0.03))
        model.anti_hebbian = AntiHebbianPlasticity(model.graph,
                                                   lr=checkpoint.get("engine_config", {}).get("anti_hebbian_lr", 0.02))
        model.structural = StructuralPlasticity(model.graph,
                                                prune_threshold=0.005,
                                                form_threshold=0.3)

        # Restore RLM scalars
        scalars = checkpoint["scalars"]
        model._step_counter = scalars["step_counter"]
        model.sleep_cycles_completed = scalars["sleep_cycles_completed"]
        model.total_free_energy = scalars["total_free_energy"]
        model.conceptual_accuracy = scalars["conceptual_accuracy"]
        model.n_predictions = scalars["n_predictions"]
        model._edges_learned = scalars["edges_learned"]
        model.context_bias = scalars["context_bias"]
        model.context_scale = scalars["context_scale"]
        model.settle_steps = scalars.get("settle_steps", 5)
        model.settle_lr = scalars.get("settle_lr", 0.05)
        model.settle_damping = scalars.get("settle_damping", 0.9)
        model.noise_sigma = scalars.get("noise_sigma", 0.0)

        # Restore token-concept mapping
        model._token_concept_map = checkpoint["token_concept_map"]

        # Restore binding map (was lost in old code — fresh one created by constructor)
        if "binding_map" in checkpoint:
            model.binding_map = checkpoint["binding_map"]

        # Restore concept-to-tokens inverted index
        if "concept_to_tokens" in checkpoint:
            from collections import defaultdict
            model._concept_to_tokens = defaultdict(set, {
                int(k): set(v) for k, v in checkpoint["concept_to_tokens"].items()
            })

        # Restore settle loop anti-collapse state
        if "running_avg_states" in checkpoint:
            model._running_avg_states = checkpoint["running_avg_states"]

        # Restore free energy accumulator state
        ps = checkpoint.get("free_energy_state", checkpoint.get("pressure_state", {}))
        model.free_energy_engine.semantic_free_energy = ps.get("semantic_free_energy", 0.0)
        model.free_energy_engine.episodic_free_energy = ps.get("episodic_free_energy", 0.0)
        model.free_energy_engine.contradiction_free_energy = ps.get("contradiction_free_energy", 0.0)
        model.free_energy_engine.linguistic_free_energy = ps.get("linguistic_free_energy", 0.0)
        model.free_energy_engine.abstraction_free_energy = ps.get("abstraction_free_energy", 0.0)

        # Restore cognitive state
        cs = checkpoint.get("cognitive_state", {})
        if cs:
            model.identity_strength = cs.get("identity_strength", 0.5)
            model.identity_momentum = cs.get("identity_momentum", 0.0)
            model.identity_history = cs.get("identity_history", [])
            model.valence = cs.get("valence", 0.0)
            model.arousal = cs.get("arousal", 0.3)
            model.dominance = cs.get("dominance", 0.5)
            model.accumulated_meaning = cs.get("accumulated_meaning", 0.0)
            model.meaning_history = cs.get("meaning_history", [])
            model.sleep_pressure = cs.get("sleep_pressure", 0.0)
            model.sleep_pressure_threshold = cs.get("sleep_pressure_threshold", 0.7)
            model.regulation_mode = cs.get("regulation_mode", "NORMAL")
            model.dissonance_ema = cs.get("dissonance_ema", 0.5)
            model._episodic_buffer = cs.get("episodic_buffer", [])
            model._semantic_memories = cs.get("semantic_memories", {})
            model._concept_vad = {int(k): tuple(v) for k, v in cs.get("concept_vad", {}).items()}

        return model

    # ──────────────────────────────────────────────────────────────
    # Zip Save / Load (human-readable, safe, partial-load)
    # ──────────────────────────────────────────────────────────────

    def save_zip(self, path: str):
        """Save model as a zip archive with separate files.

        Layout:
            arrays.npz      — all numpy arrays (weight tensors + node vectors)
            graph.json      — graph topology + node/edge metadata (no arrays)
            metadata.json   — config, scalars, free energy state, cognitive metadata

        Advantages over pickle:
            - Human-readable graph and metadata (JSON)
            - Safe (no arbitrary code execution on load)
            - Partial loading possible (can inspect without loading arrays)
            - Version-tolerant (JSON schema can evolve)
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # ── Collect arrays ──
        arrays = {}
        state_dict_meta = {}
        sd = self.state_dict()
        for name, entry in sd.items():
            key = f"weight/{name}"
            arrays[key] = entry["data"]
            state_dict_meta[name] = {
                "salience": entry.get("salience", 0.5),
                "free_energy": entry.get("free_energy", entry.get("pressure", 0.0)),
                "stability": entry.get("stability", 0.5),
            }

        # Node vectors (active + core + genesis for full identity tracking)
        for nid, node in self.graph.nodes.items():
            arrays[f"node/{nid}"] = node.vector
            arrays[f"node_core/{nid}"] = node.core_vector
            arrays[f"node_genesis/{nid}"] = node.genesis_vector

        # Graph-level temporal context vector
        if hasattr(self.graph, 'temporal_context') and self.graph.temporal_context is not None:
            arrays["graph/temporal_context"] = self.graph.temporal_context

        # Settle loop anti-collapse state
        if self._running_avg_states is not None:
            for i, state in enumerate(self._running_avg_states):
                arrays[f"settle/running_avg_{i}"] = state

        # ── Build graph JSON (no arrays) ──
        nodes_json = {}
        for nid, node in self.graph.nodes.items():
            nodes_json[str(nid)] = {
                "id": node.id,
                "label": node.label,
                "activation": float(node.activation),
                "salience": float(node.salience),
                "free_energy": float(node.prediction_free_energy),
                "stability": float(node.stability),
                "confidence": float(node.confidence),
                "timestamp": float(node.timestamp),
                "contradiction_count": int(node.contradiction_count),
                "level": int(node.level),
                "parent": int(node.parent) if node.parent is not None else None,
                "children": sorted(int(c) for c in node.children),
                "abstraction_degree": float(node.abstraction_degree),
                # Temporal fields for identity tracking
                "last_activated": float(node.last_activated) if node.last_activated else None,
                "activation_history": [float(x) for x in (node.activation_history or [])],
                "temporal_context": node.temporal_context.tolist() if node.temporal_context is not None else None,
                "fatigue": float(node.fatigue),
            }

        # Relation vectors for edges (learned relational embeddings)
        edge_relation_vectors = {}
        edges_json = {}
        for (s, t), edge in self.graph.edges.items():
            edges_json[f"({s}, {t})"] = {
                "source": int(edge.source),
                "target": int(edge.target),
                "weight": float(edge.weight),
                "confidence": float(edge.confidence),
                "free_energy": float(edge.prediction_free_energy),
                "stability": float(edge.stability),
                "timestamp": float(edge.timestamp),
                "prediction_count": int(edge.prediction_count),
                "shortcut": bool(edge.shortcut),
                "edge_type": edge.edge_type,
                "relation_type": edge.relation_type,
            }
            if edge.relation_vector is not None:
                edge_relation_vectors[f"({s}, {t})"] = edge.relation_vector

        # Add relation vectors to arrays
        for key, rvec in edge_relation_vectors.items():
            safe_key = key.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
            arrays[f"edge_rel/{safe_key}"] = rvec

        graph_json = {
            "dim": self.graph.dim,
            "max_nodes": self.graph.max_nodes,
            "next_id": self.graph.next_id,
            "total_free_energy": float(self.graph.total_free_energy),
            "contradiction_hotspots": sorted(int(x) for x in self.graph.contradiction_hotspots),
            "nodes": nodes_json,
            "edges": edges_json,
            "temporal_context_drift_rate": float(self.graph.temporal_context_drift_rate),
        }

        # ── Build metadata JSON ──
        metadata_json = {
            "format": "ravana_zip",
            "version": 1,
            "config": {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "concept_dim": self.concept_dim,
                "n_concepts": self.n_concepts,
                "n_hidden": self.n_hidden,
                "n_layers": self.n_layers,
                "max_seq_len": self.max_seq_len,
                "free_energy_threshold": self.free_energy_threshold,
                "sleep_interval": self.sleep_interval,
            },
            "scalars": {
                "step_counter": self._step_counter,
                "sleep_cycles_completed": self.sleep_cycles_completed,
                "total_free_energy": float(self.total_free_energy),
                "conceptual_accuracy": float(self.conceptual_accuracy),
                "n_predictions": self.n_predictions,
                "edges_learned": self._edges_learned,
                "context_bias": float(self.context_bias),
                "context_scale": float(self.context_scale),
                "settle_steps": self.settle_steps,
                "settle_lr": float(self.settle_lr),
                "settle_damping": float(self.settle_damping),
                "noise_sigma": float(self.noise_sigma),
            },
            "free_energy_state": {
                "semantic_free_energy": float(self.free_energy_engine.semantic_free_energy),
                "episodic_free_energy": float(self.free_energy_engine.episodic_free_energy),
                "contradiction_free_energy": float(self.free_energy_engine.contradiction_free_energy),
                "linguistic_free_energy": float(self.free_energy_engine.linguistic_free_energy),
                "abstraction_free_energy": float(self.free_energy_engine.abstraction_free_energy),
            },
            "engine_config": {
                "hebbian_lr": float(self.hebbian.lr),
                "anti_hebbian_lr": float(self.anti_hebbian.lr),
            },
            "token_concept_map": self._token_concept_map,
            # Concept-to-tokens inverted index
            "concept_to_tokens": {str(k): sorted(v) for k, v in self._concept_to_tokens.items()},
            # Binding map (token↔concept↔memory bindings)
            "bindings": [
                {
                    "token_id": b.token_id,
                    "concept_id": b.concept_id,
                    "confidence": float(b.confidence),
                    "source": b.source,
                    "reinforcement_count": b.reinforcement_count,
                    "decay_score": float(b.decay_score),
                    "ambiguity": float(b.ambiguity),
                }
                for b in self.binding_map._index.values()
            ],
            "state_dict_meta": state_dict_meta,
        }

        # ── Cognitive State ──
        # Serialize episodic buffer (convert numpy arrays to lists for JSON)
        episodes_json = []
        for ep in self._episodic_buffer:
            ep_copy = dict(ep)
            if ep_copy.get('vector') is not None:
                ep_copy['vector'] = ep_copy['vector'].tolist()
            episodes_json.append(ep_copy)

        metadata_json["cognitive_state"] = {
            "identity_strength": self.identity_strength,
            "identity_momentum": self.identity_momentum,
            "identity_history": self.identity_history,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "accumulated_meaning": self.accumulated_meaning,
            "meaning_history": self.meaning_history,
            "sleep_pressure": self.sleep_pressure,
            "sleep_pressure_threshold": self.sleep_pressure_threshold,
            "regulation_mode": self.regulation_mode,
            "dissonance_ema": self.dissonance_ema,
            "episodic_buffer": episodes_json,
            "semantic_memories": self._semantic_memories,
            "concept_vad": {str(k): list(v) for k, v in self._concept_vad.items()},
        }

        # ── CognitiveRegulator state ──
        if hasattr(self.graph, '_regulator') and self.graph._regulator is not None:
            reg = self.graph._regulator
            metadata_json["regulator"] = {
                "current_phase": reg.current_phase,
                "phase_confidence": float(reg.phase_confidence),
                "fast_inhibition_boost": float(reg._fast_inhibition_boost),
                "fast_cooldown": int(reg._fast_cooldown),
                "fast_damping": float(reg._fast_damping),
                "medium_sleep_urgency": float(reg._medium_sleep_urgency),
                "medium_cooldown": int(reg._medium_cooldown),
                "medium_damping": float(reg._medium_damping),
                "slow_plasticity_boost": float(reg._slow_plasticity_boost),
                "slow_cooldown": int(reg._slow_cooldown),
                "slow_damping": float(reg._slow_damping),
                "phase_buffer": list(reg._phase_buffer),
                "hysteresis_dead_zone": float(reg._hysteresis_dead_zone),
                "step": int(reg._step),
                "adjustments_made": int(reg._adjustments_made),
                "oscillation_count": int(reg._oscillation_count),
                "last_phases": list(reg._last_phases),
            }

        # ── GeometryHistory snapshots ──
        if hasattr(self.graph, '_geometry_history') and self.graph._geometry_history is not None:
            gh = self.graph._geometry_history
            if hasattr(gh, 'snapshots'):
                metadata_json["geometry_history"] = gh.snapshots

        # ── Successful inference paths ──
        if hasattr(self.graph, '_successful_paths'):
            metadata_json["successful_paths"] = {
                f"{k[0]},{k[1]}": v
                for k, v in self.graph._successful_paths.items()
            }

        # ── Write zip ──
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Arrays
            from io import BytesIO
            buf = BytesIO()
            np.savez(buf, **arrays)
            zf.writestr("arrays.npz", buf.getvalue())

            # Graph
            zf.writestr("graph.json", json.dumps(graph_json, indent=2))

            # Metadata
            zf.writestr("metadata.json", json.dumps(metadata_json, indent=2))

    @classmethod
    def load_zip(cls, path: str) -> 'RLM':
        """Load a model from a zip archive.

        Reads arrays.npz, graph.json, and metadata.json to reconstruct
        a complete RLM instance with all weights, graph, and state.
        """
        with zipfile.ZipFile(path, 'r') as zf:
            # ── Metadata ──
            meta = json.loads(zf.read("metadata.json"))
            cfg = meta["config"]
            model = cls(**cfg)

            # ── Arrays ──
            from io import BytesIO
            buf = BytesIO(zf.read("arrays.npz"))
            npz = np.load(buf)

            # Restore weight arrays
            sd = {}
            for name, m in meta["state_dict_meta"].items():
                key = f"weight/{name}"
                if key in npz:
                    sd[name] = {
                        "data": npz[key],
                        "salience": m["salience"],
                        "free_energy": m.get("free_energy", m.get("pressure", 0.0)),
                        "stability": m["stability"],
                    }
            model.load_state_dict(sd)

            # ── Graph ──
            graph_data = json.loads(zf.read("graph.json"))

            # Clear default graph and rebuild from saved data
            from ..graph import ConceptNode, ConceptEdge
            model.graph = ConceptGraph(dim=graph_data["dim"],
                                       max_nodes=graph_data["max_nodes"])
            model.graph.next_id = graph_data["next_id"]
            model.graph.total_free_energy = graph_data["total_free_energy"]
            model.graph.contradiction_hotspots = set(graph_data["contradiction_hotspots"])

            # Restore nodes (with full identity vectors + temporal fields)
            for nid_str, nd in graph_data["nodes"].items():
                nid = int(nid_str)
                vec_key = f"node/{nid}"
                vector = npz[vec_key] if vec_key in npz else np.zeros(graph_data["dim"], dtype=np.float32)
                node = ConceptNode(
                    node_id=nd["id"],
                    vector=vector,
                    label=nd["label"],
                )
                # Restore identity vectors (core = drift-resistant anchor, genesis = original)
                core_key = f"node_core/{nid}"
                if core_key in npz:
                    node.core_vector = npz[core_key]
                else:
                    node.core_vector = vector.copy()
                genesis_key = f"node_genesis/{nid}"
                if genesis_key in npz:
                    node.genesis_vector = npz[genesis_key]
                else:
                    node.genesis_vector = vector.copy()

                node.activation = nd["activation"]
                node.salience = nd["salience"]
                node.prediction_free_energy = nd.get("free_energy", nd.get("pressure", 0.0))
                node.stability = nd["stability"]
                node.confidence = nd["confidence"]
                node.timestamp = nd["timestamp"]
                node.contradiction_count = nd["contradiction_count"]
                node.level = nd["level"]
                node.parent = nd["parent"]
                node.children = set(nd["children"])
                node.abstraction_degree = nd["abstraction_degree"]
                # Temporal fields
                node.fatigue = nd.get("fatigue", 0.0)
                node.last_activated = nd.get("last_activated")
                node.activation_history = nd.get("activation_history", [])
                tc = nd.get("temporal_context")
                node.temporal_context = np.array(tc, dtype=np.float32) if tc is not None else None
                model.graph.nodes[nid] = node

            # Restore edges (with relation vectors)
            for key, ed in graph_data["edges"].items():
                edge = ConceptEdge(
                    source=ed["source"],
                    target=ed["target"],
                    weight=ed["weight"],
                    edge_type=ed.get("edge_type", "excitatory"),
                    relation_type=ed.get("relation_type", "semantic"),
                )
                edge.confidence = ed["confidence"]
                edge.prediction_free_energy = ed.get("free_energy", ed.get("pressure", 0.0))
                edge.stability = ed["stability"]
                edge.timestamp = ed["timestamp"]
                edge.prediction_count = ed["prediction_count"]
                edge.shortcut = ed["shortcut"]
                # Restore relation vector from arrays
                safe_key = key.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
                rel_key = f"edge_rel/{safe_key}"
                if rel_key in npz:
                    edge.relation_vector = npz[rel_key]
                model.graph.edges[(ed["source"], ed["target"])] = edge

            # Rebuild engines with restored graph
            model.propagation = PropagationEngine(model.graph)
            model.free_energy_engine = FreeEnergyAccumulator(model.graph)
            model.hebbian = HebbianPlasticity(model.graph,
                                              lr=meta["engine_config"]["hebbian_lr"])
            model.anti_hebbian = AntiHebbianPlasticity(model.graph,
                                                       lr=meta["engine_config"]["anti_hebbian_lr"])
            model.structural = StructuralPlasticity(model.graph,
                                                    prune_threshold=0.005,
                                                    form_threshold=0.3)

            # Restore scalars
            s = meta["scalars"]
            model._step_counter = s["step_counter"]
            model.sleep_cycles_completed = s["sleep_cycles_completed"]
            model.total_free_energy = s["total_free_energy"]
            model.conceptual_accuracy = s["conceptual_accuracy"]
            model.n_predictions = s["n_predictions"]
            model._edges_learned = s["edges_learned"]
            model.context_bias = s["context_bias"]
            model.context_scale = s["context_scale"]
            model.settle_steps = s.get("settle_steps", 5)
            model.settle_lr = s.get("settle_lr", 0.05)
            model.settle_damping = s.get("settle_damping", 0.9)
            model.noise_sigma = s.get("noise_sigma", 0.0)

            # Restore token-concept map
            model._token_concept_map = meta["token_concept_map"]

            # Restore bindings
            for b_data in meta.get("bindings", []):
                model.binding_map.bind(
                    b_data["token_id"], b_data["concept_id"],
                    confidence=b_data["confidence"], source=b_data["source"]
                )
                b = model.binding_map._index.get((b_data["token_id"], b_data["concept_id"]))
                if b:
                    b.reinforcement_count = b_data.get("reinforcement_count", 0)
                    b.decay_score = b_data.get("decay_score", 0.0)
                    b.ambiguity = b_data.get("ambiguity", 0.0)

            # Restore free energy state
            ps = meta.get("free_energy_state", meta.get("pressure_state", {}))
            model.free_energy_engine.semantic_free_energy = ps["semantic_free_energy"]
            model.free_energy_engine.episodic_free_energy = ps["episodic_free_energy"]
            model.free_energy_engine.contradiction_free_energy = ps["contradiction_free_energy"]
            model.free_energy_engine.linguistic_free_energy = ps["linguistic_free_energy"]
            model.free_energy_engine.abstraction_free_energy = ps["abstraction_free_energy"]

            # Restore graph-level temporal context
            tc_key = "graph/temporal_context"
            if tc_key in npz:
                model.graph.temporal_context = npz[tc_key]
            model.graph.temporal_context_drift_rate = graph_data.get("temporal_context_drift_rate", 0.05)

            # Restore CognitiveRegulator state
            if "regulator" in meta and hasattr(model.graph, '_regulator') and model.graph._regulator is not None:
                reg_data = meta["regulator"]
                reg = model.graph._regulator
                reg.current_phase = reg_data.get("current_phase", "exploratory")
                reg.phase_confidence = reg_data.get("phase_confidence", 0.0)
                reg._fast_inhibition_boost = reg_data.get("fast_inhibition_boost", 0.0)
                reg._fast_cooldown = reg_data.get("fast_cooldown", 0)
                reg._fast_damping = reg_data.get("fast_damping", 0.7)
                reg._medium_sleep_urgency = reg_data.get("medium_sleep_urgency", 0.0)
                reg._medium_cooldown = reg_data.get("medium_cooldown", 0)
                reg._medium_damping = reg_data.get("medium_damping", 0.8)
                reg._slow_plasticity_boost = reg_data.get("slow_plasticity_boost", 0.0)
                reg._slow_cooldown = reg_data.get("slow_cooldown", 0)
                reg._slow_damping = reg_data.get("slow_damping", 0.9)
                reg._phase_buffer = list(reg_data.get("phase_buffer", []))
                reg._hysteresis_dead_zone = reg_data.get("hysteresis_dead_zone", 0.1)
                reg._step = reg_data.get("step", 0)
                reg._adjustments_made = reg_data.get("adjustments_made", 0)
                reg._oscillation_count = reg_data.get("oscillation_count", 0)
                reg._last_phases = list(reg_data.get("last_phases", []))

            # Restore GeometryHistory snapshots
            if "geometry_history" in meta and hasattr(model.graph, '_geometry_history'):
                model.graph._geometry_history.snapshots = meta["geometry_history"]

            # Restore successful inference paths (must be defaultdict for record_path += 1)
            if "successful_paths" in meta:
                from collections import defaultdict
                paths = defaultdict(int)
                for k, v in meta["successful_paths"].items():
                    paths[tuple(int(x) for x in k.split(","))] = v
                model.graph._successful_paths = paths

            # Restore concept-to-tokens inverted index
            if "concept_to_tokens" in meta:
                from collections import defaultdict
                model._concept_to_tokens = defaultdict(set, {
                    int(k): set(v) for k, v in meta["concept_to_tokens"].items()
                })

            # Restore settle loop anti-collapse state
            settle_keys = [k for k in npz.keys() if k.startswith("settle/running_avg_")]
            if settle_keys:
                model._running_avg_states = [npz[k] for k in sorted(settle_keys)]

            # Restore cognitive state
            cs = meta.get("cognitive_state", {})
            if cs:
                model.identity_strength = cs.get("identity_strength", 0.5)
                model.identity_momentum = cs.get("identity_momentum", 0.0)
                model.identity_history = cs.get("identity_history", [])
                model.valence = cs.get("valence", 0.0)
                model.arousal = cs.get("arousal", 0.3)
                model.dominance = cs.get("dominance", 0.5)
                model.accumulated_meaning = cs.get("accumulated_meaning", 0.0)
                model.meaning_history = cs.get("meaning_history", [])
                model.sleep_pressure = cs.get("sleep_pressure", 0.0)
                model.sleep_pressure_threshold = cs.get("sleep_pressure_threshold", 0.7)
                model.regulation_mode = cs.get("regulation_mode", "NORMAL")
                model.dissonance_ema = cs.get("dissonance_ema", 0.5)
                # Restore episodic buffer (convert vector lists back to numpy)
                model._episodic_buffer = []
                for ep in cs.get("episodic_buffer", []):
                    ep_copy = dict(ep)
                    if ep_copy.get('vector') is not None:
                        ep_copy['vector'] = np.array(ep_copy['vector'], dtype=np.float32)
                    model._episodic_buffer.append(ep_copy)
                model._semantic_memories = cs.get("semantic_memories", {})
                model._concept_vad = {int(k): tuple(v) for k, v in cs.get("concept_vad", {}).items()}

            return model
