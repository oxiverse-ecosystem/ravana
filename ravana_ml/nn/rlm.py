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
from .module import Module, Linear, Embedding


class RLM(Module):
    """
    Recursive Learning Model (RLM)

    An alternative to the traditional LLM. Instead of transformer attention
    and backprop, RLM uses concept graphs, Hebbian plasticity, competitive
    inhibition, and free-energy-driven sleep cycles. Maps input sequences to
    conceptual trajectories in a ConceptGraph.
    """
    def __init__(self, vocab_size: int, embed_dim: int, concept_dim: int,
                 n_concepts: int, n_hidden: int, n_layers: int = 1,
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
        self.sleep_cycles_completed = 0
        self.total_free_energy = 0.0
        self.conceptual_accuracy = 0.0
        self.n_predictions = 0
        self._step_counter = 0

        self._edges_learned = 0

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
            token_idx = int(i * self.vocab_size / n_token) if n_token > 0 else i
            vec = token_vecs[token_idx] / (np.linalg.norm(token_vecs[token_idx]) + 1e-15)
            self.graph.add_node(vec, label=f"tok_{token_idx}")

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
        results = self.graph.find_similar(embed_vec, k=1)
        return results[0][0] if results else -1

    def _nearest_concepts(self, embed_vec: np.ndarray, k: int = 3) -> List[int]:
        results = self.graph.find_similar(embed_vec, k=k)
        return [r[0] for r in results]

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
        node_sims = []
        for nid, node in self.graph.nodes.items():
            sim = np.dot(node.vector, z_norm)
            node_sims.append((nid, sim))
        node_sims.sort(key=lambda x: x[1], reverse=True)

        edge_pred_list = []
        for nid, sim in node_sims[:7]:
            self.graph.activate(nid, max(0.01, sim))
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

        # ── Context Priming with Temporal Decay (inverted index) ──
        if T > 1:
            # Build context vector for disambiguation (must be concept_dim, not n_hidden)
            if np.linalg.norm(self.graph.temporal_context) > 0:
                ctx_vec = self.graph.temporal_context
            else:
                # Project hidden state into concept space
                ctx_vec = self.concept_predictor(StateTensor(h[np.newaxis, :])).data[0]

            # Only iterate tokens reachable from context concepts (O(B*T) not O(V*T))
            candidate_tokens: Set[int] = set()
            for ctx_nid in context_concepts:
                candidate_tokens.update(self._concept_to_tokens.get(ctx_nid, set()))
            candidate_tokens.discard(int(token_ids[0, -1]))

            for tok_id in candidate_tokens:
                # Use binding map for disambiguation if token is ambiguous
                if self.binding_map.is_ambiguous(tok_id):
                    tok_concept = self.binding_map.disambiguate(
                        tok_id, ctx_vec, self.graph, suppression_rate=0.05
                    )
                else:
                    tok_concept = self._token_concept_map[tok_id]
                    if tok_concept < 0:
                        continue

                for i, ctx_nid in enumerate(context_concepts):
                    ce = self.graph.get_edge(ctx_nid, tok_concept)
                    if ce is not None and ce.weight > 0.01:
                        # Temporal decay: more recent context is stronger
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
        self.free_energy_engine.accumulate_semantic(conceptual_error * 1.5, salience=0.5)

        edge_pred_set = set(self._last_edge_pred)
        single_correct = output_concept in edge_pred_set

        if not single_correct and input_concept >= 0:
            inode = self.graph.get_node(input_concept)
            if inode:
                inode.contradiction_count += 1
                inode.prediction_free_energy += 0.5
                # Trigger hotspot tracking when free energy exceeds threshold
                if inode.prediction_free_energy > 5.0:
                    self.graph.contradiction_hotspots.add(input_concept)

            # Also track contradictions on predicted concepts that missed
            if len(edge_pred_set) > 0:
                self.graph.apply_prediction_error(
                    list(edge_pred_set), next_embed
                )

        if single_correct and input_concept >= 0:
            inode = self.graph.get_node(input_concept)
            if inode and inode.contradiction_count > 0:
                inode.contradiction_count = max(0, inode.contradiction_count - 1)
                inode.prediction_free_energy = max(0.0, inode.prediction_free_energy - 0.3)

        if T > 1:
            target_onehot = np.zeros(self.vocab_size, dtype=np.float32)
            target_onehot[next_id] = 1.0

            # === Predictive Coding: Settle + Local Error ===
            # Collect hidden states at each layer
            h_states = [self._last_hidden_state]
            h_temp = self._last_hidden_state
            for layer in self.hidden_layers:
                h_temp = layer(StateTensor(h_temp[np.newaxis, :])).data[0]
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
            ctx_logits_now = self.context_logits(
                StateTensor(self._last_hidden_state[np.newaxis, :])
            ).data.flatten()
            ctx_probs_now = F.softmax(
                StateTensor(ctx_logits_now[np.newaxis, :]), dim=-1
            ).data.flatten()
            raw_error = target_onehot - ctx_probs_now
            h_2d = self._last_hidden_state.reshape(1, -1)
            e_2d = raw_error.reshape(1, -1)
            direct_lr = 0.0001
            direct_update = (e_2d.T @ h_2d) * direct_lr
            self.context_logits.weight.data += direct_update
            np.clip(self.context_logits.weight.data, -5.0, 5.0,
                    out=self.context_logits.weight.data)

            # Hidden layers: error between what layer above predicted and actual
            for i, layer in enumerate(self.hidden_layers):
                layer_err = local_errors[i] * 2.0
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

        if self._step_counter % self.sleep_interval == 0:
            self.sleep_cycle()

        return conceptual_error

    def _update_token_concept_map(self):
        # Use vectorized find_similar for batch concept lookup
        if not self.graph.nodes:
            return
        # Pre-compute all token embeddings
        token_embeds = np.zeros((self.vocab_size, self.embed_dim), dtype=np.float32)
        for tid in range(self.vocab_size):
            token_embeds[tid] = self.token_embed(StateTensor(np.array([tid]))).data[0]
        # Batch nearest concept lookup via graph's vectorized find_similar
        for tid in range(self.vocab_size):
            results = self.graph.find_similar(token_embeds[tid], k=1)
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

        for step in range(self.settle_steps):
            errors = []

            for i in range(n_states):
                if i < n_hidden:
                    # Hidden layer i predicts layer i+1's current state
                    pred = self.hidden_layers[i](
                        StateTensor(states[i][np.newaxis, :])
                    ).data[0]
                    pred = np.tanh(pred)

                    # A. Prediction residual normalization
                    # Prevents giant attractors from dominating error landscape
                    pred_norm = eps + np.linalg.norm(pred)
                    e = (states[i + 1] - pred) / pred_norm
                else:
                    # Top hidden layer predicts context logits
                    ctx = self.context_logits(
                        StateTensor(states[i][np.newaxis, :])
                    ).data.flatten()
                    ctx_dist = F.softmax(
                        StateTensor(ctx[np.newaxis, :]), dim=-1
                    ).data.flatten()

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
                    top_down = errors[i] @ self.context_logits.weight.data

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
                pred = self.hidden_layers[i](
                    StateTensor(states[i][np.newaxis, :])
                ).data[0]
                pred = np.tanh(pred)
                pred_norm = eps + np.linalg.norm(pred)
                e = (states[i + 1] - pred) / pred_norm
            else:
                ctx = self.context_logits(
                    StateTensor(states[i][np.newaxis, :])
                ).data.flatten()
                ctx_dist = F.softmax(
                    StateTensor(ctx[np.newaxis, :]), dim=-1
                ).data.flatten()
                if target is not None:
                    pred_norm = eps + np.linalg.norm(ctx_dist)
                    e = (target - ctx_dist) / pred_norm
                else:
                    pred_norm = eps + np.linalg.norm(ctx_dist)
                    e = -ctx_dist / pred_norm
            final_errors.append(e)

        return states, final_errors

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

        # ── Contradiction Resolution ──
        # Convert weak excitatory edges between contradictory concepts to inhibitory
        self.graph.form_inhibitory_edges()

        # Split concepts that have accumulated enough contradiction + drift + entropy
        for nid in list(self.graph.contradiction_hotspots):
            if self.graph.should_split(nid):
                self.graph.split_concept(nid, binding_map=self.binding_map)

        # Global synaptic downscaling — prevents runaway weights
        self.graph.homeostatic_downscale()

        # Reconcile: reset contradiction counts, reduce free energy on hotspots
        self.graph.reconcile_contradictions()

        # Binding maintenance: decay old bindings, prune weak ones
        self.binding_map.decay_all(rate=0.005)
        self.binding_map.prune(min_strength=0.05)

        self.sleep_cycles_completed += 1

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
        x = self.token_embed(StateTensor(np.array([token_id]))).data[0]

        # Recurrent step
        combined = np.concatenate([x, h_prev])
        h = self.recurrent_cell(StateTensor(combined[np.newaxis, :])).data[0]
        h = np.tanh(h)
        for layer in self.hidden_layers:
            h = layer(StateTensor(h[np.newaxis, :])).data[0]
            h = np.tanh(h)

        # Concept prediction from hidden state
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
        node_sims = []
        for nid, node in self.graph.nodes.items():
            sim = np.dot(node.vector, z_norm)
            effective_sim = sim * (1.0 - getattr(node, 'fatigue', 0.0))
            node_sims.append((nid, effective_sim))
        node_sims.sort(key=lambda x: x[1], reverse=True)

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
            norm = np.linalg.norm(node.vector) + 1e-15
            vec_norm = node.vector / norm
            local = (token_norms @ vec_norm) * node.effective_activation
            concept_scores = np.maximum(concept_scores, local)

        # ── Context Priming with Temporal Decay ──
        if context_concepts is not None and len(context_concepts) > 0:
            T_len = len(context_concepts)
            if np.linalg.norm(self.graph.temporal_context) > 0:
                ctx_vec = self.graph.temporal_context
            else:
                ctx_vec = z
                         

            for tok_id in range(self.vocab_size):
                if tok_id == token_id:
                    continue

                if self.binding_map.is_ambiguous(tok_id):
                    tok_concept = self.binding_map.disambiguate(
                        tok_id, ctx_vec, self.graph, suppression_rate=0.05
                    )
                else:
                    tok_concept = self._token_concept_map[tok_id]
                    if tok_concept < 0:
                        x_tok = self.token_embed(StateTensor(np.array([tok_id]))).data[0]
                        tok_concept = self._nearest_concept(x_tok)
                
                for i, ctx_nid in enumerate(context_concepts):
                    ce = self.graph.get_edge(ctx_nid, tok_concept)
                    if ce is not None and ce.weight > 0.01:
                        dist = T_len - 1 - i
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
            # Free energy accumulator state
            "free_energy_state": {
                "semantic_free_energy": self.free_energy_engine.semantic_free_energy,
                "episodic_free_energy": self.free_energy_engine.episodic_free_energy,
                "contradiction_free_energy": self.free_energy_engine.contradiction_free_energy,
                "linguistic_free_energy": self.free_energy_engine.linguistic_free_energy,
                "abstraction_free_energy": self.free_energy_engine.abstraction_free_energy,
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

        # Restore free energy accumulator state
        ps = checkpoint.get("free_energy_state", checkpoint.get("pressure_state", {}))
        model.free_energy_engine.semantic_free_energy = ps.get("semantic_free_energy", 0.0)
        model.free_energy_engine.episodic_free_energy = ps.get("episodic_free_energy", 0.0)
        model.free_energy_engine.contradiction_free_energy = ps.get("contradiction_free_energy", 0.0)
        model.free_energy_engine.linguistic_free_energy = ps.get("linguistic_free_energy", 0.0)
        model.free_energy_engine.abstraction_free_energy = ps.get("abstraction_free_energy", 0.0)

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

        # Node vectors
        for nid, node in self.graph.nodes.items():
            arrays[f"node/{nid}"] = node.vector

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
            }

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
            }

        graph_json = {
            "dim": self.graph.dim,
            "max_nodes": self.graph.max_nodes,
            "next_id": self.graph.next_id,
            "total_free_energy": float(self.graph.total_free_energy),
            "contradiction_hotspots": sorted(int(x) for x in self.graph.contradiction_hotspots),
            "nodes": nodes_json,
            "edges": edges_json,
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

            # Restore nodes
            for nid_str, nd in graph_data["nodes"].items():
                nid = int(nid_str)
                vec_key = f"node/{nid}"
                vector = npz[vec_key] if vec_key in npz else np.zeros(graph_data["dim"], dtype=np.float32)
                node = ConceptNode(
                    node_id=nd["id"],
                    vector=vector,
                    label=nd["label"],
                )
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
                model.graph.nodes[nid] = node

            # Restore edges
            for key, ed in graph_data["edges"].items():
                edge = ConceptEdge(
                    source=ed["source"],
                    target=ed["target"],
                    weight=ed["weight"],
                    edge_type=ed.get("edge_type", "excitatory"),  # backwards compatible
                )
                edge.confidence = ed["confidence"]
                edge.prediction_free_energy = ed.get("free_energy", ed.get("pressure", 0.0))
                edge.stability = ed["stability"]
                edge.timestamp = ed["timestamp"]
                edge.prediction_count = ed["prediction_count"]
                edge.shortcut = ed["shortcut"]
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

            return model
