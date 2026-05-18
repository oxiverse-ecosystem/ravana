import numpy as np
import time
from typing import Optional, List, Tuple, Dict
from ..tensor import StateTensor, RawTensor, tensor, Parameter
from ..graph import ConceptGraph
from ..propagation import PropagationEngine
from ..pressure import PressureAccumulator
from ..plasticity import HebbianPlasticity, AntiHebbianPlasticity, StructuralPlasticity
from . import functional as F
from .module import Module, Embedding, Linear, LayerNorm, Dropout


class RLM(Module):
    def __init__(self, vocab_size: int = 4096, embed_dim: int = 256,
                 concept_dim: int = 256, n_concepts: int = 8192,
                 n_hidden: int = 512, n_layers: int = 4,
                 max_seq_len: int = 128, pressure_threshold: float = 8.0,
                 sleep_interval: int = 20,
                 structured_embeddings: bool = True):
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

        self.token_embed = Embedding(vocab_size, embed_dim)
        self.pos_embed = Embedding(max_seq_len, embed_dim)
        if structured_embeddings:
            self._init_structured_embeddings()

        self.hidden_layers = []
        for i in range(n_layers):
            layer = Linear(embed_dim if i == 0 else n_hidden, n_hidden, bias=True)
            self.hidden_layers.append(layer)
            self.register_module(f'hidden_{i}', layer)

        self.layer_norm = LayerNorm(n_hidden)
        self.out_proj = Linear(concept_dim, vocab_size, bias=True)

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

        self._token_trace: List[int] = []
        self._last_predicted_concepts: List[int] = []
        self._last_input_concepts: List[int] = []
        self.sleep_cycles_completed = 0
        self.total_pressure = 0.0
        self.conceptual_accuracy = 0.0
        self.n_predictions = 0
        self._step_counter = 0

        # Log: edges learned per step
        self._edges_learned = 0

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
        token_vecs = self.token_embed.weight.data.copy()
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
            sim = float(np.dot(embed_vec, node.vector) /
                        (np.linalg.norm(embed_vec) * np.linalg.norm(node.vector) + 1e-15))
            if sim > best_sim:
                best_sim = sim
                best_id = nid
        return best_id

    def _nearest_concepts(self, embed_vec: np.ndarray, k: int = 5) -> List[int]:
        return [nid for nid, _ in self.graph.find_similar(embed_vec, k=k)]

    # ──────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────

    def forward(self, token_ids: np.ndarray) -> StateTensor:
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]
        B, T = token_ids.shape
        assert T <= self.max_seq_len

        # Raw token embedding for concept binding (preserves semantic topology)
        raw_tok = self.token_embed(StateTensor(token_ids))
        positions = np.arange(T, dtype=np.int64)
        pos = self.pos_embed(StateTensor(positions))
        tok_plus_pos = raw_tok + pos

        # Surface fluency path (hidden layers for token-level processing)
        x = tok_plus_pos
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.layer_norm(x)
        hidden = x[:, -1, :].data if x.ndim == 3 else x.data[-1:] if x.ndim == 2 else x.data
        if hidden.ndim == 1:
            hidden = hidden[np.newaxis, :]

        # Use raw token embedding (no position) for concept binding.
        # Position embedding corrupts the structured topology (random weights, large norm).
        bind_vec = raw_tok[:, -1, :].data if raw_tok.ndim == 3 else raw_tok.data
        if bind_vec.ndim == 1:
            bind_vec = bind_vec[np.newaxis, :]

        self.graph.reset_activation()
        input_concepts = self.graph.bind_input(bind_vec[0], k=5)

        self.graph.spread_activation(steps=2, k_active=7, decay=0.5)
        edge_pred = self.propagation.get_prediction(input_concepts, top_k=5)
        all_active = [n.id for n in sorted(self.graph.nodes.values(),
                                            key=lambda n: n.activation, reverse=True)
                      if n.activation > 0.01][:5]

        # For tracking: full prediction set (edge + geometric + input)
        predicted = list(dict.fromkeys(edge_pred + all_active + input_concepts))[:5]
        self._last_input_concepts = input_concepts
        self._last_predicted_concepts = predicted

        # For DECODING: use only edge-predicted concepts (interpolated ones excluded).
        # Edge predictions follow learned causal chains; geometric neighbors are noise.
        token_norms = self.token_embed.weight.data
        token_norms = token_norms / (np.linalg.norm(token_norms, axis=1, keepdims=True) + 1e-15)
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        for nid in edge_pred:
            if nid >= self.vocab_size:
                continue
            node = self.graph.get_node(nid)
            if node and node.activation > 0.01:
                norm = np.sqrt(np.dot(node.vector, node.vector) + 1e-15)
                vec_norm = node.vector / norm
                scores += token_norms @ vec_norm * node.activation * node.confidence
        logits = StateTensor(scores[np.newaxis, :] * 15.0)
        return logits[0] if logits.shape[0] == 1 else logits

    # ──────────────────────────────────────────────────────────────
    # Learn: direct concept-token binding + targeted Hebbian
    # ──────────────────────────────────────────────────────────────

    def learn(self, token_ids: np.ndarray, next_token_ids: np.ndarray):
        self._step_counter += 1
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]

        logits = self.forward(token_ids)

        if isinstance(next_token_ids, np.ndarray) and next_token_ids.ndim > 0:
            next_id = int(next_token_ids[0]) if next_token_ids.ndim == 1 else int(next_token_ids[0, 0])
        else:
            next_id = int(next_token_ids)

        last_input_id = int(token_ids[0, -1]) if token_ids.ndim > 1 else int(token_ids[-1])

        next_embed = self.token_embed(StateTensor(np.array([next_id]))).data[0]

        # ── Map input and output tokens to their nearest concepts ──
        input_concept = self._nearest_concept(
            self.token_embed(StateTensor(np.array([last_input_id]))).data[0])
        output_concept = self._nearest_concept(next_embed)

        # ════════════════════════════════════════════════════════════
        # PRIMARY: Direct error-correcting Hebbian on concept pair
        # ════════════════════════════════════════════════════════════

        if input_concept >= 0 and output_concept >= 0 and input_concept != output_concept:
            edge = self.graph.get_or_create_edge(input_concept, output_concept, weight=0.3)
            edge.weight = min(1.0, edge.weight + 0.05)
            edge.confidence = min(1.0, edge.confidence + 0.03)
            self._edges_learned += 1

        # ════════════════════════════════════════════════════════════
        # SECONDARY: Conceptual prediction error (for pressure)
        # ════════════════════════════════════════════════════════════

        predicted_set = set(self._last_predicted_concepts)
        actual_set = set(self._nearest_concepts(next_embed, k=5))

        n_overlap = len(predicted_set & actual_set)
        n_union = max(1, len(predicted_set | actual_set))
        overlap_ratio = n_overlap / n_union
        conceptual_error = 1.0 - overlap_ratio
        self.pressure.accumulate_semantic(conceptual_error * 1.5, salience=0.5)

        single_correct = output_concept in predicted_set

        # ════════════════════════════════════════════════════════════
        # Expression pressure (token confidence)
        # ════════════════════════════════════════════════════════════

        logit_dist = F.softmax(logits, dim=-1).data.flatten()
        entropy = -np.sum(logit_dist * np.log(logit_dist + 1e-15))
        entropy /= np.log(self.vocab_size)
        self.pressure.accumulate_episodic(entropy * 0.3)

        # ── Track ──
        self._token_trace.append(next_id)
        self.total_pressure = self.pressure.total

        factor = 0.95 if single_correct else 0.05
        self.conceptual_accuracy = 0.9 * self.conceptual_accuracy + 0.1 * factor
        self.n_predictions += 1

        if self._step_counter % self.sleep_interval == 0:
            self.sleep_cycle()

        return conceptual_error

    # ──────────────────────────────────────────────────────────────
    # Sleep
    # ──────────────────────────────────────────────────────────────

    def sleep_cycle(self):
        if self.pressure.total < self.pressure_threshold * 0.25:
            return None

        stages = {'hotspots': 0, 'reconciled': 0, 'pruned': 0, 'formed': 0, 'vector_adj': 0}

        hotspots = self.graph.get_hotspots(threshold=2.0)
        stages['hotspots'] = len(hotspots)

        reconciled = self.graph.reconcile_contradictions()
        stages['reconciled'] = reconciled

        pruned, formed = self.structural.step()
        stages['pruned'] = pruned
        stages['formed'] = formed

        vec_adj = 0
        token_vecs = self.token_embed.weight.data
        for i in range(min(len(token_vecs), len(self.graph.nodes))):
            node = self.graph.get_node(i)
            if node is None:
                continue
            tok_norm = token_vecs[i] / (np.linalg.norm(token_vecs[i]) + 1e-15)
            delta = tok_norm - (node.vector / (np.linalg.norm(node.vector) + 1e-15))
            if np.linalg.norm(delta) > 0.01:
                self.graph.adjust_vector(i, delta, lr=0.1)
                vec_adj += 1
        stages['vector_adj'] = vec_adj

        self.pressure.decay(rate=0.3)
        self.total_pressure = self.pressure.total
        self.sleep_cycles_completed += 1

        for mod in self._modules.values():
            if hasattr(mod, 'sleep_cycle'):
                mod.sleep_cycle()

        return stages

    # ──────────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────────

    def generate(self, prompt_ids: List[int], max_tokens: int = 50,
                 temperature: float = 0.8, learn_on_fly: bool = False) -> List[int]:
        generated = list(prompt_ids)
        for _ in range(max_tokens):
            if len(generated) > self.max_seq_len:
                generated = generated[-self.max_seq_len:]
            input_ids = np.array([generated], dtype=np.int64)
            logits = self.forward(input_ids)
            probs = F.softmax(logits / temperature, dim=-1).data.flatten()
            probs = np.maximum(probs, 1e-15)
            probs /= probs.sum()
            next_id = int(np.random.choice(self.vocab_size, p=probs))
            generated.append(next_id)
            if next_id == 0:
                break
        return generated

    def __repr__(self):
        return (f"RLM(vocab={self.vocab_size}, embed={self.embed_dim}, "
                f"concepts={len(self.graph.nodes)}, edges={len(self.graph.edges)}, "
                f"sleep={self.sleep_cycles_completed}, acc={self.conceptual_accuracy:.3f}, "
                f"learned_edges={self._edges_learned})")
