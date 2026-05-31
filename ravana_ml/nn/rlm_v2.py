"""
RLM v2 — Triple-Based Cognitive Architecture

Brain-inspired semantic memory that decomposes input into (subject, relation_type, object)
triples and uses spreading activation over a concept graph for inference.

Key differences from v1 (rlm.py):
- No character-level GRU — triple decomposition replaces sequential processing
- No 5-path logit blend — spreading activation is the sole inference mechanism
- Learned relation type embeddings — not keyword-based classification
- Hebbian learning on (subject ⊗ relation_type) → object associations
- Sleep cycles consolidate triple associations and prune weak edges

Architecture:
    input "heat causes expansion"
        → decompose: (subject="heat", relation="causes", object="expansion")
        → classify relation: "causes" → CAUSAL type embedding
        → find subject concept node in graph
        → spread activation from subject, filtered by relation type
        → score activated nodes against all token embeddings
        → return logits over vocab

    Learning: Hebbian edge strengthening + relation vector updates
    Sleep: consolidate triples, prune weak edges, replay important triples

Inspired by:
- Spreading activation in human semantic memory (Collins & Loftus, 1975)
- Compositional knowledge representation (Tenenbaum et al.)
- Hebbian synaptic plasticity (Hebb, 1949)
"""

import numpy as np
import time
import pickle
from typing import Optional, List, Tuple, Dict, Set
from collections import defaultdict

from .module import Module, Linear, Embedding
from ..graph import ConceptGraph, ConceptNode, ConceptEdge, ConceptBindingMap
from ..plasticity import HebbianPlasticity, AntiHebbianPlasticity, StructuralPlasticity
from ..propagation import PropagationEngine


# ─── Relation Type Definitions ───────────────────────────────────────────────

RELATION_TYPES = [
    "causal",       # causes, produces, leads to, results in, makes, triggers
    "semantic",     # is, are, represents, defines, means
    "temporal",     # then, after, before, next, later, during
    "possessive",   # has, contains, includes, belongs to, part of
    "analogical",   # like, similar to, resembles, analogous to
    "contextual",   # in, at, on, with, under, over
]

# Keyword → relation type mapping for initial classification
_KEYWORD_MAP = {
    "causal": [
        "causes", "cause", "produces", "produce", "leads", "results",
        "makes", "make", "triggers", "trigger", "creates", "create",
        "generates", "generate", "melts", "melt", "burns", "burn",
        "breaks", "break", "destroys", "destroy", "builds", "build",
        "grows", "grow", "changes", "change", "transforms", "transform",
        "converts", "convert", "affects", "affect", "influences", "influence",
        "powers", "power", "drives", "drive", "forces", "force",
        "heats", "heat", "cools", "cool", "freezes", "freeze",
        "dissolves", "dissolve", "evaporates", "evaporate",
        "compresses", "compress", "expands", "expand",
    ],
    "temporal": [
        "then", "after", "before", "next", "later", "during",
        "when", "while", "until", "since", "follows", "follow",
        "precedes", "precede", "succeeds", "succeed",
    ],
    "possessive": [
        "has", "have", "contains", "contain", "includes", "include",
        "belongs", "comprises", "comprise", "holds", "hold",
        "carries", "carry", "bears", "bear",
    ],
    "analogical": [
        "like", "similar", "resembles", "resemble", "analogous",
        "comparable", "equivalent", "parallel", "mirrors", "mirror",
    ],
    "contextual": [
        "in", "at", "on", "with", "under", "over", "near", "beside",
        "within", "among", "between", "through", "across", "along",
    ],
}


class RLMv2(Module):
    """Triple-based cognitive architecture with spreading activation inference.

    This model decomposes input text into (subject, relation_type, object) triples,
    stores them as typed edges in a concept graph, and infers predictions via
    spreading activation from subject nodes filtered by relation type.

    Args:
        vocab_size: Number of tokens in vocabulary
        embed_dim: Dimensionality of token embeddings
        concept_dim: Dimensionality of concept vectors (must match graph.dim)
        n_concepts: Initial number of concept nodes to create
        max_seq_len: Maximum sequence length (for positional encoding)
        sleep_interval: Learn steps between sleep cycles
        gate_concept_creation: If True, only create new concepts when similarity < threshold
        anchor_relation_vectors: If True, relation vectors seeded deterministically per type
    """

    def __init__(self, vocab_size: int, embed_dim: int, concept_dim: int,
                 n_concepts: int, max_seq_len: int = 128,
                 sleep_interval: int = 100,
                 gate_concept_creation: bool = True,
                 anchor_relation_vectors: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.concept_dim = concept_dim
        self.n_concepts = n_concepts
        self.max_seq_len = max_seq_len
        self.sleep_interval = sleep_interval
        self._gate_concept_creation = gate_concept_creation

        # ── Token Embeddings ──
        self.token_embed = Embedding(vocab_size, embed_dim)

        # ── Projection Layers (embed_dim ↔ concept_dim) ──
        # Project token embeddings into concept space
        self.subject_proj = Linear(embed_dim, concept_dim, bias=False)
        # Project concept vectors back to token embedding space for scoring
        self.concept_to_embed = Linear(concept_dim, embed_dim, bias=False)

        # ── Relation Type Classifier (the ONLY backprop-trained component) ──
        # Learned embeddings for each relation type
        n_rel_types = len(RELATION_TYPES)
        self.relation_type_embed = Embedding(n_rel_types, concept_dim)
        # Classifier: concept_dim → n_rel_types
        self.relation_classifier = Linear(concept_dim, n_rel_types, bias=True)

        # ── Concept Graph ──
        self.graph = ConceptGraph(
            dim=concept_dim,
            max_nodes=10000,
            anchor_relation_vectors=anchor_relation_vectors,
            adaptive_downscale=True,
        )

        # ── Binding Map (token ↔ concept) ──
        self.binding_map = ConceptBindingMap()

        # ── Propagation Engine ──
        self.propagation = PropagationEngine(self.graph)

        # ── Plasticity ──
        self.hebbian = HebbianPlasticity(self.graph, lr=0.01)
        self.anti_hebbian = AntiHebbianPlasticity(self.graph, lr=0.01)
        self.structural = StructuralPlasticity(self.graph)

        # ── Cognitive State ──
        self._step_counter = 0
        self._sleep_pressure = 0.0
        self._base_lr = 0.005

        # ── Concept creation gating ──
        self._concept_similarity_threshold = 0.7 if gate_concept_creation else 0.0
        self._max_concepts = max(n_concepts, int(vocab_size * 0.5))

        # ── Relation type classifier learning rate ──
        self._classifier_lr = 0.01

        # ── Episodic memory (triple store) ──
        # Stores (subject_cid, rel_type_idx, object_cid, timestamp) for replay
        self._episodic_triples: List[Tuple[int, int, int, float]] = []
        self._max_episodic = 500

        # ── Initialize structured concepts (DISABLED — 1-to-1 mapping) ──
        # self._init_structured_concepts()

        # ── Performance tracking ──
        self._train_correct = 0
        self._train_total = 0
        self._last_loss = 0.0

    def _init_structured_concepts(self):
        """Create initial concept nodes distributed across the concept space."""
        rng = np.random.RandomState(42)
        for i in range(self.n_concepts):
            vec = rng.randn(self.concept_dim).astype(np.float32) * 0.1
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            self.graph.add_node(vector=vec, label=f"init_{i}")

    # ── Dimension Bridging ──────────────────────────────────────────────────

    def _project_to_concept(self, embed_vec: np.ndarray) -> np.ndarray:
        """Project embed_dim vector to concept_dim space via learned projection."""
        if len(embed_vec) == self.concept_dim:
            return embed_vec
        # Use the subject projection layer
        return self.subject_proj(embed_vec.reshape(1, -1)).flatten()

    def _project_to_embed(self, concept_vec: np.ndarray) -> np.ndarray:
        """Project concept_dim vector to embed_dim space via learned projection."""
        if len(concept_vec) == self.embed_dim:
            return concept_vec
        return self.concept_to_embed(concept_vec.reshape(1, -1)).flatten()

    # ── Triple Decomposition ────────────────────────────────────────────────

    def decompose_triple(self, token_ids: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """Decompose token sequence into (subject, relation, object) triple.

        For 3-token input: subject=[t0], relation=[t1], object=[t2]
        For 2-token input: subject=[t0], relation=[], object=[t1]
        For 4+ tokens: subject=[t0], relation=[t1..tN-2], object=[tN-1]

        This mirrors how the brain decomposes "heat causes expansion" into
        (heat, causal-relation, expansion) rather than processing it as a
        flat character sequence.
        """
        n = len(token_ids)
        if n >= 3:
            return [token_ids[0]], token_ids[1:-1], [token_ids[-1]]
        elif n == 2:
            return [token_ids[0]], [], [token_ids[1]]
        elif n == 1:
            return [token_ids[0]], [], []
        else:
            return [], [], []

    def classify_relation(self, relation_token_ids: List[int]) -> int:
        """Classify relation tokens into a relation type index.

        Uses keyword matching for initial classification, then learned embeddings
        for refinement. Returns index into RELATION_TYPES.

        This is what makes "melts" → CAUSAL work (same pathway as "causes").
        """
        if not relation_token_ids:
            return RELATION_TYPES.index("semantic")  # default

        # Phase 1: keyword-based classification
        # Decode relation tokens to text
        rel_words = set()
        for tid in relation_token_ids:
            try:
                word = self._decode_token(tid).lower().strip()
                if word:
                    rel_words.add(word)
            except Exception:
                pass

        # Check keyword map
        for rel_type, keywords in _KEYWORD_MAP.items():
            for word in rel_words:
                if word in keywords:
                    return RELATION_TYPES.index(rel_type)

        # Default to semantic
        return RELATION_TYPES.index("semantic")

    def _classify_relation_learned(self, relation_token_ids: List[int]) -> Tuple[int, np.ndarray]:
        """Classify relation using learned embeddings + keyword fallback.

        Returns (type_index, type_embedding) where type_embedding is the
        learned vector for that relation type.

        This is the learned version that improves over time as the model
        sees more examples of each relation type.
        """
        # Get keyword-based classification as fallback
        keyword_idx = self.classify_relation(relation_token_ids)

        if not relation_token_ids:
            return keyword_idx, self.relation_type_embed.weight.data[keyword_idx]

        # Average the relation token embeddings
        rel_embeds = []
        for tid in relation_token_ids:
            rel_embeds.append(self.token_embed.weight.data[tid])
        rel_vec = np.mean(rel_embeds, axis=0)

        # Project to concept space
        rel_concept = self._project_to_concept(rel_vec)

        # Score against all relation type embeddings
        type_embeds = self.relation_type_embed.weight.data  # (n_types, concept_dim)
        # Cosine similarity
        rel_norm = np.linalg.norm(rel_concept)
        type_norms = np.linalg.norm(type_embeds, axis=1)
        if rel_norm > 0 and np.all(type_norms > 0):
            sims = type_embeds @ rel_concept / (type_norms * rel_norm)
        else:
            sims = np.zeros(len(RELATION_TYPES))

        # Softmax with temperature
        temp = 0.5
        exp_sims = np.exp(sims / temp - np.max(sims / temp))
        probs = exp_sims / (np.sum(exp_sims) + 1e-10)

        # Blend: 70% learned, 30% keyword
        blended = np.zeros(len(RELATION_TYPES))
        blended[keyword_idx] += 0.3
        blended += 0.7 * probs
        blended /= np.sum(blended) + 1e-10

        learned_idx = int(np.argmax(blended))
        return learned_idx, self.relation_type_embed.weight.data[learned_idx]

    def _decode_token(self, token_id: int) -> str:
        """Decode a single token ID to text."""
        try:
            if hasattr(self, '_tokenizer') and self._tokenizer is not None:
                return self._tokenizer.decode([token_id])
            # Fallback: treat as character
            if 0 <= token_id < 128:
                return chr(token_id)
            return f"tok_{token_id}"
        except Exception:
            return f"tok_{token_id}"

    # ── Concept Lookup ──────────────────────────────────────────────────────

    def _nearest_concept(self, embed_vec: np.ndarray) -> Tuple[int, float]:
        """Find the nearest concept node to an embedding vector.

        Uses the subject projection to map embed_dim → concept_dim,
        then cosine similarity against all concept vectors.

        Returns (concept_id, similarity_score).
        """
        concept_vec = self._project_to_concept(embed_vec)
        results = self.graph.find_similar(concept_vec, k=1)
        if results:
            return results[0]
        # Fallback: create a new concept
        return -1, 0.0

    def _get_or_create_concept(self, token_id: int, embed_vec: np.ndarray) -> int:
        """Get existing concept for a token, or create one if needed.

        1-to-1 mapping: each token gets exactly one concept. No merging.
        This prevents unrelated concepts from being collapsed together.
        """
        # Check binding map first — reuse existing concept for this token
        bindings = self.binding_map.get_concepts(token_id, min_confidence=0.1)
        if bindings:
            return bindings[0].concept_id

        # Always create a new concept for this token (no merging)
        concept_vec = self._project_to_concept(embed_vec)
        if len(self.graph.nodes) < self._max_concepts:
            node = self.graph.add_node(vector=concept_vec, label=self._decode_token(token_id))
            nid = node.id
        else:
            # At capacity — shouldn't happen with reasonable n_concepts
            nid = len(self.graph.nodes)

        # Bind token to concept
        self.binding_map.bind(token_id, nid, confidence=0.9)
        return nid

    # ── Spreading Activation Inference ──────────────────────────────────────

    def forward(self, token_ids: np.ndarray) -> 'tensor':
        """Forward pass: predict next token via spreading activation.

        Input: token_ids array (1D or 2D with batch dim)
        Output: logits over vocab_size

        This is the core inference mechanism:
        1. Decompose into (subject, relation, object) — object is what we predict
        2. Find subject concept node
        3. Classify relation type
        4. Spread activation from subject, filtered by relation type
        5. Score activated nodes against all token embeddings
        6. Return logits over vocab
        """
        from ..tensor import tensor as make_tensor

        # Flatten input
        if token_ids.ndim > 1:
            token_ids = token_ids.flatten()
        token_ids = token_ids.tolist()

        # Decompose triple — we predict the object given subject + relation
        subject_ids, relation_ids, object_ids = self.decompose_triple(token_ids)

        # If no subject, return uniform logits
        if not subject_ids:
            return make_tensor(np.zeros(self.vocab_size, dtype=np.float32))

        # Get subject embedding and concept
        subject_tid = subject_ids[0]
        subject_embed = self.token_embed.weight.data[subject_tid]
        subject_cid = self._get_or_create_concept(subject_tid, subject_embed)

        # Classify relation type
        rel_type_idx, rel_type_embed = self._classify_relation_learned(relation_ids)
        rel_type_name = RELATION_TYPES[rel_type_idx]

        # ── Vector Arithmetic (word2vec-style analogy) ──
        # Compute expected output: subject_embed + relation_vector
        # This enables cross-domain transfer via embedding space arithmetic.
        # "king - man + woman = queen" style: "anger + causal_vector ≈ expansion"
        analogy_targets = {}
        # Collect relation vectors from edges matching the query relation type
        rvs = []
        for (src, tgt), edge in self.graph.edges.items():
            if edge.relation_type == rel_type_name and hasattr(edge, 'relation_vector') and edge.relation_vector is not None:
                rvs.append(edge.relation_vector)
        if rvs:
            avg_rv = np.mean(rvs, axis=0)
            expected = subject_embed + avg_rv  # expected output embedding
            # Compare with all concept vectors
            for cid, node in self.graph.nodes.items():
                if cid == subject_cid:
                    continue
                cv = self._project_to_concept(expected)[:len(node.vector)]
                cv_norm = np.linalg.norm(cv)
                nv_norm = np.linalg.norm(node.vector)
                if cv_norm > 0 and nv_norm > 0:
                    sim = float(np.dot(cv, node.vector) / (cv_norm * nv_norm))
                    if sim > 0.3:  # threshold for analogy candidates
                        analogy_targets[cid] = sim * 2.0  # boost factor

        # ── Spreading Activation ──
        # Reset all activations
        for node in self.graph.nodes.values():
            node.activation = 0.0

        # Activate subject concept
        self.graph.activate(subject_cid, amount=1.0)

        # Phase 1: General spreading (3 steps, decay 0.3)
        self.graph.spread_activation(steps=3, k_active=10, decay=0.3)

        # Phase 2: Relation-aware spreading (2 extra steps along matching edges)
        # This preferentially spreads activation along edges that match the query
        # relation type, enabling deeper cross-domain paths.
        for _ in range(2):
            to_activate = []
            for nid, node in self.graph.nodes.items():
                if node.activation < 0.005:
                    continue
                for tgt_id, edge in self.graph.get_outgoing(nid):
                    if edge.relation_type == rel_type_name:
                        tgt_node = self.graph.get_node(tgt_id)
                        if tgt_node is not None:
                            to_activate.append((tgt_id, node.activation * 0.5 * edge.weight * edge.confidence))
            for nid, amount in to_activate:
                self.graph.activate(nid, amount=amount)

        # ── Score tokens via activated concepts ──
        concept_scores = np.zeros(self.vocab_size, dtype=np.float32)

        # Collect all active nodes with their activations
        active_nodes = []
        for nid, node in self.graph.nodes.items():
            if node.activation > 0.01:
                active_nodes.append((nid, node))

        # For each active node, check outgoing edges
        matching_targets = {}  # target_concept_id → score
        for nid, node in active_nodes:
            outgoing = self.graph.get_outgoing(nid)
            for tgt_id, edge in outgoing:
                # Filter by relation type — the KEY to cross-domain transfer
                # "causes" edges from heat activate expansion, not kindness
                type_match = (edge.relation_type == rel_type_name)

                # Compute edge score
                base_score = node.activation * edge.weight * edge.confidence

                # Relation type matching bonus
                if type_match:
                    base_score *= 3.0  # strong boost for matching type
                else:
                    base_score *= 0.1  # heavy penalty for non-matching type

                # Relation vector alignment bonus
                if hasattr(edge, 'relation_vector') and edge.relation_vector is not None:
                    rv = edge.relation_vector
                    rv_norm = np.linalg.norm(rv)
                    node_norm = np.linalg.norm(node.vector)
                    if rv_norm > 0 and node_norm > 0:
                        min_len = min(len(rv), len(node.vector))
                        alignment = float(np.dot(rv[:min_len], node.vector[:min_len]) / (rv_norm * node_norm))
                        base_score *= (1.0 + 0.5 * max(0, alignment))

                if tgt_id in matching_targets:
                    matching_targets[tgt_id] = max(matching_targets[tgt_id], base_score)
                else:
                    matching_targets[tgt_id] = base_score

        # ── Activation-gated relation-type query (cross-domain transfer) ──
        # Only boost causal edges whose source was ACTIVATED by spreading activation.
        # This prevents flooding from unrelated causal edges while still enabling
        # cross-domain transfer through activated intermediaries.
        if rel_type_name != "semantic":
            for nid, node in active_nodes:
                if node.activation < 0.01:
                    continue
                outgoing = self.graph.get_outgoing(nid)
                for tgt_id, edge in outgoing:
                    if edge.relation_type != rel_type_name:
                        continue
                    if tgt_id == subject_cid:
                        continue  # Don't predict subject
                    # Score: activation × edge weight × confidence × 2.0 boost
                    cross_score = node.activation * edge.weight * edge.confidence * 2.0
                    if tgt_id in matching_targets:
                        matching_targets[tgt_id] = max(matching_targets[tgt_id], cross_score)
                    else:
                        matching_targets[tgt_id] = cross_score

        # ── 2-Hop edge traversal (compositionality) ──
        # "fire causes ?" → fire→heat (any type), heat→expansion (causal) → boost expansion
        # This enables cross-subject transfer: fire isn't trained with expansion,
        # but heat is, and fire→heat exists.
        subject_outgoing = self.graph.get_outgoing(subject_cid)
        for mid_cid, mid_edge in subject_outgoing:
            mid_node = self.graph.get_node(mid_cid)
            if mid_node is None:
                continue
            # Check mid's outgoing edges for matching relation type
            for tgt_cid, tgt_edge in self.graph.get_outgoing(mid_cid):
                if tgt_cid == subject_cid:
                    continue
                if tgt_edge.relation_type == rel_type_name:
                    # 2-hop score: subject→mid weight × mid→target weight
                    hop_score = mid_edge.weight * mid_edge.confidence * tgt_edge.weight * tgt_edge.confidence * 3.0
                    if tgt_cid in matching_targets:
                        matching_targets[tgt_cid] = max(matching_targets[tgt_cid], hop_score)
                    else:
                        matching_targets[tgt_cid] = hop_score

        # Also include directly active concepts (for same-type queries)
        # Aggressive boosting: activated concepts get full activation as score
        for nid, node in active_nodes:
            if nid == subject_cid:
                continue  # Don't predict subject itself
            # Score based on activation strength (boosted heavily for multi-hop)
            act_score = node.activation * 3.0
            if nid in matching_targets:
                matching_targets[nid] = max(matching_targets[nid], act_score)
            else:
                matching_targets[nid] = act_score

            # Also boost targets of activated nodes' outgoing edges
            for tgt_id, edge in self.graph.get_outgoing(nid):
                if tgt_id == subject_cid:
                    continue
                edge_score = node.activation * edge.weight * edge.confidence * 0.8
                if edge.relation_type == rel_type_name:
                    edge_score *= 2.0  # boost matching relation type
                if tgt_id in matching_targets:
                    matching_targets[tgt_id] = max(matching_targets[tgt_id], edge_score)
                else:
                    matching_targets[tgt_id] = edge_score

        # ── Merge analogy targets (vector arithmetic) ──
        for cid, score in analogy_targets.items():
            if cid == subject_cid:
                continue
            if cid in matching_targets:
                matching_targets[cid] = max(matching_targets[cid], score)
            else:
                matching_targets[cid] = score

        # Map target concepts to token scores
        for tgt_cid, score in matching_targets.items():
            tgt_node = self.graph.get_node(tgt_cid)
            if tgt_node is None:
                continue

            # Method 1: Check binding map for bound tokens
            bindings = self.binding_map.get_tokens(tgt_cid, min_confidence=0.1)
            for binding in bindings:
                tok_id = binding.token_id
                if 0 <= tok_id < self.vocab_size:
                    concept_scores[tok_id] += score * binding.confidence

            # Method 2: Cosine similarity with all token embeddings
            tgt_embed = self._project_to_embed(tgt_node.vector)
            tgt_norm = np.linalg.norm(tgt_embed)
            if tgt_norm > 0:
                token_embeds = self.token_embed.weight.data  # (vocab_size, embed_dim)
                token_norms = np.linalg.norm(token_embeds, axis=1)
                valid = token_norms > 0
                sims = np.zeros(self.vocab_size, dtype=np.float32)
                sims[valid] = token_embeds[valid] @ tgt_embed / (token_norms[valid] * tgt_norm)
                concept_scores += sims * score * 0.3  # weight the similarity contribution

        # Suppress subject token self-prediction
        if 0 <= subject_tid < self.vocab_size:
            concept_scores[subject_tid] *= 0.1

        # Apply temperature-modulated softmax
        temp = 0.42  # Low temperature for discrimination (v1 breakthrough finding)
        if np.max(concept_scores) > 0:
            exp_scores = np.exp((concept_scores - np.max(concept_scores)) / temp)
            probs = exp_scores / (np.sum(exp_scores) + 1e-10)
            logits = np.log(probs + 1e-10)
        else:
            logits = concept_scores

        return make_tensor(logits.astype(np.float32))

    # ── Hebbian Learning ────────────────────────────────────────────────────

    def learn(self, token_ids: np.ndarray, target_ids: np.ndarray) -> Dict[str, float]:
        """Learn from a (context, target) pair via Hebbian triple updates.

        This is the core learning mechanism:
        1. Forward pass to get prediction
        2. Decompose into (subject, relation, object) triple
        3. Create/update concept nodes for subject and object
        4. Create/update typed edge (subject → object) with relation_type
        5. Hebbian update on edge weight
        6. Update relation vectors (pull toward target, push from negatives)
        7. Update relation classifier weights
        8. Update concept vectors (pull toward token embeddings)
        """
        from ..tensor import tensor as make_tensor

        # Flatten inputs
        if token_ids.ndim > 1:
            token_ids = token_ids.flatten()
        if target_ids.ndim > 1:
            target_ids = target_ids.flatten()

        input_ids = token_ids.tolist()
        target_id = int(target_ids.flatten()[0])

        # ── Forward pass ──
        logits_tensor = self.forward(token_ids)
        logits = logits_tensor.data.flatten() if hasattr(logits_tensor.data, 'flatten') else logits_tensor.data

        # Prediction error
        target_onehot = np.zeros(self.vocab_size, dtype=np.float32)
        target_onehot[target_id] = 1.0
        prediction_error = target_onehot - np.exp(logits)  # error in probability space

        # Track accuracy
        pred_id = int(np.argmax(logits))
        is_correct = pred_id == target_id
        if is_correct:
            self._train_correct += 1
        self._train_total += 1

        # ── Decompose triple ──
        # Reconstruct full triple from context + target for proper decomposition
        # e.g., "heat causes" + "expansion" → "heat causes expansion" → (heat, causes, expansion)
        full_triple_ids = input_ids + [target_id]
        subject_ids, relation_ids, object_ids = self.decompose_triple(full_triple_ids)

        # The target IS the object (what we're trying to predict)
        object_tid = target_id

        if not subject_ids:
            self._step_counter += 1
            return {"loss": float(np.mean(prediction_error ** 2)), "accuracy": self._train_correct / max(1, self._train_total)}

        subject_tid = subject_ids[0]

        # ── Get/create concept nodes ──
        subject_embed = self.token_embed.weight.data[subject_tid]
        object_embed = self.token_embed.weight.data[object_tid]

        subject_cid = self._get_or_create_concept(subject_tid, subject_embed)
        object_cid = self._get_or_create_concept(object_tid, object_embed)

        # ── Classify relation type ──
        rel_type_idx, rel_type_embed = self._classify_relation_learned(relation_ids)
        rel_type_name = RELATION_TYPES[rel_type_idx]

        # ── Create/update typed edge ──
        edge = self.graph.add_edge(
            source=subject_cid,
            target=object_cid,
            weight=0.3,
            relation_type=rel_type_name,
        )

        # Store predicate token (the verb/relation token)
        if relation_ids:
            edge.predicate_token_id = relation_ids[0]

        # ── Hebbian edge update ──
        # Strengthen edge on correct prediction, weaken on incorrect
        pred_error = 1.0 - edge.confidence
        surprise = abs(pred_error) * edge.confidence
        effective_lr = self._base_lr * (1.0 + surprise * 5.0)

        src_node = self.graph.get_node(subject_cid)
        tgt_node = self.graph.get_node(object_cid)

        if src_node is not None and tgt_node is not None:
            # Hebbian update: co-activation strengthens edge
            delta = effective_lr * src_node.activation * tgt_node.activation * pred_error
            edge.weight = max(0.0, min(1.0, edge.weight + delta))
            edge.confidence = min(1.0, edge.confidence + 0.03)
            edge.stability = min(1.0, edge.stability + 0.01)
            edge.prediction_count += 1

            if is_correct:
                edge.forward_pred_count += 1

            # ── Relation vector update (Hebbian pull + type anchor) ──
            # The relation vector encodes the relational pattern.
            # It pulls toward the target signal while maintaining type structure.
            tgt_vec = tgt_node.vector
            tgt_norm = np.linalg.norm(tgt_vec)
            if tgt_norm > 0:
                tgt_signal = tgt_vec / tgt_norm
                # 3-way blend: 70% current + 20% target + 10% type seed
                # This prevents EMA from erasing type-specific structure
                type_seed = ConceptEdge._init_relation_vector(rel_type_name, len(edge.relation_vector))
                edge.relation_vector = (
                    0.70 * edge.relation_vector +
                    0.20 * tgt_signal[:len(edge.relation_vector)] +
                    0.10 * type_seed
                )
                rv_norm = np.linalg.norm(edge.relation_vector)
                if rv_norm > 0:
                    edge.relation_vector /= rv_norm

            # ── Contrastive push: repel from different-target edges ──
            outgoing = self.graph.get_outgoing(subject_cid)
            for other_tgt_id, other_edge in outgoing:
                if other_tgt_id == object_cid:
                    continue
                # Push relation vectors apart for different targets
                other_rv = other_edge.relation_vector
                push_strength = 0.05
                edge.relation_vector -= push_strength * other_rv[:len(edge.relation_vector)]
                rv_norm = np.linalg.norm(edge.relation_vector)
                if rv_norm > 0:
                    edge.relation_vector /= rv_norm

            # ── Concept vector updates ──
            # Pull concept vectors toward their bound token embeddings
            pull_lr = 0.005

            # Subject concept → subject token embedding
            subject_concept_vec = self._project_to_concept(subject_embed)
            src_delta = pull_lr * (subject_concept_vec - src_node.vector)
            src_delta = np.clip(src_delta, -self.graph.max_step_delta, self.graph.max_step_delta)
            src_node.vector += src_delta
            src_norm = np.linalg.norm(src_node.vector)
            if src_norm > 0:
                src_node.vector /= src_norm

            # Object concept → object embedding
            object_concept_vec = self._project_to_concept(object_embed)
            tgt_delta = pull_lr * (object_concept_vec - tgt_node.vector)
            tgt_delta = np.clip(tgt_delta, -self.graph.max_step_delta, self.graph.max_step_delta)
            tgt_node.vector += tgt_delta
            tgt_norm = np.linalg.norm(tgt_node.vector)
            if tgt_norm > 0:
                tgt_node.vector /= tgt_norm

        # ── Update relation classifier weights (backprop on classifier only) ──
        self._update_relation_classifier(relation_ids, rel_type_idx)

        # ── Train token embeddings (co-occurrence similarity) ──
        # Pull subject and object token embeddings together — this creates
        # word2vec-style semantic similarity. "heat" and "expansion" become
        # similar because they appear in the same triple.
        # This is what makes cross-subject transfer work: "fire" and "heat"
        # both co-occur with similar objects, so their embeddings converge.
        if is_correct:
            embed_lr = 0.002
            subj_emb = self.token_embed.weight.data[subject_tid]
            obj_emb = self.token_embed.weight.data[object_tid]
            # Pull together (asymmetric: subject moves more)
            delta = embed_lr * (obj_emb - subj_emb)
            self.token_embed.weight.data[subject_tid] += delta
            self.token_embed.weight.data[object_tid] -= delta * 0.3

        # ── Store episodic triple ──
        self._episodic_triples.append((subject_cid, rel_type_idx, object_cid, time.time()))
        if len(self._episodic_triples) > self._max_episodic:
            self._episodic_triples = self._episodic_triples[-self._max_episodic:]

        # ── Sleep pressure accumulation ──
        if not is_correct:
            self._sleep_pressure += 0.015
        else:
            self._sleep_pressure += 0.005  # Even correct predictions accumulate (slowly)

        # ── Auto-sleep check ──
        self._step_counter += 1
        if (self._sleep_pressure > 0.7 and
                self._step_counter - getattr(self, '_last_sleep_step', 0) > 200):
            self.sleep_cycle()
            self._last_sleep_step = self._step_counter

        # Periodic sleep
        if self.sleep_interval > 0 and self._step_counter % self.sleep_interval == 0:
            self.sleep_cycle()

        loss = float(np.mean(prediction_error ** 2))
        self._last_loss = loss

        return {
            "loss": loss,
            "accuracy": self._train_correct / max(1, self._train_total),
            "is_correct": is_correct,
            "pred_id": pred_id,
            "target_id": target_id,
            "relation_type": rel_type_name,
            "subject_cid": subject_cid,
            "object_cid": object_cid,
        }

    def _update_relation_classifier(self, relation_token_ids: List[int], true_type_idx: int):
        """Update relation classifier weights via local Hebbian-style learning.

        This is the ONLY backprop-trained component. It learns to map
        relation token embeddings to relation type indices.

        Uses a simple local learning rule: pull weights toward correct type,
        push away from incorrect types.
        """
        if not relation_token_ids:
            return

        # Average relation token embeddings
        rel_embeds = []
        for tid in relation_token_ids:
            rel_embeds.append(self.token_embed.weight.data[tid])
        rel_vec = np.mean(rel_embeds, axis=0)

        # Project to concept space
        rel_concept = self._project_to_concept(rel_vec)

        # Current classifier output
        logits_tensor = self.relation_classifier(rel_concept.reshape(1, -1))
        logits = logits_tensor.data.flatten() if hasattr(logits_tensor.data, 'flatten') else logits_tensor.data.flatten()

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)

        # Error: target is one-hot at true_type_idx
        error = np.zeros(len(RELATION_TYPES), dtype=np.float32)
        error[true_type_idx] = 1.0 - probs[true_type_idx]
        for i in range(len(RELATION_TYPES)):
            if i != true_type_idx:
                error[i] = -probs[i]

        # Update classifier weights: dW = error^T @ input
        lr = self._classifier_lr
        input_2d = rel_concept.reshape(1, -1)
        error_2d = error.reshape(-1, 1)
        self.relation_classifier.weight.data += lr * (error_2d @ input_2d)
        if self.relation_classifier.bias is not None:
            self.relation_classifier.bias.data += lr * error

        # Also update relation type embeddings: pull the correct type toward rel_concept
        type_embed = self.relation_type_embed.weight.data[true_type_idx]
        pull = 0.01 * (rel_concept[:len(type_embed)] - type_embed)
        self.relation_type_embed.weight.data[true_type_idx] += pull
        # Renormalize
        norm = np.linalg.norm(self.relation_type_embed.weight.data[true_type_idx])
        if norm > 0:
            self.relation_type_embed.weight.data[true_type_idx] /= norm

    # ── Sleep Cycle ─────────────────────────────────────────────────────────

    def sleep_cycle(self):
        """Sleep cycle: consolidate triples, prune weak edges, replay important memories.

        This is the brain's "offline consolidation" — during sleep, the model:
        1. Replays important episodic triples (hippocampal replay)
        2. Consolidates edge weights (homeostatic downscaling)
        3. Prunes weak/unstable edges
        4. Updates concept vectors (drift defense)
        """
        # ── Hippocampal replay ──
        # Replay the most recent/important triples
        if self._episodic_triples:
            n_replay = min(10, len(self._episodic_triples))
            replay_triples = self._episodic_triples[-n_replay:]

            for subj_cid, rel_idx, obj_cid, ts in replay_triples:
                src = self.graph.get_node(subj_cid)
                tgt = self.graph.get_node(obj_cid)
                if src is None or tgt is None:
                    continue

                edge = self.graph.get_edge(subj_cid, obj_cid)
                if edge is None:
                    continue

                # Strengthen edge through replay
                edge.weight = min(1.0, edge.weight + 0.02)
                edge.confidence = min(1.0, edge.confidence + 0.01)

                # Hebbian co-activation replay
                self.graph.hebbian_update(subj_cid, obj_cid, coactivation=0.5, lr=0.005)

        # ── Homeostatic downscaling ──
        # Normalize edge weights to prevent runaway strengthening
        self._normalize_outgoing_weights(budget=3.0)

        # ── Prune weak edges ──
        self._prune_weak_edges(threshold=0.05)

        # ── Drift defense ──
        # Pull concept vectors back toward their core vectors if they've drifted too far
        for nid, node in self.graph.nodes.items():
            drift = node.drift_magnitude
            if drift > 0.4:
                # Pull back toward core vector
                pull = 0.1 * (node.core_vector - node.vector)
                node.vector += pull
                norm = np.linalg.norm(node.vector)
                if norm > 0:
                    node.vector /= norm

        # ── Reset sleep pressure ──
        self._sleep_pressure = 0.0

        # ── Reset activations ──
        for node in self.graph.nodes.values():
            node.activation = 0.0
            node.fatigue = 0.0

    def _normalize_outgoing_weights(self, budget: float = 3.0):
        """Normalize outgoing edge weights so total doesn't exceed budget.

        Prevents runaway edge strengthening from hippocampal replay.
        """
        for nid in self.graph._outgoing:
            edges = self.graph._outgoing.get(nid, [])
            if not edges:
                continue
            total_weight = sum(e.weight for _, e in edges)
            if total_weight > budget:
                scale = budget / total_weight
                for _, edge in edges:
                    edge.weight *= scale

    def _prune_weak_edges(self, threshold: float = 0.05):
        """Remove edges with weight below threshold.

        Keeps the graph clean and prevents accumulation of noise edges.
        """
        edges_to_remove = []
        for (src, tgt), edge in self.graph.edges.items():
            if edge.weight < threshold and edge.prediction_count > 5:
                edges_to_remove.append((src, tgt))

        for src, tgt in edges_to_remove:
            self.graph.remove_edge(src, tgt)

    # ── Save/Load ───────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Save model state."""
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "concept_dim": self.concept_dim,
            "n_concepts": self.n_concepts,
            "token_embed": self.token_embed.weight.data.copy(),
            "subject_proj": self.subject_proj.weight.data.copy(),
            "concept_to_embed": self.concept_to_embed.weight.data.copy(),
            "relation_type_embed": self.relation_type_embed.weight.data.copy(),
            "relation_classifier_weight": self.relation_classifier.weight.data.copy(),
            "relation_classifier_bias": self.relation_classifier.bias.data.copy() if self.relation_classifier.bias is not None else None,
            "graph_nodes": {nid: {
                "vector": n.vector.copy(),
                "core_vector": n.core_vector.copy(),
                "label": n.label,
                "activation": n.activation,
                "confidence": n.confidence,
                "stability": n.stability,
            } for nid, n in self.graph.nodes.items()},
            "graph_edges": {(s, t): {
                "weight": e.weight,
                "confidence": e.confidence,
                "relation_type": e.relation_type,
                "relation_vector": e.relation_vector.copy(),
                "predicate_token_id": e.predicate_token_id,
                "prediction_count": e.prediction_count,
            } for (s, t), e in self.graph.edges.items()},
            "episodic_triples": self._episodic_triples.copy(),
            "step_counter": self._step_counter,
            "train_correct": self._train_correct,
            "train_total": self._train_total,
        }

    def save(self, path: str):
        """Save model to file."""
        state = self.state_dict()
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        # Restore embeddings
        self.token_embed.weight.data = state["token_embed"]
        self.subject_proj.weight.data = state["subject_proj"]
        self.concept_to_embed.weight.data = state["concept_to_embed"]
        self.relation_type_embed.weight.data = state["relation_type_embed"]
        self.relation_classifier.weight.data = state["relation_classifier_weight"]
        if state["relation_classifier_bias"] is not None:
            self.relation_classifier.bias.data = state["relation_classifier_bias"]

        # Restore graph
        self.graph.nodes.clear()
        self.graph.edges.clear()
        self.graph._outgoing.clear()
        self.graph._incoming.clear()

        for nid, ndata in state["graph_nodes"].items():
            node = ConceptNode(nid, ndata["vector"], ndata["label"])
            node.core_vector = ndata["core_vector"]
            node.activation = ndata["activation"]
            node.confidence = ndata["confidence"]
            node.stability = ndata["stability"]
            self.graph.nodes[nid] = node
            self.graph.next_id = max(self.graph.next_id, nid + 1)

        for (s, t), edata in state["graph_edges"].items():
            edge = ConceptEdge(s, t, weight=edata["weight"],
                             relation_type=edata["relation_type"],
                             relation_dim=self.concept_dim)
            edge.confidence = edata["confidence"]
            edge.relation_vector = edata["relation_vector"]
            edge.predicate_token_id = edata["predicate_token_id"]
            edge.prediction_count = edata["prediction_count"]
            self.graph.edges[(s, t)] = edge
            self.graph._outgoing[s].append((t, edge))
            self.graph._incoming[t].append((s, edge))

        # Restore state
        self._episodic_triples = state.get("episodic_triples", [])
        self._step_counter = state.get("step_counter", 0)
        self._train_correct = state.get("train_correct", 0)
        self._train_total = state.get("train_total", 0)
