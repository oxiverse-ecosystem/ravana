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
        "contributes", "contribute", "associated", "linked",
        "correlates", "correlate", "worsens", "worsen",
        "improves", "improve", "increases", "increase",
        "decreases", "decrease", "reduces", "reduce",
        "enhances", "enhance", "diminishes", "diminish",
        "prevents", "prevent", "inhibits", "inhibit",
        "strengthens", "strengthen", "weakens", "weaken",
        "restores", "restore", "provides", "provide",
        "protects", "protect", "corrupts", "corrupt",
        "damages", "damage", "harms", "harm", "heals", "heal",
        "cures", "cure", "fights", "fight", "blocks", "block",
        "accelerates", "accelerate", "slows", "slow",
        # Compound predicates (single tokens after wordpiece)
        "contributes_to", "associated_with", "linked_to",
        "may_cause", "can_cause", "leads_to", "results_in",
        "correlated_with", "is_a", "is_type_of", "type_of",
        "consists_of", "composed_of", "made_of",
        "capable_of", "able_to", "same_as", "equivalent_to",
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
        self.predictive_coding_enabled = True  # flag for A/B testing
        self._gate_concept_creation = gate_concept_creation

        # ── Token Embeddings ──
        self.token_embed = Embedding(vocab_size, embed_dim)

        # ── Projection Layers (embed_dim ↔ concept_dim) ──
        # Project token embeddings into concept space
        self.subject_proj = Linear(embed_dim, concept_dim, bias=False)
        # Project concept vectors back to token embedding space for scoring
        self.concept_to_embed = Linear(concept_dim, embed_dim, bias=False)

        # ── Relation Type Classifier (local Hebbian learning, no backprop) ──
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

        # ── Norm cache (invalidated per forward, avoids 30+ redundant linalg.norm) ──
        self._norm_cache: Dict[str, float] = {}
        self._token_embed_norms: Optional[np.ndarray] = None  # pre-computed once per forward

        # ── Concept creation gating ──
        self._concept_similarity_threshold = 0.7 if gate_concept_creation else 0.0
        # 2x headroom: tokenizer discovers words lazily during training, so the
        # first N tokens fill the graph to capacity and later tokens get bound
        # to the nearest existing concept (many-to-one contamination).
        self._max_concepts = max(n_concepts * 2, vocab_size, 100)

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

        # ── Forward pass caches (invalidated when graph changes) ──
        self._node_matrix_cache = None   # (node_ids, matrix, norms)
        self._node_matrix_version = -1   # graph version counter
        self._rel_vector_cache = {}      # rel_type_name → avg relation vector
        self._rel_vector_version = -1

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

    def _cached_norm(self, vec: np.ndarray, cache_key: str) -> float:
        """Cached linalg.norm — avoids redundant computation within a forward pass."""
        val = self._norm_cache.get(cache_key)
        if val is None:
            val = float(np.linalg.norm(vec))
            self._norm_cache[cache_key] = val
        return val

    def _get_node_matrix(self):
        """Get cached (node_ids, vector_matrix, norms) for batch operations.

        Uses a version counter to avoid rebuilding when the graph hasn't changed.
        Returns (node_ids: List[int], matrix: np.ndarray[N,D], norms: np.ndarray[N]).
        """
        graph_version = (len(self.graph.nodes), len(self.graph.edges))
        if self._node_matrix_version != graph_version or self._node_matrix_cache is None:
            node_ids = sorted(self.graph.nodes.keys())
            if not node_ids:
                self._node_matrix_cache = ([], np.empty((0, self.concept_dim)), np.empty(0))
            else:
                matrix = np.stack([self.graph.nodes[cid].vector for cid in node_ids]).astype(np.float32)
                norms = np.linalg.norm(matrix, axis=1)
                self._node_matrix_cache = (node_ids, matrix, norms)
            self._node_matrix_version = graph_version
        return self._node_matrix_cache

    def _invalidate_caches(self):
        """Invalidate forward-pass caches after graph modification."""
        self._node_matrix_version = -1
        self._rel_vector_version = -1
        self._rel_vector_cache.clear()

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

        # Use keyword classifier directly — learned relation type embeddings
        # are never trained (learn() updates edge relation_vectors, not
        # relation_type_embed.weight). With 300x hard boost, keyword-only
        # is deterministic and avoids random noise corrupting relation types.
        # Tested: 100% keyword gives 62.5% cross_domain_causal vs 37.5% blend.
        return keyword_idx, self.relation_type_embed.weight.data[keyword_idx]

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
            cid = bindings[0].concept_id
            # Validate that the node still exists in the graph
            # (binding may be stale if node was pruned)
            if self.graph.get_node(cid) is not None:
                return cid
            # Stale binding — fall through to create a fresh concept

        # Always create a new concept for this token (no merging)
        concept_vec = self._project_to_concept(embed_vec)
        if len(self.graph.nodes) < self._max_concepts:
            node = self.graph.add_node(vector=concept_vec, label=self._decode_token(token_id))
            nid = node.id
        else:
            # At capacity — find nearest existing concept to reuse
            # (better than a phantom ID that doesn't exist in the graph)
            nid, sim = self._nearest_concept(embed_vec)
            if nid < 0:
                # No concepts at all — force create
                node = self.graph.add_node(vector=concept_vec, label=self._decode_token(token_id))
                nid = node.id

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

        # ── Norm cache: clear per forward, pre-compute token embed norms ──
        self._norm_cache.clear()
        if self._token_embed_norms is None:
            self._token_embed_norms = np.linalg.norm(self.token_embed.weight.data, axis=1)

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

        # Identify relation concept node (intermediary, not a prediction target)
        relation_cid = -1
        if relation_ids:
            relation_tid = relation_ids[0]
            # Look up existing concept without creating one
            existing = self.binding_map.get_tokens(relation_tid, min_confidence=0.0)
            if existing:
                relation_cid = existing[0].concept_id

        # ── Vector Arithmetic (word2vec-style analogy) — VECTORIZED ──
        # Compute expected output: subject_embed + relation_vector
        analogy_targets = {}
        # Cache avg relation vectors per type (expensive to recompute every forward)
        rel_version = (len(self.graph.edges), rel_type_name)
        if self._rel_vector_version != rel_version:
            rvs = []
            for (src, tgt), edge in self.graph.edges.items():
                if edge.relation_type == rel_type_name and hasattr(edge, 'relation_vector') and edge.relation_vector is not None:
                    rvs.append(edge.relation_vector)
            self._rel_vector_cache[rel_type_name] = np.mean(rvs, axis=0) if rvs else None
            self._rel_vector_version = rel_version

        avg_rv = self._rel_vector_cache.get(rel_type_name)
        if avg_rv is not None:
            expected = subject_embed + avg_rv
            cv = self._project_to_concept(expected)
            cv_norm = np.linalg.norm(cv)
            if cv_norm > 0:
                # Batch cosine similarity against all concept nodes
                node_ids, node_matrix, node_norms = self._get_node_matrix()
                if len(node_ids) > 0:
                    min_len = min(len(cv), node_matrix.shape[1])
                    sims = (node_matrix[:, :min_len] @ cv[:min_len]) / (node_norms * cv_norm + 1e-15)
                    # Mask subject
                    for i, cid in enumerate(node_ids):
                        if cid == subject_cid:
                            sims[i] = -1.0
                            break
                    mask = sims > 0.3
                    for i in np.where(mask)[0]:
                        analogy_targets[node_ids[i]] = float(sims[i]) * 2.0

        # ── Spreading Activation ──
        # Reset all activations
        for node in self.graph.nodes.values():
            node.activation = 0.0

        # Activate subject concept
        self.graph.activate(subject_cid, amount=1.0)

        # Phase 0: Similarity-based priming (VECTORIZED — single matmul)
        # "Tiger is cat-like" — activate concepts with similar embeddings.
        subject_node = self.graph.get_node(subject_cid)
        if subject_node is None:
            from ..tensor import tensor as make_tensor
            return make_tensor(np.zeros(self.vocab_size, dtype=np.float32))
        subject_vec = subject_node.vector
        sv_norm = np.linalg.norm(subject_vec)
        if sv_norm > 0:
            node_ids, node_matrix, node_norms = self._get_node_matrix()
            if len(node_ids) > 0:
                # Batch cosine similarity: one matmul instead of N Python loops
                sims = (node_matrix @ subject_vec) / (node_norms * sv_norm + 1e-15)
                # Find subject index and mask it out
                subject_idx = -1
                for i, cid in enumerate(node_ids):
                    if cid == subject_cid:
                        subject_idx = i
                        break
                if subject_idx >= 0:
                    sims[subject_idx] = -1.0
                # Activate nodes above threshold
                mask = sims > 0.3
                for i in np.where(mask)[0]:
                    self.graph.activate(node_ids[i], amount=float(sims[i]) * 0.3)

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

        # Phase 2b: Direct-edge boost from subject
        # Spreading activation decays subject's activation (1.0 → ~0.8) before it
        # reaches direct targets, while similarity-primed nodes accumulate activation
        # from multiple paths. This phase gives the subject's own direct outgoing
        # edges (matching relation type) a strong boost so they compete with
        # indirect high-activation nodes. Without this, "heat causes expansion"
        # produces expansion at 0.14 while "intense" (similarity-primed) hits 1.0.
        # Phase 2b: Direct-edge boost from subject
        subject_node_final = self.graph.get_node(subject_cid)
        if subject_node_final is not None and subject_node_final.activation > 0.01:
            for tgt_id, edge in self.graph.get_outgoing(subject_cid):
                if edge.relation_type == rel_type_name and tgt_id != subject_cid:
                    tgt_node = self.graph.get_node(tgt_id)
                    if tgt_node is not None:
                        boost = subject_node_final.activation * 2.0 * edge.weight * edge.confidence
                        self.graph.activate(tgt_id, amount=boost)

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
                    rv_norm = self._cached_norm(rv, f'rv_{nid}_{tgt_id}')
                    node_norm = self._cached_norm(node.vector, f'node_{nid}')
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
            if nid == relation_cid:
                continue  # Don't predict relation node (it's an intermediary)
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
        # ── Batch concept-to-token scoring (replaces serial loop) ──
        # Collect targets for batch cosine similarity computation
        batch_targets = []  # (tgt_cid, score, concept_vec)
        for tgt_cid, score in matching_targets.items():
            tgt_node = self.graph.get_node(tgt_cid)
            if tgt_node is None:
                continue

            # Method 1: Check binding map for bound tokens (immediate)
            # Boosted from 1.0 to 2.0 — binding map is exact match, should dominate
            bindings = self.binding_map.get_tokens(tgt_cid, min_confidence=0.1)
            for binding in bindings:
                tok_id = binding.token_id
                if 0 <= tok_id < self.vocab_size:
                    concept_scores[tok_id] += score * binding.confidence * 2.0

            batch_targets.append((tgt_cid, score, tgt_node.vector))

        # Method 2: Batch cosine similarity — one matmul instead of N loops
        if batch_targets:
            tgt_vecs = np.stack([tv[2] for tv in batch_targets])  # (n_targets, concept_dim)
            # Project to embed_dim (same as _project_to_embed but batched)
            if self.concept_dim == self.embed_dim:
                tgt_embeds = tgt_vecs  # identity — no projection needed
            else:
                tgt_embeds = self.concept_to_embed(tgt_vecs).data  # (n_targets, embed_dim)
            tgt_norms = np.linalg.norm(tgt_embeds, axis=1)  # (n_targets,)
            valid_tgt = tgt_norms > 0

            token_embeds = self.token_embed.weight.data  # (vocab_size, embed_dim)
            token_norms = self._token_embed_norms  # pre-computed at forward start
            if token_norms is None:
                token_norms = np.linalg.norm(token_embeds, axis=1)
                self._token_embed_norms = token_norms
            valid_tok = token_norms > 0

            # Similarity matrix: (n_targets, vocab_size) — one big matmul
            sim_matrix = np.zeros((len(batch_targets), self.vocab_size), dtype=np.float32)
            if np.any(valid_tgt) and np.any(valid_tok):
                normed_tgt = tgt_embeds.copy()
                normed_tgt[valid_tgt] /= tgt_norms[valid_tgt, np.newaxis]
                normed_tok = token_embeds.copy()
                normed_tok[valid_tok] /= token_norms[valid_tok, np.newaxis]
                sim_matrix = normed_tgt @ normed_tok.T  # (n_targets, vocab_size)

            # Accumulate weighted similarities
            for i, (tgt_cid, score, _) in enumerate(batch_targets):
                concept_scores += sim_matrix[i] * score * 0.3

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

        # ── Create relation concept node ──
        # The relation becomes a SHARED NODE in the graph, not just edge metadata.
        # "cat has tail" and "dog has tail" both route through the same "has" node.
        # This gives the graph real topology: cat→has→tail, dog→has→tail.
        if relation_ids:
            relation_tid = relation_ids[0]  # first relation token
            relation_embed = self.token_embed.weight.data[relation_tid]
            relation_cid = self._get_or_create_concept(relation_tid, relation_embed)
        else:
            relation_cid = object_cid  # fallback: direct edge if no relation

        # ── Create relation-object concept node ──
        # Instead of a global "has" hub (which gets overloaded with 10+ facts),
        # create relation-OBJECT nodes: "has:tail", "has:wing", etc.
        # Multiple subjects sharing the same object (cat, dog → tail) route
        # through the same hub, enabling WITHIN-group transfer.
        # Cross-group transfer (tiger is cat-like → tail) uses embedding similarity.
        # Synthetic token ID to avoid collision with real tokens.
        if relation_ids:
            rel_obj_tid = 10000 + relation_ids[0] * 256 + object_tid
            rel_obj_embed = 0.5 * (relation_embed + object_embed)  # blend
            rel_obj_cid = self._get_or_create_concept(rel_obj_tid, rel_obj_embed)
        else:
            rel_obj_cid = object_cid

        # ── Create/update edges: direct + via relation-object hub ──
        # Use get_edge to avoid resetting learned weights via add_edge.
        edge_direct = self.graph.get_edge(subject_cid, object_cid)
        if edge_direct is None:
            edge_direct = self.graph.add_edge(
                source=subject_cid, target=object_cid,
                weight=0.3, relation_type=rel_type_name,
            )
        edge_sr = self.graph.get_edge(subject_cid, rel_obj_cid)
        if edge_sr is None:
            edge_sr = self.graph.add_edge(
                source=subject_cid, target=rel_obj_cid,
                weight=0.3, relation_type=rel_type_name,
            )
        edge_ro = self.graph.get_edge(rel_obj_cid, object_cid)
        if edge_ro is None:
            edge_ro = self.graph.add_edge(
                source=rel_obj_cid, target=object_cid,
                weight=0.3, relation_type=rel_type_name,
            )

        # Store predicate token
        if relation_ids:
            edge_direct.predicate_token_id = relation_ids[0]
            edge_sr.predicate_token_id = relation_ids[0]
            edge_ro.predicate_token_id = relation_ids[0]

        # ── Hebbian edge updates on ALL edges ──
        # Pure Hebbian: co-activation strengthens connections.
        # During supervised training, we activate the correct pathway
        # (subject + relation-object hub + object via teaching signal).

        src_node = self.graph.get_node(subject_cid)
        rel_obj_node = self.graph.get_node(rel_obj_cid)
        tgt_node = self.graph.get_node(object_cid)

        if src_node is not None and rel_obj_node is not None and tgt_node is not None:
            # Teaching signals: activate hub and object nodes
            rel_obj_node.activation = max(rel_obj_node.activation, 0.7)
            tgt_node.activation = max(tgt_node.activation, 0.8)

            # Direct edge: subject → object (memorization path)
            delta = self._base_lr * src_node.activation * tgt_node.activation
            edge_direct.weight = max(0.0, min(1.0, edge_direct.weight + delta))
            edge_direct.confidence = edge_direct.weight
            edge_direct.stability = min(1.0, edge_direct.stability + 0.01)
            edge_direct.prediction_count += 1
            if is_correct:
                edge_direct.forward_pred_count += 1

            # Edge 1: subject → rel_obj hub (transfer path)
            delta = self._base_lr * src_node.activation * rel_obj_node.activation
            edge_sr.weight = max(0.0, min(1.0, edge_sr.weight + delta))
            edge_sr.confidence = edge_sr.weight
            edge_sr.stability = min(1.0, edge_sr.stability + 0.01)
            edge_sr.prediction_count += 1
            if is_correct:
                edge_sr.forward_pred_count += 1

            # Edge 2: rel_obj hub → object (transfer path)
            delta = self._base_lr * rel_obj_node.activation * tgt_node.activation
            edge_ro.weight = max(0.0, min(1.0, edge_ro.weight + delta))
            edge_ro.confidence = edge_ro.weight
            edge_ro.stability = min(1.0, edge_ro.stability + 0.01)
            edge_ro.prediction_count += 1
            if is_correct:
                edge_ro.forward_pred_count += 1

            # Relation vector updates
            tgt_vec = tgt_node.vector
            tgt_norm = np.linalg.norm(tgt_vec)
            if tgt_norm > 0:
                tgt_signal = tgt_vec / tgt_norm
                type_seed = ConceptEdge._init_relation_vector(rel_type_name, len(edge_ro.relation_vector))
                for e in [edge_sr, edge_ro]:
                    e.relation_vector = (
                        0.70 * e.relation_vector +
                        0.20 * tgt_signal[:len(e.relation_vector)] +
                        0.10 * type_seed
                    )
                    rv_norm = np.linalg.norm(e.relation_vector)
                    if rv_norm > 0:
                        e.relation_vector /= rv_norm
                    e._rv_norm_cache = None

            # ── Concept vector updates ──
            # Increased from 0.005 to 0.02 — concept vectors must converge fast
            # enough that concept-to-token scoring produces correct answers
            pull_lr = 0.02

            # Subject concept → subject token embedding
            subject_concept_vec = self._project_to_concept(subject_embed)
            src_delta = pull_lr * (subject_concept_vec - src_node.vector)
            src_delta = np.clip(src_delta, -self.graph.max_step_delta, self.graph.max_step_delta)
            src_node.vector += src_delta
            src_norm = np.linalg.norm(src_node.vector)
            if src_norm > 0:
                src_node.vector /= src_norm

            # Relation-object concept → blended embedding
            if relation_ids:
                rel_obj_concept_vec = self._project_to_concept(rel_obj_embed)
                rel_obj_delta = pull_lr * (rel_obj_concept_vec - rel_obj_node.vector)
                rel_obj_delta = np.clip(rel_obj_delta, -self.graph.max_step_delta, self.graph.max_step_delta)
                rel_obj_node.vector += rel_obj_delta
                rel_obj_norm = np.linalg.norm(rel_obj_node.vector)
                if rel_obj_norm > 0:
                    rel_obj_node.vector /= rel_obj_norm

            # Object concept → object embedding
            object_concept_vec = self._project_to_concept(object_embed)
            tgt_delta = pull_lr * (object_concept_vec - tgt_node.vector)
            tgt_delta = np.clip(tgt_delta, -self.graph.max_step_delta, self.graph.max_step_delta)
            tgt_node.vector += tgt_delta
            tgt_norm = np.linalg.norm(tgt_node.vector)
            if tgt_norm > 0:
                tgt_node.vector /= tgt_norm

        # ── Update relation classifier weights (local Hebbian) ──
        self._update_relation_classifier(relation_ids, rel_type_idx)

        # ── Train token embeddings (DISABLED — corrupts semantic structure) ──
        # Pulling token embeddings together during training destroys the
        # carefully structured embedding space. "heat" gets pulled toward
        # "expansion", "cold", "melts" simultaneously — ending up in a
        # meaningless average position. Concept vectors + projection layers
        # handle the learning instead.
        # if is_correct:
        #     embed_lr = 0.002
        #     subj_emb = self.token_embed.weight.data[subject_tid]
        #     obj_emb = self.token_embed.weight.data[object_tid]
        #     delta = embed_lr * (obj_emb - subj_emb)
        #     self.token_embed.weight.data[subject_tid] += delta
        #     self.token_embed.weight.data[object_tid] -= delta * 0.3
        #     self._token_embed_norms = None

        # ── Predictive Coding: update RELEVANT edges only ──
        # Only edges touching subject, relation, or object nodes learn from this triple.
        # Updating ALL edges causes catastrophic interference — training "heat causes
        # expansion" would also modify edges for kindness, trust, rain, etc.
        if self.predictive_coding_enabled:
            pc_lr = self._base_lr * 0.15
            obj_embed_norm = np.linalg.norm(object_embed)
            obj_signal = object_embed / obj_embed_norm if obj_embed_norm > 0 else None
            # Collect relevant node IDs for this triple
            relevant_cids = {subject_cid, object_cid}
            if relation_ids:
                relevant_cids.add(relation_cid)
                relevant_cids.add(rel_obj_cid)
            for (src_id, tgt_id), edge in list(self.graph.edges.items()):
                # Only update edges that touch relevant nodes
                if src_id not in relevant_cids and tgt_id not in relevant_cids:
                    continue
                src_node = self.graph.nodes.get(src_id)
                tgt_node = self.graph.nodes.get(tgt_id)
                if src_node is None or tgt_node is None:
                    continue
                # Prediction: edge predicts target activation from source
                predicted = src_node.activation * edge.weight
                # Ground truth: actual target activation
                actual = tgt_node.activation
                # For the correct object: ensure strong signal
                if tgt_id == object_cid:
                    actual = max(actual, 0.5)
                # For the subject: ensure it's recognized as active
                if src_id == subject_cid:
                    actual_src = max(src_node.activation, 0.3)
                    predicted = actual_src * edge.weight
                # Local prediction error
                error = actual - predicted
                # Weight update (local Hebbian, scaled by source activation)
                w_delta = pc_lr * abs(error) * max(src_node.activation, 0.01)
                if error > 0:
                    edge.weight = min(1.0, edge.weight + w_delta * 0.3)
                else:
                    edge.weight = max(0.0, edge.weight - w_delta * 0.1)
                # Confidence: edges that predict correctly gain confidence
                if abs(error) < 0.3:
                    edge.confidence = min(1.0, edge.confidence + 0.005)
                # Relation vector: pull toward correct object for edges touching object
                if tgt_id == object_cid and obj_signal is not None:
                    edge.relation_vector += pc_lr * 0.3 * obj_signal[:len(edge.relation_vector)]
                    rv_n = np.linalg.norm(edge.relation_vector)
                    if rv_n > 0:
                        edge.relation_vector /= rv_n
                    edge._rv_norm_cache = None

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

    def learn_fast(self, token_ids: np.ndarray, target_ids: np.ndarray) -> Dict[str, float]:
        """Fast learn() for hard-example boosting — skips forward() call.

        Use this during hard-boost training when we already know the prediction
        is wrong. Saves ~0.7-1.0ms per call by skipping the forward pass.
        Also skips the O(E) predictive coding loop (uses only local Hebbian).
        """
        # Flatten inputs
        if token_ids.ndim > 1:
            token_ids = token_ids.flatten()
        if target_ids.ndim > 1:
            target_ids = target_ids.flatten()

        input_ids = token_ids.tolist()
        target_id = int(target_ids.flatten()[0])
        is_correct = False  # We know it's wrong (hard example)

        # ── Decompose triple ──
        full_triple_ids = input_ids + [target_id]
        subject_ids, relation_ids, object_ids = self.decompose_triple(full_triple_ids)

        if not subject_ids:
            self._step_counter += 1
            return {"loss": -1.0, "accuracy": self._train_correct / max(1, self._train_total)}

        subject_tid = subject_ids[0]
        subject_embed = self.token_embed.weight.data[subject_tid]
        object_embed = self.token_embed.weight.data[target_id]

        subject_cid = self._get_or_create_concept(subject_tid, subject_embed)
        object_cid = self._get_or_create_concept(target_id, object_embed)

        # ── Classify relation type ──
        rel_type_idx, rel_type_embed = self._classify_relation_learned(relation_ids)
        rel_type_name = RELATION_TYPES[rel_type_idx]

        # ── Create relation concept node ──
        if relation_ids:
            relation_tid = relation_ids[0]
            relation_embed = self.token_embed.weight.data[relation_tid]
            relation_cid = self._get_or_create_concept(relation_tid, relation_embed)
        else:
            relation_cid = object_cid

        # ── Create relation-object concept node ──
        if relation_ids:
            rel_obj_tid = 10000 + relation_ids[0] * 256 + target_id
            rel_obj_embed = 0.5 * (relation_embed + object_embed)
            rel_obj_cid = self._get_or_create_concept(rel_obj_tid, rel_obj_embed)
        else:
            rel_obj_cid = object_cid

        # ── Create/update edges: direct + via relation-object hub ──
        edge_direct = self.graph.get_edge(subject_cid, object_cid)
        if edge_direct is None:
            edge_direct = self.graph.add_edge(
                source=subject_cid, target=object_cid,
                weight=0.3, relation_type=rel_type_name,
            )
        edge_sr = self.graph.get_edge(subject_cid, rel_obj_cid)
        if edge_sr is None:
            edge_sr = self.graph.add_edge(
                source=subject_cid, target=rel_obj_cid,
                weight=0.3, relation_type=rel_type_name,
            )
        edge_ro = self.graph.get_edge(rel_obj_cid, object_cid)
        if edge_ro is None:
            edge_ro = self.graph.add_edge(
                source=rel_obj_cid, target=object_cid,
                weight=0.3, relation_type=rel_type_name,
            )

        if relation_ids:
            edge_direct.predicate_token_id = relation_ids[0]
            edge_sr.predicate_token_id = relation_ids[0]
            edge_ro.predicate_token_id = relation_ids[0]

        # ── Hebbian edge updates (local only — no forward pass needed) ──
        src_node = self.graph.get_node(subject_cid)
        rel_obj_node = self.graph.get_node(rel_obj_cid)
        tgt_node = self.graph.get_node(object_cid)

        if src_node is not None and rel_obj_node is not None and tgt_node is not None:
            rel_obj_node.activation = max(rel_obj_node.activation, 0.7)
            tgt_node.activation = max(tgt_node.activation, 0.8)

            # Direct edge: subject → object
            delta = self._base_lr * src_node.activation * tgt_node.activation
            edge_direct.weight = max(0.0, min(1.0, edge_direct.weight + delta))
            edge_direct.confidence = edge_direct.weight
            edge_direct.stability = min(1.0, edge_direct.stability + 0.01)
            edge_direct.prediction_count += 1

            # Edge 1: subject → rel_obj hub
            delta = self._base_lr * src_node.activation * rel_obj_node.activation
            edge_sr.weight = max(0.0, min(1.0, edge_sr.weight + delta))
            edge_sr.confidence = edge_sr.weight
            edge_sr.stability = min(1.0, edge_sr.stability + 0.01)
            edge_sr.prediction_count += 1

            # Edge 2: rel_obj hub → object
            delta = self._base_lr * rel_obj_node.activation * tgt_node.activation
            edge_ro.weight = max(0.0, min(1.0, edge_ro.weight + delta))
            edge_ro.confidence = edge_ro.weight
            edge_ro.stability = min(1.0, edge_ro.stability + 0.01)
            edge_ro.prediction_count += 1

            # Relation vector updates
            tgt_vec = tgt_node.vector
            tgt_norm = np.linalg.norm(tgt_vec)
            if tgt_norm > 0:
                tgt_signal = tgt_vec / tgt_norm
                type_seed = ConceptEdge._init_relation_vector(rel_type_name, len(edge_ro.relation_vector))
                for e in [edge_sr, edge_ro]:
                    e.relation_vector = (
                        0.70 * e.relation_vector +
                        0.20 * tgt_signal[:len(e.relation_vector)] +
                        0.10 * type_seed
                    )
                    rv_norm = np.linalg.norm(e.relation_vector)
                    if rv_norm > 0:
                        e.relation_vector /= rv_norm
                    e._rv_norm_cache = None

            # Concept vector updates
            pull_lr = 0.02
            subject_concept_vec = self._project_to_concept(subject_embed)
            src_delta = pull_lr * (subject_concept_vec - src_node.vector)
            src_delta = np.clip(src_delta, -self.graph.max_step_delta, self.graph.max_step_delta)
            src_node.vector += src_delta
            src_norm = np.linalg.norm(src_node.vector)
            if src_norm > 0:
                src_node.vector /= src_norm

            if relation_ids:
                rel_obj_concept_vec = self._project_to_concept(rel_obj_embed)
                rel_obj_delta = pull_lr * (rel_obj_concept_vec - rel_obj_node.vector)
                rel_obj_delta = np.clip(rel_obj_delta, -self.graph.max_step_delta, self.graph.max_step_delta)
                rel_obj_node.vector += rel_obj_delta
                rel_obj_norm = np.linalg.norm(rel_obj_node.vector)
                if rel_obj_norm > 0:
                    rel_obj_node.vector /= rel_obj_norm

            object_concept_vec = self._project_to_concept(object_embed)
            tgt_delta = pull_lr * (object_concept_vec - tgt_node.vector)
            tgt_delta = np.clip(tgt_delta, -self.graph.max_step_delta, self.graph.max_step_delta)
            tgt_node.vector += tgt_delta
            tgt_n = np.linalg.norm(tgt_node.vector)
            if tgt_n > 0:
                tgt_node.vector /= tgt_n

        # ── Update relation classifier ──
        self._update_relation_classifier(relation_ids, rel_type_idx)

        # ── Predictive coding: update relevant edges only ──
        if self.predictive_coding_enabled:
            pc_lr = self._base_lr * 0.15
            obj_embed_norm = np.linalg.norm(object_embed)
            obj_signal = object_embed / obj_embed_norm if obj_embed_norm > 0 else None
            relevant_cids = {subject_cid, object_cid}
            if relation_ids:
                relevant_cids.add(relation_cid)
                relevant_cids.add(rel_obj_cid)
            for (src_id, tgt_id), edge in list(self.graph.edges.items()):
                if src_id not in relevant_cids and tgt_id not in relevant_cids:
                    continue
                src_n = self.graph.nodes.get(src_id)
                tgt_n = self.graph.nodes.get(tgt_id)
                if src_n is None or tgt_n is None:
                    continue
                predicted = src_n.activation * edge.weight
                actual = tgt_n.activation
                if tgt_id == object_cid:
                    actual = max(actual, 0.5)
                if src_id == subject_cid:
                    actual_src = max(src_n.activation, 0.3)
                    predicted = actual_src * edge.weight
                error = actual - predicted
                w_delta = pc_lr * abs(error) * max(src_n.activation, 0.01)
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

        # ── Episodic + sleep pressure + auto-sleep ──
        self._episodic_triples.append((subject_cid, rel_type_idx, object_cid, time.time()))
        if len(self._episodic_triples) > self._max_episodic:
            self._episodic_triples = self._episodic_triples[-self._max_episodic:]
        self._sleep_pressure += 0.015

        self._step_counter += 1

        # Auto-sleep check
        if (self._sleep_pressure > 0.7 and
                self._step_counter - getattr(self, '_last_sleep_step', 0) > 200):
            self.sleep_cycle()
            self._last_sleep_step = self._step_counter
        if self.sleep_interval > 0 and self._step_counter % self.sleep_interval == 0:
            self.sleep_cycle()

        return {"loss": 0.0, "accuracy": self._train_correct / max(1, self._train_total)}

    def _update_relation_classifier(self, relation_token_ids: List[int], true_type_idx: int):
        """Update relation classifier via local Hebbian learning (no backprop).

        Each weight update uses only LOCAL information:
        - Local prediction error (softmax output vs target)
        - Local input (relation embedding)
        No chain rule, no gradient propagation through other layers.
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

        # Local prediction error: target is one-hot at true_type_idx
        error = np.zeros(len(RELATION_TYPES), dtype=np.float32)
        error[true_type_idx] = 1.0 - probs[true_type_idx]
        for i in range(len(RELATION_TYPES)):
            if i != true_type_idx:
                error[i] = -probs[i]

        # Local Hebbian update: ΔW = lr × error ⊗ input
        # This is Hebbian (pre × post × error), NOT backprop.
        # No chain rule — uses only the local error signal and local input.
        lr = self._classifier_lr
        input_2d = rel_concept.reshape(1, -1)
        error_2d = error.reshape(-1, 1)
        delta_w = lr * (error_2d @ input_2d)
        self.relation_classifier.weight.data += delta_w
        if self.relation_classifier.bias is not None:
            self.relation_classifier.bias.data += lr * error

        # Update relation type embeddings: pull the correct type toward rel_concept
        # This is already Hebbian (local: pre * post)
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
        # Budget increased from 3.0 to 5.0 — less aggressive, preserves learned edges
        self._normalize_outgoing_weights(budget=5.0)

        # ── Prune weak edges ──
        # Only prune edges that have been seen many times AND are still weak.
        # Higher threshold for prediction_count prevents killing new cross-domain bridges.
        self._prune_weak_edges(threshold=0.03)

        # ── Drift defense ──
        # Pull concept vectors back toward their core vectors if they've drifted too far.
        # Threshold increased from 0.4 to 0.7 — allow more movement before correction.
        # Pull strength reduced from 0.1 to 0.05 — gentler correction.
        for nid, node in self.graph.nodes.items():
            drift = node.drift_magnitude
            if drift > 0.7:
                # Pull back toward core vector (gentler)
                pull = 0.05 * (node.core_vector - node.vector)
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

    def _prune_weak_edges(self, threshold: float = 0.03):
        """Remove edges with weight below threshold.

        Keeps the graph clean and prevents accumulation of noise edges.
        Only prunes edges that have been seen enough times (prediction_count > 15)
        to give them a fair chance to learn.
        """
        edges_to_remove = []
        for (src, tgt), edge in self.graph.edges.items():
            if edge.weight < threshold and edge.prediction_count > 15:
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
            "binding_map": {
                "by_token": {tid: [(b.concept_id, b.confidence, b.source) for b in blist]
                             for tid, blist in self.binding_map._by_token.items()},
                "by_concept": {cid: [(b.token_id, b.confidence, b.source) for b in blist]
                               for cid, blist in self.binding_map._by_concept.items()},
            },
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

        # Restore binding map
        self.binding_map = ConceptBindingMap()
        bm_data = state.get("binding_map", {})
        for tid, bindings in bm_data.get("by_token", {}).items():
            for cid, conf, src in bindings:
                self.binding_map.bind(int(tid), int(cid), confidence=conf, source=src)
