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
from typing import Optional, List, Tuple, Dict, Set, Any
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
                 anchor_relation_vectors: bool = True,
                 latent_dim: int = 32,
                 hidden_dim: int = 48):
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

        # ── Learned Relation Predictor MLP (For compatibility/backup) ──
        rp_in = concept_dim * 2
        rp_dim = concept_dim
        self._rp_W1 = np.random.randn(rp_dim, rp_in).astype(np.float32) * np.sqrt(2.0 / rp_in)
        self._rp_b1 = np.zeros(rp_dim, dtype=np.float32)
        self._rp_W2 = np.random.randn(vocab_size, rp_dim).astype(np.float32) * np.sqrt(2.0 / rp_dim)
        self._rp_b2 = np.zeros(vocab_size, dtype=np.float32)

        self._rp_mW1 = np.zeros_like(self._rp_W1)
        self._rp_mb1 = np.zeros_like(self._rp_b1)
        self._rp_mW2 = np.zeros_like(self._rp_W2)
        self._rp_mb2 = np.zeros_like(self._rp_b2)
        self._rp_lr = 0.01
        self._rp_encoder_lr = 0.0001  # Separate, much lower LR for encoder updates during RP training
        self._rp_momentum = 0.9
        self._rp_cache = None
        self.use_rp_for_analogy = True

        # ── Domain-Agnostic Bilinear Relation Predictor ──
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder weights
        scale_enc1 = np.sqrt(2.0 / embed_dim)
        scale_enc2 = np.sqrt(2.0 / self.hidden_dim)
        self._enc_W1 = np.random.randn(self.hidden_dim, embed_dim).astype(np.float32) * scale_enc1
        self._enc_b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self._enc_W2 = np.random.randn(self.latent_dim, self.hidden_dim).astype(np.float32) * scale_enc2
        self._enc_b2 = np.zeros(self.latent_dim, dtype=np.float32)
        
        self._enc_mW1 = np.zeros_like(self._enc_W1)
        self._enc_mb1 = np.zeros_like(self._enc_b1)
        self._enc_mW2 = np.zeros_like(self._enc_W2)
        self._enc_mb2 = np.zeros_like(self._enc_b2)
        
        # Decoder weights (for autoencoder pre-training only)
        scale_dec1 = np.sqrt(2.0 / self.latent_dim)
        scale_dec2 = np.sqrt(2.0 / self.hidden_dim)
        self._dec_W1 = np.random.randn(self.hidden_dim, self.latent_dim).astype(np.float32) * scale_dec1
        self._dec_b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self._dec_W2 = np.random.randn(embed_dim, self.hidden_dim).astype(np.float32) * scale_dec2
        self._dec_b2 = np.zeros(embed_dim, dtype=np.float32)
        
        # Relation matrices (one for each relation type in RELATION_TYPES)
        n_rel_types = len(RELATION_TYPES)
        self._rp_rel_matrices = np.random.randn(n_rel_types, self.latent_dim, self.latent_dim).astype(np.float32) * np.sqrt(2.0 / self.latent_dim)
        self._rp_mrel_matrices = np.zeros_like(self._rp_rel_matrices)
        self.freeze_encoder = True
        self.rp_scale = 16.0
        
        # ── Contrastive Regularization Attributes ──
        self.use_contrastive_reg = False
        self.lambda_contrastive = 0.5
        self.semantic_pairs = []
        self.neg_sample_size = 5

        # ── Graph-Aware Encoder Alignment Attributes ──
        self.alignment_margin = 0.15
        self.lambda_anchor = 0.05
        self.max_alignment_epochs = 10
        self.alignment_lr = 0.005
        self.alignment_edge_threshold = 0.25
        self.alignment_needed = False  # flag: encoder changed, needs re-alignment
        self.wake_epochs_since_sleep = 0  # count wake epochs for fixed-cadence sleep
        self.sleep_every_n_wake_epochs = 3  # sleep every N wake epochs (default 3)


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

    @property
    def _tokenizer(self):
        return getattr(self, '_tokenizer_val', None)

    @_tokenizer.setter
    def _tokenizer(self, val):
        self._tokenizer_val = val
        if val is not None:
            # 1. Initialize token embeddings using LearnedEmbedder to capture morphological similarity
            self._initialize_token_embeddings_from_tokenizer()
            # 2. Pre-train the encoder as an autoencoder over the whole vocabulary (Domain A + B)
            self._pretrain_encoder_autoencoder(epochs=300, lr=0.01)

    def _initialize_token_embeddings_from_tokenizer(self):
        """Initialize token embeddings using LearnedEmbedder to capture character n-gram structure."""
        from ..embedder import LearnedEmbedder
        embedder = LearnedEmbedder(dim=self.embed_dim)
        
        tokenizer = self._tokenizer_val
        # Determine vocabulary mapping
        mapping = {}
        if hasattr(tokenizer, 'word_to_id') and tokenizer.word_to_id:
            mapping = tokenizer.word_to_id
        elif hasattr(tokenizer, 'char_to_id') and tokenizer.char_to_id:
            mapping = tokenizer.char_to_id
        else:
            # Fallback if no mapping is found: use decoded tokens
            mapping = {self._decode_token(tid): tid for tid in range(self.vocab_size)}
            
        # Fit IDF if words are available
        words = [w for w in mapping.keys() if isinstance(w, str)]
        if hasattr(embedder, 'fit') and words:
            try:
                embedder.fit(words)
            except Exception:
                pass
                
        for key, tid in mapping.items():
            if isinstance(tid, int) and 0 <= tid < self.vocab_size:
                word_str = str(key)
                vec = embedder.encode(word_str)
                self.token_embed.weight.data[tid] = vec
                
        # Clear/invalidate cached norms
        self._token_embed_norms = None

    def _pretrain_encoder_autoencoder(self, epochs=300, lr=0.01):
        """Pre-train the encoder as an autoencoder over all vocabulary tokens."""
        X = self.token_embed.weight.data  # (vocab_size, embed_dim)
        
        # Momentum buffers for autoencoder weights
        dec_mW1 = np.zeros_like(self._dec_W1)
        dec_mb1 = np.zeros_like(self._dec_b1)
        dec_mW2 = np.zeros_like(self._dec_W2)
        dec_mb2 = np.zeros_like(self._dec_b2)
        
        for epoch in range(epochs):
            # Forward pass: Encoder
            z1 = X @ self._enc_W1.T + self._enc_b1      # (V, hidden_dim)
            h1 = np.tanh(z1)                           # (V, hidden_dim)
            z2 = h1 @ self._enc_W2.T + self._enc_b2    # (V, latent_dim)
            latent = np.tanh(z2)                       # (V, latent_dim)
            
            # Forward pass: Decoder
            dec_z1 = latent @ self._dec_W1.T + self._dec_b1  # (V, hidden_dim)
            dec_h1 = np.tanh(dec_z1)                        # (V, hidden_dim)
            dec_z2 = dec_h1 @ self._dec_W2.T + self._dec_b2  # (V, embed_dim)
            recon = dec_z2                                  # (V, embed_dim)
            
            # Loss: Mean Squared Error
            loss = np.mean((recon - X) ** 2)
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"  [Autoencoder] Epoch {epoch:3d} Loss: {loss:.6f}")
            
            # Backward pass: Decoder
            d_recon = 2.0 * (recon - X) / len(X)       # (V, embed_dim)
            
            d_dec_W2 = d_recon.T @ dec_h1              # (embed_dim, hidden_dim)
            d_dec_b2 = np.sum(d_recon, axis=0)         # (embed_dim,)
            
            d_dec_h1 = d_recon @ self._dec_W2          # (V, hidden_dim)
            d_dec_z1 = d_dec_h1 * (1.0 - dec_h1 * dec_h1) # (V, hidden_dim)
            
            d_dec_W1 = d_dec_z1.T @ latent             # (hidden_dim, latent_dim)
            d_dec_b1 = np.sum(d_dec_z1, axis=0)        # (hidden_dim,)
            
            d_latent = d_dec_z1 @ self._dec_W1         # (V, latent_dim)
            
            # Backward pass: Encoder
            d_z2 = d_latent * (1.0 - latent * latent)  # (V, latent_dim)
            d_enc_W2 = d_z2.T @ h1                     # (latent_dim, hidden_dim)
            d_enc_b2 = np.sum(d_z2, axis=0)            # (latent_dim,)
            
            d_h1 = d_z2 @ self._enc_W2                 # (V, hidden_dim)
            d_z1 = d_h1 * (1.0 - h1 * h1)              # (V, hidden_dim)
            d_enc_W1 = d_z1.T @ X                      # (hidden_dim, embed_dim)
            d_enc_b1 = np.sum(d_z1, axis=0)            # (hidden_dim,)
            
            # Update Decoder
            dec_mW2 = self._rp_momentum * dec_mW2 - lr * d_dec_W2
            dec_mb2 = self._rp_momentum * dec_mb2 - lr * d_dec_b2
            dec_mW1 = self._rp_momentum * dec_mW1 - lr * d_dec_W1
            dec_mb1 = self._rp_momentum * dec_mb1 - lr * d_dec_b1
            
            self._dec_W2 += dec_mW2
            self._dec_b2 += dec_mb2
            self._dec_W1 += dec_mW1
            self._dec_b1 += dec_mb1
            
            # Update Encoder
            self._enc_mW2 = self._rp_momentum * self._enc_mW2 - lr * d_enc_W2
            self._enc_mb2 = self._rp_momentum * self._enc_mb2 - lr * d_enc_b2
            self._enc_mW1 = self._rp_momentum * self._enc_mW1 - lr * d_enc_W1
            self._enc_mb1 = self._rp_momentum * self._enc_mb1 - lr * d_enc_b1
            
            self._enc_W2 += self._enc_mW2
            self._enc_b2 += self._enc_mb2
            self._enc_W1 += self._enc_mW1
            self._enc_b1 += self._enc_mb1
            # Encoder changed - need re-alignment on next sleep
            self.mark_alignment_needed()

    def _encoder_forward_full(self, X):
        """Pass inputs through the encoder, returning all activations for backpropagation.
        X: (B, embed_dim) or (embed_dim,)
        """
        is_flat = X.ndim == 1
        if is_flat:
            X_batch = X[np.newaxis, :]
        else:
            X_batch = X
        z1 = X_batch @ self._enc_W1.T + self._enc_b1       # (B, hidden_dim)
        h1 = np.tanh(z1)                            # (B, hidden_dim)
        z2 = h1 @ self._enc_W2.T + self._enc_b2     # (B, latent_dim)
        latent = np.tanh(z2)                        # (B, latent_dim)
        if is_flat:
            return latent[0], z1[0], h1[0], z2[0]
        return latent, z1, h1, z2

    def _encoder_backward(self, X, z1, h1, z2, h2, d_h2):
        """Compute encoder parameter gradients.
        X: (B, embed_dim)
        z1, h1: (B, hidden_dim)
        z2, h2: (B, latent_dim)
        d_h2: (B, latent_dim)
        """
        d_z2 = d_h2 * (1.0 - h2 * h2)                # (B, latent_dim)
        d_W2 = d_z2.T @ h1                           # (latent_dim, hidden_dim)
        d_b2 = np.sum(d_z2, axis=0)                  # (latent_dim,)
        
        d_h1 = d_z2 @ self._enc_W2                   # (B, hidden_dim)
        d_z1 = d_h1 * (1.0 - h1 * h1)                # (B, hidden_dim)
        d_W1 = d_z1.T @ X                            # (hidden_dim, embed_dim)
        d_b1 = np.sum(d_z1, axis=0)                  # (hidden_dim,)
        
        return d_W1, d_b1, d_W2, d_b2

    def _rp_forward(self, subject_tid, rel_type_idx):
        """Relation predictor forward pass using domain-agnostic latent space."""
        subject_tid = int(subject_tid)
        rel_type_idx = int(rel_type_idx)

        # Get source and target embeddings
        source_embed = self.token_embed.weight.data[subject_tid] # (embed_dim,)
        target_embeds = self.token_embed.weight.data             # (vocab_size, embed_dim)
        
        # Pass through encoder
        source_latent, src_z1, src_h1, src_z2 = self._encoder_forward_full(source_embed) # (latent_dim,)
        target_latents, tgt_z1, tgt_h1, tgt_z2 = self._encoder_forward_full(target_embeds) # (vocab_size, latent_dim)
        
        # Get relation matrix
        rel_matrix = self._rp_rel_matrices[rel_type_idx] # (latent_dim, latent_dim)
        
        # Compute bilinear product
        proj_latent = source_latent @ rel_matrix
        logits = (proj_latent @ target_latents.T) * getattr(self, "rp_scale", 16.0) # (vocab_size,)
        
        # Cache for backprop
        self._rp_cache = (
            subject_tid, rel_type_idx,
            source_embed, target_embeds,
            source_latent, src_z1, src_h1, src_z2,
            target_latents, tgt_z1, tgt_h1, tgt_z2,
            rel_matrix, logits
        )
        return logits

    def _compute_contrastive_gradients(self):
        """Compute contrastive loss gradients w.r.t. encoder parameters."""
        d_con_W1 = np.zeros_like(self._enc_W1)
        d_con_b1 = np.zeros_like(self._enc_b1)
        d_con_W2 = np.zeros_like(self._enc_W2)
        d_con_b2 = np.zeros_like(self._enc_b2)
        
        pairs = getattr(self, "semantic_pairs", [])
        if not pairs:
            return d_con_W1, d_con_b1, d_con_W2, d_con_b2, 0.0
            
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None:
            return d_con_W1, d_con_b1, d_con_W2, d_con_b2, 0.0
            
        # Draw negative samples
        vocab_words = list(tokenizer.word_to_id.keys())
        neg_size = getattr(self, "neg_sample_size", 5)
        
        total_loss = 0.0
        n_pairs_processed = 0
        
        for word_a, word_b in pairs:
            tid_a = tokenizer.word_to_id.get(word_a)
            tid_b = tokenizer.word_to_id.get(word_b)
            if tid_a is None or tid_b is None:
                continue
                
            embed_a = self.token_embed.weight.data[tid_a]
            embed_b = self.token_embed.weight.data[tid_b]
            
            lat_a, z1_a, h1_a, z2_a = self._encoder_forward_full(embed_a)
            lat_b, z1_b, h1_b, z2_b = self._encoder_forward_full(embed_b)
            
            norm_a = np.linalg.norm(lat_a)
            norm_b = np.linalg.norm(lat_b)
            
            unit_a = lat_a / (norm_a + 1e-15)
            unit_b = lat_b / (norm_b + 1e-15)
            
            s = np.dot(unit_a, unit_b)
            sig_s = 1.0 / (1.0 + np.exp(-s) + 1e-15)
            
            # Positive loss: -log(sigmoid(s))
            total_loss -= np.log(sig_s + 1e-15)
            n_pairs_processed += 1
            
            # Gradients of positive loss w.r.t lat_a and lat_b
            d_s_d_lat_a = (unit_b - s * unit_a) / (norm_a + 1e-15)
            d_s_d_lat_b = (unit_a - s * unit_b) / (norm_b + 1e-15)
            
            d_lat_a = (sig_s - 1.0) * d_s_d_lat_a
            d_lat_b = (sig_s - 1.0) * d_s_d_lat_b
            
            # Backprop for word_a and word_b
            dW1_a, db1_a, dW2_a, db2_a = self._encoder_backward(
                embed_a[np.newaxis, :], z1_a[np.newaxis, :], h1_a[np.newaxis, :], z2_a[np.newaxis, :],
                lat_a[np.newaxis, :], d_lat_a[np.newaxis, :]
            )
            dW1_b, db1_b, dW2_b, db2_b = self._encoder_backward(
                embed_b[np.newaxis, :], z1_b[np.newaxis, :], h1_b[np.newaxis, :], z2_b[np.newaxis, :],
                lat_b[np.newaxis, :], d_lat_b[np.newaxis, :]
            )
            
            d_con_W1 += dW1_a + dW1_b
            d_con_b1 += db1_a + db1_b
            d_con_W2 += dW2_a + dW2_b
            d_con_b2 += db2_a + db2_b
            
            # Negative sampling
            if vocab_words:
                neg_words = np.random.choice(vocab_words, size=min(neg_size, len(vocab_words)), replace=False)
                for word_neg in neg_words:
                    if word_neg == word_a or word_neg == word_b:
                        continue
                    tid_neg = tokenizer.word_to_id.get(word_neg)
                    if tid_neg is None:
                        continue
                    embed_neg = self.token_embed.weight.data[tid_neg]
                    lat_neg, z1_neg, h1_neg, z2_neg = self._encoder_forward_full(embed_neg)
                    
                    norm_neg = np.linalg.norm(lat_neg)
                    unit_neg = lat_neg / (norm_neg + 1e-15)
                    
                    s_neg = np.dot(unit_a, unit_neg)
                    sig_s_neg = 1.0 / (1.0 + np.exp(-s_neg) + 1e-15)
                    
                    # Negative loss: log(sigmoid(s_neg))
                    total_loss += np.log(sig_s_neg + 1e-15)
                    
                    # Gradients of negative loss w.r.t lat_a and lat_neg
                    d_s_d_lat_a_neg = (unit_neg - s_neg * unit_a) / (norm_a + 1e-15)
                    d_s_d_lat_neg = (unit_a - s_neg * unit_neg) / (norm_neg + 1e-15)
                    
                    d_lat_a_neg = (1.0 - sig_s_neg) * d_s_d_lat_a_neg
                    d_lat_neg = (1.0 - sig_s_neg) * d_s_d_lat_neg
                    
                    # Backprop
                    dW1_a_neg, db1_a_neg, dW2_a_neg, db2_a_neg = self._encoder_backward(
                        embed_a[np.newaxis, :], z1_a[np.newaxis, :], h1_a[np.newaxis, :], z2_a[np.newaxis, :],
                        lat_a[np.newaxis, :], d_lat_a_neg[np.newaxis, :]
                    )
                    dW1_neg, db1_neg, dW2_neg, db2_neg = self._encoder_backward(
                        embed_neg[np.newaxis, :], z1_neg[np.newaxis, :], h1_neg[np.newaxis, :], z2_neg[np.newaxis, :],
                        lat_neg[np.newaxis, :], d_lat_neg[np.newaxis, :]
                    )
                    
                    d_con_W1 += dW1_a_neg + dW1_neg
                    d_con_b1 += db1_a_neg + db1_neg
                    d_con_W2 += dW2_a_neg + dW2_neg
                    d_con_b2 += db2_a_neg + db2_neg
                    
        if n_pairs_processed > 0:
            scale = 1.0 / n_pairs_processed
            d_con_W1 *= scale
            d_con_b1 *= scale
            d_con_W2 *= scale
            d_con_b2 *= scale
            total_loss *= scale
            
        return d_con_W1, d_con_b1, d_con_W2, d_con_b2, total_loss

    def _rp_backward(self, target_id, lr_scale=1.0):
        """Relation predictor backward pass for domain-agnostic bilinear relation embedding."""
        if self._rp_cache is None:
            return
            
        (
            subject_tid, rel_type_idx,
            source_embed, target_embeds,
            source_latent, src_z1, src_h1, src_z2,
            target_latents, tgt_z1, tgt_h1, tgt_z2,
            rel_matrix, logits
        ) = self._rp_cache
        
        # Softmax loss gradient
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)
        d_logits = probs.copy()
        if 0 <= target_id < len(d_logits):
            d_logits[target_id] -= 1.0
        d_logits *= getattr(self, "rp_scale", 16.0)
            
        # Let p = rel_matrix.T @ source_latent (shape (latent_dim,))
        p = rel_matrix.T @ source_latent # (latent_dim,)
        
        # Gradient with respect to target_latents (shape (vocab_size, latent_dim))
        d_target_latents = np.outer(d_logits, p)
        
        # Gradient with respect to p
        d_p = target_latents.T @ d_logits # (latent_dim,)
        
        # Gradient with respect to rel_matrix
        d_rel_matrix = np.outer(source_latent, d_p) # (latent_dim, latent_dim)
        
        # Gradient with respect to source_latent
        d_source_latent = rel_matrix @ d_p # (latent_dim,)
        
        # Backprop through encoder for source embedding path
        d_W1_src, d_b1_src, d_W2_src, d_b2_src = self._encoder_backward(
            source_embed[np.newaxis, :],
            src_z1[np.newaxis, :],
            src_h1[np.newaxis, :],
            src_z2[np.newaxis, :],
            source_latent[np.newaxis, :],
            d_source_latent[np.newaxis, :]
        )
        
        # Backprop through encoder for target embeddings path
        d_W1_tgt, d_b1_tgt, d_W2_tgt, d_b2_tgt = self._encoder_backward(
            target_embeds,
            tgt_z1,
            tgt_h1,
            tgt_z2,
            target_latents,
            d_target_latents
        )
        
        # Sum encoder gradients
        d_enc_W1 = d_W1_src + d_W1_tgt
        d_enc_b1 = d_b1_src + d_b1_tgt
        d_enc_W2 = d_W2_src + d_W2_tgt
        d_enc_b2 = d_b2_src + d_b2_tgt
        
        # Add contrastive loss regularizer gradients
        if not getattr(self, "freeze_encoder", False) and getattr(self, "use_contrastive_reg", False):
            d_con_W1, d_con_b1, d_con_W2, d_con_b2, con_loss = self._compute_contrastive_gradients()
            lambda_c = getattr(self, "lambda_contrastive", 0.5)
            d_enc_W1 += lambda_c * d_con_W1
            d_enc_b1 += lambda_c * d_con_b1
            d_enc_W2 += lambda_c * d_con_W2
            d_enc_b2 += lambda_c * d_con_b2
        
        # Update weights with momentum
        lr = self._rp_lr * lr_scale
        encoder_lr = self._rp_encoder_lr * lr_scale

        # Update encoder
        if not getattr(self, "freeze_encoder", False):
            self._enc_mW1 = self._rp_momentum * self._enc_mW1 - encoder_lr * d_enc_W1
            self._enc_mb1 = self._rp_momentum * self._enc_mb1 - encoder_lr * d_enc_b1
            self._enc_mW2 = self._rp_momentum * self._enc_mW2 - encoder_lr * d_enc_W2
            self._enc_mb2 = self._rp_momentum * self._enc_mb2 - encoder_lr * d_enc_b2
            
            self._enc_W1 += self._enc_mW1
            self._enc_b1 += self._enc_mb1
            self._enc_W2 += self._enc_mW2
            self._enc_b2 += self._enc_mb2
            # Encoder changed - need re-alignment on next sleep
            self.mark_alignment_needed()
        
        # Update relation matrix
        self._rp_mrel_matrices[rel_type_idx] = (
            self._rp_momentum * self._rp_mrel_matrices[rel_type_idx] - lr * d_rel_matrix
        )
        self._rp_rel_matrices[rel_type_idx] += self._rp_mrel_matrices[rel_type_idx]
        self._rp_cache = None


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

        # Decompose input prompt (which has no object at the end)
        if len(token_ids) >= 1:
            subject_ids = [token_ids[0]]
            relation_ids = token_ids[1:]
        else:
            subject_ids = []
            relation_ids = []

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
            existing = self.binding_map.get_concepts(relation_tid, min_confidence=0.0)
            if existing:
                relation_cid = existing[0].concept_id

        # ── Vector Arithmetic or MLP Analogy (cross-domain transfer) ──
        analogy_targets = {}
        subject_node = self.graph.get_node(subject_cid)
        if subject_node is None:
            return make_tensor(np.zeros(self.vocab_size, dtype=np.float32))

        self._rp_probs_cache = None
        if self.use_rp_for_analogy:
            # Run the learned relation predictor
            rp_logits = self._rp_forward(subject_tid, rel_type_idx)
            exp_logits = np.exp(rp_logits - np.max(rp_logits))
            probs = exp_logits / (np.sum(exp_logits) + 1e-10)
            self._rp_probs_cache = probs
            
            # Map top tokens to analogy targets
            top_tok_indices = np.argsort(probs)[::-1][:10]
            for tok_id in top_tok_indices:
                if tok_id == subject_tid:
                    continue
                prob = float(probs[tok_id])
                if prob < 0.01:
                    continue
                bindings = self.binding_map.get_concepts(tok_id, min_confidence=0.1)
                for b in bindings:
                    cid = b.concept_id
                    score = prob * 8.0  # Boosted weight
                    if cid in analogy_targets:
                        analogy_targets[cid] = max(analogy_targets[cid], score)
                    else:
                        analogy_targets[cid] = score
        else:
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
        if getattr(self, '_rp_probs_cache', None) is not None:
            gated_rp = np.zeros_like(self._rp_probs_cache)
            for tok_id in range(self.vocab_size):
                prob = self._rp_probs_cache[tok_id]
                bindings = self.binding_map.get_concepts(tok_id, min_confidence=0.0)
                if bindings:
                    max_act = 0.0
                    for b in bindings:
                        node = self.graph.get_node(b.concept_id)
                        if node is not None:
                            max_act = max(max_act, node.activation)
                    gated_rp[tok_id] = prob * (0.15 + max_act)
                else:
                    gated_rp[tok_id] = prob * 0.8
            concept_scores += gated_rp * 35.0


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

        # ── N-Hop BFS traversal (compositionality) ──
        # "fire causes ?" → fire→heat (any type), heat→expansion (causal) → boost expansion
        # "exercise produces crashes" → exercise→stress→bugs→crashes (3 hops)
        # First hop: follow ALL edges (any type) to reach domain intermediaries.
        # Subsequent hops: only follow edges matching the query relation type.
        from collections import deque
        _max_hops = 3
        _hop_base_boost = 8.0
        _hop_decay = 0.6  # per-hop score decay
        # BFS queue: (node_id, cumulative_score, depth, visited_set)
        bfs_queue = deque()
        for mid_cid, mid_edge in self.graph.get_outgoing(subject_cid):
            if mid_cid == subject_cid:
                continue
            mid_node = self.graph.get_node(mid_cid)
            if mid_node is None:
                continue
            hop_score = mid_edge.weight * mid_edge.confidence
            bfs_queue.append((mid_cid, hop_score, 1, {subject_cid}))

        while bfs_queue:
            nid, cum_score, depth, visited = bfs_queue.popleft()
            if depth > _max_hops:
                continue
            for tgt_cid, tgt_edge in self.graph.get_outgoing(nid):
                if tgt_cid in visited or tgt_cid == subject_cid:
                    continue
                # After first hop (depth>=1), only follow edges matching relation type
                if depth >= 1 and tgt_edge.relation_type != rel_type_name:
                    continue
                edge_score = cum_score * tgt_edge.weight * tgt_edge.confidence
                # Apply boost with per-hop decay
                final_score = edge_score * _hop_base_boost * (_hop_decay ** (depth - 1))
                if tgt_cid in matching_targets:
                    matching_targets[tgt_cid] = max(matching_targets[tgt_cid], final_score)
                else:
                    matching_targets[tgt_cid] = final_score
                # Continue BFS if we haven't reached max depth
                if depth < _max_hops:
                    new_visited = visited | {nid}
                    bfs_queue.append((tgt_cid, edge_score, depth + 1, new_visited))

        # Also include directly active concepts (for same-type queries)
        # Hub suppression: penalize low-activation high-in-degree noise nodes
        for nid, node in active_nodes:
            if nid == subject_cid:
                continue  # Don't predict subject itself
            if nid == relation_cid:
                continue  # Don't predict relation node (it's an intermediary)
            # Hub suppression: penalize low-activation nodes with high in-degree
            in_deg = len(self.graph._incoming.get(nid, []))
            if node.activation < 0.5 and in_deg > 5:
                hub_factor = 0.3
            elif node.activation < 0.3 and in_deg > 3:
                hub_factor = 0.5
            else:
                hub_factor = 1.0
            act_score = node.activation * 3.0 * hub_factor
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

        # Reverse edges: weak bidirectional inference
        # If "anger → heat" exists, create weak "heat → anger" so cross-domain
        # queries like "heat causes ?" can traverse back through anger.
        reverse_direct = self.graph.get_edge(object_cid, subject_cid)
        if reverse_direct is None:
            reverse_direct = self.graph.add_edge(
                source=object_cid, target=subject_cid,
                weight=0.1, relation_type=rel_type_name,
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

            # Path-aware concept vector update: pull subject and object vectors
            # closer together. This builds analogical structure — if anger→heat
            # and heat→expansion, anger and expansion become closer in concept
            # space, enabling cross-domain transfer via embedding similarity.
            path_lr = 0.005  # gentle to avoid destroying existing structure
            src_to_tgt = tgt_node.vector - src_node.vector
            path_delta_src = path_lr * src_to_tgt
            path_delta_tgt = -path_lr * src_to_tgt * 0.3  # asymmetric: move src more
            path_delta_src = np.clip(path_delta_src, -self.graph.max_step_delta, self.graph.max_step_delta)
            path_delta_tgt = np.clip(path_delta_tgt, -self.graph.max_step_delta, self.graph.max_step_delta)
            src_node.vector += path_delta_src
            tgt_node.vector += path_delta_tgt
            # Re-normalize
            src_norm = np.linalg.norm(src_node.vector)
            if src_norm > 0:
                src_node.vector /= src_norm
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

        # ── Train learned relation predictor MLP ──
        self._rp_forward(subject_tid, rel_type_idx)
        self._rp_backward(target_id)

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

        # Reverse edges: weak bidirectional inference
        reverse_direct = self.graph.get_edge(object_cid, subject_cid)
        if reverse_direct is None:
            reverse_direct = self.graph.add_edge(
                source=object_cid, target=subject_cid,
                weight=0.1, relation_type=rel_type_name,
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

    def sleep_cycle(self, validation_queries: Optional[List[Dict[str, Any]]] = None,
                    force_alignment: bool = False):
        """Sleep cycle: consolidate triples, prune weak edges, replay important memories.

        This is the brain's "offline consolidation" — during sleep, the model:
        1. Replays important episodic triples (hippocampal replay)
        2. Consolidates edge weights (homeostatic downscaling)
        3. Prunes weak/unstable edges
        4. Dynamically aligns the encoder to the graph topology (if needed)
        5. Updates concept vectors (drift defense)
        6. Prunes phantom nodes (unbound concepts)

        Args:
            validation_queries: Held-out queries for validation during alignment
            force_alignment: If True, run alignment even if encoder hasn't changed
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
        # Prune edges with weight below 0.1 to clear out spurious semantic edges.
        self._prune_weak_edges(threshold=0.1)

        # ── Prune phantom nodes ──
        # Remove concept nodes that have no token binding (token_id == None)
        # and degree < 2 (isolated or single-edge artifacts from tokenizer expansion)
        self._prune_phantom_nodes(min_degree=2)

        # ── Representation Alignment ──
        # Align encoder representations to graph topology only if encoder changed
        # or forced (e.g., initial consolidation, manual call)
        if self.alignment_needed or force_alignment:
            self.align_encoder_to_graph(validation_queries=validation_queries)
            self.alignment_needed = False  # reset flag after alignment

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
        self.wake_epochs_since_sleep = 0

        # ── Reset activations ──
        for node in self.graph.nodes.values():
            node.activation = 0.0
            node.fatigue = 0.0

    def _prune_phantom_nodes(self, min_degree: int = 2):
        """Remove concept nodes without token bindings that are structural artifacts.

        Phantom nodes (token_id == None) created during tokenizer vocabulary expansion
        but never bound to actual tokens act as noise hubs during multi-seed traversal.
        Only remove if degree < min_degree to preserve legitimate hub concepts.
        """
        nodes_to_remove = []
        for nid, node in self.graph.nodes.items():
            bindings = self.binding_map.get_tokens(nid, min_confidence=0.0)
            has_token_binding = len(bindings) > 0
            if not has_token_binding:
                # Count edges
                out_degree = len(self.graph._outgoing.get(nid, []))
                in_degree = len(self.graph._incoming.get(nid, []))
                total_degree = out_degree + in_degree
                if total_degree < min_degree:
                    nodes_to_remove.append(nid)

        for nid in nodes_to_remove:
            # Remove all edges connected to this node
            for _, edge in self.graph._outgoing.get(nid, []):
                self.graph.remove_edge(nid, edge.target)
            for edge, _ in self.graph._incoming.get(nid, []):
                self.graph.remove_edge(edge.source, nid)
            # Remove the node
            if nid in self.graph.nodes:
                del self.graph.nodes[nid]
            if nid in self.graph._outgoing:
                del self.graph._outgoing[nid]
            if nid in self.graph._incoming:
                del self.graph._incoming[nid]

        if nodes_to_remove:
            self._invalidate_caches()
            print(f"[Sleep] Pruned {len(nodes_to_remove)} phantom nodes (degree < {min_degree})")

    def end_wake_epoch(self, validation_queries: Optional[List[Dict[str, Any]]] = None):
        """Call at the end of each wake epoch to track sleep cadence.

        Increments wake_epochs_since_sleep and triggers sleep if
        sleep_every_n_wake_epochs threshold reached.
        """
        self.wake_epochs_since_sleep += 1
        if self.wake_epochs_since_sleep >= self.sleep_every_n_wake_epochs:
            self.sleep_cycle(validation_queries=validation_queries)

    def mark_alignment_needed(self):
        """Mark that encoder has changed and alignment is needed on next sleep."""
        self.alignment_needed = True

    def align_encoder_to_graph(self, validation_queries: Optional[List[Dict[str, Any]]] = None):
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

    def compute_neighbor_recall_at_5(self) -> float:
        """Calculate the fraction of strong topological neighbors in the top-5 candidate seeds."""
        tok = self._tokenizer
        if tok is None or not self.graph.nodes:
            return 0.0
            
        recalls = []
        node_ids = list(self.graph.nodes.keys())
        if not node_ids:
            return 0.0
            
        # Pre-compute latent representations
        latents = {}
        for nid in node_ids:
            tokens = self.binding_map.get_tokens(nid, 0.0)
            if tokens:
                tid = tokens[0].token_id
                if tid < self.token_embed.weight.data.shape[0]:
                    emb = self.token_embed.weight.data[tid]
                    lat, *_ = self._encoder_forward_full(emb)
                    latents[nid] = lat

        for u in node_ids:
            strong_neighbors = [
                v for v, edge in self.graph.get_outgoing(u)
                if edge.weight >= self.alignment_edge_threshold
            ]
            if not strong_neighbors:
                continue
                
            if u not in latents:
                continue
            lat_u = latents[u]
            
            scores = []
            for v in node_ids:
                if v == u or v not in latents:
                    continue
                lat_v = latents[v]
                na = np.linalg.norm(lat_u)
                nb = np.linalg.norm(lat_v)
                sim = np.dot(lat_u, lat_v) / (na * nb + 1e-15) if na > 0 and nb > 0 else 0.0
                scores.append((v, sim))
                
            scores.sort(key=lambda x: x[1], reverse=True)
            top_5_cids = {item[0] for item in scores[:5]}
            
            hits = sum(1 for v in strong_neighbors if v in top_5_cids)
            recalls.append(hits / len(strong_neighbors))
            
        return float(np.mean(recalls)) if recalls else 1.0

    def align_encoder_to_graph(self, validation_queries: Optional[List[Dict[str, Any]]] = None):
        """Perform offline graph-aware contrastive alignment to sync embeddings with graph topology.
        
        Implements Bridge Alignment: combines graph-topology pairs with cross-domain semantic pairs
        and validation query analogy pairs to provide training signal for hard/out-of-distribution cases.
        """
        tok = self._tokenizer
        if tok is None or not self.graph.nodes:
            return
           
        # Save initial checkpoint
        checkpoint_W1 = self._enc_W1.copy()
        checkpoint_b1 = self._enc_b1.copy()
        checkpoint_W2 = self._enc_W2.copy()
        checkpoint_b2 = self._enc_b2.copy()
        
        # Track peak performance starting from initial checkpoint
        peak_encoder_state = (checkpoint_W1.copy(), checkpoint_b1.copy(), checkpoint_W2.copy(), checkpoint_b2.copy())
        
        # Calculate pre-alignment validation accuracy
        pre_acc = 0.0
        if validation_queries:
            successes = 0
            for tc in validation_queries:
                q = tc["query"]
                expected = tc["expected"]
                res, _ = self.retrieval_v2_multi_seed(q, k_neighbors=5, gate_mode="margin_multi")
                rank = next((i+1 for i, x in enumerate(res) if x[0] == expected), 99)
                if rank <= 10:
                    successes += 1
            pre_acc = successes / len(validation_queries)
        peak_validation_acc = pre_acc
        
        # ── Pair Extraction ──
        # 1. Graph topology pairs: strong edges (weight >= threshold) across all relation types
        positive_pairs = []
        seen_pairs = set()  # for deduplication
        
        for u in self.graph.nodes:
            tokens_u = self.binding_map.get_tokens(u, 0.0)
            if not tokens_u:
                continue
            tid_u = tokens_u[0].token_id
            if tid_u >= self.token_embed.weight.data.shape[0]:
                continue
            word_u = tok.decode([tid_u])
            
            for v, edge in self.graph.get_outgoing(u):
                if edge.weight >= self.alignment_edge_threshold:
                    tokens_v = self.binding_map.get_tokens(v, 0.0)
                    if not tokens_v:
                        continue
                    tid_v = tokens_v[0].token_id
                    if tid_v >= self.token_embed.weight.data.shape[0]:
                        continue
                    word_v = tok.decode([tid_v])
                    pair_key = (word_u, word_v)
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        positive_pairs.append((word_u, word_v, u, v))
        
        # 2. Bridge Alignment: cross-domain semantic/analogy pairs from self.semantic_pairs
        for word_a, word_b in getattr(self, "semantic_pairs", []):
            tid_a = tok.word_to_id.get(word_a)
            tid_b = tok.word_to_id.get(word_b)
            if tid_a is None or tid_b is None:
                continue
            if tid_a >= self.token_embed.weight.data.shape[0] or tid_b >= self.token_embed.weight.data.shape[0]:
                continue
            # Get concept IDs if they exist
            bindings_a = self.binding_map.get_concepts(tid_a, min_confidence=0.1)
            bindings_b = self.binding_map.get_concepts(tid_b, min_confidence=0.1)
            cid_a = bindings_a[0].concept_id if bindings_a else None
            cid_b = bindings_b[0].concept_id if bindings_b else None
            pair_key = (word_a, word_b)
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                positive_pairs.append((word_a, word_b, cid_a, cid_b))
        
        # 3. Bridge Alignment: validation query analogy pairs (query_word -> expected_seed)
        # These are USED ONLY for validation/early stopping, NOT for training.
        # This prevents overfitting to specific validation queries.
        validation_pairs = []
        if validation_queries:
            for tc in validation_queries:
                q = tc["query"]
                expected_seed = tc.get("expected_seed")
                if not expected_seed:
                    continue
                query_word = q.split()[0] if q else None
                if not query_word:
                    continue
                tid_a = tok.word_to_id.get(query_word)
                tid_b = tok.word_to_id.get(expected_seed)
                if tid_a is None or tid_b is None:
                    continue
                if tid_a >= self.token_embed.weight.data.shape[0] or tid_b >= self.token_embed.weight.data.shape[0]:
                    continue
                bindings_a = self.binding_map.get_concepts(tid_a, min_confidence=0.1)
                bindings_b = self.binding_map.get_concepts(tid_b, min_confidence=0.1)
                cid_a = bindings_a[0].concept_id if bindings_a else None
                cid_b = bindings_b[0].concept_id if bindings_b else None
                pair_key = (query_word, expected_seed)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    validation_pairs.append((query_word, expected_seed, cid_a, cid_b))
                    
        if not positive_pairs:
            print(f"[Debug Alignment] No positive pairs found! Graph has {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")
            if self.graph.edges:
                max_w = max(e.weight for e in self.graph.edges.values())
                print(f"[Debug Alignment] Max edge weight in graph = {max_w:.4f}")
            return
            
        vocab_words = list(tok.word_to_id.keys())
        
        # ── Alignment Loop ──
        lr = getattr(self, "alignment_lr", 0.005)
        margin = getattr(self, "alignment_margin", 0.15)
        lambda_anchor = getattr(self, "lambda_anchor", 0.05)
        max_epochs = getattr(self, "max_alignment_epochs", 10)

        patience_counter = 0

        min_epochs = 5  # minimum epochs before early stopping can trigger

        for epoch in range(1, max_epochs + 1):
            # Accumulators for this epoch
            d_con_W1 = np.zeros_like(self._enc_W1)
            d_con_b1 = np.zeros_like(self._enc_b1)
            d_con_W2 = np.zeros_like(self._enc_W2)
            d_con_b2 = np.zeros_like(self._enc_b2)
            
            total_loss = 0.0
            
            for word_a, word_b, cid_a, cid_b in positive_pairs:
                tid_a = tok.word_to_id.get(word_a)
                tid_b = tok.word_to_id.get(word_b)
                if tid_a is None or tid_b is None:
                    continue
                    
                embed_a = self.token_embed.weight.data[tid_a]
                embed_b = self.token_embed.weight.data[tid_b]
                
                lat_a, z1_a, h1_a, z2_a = self._encoder_forward_full(embed_a)
                lat_b, z1_b, h1_b, z2_b = self._encoder_forward_full(embed_b)
                
                norm_a = np.linalg.norm(lat_a)
                norm_b = np.linalg.norm(lat_b)
                unit_a = lat_a / (norm_a + 1e-15) if norm_a > 0 else lat_a
                unit_b = lat_b / (norm_b + 1e-15) if norm_b > 0 else lat_b
                
                s_p = float(np.dot(unit_a, unit_b))
                
                # Dynamic Stratified Negative Sampling:
                # 3 Random + 2 Hard Negatives
                neg_candidates = []
                scored_all = []
                for word_neg, tid_neg in tok.word_to_id.items():
                    if word_neg == word_a or word_neg == word_b:
                        continue
                    if tid_neg >= self.token_embed.weight.data.shape[0]:
                        continue
                    bindings = self.binding_map.get_concepts(tid_neg, min_confidence=0.1)
                    if not bindings:
                        continue
                    cid_neg = bindings[0].concept_id
                    # Robust negative sampling: skip graph edge check if cid_a is None (e.g., for query words without concept node)
                    if cid_a is not None and self.graph.get_edge(cid_a, cid_neg) is not None:
                        continue
                    embed_neg = self.token_embed.weight.data[tid_neg]
                    lat_neg, *_ = self._encoder_forward_full(embed_neg)
                    norm_neg = np.linalg.norm(lat_neg)
                    unit_neg = lat_neg / (norm_neg + 1e-15) if norm_neg > 0 else lat_neg
                    sim_neg = float(np.dot(unit_a, unit_neg))
                    scored_all.append((word_neg, sim_neg, tid_neg))
                    
                scored_all.sort(key=lambda x: x[1], reverse=True)
                for item in scored_all[:2]:
                    neg_candidates.append(item[2])
                    
                if vocab_words:
                    random_choices = np.random.choice(vocab_words, size=min(10, len(vocab_words)), replace=False)
                    for r_word in random_choices:
                        if len(neg_candidates) >= 5:
                            break
                        if r_word == word_a or r_word == word_b:
                            continue
                        tid_neg = tok.word_to_id[r_word]
                        if tid_neg >= self.token_embed.weight.data.shape[0]:
                            continue
                        bindings = self.binding_map.get_concepts(tid_neg, min_confidence=0.1)
                        if bindings:
                            cid_neg = bindings[0].concept_id
                            # Robust negative sampling: skip graph edge check if cid_a is None
                            if cid_a is not None and self.graph.get_edge(cid_a, cid_neg) is not None:
                                continue
                        if tid_neg not in neg_candidates:
                            neg_candidates.append(tid_neg)
                            
                for tid_neg in neg_candidates:
                    embed_neg = self.token_embed.weight.data[tid_neg]
                    lat_neg, z1_neg, h1_neg, z2_neg = self._encoder_forward_full(embed_neg)
                    norm_neg = np.linalg.norm(lat_neg)
                    unit_neg = lat_neg / (norm_neg + 1e-15) if norm_neg > 0 else lat_neg
                    
                    s_n = float(np.dot(unit_a, unit_neg))
                    loss_val = max(0.0, s_n - s_p + margin)
                    if loss_val <= 0.0:
                        continue
                        
                    total_loss += loss_val
                    
                    d_sp_d_lata = (unit_b - s_p * unit_a) / (norm_a + 1e-15)
                    d_sp_d_latb = (unit_a - s_p * unit_b) / (norm_b + 1e-15)
                    d_sn_d_lata = (unit_neg - s_n * unit_a) / (norm_a + 1e-15)
                    d_sn_d_latneg = (unit_a - s_n * unit_neg) / (norm_neg + 1e-15)
                    
                    d_lat_a = -1.0 * d_sp_d_lata + 1.0 * d_sn_d_lata
                    d_lat_b = -1.0 * d_sp_d_latb
                    d_lat_neg = 1.0 * d_sn_d_latneg
                    
                    dW1_a, db1_a, dW2_a, db2_a = self._encoder_backward(
                        embed_a[np.newaxis, :], z1_a[np.newaxis, :], h1_a[np.newaxis, :], z2_a[np.newaxis, :],
                        lat_a[np.newaxis, :], d_lat_a[np.newaxis, :]
                    )
                    dW1_b, db1_b, dW2_b, db2_b = self._encoder_backward(
                        embed_b[np.newaxis, :], z1_b[np.newaxis, :], h1_b[np.newaxis, :], z2_b[np.newaxis, :],
                        lat_b[np.newaxis, :], d_lat_b[np.newaxis, :]
                    )
                    dW1_neg, db1_neg, dW2_neg, db2_neg = self._encoder_backward(
                        embed_neg[np.newaxis, :], z1_neg[np.newaxis, :], h1_neg[np.newaxis, :], z2_neg[np.newaxis, :],
                        lat_neg[np.newaxis, :], d_lat_neg[np.newaxis, :]
                    )
                    
                    d_con_W1 += dW1_a + dW1_b + dW1_neg
                    d_con_b1 += db1_a + db1_b + db1_neg
                    d_con_W2 += dW2_a + dW2_b + dW2_neg
                    d_con_b2 += db2_a + db2_b + db2_neg
            
            d_anchor_W1 = 2.0 * lambda_anchor * (self._enc_W1 - checkpoint_W1)
            d_anchor_b1 = 2.0 * lambda_anchor * (self._enc_b1 - checkpoint_b1)
            d_anchor_W2 = 2.0 * lambda_anchor * (self._enc_W2 - checkpoint_W2)
            d_anchor_b2 = 2.0 * lambda_anchor * (self._enc_b2 - checkpoint_b2)
            
            d_total_W1 = d_con_W1 + d_anchor_W1
            d_total_b1 = d_con_b1 + d_anchor_b1
            d_total_W2 = d_con_W2 + d_anchor_W2
            d_total_b2 = d_con_b2 + d_anchor_b2
            
            self._enc_mW1 = self._rp_momentum * self._enc_mW1 - lr * d_total_W1
            self._enc_mb1 = self._rp_momentum * self._enc_mb1 - lr * d_total_b1
            self._enc_mW2 = self._rp_momentum * self._enc_mW2 - lr * d_total_W2
            self._enc_mb2 = self._rp_momentum * self._enc_mb2 - lr * d_total_b2
            
            self._enc_W1 += self._enc_mW1
            self._enc_b1 += self._enc_mb1
            self._enc_W2 += self._enc_mW2
            self._enc_b2 += self._enc_mb2
            
            self._token_embed_norms = None
            
            recall_5 = self.compute_neighbor_recall_at_5()

            if validation_queries:
                successes = 0
                for tc in validation_queries:
                    q = tc["query"]
                    expected = tc["expected"]
                    res, _ = self.retrieval_v2_multi_seed(q, k_neighbors=5, gate_mode="margin_multi")
                    rank = next((i+1 for i, x in enumerate(res) if x[0] == expected), 99)
                    if rank <= 10:
                        successes += 1
                validation_acc = successes / len(validation_queries)

                if validation_acc > peak_validation_acc:
                    peak_validation_acc = validation_acc
                    peak_encoder_state = (self._enc_W1.copy(), self._enc_b1.copy(), self._enc_W2.copy(), self._enc_b2.copy())
                    patience_counter = 0  # reset patience on improvement
                else:
                    patience_counter += 1
                    if epoch >= min_epochs and patience_counter >= 3:  # patience of 3 epochs after min_epochs
                        print(f"[Align] Early stopping at epoch {epoch} (patience exhausted, peak_acc={peak_validation_acc:.3f})")
                        break
                    
        if peak_encoder_state is not None:
            self._enc_W1, self._enc_b1, self._enc_W2, self._enc_b2 = peak_encoder_state
        else:
            self._enc_W1 = checkpoint_W1
            self._enc_b1 = checkpoint_b1
            self._enc_W2 = checkpoint_W2
            self._enc_b2 = checkpoint_b2
        self._token_embed_norms = None

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

    def _prune_weak_edges(self, threshold: float = 0.1):
        """Remove edges with weight below threshold.

        Keeps the graph clean and prevents accumulation of noise edges.
        Prunes weak edges regardless of prediction count.
        """
        edges_to_remove = []
        for (src, tgt), edge in self.graph.edges.items():
            if edge.weight < threshold:
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
            "_rp_W1": self._rp_W1.copy(),
            "_rp_b1": self._rp_b1.copy(),
            "_rp_W2": self._rp_W2.copy(),
            "_rp_b2": self._rp_b2.copy(),
            "_rp_mW1": self._rp_mW1.copy(),
            "_rp_mb1": self._rp_mb1.copy(),
            "_rp_mW2": self._rp_mW2.copy(),
            "_rp_mb2": self._rp_mb2.copy(),
            "_enc_W1": self._enc_W1.copy(),
            "_enc_b1": self._enc_b1.copy(),
            "_enc_W2": self._enc_W2.copy(),
            "_enc_b2": self._enc_b2.copy(),
            "_enc_mW1": self._enc_mW1.copy(),
            "_enc_mb1": self._enc_mb1.copy(),
            "_enc_mW2": self._enc_mW2.copy(),
            "_enc_mb2": self._enc_mb2.copy(),
            "_dec_W1": self._dec_W1.copy(),
            "_dec_b1": self._dec_b1.copy(),
            "_dec_W2": self._dec_W2.copy(),
            "_dec_b2": self._dec_b2.copy(),
            "_rp_rel_matrices": self._rp_rel_matrices.copy(),
            "_rp_mrel_matrices": self._rp_mrel_matrices.copy(),
            "binding_map": {
                "by_token": {tid: [(b.concept_id, b.confidence, b.source) for b in blist]
                             for tid, blist in self.binding_map._by_token.items()},
                "by_concept": {cid: [(b.token_id, b.confidence, b.source) for b in blist]
                               for cid, blist in self.binding_map._by_concept.items()},
            },
            "_tokenizer": self._tokenizer if hasattr(self, '_tokenizer') and self._tokenizer is not None else None,
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

        # Restore tokenizer (bypass setter to avoid redundant pre-training)
        self._tokenizer_val = state.get("_tokenizer", None)

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

        # Restore Relation Predictor MLP weights
        if "_rp_W1" in state:
            self._rp_W1 = state["_rp_W1"]
            self._rp_b1 = state["_rp_b1"]
            self._rp_W2 = state["_rp_W2"]
            self._rp_b2 = state["_rp_b2"]
            self._rp_mW1 = state["_rp_mW1"]
            self._rp_mb1 = state["_rp_mb1"]
            self._rp_mW2 = state["_rp_mW2"]
            self._rp_mb2 = state["_rp_mb2"]

        # Restore Domain-Agnostic variables if they exist
        if "_enc_W1" in state:
            self._enc_W1 = state["_enc_W1"]
            self._enc_b1 = state["_enc_b1"]
            self._enc_W2 = state["_enc_W2"]
            self._enc_b2 = state["_enc_b2"]
            self._enc_mW1 = state["_enc_mW1"]
            self._enc_mb1 = state["_enc_mb1"]
            self._enc_mW2 = state["_enc_mW2"]
            self._enc_mb2 = state["_enc_mb2"]
            self._dec_W1 = state["_dec_W1"]
            self._dec_b1 = state["_dec_b1"]
            self._dec_W2 = state["_dec_W2"]
            self._dec_b2 = state["_dec_b2"]
            self._rp_rel_matrices = state["_rp_rel_matrices"]
            self._rp_mrel_matrices = state["_rp_mrel_matrices"]

        # Restore binding map
        self.binding_map = ConceptBindingMap()
        bm_data = state.get("binding_map", {})
        for tid, bindings in bm_data.get("by_token", {}).items():
            for cid, conf, src in bindings:
                self.binding_map.bind(int(tid), int(cid), confidence=conf, source=src)

        # Rebuild raw numpy caches in submodules and clear cached norms
        for mod in self.modules():
            if hasattr(mod, '_rebuild_raw_cache'):
                mod._rebuild_raw_cache()
        self._token_embed_norms = None

    def retrieval_v1(
        self,
        query: str,
        k_neighbors: int = 3,
        similarity_threshold: float = 0.1,
        max_depth: int = 3
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        """Standard top-k hybrid retrieval (version 1)."""
        from typing import Any
        parts = query.split()
        if len(parts) < 2:
            return [], {}
        subj_word, rel_word = parts[0], parts[1]
        
        causal = {"causes", "cause", "leads", "produces", "creates"}
        possessive = {"has", "have", "contains", "includes"}
        rel_type = None
        if rel_word in causal:
            rel_type = "causal"
        elif rel_word in possessive:
            rel_type = "possessive"
            
        tok = self._tokenizer
        subj_tid = tok.word_to_id.get(subj_word)
        if subj_tid is None:
            return [], {}
            
        emb = self.token_embed.weight.data[subj_tid]
        lat_query, *_ = self._encoder_forward_full(emb)
        
        # Gather all candidates in graph with similarities
        scored_neighbors = []
        for word, tid in tok.word_to_id.items():
            if word == subj_word:
                continue
            bindings = self.binding_map.get_concepts(tid, min_confidence=0.1)
            if not bindings:
                continue
            cid = bindings[0].concept_id
            
            w_emb = self.token_embed.weight.data[tid]
            lat_word, *_ = self._encoder_forward_full(w_emb)
            
            # cosine sim
            na = float(np.linalg.norm(lat_query))
            nb = float(np.linalg.norm(lat_word))
            sim = float(np.dot(lat_query, lat_word) / (na * nb + 1e-15)) if na > 0 and nb > 0 else 0.0
            
            if sim > similarity_threshold:
                scored_neighbors.append((cid, sim, word))
                
        scored_neighbors.sort(key=lambda x: x[1], reverse=True)
        seeds = scored_neighbors[:k_neighbors]
        
        # Traversal BFS
        activations = {}
        for cid, sim, _ in seeds:
            activations[cid] = sim
            
        frontier = [cid for cid, _, _ in seeds]
        for depth in range(1, max_depth + 1):
            next_frontier = []
            for nid in frontier:
                act = activations[nid]
                for tgt_id, edge in self.graph.get_outgoing(nid):
                    if rel_type and edge.relation_type != rel_type:
                        continue
                    prop = act * edge.weight
                    if tgt_id not in activations or prop > activations[tgt_id]:
                        activations[tgt_id] = prop
                        next_frontier.append(tgt_id)
            frontier = next_frontier
            
        # Decode results
        seed_words = {n[2] for n in seeds}
        results = []
        for cid, act in activations.items():
            tokens = self.binding_map.get_tokens(cid, 0.0)
            for b in tokens:
                word = tok.decode([b.token_id])
                if word not in seed_words and not word.startswith("?"):
                    results.append((word, act))
                    
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Compute telemetry
        top_seed_sim = seeds[0][1] if seeds else 0.0
        margin = (seeds[0][1] - seeds[1][1]) if len(seeds) > 1 else top_seed_sim
        activated_nodes = len(activations)
        dead_end_nodes = 0
        for cid in activations:
            matching_edges = 0
            for tgt_id, edge in self.graph.get_outgoing(cid):
                if rel_type and edge.relation_type != rel_type:
                    continue
                matching_edges += 1
            if matching_edges == 0:
                dead_end_nodes += 1
                
        metrics = {
            "seed_count": len(seeds),
            "top_seed_similarity": top_seed_sim,
            "margin": margin,
            "activated_nodes": activated_nodes,
            "dead_end_nodes": dead_end_nodes,
            "final_rank": "N/A"
        }
        return results, metrics

    def retrieval_v2_multi_seed(
        self,
        query: str,
        k_neighbors: int = 5,
        max_depth: int = 3,
        gate_mode: str = "margin_multi",  # "standard", "strict_margin", "relative_threshold", "weighted", "margin_multi", "adaptive_margin"
        adaptive_margin_factor: float = 0.5  # fraction of inter-seed spread to use as margin
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        """Multi-seed and margin-gated hybrid retrieval (version 2).

        Gate modes:
        - "standard": top-3 unconditionally
        - "strict_margin": single seed if margin >= 0.15 and sim >= 0.50
        - "relative_threshold": seeds within 0.85 * best_sim, up to k_neighbors
        - "weighted": top-k_neighbors always, with softmax weights
        - "margin_multi": strict_margin for single seed, else weighted fallback
        - "adaptive_margin": dynamic margin = spread * adaptive_margin_factor
        """
        from typing import Any
        parts = query.split()
        if len(parts) < 2:
            return [], {}
        subj_word, rel_word = parts[0], parts[1]
        
        causal = {"causes", "cause", "leads", "produces", "creates"}
        possessive = {"has", "have", "contains", "includes"}
        rel_type = None
        if rel_word in causal:
            rel_type = "causal"
        elif rel_word in possessive:
            rel_type = "possessive"
            
        tok = self._tokenizer
        subj_tid = tok.word_to_id.get(subj_word)
        if subj_tid is None:
            return [], {}
           
        emb = self.token_embed.weight.data[subj_tid]
        lat_query, *_ = self._encoder_forward_full(emb)
        
        # Gather all candidates in graph with similarities
        scored_neighbors = []
        for word, tid in tok.word_to_id.items():
            if word == subj_word:
                continue
            bindings = self.binding_map.get_concepts(tid, min_confidence=0.1)
            if not bindings:
                continue
            cid = bindings[0].concept_id
            
            w_emb = self.token_embed.weight.data[tid]
            lat_word, *_ = self._encoder_forward_full(w_emb)
            
            # cosine sim
            na = float(np.linalg.norm(lat_query))
            nb = float(np.linalg.norm(lat_word))
            sim = float(np.dot(lat_query, lat_word) / (na * nb + 1e-15)) if na > 0 and nb > 0 else 0.0
            
            scored_neighbors.append((cid, sim, word))
            
        scored_neighbors.sort(key=lambda x: x[1], reverse=True)
        
        # Apply Gating
        seeds = []
        if scored_neighbors:
            best_sim = scored_neighbors[0][1]
            
            if gate_mode == "strict_margin":
                margin = (best_sim - scored_neighbors[1][1]) if len(scored_neighbors) > 1 else best_sim
                if margin >= 0.15 and best_sim >= 0.50:
                    seeds = [scored_neighbors[0]]
                else:
                    seeds = []
            elif gate_mode == "relative_threshold":
                threshold = 0.85 * best_sim
                seeds = [n for n in scored_neighbors if n[1] >= threshold][:k_neighbors]
            elif gate_mode == "weighted":
                seeds = scored_neighbors[:k_neighbors]
            elif gate_mode == "margin_multi":
                margin = (best_sim - scored_neighbors[1][1]) if len(scored_neighbors) > 1 else best_sim
                if margin >= 0.15 and best_sim >= 0.50:
                    seeds = [scored_neighbors[0]]
                else:
                    seeds = scored_neighbors[:k_neighbors]
            elif gate_mode == "adaptive_margin":
                # Adaptive margin: use inter-seed spread to dynamically set threshold
                if len(scored_neighbors) >= 2:
                    spread = scored_neighbors[0][1] - scored_neighbors[-1][1]  # max - min in available
                    # Use spread within top-k_neighbors for more local measure
                    top_k_sims = [n[1] for n in scored_neighbors[:k_neighbors]]
                    local_spread = top_k_sims[0] - top_k_sims[-1] if len(top_k_sims) > 1 else best_sim
                    dynamic_margin = local_spread * adaptive_margin_factor
                    # Minimum margin floor to prevent noise admission when spread is tiny
                    dynamic_margin = max(dynamic_margin, 0.05)
                    # Select seeds with sim >= best_sim - dynamic_margin
                    threshold = best_sim - dynamic_margin
                    seeds = [n for n in scored_neighbors if n[1] >= threshold][:k_neighbors]
                else:
                    seeds = scored_neighbors[:k_neighbors]
            else:  # "standard" top-k
                seeds = scored_neighbors[:3]  # standard uses top-3
               
        # Traversal BFS
        activations = {}
        if gate_mode in ("weighted", "margin_multi") and seeds:
            sims = np.array([n[1] for n in seeds])
            temp = 0.15
            exp_sims = np.exp((sims - np.max(sims)) / temp)
            weights = exp_sims / np.sum(exp_sims)
            for (cid, _, _), w in zip(seeds, weights):
                activations[cid] = float(w)
        else:
            for cid, sim, _ in seeds:
                activations[cid] = sim
                
        frontier = [cid for cid, _, _ in seeds]
        for depth in range(1, max_depth + 1):
            next_frontier = []
            for nid in frontier:
                act = activations[nid]
                for tgt_id, edge in self.graph.get_outgoing(nid):
                    if rel_type and edge.relation_type != rel_type:
                        continue
                    prop = act * edge.weight
                    if tgt_id not in activations or prop > activations[tgt_id]:
                        activations[tgt_id] = prop
                        next_frontier.append(tgt_id)
            frontier = next_frontier
            
        # Decode results
        seed_words = {n[2] for n in seeds}
        results = []
        for cid, act in activations.items():
            tokens = self.binding_map.get_tokens(cid, 0.0)
            for b in tokens:
                word = tok.decode([b.token_id])
                if word not in seed_words and not word.startswith("?"):
                    results.append((word, act))
                    
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Compute telemetry
        top_seed_sim = seeds[0][1] if seeds else 0.0
        margin = (seeds[0][1] - seeds[1][1]) if len(seeds) > 1 else top_seed_sim
        activated_nodes = len(activations)
        dead_end_nodes = 0
        for cid in activations:
            matching_edges = 0
            for tgt_id, edge in self.graph.get_outgoing(cid):
                if rel_type and edge.relation_type != rel_type:
                    continue
                matching_edges += 1
            if matching_edges == 0:
                dead_end_nodes += 1
                
        metrics = {
            "seed_count": len(seeds),
            "top_seed_similarity": top_seed_sim,
            "margin": margin,
            "activated_nodes": activated_nodes,
            "dead_end_nodes": dead_end_nodes,
            "final_rank": "N/A"
        }
        return results, metrics

    def snapshot_replay_buffer(self, domain_name: str):
        """No-op stub for backward compatibility."""
        pass

    def activate_domain_memories(self, domain_name: str):
        """No-op stub for backward compatibility."""
        pass
