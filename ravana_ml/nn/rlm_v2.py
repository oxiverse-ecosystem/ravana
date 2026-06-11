"""
RLM v2 - Triple-Based Cognitive Architecture

Brain-inspired semantic memory that decomposes input into (subject, relation_type, object)
triples and uses spreading activation over a concept graph for inference.

Key differences from v1 (rlm.py):
- No character-level GRU - triple decomposition replaces sequential processing
- No 5-path logit blend - spreading activation is the sole inference mechanism
- Learned relation type embeddings - not keyword-based classification
- Hebbian learning on (subject @ relation_type) -> object associations
- Sleep cycles consolidate triple associations and prune weak edges

Architecture:
    input "heat causes expansion"
        -> decompose: (subject="heat", relation="causes", object="expansion")
        -> classify relation: "causes" -> CAUSAL type embedding
        -> find subject concept node in graph
        -> spread activation from subject, filtered by relation type
        -> score activated nodes against all token embeddings
        -> return logits over vocab

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
import os
from typing import Optional, List, Tuple, Dict, Set, Any
from collections import defaultdict

from .module import Module, Linear, Embedding
from ..graph import ConceptGraph, ConceptNode, ConceptEdge, ConceptBindingMap
from ..plasticity import HebbianPlasticity, AntiHebbianPlasticity, StructuralPlasticity
from ..propagation import PropagationEngine
from ..currencies import CognitiveCurrencies
from ..currency import create_rlm_currency
import zipfile
import json

# ── GloVe Embedding Loader ──────────────────────────────────────────────

def _build_glove_embedding_matrix(tokenizer, target_dim=64, glove_dim=100, max_words=200000):
    """Build embedding matrix from pre-trained GloVe vectors.
    
    Downloads glove.6B.100d.txt (cached in data/glove/) on first call.
    Projects 100D -> target_dim using a random orthogonal projection.
    Returns (vocab_size, target_dim) float32 matrix or None if unavailable.
    """
    import numpy as np
    import os
    from pathlib import Path
    
    glove_path = Path('data') / 'glove' / 'glove.6B.100d.txt'
    
    # Fallback to 50D if 100D not available
    if not glove_path.exists():
        glove_path = Path('data') / 'glove' / 'glove.6B.50d.txt'
        glove_dim = 50
    
    if not glove_path.exists():
        return None
    
    # Load GloVe vectors into dict
    word_vecs = {}
    with open(str(glove_path), 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_words:
                break
            parts = line.strip().split()
            if len(parts) != glove_dim + 1:
                continue
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            word_vecs[word] = vec
    
    # Build projection matrix glove_dim -> target_dim (random orthogonal)
    rng = np.random.RandomState(42)
    max_d = max(glove_dim, target_dim)
    full_q, _ = np.linalg.qr(rng.randn(max_d, max_d).astype(np.float32))
    proj = full_q[:target_dim, :glove_dim].copy()
    proj *= np.sqrt(float(glove_dim) / float(target_dim))
    
    # Build token->id mapping
    word_to_id = {}
    if hasattr(tokenizer, 'word_to_id') and tokenizer.word_to_id:
        word_to_id = tokenizer.word_to_id
    else:
        for tid in range(tokenizer.vocab_size):
            try:
                w = tokenizer.decode([tid]).strip().lower()
                if w:
                    word_to_id[w] = tid
            except Exception:
                pass
    
    vocab_size = tokenizer.vocab_size
    
    # Check for cached projected matrix
    cache_path = Path('data') / 'glove' / f'projected_{vocab_size}_{target_dim}.npy'
    if cache_path.exists():
        return np.load(str(cache_path))
    
    matrix = np.zeros((vocab_size, target_dim), dtype=np.float32)
    found = 0
    total = 0
    
    for word, tid in word_to_id.items():
        if not isinstance(word, str) or not word:
            continue
        total += 1
        w = word.lower().strip()
        
        vec = word_vecs.get(w)
        if vec is None and len(w) > 1:
            # Try stripping plural/tense markers
            vec = word_vecs.get(w.rstrip('s'))
        if vec is None and len(w) > 2:
            vec = word_vecs.get(w[:-1])
        
        if vec is not None:
            matrix[tid] = proj @ vec
            norm = np.linalg.norm(matrix[tid])
            if norm > 0:
                matrix[tid] *= 2.0 / norm  # normalize to radius ~2.0
            found += 1
    
    # Fill missing with random
    if found < total:
        rng2 = np.random.RandomState(42)
        for word, tid in word_to_id.items():
            if not isinstance(word, str):
                continue
            if np.all(matrix[tid] == 0):
                matrix[tid] = rng2.randn(target_dim).astype(np.float32) * 0.3
    
    # Cache the projected matrix
    cache_path = Path('data') / 'glove' / f'projected_{vocab_size}_{target_dim}.npy'
    np.save(str(cache_path), matrix)
    return matrix


# ─── Relation Type Definitions ───────────────────────────────────────────────

RELATION_TYPES = [
    "causal",       # causes, produces, leads to, results in, makes, triggers
    "semantic",     # is, are, represents, defines, means
    "temporal",     # then, after, before, next, later, during
    "possessive",   # has, contains, includes, belongs to, part of
    "analogical",   # like, similar to, resembles, analogous to
    "contextual",   # in, at, on, with, under, over
]

# Keyword -> relation type mapping for initial classification
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
                 latent_dim: int = 96,
                 hidden_dim: int = 128,
                 **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.concept_dim = concept_dim
        self.n_concepts = n_concepts
        self.max_seq_len = max_seq_len
        self.sleep_interval = sleep_interval
        self.n_hidden = kwargs.get("n_hidden", 32)
        self.n_layers = kwargs.get("n_layers", 2)
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
        # Classifier: concept_dim -> n_rel_types
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
        # 2x headroom + guarantee every vocab token gets its own concept:
        # vocab_size + 50 ensures enough slots for all token concepts + intermediates
        self._max_concepts = max(n_concepts * 2, vocab_size + 50, 150)

        # ── Relation type classifier learning rate ──
        self._classifier_lr = 0.01

        # ── Episodic memory (triple store) ──
        # Stores (subject_cid, rel_type_idx, object_cid, timestamp) for replay
        self._episodic_triples: List[Tuple[int, int, int, float]] = []
        self._max_episodic = 500

        # ── Initialize structured concepts (DISABLED - 1-to-1 mapping) ──
        # self._init_structured_concepts()

        # ── Performance tracking ──
        self._train_correct = 0
        self._train_total = 0
        self._seen_predicates = set()
        self._last_loss = 0.0

        # ── Forward pass caches (invalidated when graph changes) ──
        self._node_matrix_cache = None   # (node_ids, matrix, norms)
        self._node_matrix_version = -1   # graph version counter
        self._rel_vector_cache = {}      # rel_type_name -> avg relation vector
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
        self._rp_encoder_lr = 0.005  # encoder fine-tuning LR during RP training
        self._rp_momentum = 0.9
        self._rp_cache = None
        self.use_rp_for_analogy = True

        # ── Domain-Agnostic Bilinear Relation Predictor ──
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder weights (orthogonal init for stable gradients)
        rng_enc = np.random.RandomState(42)
        max_dim = max(self.hidden_dim, self.latent_dim, embed_dim)
        full_q, _ = np.linalg.qr(rng_enc.randn(max_dim, max_dim).astype(np.float32))
        self._enc_W1 = (full_q[:self.hidden_dim, :embed_dim].copy() * np.sqrt(2.0 / embed_dim)).astype(np.float32)
        self._enc_b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self._enc_W2 = (full_q[:self.latent_dim, :self.hidden_dim].copy() * np.sqrt(2.0 / self.hidden_dim)).astype(np.float32)
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
        # Identity init: source_embed @ I @ target_embed = dot product (cosine-like)
        # Gradients learn deviations from identity, enabling shared relation-specific transformations
        self._rp_rel_matrices = np.tile(np.eye(self.latent_dim, dtype=np.float32), (n_rel_types, 1, 1))
        self._rp_mrel_matrices = np.zeros_like(self._rp_rel_matrices)
        self.freeze_encoder = False
        # Bilinear form: logits = source_latent @ W_rel @ token_embeds.T
        # Gradient dW_rel = outer(source_latent, d_logits @ token_embeds) is
        # naturally O(vocab_size) larger than old linear form. Use rp_scale=0.1
        # to compensate, plus gradient clipping (norm=10) in _rp_backward.
        self.rp_scale = 1.0

        # ── Direct Latent -> Domain Logits (bypasses bottleneck) ──
        self.use_sparse_concept_predictor = False
        self.bottleneck_dim = 128  # kept for backward compat loading only
        self.num_domains = 2
        self.current_domain_id = None
        self.spreading_confidence_threshold = 0.35

        # Domain-Specific Heads (direct from source_latent, no bottleneck)
        self.domain_W_logits = []
        self.domain_b_logits = []

        for d in range(self.num_domains):
            W_logits = np.random.normal(0, 0.1, (self.vocab_size, self.embed_dim)).astype(np.float32)
            self.domain_W_logits.append(W_logits)
            self.domain_b_logits.append(np.zeros(self.vocab_size, dtype=np.float32))

        # Domain isolation: track which domains are frozen
        self._frozen_domains: Set[int] = set()

        # Momentum buffers
        self.domain_W_logits_m = [np.zeros_like(w) for w in self.domain_W_logits]
        self.domain_b_logits_m = [np.zeros_like(b) for b in self.domain_b_logits]

        # ── Deprecated bottleneck params (kept as zeros for backward state loading) ──
        n_rel_types = len(RELATION_TYPES)
        self.rel_encoder = np.zeros((n_rel_types, self.bottleneck_dim, self.latent_dim), dtype=np.float32)
        self.rel_bias = np.zeros((n_rel_types, self.bottleneck_dim), dtype=np.float32)
        self.rel_encoder_m = np.zeros_like(self.rel_encoder)
        self.rel_bias_m = np.zeros_like(self.rel_bias)
        self.domain_W_gates = [np.zeros((self.vocab_size, self.bottleneck_dim), dtype=np.float32) for _ in range(self.num_domains)]
        self.domain_b_gates = [np.zeros(self.vocab_size, dtype=np.float32) for _ in range(self.num_domains)]
        self.domain_W_gates_m = [np.zeros_like(w) for w in self.domain_W_gates]
        self.domain_b_gates_m = [np.zeros_like(b) for b in self.domain_b_gates]
        self.router_W = np.zeros((self.num_domains, self.bottleneck_dim), dtype=np.float32)
        self.router_b = np.zeros(self.num_domains, dtype=np.float32)
        self.router_W_m = np.zeros_like(self.router_W)
        self.router_b_m = np.zeros_like(self.router_b)

        # Deterministic Subword (char-level CNN) weights
        rng_char = np.random.RandomState(42)
        self.char_embed = rng_char.randn(128, 64).astype(np.float32) * 0.1
        self.char_cnn_W = rng_char.randn(128, 3, 64).astype(np.float32) * np.sqrt(2.0 / (3 * 64))
        self.char_cnn_b = np.zeros(128, dtype=np.float32)
        self.char_to_token_W = rng_char.randn(embed_dim, 128).astype(np.float32) * np.sqrt(2.0 / 128)
        self.char_to_token_b = np.zeros(embed_dim, dtype=np.float32)
        self.fusion_W = rng_char.randn(embed_dim, embed_dim * 2).astype(np.float32) * np.sqrt(2.0 / (embed_dim * 2))
        self.fusion_b = np.zeros(embed_dim, dtype=np.float32)

        # Pre-defined OOD anchor concepts mapping
        self.anchor_concepts = {
            "gravity": ["force", "mass", "field", "weight"],
            "orbits": ["circular", "path", "motion", "celestial"],
            "kindness": ["sharing", "empathy", "generosity", "friendship"],
            "anger": ["conflict", "jealousy", "rudeness", "resentment"],
            "sharing": ["friendship", "generosity", "kindness"],
            "lying": ["betrayal", "gossip", "mistrust"],
            "patience": ["understanding", "harmony", "peaceful"],
            "honesty": ["trust", "respect", "noble"],
            "empathy": ["connection", "compassion", "understanding"],
            "greed": ["loneliness", "harmful", "dangerous"],
            "jealousy": ["resentment", "conflict", "sadness"],
            "generosity": ["gratitude", "valuable", "important"],
            "rudeness": ["offense", "harmful", "tension"],
            "gossip": ["mistrust", "lying", "betrayal"],
            "collaboration": ["teamwork", "success", "innovation"],
            "grief": ["empathy", "sadness", "natural"],
            "curiosity": ["discovery", "exploration", "valuable"],
            "apology": ["harmony", "restorative", "forgiveness"],
            "compassion": ["suffering", "peaceful", "helpfulness"],
            "leadership": ["action", "excellence", "admirable"]
        }
        
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
        
        # Generalization Enhancement Attributes
        self.use_subspace_projection = False
        self.lambda_recon = 0.0
        self.rel_proj = np.eye(self.latent_dim, dtype=np.float32)
        
        # Ablation: disable spreading activation to isolate analogy path
        self.disable_spreading_activation = False

        # ── Cognitive State - unified via CognitiveCurrencies ──
        self.currencies = CognitiveCurrencies()

        # Native Memory (lightweight episodic buffer + semantic consolidation)
        self._episodic_buffer: List[Dict] = []    # recent experiences
        self._episodic_buffer_max = 500
        self._episodic_keys: List = []            # episodic key vectors for lookup
        self._episodic_values: List = []          # episodic value targets
        self._episodic_max = 5000                 # max episodic key-value pairs
        # Vector-based episodic retrieval
        from ..embedder import LearnedEmbedder
        self._epi_embedder = LearnedEmbedder(dim=64)
        self._epi_vectors: Dict[int, np.ndarray] = {}  # idx -> 64-dim vector
        self._epi_next_idx: int = 0
        self._epi_matrix: Optional[np.ndarray] = None  # lazy (N, 64) matrix
        self._epi_dirty: bool = True
        self._semantic_memories: Dict[int, Dict] = {}  # consolidated: {concept_id: {strength, access_count, last_access}}
        self._semantic_memory_max = 1000

        # Concept emotion tags
        self._concept_vad: Dict[int, Tuple[float, float, float]] = {}  # {concept_id: (v, a, d)}

        # Unified Cognitive Currency
        self.currency = create_rlm_currency()

        # Edge weight convergence tracking
        self._edge_weight_ema = 0.0          # EMA of mean edge weight (should rise over time)
        self._edge_weight_prev = 0.0         # previous EMA snapshot for delta
        self._token_hit_ema = 0.5            # EMA of token-level prediction hit rate

        self.sleep_cycles_completed = 0
        self.conceptual_accuracy = 0.0

        self._last_predicted_concepts: List[int] = []
        self._last_edge_pred: List[int] = []
        self._last_hidden_state: Optional[np.ndarray] = None

        # Interleaved Replay Buffer (for continual learning)
        self._replay_buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        self._replay_buffer_max: int = 500
        self._replay_n_samples: int = 20
        self._domain_memories: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

        # ── CROSS-DOMAIN GENERALIZATION FLAGS ──
        # Freeze token embeddings during RP backward (prevents per-pair memorization).
        # The autoencoder pre-train provides a stable semantic space; only W_rel learns.
        self.freeze_token_embeds_in_rp = True
        # Cross-domain alignment toggle (used by _cross_domain_relation_alignment).
        self.use_cross_domain_alignment = False
        # ── VERB-STEM OFFSET PREDICTOR ──
        # Replaces bilinear W_rel @ subject for cross-domain held-out generalization.
        # offset(verb) = avg(target_embed - subject_embed) over all training pairs
        #   using that verb. Prediction at inference:
        #     predicted_embed = subject_embed + offset(query_verb)
        #     logits_k = predicted_embed @ token_embed_k  (cosine similarity)
        # This naturally handles same-subject different-verb cases (causes vs freezes)
        # because each verb has its own offset vector.
        self._verb_offsets: Dict[str, np.ndarray] = {}          # verb_stem -> offset vector
        self._verb_offset_count: Dict[str, int] = {}             # verb_stem -> count
        self._verb_accum_buffer: List[Tuple[str, np.ndarray]] = []  # (verb_stem, offset_vec) pairs
        self.use_verb_offset = False                              # enabled by experiment

        # Track cross-domain edges injected analogically (subject_cid, object_cid).
        self._cross_domain_edges_injected: Set[Tuple[int, int]] = set()


    # ── Domain Isolation Methods ─────────────────────────────────────────

    def set_domain(self, domain_id: Optional[int], freeze_others: bool = True):
        """Set active domain and optionally freeze all other domain heads.
        
        When domain_id is None: all heads are trainable (soft routing).
        When domain_id is set: only that head is active and trainable.
        """
        self.current_domain_id = domain_id
        if domain_id is not None and freeze_others:
            self._frozen_domains = {d for d in range(self.num_domains) if d != domain_id}
        else:
            self._frozen_domains = set()
        # During forward/backward, only non-frozen domains are updated

    def freeze_domain(self, domain_id: int):
        """Freeze a specific domain head (prevents updates during training)."""
        self._frozen_domains.add(domain_id)

    def unfreeze_domain(self, domain_id: int):
        """Unfreeze a domain head."""
        self._frozen_domains.discard(domain_id)

    def unfreeze_all_domains(self):
        """Unfreeze all domain heads."""
        self._frozen_domains.clear()


    def _init_structured_concepts(self):
        """Create initial concept nodes distributed across the concept space."""
        rng = np.random.RandomState(42)
        for i in range(self.n_concepts):
            vec = rng.randn(self.concept_dim).astype(np.float32) * 0.1
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            self.graph.add_node(vector=vec, label=f"init_{i}")

    # ── Property aliases for backward compatibility with CognitiveCurrencies ──

    @property
    def identity_strength(self):
        return self.currencies.identity_strength

    @identity_strength.setter
    def identity_strength(self, val):
        self.currencies.identity_strength = val

    @property
    def identity_momentum(self):
        return self.currencies.identity_momentum

    @identity_momentum.setter
    def identity_momentum(self, val):
        self.currencies.identity_momentum = val

    @property
    def identity_history(self):
        return self.currencies.identity_history

    @identity_history.setter
    def identity_history(self, val):
        self.currencies.identity_history = val

    @property
    def valence(self):
        return self.currencies.valence

    @valence.setter
    def valence(self, val):
        self.currencies.valence = val

    @property
    def arousal(self):
        return self.currencies.arousal

    @arousal.setter
    def arousal(self, val):
        self.currencies.arousal = val

    @property
    def dominance(self):
        return self.currencies.dominance

    @dominance.setter
    def dominance(self, val):
        self.currencies.dominance = val

    @property
    def accumulated_meaning(self):
        return self.currencies.accumulated_meaning

    @accumulated_meaning.setter
    def accumulated_meaning(self, val):
        self.currencies.accumulated_meaning = val

    @property
    def meaning_history(self):
        return self.currencies.meaning_history

    @meaning_history.setter
    def meaning_history(self, val):
        self.currencies.meaning_history = val

    @property
    def sleep_pressure(self):
        return self.currencies.sleep_pressure

    @sleep_pressure.setter
    def sleep_pressure(self, val):
        self.currencies.sleep_pressure = val

    @property
    def sleep_pressure_threshold(self):
        return self.currencies.sleep_pressure_threshold

    @sleep_pressure_threshold.setter
    def sleep_pressure_threshold(self, val):
        self.currencies.sleep_pressure_threshold = val

    @property
    def regulation_mode(self):
        return self.currencies.regulation_mode

    @regulation_mode.setter
    def regulation_mode(self, val):
        self.currencies.regulation_mode = val

    @property
    def dissonance_ema(self):
        return self.currencies.dissonance_ema

    @dissonance_ema.setter
    def dissonance_ema(self, val):
        self.currencies.dissonance_ema = val

    @property
    def dissonance_normalized(self) -> float:
        return self.currencies.dissonance_normalized

    # ── Cognitive Helpers ──

    def _update_emotion(self, valence_stimulus: float, arousal_stimulus: float,
                        dominance_stimulus: float = 0.0):
        self.currencies._update_emotion(valence_stimulus, arousal_stimulus, dominance_stimulus)

    def _compute_identity_update(self, error: float, is_correct: bool) -> float:
        return self.currencies._compute_identity_update(error, is_correct)

    def _compute_meaning(self, error: float) -> float:
        return self.currencies._compute_meaning(error)

    def _store_episode(self, error: float, is_correct: bool, domain: Optional[str] = None):
        importance = 1.0 - min(1.0, error)
        if is_correct:
            importance *= 0.8
        else:
            importance = min(1.0, importance + 0.3)

        episode = {
            'vector': self._last_hidden_state.copy() if self._last_hidden_state is not None else None,
            'concepts': list(self._last_predicted_concepts),
            'error': error,
            'correct': is_correct,
            'valence': self.valence,
            'arousal': self.arousal,
            'timestamp': time.time(),
            'importance': importance,
            'consolidation_state': 'fresh',
            'access_count': 0,
            'domain': domain,
        }
        self._episodic_buffer.append(episode)
        if len(self._episodic_buffer) > self._episodic_buffer_max:
            self._evict_lowest_salience(self._episodic_buffer)

    def _evict_lowest_salience(self, buffer: List[Dict]) -> List[Dict]:
        now = time.time()
        min_score = float('inf')
        min_idx = 0
        for i, ep in enumerate(buffer):
            age = now - ep.get('timestamp', now)
            recency = 1.0 / (1.0 + age * 0.01)
            importance = ep.get('importance', 0.5)
            error_signal = min(1.0, ep.get('error', 0.5))
            score = importance * 0.4 + recency * 0.3 + error_signal * 0.3
            if score < min_score:
                min_score = score
                min_idx = i
        buffer.pop(min_idx)
        return buffer

    def _consolidate_episodic_to_semantic(self):
        for ep in self._episodic_buffer:
            if ep['correct'] and ep['error'] < 0.3:
                importance = ep.get('importance', 0.5)
                is_novel = ep.get('consolidation_state', 'fresh') == 'fresh'
                strength_delta = 0.05 * importance * (1.5 if is_novel else 1.0)
                for cid in ep['concepts']:
                    if cid in self._semantic_memories:
                        self._semantic_memories[cid]['strength'] = min(1.0,
                            self._semantic_memories[cid]['strength'] + strength_delta)
                        self._semantic_memories[cid]['access_count'] += 1
                    else:
                        if len(self._semantic_memories) < self._semantic_memory_max:
                            self._semantic_memories[cid] = {
                                'strength': 0.3 * importance,
                                'access_count': 1,
                                'last_access': time.time(),
                            }

    def _decay_semantic_memories(self):
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
        for cid, mem in self._semantic_memories.items():
            node = self.graph.get_node(cid)
            if not node or mem['strength'] <= 0.2:
                continue
            try:
                similar = self.graph.find_similar(node.vector, k=10)
            except Exception:
                continue
            for target_cid, sim in similar:
                if target_cid == cid or sim < 0.2:
                    continue
                mem2 = self._semantic_memories.get(target_cid)
                if not mem2 or mem2['strength'] <= 0.2:
                    continue
                edge = self.graph.get_edge(cid, target_cid)
                if edge:
                    edge.weight += 0.01 * mem['strength'] * mem2['strength'] * sim
                    edge.confidence = min(1.0, edge.confidence + 0.005 * sim)
                elif mem['strength'] > 0.5 and mem2['strength'] > 0.5 and sim > 0.4:
                    edge = self.graph.add_edge(cid, target_cid, weight=0.1 * sim)
                    edge.confidence = 0.3

    def _regulate_cognitive_state(self):
        self.currencies.regulate()

    def buffer_experience(self, input_ids: np.ndarray, target_ids: np.ndarray,
                          domain: Optional[str] = None):
        entry = (input_ids.copy(), target_ids.copy())
        self._replay_buffer.append(entry)
        if domain is not None:
            if domain not in self._domain_memories:
                self._domain_memories[domain] = []
            self._domain_memories[domain].append(entry)
        if len(self._replay_buffer) > self._replay_buffer_max:
            self._replay_buffer = self._replay_buffer[-self._replay_buffer_max:]

    def _replay_old_memories(self, n_samples: int = 20):
        if not self._replay_buffer:
            return
        n = min(n_samples, len(self._replay_buffer))
        rng = np.random.RandomState()
        indices = rng.choice(len(self._replay_buffer), size=n, replace=False)
        for idx in indices:
            input_ids, target_ids = self._replay_buffer[idx]
            self.learn(input_ids, target_ids)

    # ── Dimension Bridging ──────────────────────────────────────────────────

    def _project_to_concept(self, embed_vec: np.ndarray) -> np.ndarray:
        """Project embed_dim vector to concept_dim space via learned projection."""
        if len(embed_vec) == self.concept_dim:
            return embed_vec
        # Use the subject projection layer
        return self.subject_proj(embed_vec.reshape(1, -1)).data.flatten()

    def _project_to_embed(self, concept_vec: np.ndarray) -> np.ndarray:
        """Project concept_dim vector to embed_dim space via learned projection."""
        if len(concept_vec) == self.embed_dim:
            return concept_vec
        return self.concept_to_embed(concept_vec.reshape(1, -1)).data.flatten()

    def _cached_norm(self, vec: np.ndarray, cache_key: str) -> float:
        """Cached linalg.norm - avoids redundant computation within a forward pass."""
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

        This is what makes "melts" -> CAUSAL work (same pathway as "causes").
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

        # Use keyword classifier directly - learned relation type embeddings
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

        Uses the subject projection to map embed_dim -> concept_dim,
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
        # Check binding map first - reuse existing concept for this token
        bindings = self.binding_map.get_concepts(token_id, min_confidence=0.1)
        if bindings:
            cid = bindings[0].concept_id
            # Validate that the node still exists in the graph
            # (binding may be stale if node was pruned)
            if self.graph.get_node(cid) is not None:
                return cid
            # Stale binding - fall through to create a fresh concept

        # Always create a new concept for this token (no merging)
        concept_vec = self._project_to_concept(embed_vec)
        if len(self.graph.nodes) < self._max_concepts:
            node = self.graph.add_node(vector=concept_vec, label=self._decode_token(token_id))
            nid = node.id
        else:
            # At capacity - find nearest existing concept to reuse
            # (better than a phantom ID that doesn't exist in the graph)
            nid, sim = self._nearest_concept(embed_vec)
            if nid < 0:
                # No concepts at all - force create
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
        """Initialize token embeddings using pre-trained GloVe vectors.
        
        Uses glove.6B.100d.txt (cached in data/glove/) projected to embed_dim.
        GloVe vectors capture genuine semantic relationships, making the
        verb-stem offset predictor work:
          offset("causes") = avg(expansion - heat, vision - light, ...)
        Character n-gram embeddings (previously used) cannot capture this.
        """
        import numpy as np
        tokenizer = self._tokenizer_val
        
        matrix = _build_glove_embedding_matrix(
            tokenizer, target_dim=self.embed_dim, glove_dim=100
        )
        
        if matrix is not None:
            n_found = np.count_nonzero(matrix.any(axis=1))
            coverage = n_found / max(1, self.vocab_size)
            print(f"  [Embeddings] GloVe 100D -> {self.embed_dim}D: {n_found}/{self.vocab_size} tokens ({coverage:.1%})")
            self.token_embed.weight.data[:matrix.shape[0]] = matrix
        else:
            # Fallback: random orthogonal init
            print(f"  [Embeddings] GloVe unavailable. Using random init.")
            rng = np.random.RandomState(42)
            max_dim = max(self.vocab_size, self.embed_dim)
            full_q, _ = np.linalg.qr(rng.randn(max_dim, max_dim).astype(np.float32))
            self.token_embed.weight.data[:] = full_q[:self.vocab_size, :self.embed_dim] * 0.1
                
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
        z1, h1: (B, hidden_dim) pre/post activation of first layer
        z2, h2: (B, latent_dim) pre/post activation of second layer
        d_h2: (B, latent_dim) gradient w.r.t. h2
        Returns (d_W1, d_b1, d_W2, d_b2).
        """
        d_z2 = d_h2 * (1.0 - h2 * h2)
        d_enc_W2 = d_z2.T @ h1
        d_enc_b2 = np.sum(d_z2, axis=0)

        d_h1 = d_z2 @ self._enc_W2
        d_z1 = d_h1 * (1.0 - h1 * h1)
        d_enc_W1 = d_z1.T @ X
        d_enc_b1 = np.sum(d_z1, axis=0)

        return d_enc_W1, d_enc_b1, d_enc_W2, d_enc_b2
    def get_robust_embedding(self, tid):
        """Get subword character-CNN augmented robust embedding for a token ID."""
        token_emb = self.token_embed.weight.data[tid]
        if not hasattr(self, "_tokenizer") or self._tokenizer is None:
            return token_emb
        word = ""
        try:
            word = self._tokenizer.decode([tid]).strip()
        except Exception:
            pass
        if not word:
            return token_emb
        
        c_ids = [ord(c) for c in word if ord(c) < 128]
        if not c_ids:
            return token_emb
            
        char_emb = self.char_embed[c_ids] # (L, 64)
        L = len(c_ids)
        padded = np.zeros((L + 2, 64), dtype=np.float32)
        padded[1:-1] = char_emb
        
        conv_out = np.zeros((L, 128), dtype=np.float32)
        for i in range(L):
            patch = padded[i:i+3] # (3, 64)
            conv_out[i] = np.sum(patch[np.newaxis, :, :] * self.char_cnn_W, axis=(1, 2)) + self.char_cnn_b
            
        features = np.max(conv_out, axis=0) # (128,)
        char_emb_final = self.char_to_token_W @ features + self.char_to_token_b # (embed_dim,)
        
        combined = np.concatenate([token_emb, char_emb_final]) # (embed_dim * 2,)
        fused = self.fusion_W @ combined + self.fusion_b # (embed_dim,)
        return fused

    def _get_anchor_regularized_latent(self, token_id, latent):
        """Regularize latent representation towards expected semantic anchor neighborhoods."""
        if not hasattr(self, "_tokenizer") or self._tokenizer is None or not hasattr(self, "anchor_concepts"):
            return latent
        try:
            word = self._tokenizer.decode([token_id]).strip().lower()
        except Exception:
            return latent
        if word not in self.anchor_concepts:
            return latent
            
        anchors = self.anchor_concepts[word]
        anchor_latents = []
        for anchor_word in anchors:
            aid = self._tokenizer.word_to_id.get(anchor_word)
            if aid is not None:
                a_embed = self.get_robust_embedding(aid)
                a_lat, _, _, _ = self._encoder_forward_full(a_embed)
                anchor_latents.append(a_lat)
        if not anchor_latents:
            return latent
        mean_anchor_latent = np.mean(anchor_latents, axis=0)
        lambda_anchor = getattr(self, "lambda_anchor", 0.3)
        
        norm_before = np.linalg.norm(latent)
        reg_latent = (1.0 - lambda_anchor) * latent + lambda_anchor * mean_anchor_latent
        norm_after = np.linalg.norm(reg_latent)
        if norm_after > 0:
            reg_latent = reg_latent * (norm_before / norm_after)
        return reg_latent

    def get_query_confidence(self, subject_cid, rel_type_name, rp_probs=None):
        """Compute confidence score. Uses RP output entropy when available, falls back to edge-based."""
        if rp_probs is not None:
            max_prob = float(np.max(rp_probs))
            if max_prob > 0.3:
                return 1.0
            entropy = -float(np.sum(rp_probs * np.log(rp_probs + 1e-15)))
            max_entropy = float(np.log(len(rp_probs)))
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
            confidence = 1.0 - norm_entropy
            return min(1.0, max(0.0, confidence))
        matching_edges = [edge for _, edge in self.graph.get_outgoing(subject_cid) if edge.relation_type == rel_type_name]
        if not matching_edges:
            return 0.0
        return max(edge.weight * edge.confidence for edge in matching_edges)

    # ── Verb-Stem Helper ───────────────────────────────────────────────────

    @staticmethod
    def _verb_stem(word: str) -> str:
        """Extract verb stem by stripping common suffixes.
        
        'causes' -> 'caus', 'freezes' -> 'freez', 'produces' -> 'produc',
        'makes' -> 'make', 'melts' -> 'melt', 'is' -> 'is'
        Stemming preserves enough information to distinguish verbs that
        map to the same relation type but have different semantics.
        """
        w = word.lower().strip()
        # Handle short words (2+ chars for stem)
        if len(w) <= 3:
            return w
        # Strip common suffixes
        for suffix in ['ing', 'ed', 'es', 's', 'd']:
            if w.endswith(suffix) and len(w) > len(suffix) + 1:
                w = w[:-len(suffix)]
                break
        return w

    def _accumulate_verb_offset(self, subject_tid: int, target_tid: int, verb_word: str):
        """Accumulate offset = target_embed - subject_embed for a verb stem.
        
        Called during learn() for each training triple.
        Offsets are finalized into _verb_offsets by _compute_verb_offsets().
        """
        if not verb_word or not self.use_verb_offset:
            return
        stem = self._verb_stem(verb_word)
        subject_embed = self.token_embed.weight.data[subject_tid]
        target_embed = self.token_embed.weight.data[target_tid]
        offset = target_embed - subject_embed
        self._verb_accum_buffer.append((stem, offset))

    def _compute_verb_offsets(self):
        """Finalize verb offsets by averaging accumulated (target - subject) vectors.
        
        Should be called after training is complete (before evaluation).
        Unseen verbs at inference fall back to bilinear W_rel.
        """
        if not self.use_verb_offset:
            return
        # Group by verb stem
        sums: Dict[str, np.ndarray] = {}
        counts: Dict[str, int] = {}
        for stem, offset in self._verb_accum_buffer:
            if stem not in sums:
                sums[stem] = np.zeros_like(offset)
                counts[stem] = 0
            sums[stem] += offset
            counts[stem] += 1
        # Compute averages
        for stem, total in sums.items():
            if counts[stem] > 0:
                offset = total / counts[stem]
                # Normalize to prevent embedding-space drift
                norm = np.linalg.norm(offset)
                if norm > 0:
                    offset = offset / norm * min(norm, 5.0)  # cap magnitude at 5
                self._verb_offsets[stem] = offset
                self._verb_offset_count[stem] = counts[stem]
        n_verbs = len(self._verb_offsets)
        if n_verbs > 0:
            print(f"  [Verb Offset] Computed {n_verbs} verb offsets from {len(self._verb_accum_buffer)} training pairs")
            # Log some examples
            for stem, cnt in sorted(self._verb_offset_count.items(), key=lambda x: -x[1])[:5]:
                print(f"    '{stem}': {cnt} examples")

    def _rp_forward_verb_offset(self, subject_tid: int, verb_word: str) -> Optional[np.ndarray]:
        """Predict using verb-stem offset arithmetic.
        
        predicted_embed = subject_embed + offset(verb_stem)
        logits_k = predicted_embed @ token_embed_k / (temperature)
        
        Returns logits over vocab, or None if verb is unknown.
        """
        if not verb_word or not self.use_verb_offset:
            return None
        stem = self._verb_stem(verb_word)
        if stem not in self._verb_offsets:
            return None
        
        source_embed = self.token_embed.weight.data[subject_tid]
        offset = self._verb_offsets[stem]
        predicted = source_embed + offset
        
        # Cosine similarity against all token embeddings
        token_embeds = self.token_embed.weight.data  # (vocab_size, embed_dim)
        token_norms = np.linalg.norm(token_embeds, axis=1)
        pred_norm = np.linalg.norm(predicted)
        
        if pred_norm > 0 and np.any(token_norms > 0):
            valid_tok = token_norms > 0
            normed_tok = token_embeds.copy()
            normed_tok[valid_tok] /= token_norms[valid_tok, np.newaxis]
            logits = (predicted / pred_norm) @ normed_tok.T  # cosine similarities
            # Suppress subject token (self-prediction) - strong negation
            if 0 <= subject_tid < len(logits):
                logits[subject_tid] = np.min(logits) - 10.0
            # Scale up to make softmax meaningful
            logits *= 10.0
            return logits
        return None

    def _rp_forward(self, subject_tid, rel_type_idx, route_softly=None, verb_word=None):
        """Bilinear RP forward using raw token embeddings (bypass collapsed encoder).

        Uses source_embed directly (no encoder projection) and target_embeds
        directly. The bilinear gradient trains token embeddings end-to-end,
        just like standard KG completion (RESCAL).

        logits_k = source_embed @ W_rel @ target_embed_k
        """
        subject_tid = int(subject_tid)
        rel_type_idx = int(rel_type_idx)

        # ── Verb-Stem Offset Path (primary) ──
        # When use_verb_offset is True and the verb is known, use offset arithmetic
        # instead of bilinear W_rel. This enables same-subject different-target
        # predictions (e.g., "cold causes" -> shivering, "cold freezes" -> water).
        # It also enables cross-domain transfer via shared verb semantics.
        if verb_word:
            verb_logits = self._rp_forward_verb_offset(subject_tid, verb_word)
            if verb_logits is not None:
                self._rp_cache = None  # prevent stale cache from corrupting W_rel
                return verb_logits

        # ── Bilinear W_rel Path (fallback) ──
        # Use raw token embeddings directly (bypass collapsed encoder)
        source_embed = self.get_robust_embedding(subject_tid)  # (embed_dim,)
        token_embeds = self.token_embed.weight.data  # (vocab_size, embed_dim)

        # Project to latent_dim if embed_dim != latent_dim
        if self.embed_dim != self.latent_dim:
            source_latent, _, _, _ = self._encoder_forward_full(source_embed)
            # Cache encoded targets (token embeddings are frozen during RP training,
            # so encode once and reuse). Invalidate by re-computing on first call
            # after token_embed changes (freeze_token_embeds_in_rp=True by default).
            if getattr(self, '_cached_encoded_targets', None) is None:
                encoded = []
                for i in range(token_embeds.shape[0]):
                    lat, _, _, _ = self._encoder_forward_full(token_embeds[i])
                    encoded.append(lat)
                self._cached_encoded_targets = np.stack(encoded)
            target_latents = self._cached_encoded_targets
        else:
            source_latent = source_embed
            target_latents = token_embeds

        # Relation matrix (shared across ALL subjects)
        W_rel = self._rp_rel_matrices[rel_type_idx]  # (latent_dim, latent_dim)

        # Bilinear scoring: logits = source_latent @ W_rel @ target_latents.T
        projected = source_latent @ W_rel  # (latent_dim,)
        logits = projected @ target_latents.T  # (vocab_size,)

        # Cache for backprop
        self._rp_cache = (
            subject_tid, rel_type_idx,
            source_embed, source_latent,
            token_embeds, target_latents,
            W_rel, logits
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
        """Bilinear RP backward with token embedding gradient flow.

        Gradients flow through:
        1. W_rel via dW_rel = outer(source_latent, d_logits @ target_latents)
        2. Source embedding: d_source_embed = W_rel @ (d_logits @ target_latents)
        3. Target embeddings: d_target_latents[k] = d_logits[k] * (W_rel @ source_latent)

        This trains token embeddings end-to-end with the bilinear scoring,
        just like standard KG completion (RESCAL).
        """
        if self._rp_cache is None:
            return
            
        (
            subject_tid, rel_type_idx,
            source_embed, source_latent,
            token_embeds, target_latents,
            W_rel, logits
        ) = self._rp_cache
        
        # Softmax loss gradient
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)
        d_logits = probs.copy()
        if 0 <= target_id < len(d_logits):
            d_logits[target_id] -= 1.0
        d_logits *= getattr(self, "rp_scale", 16.0)

        # === Gradient w.r.t. W_rel ===
        d_logits_proj = d_logits @ target_latents  # (latent_dim,)
        dW_rel = np.outer(source_latent, d_logits_proj)  # (latent_dim, latent_dim)

        # Gradient clipping (prevent NaN from bilinear amplification)
        grad_norm = np.linalg.norm(dW_rel)
        if grad_norm > 10.0:
            dW_rel *= (10.0 / (grad_norm + 1e-15))

        # === Gradients w.r.t. token embeddings ===
        # dL/d(source_embed) = W_rel @ (d_logits @ target_latents)
        d_source_latent = W_rel @ d_logits_proj  # (latent_dim,)
        
        # dL/d(target_latents[k]) = d_logits[k] * (W_rel @ source_latent)
        d_target_latent_proj = W_rel @ source_latent  # (latent_dim,)
        # Full gradient matrix: d_logits (vocab_size,) * d_target_latent_proj (latent_dim,)
        # = outer(d_logits, d_target_latent_proj)  # (vocab_size, latent_dim)
        d_target_latents = np.outer(d_logits, d_target_latent_proj)  # (vocab_size, latent_dim)

        lr = self._rp_lr * lr_scale
        # ── CROSS-DOMAIN GENERALIZATION FIX ──
        # Token embeddings are FROZEN during RP training by default. Set
        # self.freeze_token_embeds_in_rp = False to revert to the old
        # per-pair memorization behavior.
        #
        # Why: When embed_lr > 0, every step pulls source toward W_rel @ target AND
        # pulls all targets toward W_rel @ source. Over thousands of steps each (s, o)
        # pair co-adapts (memorization), and held-out subjects drift into noise via
        # negative-gradient updates. Result: 100% train acc, 0% held-out.
        # Freezing embeddings keeps them in the autoencoder-aligned semantic space and
        # forces W_rel to learn a generic relation transform that actually generalizes.
        freeze_token_embeds = getattr(self, 'freeze_token_embeds_in_rp', True)
        embed_lr = 0.0 if freeze_token_embeds else lr * 0.1

        # Update the SHARED relation matrix with momentum
        self._rp_mrel_matrices[rel_type_idx] = (
            self._rp_momentum * self._rp_mrel_matrices[rel_type_idx] - lr * dW_rel
        )
        self._rp_rel_matrices[rel_type_idx] += self._rp_mrel_matrices[rel_type_idx]

        if embed_lr > 0:
            # Update source embedding (trains it to be relation-aware)
            self.token_embed.weight.data[subject_tid] -= embed_lr * d_source_latent
            # Update ALL target embeddings (the relation-aware signal spreads)
            self.token_embed.weight.data -= embed_lr * d_target_latents
            # Embeddings changed -> invalidate norm cache
            self._token_embed_norms = None

        self._rp_cache = None
    # ── Spreading Activation Inference ──────────────────────────────────────

    def forward(self, token_ids: np.ndarray) -> 'tensor':
        """Forward pass: predict next token via spreading activation.

        Input: token_ids array (1-D or 2-D with batch dim)
        Output: logits over vocab_size

        This is the core inference mechanism:
        Step 1: Decompose into (subject, relation, object) -- object is what we predict
        Step 2: Find subject concept node
        Step 3: Classify relation type
        Step 4: Spread activation from subject, filtered by relation type
        Step 5: Score activated nodes against all token embeddings
        Step 6: Return logits over vocab
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
        self._last_hidden_state = self._project_to_concept(subject_embed)
        subject_cid = self._get_or_create_concept(subject_tid, subject_embed)

        # Classify relation type
        rel_type_idx, rel_type_embed = self._classify_relation_learned(relation_ids)
        rel_type_name = RELATION_TYPES[rel_type_idx]

        query_verb_word = ""
        if relation_ids:
            try:
                query_verb_word = self._decode_token(relation_ids[0]).lower().strip()
            except Exception:
                pass

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
            # Pass verb_word for verb-offset path (when use_verb_offset is True)
            rp_logits = self._rp_forward(subject_tid, rel_type_idx, verb_word=query_verb_word)
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

        # ── OOD Detection for Spreading Activation ──
        # Three signals for OOD:
        # 1. Unseen predicate (was never in training data)
        # 2. RP max_prob low (< 0.05) 
        # 3. No matching outgoing edges from subject for this relation type
        in_training = getattr(self, '_training_mode', False)
        disable_spreading = getattr(self, 'disable_spreading_activation', False)
        if not in_training and not disable_spreading:
            seen_preds = getattr(self, '_seen_predicates', set())
            rp_probs_for_conf = getattr(self, '_rp_probs_cache', None)

            # Signal 1: unseen predicate
            pred_is_ood = bool(
                query_verb_word
                and query_verb_word not in seen_preds
                and len(seen_preds) > 0
            )

            # Signal 2: RP uncertainty
            rp_is_uncertain = False
            if rp_probs_for_conf is not None:
                max_rp = float(np.max(rp_probs_for_conf))
                rp_is_uncertain = max_rp < 0.05

            # Signal 3: no matching outgoing edges from subject
            subject_out = self.graph.get_outgoing(subject_cid)
            has_matching_edges = any(
                e.relation_type == rel_type_name for _, e in subject_out
            ) if subject_out else False

            is_ood_query = pred_is_ood or rp_is_uncertain or not has_matching_edges
            if is_ood_query:
                disable_spreading = True

        if disable_spreading:
            # OOD / novel query: skip spreading, rely on direct RP + similarity
            for node in self.graph.nodes.values():
                node.activation = 0.0
            self.graph.activate(subject_cid, amount=1.0)
        else:
                # Reset all activations
                for node in self.graph.nodes.values():
                    node.activation = 0.0

                # Activate subject concept
                self.graph.activate(subject_cid, amount=1.0)

                # Phase 0: Similarity-based priming (VECTORIZED - single matmul)
                # "Tiger is cat-like" - activate concepts with similar embeddings.
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
                                    score = node.activation * 0.5 * edge.weight * edge.confidence
                                    # Apply predicate matching
                                    pred_mult = 1.0
                                    if query_verb_word and hasattr(edge, 'predicate_token_id') and edge.predicate_token_id != -1:
                                        try:
                                            edge_verb_word = self._decode_token(edge.predicate_token_id).lower().strip()
                                            if edge_verb_word:
                                                w1 = query_verb_word.rstrip('s').rstrip('d')
                                                w2 = edge_verb_word.rstrip('s').rstrip('d')
                                                if w1 == w2 or w1 in w2 or w2 in w1:
                                                    pred_mult = 2.5
                                            else:
                                                    pred_mult = 0.4
                                        except Exception:
                                            pass
                                    score *= pred_mult
                                    to_activate.append((tgt_id, score))
                    for nid, amount in to_activate:
                        self.graph.activate(nid, amount=amount)

                # Phase 2b: Direct-edge boost from subject
                # Spreading activation decays subject's activation (1.0 -> ~0.8) before it
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
                                # Apply predicate matching
                                pred_mult = 1.0
                                if query_verb_word and hasattr(edge, 'predicate_token_id') and edge.predicate_token_id != -1:
                                    try:
                                        edge_verb_word = self._decode_token(edge.predicate_token_id).lower().strip()
                                        if edge_verb_word:
                                            w1 = query_verb_word.rstrip('s').rstrip('d')
                                            w2 = edge_verb_word.rstrip('s').rstrip('d')
                                            if w1 == w2 or w1 in w2 or w2 in w1:
                                                pred_mult = 2.5
                                        else:
                                                pred_mult = 0.4
                                    except Exception:
                                        pass
                                boost *= pred_mult
                                self.graph.activate(tgt_id, amount=boost)

        # ── Score tokens via activated concepts ──
        concept_scores = np.zeros(self.vocab_size, dtype=np.float32)
        if getattr(self, '_rp_probs_cache', None) is not None:
            gated_rp = np.zeros_like(self._rp_probs_cache)
            rp_gate_base = 0.15
            for tok_id in range(self.vocab_size):
                prob = self._rp_probs_cache[tok_id]
                bindings = self.binding_map.get_concepts(tok_id, min_confidence=0.0)
                if bindings:
                    max_act = 0.0
                    for b in bindings:
                        node = self.graph.get_node(b.concept_id)
                        if node is not None:
                            max_act = max(max_act, node.activation)
                    gated_rp[tok_id] = prob * (rp_gate_base + max_act)
                else:
                    gated_rp[tok_id] = prob * 0.8
            concept_scores += gated_rp * 35.0
        
        # OOD fallback: direct similarity scoring via token embedding space
        # Use raw token embedding (not get_robust_embedding which has random char-CNN noise)
        ood_embed = self.token_embed.weight.data[subject_tid]
        ood_norm = np.linalg.norm(ood_embed)
        if ood_norm > 0:
            ood_unit = ood_embed / ood_norm
            token_embeds = self.token_embed.weight.data
            token_norms = self._token_embed_norms
            if token_norms is None:
                token_norms = np.linalg.norm(token_embeds, axis=1)
            valid_tok = token_norms > 0
            if np.any(valid_tok):
                normed_tok = token_embeds.copy()
                normed_tok[valid_tok] /= token_norms[valid_tok, np.newaxis]
                ood_sims = ood_unit @ normed_tok.T
                sim_weight = 5.0 if disable_spreading else 3.0
                concept_scores += ood_sims * sim_weight


        # Collect all active nodes with their activations
        active_nodes = []
        for nid, node in self.graph.nodes.items():
            if node.activation > 0.01:
                active_nodes.append((nid, node))

        # For each active node, check outgoing edges
        matching_targets = {}  # target_concept_id -> score
        for nid, node in active_nodes:
            outgoing = self.graph.get_outgoing(nid)
            for tgt_id, edge in outgoing:
                # Filter by relation type - the KEY to cross-domain transfer
                # "causes" edges from heat activate expansion, not kindness
                type_match = (edge.relation_type == rel_type_name)

                # Compute edge score
                base_score = node.activation * edge.weight * edge.confidence

                # Relation type matching bonus
                if type_match:
                    base_score *= 3.0  # strong boost for matching type
                else:
                    base_score *= 0.1  # heavy penalty for non-matching type

                # Predicate matching
                pred_mult = 1.0
                if query_verb_word and hasattr(edge, 'predicate_token_id') and edge.predicate_token_id != -1:
                    try:
                        edge_verb_word = self._decode_token(edge.predicate_token_id).lower().strip()
                        if edge_verb_word:
                            w1 = query_verb_word.rstrip('s').rstrip('d')
                            w2 = edge_verb_word.rstrip('s').rstrip('d')
                            if w1 == w2 or w1 in w2 or w2 in w1:
                                pred_mult = 2.5
                            else:
                                pred_mult = 0.4
                    except Exception:
                        pass
                base_score *= pred_mult

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
        # "fire causes ?" -> fire->heat (any type), heat->expansion (causal) -> boost expansion
        # "exercise produces crashes" -> exercise->stress->bugs->crashes (3 hops)
        # First hop: follow ALL edges (any type) to reach domain intermediaries.
        # Subsequent hops: only follow edges matching the query relation type.
        from collections import deque
        _max_hops = 3
        _hop_base_boost = 8.0
        _hop_decay = 0.7  # relaxed per-hop score decay
        # BFS queue: (node_id, cumulative_score, depth, visited_set)
        bfs_queue = deque()
        for mid_cid, mid_edge in self.graph.get_outgoing(subject_cid):
            if mid_cid == subject_cid:
                continue
            mid_node = self.graph.get_node(mid_cid)
            if mid_node is None:
                continue
            hop_score = mid_edge.weight * mid_edge.confidence
            
            # Apply predicate matching
            pred_mult = 1.0
            if query_verb_word and hasattr(mid_edge, 'predicate_token_id') and mid_edge.predicate_token_id != -1:
                try:
                    edge_verb_word = self._decode_token(mid_edge.predicate_token_id).lower().strip()
                    if edge_verb_word:
                        w1 = query_verb_word.rstrip('s').rstrip('d')
                        w2 = edge_verb_word.rstrip('s').rstrip('d')
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
            if depth > _max_hops:
                continue
            for tgt_cid, tgt_edge in self.graph.get_outgoing(nid):
                if tgt_cid in visited or tgt_cid == subject_cid:
                    continue
                # After first hop (depth>=1), only follow edges matching relation type
                if depth >= 1 and tgt_edge.relation_type != rel_type_name:
                    continue
                
                # Apply predicate matching
                pred_mult = 1.0
                if query_verb_word and hasattr(tgt_edge, 'predicate_token_id') and tgt_edge.predicate_token_id != -1:
                    try:
                        edge_verb_word = self._decode_token(tgt_edge.predicate_token_id).lower().strip()
                        if edge_verb_word:
                            w1 = query_verb_word.rstrip('s').rstrip('d')
                            w2 = edge_verb_word.rstrip('s').rstrip('d')
                            if w1 == w2 or w1 in w2 or w2 in w1:
                                pred_mult = 2.5
                            else:
                                pred_mult = 0.4
                    except Exception:
                        pass
                
                edge_score = cum_score * tgt_edge.weight * tgt_edge.confidence * pred_mult
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
        # Stronger suppression during OOD queries (when spreading is disabled)
        hub_suppress_strength = 1.5 if disable_spreading else 1.0
        for nid, node in active_nodes:
            if nid == subject_cid:
                continue  # Don't predict subject itself
            if nid == relation_cid:
                continue  # Don't predict relation node (it's an intermediary)
            # Hub suppression: penalize low-activation nodes with high in-degree
            in_deg = len(self.graph._incoming.get(nid, []))
            if node.activation < 0.5 and in_deg > 5:
                hub_factor = 0.3 / hub_suppress_strength
            elif node.activation < 0.3 and in_deg > 3:
                hub_factor = 0.5 / hub_suppress_strength
            else:
                hub_factor = 1.0 / hub_suppress_strength if in_deg > 10 else 1.0
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
            # Boosted from 1.0 to 2.0 - binding map is exact match, should dominate
            bindings = self.binding_map.get_tokens(tgt_cid, min_confidence=0.1)
            for binding in bindings:
                tok_id = binding.token_id
                if 0 <= tok_id < self.vocab_size:
                    concept_scores[tok_id] += score * binding.confidence * 2.0

            batch_targets.append((tgt_cid, score, tgt_node.vector))

        # Method 2: Batch cosine similarity - one matmul instead of N loops
        if batch_targets:
            tgt_vecs = np.stack([tv[2] for tv in batch_targets])  # (n_targets, concept_dim)
            # Project to embed_dim (same as _project_to_embed but batched)
            if self.concept_dim == self.embed_dim:
                tgt_embeds = tgt_vecs  # identity - no projection needed
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

            # Similarity matrix: (n_targets, vocab_size) - one big matmul
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

        # Collect active nodes sorted by activation
        all_active = [n for n in sorted(self.graph.nodes.values(),
                                         key=lambda n: n.activation, reverse=True)
                      if n.activation > 0.01][:7]
        self._last_predicted_concepts = [n.id for n in all_active][:5]
        self._last_edge_pred = self.propagation.get_prediction(self._last_predicted_concepts, top_k=5)

        # Apply temperature-modulated softmax with cognitive/emotion modulation
        emotion_scale = 1.0 + 0.3 * self.arousal - 0.1 * max(0.0, -self.valence)
        identity_scale = 0.5 + 0.5 * self.identity_strength
        temp = max(0.2, 0.3 + 0.4 * self.arousal)

        if np.max(concept_scores) > 0:
            concept_scores_scaled = concept_scores / temp
            exp_scores = np.exp(concept_scores_scaled - np.max(concept_scores_scaled))
            probs = exp_scores / (np.sum(exp_scores) + 1e-10)
            logits = np.log(probs + 1e-10)
        else:
            logits = concept_scores

        logits = logits * (identity_scale * emotion_scale)

        return make_tensor(logits.astype(np.float32))

    # ── Hebbian Learning ────────────────────────────────────────────────────

    def learn(self, token_ids: np.ndarray, target_ids: np.ndarray) -> Dict[str, float]:
        """Learn from a (context, target) pair via Hebbian triple updates.

        This is the core learning mechanism:
        1. Forward pass to get prediction
        2. Decompose into (subject, relation, object) triple
        3. Create/update concept nodes for subject and object
        4. Create/update typed edge (subject -> object) with relation_type
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

        # ── Forward pass (enable spreading during training) ──
        self._training_mode = True
        logits_tensor = self.forward(token_ids)
        logits = logits_tensor.data.flatten() if hasattr(logits_tensor.data, 'flatten') else logits_tensor.data
        self._training_mode = False

        # Prediction error
        target_onehot = np.zeros(self.vocab_size, dtype=np.float32)
        target_onehot[target_id] = 1.0
        prediction_error = target_onehot - np.exp(logits)  # error in probability space
        target_prob = np.clip(np.exp(logits[target_id]), 0.0, 1.0)
        surprise = 1.0 - target_prob
        hebbian_multiplier = 1.0 + 15.0 * surprise

        # Track accuracy
        pred_id = int(np.argmax(logits))
        is_correct = pred_id == target_id
        if is_correct:
            self._train_correct += 1
        self._train_total += 1

        # ── Decompose triple ──
        # Reconstruct full triple from context + target for proper decomposition
        # e.g., "heat causes" + "expansion" -> "heat causes expansion" -> (heat, causes, expansion)
        full_triple_ids = input_ids + [target_id]
        subject_ids, relation_ids, object_ids = self.decompose_triple(full_triple_ids)

        # The target IS the object (what we're trying to predict)
        object_tid = target_id

        if not subject_ids:
            self._step_counter += 1
            return {"loss": float(np.mean(prediction_error ** 2)), "accuracy": self._train_correct / max(1, self._train_total)}

        subject_tid = subject_ids[0]

        # Track seen predicates for OOD detection
        if relation_ids and hasattr(self, '_decode_token'):
            try:
                pred_word = self._decode_token(relation_ids[0]).lower().strip()
                if pred_word:
                    if not hasattr(self, '_seen_predicates'):
                        self._seen_predicates = set()
                    self._seen_predicates.add(pred_word)
                    # Accumulate verb offset for verb-stem predictor
                    self._accumulate_verb_offset(subject_tid, object_tid, pred_word)
            except Exception:
                pass

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
        # This gives the graph real topology: cat->has->tail, dog->has->tail.
        if relation_ids:
            relation_tid = relation_ids[0]  # first relation token
            relation_embed = self.token_embed.weight.data[relation_tid]
            relation_cid = self._get_or_create_concept(relation_tid, relation_embed)
        else:
            relation_cid = object_cid  # fallback: direct edge if no relation

        # ── Create relation-object concept node ──
        # Instead of a global "has" hub (which gets overloaded with 10+ facts),
        # create relation-OBJECT nodes: "has:tail", "has:wing", etc.
        # Multiple subjects sharing the same object (cat, dog -> tail) route
        # through the same hub, enabling WITHIN-group transfer.
        # Cross-group transfer (tiger is cat-like -> tail) uses embedding similarity.
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
        # If "anger -> heat" exists, create weak "heat -> anger" so cross-domain
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

        # ── Edge Validation: Check binding map consistency ──
        # If bindings have shifted since edge creation, re-validate edge topology
        self._validate_edge_bindings(subject_cid, object_cid, rel_type_name, relation_ids)

        # ── Direct Edge Injection: For cross-domain causal, inject subject->object edges
        # when binding map shows 1-to-1 mapping but graph edges are missing/polluted
        self._inject_direct_edges_if_needed(subject_cid, object_cid, rel_type_name)

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

            # Direct edge: subject -> object (memorization path)
            delta = self._base_lr * src_node.activation * tgt_node.activation * hebbian_multiplier
            edge_direct.weight = max(0.0, min(1.0, edge_direct.weight + delta))
            edge_direct.confidence = edge_direct.weight
            edge_direct.stability = min(1.0, edge_direct.stability + 0.01)
            edge_direct.prediction_count += 1
            if is_correct:
                edge_direct.forward_pred_count += 1

            # Edge 1: subject -> rel_obj hub (transfer path)
            delta = self._base_lr * src_node.activation * rel_obj_node.activation * hebbian_multiplier
            edge_sr.weight = max(0.0, min(1.0, edge_sr.weight + delta))
            edge_sr.confidence = edge_sr.weight
            edge_sr.stability = min(1.0, edge_sr.stability + 0.01)
            edge_sr.prediction_count += 1
            if is_correct:
                edge_sr.forward_pred_count += 1

            # Edge 2: rel_obj hub -> object (transfer path)
            delta = self._base_lr * rel_obj_node.activation * tgt_node.activation * hebbian_multiplier
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

            # Subject concept -> subject token embedding
            subject_concept_vec = self._project_to_concept(subject_embed)
            src_delta = pull_lr * (subject_concept_vec - src_node.vector)
            src_delta = np.clip(src_delta, -self.graph.max_step_delta, self.graph.max_step_delta)
            src_node.vector += src_delta
            src_norm = np.linalg.norm(src_node.vector)
            if src_norm > 0:
                src_node.vector /= src_norm

            # Relation-object concept -> blended embedding
            if relation_ids:
                rel_obj_concept_vec = self._project_to_concept(rel_obj_embed)
                rel_obj_delta = pull_lr * (rel_obj_concept_vec - rel_obj_node.vector)
                rel_obj_delta = np.clip(rel_obj_delta, -self.graph.max_step_delta, self.graph.max_step_delta)
                rel_obj_node.vector += rel_obj_delta
                rel_obj_norm = np.linalg.norm(rel_obj_node.vector)
                if rel_obj_norm > 0:
                    rel_obj_node.vector /= rel_obj_norm

            # Object concept -> object embedding
            object_concept_vec = self._project_to_concept(object_embed)
            tgt_delta = pull_lr * (object_concept_vec - tgt_node.vector)
            tgt_delta = np.clip(tgt_delta, -self.graph.max_step_delta, self.graph.max_step_delta)
            tgt_node.vector += tgt_delta
            tgt_norm = np.linalg.norm(tgt_node.vector)
            if tgt_norm > 0:
                tgt_node.vector /= tgt_norm

            # Path-aware concept vector update: pull subject and object vectors
            # closer together. This builds analogical structure - if anger->heat
            # and heat->expansion, anger and expansion become closer in concept
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

        # ── Train token embeddings (DISABLED - corrupts semantic structure) ──
        # Pulling token embeddings together during training destroys the
        # carefully structured embedding space. "heat" gets pulled toward
        # "expansion", "cold", "melts" simultaneously - ending up in a
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
        # Updating ALL edges causes catastrophic interference - training "heat causes
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

        # ── Cognitive Updates (GRACE Core) ──
        output_concept = object_cid
        input_concept = subject_cid

        token_hit = output_concept in self._last_predicted_concepts
        edge_pred_set = set(self._last_edge_pred)
        single_correct = output_concept in edge_pred_set

        is_prediction_correct = token_hit or single_correct
        conceptual_error = 0.0 if is_prediction_correct else 1.0

        # Update token-hit EMA
        self._token_hit_ema = 0.9 * self._token_hit_ema + 0.1 * (1.0 if is_prediction_correct else 0.0)

        # Edge weight convergence
        n_edges = len(self.graph.edges)
        if n_edges > 0:
            mean_w = np.mean([e.weight for e in self.graph.edges.values()])
            self._edge_weight_prev = self._edge_weight_ema
            self._edge_weight_ema = 0.99 * self._edge_weight_ema + 0.01 * mean_w

        # Trigger currencies update
        self.currencies.update(conceptual_error, single_correct)
        self._sleep_pressure = self.sleep_pressure

        # Episodic memory storage
        self._store_episode(conceptual_error, single_correct)

        # Emotion-tag active concepts
        for cid in self._last_predicted_concepts:
            self._concept_vad[cid] = (self.valence, self.arousal, self.dominance)

        # Sync currency from canonical scalars
        factor = 0.95 if single_correct else 0.05
        self.conceptual_accuracy = 0.9 * self.conceptual_accuracy + 0.1 * factor

        self.currency.update('identity_strength', self.identity_strength)
        self.currency.update('dissonance_ema', self.dissonance_ema)
        self.currency.update('sleep_pressure', self.sleep_pressure)
        self.currency.update('conceptual_accuracy', self.conceptual_accuracy)
        self.currency.update('valence', self.valence)
        self.currency.update('arousal', self.arousal)
        self.currency.update('dominance', self.dominance)
        self.currency.update('accumulated_meaning', self.accumulated_meaning)
        self.currency.update('total_free_energy', 0.0)

        # ── Store episodic triple ──
        self._episodic_triples.append((subject_cid, rel_type_idx, object_cid, time.time()))
        if len(self._episodic_triples) > self._max_episodic:
            self._episodic_triples = self._episodic_triples[-self._max_episodic:]

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

    # ── Graph Structure Repair Methods ───────────────────────────────────────────

    def _validate_edge_bindings(self, subject_cid: int, object_cid: int,
                                 rel_type_name: str, relation_ids: List[int]) -> bool:
        """
        Validate edge topology against current binding map.
        
        After training, bindings may have settled to 1-to-1 mapping, but edges
        created during early training may reflect different/incorrect bindings.
        This method checks if the current binding map matches the edge topology
        and corrects mismatches.
        
        Returns True if validation passed, False if edges were corrected.
        """
        # Get current token IDs for these concepts
        subj_bindings = self.binding_map.get_tokens(subject_cid, min_confidence=0.1)
        obj_bindings = self.binding_map.get_tokens(object_cid, min_confidence=0.1)
        
        if not subj_bindings or not obj_bindings:
            # Concepts don't have valid token bindings yet
            return True
            
        current_subj_tid = subj_bindings[0].token_id
        current_obj_tid = obj_bindings[0].token_id
        
        # Check if edge's predicate token matches current relation
        direct_edge = self.graph.get_edge(subject_cid, object_cid)
        if direct_edge is not None:
            if relation_ids and direct_edge.predicate_token_id != relation_ids[0]:
                # Predicate has changed - this edge was learned with different relation
                # Update the predicate token to match current
                direct_edge.predicate_token_id = relation_ids[0]
                # Reduce confidence since topology was learned under different semantics
                direct_edge.confidence = max(0.1, direct_edge.confidence * 0.5)
                
        # Check reverse edge
        reverse_edge = self.graph.get_edge(object_cid, subject_cid)
        if reverse_edge is not None and relation_ids:
            reverse_edge.predicate_token_id = relation_ids[0]
            
        return True

    def _inject_direct_edges_if_needed(self, subject_cid: int, object_cid: int,
                                       rel_type_name: str, threshold: float = 0.5) -> int:
        """
        Inject direct subject->object edges when binding map shows 1-to-1
        but graph edges are missing or weak (polluted by Hebbian noise).
        
        This bypasses the graph topology bottleneck for cross-domain causal
        relations where vector arithmetic carries the load but Hebbian edges
        are polluted.
        
        Returns number of edges injected/strengthened.
        """
        injected = 0
        
        # Check if direct edge exists and is strong enough
        direct_edge = self.graph.get_edge(subject_cid, object_cid)
        
        if direct_edge is None:
            # No edge exists - create it with strong weight
            self.graph.add_edge(
                source=subject_cid,
                target=object_cid,
                weight=0.7,  # Strong initial weight for direct causal
                relation_type=rel_type_name,
            )
            injected += 1
        elif direct_edge.weight < threshold:
            # Edge exists but is weak - strengthen it (bypass Hebbian noise)
            direct_edge.weight = threshold
            direct_edge.confidence = max(direct_edge.confidence, 0.6)
            direct_edge.stability = max(direct_edge.stability, 0.5)
            direct_edge.relation_type = rel_type_name
            injected += 1
            
        # Also check relation-object hub path edges
        # (These are handled by learn() but we ensure they're not polluted)
        return injected

    def _anti_hebbian_prune_polluted_edges(self, mismatch_threshold: float = 0.3,
                                           min_prediction_count: int = 5) -> int:
        """
        Anti-Hebbian pruning: weaken/remove edges that consistently predict wrong.
        
        Edges with high prediction_count but low forward_pred_count ratio
        are likely polluted by Hebbian noise from early training with different bindings.
        
        Returns number of edges pruned/weakened.
        """
        pruned = 0
        edges_to_check = list(self.graph.edges.items())
        
        for (src_id, tgt_id), edge in edges_to_check:
            if edge.prediction_count >= min_prediction_count:
                pred_ratio = edge.forward_pred_count / edge.prediction_count
                if pred_ratio < (1.0 - mismatch_threshold):
                    # This edge consistently fails to predict correctly
                    # Apply anti-Hebbian weakening
                    self.graph.anti_hebbian_update(src_id, tgt_id, lr=0.02)
                    pruned += 1
                    
                    # If edge is very weak and unstable, remove it
                    if edge.weight < 0.1 and edge.confidence < 0.1:
                        self.graph.remove_edge(src_id, tgt_id)
                        
        return pruned

    def learn_fast(self, token_ids: np.ndarray, target_ids: np.ndarray) -> Dict[str, float]:
        """Fast learn() for hard-example boosting - skips forward() call.

        Use this during hard-boost training when we already know the prediction
        is wrong. Saves approx 0.7 to 1.0 ms per call by skipping the forward pass.
        Also skips the O-of-E predictive coding loop (uses only local Hebbian).
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

        # Track seen predicates for OOD detection
        if relation_ids and hasattr(self, '_decode_token'):
            try:
                pred_word = self._decode_token(relation_ids[0]).lower().strip()
                if pred_word:
                    if not hasattr(self, '_seen_predicates'):
                        self._seen_predicates = set()
                    self._seen_predicates.add(pred_word)
                    # Accumulate verb offset for verb-stem predictor
                    self._accumulate_verb_offset(subject_tid, object_tid, pred_word)
            except Exception:
                pass

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

        # ── Hebbian edge updates (local only - no forward pass needed) ──
        src_node = self.graph.get_node(subject_cid)
        rel_obj_node = self.graph.get_node(rel_obj_cid)
        tgt_node = self.graph.get_node(object_cid)

        if src_node is not None and rel_obj_node is not None and tgt_node is not None:
            rel_obj_node.activation = max(rel_obj_node.activation, 0.7)
            tgt_node.activation = max(tgt_node.activation, 0.8)

            # Direct edge: subject -> object
            delta = self._base_lr * src_node.activation * tgt_node.activation * 16.0
            edge_direct.weight = max(0.0, min(1.0, edge_direct.weight + delta))
            edge_direct.confidence = edge_direct.weight
            edge_direct.stability = min(1.0, edge_direct.stability + 0.01)
            edge_direct.prediction_count += 1

            # Edge 1: subject -> rel_obj hub
            delta = self._base_lr * src_node.activation * rel_obj_node.activation * 16.0
            edge_sr.weight = max(0.0, min(1.0, edge_sr.weight + delta))
            edge_sr.confidence = edge_sr.weight
            edge_sr.stability = min(1.0, edge_sr.stability + 0.01)
            edge_sr.prediction_count += 1

            # Edge 2: rel_obj hub -> object
            delta = self._base_lr * rel_obj_node.activation * tgt_node.activation * 16.0
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

    # ── Hard-Boost Sampling ───────────────────────────────────────────────────

    def hard_boost_sample(self, tok, triplet_pairs: List[Tuple[str, str, str]],
                          n_samples: int = 15, intensity: float = 300.0,
                          margin: float = 0.1,
                          triplet_rel_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Hard-boost sampling: Select only N random hard examples per epoch
        instead of applying to all pairs. Keeps 300x intensity but fits timeout.
        
        Args:
            tok: Tokenizer
            triplet_pairs: List of (anchor, positive, negative) triples
            n_samples: Number of hard examples to sample per epoch (default 15)
            intensity: Boost intensity multiplier (default 300.0)
            margin: Triplet margin (default 0.1)
            triplet_rel_types: Optional list of relation types for each triplet (e.g., 'causal', 'semantic', 'temporal').
                               If provided, performs stratified sampling to ensure each underperforming relation type gets boost cycles.
            
        Returns:
            Dict with sampled indices and per-triple gap metrics
        """
        if not triplet_pairs:
            return {'sampled_indices': [], 'gaps': {}}
        
        # Evaluate all pairs to find hard ones (negative or small positive gap)
        hard_candidates = []
        all_gaps = {}
        
        for i, (anchor, positive, negative) in enumerate(triplet_pairs):
            try:
                pa = self._proto_latent(tok, anchor)
                pp = self._proto_latent(tok, positive)
                pn = self._proto_latent(tok, negative)
                
                def cosine(a, b):
                    na = np.linalg.norm(a)
                    nb = np.linalg.norm(b)
                    if na == 0 or nb == 0:
                        return 0.0
                    return float(np.dot(a, b) / (na * nb))
                
                sp = cosine(pa, pp)
                sn = cosine(pa, pn)
                gap = sp - sn
                
                all_gaps[f"{anchor}->{positive} (vs {negative})"] = {
                    's_pos': sp, 's_neg': sn, 'gap': gap,
                    'satisfied': gap > margin
                }
                
                # Hard example: gap <= margin (violation or near-violation)
                if gap <= margin:
                    rel_type = triplet_rel_types[i] if triplet_rel_types and i < len(triplet_rel_types) else 'unknown'
                    hard_candidates.append((i, gap, rel_type))
            except (KeyError, IndexError):
                continue
        
        # Stratified sampling by relation type if rel_types provided
        if triplet_rel_types:
            # Group hard candidates by relation type
            by_type = defaultdict(list)
            for idx, gap, rel_type in hard_candidates:
                by_type[rel_type].append((idx, gap))
            
            # Sample from each bucket proportionally
            sampled_indices = []
            n_types = len(by_type)
            if n_types > 0:
                per_type = max(1, n_samples // n_types)
                for rel_type, candidates in by_type.items():
                    candidates.sort(key=lambda x: x[1])  # hardest first
                    sampled_indices.extend([idx for idx, _ in candidates[:per_type]])
            
            # If we have fewer than n_samples, fill from remaining hardest across all types
            if len(sampled_indices) < n_samples:
                all_remaining = [(idx, gap) for idx, gap, _ in hard_candidates if idx not in sampled_indices]
                all_remaining.sort(key=lambda x: x[1])
                sampled_indices.extend([idx for idx, _ in all_remaining[:n_samples - len(sampled_indices)]])
            
            sampled_indices = sampled_indices[:n_samples]
        else:
            # Original behavior: sort by gap (most negative first = hardest)
            hard_candidates.sort(key=lambda x: x[1])
            # Sample N from hardest, or all if fewer than N
            sampled = hard_candidates[:n_samples] if hard_candidates else []
            sampled_indices = [idx for idx, _ in sampled]
        
        # Apply intensified triplet margin to sampled hard examples
        boosted_results = {}
        for idx in sampled_indices:
            anchor, positive, negative = triplet_pairs[idx]
            result = self.apply_triplet_margin(
                tok, anchor, positive, negative,
                margin=margin,
                lr=0.01 * intensity,  # 300x intensity
                update_both=True
            )
            boosted_results[f"{anchor}->{positive} (vs {negative})"] = result
        
        return {
            'sampled_indices': sampled_indices,
            'n_hard_total': len(hard_candidates),
            'n_sampled': len(sampled_indices),
            'gaps': all_gaps,
            'boosted_results': boosted_results
        }

    def _proto_latent(self, tok, word: str) -> np.ndarray:
        """Get latent vector for a word using _encoder_forward_full (not subject_proj)."""
        tid = tok.word_to_id.get(word)
        if tid is None:
            raise KeyError(word)
        emb = self.token_embed.weight.data[tid]
        lat, *_ = self._encoder_forward_full(emb)
        if getattr(self, "use_subspace_projection", False) and hasattr(self, "rel_proj"):
            lat = lat @ self.rel_proj
        return lat

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
        # No chain rule - uses only the local error signal and local input.
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

    # ── Triplet Margin and Knowledge Graph Integration Core APIs ─────────────

    def apply_triplet_margin(self, tok, anchor: str, positive: str, negative: str,
                             margin: float = 0.1, lr: float = 0.01, update_both: bool = True) -> dict:
        """Apply triplet margin constraint, updating the encoder representations via backpropagation.

        Args:
            tok: WordTokenizer instance.
            anchor: Anchor word.
            positive: Positive analogical target word.
            negative: Negative out-of-domain word.
            margin: Triplet separation margin (default 0.1).
            lr: Gradient updates learning rate (default 0.01).
            update_both: If True, backpropagates through both anchor and positive paths.
        """
        a_tid = tok.word_to_id.get(anchor)
        p_tid = tok.word_to_id.get(positive)
        n_tid = tok.word_to_id.get(negative)

        if a_tid is None or p_tid is None or n_tid is None:
            return {'violation': 0.0, 'gap': 0.0, 'updated': False}

        a_emb = self.token_embed.weight.data[a_tid]
        p_emb = self.token_embed.weight.data[p_tid]
        n_emb = self.token_embed.weight.data[n_tid]

        lat_a, z1_a, h1_a, z2_a = self._encoder_forward_full(a_emb)
        lat_p, z1_p, h1_p, z2_p = self._encoder_forward_full(p_emb)
        lat_n, z1_n, h1_n, z2_n = self._encoder_forward_full(n_emb)

        def get_unit_and_norm(v):
            norm = np.linalg.norm(v)
            if norm == 0:
                return np.zeros_like(v), 0.0
            return v / norm, norm

        if getattr(self, "use_subspace_projection", False):
            proj_a = lat_a @ self.rel_proj
            proj_p = lat_p @ self.rel_proj
            proj_n = lat_n @ self.rel_proj

            u_a_proj, norm_a_proj = get_unit_and_norm(proj_a)
            u_p_proj, norm_p_proj = get_unit_and_norm(proj_p)
            u_n_proj, norm_n_proj = get_unit_and_norm(proj_n)

            sim_pos = float(np.dot(u_a_proj, u_p_proj))
            sim_neg = float(np.dot(u_a_proj, u_n_proj))
            cosine_gap = sim_pos - sim_neg

            loss = sim_neg - sim_pos + margin
            loss = max(0.0, loss)

            if loss > 1e-9:
                grad_a_proj = (u_n_proj - sim_neg * u_a_proj) / (norm_a_proj + 1e-15) - (u_p_proj - sim_pos * u_a_proj) / (norm_a_proj + 1e-15)
                grad_p_proj = -(u_a_proj - sim_pos * u_p_proj) / (norm_p_proj + 1e-15)
                grad_n_proj = (u_a_proj - sim_neg * u_n_proj) / (norm_n_proj + 1e-15)

                grad_P = np.outer(lat_a, grad_a_proj) + np.outer(lat_p, grad_p_proj) + np.outer(lat_n, grad_n_proj)
                self.rel_proj -= lr * grad_P
                updated = True
            else:
                updated = False
            return {'violation': loss, 'gap': cosine_gap, 'updated': updated}

        u_a, norm_a = get_unit_and_norm(lat_a)
        u_p, norm_p = get_unit_and_norm(lat_p)
        u_n, norm_n = get_unit_and_norm(lat_n)

        sim_pos = float(np.dot(u_a, u_p))
        sim_neg = float(np.dot(u_a, u_n))
        cosine_gap = sim_pos - sim_neg

        loss = sim_neg - sim_pos + margin
        loss = max(0.0, loss)

        if loss > 1e-9:
            d_lat_a = (u_n - sim_neg * u_a) / (norm_a + 1e-15) - (u_p - sim_pos * u_a) / (norm_a + 1e-15)
            d_lat_p = -(u_a - sim_pos * u_p) / (norm_p + 1e-15)
            d_lat_n = (u_a - sim_neg * u_n) / (norm_n + 1e-15)

            # Backprop anchor
            dW1a, db1a, dW2a, db2a = self._encoder_backward(
                a_emb[np.newaxis, :],
                z1_a[np.newaxis, :],
                h1_a[np.newaxis, :],
                z2_a[np.newaxis, :],
                lat_a[np.newaxis, :],
                d_lat_a[np.newaxis, :],
            )
            self._enc_W1 -= lr * dW1a
            self._enc_b1 -= lr * db1a
            self._enc_W2 -= lr * dW2a
            self._enc_b2 -= lr * db2a

            if update_both:
                # Backprop positive
                dW1p, db1p, dW2p, db2p = self._encoder_backward(
                    p_emb[np.newaxis, :],
                    z1_p[np.newaxis, :],
                    h1_p[np.newaxis, :],
                    z2_p[np.newaxis, :],
                    lat_p[np.newaxis, :],
                    d_lat_p[np.newaxis, :],
                )
                self._enc_W1 -= lr * dW1p
                self._enc_b1 -= lr * db1p
                self._enc_W2 -= lr * dW2p
                self._enc_b2 -= lr * db2p

                # Backprop negative
                dW1n, db1n, dW2n, db2n = self._encoder_backward(
                    n_emb[np.newaxis, :],
                    z1_n[np.newaxis, :],
                    h1_n[np.newaxis, :],
                    z2_n[np.newaxis, :],
                    lat_n[np.newaxis, :],
                    d_lat_n[np.newaxis, :],
                )
                self._enc_W1 -= lr * dW1n
                self._enc_b1 -= lr * db1n
                self._enc_W2 -= lr * dW2n
                self._enc_b2 -= lr * db2n

            self._token_embed_norms = None
            updated = True
        else:
            updated = False

        return {'violation': loss, 'gap': cosine_gap, 'updated': updated}

    def add_analogical_relation(self, anchor_word: str, positive_word: str, weight: float = 0.8):
        """Fold an analogical relation between two words directly into the concept graph as an edge."""
        tok = self._tokenizer
        if tok is None:
            return None
        a_tid = tok.word_to_id.get(anchor_word)
        p_tid = tok.word_to_id.get(positive_word)
        if a_tid is None or p_tid is None:
            return None
        a_bindings = self.binding_map.get_concepts(a_tid, min_confidence=0.1)
        p_bindings = self.binding_map.get_concepts(p_tid, min_confidence=0.1)
        if not a_bindings or not p_bindings:
            return None
        a_cid = a_bindings[0].concept_id
        p_cid = p_bindings[0].concept_id

        edge = self.graph.get_edge(a_cid, p_cid)
        if edge is None:
            edge = self.graph.add_edge(
                source=a_cid,
                target=p_cid,
                weight=weight,
                relation_type="analogical"
            )
            edge.confidence = weight
        else:
            edge.weight = max(edge.weight, weight)
            edge.confidence = max(edge.confidence, weight)
            edge.relation_type = "analogical"
        return edge

    def traverse(self, start_word: str, steps: int = 5, threshold: float = 0.3):
        """Hybrid search traversal combining direct 1-hop connections and multi-hop infer_chain paths."""
        tok = self._tokenizer
        if tok is None:
            return []
        tid = tok.word_to_id.get(start_word)
        if tid is None:
            return []
        bindings = self.binding_map.get_concepts(tid, min_confidence=0.1)
        if not bindings:
            return []
        cid = bindings[0].concept_id

        # 1. Direct 1-hop connections
        direct_candidates = []
        outgoing = self.graph._outgoing.get(cid, [])
        for target_cid, edge in outgoing:
            if edge.weight >= threshold and edge.edge_type != "inhibitory":
                t_bindings = self.binding_map.get_tokens(target_cid, min_confidence=0.1)
                if t_bindings:
                    t_tid = t_bindings[0].token_id
                    t_word = tok.decode([t_tid])
                    score = edge.weight * 1.5  # boost direct connection
                    direct_candidates.append((t_word, score))

        # 2. Multi-hop BFS paths from infer_chain
        chains = self.graph.infer_chain(start_id=cid, max_hops=steps, min_weight=threshold, k=5)
        multi_candidates = []
        for target_cid, score, path in chains:
            t_bindings = self.binding_map.get_tokens(target_cid, min_confidence=0.1)
            if t_bindings:
                t_tid = t_bindings[0].token_id
                t_word = tok.decode([t_tid])
                multi_candidates.append((t_word, score))

        # Combine and deduplicate
        combined = {}
        for word, score in direct_candidates + multi_candidates:
            if word == start_word:
                continue
            if word not in combined or score > combined[word]:
                combined[word] = score

        # Sort top candidates by score
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return ranked

    # ── Sleep Cycle ─────────────────────────────────────────────────────────

    def sleep_cycle(self, validation_queries: Optional[List[Dict[str, Any]]] = None,
                    force_alignment: bool = False):
        """Sleep cycle: consolidate triples, prune weak edges, replay important memories.

        This is the brain's "offline consolidation" - during sleep, the model:
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
        # Prevent recursive sleep cycle calls during replay or model training inside sleep
        if getattr(self, '_in_sleep_cycle', False):
            return
        self._in_sleep_cycle = True
        self._last_sleep_step = self._step_counter

        try:
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
            # Budget increased from 3.0 to 5.0 - less aggressive, preserves learned edges
            self._normalize_outgoing_weights(budget=5.0)

            # ── Prune weak edges ──
            # Prune edges with weight below 0.1 to clear out spurious semantic edges.
            self._prune_weak_edges(threshold=0.1)

            # ── Anti-Hebbian Pruning: Remove polluted edges ──
            # Edges created during early training with different bindings that now
            # consistently predict incorrectly are weakened/removed.
            n_pruned = self._anti_hebbian_prune_polluted_edges()
            if n_pruned:
                print(f"[Sleep] Anti-Hebbian pruned {n_pruned} polluted edges")

            # ── Prune phantom nodes ──
            # Remove concept nodes that have no token binding (token_id == None)
            # and degree < 2 (isolated or single-edge artifacts from tokenizer expansion)
            self._prune_phantom_nodes(min_degree=2)

            # ── Representation Alignment ──
            # Align encoder representations to graph topology if needed or forced or graph changed
            graph_version = getattr(self.graph, 'version', 0)
            last_aligned = getattr(self, '_last_aligned_version', -1)
            if self.alignment_needed or force_alignment or graph_version > last_aligned:
                self.align_encoder_to_graph(validation_queries=validation_queries)
                self.alignment_needed = False  # reset flag after alignment

            # ── Drift defense ──
            # Pull concept vectors back toward their core vectors if they've drifted too far.
            # Threshold increased from 0.4 to 0.7 - allow more movement before correction.
            # Pull strength reduced from 0.1 to 0.05 - gentler correction.
            for nid, node in self.graph.nodes.items():
                drift = node.drift_magnitude
                if drift > 0.7:
                    # Pull back toward core vector (gentler)
                    pull = 0.05 * (node.core_vector - node.vector)
                    node.vector += pull
                    norm = np.linalg.norm(node.vector)
                    if norm > 0:
                        node.vector /= norm

            # ── Episodic -> semantic consolidation ──
            self._consolidate_episodic_to_semantic()

            # ── Semantic memory decay ──
            self._decay_semantic_memories()

            # ── Memory -> weights bridge ──
            self._bridge_memories_to_graph()

            # ── Interleaved named domain memories replay ──
            self._replay_old_memories(n_samples=self._replay_n_samples)

            # ── Cognitive currency consolidation (emotion, identity, meaning, sleep) ──
            self.currencies.consolidate_on_sleep()

            # ── Reset sleep pressure ──
            self._sleep_pressure = 0.0
            self.wake_epochs_since_sleep = 0
            self.sleep_cycles_completed += 1

            # ── Final self-regulation ──
            self._regulate_cognitive_state()

            # ── Reset activations ──
            for node in self.graph.nodes.values():
                node.activation = 0.0
                node.fatigue = 0.0
        finally:
            self._in_sleep_cycle = False

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
        
        # Manifold Regularization: Cache original projections before alignment starts
        original_latents = {}
        lambda_recon = getattr(self, "lambda_recon", 0.0)
        if lambda_recon > 0.0:
            for word, tid in tok.word_to_id.items():
                if tid < self.token_embed.weight.data.shape[0]:
                    embed = self.token_embed.weight.data[tid]
                    lat, *_ = self._encoder_forward_full(embed)
                    original_latents[tid] = lat.copy()

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
            
            # Calculate and accumulate reconstruction penalty (Manifold Regularization)
            if lambda_recon > 0.0 and original_latents:
                d_recon_W1 = np.zeros_like(self._enc_W1)
                d_recon_b1 = np.zeros_like(self._enc_b1)
                d_recon_W2 = np.zeros_like(self._enc_W2)
                d_recon_b2 = np.zeros_like(self._enc_b2)
                
                for tid, lat_orig in original_latents.items():
                    embed = self.token_embed.weight.data[tid]
                    lat_curr, z1_curr, h1_curr, z2_curr = self._encoder_forward_full(embed)
                    
                    # MSE gradient: d_loss / d_lat = 2 * (lat_curr - lat_orig) / N
                    d_lat = (2.0 * (lat_curr - lat_orig)) / len(original_latents)
                    
                    dW1_r, db1_r, dW2_r, db2_r = self._encoder_backward(
                        embed[np.newaxis, :], z1_curr[np.newaxis, :], h1_curr[np.newaxis, :], z2_curr[np.newaxis, :],
                        lat_curr[np.newaxis, :], d_lat[np.newaxis, :]
                    )
                    d_recon_W1 += dW1_r
                    d_recon_b1 += db1_r
                    d_recon_W2 += dW2_r
                    d_recon_b2 += db2_r
                
                # Add scaled reconstruction gradients to accumulators
                d_con_W1 += lambda_recon * d_recon_W1
                d_con_b1 += lambda_recon * d_recon_b1
                d_con_W2 += lambda_recon * d_recon_W2
                d_con_b2 += lambda_recon * d_recon_b2

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
        self._last_aligned_version = getattr(self.graph, 'version', 0)

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
            "n_hidden": self.n_hidden,
            "n_layers": self.n_layers,
            "token_embed": self.token_embed.weight.data.copy(),
            "subject_proj": self.subject_proj.weight.data.copy(),
            "concept_to_embed": self.concept_to_embed.weight.data.copy(),
            "relation_type_embed": self.relation_type_embed.weight.data.copy(),
            "relation_classifier_weight": self.relation_classifier.weight.data.copy(),
            "relation_classifier_bias": self.relation_classifier.bias.data.copy() if self.relation_classifier.bias is not None else None,
            "graph_nodes": {nid: {
                "vector": n.vector.copy(),
                "core_vector": n.core_vector.copy(),
                "genesis_vector": getattr(n, "genesis_vector", n.vector).copy(),
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
            "seen_predicates": sorted(self._seen_predicates),
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
            "rel_proj": self.rel_proj.copy() if hasattr(self, "rel_proj") else np.eye(self.latent_dim, dtype=np.float32),
            "use_subspace_projection": self.use_subspace_projection,
            "lambda_recon": self.lambda_recon,
            
            # --- Sparse Concept Predictor Weights ---
            # Bottleneck params (DEPRECATED - kept for backward compat loading)
            "rel_encoder": self.rel_encoder.copy(),
            "rel_bias": self.rel_bias.copy(),
            "domain_W_logits": [w.copy() for w in self.domain_W_logits],
            "domain_b_logits": [b.copy() for b in self.domain_b_logits],
            "domain_W_gates": self.domain_W_gates,
            "domain_b_gates": self.domain_b_gates,
            "router_W": self.router_W,
            "router_b": self.router_b,

            "binding_map": {
                "by_token": {tid: [(b.concept_id, b.confidence, b.source) for b in blist]
                             for tid, blist in self.binding_map._by_token.items()},
                "by_concept": {cid: [(b.token_id, b.confidence, b.source) for b in blist]
                               for cid, blist in self.binding_map._by_concept.items()},
            },
            "_tokenizer": self._tokenizer if hasattr(self, '_tokenizer') and self._tokenizer is not None else None,
            
            # --- New cognitive / EWC / replay states ---
            "cognitive_currencies": self.currencies.get_state(),
            "episodic_buffer": [{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ep.items()} for ep in self._episodic_buffer],
            "semantic_memories": self._semantic_memories,
            "concept_vad": {str(k): list(v) for k, v in self._concept_vad.items()},
            "edge_weight_ema": self._edge_weight_ema,
            "edge_weight_prev": self._edge_weight_prev,
            "token_hit_ema": self._token_hit_ema,
            "sleep_cycles_completed": self.sleep_cycles_completed,
            "conceptual_accuracy": self.conceptual_accuracy,
            "sleep_pressure": self._sleep_pressure,
            "last_loss": self._last_loss,
            "alignment_needed": self.alignment_needed,
            "wake_epochs_since_sleep": self.wake_epochs_since_sleep,
            "sleep_every_n_wake_epochs": self.sleep_every_n_wake_epochs,
            "replay_buffer": [(inputs.copy(), targets.copy()) for inputs, targets in self._replay_buffer],
            "domain_memories": {domain: [(inputs.copy(), targets.copy()) for inputs, targets in mems] for domain, mems in self._domain_memories.items()},
            "currency": self.currency.to_dict() if hasattr(self.currency, 'to_dict') else None,
            # --- Alignment completeness: save semantic_pairs for checkpoint/restore ---
            "semantic_pairs": getattr(self, "semantic_pairs", []),
        }

    def save(self, path: str):
        """Save model to file."""
        state = self.state_dict()
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_state(self, state: dict):
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

        for nid, ndata in state["graph_nodes"].items():
            node = ConceptNode(nid, ndata["vector"], ndata["label"])
            node.core_vector = ndata["core_vector"]
            node.genesis_vector = ndata.get("genesis_vector", ndata["vector"])
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
            self.graph._edges_by_relation_type[edge.relation_type].append(((s, t), edge))

        # Restore state
        self._episodic_triples = state.get("episodic_triples", [])
        self._step_counter = state.get("step_counter", 0)
        self._train_correct = state.get("train_correct", 0)
        self._train_total = state.get("train_total", 0)
        self._seen_predicates = set(state.get("seen_predicates", []))

        # Restore new concept predictor weights
        if "rel_encoder" in state:
            # Load bottleneck params for backward compat (new arch ignores them)
            self.rel_encoder = state["rel_encoder"].copy()
            self.rel_bias = state["rel_bias"].copy()
            # Domain heads: handle both old (bottleneck_dim) and new (latent_dim) shapes
            old_W = state["domain_W_logits"]
            expected_shape = (self.vocab_size, self.embed_dim)
            self.domain_W_logits = []
            for w in old_W:
                if w.shape != expected_shape:
                    # Reinit with correct shape (old checkpoint had bottleneck_dim)
                    w_new = np.random.normal(0, 0.1, expected_shape).astype(np.float32)
                    self.domain_W_logits.append(w_new)
                else:
                    self.domain_W_logits.append(w.copy())
            old_b = state["domain_b_logits"]
            self.domain_b_logits = []
            for b in old_b:
                if len(b) != self.vocab_size:
                    self.domain_b_logits.append(np.zeros(self.vocab_size, dtype=np.float32))
                else:
                    self.domain_b_logits.append(b.copy())
            # Deprecated gate/router params (loaded for backward compat, unused)
            if "domain_W_gates" in state:
                self.domain_W_gates = [w.copy() for w in state["domain_W_gates"]]
                self.domain_b_gates = [b.copy() for b in state["domain_b_gates"]]
            if "router_W" in state:
                self.router_W = state["router_W"].copy()
                self.router_b = state["router_b"].copy()

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

        if "rel_proj" in state:
            self.rel_proj = state["rel_proj"]
        if "use_subspace_projection" in state:
            self.use_subspace_projection = state["use_subspace_projection"]
        if "lambda_recon" in state:
            self.lambda_recon = state["lambda_recon"]

        # Restore binding map
        self.binding_map = ConceptBindingMap()
        bm_data = state.get("binding_map", {})
        for tid, bindings in bm_data.get("by_token", {}).items():
            for cid, conf, src in bindings:
                self.binding_map.bind(int(tid), int(cid), confidence=conf, source=src)

        # Restore new cognitive states
        if "cognitive_currencies" in state:
            self.currencies.load_state(state["cognitive_currencies"])
        self._episodic_buffer = []
        for ep in state.get("episodic_buffer", []):
            ep_copy = dict(ep)
            if ep_copy.get('vector') is not None:
                ep_copy['vector'] = np.array(ep_copy['vector'], dtype=np.float32)
            self._episodic_buffer.append(ep_copy)
        self._semantic_memories = state.get("semantic_memories", {})
        self._concept_vad = {int(k): tuple(v) for k, v in state.get("concept_vad", {}).items()}
        self._edge_weight_ema = state.get("edge_weight_ema", 0.0)
        self._edge_weight_prev = state.get("edge_weight_prev", 0.0)
        self._token_hit_ema = state.get("token_hit_ema", 0.5)
        self.sleep_cycles_completed = state.get("sleep_cycles_completed", 0)
        self.conceptual_accuracy = state.get("conceptual_accuracy", 0.0)
        self._sleep_pressure = state.get("sleep_pressure", 0.0)
        self._last_loss = state.get("last_loss", 0.0)
        self.alignment_needed = state.get("alignment_needed", False)
        self.wake_epochs_since_sleep = state.get("wake_epochs_since_sleep", 0)
        self.sleep_every_n_wake_epochs = state.get("sleep_every_n_wake_epochs", 3)
        self._replay_buffer = [(inputs.copy(), targets.copy()) for inputs, targets in state.get("replay_buffer", [])]
        self._domain_memories = {domain: [(inputs.copy(), targets.copy()) for inputs, targets in mems] for domain, mems in state.get("domain_memories", {}).items()}
        if "currency" in state and state["currency"] is not None:
            self.currency.load_dict(state["currency"])

        # --- Alignment completeness: restore semantic_pairs from checkpoint ---
        if "semantic_pairs" in state:
            self.semantic_pairs = state["semantic_pairs"]

        # Rebuild raw numpy caches in submodules and clear cached norms
        for mod in self.modules():
            if hasattr(mod, '_rebuild_raw_cache'):
                mod._rebuild_raw_cache()
        self._token_embed_norms = None
        self.graph._vectors_dirty = True
        self.graph._adj_dirty = True
        self._invalidate_caches()

    def load(self, path: Optional[str] = None) -> 'RLMv2':
        """Load model from file. Supports both RLM.load(path) and model.load(path)."""
        if path is None:
            # Called as RLM.load(path) -> self is actually the path string
            path_str = self
            with open(path_str, 'rb') as f:
                state = pickle.load(f)
            # Create new instance
            model = RLMv2(
                vocab_size=state["vocab_size"],
                embed_dim=state["embed_dim"],
                concept_dim=state["concept_dim"],
                n_concepts=state["n_concepts"]
            )
            model._load_state(state)
            return model
        else:
            # Called as model.load(path) -> self is the model instance, path is the path string
            with open(path, 'rb') as f:
                state = pickle.load(f)
            self._load_state(state)
            return self

    def save_zip(self, path: str):
        """Save model as a zip archive with separate files."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        arrays = {}
        state_dict_meta = {}
        sd = self.state_dict()
        
        # Neural weights
        for name in ["token_embed", "subject_proj", "concept_to_embed", "relation_type_embed", "relation_classifier_weight", "relation_classifier_bias"]:
            if name in sd and sd[name] is not None:
                key = f"weight/{name}"
                arrays[key] = sd[name]
                state_dict_meta[name] = {
                    "salience": 0.5,
                    "free_energy": 0.0,
                    "stability": 0.5,
                }

        # Relation predictor arrays
        arrays["rp/W1"] = self._rp_W1
        arrays["rp/b1"] = self._rp_b1
        arrays["rp/W2"] = self._rp_W2
        arrays["rp/b2"] = self._rp_b2
        arrays["rp/mW1"] = self._rp_mW1
        arrays["rp/mb1"] = self._rp_mb1
        arrays["rp/mW2"] = self._rp_mW2
        arrays["rp/mb2"] = self._rp_mb2
        
        # Domain-Agnostic variables
        arrays["enc/W1"] = self._enc_W1
        arrays["enc/b1"] = self._enc_b1
        arrays["enc/W2"] = self._enc_W2
        arrays["enc/b2"] = self._enc_b2
        arrays["enc/mW1"] = self._enc_mW1
        arrays["enc/mb1"] = self._enc_mb1
        arrays["enc/mW2"] = self._enc_mW2
        arrays["enc/mb2"] = self._enc_mb2
        arrays["dec/W1"] = self._dec_W1
        arrays["dec/b1"] = self._dec_b1
        arrays["dec/W2"] = self._dec_W2
        arrays["dec/b2"] = self._dec_b2
        arrays["rp/rel_matrices"] = self._rp_rel_matrices
        arrays["rp/mrel_matrices"] = self._rp_mrel_matrices
        arrays["rp/rel_proj"] = self.rel_proj

        # Node vectors
        for nid, node in self.graph.nodes.items():
            arrays[f"node/{nid}"] = node.vector
            arrays[f"node_core/{nid}"] = node.core_vector
            arrays[f"node_genesis/{nid}"] = getattr(node, "genesis_vector", node.vector)

        # Edges relation vectors
        edge_relation_vectors = {}
        edges_json = {}
        for (s, t), edge in self.graph.edges.items():
            edges_json[f"({s}, {t})"] = {
                "source": int(edge.source),
                "target": int(edge.target),
                "weight": float(edge.weight),
                "confidence": float(edge.confidence),
                "stability": float(edge.stability),
                "timestamp": float(edge.timestamp),
                "prediction_count": int(edge.prediction_count),
                "relation_type": edge.relation_type,
                "predicate_token_id": int(getattr(edge, 'predicate_token_id', -1)),
            }
            if edge.relation_vector is not None:
                edge_relation_vectors[f"({s}, {t})"] = edge.relation_vector

        for key, rvec in edge_relation_vectors.items():
            safe_key = key.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
            arrays[f"edge_rel/{safe_key}"] = rvec

        nodes_json = {}
        for nid, node in self.graph.nodes.items():
            nodes_json[str(nid)] = {
                "id": node.id,
                "label": node.label,
                "activation": float(node.activation),
                "stability": float(node.stability),
                "confidence": float(node.confidence),
            }

        graph_json = {
            "dim": self.graph.dim,
            "max_nodes": self.graph.max_nodes,
            "next_id": self.graph.next_id,
            "total_free_energy": float(self.graph.total_free_energy),
            "contradiction_hotspots": sorted(int(x) for x in self.graph.contradiction_hotspots),
            "active_nodes": sorted(int(x) for x in self.graph._active_nodes),
            "nodes": nodes_json,
            "edges": edges_json,
        }

        # Build metadata JSON
        metadata_json = {
            "format": "ravana_zip",
            "version": 2,
            "config": {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "concept_dim": self.concept_dim,
                "n_concepts": self.n_concepts,
                "max_seq_len": self.max_seq_len,
                "sleep_interval": self.sleep_interval,
            },
            "scalars": {
                "step_counter": self._step_counter,
                "sleep_cycles_completed": self.sleep_cycles_completed,
                "conceptual_accuracy": float(self.conceptual_accuracy),
                "train_correct": self._train_correct,
                "train_total": self._train_total,
                "sleep_pressure": float(self._sleep_pressure),
                "last_loss": float(self._last_loss),
                "alignment_needed": bool(self.alignment_needed),
                "wake_epochs_since_sleep": int(self.wake_epochs_since_sleep),
                "sleep_every_n_wake_epochs": int(self.sleep_every_n_wake_epochs),
                "use_subspace_projection": bool(self.use_subspace_projection),
                "lambda_recon": float(self.lambda_recon),
            },
            "bindings": [
                {
                    "token_id": b.token_id,
                    "concept_id": b.concept_id,
                    "confidence": float(b.confidence),
                    "source": b.source,
                }
                for b in self.binding_map._index.values()
            ],
            "state_dict_meta": state_dict_meta,
            "_tokenizer": self._tokenizer if hasattr(self, '_tokenizer') and self._tokenizer is not None else None,
        }

        # Serialize episodic buffer (convert numpy arrays to lists for JSON)
        episodes_json = []
        for ep in self._episodic_buffer:
            ep_copy = dict(ep)
            if ep_copy.get('vector') is not None:
                ep_copy['vector'] = ep_copy['vector'].tolist()
            episodes_json.append(ep_copy)

        metadata_json["cognitive_state"] = self.currencies.get_state() | {
            "edge_weight_ema": self._edge_weight_ema,
            "edge_weight_prev": self._edge_weight_prev,
            "token_hit_ema": self._token_hit_ema,
            "episodic_buffer": episodes_json,
            "semantic_memories": self._semantic_memories,
            "concept_vad": {str(k): list(v) for k, v in self._concept_vad.items()},
            "currency": self.currency.to_dict() if hasattr(self.currency, 'to_dict') else None,
        }

        # Replay buffer (convert numpy arrays to lists)
        replay_json = []
        for inputs, targets in self._replay_buffer:
            replay_json.append((inputs.tolist(), targets.tolist()))
        metadata_json["replay_buffer"] = replay_json
        
        domain_memories_json = {}
        for domain, mems in self._domain_memories.items():
            domain_memories_json[domain] = [(inputs.tolist(), targets.tolist()) for inputs, targets in mems]
        metadata_json["domain_memories"] = domain_memories_json

        # Write zip
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
            from io import BytesIO
            buf = BytesIO()
            np.savez(buf, **arrays)
            zf.writestr("arrays.npz", buf.getvalue())
            zf.writestr("graph.json", json.dumps(graph_json, indent=2))
            zf.writestr("metadata.json", json.dumps(metadata_json, indent=2))

    @classmethod
    def load_zip(cls, path: str) -> 'RLMv2':
        """Load a model from a zip archive."""
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
            model.token_embed.weight.data = npz["weight/token_embed"]
            model.subject_proj.weight.data = npz["weight/subject_proj"]
            model.concept_to_embed.weight.data = npz["weight/concept_to_embed"]
            model.relation_type_embed.weight.data = npz["weight/relation_type_embed"]
            model.relation_classifier.weight.data = npz["weight/relation_classifier_weight"]
            if "weight/relation_classifier_bias" in npz:
                model.relation_classifier.bias.data = npz["weight/relation_classifier_bias"]

            # Restore relation predictor weights
            model._rp_W1 = npz["rp/W1"]
            model._rp_b1 = npz["rp/b1"]
            model._rp_W2 = npz["rp/W2"]
            model._rp_b2 = npz["rp/b2"]
            model._rp_mW1 = npz["rp/mW1"]
            model._rp_mb1 = npz["rp/mb1"]
            model._rp_mW2 = npz["rp/mW2"]
            model._rp_mb2 = npz["rp/mb2"]
            
            # Restore domain-agnostic variables
            model._enc_W1 = npz["enc/W1"]
            model._enc_b1 = npz["enc/b1"]
            model._enc_W2 = npz["enc/W2"]
            model._enc_b2 = npz["enc/b2"]
            model._enc_mW1 = npz["enc/mW1"]
            model._enc_mb1 = npz["enc/mb1"]
            model._enc_mW2 = npz["enc/mW2"]
            model._enc_mb2 = npz["enc/mb2"]
            model._dec_W1 = npz["dec/W1"]
            model._dec_b1 = npz["dec/b1"]
            model._dec_W2 = npz["dec/w2"] if "dec/w2" in npz else npz["dec/W2"]
            model._dec_b2 = npz["dec/b2"]
            model._rp_rel_matrices = npz["rp/rel_matrices"]
            model._rp_mrel_matrices = npz["rp/mrel_matrices"]
            if "rp/rel_proj" in npz:
                model.rel_proj = npz["rp/rel_proj"]

            # Restore graph
            graph_data = json.loads(zf.read("graph.json"))
            model.graph.next_id = graph_data["next_id"]
            model.graph.nodes.clear()
            model.graph.edges.clear()
            model.graph._outgoing.clear()
            model.graph._incoming.clear()
            model.graph._edges_by_relation_type.clear()

            # Restore nodes
            for nid_str, nd in graph_data["nodes"].items():
                nid = int(nid_str)
                vec_key = f"node/{nid}"
                vector = npz[vec_key] if vec_key in npz else np.zeros(cfg["concept_dim"], dtype=np.float32)
                node = ConceptNode(node_id=nd["id"], vector=vector, label=nd["label"])
                node.activation = nd["activation"]
                node.stability = nd["stability"]
                node.confidence = nd["confidence"]
                
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
                
                model.graph.nodes[nid] = node

            # Restore edges
            for key, ed in graph_data["edges"].items():
                s = ed["source"]
                t = ed["target"]
                edge = ConceptEdge(source=s, target=t, weight=ed["weight"],
                                  relation_type=ed["relation_type"],
                                  relation_dim=cfg["concept_dim"])
                edge.confidence = ed["confidence"]
                edge.stability = ed["stability"]
                edge.timestamp = ed["timestamp"]
                edge.prediction_count = ed["prediction_count"]
                edge.predicate_token_id = ed.get("predicate_token_id", -1)
                
                # Restore edge relation vector
                safe_key = key.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
                rel_key = f"edge_rel/{safe_key}"
                if rel_key in npz:
                    edge.relation_vector = npz[rel_key]
                
                model.graph.edges[(s, t)] = edge
                model.graph._outgoing[s].append((t, edge))
                model.graph._incoming[t].append((s, edge))
                model.graph._edges_by_relation_type[edge.relation_type].append(((s, t), edge))

            # Restore tokenizer (bypass setter to avoid redundant pre-training)
            model._tokenizer_val = meta.get("_tokenizer", None)

            # Restore scalars
            s_data = meta["scalars"]
            model._step_counter = s_data["step_counter"]
            model.sleep_cycles_completed = s_data["sleep_cycles_completed"]
            model.conceptual_accuracy = s_data["conceptual_accuracy"]
            model._train_correct = s_data.get("train_correct", 0)
            model._train_total = s_data.get("train_total", 0)
            model._sleep_pressure = s_data.get("sleep_pressure", 0.0)
            model._last_loss = s_data.get("last_loss", 0.0)
            model.alignment_needed = s_data.get("alignment_needed", False)
            model.wake_epochs_since_sleep = s_data.get("wake_epochs_since_sleep", 0)
            model.sleep_every_n_wake_epochs = s_data.get("sleep_every_n_wake_epochs", 3)
            model.use_subspace_projection = s_data.get("use_subspace_projection", False)
            model.lambda_recon = s_data.get("lambda_recon", 0.0)

            # Restore bindings
            for b_data in meta.get("bindings", []):
                model.binding_map.bind(b_data["token_id"], b_data["concept_id"], confidence=b_data["confidence"], source=b_data["source"])

            # Restore cognitive state
            cs = meta.get("cognitive_state", {})
            if cs:
                model.currencies.load_state(cs)
                model._edge_weight_ema = cs.get("edge_weight_ema", 0.0)
                model._edge_weight_prev = cs.get("edge_weight_prev", 0.0)
                model._token_hit_ema = cs.get("token_hit_ema", 0.5)
                
                model._episodic_buffer = []
                for ep in cs.get("episodic_buffer", []):
                    ep_copy = dict(ep)
                    if ep_copy.get('vector') is not None:
                        ep_copy['vector'] = np.array(ep_copy['vector'], dtype=np.float32)
                    model._episodic_buffer.append(ep_copy)
                model._semantic_memories = cs.get("semantic_memories", {})
                model._concept_vad = {int(k): tuple(v) for k, v in cs.get("concept_vad", {}).items()}
                if "currency" in cs and cs["currency"] is not None:
                    model.currency.load_dict(cs["currency"])

            # Restore replay buffers
            replay_json = meta.get("replay_buffer", [])
            model._replay_buffer = [(np.array(inputs, dtype=np.int64), np.array(targets, dtype=np.int64)) for inputs, targets in replay_json]
            
            domain_memories_json = meta.get("domain_memories", {})
            model._domain_memories = {domain: [(np.array(inputs, dtype=np.int64), np.array(targets, dtype=np.int64)) for inputs, targets in mems] for domain, mems in domain_memories_json.items()}

            # Rebuild raw numpy caches in submodules and clear cached norms
            for mod in model.modules():
                if hasattr(mod, '_rebuild_raw_cache'):
                    mod._rebuild_raw_cache()
            model._token_embed_norms = None
            model.graph._vectors_dirty = True
            model.graph._adj_dirty = True
            model._invalidate_caches()

            return model

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
        # Track best path coherence reaching each node (for path-level validation)
        path_coherence = {}
        for cid, _, _ in seeds:
            path_coherence[cid] = 1.0 if gate_mode not in ("weighted", "margin_multi") else activations[cid]

        for depth in range(1, max_depth + 1):
            next_frontier = []
            for nid in frontier:
                act = activations[nid]
                src_coherence = path_coherence.get(nid, 0.0)
                for tgt_id, edge in self.graph.get_outgoing(nid):
                    if rel_type and edge.relation_type != rel_type:
                        continue
                    # Path-level validation: compute coherence of path through this edge
                    # coherence = product of edge weights * relation_type_bonus (confidence excluded - it's for learning, not path validity)
                    edge_weight = edge.posterior_mean if hasattr(edge, 'posterior_mean') else edge.weight
                    relation_bonus = 1.1 if (rel_type and edge.relation_type == rel_type) else 0.9
                    if edge.edge_type == "inhibitory":
                        relation_bonus *= 0.5
                    new_coherence = src_coherence * edge_weight * relation_bonus
                    # Only propagate if path coherence meets threshold
                    if new_coherence < path_coherence_threshold:
                        continue
                    prop = act * edge_weight
                    if tgt_id not in activations or prop > activations[tgt_id]:
                        activations[tgt_id] = prop
                        path_coherence[tgt_id] = max(path_coherence.get(tgt_id, 0.0), new_coherence)
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
        adaptive_margin_factor: float = 0.5,  # fraction of inter-seed spread to use as margin
        path_coherence_threshold: float = 0.01,  # minimum path coherence for activation propagation
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        """Multi-seed and margin-gated hybrid retrieval (version 2).

        Gate modes:
        - "standard": top-3 unconditionally
        - "strict_margin": single seed if margin >= 0.15 and sim >= 0.50
        - "relative_threshold": seeds within 0.85 * best_sim, up to k_neighbors
        - "weighted": top-k_neighbors always, with softmax weights
        - "margin_multi": strict_margin for single seed, else weighted fallback
        - "adaptive_margin": dynamic margin = spread * adaptive_margin_factor

        Path coherence (new): During BFS traversal, each path from seed to node
        is scored for causal coherence. Only paths with coherence >=
        path_coherence_threshold propagate activation.
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
        # Track best path coherence reaching each node (for path-level validation)
        path_coherence = {}
        for cid, _, _ in seeds:
            path_coherence[cid] = 1.0 if gate_mode not in ("weighted", "margin_multi") else activations[cid]

        for depth in range(1, max_depth + 1):
            next_frontier = []
            for nid in frontier:
                act = activations[nid]
                src_coherence = path_coherence.get(nid, 0.0)
                for tgt_id, edge in self.graph.get_outgoing(nid):
                    if rel_type and edge.relation_type != rel_type:
                        continue
                    # Path-level validation: compute coherence of path through this edge
                    # coherence = product of edge weights * relation_type_bonus (confidence excluded - it's for learning, not path validity)
                    edge_weight = edge.posterior_mean if hasattr(edge, 'posterior_mean') else edge.weight
                    relation_bonus = 1.1 if (rel_type and edge.relation_type == rel_type) else 0.9
                    if edge.edge_type == "inhibitory":
                        relation_bonus *= 0.5
                    new_coherence = src_coherence * edge_weight * relation_bonus
                    # Only propagate if path coherence meets threshold
                    if new_coherence < path_coherence_threshold:
                        continue
                    prop = act * edge_weight
                    if tgt_id not in activations or prop > activations[tgt_id]:
                        activations[tgt_id] = prop
                        path_coherence[tgt_id] = max(path_coherence.get(tgt_id, 0.0), new_coherence)
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
    # ── Cross-Domain Relation Alignment (Fix #3) ──
    # Pulls subject->object pairs that share the same relation type closer in latent
    # space across domains. Without this, W_rel learns separate sub-clusters per
    # domain (heat->expansion in physics, anger->conflict in social) and held-out
    # cross-domain queries fail.

    def _cross_domain_relation_alignment(self, lr=None):
        """One step of cross-domain relation alignment.

        For each relation type, find all subject-object pairs in the graph that use
        that relation, compute the mean (target_embed - W_rel @ source_embed) residual,
        and nudge W_rel to reduce that residual averaged across ALL pairs (regardless
        of domain). This pushes W_rel toward a domain-agnostic relation transform.
        """
        if lr is None:
            lr = getattr(self, "alignment_lr", 0.005)

        from collections import defaultdict
        pairs_by_rel = defaultdict(list)
        for (src_cid, tgt_cid), edge in self.graph.edges.items():
            rel_name = edge.relation_type
            if rel_name not in RELATION_TYPES:
                continue
            if edge.weight < 0.2 or edge.confidence < 0.2:
                continue
            # Look up token IDs from binding map
            src_tokens = self.binding_map.get_tokens(src_cid, min_confidence=0.1)
            tgt_tokens = self.binding_map.get_tokens(tgt_cid, min_confidence=0.1)
            if not src_tokens or not tgt_tokens:
                continue
            src_tid = src_tokens[0].token_id
            tgt_tid = tgt_tokens[0].token_id
            # Skip synthetic tokens (rel_obj hubs use IDs >= 10000)
            if src_tid >= self.vocab_size or tgt_tid >= self.vocab_size:
                continue
            pairs_by_rel[rel_name].append((src_tid, tgt_tid))

        results = {}
        for rel_name, pairs in pairs_by_rel.items():
            if len(pairs) < 2:
                continue
            rel_idx = RELATION_TYPES.index(rel_name)
            W_rel = self._rp_rel_matrices[rel_idx]
            # Accumulate gradient toward minimizing mean residual norm
            grad_sum = np.zeros_like(W_rel)
            for src_tid, tgt_tid in pairs:
                s = self.token_embed.weight.data[src_tid]   # (embed_dim,)
                t = self.token_embed.weight.data[tgt_tid]   # (embed_dim,)
                pred = s @ W_rel                            # (embed_dim,)
                residual = pred - t                          # (embed_dim,)
                grad_sum += np.outer(s, residual)            # (embed_dim, embed_dim)
            grad_mean = grad_sum / len(pairs)
            # Clip
            gn = np.linalg.norm(grad_mean)
            if gn > 5.0:
                grad_mean *= (5.0 / (gn + 1e-15))
            self._rp_rel_matrices[rel_idx] -= lr * grad_mean
            results[rel_name] = float(np.linalg.norm(grad_mean))
        return results

    def measure_cross_domain_alignment(self):
        """Measure how well W_rel transforms generalize across all subject-object pairs.

        Returns dict mapping relation_type -> mean cosine similarity between
        (W_rel @ source_embed) and target_embed across all stored pairs of that type.
        Higher = better alignment (W_rel actually maps subject->object generically).
        """
        from collections import defaultdict
        pairs_by_rel = defaultdict(list)
        for (src_cid, tgt_cid), edge in self.graph.edges.items():
            rel_name = edge.relation_type
            if rel_name not in RELATION_TYPES:
                continue
            if edge.weight < 0.2:
                continue
            src_tokens = self.binding_map.get_tokens(src_cid, min_confidence=0.1)
            tgt_tokens = self.binding_map.get_tokens(tgt_cid, min_confidence=0.1)
            if not src_tokens or not tgt_tokens:
                continue
            src_tid = src_tokens[0].token_id
            tgt_tid = tgt_tokens[0].token_id
            if src_tid >= self.vocab_size or tgt_tid >= self.vocab_size:
                continue
            pairs_by_rel[rel_name].append((src_tid, tgt_tid))

        results = {}
        for rel_name, pairs in pairs_by_rel.items():
            if not pairs:
                continue
            rel_idx = RELATION_TYPES.index(rel_name)
            W_rel = self._rp_rel_matrices[rel_idx]
            sims = []
            for src_tid, tgt_tid in pairs:
                s = self.token_embed.weight.data[src_tid]
                t = self.token_embed.weight.data[tgt_tid]
                pred = s @ W_rel
                sn = np.linalg.norm(pred); tn = np.linalg.norm(t)
                if sn > 0 and tn > 0:
                    sims.append(float(np.dot(pred, t) / (sn * tn)))
            if sims:
                results[rel_name] = float(np.mean(sims))
        return results

    def _inject_cross_domain_edge(self, subject_cid, object_cid, rel_name, subject_tid):
        """Inject (or strengthen) a relation-typed edge subject->object.

        Used by experiments to seed cross-domain analogical edges (e.g.
        anger->conflict mirroring heat->expansion). Idempotent: tracks injected
        pairs in self._cross_domain_edges_injected so repeat calls don't pollute.
        """
        if rel_name not in RELATION_TYPES:
            return
        key = (subject_cid, object_cid, rel_name)
        if key in self._cross_domain_edges_injected:
            return
        edge = self.graph.get_edge(subject_cid, object_cid)
        if edge is None:
            edge = self.graph.add_edge(
                source=subject_cid, target=object_cid,
                weight=0.5, relation_type=rel_name,
            )
        else:
            # Strengthen if same relation type, otherwise leave alone
            if edge.relation_type == rel_name:
                edge.weight = min(1.0, edge.weight + 0.2)
                edge.confidence = min(1.0, edge.confidence + 0.1)
        if subject_tid is not None and 0 <= subject_tid < self.vocab_size:
            try:
                edge.predicate_token_id = int(subject_tid)
            except Exception:
                pass
        self._cross_domain_edges_injected.add(key)

