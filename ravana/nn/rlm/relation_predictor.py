"""
Relation Predictor - Learns relation type embeddings and classifiers.
Handles: relation classification, bilinear W_rel matrices, verb-stem offset predictor.
"""
import numpy as np
import time
from typing import Optional, List, Tuple, Dict, Any, Set
from collections import defaultdict

from ravana_ml.nn.module import Module, Linear, Embedding
from ravana_ml.graph import ConceptGraph, ConceptEdge


RELATION_TYPES = [
    "causal", "semantic", "temporal", "possessive", "analogical", "contextual",
]

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


class RelationPredictor:
    """Learns relation type embeddings, classifiers, and bilinear W_rel for triple prediction."""

    def __init__(self, vocab_size: int, embed_dim: int, concept_dim: int,
                 latent_dim: int = 96, hidden_dim: int = 128, n_hidden: int = 32,
                 n_layers: int = 2):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.concept_dim = concept_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        n_rel_types = len(RELATION_TYPES)

        # Relation type embeddings and classifier
        self.relation_type_embed = Embedding(n_rel_types, concept_dim)
        self.relation_classifier = Linear(concept_dim, n_rel_types, bias=True)
        self._classifier_lr = 0.01

        # Bilinear W_rel matrices (one per relation type)
        # Identity init: source @ I @ target = dot product
        self._rp_rel_matrices = np.tile(np.eye(latent_dim, dtype=np.float32), (n_rel_types, 1, 1))
        self._rp_mrel_matrices = np.zeros_like(self._rp_rel_matrices)

        # Encoder (embed_dim -> hidden_dim -> latent_dim)
        rng_enc = np.random.RandomState(42)
        max_dim = max(hidden_dim, latent_dim, embed_dim)
        full_q, _ = np.linalg.qr(rng_enc.randn(max_dim, max_dim).astype(np.float32))
        self._enc_W1 = (full_q[:hidden_dim, :embed_dim].copy() * np.sqrt(2.0 / embed_dim)).astype(np.float32)
        self._enc_b1 = np.zeros(hidden_dim, dtype=np.float32)
        self._enc_W2 = (full_q[:latent_dim, :hidden_dim].copy() * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self._enc_b2 = np.zeros(latent_dim, dtype=np.float32)

        self._enc_mW1 = np.zeros_like(self._enc_W1)
        self._enc_mb1 = np.zeros_like(self._enc_b1)
        self._enc_mW2 = np.zeros_like(self._enc_W2)
        self._enc_mb2 = np.zeros_like(self._enc_b2)

        # Decoder (for autoencoder pre-training)
        scale_dec1 = np.sqrt(2.0 / latent_dim)
        scale_dec2 = np.sqrt(2.0 / hidden_dim)
        self._dec_W1 = np.random.randn(hidden_dim, latent_dim).astype(np.float32) * scale_dec1
        self._dec_b1 = np.zeros(hidden_dim, dtype=np.float32)
        self._dec_W2 = np.random.randn(embed_dim, hidden_dim).astype(np.float32) * scale_dec2
        self._dec_b2 = np.zeros(embed_dim, dtype=np.float32)

        # MLP relation predictor (legacy compat)
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
        self._rp_encoder_lr = 0.005
        self._rp_momentum = 0.9
        self._rp_cache = None
        self.rp_scale = 1.0
        self.use_rp_for_analogy = True

        # Verb-stem offset predictor
        self._verb_offsets: Dict[str, np.ndarray] = {}
        self._verb_offset_count: Dict[str, int] = {}
        self._verb_accum_buffer: List[Tuple[str, np.ndarray]] = []
        self.use_verb_offset = False

        # Cross-domain alignment
        self.use_cross_domain_alignment = False
        self.freeze_token_embeds_in_rp = True
        self.alignment_needed = False
        self.alignment_lr = 0.005
        self.alignment_margin = 0.15
        self.max_alignment_epochs = 10

        # Domain heads (legacy)
        self.bottleneck_dim = 32
        self.num_domains = 4
        self._domain_heads = {}
        self.current_domain_id = None
        self._frozen_domains = set()
        self._init_domain_heads()

    def _init_domain_heads(self):
        """Compatibility hook for older domain-head routing."""
        pass

    def set_domain(self, domain_id: Optional[int], freeze_others: bool = True):
        self.current_domain_id = domain_id
        if domain_id is not None and freeze_others:
            self._frozen_domains = {d for d in range(self.num_domains) if d != domain_id}
        else:
            self._frozen_domains = set()

    def freeze_domain(self, domain_id: int):
        self._frozen_domains.add(domain_id)

    def unfreeze_domain(self, domain_id: int):
        self._frozen_domains.discard(domain_id)

    def unfreeze_all_domains(self):
        self._frozen_domains.clear()

    # ── Classification ──

    def classify_relation(self, relation_token_ids: List[int],
                          decode_token_fn) -> int:
        """Keyword-based relation classification."""
        if not relation_token_ids:
            return RELATION_TYPES.index("semantic")

        rel_words = set()
        for tid in relation_token_ids:
            try:
                word = decode_token_fn(tid).lower().strip()
                if word:
                    rel_words.add(word)
            except Exception:
                pass

        for rel_type, keywords in _KEYWORD_MAP.items():
            for word in rel_words:
                if word in keywords:
                    return RELATION_TYPES.index(rel_type)
        return RELATION_TYPES.index("semantic")

    def classify_relation_learned(self, relation_token_ids: List[int],
                                   decode_token_fn,
                                   token_embed,
                                   subject_proj) -> Tuple[int, np.ndarray]:
        """Learned relation classification with keyword fallback."""
        keyword_idx = self.classify_relation(relation_token_ids, decode_token_fn)

        if not relation_token_ids:
            return keyword_idx, self.relation_type_embed.weight.data[keyword_idx]

        rel_embeds = [token_embed.weight.data[tid] for tid in relation_token_ids]
        rel_vec = np.mean(rel_embeds, axis=0)
        rel_concept = subject_proj(rel_vec.reshape(1, -1)).data.flatten()

        type_embeds = self.relation_type_embed.weight.data
        rel_norm = np.linalg.norm(rel_concept)
        type_norms = np.linalg.norm(type_embeds, axis=1)
        if rel_norm > 0 and np.all(type_norms > 0):
            sims = type_embeds @ rel_concept / (type_norms * rel_norm)
        else:
            sims = np.zeros(len(RELATION_TYPES))

        # Keyword-only for deterministic behavior (tested: 100% keyword gives 62.5% cross_domain_causal)
        return keyword_idx, self.relation_type_embed.weight.data[keyword_idx]

    def _update_relation_classifier(self, relation_token_ids: List[int],
                                     true_type_idx: int,
                                     token_embed, subject_proj):
        """Local Hebbian update of relation classifier."""
        if not relation_token_ids:
            return

        rel_embeds = [token_embed.weight.data[tid] for tid in relation_token_ids]
        rel_vec = np.mean(rel_embeds, axis=0)
        rel_concept = subject_proj(rel_vec.reshape(1, -1)).data.flatten()

        logits_tensor = self.relation_classifier(rel_concept.reshape(1, -1))
        logits = logits_tensor.data.flatten()
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)

        error = np.zeros(len(RELATION_TYPES), dtype=np.float32)
        error[true_type_idx] = 1.0 - probs[true_type_idx]
        for i in range(len(RELATION_TYPES)):
            if i != true_type_idx:
                error[i] = -probs[i]

        lr = self._classifier_lr
        input_2d = rel_concept.reshape(1, -1)
        error_2d = error.reshape(-1, 1)
        delta_w = lr * (error_2d @ input_2d)
        self.relation_classifier.weight.data += delta_w
        if self.relation_classifier.bias is not None:
            self.relation_classifier.bias.data += lr * error

        # Update relation type embedding
        type_embed = self.relation_type_embed.weight.data[true_type_idx]
        pull = 0.01 * (rel_concept[:len(type_embed)] - type_embed)
        self.relation_type_embed.weight.data[true_type_idx] += pull
        norm = np.linalg.norm(self.relation_type_embed.weight.data[true_type_idx])
        if norm > 0:
            self.relation_type_embed.weight.data[true_type_idx] /= norm

    # ── Verb-Stem Offset Predictor ──

    @staticmethod
    def _verb_stem(word: str) -> str:
        w = word.lower().strip()
        if len(w) <= 3:
            return w
        for suffix in ['ing', 'ed', 'es', 's', 'd']:
            if w.endswith(suffix) and len(w) > len(suffix) + 1:
                w = w[:-len(suffix)]
                break
        return w

    def accumulate_verb_offset(self, subject_tid: int, target_tid: int,
                                verb_word: str, token_embed):
        """Accumulate offset = target_embed - subject_embed for verb stem."""
        if not verb_word or not self.use_verb_offset:
            return
        stem = self._verb_stem(verb_word)
        subject_embed = token_embed.weight.data[subject_tid]
        target_embed = token_embed.weight.data[target_tid]
        offset = target_embed - subject_embed
        self._verb_accum_buffer.append((stem, offset))

    def compute_verb_offsets(self):
        """Finalize verb offsets by averaging accumulated vectors."""
        if not self.use_verb_offset:
            return
        sums: Dict[str, np.ndarray] = {}
        counts: Dict[str, int] = {}
        for stem, offset in self._verb_accum_buffer:
            if stem not in sums:
                sums[stem] = np.zeros_like(offset)
                counts[stem] = 0
            sums[stem] += offset
            counts[stem] += 1
        for stem, total in sums.items():
            if counts[stem] > 0:
                offset = total / counts[stem]
                norm = np.linalg.norm(offset)
                if norm > 0:
                    offset = offset / norm * min(norm, 5.0)
                self._verb_offsets[stem] = offset
                self._verb_offset_count[stem] = counts[stem]
        if self._verb_offsets:
            print(f"  [Verb Offset] Computed {len(self._verb_offsets)} verb offsets")

    def forward_verb_offset(self, subject_tid: int, verb_word: str,
                             token_embed) -> Optional[np.ndarray]:
        """Predict using verb-stem offset arithmetic."""
        if not verb_word or not self.use_verb_offset:
            return None
        stem = self._verb_stem(verb_word)
        if stem not in self._verb_offsets:
            return None
        source_embed = token_embed.weight.data[subject_tid]
        offset = self._verb_offsets[stem]
        predicted = source_embed + offset

        token_embeds = token_embed.weight.data
        token_norms = np.linalg.norm(token_embeds, axis=1)
        pred_norm = np.linalg.norm(predicted)
        if pred_norm > 0 and np.any(token_norms > 0):
            valid_tok = token_norms > 0
            normed_tok = token_embeds.copy()
            normed_tok[valid_tok] /= token_norms[valid_tok, np.newaxis]
            logits = (predicted / pred_norm) @ normed_tok.T
            if 0 <= subject_tid < len(logits):
                logits[subject_tid] = np.min(logits) - 10.0
            logits *= 10.0
            return logits
        return None

    # ── Bilinear W_rel Forward/Backward ──

    def _encoder_forward_full(self, X):
        is_flat = X.ndim == 1
        if is_flat:
            X_batch = X[np.newaxis, :]
        else:
            X_batch = X
        z1 = X_batch @ self._enc_W1.T + self._enc_b1
        h1 = np.tanh(z1)
        z2 = h1 @ self._enc_W2.T + self._enc_b2
        latent = np.tanh(z2)
        if is_flat:
            return latent[0], z1[0], h1[0], z2[0]
        return latent, z1, h1, z2

    def _encoder_backward(self, X, z1, h1, z2, h2, d_h2):
        d_z2 = d_h2 * (1.0 - h2 * h2)
        d_enc_W2 = d_z2.T @ h1
        d_enc_b2 = np.sum(d_z2, axis=0)
        d_h1 = d_z2 @ self._enc_W2
        d_z1 = d_h1 * (1.0 - h1 * h1)
        d_enc_W1 = d_z1.T @ X
        d_enc_b1 = np.sum(d_z1, axis=0)
        return d_enc_W1, d_enc_b1, d_enc_W2, d_enc_b2

    def rp_forward(self, subject_tid: int, rel_type_idx: int,
                    token_embed, verb_word: str = None) -> np.ndarray:
        """Bilinear RP forward using raw token embeddings."""
        # Try verb offset first
        if verb_word:
            verb_logits = self.forward_verb_offset(subject_tid, verb_word, token_embed)
            if verb_logits is not None:
                self._rp_cache = None
                return verb_logits

        source_embed = token_embed.weight.data[subject_tid]
        token_embeds = token_embed.weight.data

        if self.embed_dim != self.latent_dim:
            source_latent, _, _, _ = self._encoder_forward_full(source_embed)
            target_latents, _, _, _ = self._encoder_forward_full(token_embeds)
        else:
            source_latent = source_embed
            target_latents = token_embeds

        W_rel = self._rp_rel_matrices[rel_type_idx]
        projected = source_latent @ W_rel
        logits = projected @ target_latents.T

        self._rp_cache = (subject_tid, rel_type_idx,
                          source_embed, source_latent,
                          token_embeds, target_latents,
                          W_rel, logits)
        return logits

    def rp_backward(self, target_id: int, lr_scale: float = 1.0):
        """Bilinear RP backward with gradient clipping."""
        if self._rp_cache is None:
            return

        (subject_tid, rel_type_idx,
         source_embed, source_latent,
         token_embeds, target_latents,
         W_rel, logits) = self._rp_cache

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)
        d_logits = probs.copy()
        if 0 <= target_id < len(d_logits):
            d_logits[target_id] -= 1.0
        d_logits *= getattr(self, "rp_scale", 16.0)

        d_logits_proj = d_logits @ target_latents
        dW_rel = np.outer(source_latent, d_logits_proj)

        grad_norm = np.linalg.norm(dW_rel)
        if grad_norm > 10.0:
            dW_rel *= (10.0 / (grad_norm + 1e-15))

        lr = self._rp_lr * lr_scale
        freeze_token_embeds = self.freeze_token_embeds_in_rp
        embed_lr = 0.0 if freeze_token_embeds else lr * 0.1

        self._rp_mrel_matrices[rel_type_idx] = (
            self._rp_momentum * self._rp_mrel_matrices[rel_type_idx] - lr * dW_rel)
        self._rp_rel_matrices[rel_type_idx] += self._rp_mrel_matrices[rel_type_idx]

        if embed_lr > 0:
            d_source_latent = W_rel @ d_logits_proj
            d_target_latent_proj = W_rel @ source_latent
            d_target_latents = np.outer(d_logits, d_target_latent_proj)
            # Note: token_embed is external, updated by caller

        self._rp_cache = None

    # ── Cross-Domain Alignment ──

    def cross_domain_relation_alignment(self, graph: ConceptGraph,
                                         binding_map, token_embed, lr=None) -> Dict[str, float]:
        """Align W_rel across domains by minimizing mean residual."""
        if lr is None:
            lr = self.alignment_lr

        pairs_by_rel = defaultdict(list)
        for (src_cid, tgt_cid), edge in graph.edges.items():
            rel_name = edge.relation_type
            if rel_name not in RELATION_TYPES:
                continue
            if edge.weight < 0.2 or edge.confidence < 0.2:
                continue
            src_tokens = binding_map.get_tokens(src_cid, min_confidence=0.1)
            tgt_tokens = binding_map.get_tokens(tgt_cid, min_confidence=0.1)
            if not src_tokens or not tgt_tokens:
                continue
            src_tid = src_tokens[0].token_id
            tgt_tid = tgt_tokens[0].token_id
            if src_tid >= self.vocab_size or tgt_tid >= self.vocab_size:
                continue
            pairs_by_rel[rel_name].append((src_tid, tgt_tid))

        results = {}
        for rel_name, pairs in pairs_by_rel.items():
            if len(pairs) < 2:
                continue
            rel_idx = RELATION_TYPES.index(rel_name)
            W_rel = self._rp_rel_matrices[rel_idx]
            grad_sum = np.zeros_like(W_rel)
            for src_tid, tgt_tid in pairs:
                s = token_embed.weight.data[src_tid]
                t = token_embed.weight.data[tgt_tid]
                pred = s @ W_rel
                residual = pred - t
                grad_sum += np.outer(s, residual)
            grad_mean = grad_sum / len(pairs)
            gn = np.linalg.norm(grad_mean)
            if gn > 5.0:
                grad_mean *= (5.0 / (gn + 1e-15))
            self._rp_rel_matrices[rel_idx] -= lr * grad_mean
            results[rel_name] = float(np.linalg.norm(grad_mean))
        return results

    def measure_cross_domain_alignment(self, graph: ConceptGraph,
                                        binding_map, token_embed) -> Dict[str, float]:
        """Measure alignment quality across all pairs."""
        pairs_by_rel = defaultdict(list)
        for (src_cid, tgt_cid), edge in graph.edges.items():
            rel_name = edge.relation_type
            if rel_name not in RELATION_TYPES or edge.weight < 0.2:
                continue
            src_tokens = binding_map.get_tokens(src_cid, min_confidence=0.1)
            tgt_tokens = binding_map.get_tokens(tgt_cid, min_confidence=0.1)
            if not src_tokens or not tgt_tokens:
                continue
            src_tid, tgt_tid = src_tokens[0].token_id, tgt_tokens[0].token_id
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
                s = token_embed.weight.data[src_tid]
                t = token_embed.weight.data[tgt_tid]
                pred = s @ W_rel
                sn, tn = np.linalg.norm(pred), np.linalg.norm(t)
                if sn > 0 and tn > 0:
                    sims.append(float(np.dot(pred, t) / (sn * tn)))
            if sims:
                results[rel_name] = float(np.mean(sims))
        return results

    def inject_cross_domain_edge(self, graph: ConceptGraph,
                                  subject_cid: int, object_cid: int,
                                  rel_name: str, subject_tid: Optional[int] = None):
        """Inject analogical edge across domains."""
        if rel_name not in RELATION_TYPES:
            return
        key = (subject_cid, object_cid, rel_name)
        if key in getattr(self, '_cross_domain_edges_injected', set()):
            return
        if not hasattr(self, '_cross_domain_edges_injected'):
            self._cross_domain_edges_injected = set()
        edge = graph.get_edge(subject_cid, object_cid)
        if edge is None:
            edge = graph.add_edge(source=subject_cid, target=object_cid,
                                  weight=0.5, relation_type=rel_name)
        else:
            if edge.relation_type == rel_name:
                edge.weight = min(1.0, edge.weight + 0.2)
                edge.confidence = min(1.0, edge.confidence + 0.1)
        if subject_tid is not None and 0 <= subject_tid < self.vocab_size:
            edge.predicate_token_id = subject_tid
        self._cross_domain_edges_injected.add(key)

    # ── State Management ──

    def state_dict(self) -> dict:
        return {
            "relation_type_embed": self.relation_type_embed.weight.data.copy(),
            "relation_classifier_weight": self.relation_classifier.weight.data.copy(),
            "relation_classifier_bias": self.relation_classifier.bias.data.copy() if self.relation_classifier.bias is not None else None,
            "_rp_rel_matrices": self._rp_rel_matrices.copy(),
            "_rp_mrel_matrices": self._rp_mrel_matrices.copy(),
            "_enc_W1": self._enc_W1.copy(), "_enc_b1": self._enc_b1.copy(),
            "_enc_W2": self._enc_W2.copy(), "_enc_b2": self._enc_b2.copy(),
            "_enc_mW1": self._enc_mW1.copy(), "_enc_mb1": self._enc_mb1.copy(),
            "_enc_mW2": self._enc_mW2.copy(), "_enc_mb2": self._enc_mb2.copy(),
            "_dec_W1": self._dec_W1.copy(), "_dec_b1": self._dec_b1.copy(),
            "_dec_W2": self._dec_W2.copy(), "_dec_b2": self._dec_b2.copy(),
            "_rp_W1": self._rp_W1.copy(), "_rp_b1": self._rp_b1.copy(),
            "_rp_W2": self._rp_W2.copy(), "_rp_b2": self._rp_b2.copy(),
            "_rp_mW1": self._rp_mW1.copy(), "_rp_mb1": self._rp_mb1.copy(),
            "_rp_mW2": self._rp_mW2.copy(), "_rp_mb2": self._rp_mb2.copy(),
            "_verb_offsets": {k: v.copy() for k, v in self._verb_offsets.items()},
            "_verb_offset_count": self._verb_offset_count.copy(),
            "use_verb_offset": self.use_verb_offset,
            "freeze_token_embeds_in_rp": self.freeze_token_embeds_in_rp,
            "use_cross_domain_alignment": self.use_cross_domain_alignment,
            "current_domain_id": self.current_domain_id,
            "_frozen_domains": self._frozen_domains.copy(),
        }

    def load_state(self, state: dict):
        self.relation_type_embed.weight.data = state["relation_type_embed"]
        self.relation_classifier.weight.data = state["relation_classifier_weight"]
        if state["relation_classifier_bias"] is not None:
            self.relation_classifier.bias.data = state["relation_classifier_bias"]
        self._rp_rel_matrices = state["_rp_rel_matrices"]
        self._rp_mrel_matrices = state["_rp_mrel_matrices"]
        self._enc_W1 = state["_enc_W1"]; self._enc_b1 = state["_enc_b1"]
        self._enc_W2 = state["_enc_W2"]; self._enc_b2 = state["_enc_b2"]
        self._enc_mW1 = state["_enc_mW1"]; self._enc_mb1 = state["_enc_mb1"]
        self._enc_mW2 = state["_enc_mW2"]; self._enc_mb2 = state["_enc_mb2"]
        self._dec_W1 = state["_dec_W1"]; self._dec_b1 = state["_dec_b1"]
        self._dec_W2 = state["_dec_W2"]; self._dec_b2 = state["_dec_b2"]
        self._rp_W1 = state["_rp_W1"]; self._rp_b1 = state["_rp_b1"]
        self._rp_W2 = state["_rp_W2"]; self._rp_b2 = state["_rp_b2"]
        self._rp_mW1 = state["_rp_mW1"]; self._rp_mb1 = state["_rp_mb1"]
        self._rp_mW2 = state["_rp_mW2"]; self._rp_mb2 = state["_rp_mb2"]
        self._verb_offsets = {k: v.copy() for k, v in state.get("_verb_offsets", {}).items()}
        self._verb_offset_count = state.get("_verb_offset_count", {}).copy()
        self.use_verb_offset = state.get("use_verb_offset", False)
        self.freeze_token_embeds_in_rp = state.get("freeze_token_embeds_in_rp", True)
        self.use_cross_domain_alignment = state.get("use_cross_domain_alignment", False)
        self.current_domain_id = state.get("current_domain_id", None)
        self._frozen_domains = state.get("_frozen_domains", set()).copy()