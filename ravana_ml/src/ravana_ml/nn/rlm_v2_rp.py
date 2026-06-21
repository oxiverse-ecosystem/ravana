"""
Mixin: RPMixin — rlm_v2_rp methods for RLMv2.

Auto-extracted from rlm_v2.py. Edit in the source or directly here.
"""
import numpy as np
from typing import Optional, List, Tuple, Dict, Set, Any


class RPMixin:
    # ── LSH Token Scoring (P3 Sprint 6) ──
    def _lsh_init(self, n_buckets: int = 8, n_hashes: int = 3):
        """Initialise LSH random projection hash for token scoring.

        Hashes token embeddings into buckets using random projections (simhash).
        h(x) = sign(R . x) where each row of R is a
        random unit vector.  n_hashes such hash functions give a (n_hashes,
        n_buckets)-sensitive LSH family.

        Args:
            n_buckets: Number of buckets per hash function (default 16).
            n_hashes: Number of independent hash functions (default 4).
        """
        rng = np.random.RandomState(42)
        self._lsh_n_buckets = n_buckets
        self._lsh_n_hashes = n_hashes
        # Each hash uses a random projection matrix (latent_dim,)
        self._lsh_projections = rng.randn(n_hashes, self.latent_dim).astype(np.float32)
        for i in range(n_hashes):
            nrm = np.linalg.norm(self._lsh_projections[i])
            if nrm > 0:
                self._lsh_projections[i] /= nrm
        # Pre-computed bucket assignment for all tokens (built on first use)
        self._lsh_buckets: Optional[np.ndarray] = None
        self._lsh_bucket_map: Optional[Dict[int, List[int]]] = None
        self._lsh_dirty: bool = True  # invalidated when token_embeds change

    def _lsh_hash(self, embeddings: np.ndarray) -> np.ndarray:
        """Hash embeddings into LSH buckets.

        Args:
            embeddings: (N, latent_dim) matrix.

        Returns:
            bucket_ids: (N,) integer bucket IDs (0 .. n_buckets**n_hashes - 1).
        """
        N = embeddings.shape[0]
        # projections: (n_hashes, latent_dim)
        # signs: (N, n_hashes)  {0, 1} after binarising
        dots = embeddings @ self._lsh_projections.T  # (N, n_hashes)
        signs = (dots > 0).astype(np.int32)
        # Combine into scalar bucket ID: bucket = sum(signs[h] * bucket_base^h)
        bucket_ids = np.zeros(N, dtype=np.int32)
        for h in range(self._lsh_n_hashes):
            bucket_ids += signs[:, h] * (self._lsh_n_buckets ** h)
        return bucket_ids

    def _lsh_build_buckets(self):
        """Build or refresh the LSH bucket -> token list map."""
        if not self._lsh_dirty and self._lsh_bucket_map is not None:
            return
        token_embeds = self.token_embed.weight.data
        if self._rp_use_encoder_latent:
            token_embeds, _, _, _ = self._encoder_forward_full(token_embeds)
        bucket_ids = self._lsh_hash(token_embeds)
        self._lsh_buckets = bucket_ids
        bucket_map: Dict[int, List[int]] = {}
        for tid, bid in enumerate(bucket_ids):
            bucket_map.setdefault(int(bid), []).append(tid)
        self._lsh_bucket_map = bucket_map
        self._lsh_dirty = False

    def _lsh_scoring(self, source_latent: np.ndarray, target_latents: np.ndarray,
                     W_rel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Score only tokens in the same LSH bucket as the source.

        Falls back to full scoring if LSH is not initialised or the source
        bucket is too large.

        Args:
            source_latent: (latent_dim,) source embedding.
            target_latents: (vocab_size, latent_dim) target embeddings.
            W_rel: (latent_dim, latent_dim) composed relation matrix.

        Returns:
            (logits, candidate_mask) where candidate_mask is boolean (vocab_size,).
        """
        if not hasattr(self, '_lsh_n_hashes'):
            # Lazy init on first use (P3 Sprint 6)
            self._lsh_init()

        self._lsh_build_buckets()
        source_hash = self._lsh_hash(source_latent.reshape(1, -1))[0]
        candidates = self._lsh_bucket_map.get(int(source_hash), [])
        if not candidates or len(candidates) > len(target_latents) // 2:
            return None, None  # Fallback: bucket too large or empty

        projected = source_latent @ W_rel
        cand_embeds = target_latents[candidates]
        cand_logits = projected @ cand_embeds.T

        # Build full-size logits, fill candidates, leave others at -inf
        logits = np.full(len(target_latents), -np.inf, dtype=np.float32)
        for j, tid in enumerate(candidates):
            logits[tid] = cand_logits[j]

        return logits, (0, len(candidates))  # sparsity info


    """Mixin providing rlm_v2_rp methods for RLMv2."""



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

            subject_tid, rel_type_idx, domain_id,

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



        # === Gradient w.r.t. Low-Rank W_rel factors (P2 Sprint 5) ===

        d_logits_proj = d_logits @ target_latents

        dW_rel = np.outer(source_latent, d_logits_proj)



        # Gradient clipping (prevent NaN from bilinear amplification)

        grad_norm = np.linalg.norm(dW_rel)

        if grad_norm > 10.0:

            dW_rel *= (10.0 / (grad_norm + 1e-15))



        lr = self._rp_lr * lr_scale

        freeze_token_embeds = getattr(self, 'freeze_token_embeds_in_rp', True)

        embed_lr = 0.0 if freeze_token_embeds or self._rp_use_encoder_latent else lr * 0.1



        # Low-rank gradient routing:
        # W_rel = diag(W_base[type]) + U_d[domain,type].T @ V_d[domain,type]
        # dL/dW_base[type] = diag(dW_rel)
        # dL/dU = V @ dW_rel.T  (shape: rank, latent_dim)
        # dL/dV = U @ dW_rel   (shape: rank, latent_dim)
        U = self._rp_U_d[domain_id, rel_type_idx]
        V = self._rp_V_d[domain_id, rel_type_idx]

        d_w = np.diag(dW_rel)
        dU = V @ dW_rel.T
        dV = U @ dW_rel

        # Gradient clipping for low-rank factors (separate from full dW_rel)
        for g in [dU, dV]:
            gn = np.linalg.norm(g)
            if gn > 10.0:
                g *= (10.0 / (gn + 1e-15))

        # Momentum updates
        self._rp_mW_base[rel_type_idx] = (
            self._rp_momentum * self._rp_mW_base[rel_type_idx] - lr * d_w
        )
        self._rp_mU_d[domain_id, rel_type_idx] = (
            self._rp_momentum * self._rp_mU_d[domain_id, rel_type_idx] - lr * dU
        )
        self._rp_mV_d[domain_id, rel_type_idx] = (
            self._rp_momentum * self._rp_mV_d[domain_id, rel_type_idx] - lr * dV
        )

        self._rp_W_base[rel_type_idx] += self._rp_mW_base[rel_type_idx]
        self._rp_U_d[domain_id, rel_type_idx] += self._rp_mU_d[domain_id, rel_type_idx]
        self._rp_V_d[domain_id, rel_type_idx] += self._rp_mV_d[domain_id, rel_type_idx]



        if self._rp_use_encoder_latent:

            self._rp_cache = None

            return

        

        # === Gradients w.r.t. token embeddings ===

        d_source_latent = W_rel @ d_logits_proj

        d_target_latent_proj = W_rel @ source_latent

        d_target_latents = np.outer(d_logits, d_target_latent_proj)



        # === Gradient w.r.t. Entity Adapter (Fix #3) ===

        adapter = self._entity_adapters.get(subject_tid)

        if adapter is not None:

            U, V = adapter  # U: (rank, latent_dim), V: (rank, latent_dim)

            mU, mV = self._entity_adapter_momentums[subject_tid]

            

            # Need gradient of loss w.r.t source_latent BEFORE adapter

            # The adapter does: source_latent -> source_latent @ U.T @ V

            # Let's call the adapted latent: adapted = source_latent @ U.T @ V

            # dL/d_source_latent = dL/d_adapted @ V.T @ U

            

            # dL/d_adapted is what we'd normally compute for embeddings

            d_adapted = d_source_latent  # this is dL/d_adapted

            

            # Let z = source_latent @ U.T (1, rank)

            # adapted = z @ V (1, latent_dim)

            # dL/dV = outer(z, d_adapted)

            # dL/dU = outer(source_latent, d_adapted @ V.T)

            

            z = source_latent @ U.T  # (rank,)

            dV = np.outer(z, d_adapted)  # (rank, latent_dim)

            dU = np.outer(d_adapted @ V.T, source_latent)  # (rank, latent_dim)

            

            # Clip gradients

            for g in [dU, dV]:

                gn = np.linalg.norm(g)

                if gn > 5.0:

                    g *= (5.0 / (gn + 1e-15))

            

            # Momentum update

            adapter_lr = self._entity_adapter_lr

            mU = self._entity_adapter_momentum * mU - adapter_lr * dU

            mV = self._entity_adapter_momentum * mV - adapter_lr * dV

            U += mU

            V += mV

            

            self._entity_adapters[subject_tid] = (U, V)

            self._entity_adapter_momentums[subject_tid] = (mU, mV)



        if embed_lr > 0:

            self.token_embed.weight.data[subject_tid] -= embed_lr * d_source_latent

            self.token_embed.weight.data -= embed_lr * d_target_latents

            self._token_embed_norms = None



        self._rp_cache = None

    # ── Spreading Activation Inference ──────────────────────────────────────






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

        # When use_verb_offset is True, compute verb offset logits and blend

        # with W_rel output using confidence-weighted interpolation.

        # For well-trained verbs (count >= 10), verb offset dominates.

        # For rare verbs, fall back more heavily on W_rel.

        # IMPORTANT: Verb offsets are computed on RAW embeddings (target - subject),

        # so we use the raw subject embedding here, NOT the entity-adapted one.

        verb_blend_logits = None

        verb_blend_weight = 0.0

        if verb_word and self.use_verb_offset:

            verb_result = self._rp_forward_verb_offset(subject_tid, verb_word, return_count=True)

            if verb_result is not None and verb_result[0] is not None:

                verb_logits, verb_count, verb_variance = verb_result

                # Compute blend weight: count-based S-curve modified by variance
                base_blend = verb_count / (verb_count + 5.0)
                # Variance penalty: higher variance reduces effective blend weight
                variance_penalty = max(0.1, 1.0 - verb_variance)
                verb_blend_weight = base_blend * variance_penalty
                verb_blend_weight = float(np.clip(verb_blend_weight, 0.0, 1.0))

                verb_blend_logits = verb_logits

                if verb_blend_weight > 0.95:

                    # Fully confident in verb offset - return directly (no W_rel needed)

                    self._rp_cache = None

                    return verb_logits



        # Get source embedding and apply entity-specific adapter (Fix #3)

        # Only for bilinear W_rel fallback path

        source_embed = self.get_robust_embedding(subject_tid)  # (embed_dim,)

        

        # Project to latent space if needed before applying adapter

        if self._rp_use_encoder_latent:

            source_embed, _, _, _ = self._encoder_forward_full(source_embed)

        

        # For held-out subjects, initialize adapter from nearest neighbor

        adapter = self._get_or_adapt_entity_adapter(subject_tid, verb_word=verb_word)

        if adapter is not None:

            U, V = adapter  # U: (rank, latent_dim), V: (rank, latent_dim)

            # source_embed @ U.T gives (rank,), then @ V gives (latent_dim,)

            source_embed = (source_embed @ U.T) @ V



        # ── Verb Offset Path with Adapted Source (test-time adaptation for held-out) ──

        # If we have a verb offset AND the source was adapted (or _test_time_adapt_mode is set),

        # use cosine scoring with the adapted source + offset.

        # This enables test-time adaptation to generalize to non-functional verbs.

        if verb_word and self.use_verb_offset and getattr(self, '_test_time_adapt_mode', False):

            stem = self._verb_stem(verb_word)

            domain_id = self.current_domain_id if self.current_domain_id is not None else 0

            if domain_id in self._verb_offsets and stem in self._verb_offsets[domain_id]:

                offset = self._verb_offsets[domain_id][stem]

                predicted = source_embed + offset

                

                # Cosine similarity against all token embeddings

                token_embeds = self.token_embed.weight.data

                token_norms = np.linalg.norm(token_embeds, axis=1)

                pred_norm = np.linalg.norm(predicted)

                

                if pred_norm > 0 and np.any(token_norms > 0):

                    valid_tok = token_norms > 0

                    normed_tok = token_embeds.copy()

                    normed_tok[valid_tok] /= token_norms[valid_tok, np.newaxis]

                    logits = (predicted / pred_norm) @ normed_tok.T

                    # Suppress subject token

                    if 0 <= subject_tid < len(logits):

                        logits[subject_tid] = np.min(logits) - 10.0

                    logits *= 10.0

                    self._rp_cache = None

                    return logits



        # ── Bilinear W_rel Path (fallback) ──

        token_embeds = self.token_embed.weight.data  # (vocab_size, embed_dim)



        if self._rp_use_encoder_latent:

            # If we already projected source_embed (for adapter), it's already in latent space

            if source_embed.shape[-1] == self.latent_dim:

                source_latent = source_embed

            else:

                source_latent, _, _, _ = self._encoder_forward_full(source_embed)

            target_latents, _, _, _ = self._encoder_forward_full(token_embeds)

        else:

            source_latent = source_embed  # (embed_dim,)

            target_latents = token_embeds  # (vocab_size, embed_dim)



        # ── Low-Rank W_rel Composition (P2 Sprint 5) ──
        # W_rel[domain, type] = diag(W_base[type]) + U_d[domain,type].T @ V_d[domain,type]
        domain_id = self.current_domain_id if self.current_domain_id is not None else 0

        w_vec = self._rp_W_base[rel_type_idx]           # (latent_dim,)
        U = self._rp_U_d[domain_id, rel_type_idx]        # (rank, latent_dim)
        V = self._rp_V_d[domain_id, rel_type_idx]        # (rank, latent_dim)
        W_rel = np.diag(w_vec).astype(np.float32) + U.T @ V  # (latent_dim, latent_dim)

        # ── LSH-Accelerated Scoring (P3 Sprint 6) ──
        # Hash source latent into LSH bucket and score only tokens sharing that bucket.
        # Falls back to full O(V) bilinear scoring when LSH is not initialised.
        lsh_result = self._lsh_scoring(source_latent, target_latents, W_rel)
        if lsh_result[0] is not None:
            logits = lsh_result[0]
        else:
            # Fallback: full bilinear scoring
            projected = source_latent @ W_rel
            logits = projected @ target_latents.T



        # ── Blend verb offset with W_rel (confidence-weighted) ──

        # When verb offset is available (even for rare verbs), blend it with

        # the bilinear W_rel output. High-frequency verbs dominate; rare verbs

        # contribute proportionally.

        if verb_blend_logits is not None and verb_blend_weight > 0.0:

            # Guard: verb_blend_logits must not be a numpy array with False truth value

            if isinstance(verb_blend_logits, np.ndarray) and verb_blend_logits.size > 0:

                # Blend: final = blend_weight * verb_logits + (1 - blend_weight) * w_rel_logits

                blended = verb_blend_weight * verb_blend_logits + (1.0 - verb_blend_weight) * logits

                # Cache W_rel path for backprop (verb offsets are not trained via backprop)

                self._rp_cache = (

                    subject_tid, rel_type_idx, domain_id,

                    source_embed, source_latent,

                    token_embeds, target_latents,

                    W_rel, blended

                )

                return blended



        # Cache for backprop

        self._rp_cache = (

            subject_tid, rel_type_idx, domain_id,

            source_embed, source_latent,

            token_embeds, target_latents,

            W_rel, logits

        )

        return logits



