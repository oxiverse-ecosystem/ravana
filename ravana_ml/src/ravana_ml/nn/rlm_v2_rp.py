"""
Mixin: RPMixin — rlm_v2_rp methods for RLMv2.

Auto-extracted from rlm_v2.py. Edit in the source or directly here.
"""
import numpy as np
from typing import Optional, List, Tuple, Dict, Set, Any


class RPMixin:
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



        # === Gradient w.r.t. W_rel ===

        d_logits_proj = d_logits @ target_latents

        dW_rel = np.outer(source_latent, d_logits_proj)



        # Gradient clipping (prevent NaN from bilinear amplification)

        grad_norm = np.linalg.norm(dW_rel)

        if grad_norm > 10.0:

            dW_rel *= (10.0 / (grad_norm + 1e-15))



        lr = self._rp_lr * lr_scale

        freeze_token_embeds = getattr(self, 'freeze_token_embeds_in_rp', True)

        embed_lr = 0.0 if freeze_token_embeds or self._rp_use_encoder_latent else lr * 0.1



        self._rp_mrel_matrices[domain_id, rel_type_idx] = (

            self._rp_momentum * self._rp_mrel_matrices[domain_id, rel_type_idx] - lr * dW_rel

        )

        self._rp_rel_matrices[domain_id, rel_type_idx] += self._rp_mrel_matrices[domain_id, rel_type_idx]



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



        # Relation matrix (domain-specific, per relation type)

        domain_id = self.current_domain_id if self.current_domain_id is not None else 0

        W_rel = self._rp_rel_matrices[domain_id, rel_type_idx]



        # Bilinear scoring: logits = source_latent @ W_rel @ target_latents.T

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



