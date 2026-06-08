"""Apply embedding-trained bilinear RP to rlm_v2.py.

Root cause of 0% cross-domain transfer:
- The encoder is autoencoder-pretrained (300 epochs, LR=0.01) which collapses all
  latents to similar values (low reconstruction error → latents near origin)
- RP gradient (rp_scale=0.1, _rp_encoder_lr=0.005) is too weak to overcome this
- Result: source_latent ~ target_latent for all tokens → uniform logits → 0% transfer

Fix: BYPASS the collapsed encoder. Use raw token embeddings directly and train
them end-to-end with the bilinear scoring, just like standard KG completion
(RESCAL, DistMult, etc.).
"""
import sys

path = 'ravana_ml/nn/rlm_v2.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# ============================================================
# Change: Replace _rp_forward with embedding-based bilinear (bypass encoder)
# ============================================================
idx_fwd = content.find('def _rp_forward(self, subject_tid, rel_type_idx, route_softly=None):')
if idx_fwd >= 0:
    end_search = idx_fwd + len('def _rp_forward(self, subject_tid, rel_type_idx, route_softly=None):')
    next_def = content.find('\n    def ', end_search)
    if next_def >= 0:
        old_fwd = content[idx_fwd:next_def]
        new_fwd = (
            '    def _rp_forward(self, subject_tid, rel_type_idx, route_softly=None):\n'
            '        """Bilinear RP forward using raw token embeddings (bypass collapsed encoder).\n'
            '\n'
            '        Uses source_embed directly (no encoder projection) and target_embeds\n'
            '        directly. The bilinear gradient trains token embeddings end-to-end,\n'
            '        just like standard KG completion (RESCAL).\n'
            '\n'
            '        logits_k = source_embed @ W_rel @ target_embed_k\n'
            '        """\n'
            '        subject_tid = int(subject_tid)\n'
            '        rel_type_idx = int(rel_type_idx)\n'
            '\n'
            '        # Use raw token embeddings directly (bypass collapsed encoder)\n'
            '        source_embed = self.get_robust_embedding(subject_tid)  # (embed_dim,)\n'
            '        token_embeds = self.token_embed.weight.data  # (vocab_size, embed_dim)\n'
            '\n'
            '        # Source latent = source embedding (preserves discriminative info)\n'
            '        source_latent = source_embed  # (embed_dim,) — must match latent_dim\n'
            '        target_latents = token_embeds  # (vocab_size, embed_dim) — must match latent_dim\n'
            '\n'
            '        # Relation matrix (shared across ALL subjects)\n'
            '        W_rel = self._rp_rel_matrices[rel_type_idx]  # (latent_dim, latent_dim)\n'
            '\n'
            '        # Bilinear scoring: logits = source_latent @ W_rel @ target_latents.T\n'
            '        projected = source_latent @ W_rel  # (latent_dim,)\n'
            '        logits = projected @ target_latents.T  # (vocab_size,)\n'
            '\n'
            '        # Cache for backprop\n'
            '        self._rp_cache = (\n'
            '            subject_tid, rel_type_idx,\n'
            '            source_embed, source_latent,\n'
            '            token_embeds, target_latents,\n'
            '            W_rel, logits\n'
            '        )\n'
            '        return logits'
        )
        content = content.replace(old_fwd, new_fwd, 1)
        changes += 1
        print(f'Change 1: Embedding-bypass _rp_forward ({len(old_fwd)} -> {len(new_fwd)} chars)')
    else:
        print('Change 1: FAILED - no next def found')
else:
    print('Change 1: FAILED - _rp_forward not found')

# ============================================================
# Change: Replace _rp_backward with embedding gradient flow
# ============================================================
idx_bwd = content.find('def _rp_backward(self, target_id, lr_scale=1.0):')
if idx_bwd >= 0:
    end_search = idx_bwd + len('def _rp_backward(self, target_id, lr_scale=1.0):')
    next_def = content.find('\n    def ', end_search)
    next_section = content.find('\n    # ── Spreading Activation', end_search)
    if next_section >= 0 and (next_def < 0 or next_section < next_def):
        end_idx = next_section
    elif next_def >= 0:
        end_idx = next_def
    else:
        print('Change 2: FAILED - no end found')
        end_idx = -1

    if end_idx >= 0:
        old_bwd = content[idx_bwd:end_idx]
        new_bwd = (
            '    def _rp_backward(self, target_id, lr_scale=1.0):\n'
            '        """Bilinear RP backward with token embedding gradient flow.\n'
            '\n'
            '        Gradients flow through:\n'
            '        1. W_rel via dW_rel = outer(source_latent, d_logits @ target_latents)\n'
            '        2. Source embedding: d_source_embed = W_rel @ (d_logits @ target_latents)\n'
            '        3. Target embeddings: d_target_latents[k] = d_logits[k] * (W_rel @ source_latent)\n'
            '\n'
            '        This trains token embeddings end-to-end with the bilinear scoring,\n'
            '        just like standard KG completion (RESCAL).\n'
            '        """\n'
            '        if self._rp_cache is None:\n'
            '            return\n'
            '            \n'
            '        (\n'
            '            subject_tid, rel_type_idx,\n'
            '            source_embed, source_latent,\n'
            '            token_embeds, target_latents,\n'
            '            W_rel, logits\n'
            '        ) = self._rp_cache\n'
            '        \n'
            '        # Softmax loss gradient\n'
            '        exp_logits = np.exp(logits - np.max(logits))\n'
            '        probs = exp_logits / (np.sum(exp_logits) + 1e-10)\n'
            '        d_logits = probs.copy()\n'
            '        if 0 <= target_id < len(d_logits):\n'
            '            d_logits[target_id] -= 1.0\n'
            '        d_logits *= getattr(self, "rp_scale", 16.0)\n'
            '\n'
            '        # === Gradient w.r.t. W_rel ===\n'
            '        d_logits_proj = d_logits @ target_latents  # (latent_dim,)\n'
            '        dW_rel = np.outer(source_latent, d_logits_proj)  # (latent_dim, latent_dim)\n'
            '\n'
            '        # Gradient clipping (prevent NaN from bilinear amplification)\n'
            '        grad_norm = np.linalg.norm(dW_rel)\n'
            '        if grad_norm > 10.0:\n'
            '            dW_rel *= (10.0 / (grad_norm + 1e-15))\n'
            '\n'
            '        # === Gradients w.r.t. token embeddings ===\n'
            '        # dL/d(source_embed) = W_rel @ (d_logits @ target_latents)\n'
            '        d_source_latent = W_rel @ d_logits_proj  # (latent_dim,)\n'
            '        \n'
            '        # dL/d(target_latents[k]) = d_logits[k] * (W_rel @ source_latent)\n'
            '        d_target_latent_proj = W_rel @ source_latent  # (latent_dim,)\n'
            '        # Full gradient matrix: d_logits (vocab_size,) * d_target_latent_proj (latent_dim,)\n'
            '        # = outer(d_logits, d_target_latent_proj)  # (vocab_size, latent_dim)\n'
            '        d_target_latents = np.outer(d_logits, d_target_latent_proj)  # (vocab_size, latent_dim)\n'
            '\n'
            '        lr = self._rp_lr * lr_scale\n'
            '        embed_lr = lr * 0.1  # slower LR for embeddings (shared with Hebbian path)\n'
            '\n'
            '        # Update the SHARED relation matrix with momentum\n'
            '        self._rp_mrel_matrices[rel_type_idx] = (\n'
            '            self._rp_momentum * self._rp_mrel_matrices[rel_type_idx] - lr * dW_rel\n'
            '        )\n'
            '        self._rp_rel_matrices[rel_type_idx] += self._rp_mrel_matrices[rel_type_idx]\n'
            '\n'
            '        # Update source embedding (trains it to be relation-aware)\n'
            '        self.token_embed.weight.data[subject_tid] -= embed_lr * d_source_latent\n'
            '\n'
            '        # Update ALL target embeddings (the relation-aware signal spreads)\n'
            '        # This is the key to KG completion: all entities get trained\n'
            '        self.token_embed.weight.data -= embed_lr * d_target_latents\n'
            '\n'
            '        self._rp_cache = None'
        )
        content = content.replace(old_bwd, new_bwd, 1)
        changes += 1
        print(f'Change 2: Embedding-backprop _rp_backward ({len(old_bwd)} -> {len(new_bwd)} chars)')
    else:
        print('Change 2: FAILED - end_idx not found')

# ============================================================
# Write back
# ============================================================
if changes > 0:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'\nApplied {changes} changes successfully!')
else:
    print('\nNo changes applied!')
