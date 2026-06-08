"""Apply ENCODER-trained bilinear RP to rlm_v2.py.

Key insight: The encoder was autoencoder-pretrained (loss 0.003) and
preserves discriminative information. The "collapse" comment was about
the OLD linear head needing small random logits. For the bilinear form,
we NEED structured latents.

Approach:
1. _rp_forward: pass source_embed through encoder -> source_latent
2. _rp_backward: propagate gradients through encoder (via _encoder_backward)
3. This lets the encoder learn relation-aware representations jointly with W_rel
4. Same as standard KG completion: trained entity embeddings + relation matrices
"""
import sys

path = 'ravana_ml/nn/rlm_v2.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# ============================================================
# Change: Replace _rp_forward with encoder-based bilinear
# ============================================================
idx_fwd = content.find('def _rp_forward(self, subject_tid, rel_type_idx, route_softly=None):')
if idx_fwd >= 0:
    end_search = idx_fwd + len('def _rp_forward(self, subject_tid, rel_type_idx, route_softly=None):')
    next_def = content.find('\n    def ', end_search)
    if next_def >= 0:
        old_fwd = content[idx_fwd:next_def]
        new_fwd = (
            '    def _rp_forward(self, subject_tid, rel_type_idx, route_softly=None):\n'
            '        """Bilinear RP forward with encoder-trained latents.\n'
            '\n'
            '        Passes source_embed through the encoder (autoencoder-pretrained)\n'
            '        to get structured latents, then scores via bilinear form.\n'
            '        Encoder gradients flow during backward, training it to produce\n'
            '        relation-aware representations jointly with W_rel.\n'
            '        """\n'
            '        subject_tid = int(subject_tid)\n'
            '        rel_type_idx = int(rel_type_idx)\n'
            '\n'
            '        # Get source embedding\n'
            '        source_embed = self.get_robust_embedding(subject_tid)  # (embed_dim,)\n'
            '\n'
            '        # Pass through encoder to get structured latent (autoencoder-pretrained)\n'
            '        source_latent, z1, h1, z2 = self._encoder_forward_full(source_embed)  # (latent_dim,)\n'
            '\n'
            '        # Project ALL token embeddings through encoder for target latents\n'
            '        # (batched: single matmul over full vocab)\n'
            '        token_embeds = self.token_embed.weight.data  # (vocab_size, embed_dim)\n'
            '        target_latents, t_z1, t_h1, t_z2 = self._encoder_forward_full(token_embeds)  # (vocab_size, latent_dim)\n'
            '\n'
            '        # Relation matrix (shared across ALL subjects)\n'
            '        W_rel = self._rp_rel_matrices[rel_type_idx]  # (latent_dim, latent_dim)\n'
            '\n'
            '        # Bilinear scoring: logits = source_latent @ W_rel @ target_latents.T\n'
            '        projected = source_latent @ W_rel  # (latent_dim,)\n'
            '        logits = projected @ target_latents.T  # (vocab_size,)\n'
            '\n'
            '        # Cache for backprop (includes encoder activations for gradient flow)\n'
            '        self._rp_cache = (\n'
            '            subject_tid, rel_type_idx,\n'
            '            source_embed, source_latent, z1, h1, z2,\n'
            '            token_embeds, target_latents, t_z1, t_h1, t_z2,\n'
            '            W_rel, logits\n'
            '        )\n'
            '        return logits'
        )
        content = content.replace(old_fwd, new_fwd, 1)
        changes += 1
        print(f'Change 1: Encoder-trained bilinear _rp_forward ({len(old_fwd)} -> {len(new_fwd)} chars)')
    else:
        print('Change 1: FAILED - no next def found')
else:
    print('Change 1: FAILED - _rp_forward not found')

# ============================================================
# Change: Replace _rp_backward with encoder gradient flow
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
            '        """Bilinear RP backward with encoder gradient flow.\n'
            '\n'
            '        Gradients flow through:\n'
            '        1. W_rel via dW_rel = outer(source_latent, d_logits @ target_latents)\n'
            '        2. Encoder via _encoder_backward (trains it to produce\n'
            '           relation-aware latents jointly with W_rel)\n'
            '        """\n'
            '        if self._rp_cache is None:\n'
            '            return\n'
            '            \n'
            '        (\n'
            '            subject_tid, rel_type_idx,\n'
            '            source_embed, source_latent, z1, h1, z2,\n'
            '            token_embeds, target_latents, t_z1, t_h1, t_z2,\n'
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
            '        # Gradient clipping\n'
            '        grad_norm = np.linalg.norm(dW_rel)\n'
            '        if grad_norm > 10.0:\n'
            '            dW_rel *= (10.0 / (grad_norm + 1e-15))\n'
            '\n'
            '        # === Gradient w.r.t. encoder (flows through source_latent) ===\n'
            '        # dL/d(source_latent) = W_rel @ d_logits_proj^T  (latent_dim,)\n'
            '        d_source_latent = W_rel @ d_logits_proj  # (latent_dim,)\n'
            '        \n'
            '        # Backprop through encoder (trains it to produce better latents)\n'
            '        if not getattr(self, "freeze_encoder", False):\n'
            '            d_enc_W1, d_enc_b1, d_enc_W2, d_enc_b2 = self._encoder_backward(\n'
            '                source_embed[np.newaxis, :],\n'
            '                z1[np.newaxis, :], h1[np.newaxis, :],\n'
            '                z2[np.newaxis, :], source_latent[np.newaxis, :],\n'
            '                d_source_latent[np.newaxis, :]\n'
            '            )\n'
            '        else:\n'
            '            d_enc_W1 = d_enc_b1 = d_enc_W2 = d_enc_b2 = 0\n'
            '\n'
            '        lr = self._rp_lr * lr_scale\n'
            '        enc_lr = getattr(self, "_rp_encoder_lr", 0.0001) * lr_scale\n'
            '\n'
            '        # Update the SHARED relation matrix with momentum\n'
            '        self._rp_mrel_matrices[rel_type_idx] = (\n'
            '            self._rp_momentum * self._rp_mrel_matrices[rel_type_idx] - lr * dW_rel\n'
            '        )\n'
            '        self._rp_rel_matrices[rel_type_idx] += self._rp_mrel_matrices[rel_type_idx]\n'
            '\n'
            '        # Update encoder parameters (fine-tune for relation-aware latents)\n'
            '        if not getattr(self, "freeze_encoder", False):\n'
            '            self._enc_mW1 = self._rp_momentum * self._enc_mW1 - enc_lr * d_enc_W1\n'
            '            self._enc_mb1 = self._rp_momentum * self._enc_mb1 - enc_lr * d_enc_b1\n'
            '            self._enc_mW2 = self._rp_momentum * self._enc_mW2 - enc_lr * d_enc_W2\n'
            '            self._enc_mb2 = self._rp_momentum * self._enc_mb2 - enc_lr * d_enc_b2\n'
            '            \n'
            '            self._enc_W1 += self._enc_mW1\n'
            '            self._enc_b1 += self._enc_mb1\n'
            '            self._enc_W2 += self._enc_mW2\n'
            '            self._enc_b2 += self._enc_mb2\n'
            '\n'
            '        self._rp_cache = None'
        )
        content = content.replace(old_bwd, new_bwd, 1)
        changes += 1
        print(f'Change 2: Encoder-gradient _rp_backward ({len(old_bwd)} -> {len(new_bwd)} chars)')
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
