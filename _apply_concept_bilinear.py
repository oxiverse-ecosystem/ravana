"""Apply concept-vector-based bilinear RP to rlm_v2.py.

Key change: Use Hebbian-trained concept node vectors (via binding map)
instead of frozen token embeddings for the bilinear scoring. This gives
W_rel meaningful learned representations to transform.

Fallback: For tokens without concept nodes, use encoder-projected embeddings.
"""
import sys

path = 'ravana_ml/nn/rlm_v2.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# ============================================================
# Change 1: Replace _rp_forward with concept-vector-based bilinear
# ============================================================
idx_fwd = content.find('def _rp_forward(self, subject_tid, rel_type_idx, route_softly=None):')
if idx_fwd >= 0:
    end_search = idx_fwd + len('def _rp_forward(self, subject_tid, rel_type_idx, route_softly=None):')
    next_def = content.find('\n    def ', end_search)
    if next_def >= 0:
        old_fwd = content[idx_fwd:next_def]
        new_fwd = (
            '    def _rp_forward(self, subject_tid, rel_type_idx, route_softly=None):\n'
            '        """Bilinear relation predictor forward pass using concept vectors.\n'
            '\n'
            '        Uses Hebbian-trained concept node vectors (via binding map) for\n'
            '        both source and target representations. Concept vectors encode\n'
            '        learned semantics from training, unlike frozen n-gram embeddings.\n'
            '        Falls back to encoder-projected embeddings for unknown tokens.\n'
            '        """\n'
            '        subject_tid = int(subject_tid)\n'
            '        rel_type_idx = int(rel_type_idx)\n'
            '\n'
            '        # Get source embedding (fallback)\n'
            '        source_embed = self.get_robust_embedding(subject_tid)  # (embed_dim,)\n'
            '\n'
            '        # Try to use concept node vector for source (Hebbian-trained semantics)\n'
            '        bindings = self.binding_map.get_concepts(subject_tid, min_confidence=0.1)\n'
            '        if bindings:\n'
            '            src_cid = bindings[0].concept_id\n'
            '            src_node = self.graph.get_node(src_cid)\n'
            '            if src_node is not None and np.linalg.norm(src_node.vector) > 0:\n'
            '                source_latent = src_node.vector.copy()  # (concept_dim,)\n'
            '            else:\n'
            '                source_latent, _, _, _ = self._encoder_forward_full(source_embed)  # (latent_dim,)\n'
            '        else:\n'
            '            source_latent, _, _, _ = self._encoder_forward_full(source_embed)  # (latent_dim,)\n'
            '\n'
            '        # Handle dimension mismatch: project concept vector if needed\n'
            '        if len(source_latent) != self.latent_dim:\n'
            '            source_latent, _, _, _ = self._encoder_forward_full(source_latent)  # (latent_dim,)\n'
            '\n'
            '        # Build target latent matrix using concept vectors where available\n'
            '        token_embeds = self.token_embed.weight.data  # (vocab_size, embed_dim)\n'
            '        target_latents = np.zeros((self.vocab_size, self.latent_dim), dtype=np.float32)\n'
            '        concept_dim = len(src_node.vector) if src_node is not None else self.embed_dim\n'
            '\n'
            '        for tok_id in range(self.vocab_size):\n'
            '            bindings_t = self.binding_map.get_concepts(tok_id, min_confidence=0.1)\n'
            '            if bindings_t:\n'
            '                tgt_cid = bindings_t[0].concept_id\n'
            '                tgt_node = self.graph.get_node(tgt_cid)\n'
            '                if tgt_node is not None and np.linalg.norm(tgt_node.vector) > 0:\n'
            '                    tgt_vec = tgt_node.vector  # (concept_dim,)\n'
            '                    if len(tgt_vec) == self.latent_dim:\n'
            '                        target_latents[tok_id] = tgt_vec\n'
            '                    else:\n'
            '                        tgt_lat, _, _, _ = self._encoder_forward_full(tgt_vec)\n'
            '                        target_latents[tok_id] = tgt_lat\n'
            '                else:\n'
            '                    tgt_lat, _, _, _ = self._encoder_forward_full(token_embeds[tok_id])\n'
            '                    target_latents[tok_id] = tgt_lat\n'
            '            else:\n'
            '                tgt_lat, _, _, _ = self._encoder_forward_full(token_embeds[tok_id])\n'
            '                target_latents[tok_id] = tgt_lat\n'
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
            '            source_embed, source_latent, W_rel, target_latents, logits\n'
            '        )\n'
            '        return logits'
        )
        content = content.replace(old_fwd, new_fwd, 1)
        changes += 1
        print(f'Change 1: Concept-vector bilinear _rp_forward ({len(old_fwd)} -> {len(new_fwd)} chars)')
    else:
        print('Change 1: FAILED - no next def found')
else:
    print('Change 1: FAILED - _rp_forward not found')

# ============================================================
# Change 2: Replace _rp_backward (target_latents instead of token_embeds)
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
            '        """Bilinear relation predictor backward pass (concept-vector based).\n'
            '\n'
            '        logits = source_latent @ W_rel @ target_latents.T\n'
            '        dW_rel = outer(source_latent, d_logits @ target_latents)\n'
            '\n'
            '        Updates the shared relation matrix W_rel using concept-vector\n'
            '        representations (Hebbian-trained, semantically meaningful).\n'
            '        """\n'
            '        if self._rp_cache is None:\n'
            '            return\n'
            '            \n'
            '        (\n'
            '            subject_tid, rel_type_idx,\n'
            '            source_embed, source_latent, W_rel, target_latents, logits\n'
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
            '        # Bilinear gradient: dW_rel = outer(source_latent, d_logits @ target_latents)\n'
            '        d_logits_proj = d_logits @ target_latents  # (latent_dim,)\n'
            '        dW_rel = np.outer(source_latent, d_logits_proj)  # (latent_dim, latent_dim)\n'
            '\n'
            '        # Gradient clipping to prevent NaN from bilinear amplification\n'
            '        grad_norm = np.linalg.norm(dW_rel)\n'
            '        if grad_norm > 10.0:\n'
            '            dW_rel *= (10.0 / (grad_norm + 1e-15))\n'
            '\n'
            '        lr = self._rp_lr * lr_scale\n'
            '\n'
            '        # Update the SHARED relation matrix with momentum\n'
            '        self._rp_mrel_matrices[rel_type_idx] = (\n'
            '            self._rp_momentum * self._rp_mrel_matrices[rel_type_idx] - lr * dW_rel\n'
            '        )\n'
            '        self._rp_rel_matrices[rel_type_idx] += self._rp_mrel_matrices[rel_type_idx]\n'
            '\n'
            '        self._rp_cache = None'
        )
        content = content.replace(old_bwd, new_bwd, 1)
        changes += 1
        print(f'Change 2: Concept-vector bilinear _rp_backward ({len(old_bwd)} -> {len(new_bwd)} chars)')
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
