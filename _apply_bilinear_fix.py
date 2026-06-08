"""Apply bilinear relation predictor changes to rlm_v2.py."""
import sys

path = 'ravana_ml/nn/rlm_v2.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# ============================================================
# Change 1: Init _rp_rel_matrices as identity
# ============================================================
old1 = (
    '        self._rp_rel_matrices = np.random.randn(n_rel_types, self.latent_dim, self.latent_dim).astype(np.float32) * np.sqrt(2.0 / self.latent_dim)\n'
    '        self._rp_mrel_matrices = np.zeros_like(self._rp_rel_matrices)'
)
new1 = (
    '        # Identity init: source_embed @ I @ target_embed = dot product (cosine-like)\n'
    '        # Gradients learn deviations from identity, enabling shared relation-specific transformations\n'
    '        self._rp_rel_matrices = np.tile(np.eye(self.latent_dim, dtype=np.float32), (n_rel_types, 1, 1))\n'
    '        self._rp_mrel_matrices = np.zeros_like(self._rp_rel_matrices)'
)
if old1 in content:
    content = content.replace(old1, new1, 1)
    changes += 1
    print('Change 1: Identity init for _rp_rel_matrices')
else:
    print('Change 1: FAILED - pattern not found')
    idx = content.find('self._rp_rel_matrices = np.random.randn')
    if idx >= 0:
        print(f'  Found at char {idx}')

# ============================================================
# Change 2: Replace _rp_forward with bilinear version
# ============================================================
idx_fwd = content.find('def _rp_forward(self, subject_tid, rel_type_idx, route_softly=None):')
if idx_fwd >= 0:
    end_search = idx_fwd + len('def _rp_forward(self, subject_tid, rel_type_idx, route_softly=None):')
    next_def = content.find('\n    def ', end_search)
    if next_def >= 0:
        old_fwd = content[idx_fwd:next_def]
        new_fwd = (
            '    def _rp_forward(self, subject_tid, rel_type_idx, route_softly=None):\n'
            '        """Bilinear relation predictor forward pass.\n'
            '\n'
            '        Score(source, rel, target) = source_latent @ W_rel @ target_embed\n'
            '\n'
            '        W_rel is SHARED across ALL subjects (one per relation type).\n'
            '        This enables held-out generalization: a held-out subject with the same\n'
            '        relation uses the SAME W_rel transformation as trained subjects,\n'
            '        unlike the old linear head which memorized subject->target per subject.\n'
            '        """\n'
            '        subject_tid = int(subject_tid)\n'
            '        rel_type_idx = int(rel_type_idx)\n'
            '\n'
            '        # Get source embedding (bypass collapsed encoder)\n'
            '        source_embed = self.get_robust_embedding(subject_tid)  # (embed_dim,)\n'
            '\n'
            '        # Project to latent space if dimensions differ\n'
            '        if self.latent_dim == self.embed_dim:\n'
            '            source_latent = source_embed  # identity\n'
            '            token_embeds = self.token_embed.weight.data  # (vocab_size, embed_dim)\n'
            '        else:\n'
            '            # Use encoder for projection (loses some info but enables dim flexibility)\n'
            '            source_latent, _, _, _ = self._encoder_forward_full(source_embed)  # (latent_dim,)\n'
            '            token_embeds_all = self.token_embed.weight.data  # (vocab_size, embed_dim)\n'
            '            token_embeds, _, _, _ = self._encoder_forward_full(token_embeds_all)  # (vocab_size, latent_dim)\n'
            '\n'
            '        # Relation matrix (shared across ALL subjects - key to generalization)\n'
            '        W_rel = self._rp_rel_matrices[rel_type_idx]  # (latent_dim, latent_dim)\n'
            '\n'
            '        # Bilinear scoring: logits = source_latent @ W_rel @ token_embeds.T\n'
            '        # This computes score(source, rel, target_k) = source_latent @ W_rel @ target_embed_k for all k\n'
            '        projected = source_latent @ W_rel  # (latent_dim,)\n'
            '        logits = projected @ token_embeds.T  # (vocab_size,)\n'
            '\n'
            '        # Cache for backprop\n'
            '        self._rp_cache = (\n'
            '            subject_tid, rel_type_idx,\n'
            '            source_embed, source_latent, W_rel, token_embeds, logits\n'
            '        )\n'
            '        return logits'
        )
        content = content.replace(old_fwd, new_fwd, 1)
        changes += 1
        print(f'Change 2: Bilinear _rp_forward ({len(old_fwd)} -> {len(new_fwd)} chars)')
    else:
        print('Change 2: FAILED - no next def found')
else:
    print('Change 2: FAILED - _rp_forward not found')

# ============================================================
# Change 3: Replace _rp_backward with bilinear gradient version
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
        print('Change 3: FAILED - no end found')
        end_idx = -1

    if end_idx >= 0:
        old_bwd = content[idx_bwd:end_idx]
        new_bwd = (
            '    def _rp_backward(self, target_id, lr_scale=1.0):\n'
            '        """Bilinear relation predictor backward pass.\n'
            '\n'
            '        logits = source_latent @ W_rel @ token_embeds.T\n'
            '        dW_rel = outer(source_latent, d_logits @ token_embeds)\n'
            '\n'
            '        Updates the shared relation matrix W_rel (NOT domain-specific heads) -\n'
            '        so held-out subjects benefit from the same learned transformation.\n'
            '        """\n'
            '        if self._rp_cache is None:\n'
            '            return\n'
            '            \n'
            '        (\n'
            '            subject_tid, rel_type_idx,\n'
            '            source_embed, source_latent, W_rel, token_embeds, logits\n'
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
            '        # Bilinear gradient: dW_rel = outer(source_latent, d_logits @ token_embeds)\n'
            '        # logits_k = source_latent_i @ W_ij @ token_embeds_kj\n'
            '        # dL/dW_ij = source_latent_i * sum_k(dL/dlogits_k * token_embeds_kj)\n'
            '        d_logits_proj = d_logits @ token_embeds  # (latent_dim,) gradient projected through all targets\n'
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
            '        # This single update benefits ALL subjects, including held-out ones.\n'
            '        self._rp_mrel_matrices[rel_type_idx] = (\n'
            '            self._rp_momentum * self._rp_mrel_matrices[rel_type_idx] - lr * dW_rel\n'
            '        )\n'
            '        self._rp_rel_matrices[rel_type_idx] += self._rp_mrel_matrices[rel_type_idx]\n'
            '\n'
            '        self._rp_cache = None'
        )
        content = content.replace(old_bwd, new_bwd, 1)
        changes += 1
        print(f'Change 3: Bilinear _rp_backward ({len(old_bwd)} -> {len(new_bwd)} chars)')
    else:
        print('Change 3: FAILED - end_idx not found')
else:
    print('Change 3: FAILED - _rp_backward not found')

# ============================================================
# Change 4: Set rp_scale = 0.1 for bilinear form
# ============================================================
old_rp = (
    '        self.freeze_encoder = False\n'
    '        self.rp_scale = 16.0'
)
new_rp = (
    '        self.freeze_encoder = False\n'
    '        # Bilinear form: logits = source_latent @ W_rel @ token_embeds.T\n'
    '        # Gradient dW_rel = outer(source_latent, d_logits @ token_embeds) is\n'
    '        # naturally O(vocab_size) larger than old linear form. Use rp_scale=0.1\n'
    '        # to compensate, plus gradient clipping (norm=10) in _rp_backward.\n'
    '        self.rp_scale = 0.1'
)
if old_rp in content:
    content = content.replace(old_rp, new_rp, 1)
    changes += 1
    print('Change 4: rp_scale = 0.1 for bilinear form')
else:
    print('Change 4: FAILED - pattern not found')
    idx = content.find('self.rp_scale = 16.0')
    if idx >= 0:
        print(f'  rp_scale found at {idx}')

# ============================================================
# Write back
# ============================================================
if changes > 0:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'\nApplied {changes} changes successfully!')
else:
    print('\nNo changes applied!')
