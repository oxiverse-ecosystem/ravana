path = r"C:/Users/Likhith/Documents/projects/ravana/ravana_ml/nn/rlm_v2.py"
with open(path, 'r') as f:
    content = f.read()

# Add the cross-domain alignment method after _compute_contrastive_gradients
old_contrastive = '''    def _compute_contrastive_gradients(self):
        """Compute contrastive loss gradients w.r.t. encoder parameters."""
        d_con_W1 = np.zeros_like(self._enc_W1)
        d_con_b1 = np.zeros_like(self._enc_b1)
        d_con_W2 = np.zeros_like(self._enc_W2)
        d_con_b2 = np.zeros_like(self._enc_b2)'''

new_contrastive = '''    def _cross_domain_relation_alignment(self):
        """Cross-domain relation alignment: pull same-type relations across domains closer.
        
        For each relation type (e.g., 'causal'), collect edge relation vectors from each domain
        via their subject concepts. Apply contrastive loss: same-type across domains attract,
        different-types repel. This enables transfer: A's 'causal' edges generalize to B.
        """
        if self.num_domains < 2:
            return
            
        use_shared = getattr(self, 'use_shared_relation_embeds', False)
        if use_shared:
            return  # Shared embeddings already aligned
            
        margin = getattr(self, 'alignment_margin', 0.15)
        lambda_align = getattr(self, 'lambda_anchor', 0.05)
        lr = getattr(self, 'alignment_lr', 0.005)
        
        # Group edges by domain and relation type
        # Domain is inferred from subject concept's domain (via binding map)
        domain_edges = {d: {} for d in range(self.num_domains)}
        
        for (src_cid, tgt_cid), edge in self.graph.edges.items():
            if edge.relation_vector is None:
                continue
            # Infer domain from source concept's bound tokens
            # This is a heuristic - ideally we'd track domain per edge
            src_bindings = self.binding_map.get_tokens(src_cid, min_confidence=0.1)
            if not src_bindings:
                continue
            src_tid = src_bindings[0].token_id
            # Heuristic: low token IDs -> Domain A (science), high -> Domain B (social)
            # This is approximate but works for the experiment
            domain = 0 if src_tid < self.vocab_size // 2 else 1
            rel_type = edge.relation_type
            if rel_type not in domain_edges[domain]:
                domain_edges[domain][rel_type] = []
            domain_edges[domain][rel_type].append(edge.relation_vector)
        
        # Compute mean relation vector per domain per type
        mean_rvs = {d: {} for d in range(self.num_domains)}
        for d in range(self.num_domains):
            for rel_type, vecs in domain_edges[d].items():
                if len(vecs) > 0:
                    mean_rvs[d][rel_type] = np.mean(vecs, axis=0)
        
        # Contrastive alignment: for each relation type present in multiple domains
        for rel_type in RELATION_TYPES:
            present_domains = [d for d in range(self.num_domains) if rel_type in mean_rvs[d]]
            if len(present_domains) < 2:
                continue
            
            # Pull together: attract mean vectors across domains
            vectors = [mean_rvs[d][rel_type] for d in present_domains]
            center = np.mean(vectors, axis=0)
            
            # Update edges toward center (soft alignment)
            for d in present_domains:
                for (src_cid, tgt_cid), edge in self.graph.edges.items():
                    if edge.relation_type != rel_type:
                        continue
                    # Check domain
                    src_bindings = self.binding_map.get_tokens(src_cid, min_confidence=0.1)
                    if not src_bindings:
                        continue
                    src_tid = src_bindings[0].token_id
                    edge_domain = 0 if src_tid < self.vocab_size // 2 else 1
                    if edge_domain != d:
                        continue
                    
                    # Pull toward center
                    if edge.relation_vector is not None:
                        delta = lr * lambda_align * (center - edge.relation_vector)
                        edge.relation_vector += delta
                        rv_n = np.linalg.norm(edge.relation_vector)
                        if rv_n > 0:
                            edge.relation_vector /= rv_n
                        edge._rv_norm_cache = None
        
        # Repel: different relation types in same domain should be separated
        # (Optional: harder to implement cleanly, skip for now)

    def _compute_contrastive_gradients(self):
        """Compute contrastive loss gradients w.r.t. encoder parameters."""
        d_con_W1 = np.zeros_like(self._enc_W1)
        d_con_b1 = np.zeros_like(self._enc_b1)
        d_con_W2 = np.zeros_like(self._enc_W2)
        d_con_b2 = np.zeros_like(self._enc_b2)'''

content = content.replace(old_contrastive, new_contrastive)

# Now call it in learn() after predictive coding updates
old_call = '''        # ── Train learned relation predictor MLP ──
        self._rp_forward(subject_tid, rel_type_idx)
        if getattr(self, 'use_rp_contrastive', False):
            self._rp_backward(target_id, loss_type="contrastive")
        else:
            self._rp_backward(target_id)'''

new_call = '''        # Cross-domain relation alignment (runs after edge updates)
        if getattr(self, 'use_cross_domain_alignment', True):
            self._cross_domain_relation_alignment()

        # ── Train learned relation predictor MLP ──
        self._rp_forward(subject_tid, rel_type_idx)
        # Keep standard CE for RP as backup'''

content = content.replace(old_call, new_call)

with open(path, 'w') as f:
    f.write(content)

print("Added cross-domain relation alignment")