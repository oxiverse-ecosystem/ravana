path = r"C:\Users\Likhith\Documents\projects\ravana\ravana_ml\nn\rlm_v2.py"
with open(path, 'r') as f:
    content = f.read()

# Add shared relation embeddings option after rp_relation_dim
old_rel = '''        self.rp_relation_dim = 32
        n_rel_types = len(RELATION_TYPES)
        self.rp_relation_embed = np.random.randn(self.num_domains, n_rel_types, self.rp_relation_dim).astype(np.float32) * 0.1
        self.rp_rel_m = np.zeros_like(self.rp_relation_embed)'''

new_rel = '''        self.rp_relation_dim = 32
        n_rel_types = len(RELATION_TYPES)
        # Shared relation embeddings across domains for cross-domain transfer
        self.use_shared_relation_embeds = False  # When True, use single (n_rel_types, rp_relation_dim) shared by all domains
        if getattr(self, 'use_shared_relation_embeds', False):
            self.rp_relation_embed = np.random.randn(n_rel_types, self.rp_relation_dim).astype(np.float32) * 0.1
            self.rp_rel_m = np.zeros_like(self.rp_relation_embed)
        else:
            self.rp_relation_embed = np.random.randn(self.num_domains, n_rel_types, self.rp_relation_dim).astype(np.float32) * 0.1
            self.rp_rel_m = np.zeros_like(self.rp_relation_embed)'''

content = content.replace(old_rel, new_rel)

# Also update _rp_forward to handle shared embeddings
old_forward_rel = '''            if route_softly:
            # Each domain uses its OWN relation embedding for its OWN head
            logits = np.zeros(self.vocab_size, dtype=np.float32)
            active_count = 0
            hidden_cache = None  # cache hidden from first active domain for backward
            rel_embed_cache = None
            combined_cache = None
            for d in range(self.num_domains):
                if d not in self._frozen_domains:
                    d_rel_embed = self.rp_relation_embed[d, rel_type_idx]
                    d_combined = np.concatenate([source_embed, d_rel_embed])'''

new_forward_rel = '''            if route_softly:
            # Each domain uses its OWN relation embedding for its OWN head
            logits = np.zeros(self.vocab_size, dtype=np.float32)
            active_count = 0
            hidden_cache = None  # cache hidden from first active domain for backward
            rel_embed_cache = None
            combined_cache = None
            use_shared_rel = getattr(self, 'use_shared_relation_embeds', False)
            for d in range(self.num_domains):
                if d not in self._frozen_domains:
                    if use_shared_rel:
                        d_rel_embed = self.rp_relation_embed[rel_type_idx]  # shared across domains
                    else:
                        d_rel_embed = self.rp_relation_embed[d, rel_type_idx]
                    d_combined = np.concatenate([source_embed, d_rel_embed])'''

content = content.replace(old_forward_rel, new_forward_rel)

# Also update the else branch (single domain)
old_forward_rel2 = '''        else:
            d = self.current_domain_id if self.current_domain_id is not None else 0
            # Use this domain's specific relation embedding
            rel_embed = self.rp_relation_embed[d, rel_type_idx]
            combined = np.concatenate([source_embed, rel_embed])'''

new_forward_rel2 = '''        else:
            d = self.current_domain_id if self.current_domain_id is not None else 0
            # Use this domain's specific relation embedding (or shared)
            use_shared_rel = getattr(self, 'use_shared_relation_embeds', False)
            if use_shared_rel:
                rel_embed = self.rp_relation_embed[rel_type_idx]
            else:
                rel_embed = self.rp_relation_embed[d, rel_type_idx]
            combined = np.concatenate([source_embed, rel_embed])'''

content = content.replace(old_forward_rel2, new_forward_rel2)

# Also update _rp_backward for shared relation embeddings
old_backward_rel = '''        # Update relation embedding (momentum) - FIX: use 'd' instead of undefined 'active_domain'
        active_d = self.current_domain_id if self.current_domain_id is not None else 0
        if route_softly and active_domains:
            active_d = active_domains[0]  # first active domain
        self.rp_rel_m[active_d, rel_type_idx] = self._rp_momentum * self.rp_rel_m[active_d, rel_type_idx] - lr * d_rel_embed
        self.rp_relation_embed[active_d, rel_type_idx] += self.rp_rel_m[active_d, rel_type_idx]'''

new_backward_rel = '''        # Update relation embedding (momentum) - FIX: use 'd' instead of undefined 'active_domain'
        use_shared_rel = getattr(self, 'use_shared_relation_embeds', False)
        if use_shared_rel:
            # Shared relation embeddings: update single embedding
            self.rp_rel_m[rel_type_idx] = self._rp_momentum * self.rp_rel_m[rel_type_idx] - lr * d_rel_embed
            self.rp_relation_embed[rel_type_idx] += self.rp_rel_m[rel_type_idx]
        else:
            active_d = self.current_domain_id if self.current_domain_id is not None else 0
            if route_softly and active_domains:
                active_d = active_domains[0]  # first active domain
            self.rp_rel_m[active_d, rel_type_idx] = self._rp_momentum * self.rp_rel_m[active_d, rel_type_idx] - lr * d_rel_embed
            self.rp_relation_embed[active_d, rel_type_idx] += self.rp_rel_m[active_d, rel_type_idx]'''

content = content.replace(old_backward_rel, new_backward_rel)

with open(path, 'w') as f:
    f.write(content)

print("Added shared relation embeddings option")