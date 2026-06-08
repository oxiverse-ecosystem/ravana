path = r"C:\Users\Likhith\Documents\projects\ravana\ravana_ml\nn\rlm_v2.py"
with open(path, 'r') as f:
    lines = f.readlines()

# New function to replace lines 1373-1434 (0-indexed)
new_function = '''    def _rp_backward(self, target_id, lr_scale=1.0, loss_type="cross_entropy"):
        """Relation predictor backward pass with optional nonlinear hidden layer and contrastive loss.

        Gradient flows back through the concatenated vector to update both
        domain_W_logits and rp_relation_embed (but NOT the encoder -- bypassed).
        Supports:
        - 'cross_entropy': standard softmax CE loss (default)
        - 'contrastive': margin-based contrastive loss on RP logits
        """
        if self._rp_cache is None:
            return
           
        # Unpack cache (now includes hidden activation)
        cache = self._rp_cache
        if len(cache) == 8:
            subject_tid, rel_type_idx, source_embed, rel_embed, combined, route_softly, logits, hidden = cache
        else:
            # Backward compat: old cache format without hidden
            subject_tid, rel_type_idx, source_embed, rel_embed, combined, route_softly, logits = cache
            hidden = None
        
        use_hidden = getattr(self, 'use_rp_hidden', True) and hidden is not None
        rp_hidden_dim = getattr(self, 'rp_hidden_dim', 256)
        
        # Compute loss gradient w.r.t logits
        if loss_type == "contrastive" and getattr(self, 'use_rp_contrastive', False):
            d_logits, contrastive_loss = self._rp_contrastive_loss_grad(logits, target_id)
            # Scale by lambda
            d_logits *= getattr(self, 'rp_contrastive_lambda', 1.0)
        else:
            # Standard cross-entropy gradient
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / (np.sum(exp_logits) + 1e-10)
            d_logits = probs.copy()
            if 0 <= target_id < len(d_logits):
                d_logits[target_id] -= 1.0
            d_logits *= getattr(self, "rp_scale", 16.0)

        # Gradients for domain heads
        d_domain_W_logits = [np.zeros_like(w) for w in self.domain_W_logits]
        d_domain_b_logits = [np.zeros_like(b) for b in self.domain_b_logits]
        
        # Gradients for hidden layer (if used)
        if use_hidden:
            d_domain_W_hidden = [np.zeros_like(w) for w in self.domain_rp_W_hidden]
            d_domain_b_hidden = [np.zeros_like(b) for b in self.domain_rp_b_hidden]
        else:
            d_combined = np.zeros(self.rp_input_dim, dtype=np.float32)

        if route_softly:
            active_domains = [d for d in range(self.num_domains) if d not in self._frozen_domains]
            w_d = 1.0 / max(1, len(active_domains))
            for d in active_domains:
                if use_hidden:
                    # Output layer gradient: d_logits @ hidden.T
                    d_domain_W_logits[d] = np.outer(d_logits * w_d, hidden)
                    d_domain_b_logits[d] = (d_logits * w_d).copy()
                    
                    # Hidden layer gradient: backprop through tanh
                    d_hidden = self.domain_W_logits[d].T @ (d_logits * w_d)
                    # tanh derivative: 1 - hidden^2
                    d_hidden_pre = d_hidden * (1.0 - hidden * hidden)
                    d_domain_W_hidden[d] = np.outer(d_hidden_pre, combined)
                    d_domain_b_hidden[d] = d_hidden_pre.copy()
                    
                    # Gradient w.r.t combined input
                    d_combined_d = self.domain_rp_W_hidden[d].T @ d_hidden_pre
                else:
                    d_domain_W_logits[d] = np.outer(d_logits * w_d, combined)
                    d_domain_b_logits[d] = (d_logits * w_d).copy()
                    d_combined_d = self.domain_W_logits[d].T @ (d_logits * w_d)
                
                if 'd_combined' not in dir():
                    d_combined = np.zeros(self.rp_input_dim, dtype=np.float32)
                if use_hidden:
                    d_combined += d_combined_d
                else:
                    d_combined += d_combined_d
        else:
            d = self.current_domain_id if self.current_domain_id is not None else 0
            if use_hidden:
                d_domain_W_logits[d] = np.outer(d_logits, hidden)
                d_domain_b_logits[d] = d_logits.copy()
                
                d_hidden = self.domain_W_logits[d].T @ d_logits
                d_hidden_pre = d_hidden * (1.0 - hidden * hidden)
                d_domain_W_hidden[d] = np.outer(d_hidden_pre, combined)
                d_domain_b_hidden[d] = d_hidden_pre.copy()
                d_combined = self.domain_rp_W_hidden[d].T @ d_hidden_pre
            else:
                d_domain_W_logits[d] = np.outer(d_logits, combined)
                d_domain_b_logits[d] = d_logits.copy()
                d_combined = self.domain_W_logits[d].T @ d_logits

        # Split gradient: first embed_dim goes to source (not updated -- encoder bypassed),
        # last rp_relation_dim goes to relation embedding
        d_rel_embed = d_combined[self.embed_dim:]  # (rp_relation_dim,)

        lr = self._rp_lr * lr_scale

        # Update domain heads
        for d in range(self.num_domains):
            if d in self._frozen_domains:
                continue
            self.domain_W_logits_m[d] = self._rp_momentum * self.domain_W_logits_m[d] - lr * d_domain_W_logits[d]
            self.domain_b_logits_m[d] = self._rp_momentum * self.domain_b_logits_m[d] - lr * d_domain_b_logits[d]
            
            self.domain_W_logits[d] += self.domain_W_logits_m[d]
            self.domain_b_logits[d] += self.domain_b_logits_m[d]
            
            # Update hidden layer weights if used
            if use_hidden:
                self.domain_rp_W_hidden_m[d] = self._rp_momentum * self.domain_rp_W_hidden_m[d] - lr * d_domain_W_hidden[d]
                self.domain_rp_b_hidden_m[d] = self._rp_momentum * self.domain_rp_b_hidden_m[d] - lr * d_domain_b_hidden[d]
                
                self.domain_rp_W_hidden[d] += self.domain_rp_W_hidden_m[d]
                self.domain_rp_b_hidden[d] += self.domain_rp_b_hidden_m[d]

        # Update relation embedding (momentum) - FIX: use 'd' instead of undefined 'active_domain'
        active_d = self.current_domain_id if self.current_domain_id is not None else 0
        if route_softly and active_domains:
            active_d = active_domains[0]  # first active domain
        self.rp_rel_m[active_d, rel_type_idx] = self._rp_momentum * self.rp_rel_m[active_d, rel_type_idx] - lr * d_rel_embed
        self.rp_relation_embed[active_d, rel_type_idx] += self.rp_rel_m[active_d, rel_type_idx]

        self._rp_cache = None

    def _rp_contrastive_loss_grad(self, logits, target_id):
        """Compute contrastive margin loss gradient on RP logits.
        
        Loss = max(0, margin + max_neg_score - pos_score)
        Gradient: push target up, push top-k negatives down
        """
        margin = getattr(self, 'rp_contrastive_margin', 0.5)
        neg_k = getattr(self, 'rp_contrastive_neg_samples', 10)
        
        pos_score = logits[target_id]
        
        # Find top-k negative scores (excluding target)
        topk_indices = np.argsort(logits)[::-1]
        neg_indices = [i for i in topk_indices if i != target_id][:neg_k]
        
        if not neg_indices:
            return np.zeros_like(logits), 0.0
        
        neg_scores = logits[neg_indices]
        max_neg_score = np.max(neg_scores)
        
        loss = max(0.0, margin + max_neg_score - pos_score)
        
        d_logits = np.zeros_like(logits)
        if loss > 0:
            # Positive gradient: increase target score
            d_logits[target_id] = -1.0
            # Negative gradient: decrease hardest negative score
            hardest_neg_idx = neg_indices[np.argmax(neg_scores)]
            d_logits[hardest_neg_idx] = 1.0
        
        return d_logits, loss

'''

# Build new file content
# lines[1373:1435] is the old function (1373 to 1434 inclusive)
new_lines = lines[:1373] + [new_function] + lines[1435:]

with open(path, 'w') as f:
    f.writelines(new_lines)

print(f"Replaced lines 1374-1435 with new function ({len(new_lines)} total lines)")