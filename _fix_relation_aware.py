"""Fix 1: Relation Embeddings as Features

Instead of using source_embed alone → domain_W_logits (subject-specific memorization),
concatenate source_embed + relation_embed so the RP learns (subject + relation) → target.

This forces generalization: a held-out subject with the same relation should produce
similar combined features as trained subjects with that relation.

Changes:
1. __init__: Add rp_relation_dim=32, rp_relation_embed, rp_rel_m
2. __init__: domain_W_logits shape = (vocab, embed_dim + rp_relation_dim) = (vocab, 96)
3. _rp_forward: combined = concat(source_embed, rel_embed)
4. _rp_backward: split d_combined, update rel_embed
5. state_dict: add rp_relation_embed, rp_rel_m, relation_dim meta
6. _load_state: load new params, update expected shape
"""
import sys

path = 'ravana_ml/nn/rlm_v2.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# ============================================================
# Change 1: __init__ - Add relation embeddings, update domain_W_logits shape
# ============================================================
# Find the section right before "Domain-Specific Heads"
old_init = """        self.spreading_confidence_threshold = 0.35

        # Domain-Specific Heads (direct from source_latent, no bottleneck)
        self.domain_W_logits = []
        self.domain_b_logits = []

        for d in range(self.num_domains):
            W_logits = np.random.normal(0, 0.1, (self.vocab_size, self.embed_dim)).astype(np.float32)
            self.domain_W_logits.append(W_logits)
            self.domain_b_logits.append(np.zeros(self.vocab_size, dtype=np.float32))

        # Domain isolation: track which domains are frozen
        self._frozen_domains: Set[int] = set()

        # Momentum buffers
        self.domain_W_logits_m = [np.zeros_like(w) for w in self.domain_W_logits]
        self.domain_b_logits_m = [np.zeros_like(b) for b in self.domain_b_logits]"""

new_init = """        self.spreading_confidence_threshold = 0.35

        # ── Relation-Aware RP Features ──
        # Concatenate source_embed + relation_embed so the RP learns
        # (subject + relation) → target, not just subject → target.
        # This is critical for generalization to held-out subjects:
        # a new subject with the same relation will have a similar combined vector.
        self.rp_relation_dim = 32
        n_rel_types = len(RELATION_TYPES)
        self.rp_relation_embed = np.random.randn(n_rel_types, self.rp_relation_dim).astype(np.float32) * 0.1
        self.rp_rel_m = np.zeros_like(self.rp_relation_embed)

        # RP input dimension = embed_dim + relation_dim
        self.rp_input_dim = self.embed_dim + self.rp_relation_dim

        # Domain-Specific Heads (relation-aware features)
        self.domain_W_logits = []
        self.domain_b_logits = []

        for d in range(self.num_domains):
            W_logits = np.random.normal(0, 0.1, (self.vocab_size, self.rp_input_dim)).astype(np.float32)
            self.domain_W_logits.append(W_logits)
            self.domain_b_logits.append(np.zeros(self.vocab_size, dtype=np.float32))

        # Domain isolation: track which domains are frozen
        self._frozen_domains: Set[int] = set()

        # Momentum buffers
        self.domain_W_logits_m = [np.zeros_like(w) for w in self.domain_W_logits]
        self.domain_b_logits_m = [np.zeros_like(b) for b in self.domain_b_logits]"""

if old_init in content:
    content = content.replace(old_init, new_init, 1)
    print("1. __init__: Added rp_relation_embed, rp_input_dim, updated domain_W_logits shape")
else:
    print("1. __init__: FAILED")
    idx = content.find("spreading_confidence_threshold = 0.35")
    if idx >= 0:
        print(f"   Found at {idx}")
        print(f"   Next 300 chars: {repr(content[idx:idx+300])}")

# ============================================================
# Change 2: _rp_forward - concatenate source_embed + relation_embed
# ============================================================
old_rp_forward = """        # Get robust source embedding (BY-PASS ENCODER: use embed directly)
        # The encoder collapses all inputs to identical latents (autoencoder effect).
        # Using source_embed (raw embedding) preserves discriminative information.
        source_embed = self.get_robust_embedding(subject_tid)  # (embed_dim,)
        source_latent = source_embed  # use raw embedding as the \"latent\"
        if route_softly is None:
            route_softly = (self.current_domain_id is None)
        
        if route_softly:
            logits = np.zeros(self.vocab_size, dtype=np.float32)
            for d in range(self.num_domains):
                if d not in self._frozen_domains:
                    logits += self.domain_W_logits[d] @ source_latent + self.domain_b_logits[d]
            # Div by active domain count (not fixed num_domains)
            logits /= max(1, sum(1 for _d_ in range(self.num_domains) if _d_ not in self._frozen_domains))
        else:
            d = self.current_domain_id if self.current_domain_id is not None else 0
            logits = self.domain_W_logits[d] @ source_latent + self.domain_b_logits[d]

        # Cache for backprop (no encoder activations - bypassing collapsed encoder)
        self._rp_cache = (
            subject_tid, rel_type_idx,
            source_embed, source_latent, route_softly, logits
        )
        return logits"""

new_rp_forward = """        # Get robust source embedding
        source_embed = self.get_robust_embedding(subject_tid)  # (embed_dim,)
        
        # Get relation embedding — CRITICAL for held-out generalization
        # This makes the RP learn (subject + relation) → target instead of just subject → target.
        rel_embed = self.rp_relation_embed[rel_type_idx]  # (rp_relation_dim,)
        
        # Concatenate: combined = [source_embed | rel_embed]  (embed_dim + relation_dim)
        # A held-out subject with the same relation produces a similar combined vector
        # because the relation part is identical.
        combined = np.concatenate([source_embed, rel_embed])  # (rp_input_dim,)
        
        if route_softly is None:
            route_softly = (self.current_domain_id is None)
        
        if route_softly:
            logits = np.zeros(self.vocab_size, dtype=np.float32)
            for d in range(self.num_domains):
                if d not in self._frozen_domains:
                    logits += self.domain_W_logits[d] @ combined + self.domain_b_logits[d]
            logits /= max(1, sum(1 for _d_ in range(self.num_domains) if _d_ not in self._frozen_domains))
        else:
            d = self.current_domain_id if self.current_domain_id is not None else 0
            logits = self.domain_W_logits[d] @ combined + self.domain_b_logits[d]

        # Cache for backprop
        self._rp_cache = (
            subject_tid, rel_type_idx,
            source_embed, rel_embed, combined, route_softly, logits
        )
        return logits"""

if old_rp_forward in content:
    content = content.replace(old_rp_forward, new_rp_forward, 1)
    print("2. _rp_forward: Combined source_embed + rel_embed into relation-aware features")
else:
    print("2. _rp_forward: FAILED")
    idx = content.find("Get robust source embedding (BY-PASS ENCODER")
    if idx >= 0:
        print(f"   Found at {idx}")
    else:
        idx = content.find("Get robust source embedding")
        if idx >= 0:
            print(f"   Found alternate at {idx}")

# ============================================================
# Change 3: _rp_backward - split gradient, update relation embeddings
# ============================================================
old_backward = """    def _rp_backward(self, target_id, lr_scale=1.0):
        \"\"\"Relation predictor backward pass. Uses source_embed directly (bypasses encoder).

        No encoder backprop (encoder causes latent collapse during RP training).
        \"\"\"
        if self._rp_cache is None:
            return
            
        (
            subject_tid, rel_type_idx,
            source_embed, source_latent, route_softly, logits
        ) = self._rp_cache
        
        # Softmax loss gradient
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)
        d_logits = probs.copy()
        if 0 <= target_id < len(d_logits):
            d_logits[target_id] -= 1.0
        d_logits *= getattr(self, \"rp_scale\", 16.0)

        # Gradients for domain heads (no gates, no router, no encoder backprop)
        d_domain_W_logits = [np.zeros_like(w) for w in self.domain_W_logits]
        d_domain_b_logits = [np.zeros_like(b) for b in self.domain_b_logits]

        if route_softly:
            w_d = 1.0 / self.num_domains
            for d in range(self.num_domains):
                if d in self._frozen_domains:
                    continue
                d_domain_W_logits[d] = np.outer(d_logits * w_d, source_latent)
                d_domain_b_logits[d] = (d_logits * w_d).copy()
        else:
            d = self.current_domain_id if self.current_domain_id is not None else 0
            d_domain_W_logits[d] = np.outer(d_logits, source_latent)
            d_domain_b_logits[d] = d_logits.copy()

        lr = self._rp_lr * lr_scale

        # Update Domain Heads - no encoder backprop (avoids latent collapse)
        for d in range(self.num_domains):
            if d in self._frozen_domains:
                continue
            self.domain_W_logits_m[d] = self._rp_momentum * self.domain_W_logits_m[d] - lr * d_domain_W_logits[d]
            self.domain_b_logits_m[d] = self._rp_momentum * self.domain_b_logits_m[d] - lr * d_domain_b_logits[d]
            
            self.domain_W_logits[d] += self.domain_W_logits_m[d]
            self.domain_b_logits[d] += self.domain_b_logits_m[d]

        self._rp_cache = None"""

new_backward = """    def _rp_backward(self, target_id, lr_scale=1.0):
        \"\"\"Relation predictor backward pass. Uses combined source_embed + rel_embed.

        Gradient flows back through the concatenated vector to update both
        domain_W_logits and rp_relation_embed (but NOT the encoder — bypassed).
        \"\"\"
        if self._rp_cache is None:
            return
            
        (
            subject_tid, rel_type_idx,
            source_embed, rel_embed, combined, route_softly, logits
        ) = self._rp_cache
        
        # Softmax loss gradient
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)
        d_logits = probs.copy()
        if 0 <= target_id < len(d_logits):
            d_logits[target_id] -= 1.0
        d_logits *= getattr(self, \"rp_scale\", 16.0)

        # Gradients for domain heads
        d_domain_W_logits = [np.zeros_like(w) for w in self.domain_W_logits]
        d_domain_b_logits = [np.zeros_like(b) for b in self.domain_b_logits]
        d_combined = np.zeros(self.rp_input_dim, dtype=np.float32)

        if route_softly:
            w_d = 1.0 / max(1, sum(1 for _d_ in range(self.num_domains) if _d_ not in self._frozen_domains))
            for d in range(self.num_domains):
                if d in self._frozen_domains:
                    continue
                d_domain_W_logits[d] = np.outer(d_logits * w_d, combined)
                d_domain_b_logits[d] = (d_logits * w_d).copy()
                d_combined += self.domain_W_logits[d].T @ (d_logits * w_d)
        else:
            d = self.current_domain_id if self.current_domain_id is not None else 0
            d_domain_W_logits[d] = np.outer(d_logits, combined)
            d_domain_b_logits[d] = d_logits.copy()
            d_combined = self.domain_W_logits[d].T @ d_logits

        # Split gradient: first embed_dim goes to source (not updated — encoder bypassed),
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

        # Update relation embedding (momentum)
        self.rp_rel_m[rel_type_idx] = self._rp_momentum * self.rp_rel_m[rel_type_idx] - lr * d_rel_embed
        self.rp_relation_embed[rel_type_idx] += self.rp_rel_m[rel_type_idx]

        self._rp_cache = None"""

if old_backward in content:
    content = content.replace(old_backward, new_backward, 1)
    print("3. _rp_backward: Added gradient split + relation embedding update")
else:
    print("3. _rp_backward: FAILED")
    idx = content.find("def _rp_backward(self, target_id, lr_scale=1.0):")
    if idx >= 0:
        print(f"   Found at {idx}")

# ============================================================
# Change 4: state_dict - add rp_relation_embed and rp_rel_m
# ============================================================
old_state_section = "            \"router_b\": self.router_b,"

# Add before the closing brace section or after router_b
# Let me find the last bottleneck param and add after it
old_state = """            \"router_b\": self.router_b,

            \"binding_map\": {"""

new_state = """            \"router_b\": self.router_b,
            \"rp_relation_dim\": self.rp_relation_dim,
            \"rp_relation_embed\": self.rp_relation_embed.copy(),
            \"rp_rel_m\": self.rp_rel_m.copy(),
            \"rp_input_dim\": self.rp_input_dim,

            \"binding_map\": {"""

if old_state in content:
    content = content.replace(old_state, new_state, 1)
    print("4. state_dict: Added rp_relation_embed, rp_rel_m, rp_input_dim")
else:
    print("4. state_dict: FAILED")
    idx = content.find('\"router_b\": self.router_b,')
    if idx >= 0:
        print(f"   Found 'router_b' at {idx}")
        print(f"   Context: {repr(content[idx:idx+60])}")

# ============================================================
# Change 5: _load_state - load new params, update expected shape
# ============================================================
# Update expected_shape 
old_load_shape = "            expected_shape = (self.vocab_size, self.embed_dim)"
new_load_shape = "            expected_shape = (self.vocab_size, self.embed_dim + self.rp_relation_dim)"
if old_load_shape in content:
    content = content.replace(old_load_shape, new_load_shape, 1)
    print("5a. _load_state: Updated expected_shape for domain_W_logits")
else:
    print("5a. _load_state: FAILED")
    idx = content.find("expected_shape = (self.vocab_size, self.embed_dim)")
    if idx >= 0:
        print(f"   Found at {idx}")

# Add rp_relation_embed loading after the router/hidden section loading
# Find the section where we load cognitive states
old_load_extra = "        # Rebuild raw numpy caches in submodules and clear cached norms"
new_load_extra = """        # Restore relation-aware RP features
        if "rp_relation_embed" in state:
            self.rp_relation_embed = state["rp_relation_embed"].copy()
            self.rp_rel_m = state.get("rp_rel_m", np.zeros_like(self.rp_relation_embed)).copy()
        if "rp_relation_dim" in state:
            self.rp_relation_dim = state["rp_relation_dim"]
        if "rp_input_dim" in state:
            self.rp_input_dim = state["rp_input_dim"]

        # Rebuild raw numpy caches in submodules and clear cached norms"""

if old_load_extra in content:
    content = content.replace(old_load_extra, new_load_extra, 1)
    print("5b. _load_state: Added rp_relation_embed restoration")
else:
    print("5b. _load_state: FAILED")
    idx = content.find("Rebuild raw numpy caches")
    if idx >= 0:
        print(f"   Found at {idx}")

# ============================================================
# Write back
# ============================================================
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("\nDone! Fix 1 applied.")
