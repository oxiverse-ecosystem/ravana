path = r"C:\Users\Likhith\Documents\projects\ravana\ravana_ml\nn\rlm_v2.py"
with open(path, 'r') as f:
    content = f.read()

# Add parameter to __init__ signature
old_sig = '''    def __init__(self, vocab_size: int, embed_dim: int, concept_dim: int,
                 n_concepts: int, max_seq_len: int = 128,
                 sleep_interval: int = 100,
                 gate_concept_creation: bool = True,
                 anchor_relation_vectors: bool = True,
                 latent_dim: int = 96,
                 hidden_dim: int = 128,
                 **kwargs):'''

new_sig = '''    def __init__(self, vocab_size: int, embed_dim: int, concept_dim: int,
                 n_concepts: int, max_seq_len: int = 128,
                 sleep_interval: int = 100,
                 gate_concept_creation: bool = True,
                 anchor_relation_vectors: bool = True,
                 latent_dim: int = 96,
                 hidden_dim: int = 128,
                 use_shared_relation_embeds: bool = False,
                 **kwargs):'''

content = content.replace(old_sig, new_sig)

# Fix the initialization logic
old_init = '''        # Shared relation embeddings across domains for cross-domain transfer
        self.use_shared_relation_embeds = False  # When True, use single (n_rel_types, rp_relation_dim) shared by all domains
        if getattr(self, 'use_shared_relation_embeds', False):
            self.rp_relation_embed = np.random.randn(n_rel_types, self.rp_relation_dim).astype(np.float32) * 0.1
            self.rp_rel_m = np.zeros_like(self.rp_relation_embed)
        else:
            self.rp_relation_embed = np.random.randn(self.num_domains, n_rel_types, self.rp_relation_dim).astype(np.float32) * 0.1
            self.rp_rel_m = np.zeros_like(self.rp_relation_embed)'''

new_init = '''        # Shared relation embeddings across domains for cross-domain transfer
        self.use_shared_relation_embeds = use_shared_relation_embeds
        if self.use_shared_relation_embeds:
            self.rp_relation_embed = np.random.randn(n_rel_types, self.rp_relation_dim).astype(np.float32) * 0.1
            self.rp_rel_m = np.zeros_like(self.rp_relation_embed)
        else:
            self.rp_relation_embed = np.random.randn(self.num_domains, n_rel_types, self.rp_relation_dim).astype(np.float32) * 0.1
            self.rp_rel_m = np.zeros_like(self.rp_relation_embed)'''

content = content.replace(old_init, new_init)

with open(path, 'w') as f:
    f.write(content)

print("Fixed init to use constructor parameter")