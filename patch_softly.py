path = r"C:\Users\Likhith\Documents\projects\ravana\ravana_ml\nn\rlm_v2.py"
with open(path, 'r') as f:
    content = f.read()

# Fix the route_softly branch to handle shared embeddings
old_softly = '''            if route_softly:
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

new_softly = '''            if route_softly:
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
                        d_rel_embed = self.rp_relation_embed[rel_type_idx]
                    else:
                        d_rel_embed = self.rp_relation_embed[d, rel_type_idx]
                    d_combined = np.concatenate([source_embed, d_rel_embed])'''

content = content.replace(old_softly, new_softly)

with open(path, 'w') as f:
    f.write(content)

print("Fixed route_softly branch for shared embeddings")