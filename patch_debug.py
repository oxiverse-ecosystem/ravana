path = r"C:\Users\Likhith\Documents\projects\ravana\ravana_ml\nn\rlm_v2.py"
with open(path, 'r') as f:
    content = f.read()

# Add debug print in _rp_forward
old_debug = '''            use_shared_rel = getattr(self, 'use_shared_relation_embeds', False)
            for d in range(self.num_domains):
                if d not in self._frozen_domains:
                    if use_shared_rel:
                        d_rel_embed = self.rp_relation_embed[rel_type_idx]
                    else:
                        d_rel_embed = self.rp_relation_embed[d, rel_type_idx]'''

new_debug = '''            use_shared_rel = getattr(self, 'use_shared_relation_embeds', False)
            for d in range(self.num_domains):
                if d not in self._frozen_domains:
                    if use_shared_rel:
                        d_rel_embed = self.rp_relation_embed[rel_type_idx]
                        # DEBUG
                        print(f"DEBUG shared: rel_type_idx={rel_type_idx}, embed shape={d_rel_embed.shape}, embed ndim={d_rel_embed.ndim}")
                    else:
                        d_rel_embed = self.rp_relation_embed[d, rel_type_idx]'''

content = content.replace(old_debug, new_debug)

with open(path, 'w') as f:
    f.write(content)

print("Added debug")