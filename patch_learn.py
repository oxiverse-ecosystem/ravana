path = r"C:\Users\Likhith\Documents\projects\ravana\ravana_ml\nn\rlm_v2.py"
with open(path, 'r') as f:
    content = f.read()

old = '''        # ── Train learned relation predictor MLP ──
        self._rp_forward(subject_tid, rel_type_idx)
        self._rp_backward(target_id)'''

new = '''        # ── Train learned relation predictor MLP ──
        self._rp_forward(subject_tid, rel_type_idx)
        if getattr(self, 'use_rp_contrastive', False):
            self._rp_backward(target_id, loss_type="contrastive")
        else:
            self._rp_backward(target_id)'''

new_content = content.replace(old, new)

with open(path, 'w') as f:
    f.write(new_content)

print("Updated learn() to use contrastive loss")