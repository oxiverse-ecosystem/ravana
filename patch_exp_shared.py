path = r"C:\Users\Likhith\Documents\projects\ravana\experiments\experiment_cross_domain.py"
with open(path, 'r') as f:
    content = f.read()

# Enable shared relation embeddings
old_shared = '''    # Enable new RP features
    model.use_rp_hidden = True
    model.use_rp_contrastive = True
    model.rp_contrastive_lambda = 1.0
    model.rp_contrastive_margin = 0.5
    model.rp_contrastive_neg_samples = 10'''

new_shared = '''    # Enable new RP features
    model.use_rp_hidden = True
    model.use_rp_contrastive = True
    model.use_shared_relation_embeds = True  # Share relation embeddings across domains
    model.rp_contrastive_lambda = 1.0
    model.rp_contrastive_margin = 0.5
    model.rp_contrastive_neg_samples = 10'''

content = content.replace(old_shared, new_shared)

# Also change: during Phase 2, the model trains Domain B with shared relation embeddings
# No need to change the training loop since shared relation embeddings are always shared

with open(path, 'w') as f:
    f.write(content)

print("Enabled shared relation embeddings in experiment")