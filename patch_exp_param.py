path = r"C:\Users\Likhith\Documents\projects\ravana\experiments\experiment_cross_domain.py"
with open(path, 'r') as f:
    content = f.read()

# Update model creation to pass the parameter
old_model = '''    model = RLMv2(
        vocab_size=vocab_size + 5,
        embed_dim=config.embed_dim,
        concept_dim=config.concept_dim,
        n_concepts=vocab_size,
        sleep_interval=config.sleep_interval,
        gate_concept_creation=False,
        latent_dim=96,
        hidden_dim=128,
    )'''

new_model = '''    model = RLMv2(
        vocab_size=vocab_size + 5,
        embed_dim=config.embed_dim,
        concept_dim=config.concept_dim,
        n_concepts=vocab_size,
        sleep_interval=config.sleep_interval,
        gate_concept_creation=False,
        latent_dim=96,
        hidden_dim=128,
        use_shared_relation_embeds=True,  # Share relation embeddings across domains
    )'''

content = content.replace(old_model, new_model)

# Remove the line that sets it after creation
old_lines = '''    # Enable new RP features
    model.use_rp_hidden = True
    model.use_rp_contrastive = True
    model.use_shared_relation_embeds = True  # Share relation embeddings across domains
    model.rp_contrastive_lambda = 1.0
    model.rp_contrastive_margin = 0.5
    model.rp_contrastive_neg_samples = 10'''

new_lines = '''    # Enable new RP features
    model.use_rp_hidden = True
    model.use_rp_contrastive = True
    model.rp_contrastive_lambda = 1.0
    model.rp_contrastive_margin = 0.5
    model.rp_contrastive_neg_samples = 10'''

content = content.replace(old_lines, new_lines)

with open(path, 'w') as f:
    f.write(content)

print("Updated experiment to pass parameter")