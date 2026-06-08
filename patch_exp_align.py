path = r"C:\Users\Likhith\Documents\projects\ravana\experiments\experiment_cross_domain.py"
with open(path, 'r') as f:
    content = f.read()

# Update model config - enable cross-domain alignment, disable RP contrastive
old_model = '''    model = RLMv2(
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

new_model = '''    model = RLMv2(
        vocab_size=vocab_size + 5,
        embed_dim=config.embed_dim,
        concept_dim=config.concept_dim,
        n_concepts=vocab_size,
        sleep_interval=config.sleep_interval,
        gate_concept_creation=False,
        latent_dim=96,
        hidden_dim=128,
        use_shared_relation_embeds=False,  # Domain-specific, aligned via cross-domain loss
    )
    model.use_cross_domain_alignment = True  # Enable explicit relation alignment
    model.use_rp_contrastive = False  # Disable RP contrastive (not the primary path)'''

content = content.replace(old_model, new_model)

# Remove the lines after model creation that set these features
old_lines = '''    # Enable new RP features
    model.use_rp_hidden = True
    model.use_rp_contrastive = True
    model.rp_contrastive_lambda = 1.0
    model.rp_contrastive_margin = 0.5
    model.rp_contrastive_neg_samples = 10'''

new_lines = '''    # Spreading activation is primary; RP is backup
    model.use_rp_hidden = True
    model.use_rp_contrastive = False
    model.use_cross_domain_alignment = True'''

content = content.replace(old_lines, new_lines)

with open(path, 'w') as f:
    f.write(content)

print("Updated experiment config")