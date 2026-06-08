path = r"C:\Users\Likhith\Documents\projects\ravana\experiments\experiment_cross_domain.py"
with open(path, 'r') as f:
    content = f.read()

# The model creation - ensure tokenizer is attached BEFORE any forward/learn
old_model = '''    np.random.seed(config.seed)
    model = RLMv2(
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

new_model = '''    np.random.seed(config.seed)
    model = RLMv2(
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
    model._tokenizer = tokenizer  # Attach tokenizer for relation classification
    model.use_cross_domain_alignment = True  # Enable explicit relation alignment
    model.use_rp_contrastive = False  # Disable RP contrastive (not the primary path)'''

content = content.replace(old_model, new_model)

with open(path, 'w') as f:
    f.write(content)

print("Fixed tokenizer attachment")