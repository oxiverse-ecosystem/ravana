path = r"C:\Users\Likhith\Documents\projects\ravana\experiments\experiment_cross_domain.py"
with open(path, 'r') as f:
    content = f.read()

# Replace the model creation
old = '''    np.random.seed(config.seed)
    model = RLMv2(
        vocab_size=vocab_size + 5,
        embed_dim=config.embed_dim,
        concept_dim=config.concept_dim,
        n_concepts=vocab_size,
        sleep_interval=config.sleep_interval,
        gate_concept_creation=False,
    )
    model._tokenizer = tokenizer  # triggers embed init + autoencoder pre-training'''

new = '''    np.random.seed(config.seed)
    model = RLMv2(
        vocab_size=vocab_size + 5,
        embed_dim=config.embed_dim,
        concept_dim=config.concept_dim,
        n_concepts=vocab_size,
        sleep_interval=config.sleep_interval,
        gate_concept_creation=False,
        latent_dim=96,
        hidden_dim=128,
    )
    model._tokenizer = tokenizer  # triggers embed init + autoencoder pre-training
    
    # Enable new RP features
    model.use_rp_hidden = True
    model.use_rp_contrastive = True
    model.rp_contrastive_lambda = 1.0
    model.rp_contrastive_margin = 0.5
    model.rp_contrastive_neg_samples = 10
    
    # For evaluation, also test with spreading disabled to isolate RP
    # model.disable_spreading_activation = True  # Keep spreading enabled for main eval'''

new_content = content.replace(old, new)

with open(path, 'w') as f:
    f.write(new_content)

print("Updated experiment_cross_domain.py")