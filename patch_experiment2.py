path = r"C:\Users\Likhith\Documents\projects\ravana\experiments\experiment_cross_domain.py"
with open(path, 'r') as f:
    content = f.read()

# Change n_train_repeats default from 3 to 10
old_config = '''@dataclass
class CrossDomainConfig:
    n_train_repeats: int = 3            # repeats of each fact during training
    n_test_probes: int = 50             # probes per test'''

new_config = '''@dataclass
class CrossDomainConfig:
    n_train_repeats: int = 10           # repeats of each fact during training
    n_test_probes: int = 50             # probes per test'''

content = content.replace(old_config, new_config)

# Also add RP-only evaluation 
old_eval = '''    model.set_domain(None)  # Soft routing for evaluation
    post_a_on_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    post_a_on_b = evaluate_rlm(model, domain_b["test"], tokenizer)'''

new_eval = '''    model.set_domain(None)  # Soft routing for evaluation
    
    # Standard evaluation (with spreading activation)
    post_a_on_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    post_a_on_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    
    # RP-only evaluation (disable spreading activation)
    model.disable_spreading_activation = True
    rp_post_a_on_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    rp_post_a_on_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    model.disable_spreading_activation = False
    
    print(f"  RP-only Domain A test: top1={rp_post_a_on_a['top1_accuracy']:.1%}, top10={rp_post_a_on_a['top10_accuracy']:.1%}")
    print(f"  RP-only Domain B zero-shot: top1={rp_post_a_on_b['top1_accuracy']:.1%}, top10={rp_post_a_on_b['top10_accuracy']:.1%}")'''

content = content.replace(old_eval, new_eval)

# Also update Phase 2 evaluation
old_eval2 = '''    model.set_domain(None)
    post_b_on_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    post_b_on_a = evaluate_rlm(model, domain_a["test"], tokenizer)'''

new_eval2 = '''    model.set_domain(None)
    post_b_on_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    post_b_on_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    
    # RP-only evaluation
    model.disable_spreading_activation = True
    rp_post_b_on_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    rp_post_b_on_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    model.disable_spreading_activation = False
    
    print(f"  RP-only Domain B test: top1={rp_post_b_on_b['top1_accuracy']:.1%}, top10={rp_post_b_on_b['top10_accuracy']:.1%}")
    print(f"  RP-only Domain A retention: top1={rp_post_b_on_a['top1_accuracy']:.1%}, top10={rp_post_b_on_a['top10_accuracy']:.1%}")'''

content = content.replace(old_eval2, new_eval2)

# Update Phase 3 and 4 similarly
old_eval3 = '''    model.set_domain(None)
    transfer_results = test_structural_transfer(model, tokenizer, domain_a["test"], domain_b["test"])
    print(f"  Cross-domain top-1 accuracy: {transfer_results['top1_accuracy']:.1%}")
    print(f"  Cross-domain top-10 accuracy: {transfer_results['top10_accuracy']:.1%}")'''

new_eval3 = '''    model.set_domain(None)
    transfer_results = test_structural_transfer(model, tokenizer, domain_a["test"], domain_b["test"])
    print(f"  Cross-domain top-1 accuracy: {transfer_results['top1_accuracy']:.1%}")
    print(f"  Cross-domain top-10 accuracy: {transfer_results['top10_accuracy']:.1%}")
    
    # RP-only transfer
    model.disable_spreading_activation = True
    rp_transfer_results = test_structural_transfer(model, tokenizer, domain_a["test"], domain_b["test"])
    model.disable_spreading_activation = False
    print(f"  RP-only cross-domain top-1: {rp_transfer_results['top1_accuracy']:.1%}")
    print(f"  RP-only cross-domain top-10: {rp_transfer_results['top10_accuracy']:.1%}")'''

content = content.replace(old_eval3, new_eval3)

old_eval4 = '''    model.set_domain(None)
    post_sleep_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    post_sleep_b = evaluate_rlm(model, domain_b["test"], tokenizer)'''

new_eval4 = '''    model.set_domain(None)
    post_sleep_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    post_sleep_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    
    # RP-only after sleep
    model.disable_spreading_activation = True
    rp_post_sleep_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    rp_post_sleep_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    model.disable_spreading_activation = False
    print(f"  RP-only Domain A after sleep: top1={rp_post_sleep_a['top1_accuracy']:.1%}, top10={rp_post_sleep_a['top10_accuracy']:.1%}")
    print(f"  RP-only Domain B after sleep: top1={rp_post_sleep_b['top1_accuracy']:.1%}, top10={rp_post_sleep_b['top10_accuracy']:.1%}")'''

content = content.replace(old_eval4, new_eval4)

with open(path, 'w') as f:
    f.write(content)

print("Updated experiment with RP-only eval and more training repeats")