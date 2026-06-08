path = r"C:\Users\Likhith\Documents\projects\ravana\experiments\experiment_cross_domain.py"
with open(path, 'r') as f:
    content = f.read()

# Find Phase 3 (after both domains trained) and add alignment
old_phase3 = '''[Phase 3] Cross-domain transfer probes...
    model.set_domain(None)
    transfer_results = test_structural_transfer(model, tokenizer, domain_a["test"], domain_b["test"])
    print(f"  Cross-domain top-1 accuracy: {transfer_results['top1_accuracy']:.1%}")
    print(f"  Cross-domain top-10 accuracy: {transfer_results['top10_accuracy']:.1%}")'''

new_phase3 = '''[Phase 3] Cross-domain relation alignment (joint, after both domains trained)...
    model.set_domain(None)
    print("  Running cross-domain relation alignment...")
    for align_step in range(20):
        model._cross_domain_relation_alignment()
    print("  Alignment complete.")

[Phase 3.1] Cross-domain transfer probes (after alignment)...
    model.set_domain(None)
    transfer_results = test_structural_transfer(model, tokenizer, domain_a["test"], domain_b["test"])
    print(f"  Cross-domain top-1 accuracy: {transfer_results['top1_accuracy']:.1%}")
    print(f"  Cross-domain top-10 accuracy: {transfer_results['top10_accuracy']:.1%}")

    # RP-only transfer
    model.disable_spreading_activation = True
    rp_transfer_results = test_structural_transfer(model, tokenizer, domain_a["test"], domain_b["test"])
    model.disable_spreading_activation = False
    print(f"  RP-only cross-domain top-1: {rp_transfer_results['top1_accuracy']:.1%}")
    print(f"  RP-only cross-domain top-10: {rp_transfer_results['top10_accuracy']:.1%}")'''

content = content.replace(old_phase3, new_phase3)

# Also update Phase 4 (after sleep) to run alignment again
old_phase4 = '''[Phase 4] After sleep cycle...
[Sleep] Anti-Hebbian pruned 341 polluted edges
  Domain A after sleep: top1=0.0%, top10=0.0%
  Domain B after sleep: top1=0.0%, top10=0.0%
  Graph: 352 nodes, 1426 edges
  Cross-domain probes after sleep: top1=0.0%, top10=0.0%'''

new_phase4 = '''[Phase 4] After sleep cycle...
[Sleep] Anti-Hebbian pruned 341 polluted edges
  Domain A after sleep: top1=0.0%, top10=0.0%
  Domain B after sleep: top1=0.0%, top10=0.0%
  Graph: 352 nodes, 1426 edges
  Cross-domain probes after sleep: top1=0.0%, top10=0.0%

  # Run alignment after sleep too
  print("  Running cross-domain relation alignment (post-sleep)...")
  for align_step in range(20):
      model._cross_domain_relation_alignment()
  
  model.set_domain(None)
  post_sleep_transfer = test_structural_transfer(model, tokenizer, domain_a["test"], domain_b["test"])
  print(f"  Post-sleep alignment cross-domain top-1: {post_sleep_transfer['top1_accuracy']:.1%}")
  print(f"  Post-sleep alignment cross-domain top-10: {post_sleep_transfer['top10_accuracy']:.1%}")'''

content = content.replace(old_phase4, new_phase4)

with open(path, 'w') as f:
    f.write(content)

print("Added joint alignment phases")