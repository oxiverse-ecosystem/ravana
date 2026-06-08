path = r"C:\Users\Likhith\Documents\projects\ravana\experiments\experiment_cross_domain.py"
with open(path, 'r') as f:
    content = f.read()

# Phase 3 - after Domain B training, add alignment
old_phase3 = '''    # ── Phase 3: Cross-Domain Transfer Probes ──
    print("\\n[Phase 3] Cross-domain transfer probes...")
    transfer_probes = test_structural_transfer(
        model, tokenizer, domain_a["test"], domain_b["test"])
    print(f"  Cross-domain top-1 accuracy: {transfer_probes['top1_accuracy']:.1%}")
    print(f"  Cross-domain top-10 accuracy: {transfer_probes['top10_accuracy']:.1%}")
    for probe in transfer_probes["probes"]:
        status = "OK" if probe["correct"] else ("~" if probe["in_top10"] else "X")
        print(f"    [{status}] '{probe['input'].strip()}' -> expected '{probe['expected']}'"
              f"  got '{probe['predicted']}'  ({probe['description']})")

    # ── Phase 4: Sleep cycle and re-evaluate ──
    print("\\n[Phase 4] After sleep cycle...")
    model.sleep_cycle()
    post_sleep_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    post_sleep_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    graph_after_sleep = measure_graph_overlap(model)

    print(f"  Domain A after sleep: top1={post_sleep_a['top1_accuracy']:.1%}, top10={post_sleep_a['top10_accuracy']:.1%}")
    print(f"  Domain B after sleep: top1={post_sleep_b['top1_accuracy']:.1%}, top10={post_sleep_b['top10_accuracy']:.1%}")
    print(f"  Graph: {graph_after_sleep['n_nodes']} nodes, {graph_after_sleep['n_edges']} edges")

    # Re-run transfer probes after sleep
    post_sleep_probes = test_structural_transfer(
        model, tokenizer, domain_a["test"], domain_b["test"])
    print(f"  Cross-domain probes after sleep: top1={post_sleep_probes['top1_accuracy']:.1%}, top10={post_sleep_probes['top10_accuracy']:.1%}")'''

new_phase3 = '''    # ── Phase 3: Cross-Domain Relation Alignment (joint) ──
    print("\\n[Phase 3] Cross-domain relation alignment (joint)...")
    model.set_domain(None)
    for align_step in range(20):
        model._cross_domain_relation_alignment()
    print("  Alignment complete.")

    # ── Phase 3.1: Cross-Domain Transfer Probes (after alignment) ──
    print("\\n[Phase 3.1] Cross-domain transfer probes (after alignment)...")
    transfer_probes = test_structural_transfer(
        model, tokenizer, domain_a["test"], domain_b["test"])
    print(f"  Cross-domain top-1 accuracy: {transfer_probes['top1_accuracy']:.1%}")
    print(f"  Cross-domain top-10 accuracy: {transfer_probes['top10_accuracy']:.1%}")
    for probe in transfer_probes["probes"]:
        status = "OK" if probe["correct"] else ("~" if probe["in_top10"] else "X")
        print(f"    [{status}] '{probe['input'].strip()}' -> expected '{probe['expected']}'"
              f"  got '{probe['predicted']}'  ({probe['description']})")

    # RP-only transfer evaluation
    model.disable_spreading_activation = True
    rp_transfer_probes = test_structural_transfer(
        model, tokenizer, domain_a["test"], domain_b["test"])
    model.disable_spreading_activation = False
    print(f"  RP-only cross-domain top-1: {rp_transfer_probes['top1_accuracy']:.1%}")
    print(f"  RP-only cross-domain top-10: {rp_transfer_probes['top10_accuracy']:.1%}")

    # ── Phase 4: Sleep cycle and re-evaluate ──
    print("\\n[Phase 4] After sleep cycle...")
    model.sleep_cycle()
    post_sleep_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    post_sleep_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    graph_after_sleep = measure_graph_overlap(model)

    print(f"  Domain A after sleep: top1={post_sleep_a['top1_accuracy']:.1%}, top10={post_sleep_a['top10_accuracy']:.1%}")
    print(f"  Domain B after sleep: top1={post_sleep_b['top1_accuracy']:.1%}, top10={post_sleep_b['top10_accuracy']:.1%}")
    print(f"  Graph: {graph_after_sleep['n_nodes']} nodes, {graph_after_sleep['n_edges']} edges")

    # Run alignment after sleep too
    print("  Running cross-domain relation alignment (post-sleep)...")
    for align_step in range(20):
        model._cross_domain_relation_alignment()

    # Re-run transfer probes after sleep + alignment
    post_sleep_probes = test_structural_transfer(
        model, tokenizer, domain_a["test"], domain_b["test"])
    print(f"  Cross-domain probes after sleep+align: top1={post_sleep_probes['top1_accuracy']:.1%}, top10={post_sleep_probes['top10_accuracy']:.1%}")'''

content = content.replace(old_phase3, new_phase3)

with open(path, 'w') as f:
    f.write(content)

print("Updated Phase 3 and 4 with alignment")