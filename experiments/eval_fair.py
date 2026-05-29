"""Fair RLM evaluation — top-1 accuracy + novel probes + out-of-distribution.

The existing benchmark uses top-10 accuracy which inflates scores (58% top-10
with 100+ vocab = only ~5x random). This script provides a rigorous eval:

1. Top-1 accuracy (true prediction, not "in top-10")
2. Novel token pairs (never seen during training)
3. Out-of-distribution probes (cross-domain patterns)
4. Statistical significance (multiple seeds)
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import time
from ravana_ml.nn.rlm import RLM


def measure_accuracy(model, pairs, k=1):
    """Measure top-k accuracy on token pairs."""
    if not pairs:
        return 0.0
    correct = 0
    for src, tgt in pairs:
        logits = np.asarray(model.forward(np.array([src, tgt])).data).flatten()
        ranked = np.argsort(logits)[::-1]
        if tgt in set(ranked[:k]):
            correct += 1
    return correct / len(pairs)


def run_fair_eval(vocab_size=30, n_train=10, n_test=5, train_steps=500, seed=42):
    """Run a fair evaluation with train/test split."""
    np.random.seed(seed)
    
    # Split tokens into train and test sets
    all_tokens = list(range(vocab_size))
    np.random.shuffle(all_tokens)
    
    # Create train pairs (sequential: token[i] -> token[i+1])
    train_tokens = all_tokens[:n_train * 2]
    train_pairs = [(train_tokens[i], train_tokens[i+1]) for i in range(0, n_train * 2, 2)]
    
    # Create test pairs (novel: token[i] -> token[i+1] from held-out tokens)
    test_tokens = all_tokens[n_train * 2:n_train * 2 + n_test * 2]
    test_pairs = [(test_tokens[i], test_tokens[i+1]) for i in range(0, n_test * 2, 2)]
    
    # Create OOD pairs (cross-domain: use remaining tokens)
    ood_tokens = all_tokens[n_train * 2 + n_test * 2:]
    ood_pairs = [(ood_tokens[i], ood_tokens[i+1]) for i in range(0, len(ood_tokens) - 1, 2)]
    
    model = RLM(vocab_size=vocab_size, embed_dim=64, concept_dim=64, n_hidden=128,
                n_concepts=vocab_size, sleep_interval=999999)  # no sleep for clean eval
    
    # Baseline (before training)
    baseline_train = measure_accuracy(model, train_pairs, k=1)
    baseline_test = measure_accuracy(model, test_pairs, k=1)
    
    # Train
    t0 = time.time()
    for step in range(train_steps):
        for s, t in train_pairs:
            model.forward(np.array([s, t]))
            model.learn(np.array([s, t]), np.array([t]))
    train_time = time.time() - t0
    
    # Evaluate
    train_top1 = measure_accuracy(model, train_pairs, k=1)
    train_top5 = measure_accuracy(model, train_pairs, k=5)
    test_top1 = measure_accuracy(model, test_pairs, k=1)
    test_top5 = measure_accuracy(model, test_pairs, k=5)
    
    ood_top1 = 0.0
    ood_top5 = 0.0
    if ood_pairs:
        ood_top1 = measure_accuracy(model, ood_pairs, k=1)
        ood_top5 = measure_accuracy(model, ood_pairs, k=5)
    
    return {
        'seed': seed,
        'n_train': len(train_pairs),
        'n_test': len(test_pairs),
        'n_ood': len(ood_pairs),
        'train_steps': train_steps,
        'train_time_s': train_time,
        'baseline_train_top1': baseline_train,
        'baseline_test_top1': baseline_test,
        'train_top1': train_top1,
        'train_top5': train_top5,
        'test_top1': test_top1,
        'test_top5': test_top5,
        'ood_top1': ood_top1,
        'ood_top5': ood_top5,
        'edges': len(model.graph.edges),
        'concepts': len(model.graph.nodes),
        'sleep_cycles': model.sleep_cycles_completed,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("FAIR RLM EVALUATION — Top-1 Accuracy")
    print("=" * 60)
    
    seeds = [42, 123, 456]
    all_results = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        r = run_fair_eval(vocab_size=30, n_train=10, n_test=5, train_steps=500, seed=seed)
        all_results.append(r)
        print(f"  Train top-1: {r['train_top1']:.0%}  top-5: {r['train_top5']:.0%}")
        print(f"  Test  top-1: {r['test_top1']:.0%}  top-5: {r['test_top5']:.0%}")
        if r['n_ood'] > 0:
            print(f"  OOD   top-1: {r['ood_top1']:.0%}  top-5: {r['ood_top5']:.0%}")
        print(f"  Edges: {r['edges']}, Concepts: {r['concepts']}, Time: {r['train_time_s']:.1f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (mean ± std)")
    print("=" * 60)
    for metric in ['train_top1', 'train_top5', 'test_top1', 'test_top5']:
        vals = [r[metric] for r in all_results]
        print(f"  {metric}: {np.mean(vals):.1%} ± {np.std(vals):.1%}")
    
    ood_vals = [r['ood_top1'] for r in all_results if r['n_ood'] > 0]
    if ood_vals:
        print(f"  ood_top1: {np.mean(ood_vals):.1%} ± {np.std(ood_vals):.1%}")
