"""A/B test: Predictive coding ON vs OFF.

Tests whether graph-wide learning signals help the model generalize
vs just memorize. Uses character-level prediction (the actual interface).
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import time
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import SimpleTokenizer


def weight_entropy(graph):
    weights = np.array([e.weight for e in graph.edges.values()])
    if len(weights) == 0:
        return 0.0
    total = weights.sum()
    if total <= 0:
        return 0.0
    p = weights / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def eval_model(model, tok, facts):
    correct = 0
    total = len(facts)
    details = []
    for fact in facts:
        ids = tok.encode(fact)
        if len(ids) < 2:
            continue
        logits = model.forward(np.array([ids[:-1]], dtype=np.int64))
        pred_id = int(np.argmax(logits.data.flatten()))
        target_id = ids[-1]
        is_correct = pred_id == target_id
        if is_correct:
            correct += 1
        details.append((fact, tok.decode([target_id]), tok.decode([pred_id]), is_correct))
    return correct / max(1, total), details


def run_variant(variant, tok, train_facts, novel_facts, epochs=200):
    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=tok.vocab_size, sleep_interval=999999)
    
    if variant == 'baseline':
        model.predictive_coding_enabled = False
    
    entropy_log = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        for fact in train_facts:
            ids = tok.encode(fact)
            if len(ids) < 2:
                continue
            model.learn(np.array([ids[:-1]], dtype=np.int64), np.array([[ids[-1]]], dtype=np.int64))
        if epoch % 50 == 0:
            H = weight_entropy(model.graph)
            entropy_log.append((epoch, H))
    train_time = time.perf_counter() - t0
    
    train_acc, train_details = eval_model(model, tok, train_facts)
    novel_acc, novel_details = eval_model(model, tok, novel_facts)
    H = weight_entropy(model.graph)
    strong = len([e for e in model.graph.edges.values() if e.weight > 0.5])
    
    return {
        'train_acc': train_acc, 'novel_acc': novel_acc,
        'entropy': H, 'strong_edges': strong,
        'nodes': len(model.graph.nodes), 'edges': len(model.graph.edges),
        'train_time': train_time, 'entropy_log': entropy_log,
        'train_details': train_details, 'novel_details': novel_details,
    }


def main():
    tok = SimpleTokenizer()
    
    train_facts = [
        'cat has tail', 'dog has tail', 'fish has fin',
        'bird has wing', 'horse has leg', 'snake has fang',
        'frog has tongue', 'whale has flipper', 'deer has antler',
        'lion has mane', 'eagle has talon', 'shark has tooth',
    ]
    
    novel_facts = [
        'tiger has tail',   # known relation "has tail", unseen subject
        'wolf has fang',    # known relation "has fang", unseen subject
        'hawk has wing',    # known relation "has wing", unseen subject
    ]
    
    print("=" * 60)
    print("A/B TEST: Predictive Coding ON vs OFF")
    print("=" * 60)
    print(f"Training: {len(train_facts)} facts, 200 epochs")
    print(f"Novel: {len(novel_facts)} facts (unseen subjects, known relations)")
    
    results = {}
    for variant in ['baseline', 'predictive_coding']:
        print(f"\n{'─'*40}")
        print(f"  {variant.upper().replace('_', ' ')}")
        print(f"{'─'*40}")
        
        r = run_variant(variant, tok, train_facts, novel_facts, epochs=200)
        results[variant] = r
        
        print(f"  Graph: {r['nodes']} nodes, {r['edges']} edges, {r['strong_edges']} strong")
        print(f"  Train acc:  {r['train_acc']:.0%}")
        print(f"  Novel acc:  {r['novel_acc']:.0%}")
        print(f"  Entropy:    {r['entropy']:.3f}")
        print(f"  Time:       {r['train_time']:.1f}s")
        
        print(f"  Novel details:")
        for fact, target, pred, ok in r['novel_details']:
            print(f"    {fact}: target={target!r} pred={pred!r} {'✓' if ok else '✗'}")
    
    # Comparison
    b = results['baseline']
    p = results['predictive_coding']
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Baseline':>10} {'PredCode':>10} {'Delta':>10}")
    print(f"  {'─'*50}")
    print(f"  {'train_acc':<20} {b['train_acc']:>9.0%} {p['train_acc']:>9.0%} {p['train_acc']-b['train_acc']:>+9.0%}")
    print(f"  {'novel_acc':<20} {b['novel_acc']:>9.0%} {p['novel_acc']:>9.0%} {p['novel_acc']-b['novel_acc']:>+9.0%}")
    print(f"  {'entropy':<20} {b['entropy']:>10.3f} {p['entropy']:>10.3f} {p['entropy']-b['entropy']:>+10.3f}")
    print(f"  {'strong_edges':<20} {b['strong_edges']:>10} {p['strong_edges']:>10} {p['strong_edges']-b['strong_edges']:>+10}")
    print(f"  {'edges':<20} {b['edges']:>10} {p['edges']:>10} {p['edges']-b['edges']:>+10}")
    print(f"  {'time_s':<20} {b['train_time']:>10.1f} {p['train_time']:>10.1f} {p['train_time']-b['train_time']:>+10.1f}")


if __name__ == '__main__':
    main()
