"""A/B test with WORD-LEVEL tokenization.

Tests whether predictive coding helps when the model operates on words.
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import time
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.word_tokenizer import WordTokenizer


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
    return correct / max(1, len(facts)), details


def run_variant(pc_enabled, tok, train_facts, epochs=200):
    model = RLMv2(vocab_size=2000, embed_dim=32, concept_dim=32,
                  n_concepts=2000, sleep_interval=999999)
    model.predictive_coding_enabled = pc_enabled
    
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
    
    return model, train_time, entropy_log


def main():
    tok = WordTokenizer()
    
    train_facts = [
        'cat has tail', 'dog has tail', 'fish has fin',
        'bird has wing', 'horse has leg', 'snake has fang',
        'frog has tongue', 'whale has flipper', 'deer has antler',
        'lion has mane', 'eagle has talon', 'shark has tooth',
    ]
    
    novel_transfer = [
        'tiger has tail',   # "has tail" trained with cat/dog, tiger unseen
        'wolf has fang',    # "has fang" trained with snake, wolf unseen
        'hawk has wing',    # "has wing" trained with bird, hawk unseen
        'bear has mane',    # "has mane" trained with lion, bear unseen
    ]
    
    novel_hard = [
        'tiger has stripe', # both tiger and stripe unseen
        'wolf has howl',    # both wolf and howl unseen
    ]
    
    print("=" * 60)
    print("A/B TEST: Predictive Coding ON vs OFF (Word-Level)")
    print("=" * 60)
    print(f"Training: {len(train_facts)} facts, 200 epochs")
    print(f"Novel (transfer): {len(novel_transfer)} facts")
    print(f"Novel (hard): {len(novel_hard)} facts")
    
    results = {}
    for variant_name, pc_flag in [('BASELINE', False), ('PREDICTIVE CODING', True)]:
        print(f"\n{'─'*40}")
        print(f"  {variant_name}")
        print(f"{'─'*40}")
        
        model, train_time, entropy_log = run_variant(pc_flag, tok, train_facts, epochs=200)
        
        train_acc, train_det = eval_model(model, tok, train_facts)
        transfer_acc, transfer_det = eval_model(model, tok, novel_transfer)
        hard_acc, hard_det = eval_model(model, tok, novel_hard)
        H = weight_entropy(model.graph)
        strong = len([e for e in model.graph.edges.values() if e.weight > 0.5])
        
        r = {
            'train_acc': train_acc, 'transfer_acc': transfer_acc, 'hard_acc': hard_acc,
            'entropy': H, 'strong_edges': strong,
            'nodes': len(model.graph.nodes), 'edges': len(model.graph.edges),
            'train_time': train_time, 'entropy_log': entropy_log,
            'transfer_det': transfer_det, 'hard_det': hard_det,
        }
        results[variant_name] = r
        
        print(f"  Graph: {r['nodes']} nodes, {r['edges']} edges, {r['strong_edges']} strong")
        print(f"  Train acc:      {r['train_acc']:.0%}")
        print(f"  Transfer acc:   {r['transfer_acc']:.0%}")
        print(f"  Hard novel:     {r['hard_acc']:.0%}")
        print(f"  Entropy:        {r['entropy']:.3f}")
        print(f"  Time:           {r['train_time']:.1f}s")
        
        print(f"  Transfer details:")
        for fact, target, pred, ok in r['transfer_det']:
            print(f"    {fact}: target={target!r} pred={pred!r} {'✓' if ok else '✗'}")
        print(f"  Hard novel details:")
        for fact, target, pred, ok in r['hard_det']:
            print(f"    {fact}: target={target!r} pred={pred!r} {'✓' if ok else '✗'}")
    
    # Comparison
    b = results['BASELINE']
    p = results['PREDICTIVE CODING']
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Baseline':>10} {'PredCode':>10} {'Delta':>10}")
    print(f"  {'─'*50}")
    for metric in ['train_acc', 'transfer_acc', 'hard_acc', 'entropy', 'strong_edges', 'edges', 'train_time']:
        bv = b[metric]
        pv = p[metric]
        if metric in ('train_acc', 'transfer_acc', 'hard_acc'):
            print(f"  {metric:<20} {bv:>9.0%} {pv:>9.0%} {pv-bv:>+9.0%}")
        elif metric == 'entropy':
            print(f"  {metric:<20} {bv:>10.3f} {pv:>10.3f} {pv-bv:>+10.3f}")
        elif metric == 'train_time':
            print(f"  {metric:<20} {bv:>10.1f} {pv:>10.1f} {pv-bv:>+10.1f}")
        else:
            print(f"  {metric:<20} {bv:>10} {pv:>10} {pv-bv:>+10}")


if __name__ == '__main__':
    main()
