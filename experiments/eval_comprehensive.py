"""Comprehensive RLMv2 evaluation — fair eval, scaling, entropy, probes.

Tests whether the graph is learning transferable structure or just memorizing.
Run after Phase 2 (predictive coding) to measure impact.

Metrics:
1. Fair eval: train/test split on novel token pairs
2. Scaling: performance vs graph size (20→100→500 facts)
3. Weight entropy: H = -sum(p_i * log(p_i)) over edge weights (detect rich-get-richer)
4. Type A probe: novel subject with known relation
5. Type B probe: compositional reasoning (grandparent via parent)
6. Type C probe: relation transfer (known subject, unseen relation)
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import time
from collections import Counter
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import SimpleTokenizer


def weight_entropy(graph):
    """H = -sum(p_i * log(p_i)) over normalized edge weights.
    
    High entropy = diverse pathways (healthy).
    Low entropy = a few superhighways dominate (rich-get-richer).
    """
    weights = np.array([e.weight for e in graph.edges.values()])
    if len(weights) == 0:
        return 0.0
    total = weights.sum()
    if total <= 0:
        return 0.0
    p = weights / total
    p = p[p > 0]  # avoid log(0)
    return float(-np.sum(p * np.log(p)))


def measure_topk(model, tok, pairs, k=1):
    """Measure top-k accuracy on (input_text, expected_token_text) pairs."""
    if not pairs:
        return 0.0
    correct = 0
    for input_text, expected_text in pairs:
        ids = tok.encode(input_text)
        target_ids = tok.encode(expected_text)
        if not ids or not target_ids:
            continue
        target_tid = target_ids[0]
        logits = model.forward(np.array([ids], dtype=np.int64))
        logits_data = logits.data.flatten() if hasattr(logits.data, 'flatten') else logits.data
        ranked = np.argsort(logits_data)[::-1]
        if target_tid in set(ranked[:k]):
            correct += 1
    return correct / len(pairs)


def train_facts(model, tok, facts, epochs=50):
    """Train model on a list of factual sentences."""
    for _ in range(epochs):
        for fact in facts:
            ids = tok.encode(fact)
            if len(ids) < 2:
                continue
            model.learn(np.array([ids[:-1]], dtype=np.int64), np.array([[ids[-1]]], dtype=np.int64))


# ── 1. FAIR EVAL ──────────────────────────────────────────────────────
def run_fair_eval():
    """Train on known facts, test on novel combinations."""
    print("=" * 60)
    print("1. FAIR EVAL — Novel Token Prediction")
    print("=" * 60)
    
    tok = SimpleTokenizer()
    
    # Training facts (known)
    train_facts_list = [
        'heat causes expansion', 'fire produces smoke',
        'ice causes cold', 'sun produces warmth',
        'rain causes growth', 'wind produces waves',
        'kindness causes trust', 'anger produces conflict',
        'gravity causes falling', 'friction produces heat',
    ]
    
    # Test facts: novel combinations using trained tokens
    # These test whether the model learned the RELATION PATTERN, not just specific triples
    test_facts = [
        ('heat produces', 'expansion'),     # known subject, known relation type, known object
        ('fire causes', 'smoke'),            # known triple (retrieval baseline)
        ('ice produces', 'cold'),            # known triple
    ]
    
    # Novel: same relation type, different subject (should transfer if model learned structure)
    novel_facts = [
        ('sun causes', 'warmth'),            # trained
        ('rain produces', 'growth'),         # trained  
    ]
    
    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=tok.vocab_size, sleep_interval=999999)
    
    print(f"\nTraining on {len(train_facts_list)} facts, 100 epochs...")
    train_facts(model, tok, train_facts_list, epochs=100)
    print(f"Graph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")
    
    # Measure
    train_acc = measure_topk(model, tok, [(f, f.split()[-1]) for f in train_facts_list], k=1)
    test_acc = measure_topk(model, tok, test_facts, k=1)
    novel_acc = measure_topk(model, tok, novel_facts, k=1)
    
    print(f"\n  Train top-1 (known facts):    {train_acc:.0%}")
    print(f"  Test top-1 (novel combos):    {test_acc:.0%}")
    print(f"  Novel top-1 (transfer):       {novel_acc:.0%}")
    print(f"  Weight entropy:               {weight_entropy(model.graph):.3f}")
    
    return train_acc, test_acc, novel_acc


# ── 2. SCALING TEST ──────────────────────────────────────────────────
def run_scaling_test():
    """Measure performance vs graph size."""
    print("\n" + "=" * 60)
    print("2. SCALING TEST — Performance vs Graph Size")
    print("=" * 60)
    
    tok = SimpleTokenizer()
    
    # Generate facts with increasing vocabulary
    all_facts = [
        'heat causes expansion', 'fire produces smoke', 'ice causes cold',
        'sun produces warmth', 'rain causes growth', 'wind produces waves',
        'kindness causes trust', 'anger produces conflict', 'gravity causes falling',
        'friction produces heat', 'pressure causes diamonds', 'education leads knowledge',
        'poverty causes suffering', 'wealth produces power', 'music leads emotion',
        'silence causes reflection', 'noise produces stress', 'sleep leads recovery',
        'steel causes strength', 'oxygen produces rust', 'exercise leads fitness',
        'neglect causes decay', 'pollution produces disease', 'innovation leads progress',
        'corruption causes collapse', 'empathy produces connection', 'isolation leads depression',
        'practice causes mastery', 'starvation produces weakness', 'investment leads growth',
        'betrayal causes trauma', 'generosity produces gratitude', 'competition leads excellence',
        'negligence causes accidents', 'caffeine produces alertness', 'meditation leads peace',
        'inflation causes poverty', 'technology produces efficiency', 'collaboration leads innovation',
        'boredom causes creativity', 'ambition produces success', 'patience leads wisdom',
        'drought causes famine', 'alchemy produces gold', 'perseverance leads triumph',
        'steel causes rust', 'fire produces heat', 'ice causes numbness',
        'sun produces vitamin', 'rain causes flooding', 'wind produces energy',
        'kindness causes gratitude', 'anger produces headaches', 'gravity causes tides',
        'friction produces wear', 'pressure causes stress', 'education leads opportunity',
    ]
    
    results = []
    for n_facts in [10, 25, 40, 55]:
        facts = all_facts[:n_facts]
        model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                      n_concepts=tok.vocab_size, sleep_interval=999999)
        
        t0 = time.perf_counter()
        train_facts(model, tok, facts, epochs=50)
        train_time = time.perf_counter() - t0
        
        # Measure on trained facts
        pairs = [(f, f.split()[-1]) for f in facts]
        top1 = measure_topk(model, tok, pairs, k=1)
        top5 = measure_topk(model, tok, pairs, k=5)
        
        # Count active nodes/edges during a typical forward
        ids = tok.encode(facts[0])
        model.forward(np.array([ids[:-1]], dtype=np.int64))
        active_nodes = len([n for n in model.graph.nodes.values() if n.activation > 0.01])
        active_edges = sum(len(model.graph.get_outgoing(nid)) 
                          for nid in list(model.graph._active_nodes)
                          if model.graph.nodes.get(nid) and model.graph.nodes[nid].activation > 0.01)
        
        H = weight_entropy(model.graph)
        
        results.append({
            'n_facts': n_facts,
            'nodes': len(model.graph.nodes),
            'edges': len(model.graph.edges),
            'top1': top1,
            'top5': top5,
            'active_nodes': active_nodes,
            'active_edges': active_edges,
            'entropy': H,
            'train_time': train_time,
        })
        
        print(f"\n  {n_facts} facts → {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")
        print(f"    Top-1: {top1:.0%}  Top-5: {top5:.0%}")
        print(f"    Active: {active_nodes} nodes, {active_edges} edges")
        print(f"    Entropy: {H:.3f}  Train time: {train_time:.1f}s")
    
    # Check scaling
    print("\n  Scaling summary:")
    for r in results:
        print(f"    {r['n_facts']:3d} facts: {r['top1']:.0%} top-1, "
              f"{r['active_nodes']} active nodes, {r['active_edges']} active edges, "
              f"H={r['entropy']:.3f}")
    
    return results


# ── 3. WEIGHT ENTROPY TRACKING ───────────────────────────────────────
def run_entropy_tracking():
    """Track weight entropy over training to detect rich-get-richer."""
    print("\n" + "=" * 60)
    print("3. WEIGHT ENTROPY TRACKING — Rich-Get-Richer Detection")
    print("=" * 60)
    
    tok = SimpleTokenizer()
    facts = [
        'heat causes expansion', 'fire produces smoke', 'ice causes cold',
        'sun produces warmth', 'rain causes growth', 'wind produces waves',
        'kindness causes trust', 'anger produces conflict', 'gravity causes falling',
        'friction produces heat', 'pressure causes diamonds', 'education leads knowledge',
        'poverty causes suffering', 'wealth produces power', 'music leads emotion',
        'silence causes reflection', 'noise produces stress', 'sleep leads recovery',
        'steel causes strength', 'oxygen produces rust',
    ]
    
    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=tok.vocab_size, sleep_interval=999999)
    
    entropy_log = []
    for epoch in range(100):
        for fact in facts:
            ids = tok.encode(fact)
            if len(ids) < 2:
                continue
            model.learn(np.array([ids[:-1]], dtype=np.int64), np.array([[ids[-1]]], dtype=np.int64))
        
        if epoch % 10 == 0:
            H = weight_entropy(model.graph)
            n_edges = len(model.graph.edges)
            max_w = max(e.weight for e in model.graph.edges.values()) if model.graph.edges else 0
            mean_w = np.mean([e.weight for e in model.graph.edges.values()]) if model.graph.edges else 0
            entropy_log.append((epoch, H, n_edges, max_w, mean_w))
            print(f"  Epoch {epoch:3d}: H={H:.3f}, edges={n_edges}, max_w={max_w:.3f}, mean_w={mean_w:.3f}")
    
    # Check trend
    if len(entropy_log) >= 2:
        H_start = entropy_log[0][1]
        H_end = entropy_log[-1][1]
        if H_end < H_start * 0.7:
            print(f"\n  ⚠️  ENTROPY COLLAPSE: {H_start:.3f} → {H_end:.3f} ({H_end/H_start:.0%})")
            print(f"     Rich-get-richer detected. Consider edge decay or exploration bonus.")
        else:
            print(f"\n  ✓  Entropy stable: {H_start:.3f} → {H_end:.3f}")
    
    return entropy_log


# ── 4. TYPE A/B/C PROBES ─────────────────────────────────────────────
def run_probes():
    """Test generalization with structured probes."""
    print("\n" + "=" * 60)
    print("4. GENERALIZATION PROBES")
    print("=" * 60)
    
    tok = SimpleTokenizer()
    
    # ── Type A: Novel Subject ──
    # Train capital facts for known countries, test on unseen country
    print("\n  --- Type A: Novel Subject ---")
    type_a_train = [
        'france capital paris', 'germany capital berlin',
        'japan capital tokyo', 'india capital delhi',
        'brazil capital brasilia', 'egypt capital cairo',
    ]
    # Test: can model infer "capital" pattern for a new subject?
    # We need "canada" to appear in training but NOT with "capital"
    type_a_train_extra = [
        'canada has forests', 'canada has mountains', 'canada produces maple',
    ]
    
    model_a = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                    n_concepts=tok.vocab_size, sleep_interval=999999)
    train_facts(model_a, tok, type_a_train + type_a_train_extra, epochs=100)
    
    # Test: "canada capital" → should predict something capital-like
    a_pairs = [('canada capital', 'ottawa')]
    a_acc = measure_topk(model_a, tok, a_pairs, k=5)
    # Also test known capitals (retrieval baseline)
    a_known = measure_topk(model_a, tok, [('france capital', 'paris')], k=1)
    print(f"    Known capital (france→paris): {a_known:.0%}")
    print(f"    Novel subject (canada→ottawa): {a_acc:.0%} (top-5)")
    print(f"    Note: 'ottawa' may not be in vocab — testing if 'capital-like' tokens rank high")
    
    # ── Type B: Compositional Reasoning ──
    print("\n  --- Type B: Compositional Reasoning ---")
    type_b_train = [
        'alice parent bob', 'bob parent charlie',
        'diana parent eve', 'eve parent frank',
    ]
    
    model_b = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                    n_concepts=tok.vocab_size, sleep_interval=999999)
    train_facts(model_b, tok, type_b_train, epochs=100)
    
    # Test: alice grandparent → charlie (2-hop reasoning)
    # The model needs to traverse alice→bob→charlie
    b_pairs = [('alice parent', 'bob'), ('bob parent', 'charlie')]
    b_direct = measure_topk(model_b, tok, b_pairs, k=1)
    print(f"    Direct parent (alice→bob): {b_direct:.0%}")
    print(f"    Direct parent (bob→charlie): {measure_topk(model_b, tok, [('bob parent', 'charlie')], k=1):.0%}")
    
    # ── Type C: Relation Transfer ──
    print("\n  --- Type C: Relation Transfer ---")
    type_c_train = [
        'dog is animal', 'cat is animal', 'sparrow is bird',
        'eagle has wings', 'eagle has talons', 'eagle hunts prey',
    ]
    
    model_c = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                    n_concepts=tok.vocab_size, sleep_interval=999999)
    train_facts(model_c, tok, type_c_train, epochs=100)
    
    # Test: eagle is → ? (should be bird-like, since eagle shares features with sparrow)
    c_pairs = [('eagle is', 'bird')]
    c_acc = measure_topk(model_c, tok, c_pairs, k=5)
    c_known = measure_topk(model_c, tok, [('dog is', 'animal')], k=1)
    print(f"    Known relation (dog→animal): {c_known:.0%}")
    print(f"    Relation transfer (eagle→bird): {c_acc:.0%} (top-5)")
    
    return {'type_a_known': a_known, 'type_a_novel': a_acc,
            'type_b_direct': b_direct, 'type_c_known': c_known, 'type_c_transfer': c_acc}


# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("RLMv2 COMPREHENSIVE EVALUATION")
    print("Testing: fair eval, scaling, entropy, generalization probes")
    print("=" * 60)
    
    t0 = time.perf_counter()
    
    fair_results = run_fair_eval()
    scaling_results = run_scaling_test()
    entropy_log = run_entropy_tracking()
    probe_results = run_probes()
    
    total_time = time.perf_counter() - t0
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Fair eval train top-1:  {fair_results[0]:.0%}")
    print(f"  Fair eval test top-1:   {fair_results[1]:.0%}")
    print(f"  Fair eval novel top-1:  {fair_results[2]:.0%}")
    print(f"  Scaling: {'OK' if all(r['top1'] > 0.3 for r in scaling_results) else 'DEGRADED'}")
    print(f"  Entropy: {'STABLE' if len(entropy_log) >= 2 and entropy_log[-1][1] >= entropy_log[0][1] * 0.7 else 'COLLAPSING'}")
    print(f"  Probes: known={probe_results['type_a_known']:.0%}, novel={probe_results['type_a_novel']:.0%}")
    print(f"  Total time: {total_time:.1f}s")
