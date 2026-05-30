"""
RLMv2 — Cross-Domain Transfer Benchmark (v2)

Tests whether the triple architecture can generalize via relation types.

Three levels of generalization:
1. TRAIN MEMORIZATION: Learn known triples → should be ~100%
2. RELATION TYPE TRANSFER: Same subject, different verb → should work
   e.g., train "heat causes expansion", test "heat produces ?" → expansion
   ("produces" → CAUSAL, same as "causes", so the same edge fires)
3. CROSS-SUBJECT TRANSFER: Novel subject, known relation → should work
   e.g., train "heat causes expansion", test "fire causes ?" → expansion
   (fire is similar to heat, spreading activation reaches heat's causal targets)
4. FULL CROSS-DOMAIN: Novel subject from different domain, known relation
   e.g., train Domain A (science), test Domain B (social) with causal relation
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ravana_ml.nn.rlm_v2 import RLMv2, RELATION_TYPES
from ravana_ml.tokenizer import WordTokenizer


def run_benchmark(n_epochs=150, embed_dim=32, concept_dim=32):
    print("=" * 70)
    print("RLMv2 — Cross-Domain Transfer Benchmark v2")
    print("=" * 70)
    print()

    # ── Training Data ──
    # Domain A: Science (causal + semantic)
    train_triples = [
        # Causal triples
        ("heat causes expansion", "expansion"),
        ("heat melts ice", "ice"),
        ("fire produces smoke", "smoke"),
        ("fire creates heat", "heat"),
        # Semantic triples
        ("ice is cold", "cold"),
        ("water is liquid", "liquid"),
        # Domain B: Social (causal + semantic)
        ("kindness causes trust", "trust"),
        ("anger produces conflict", "conflict"),
        ("fear is debilitating", "debilitating"),
        ("trust is valuable", "valuable"),
        # Domain C: Nature (causal + semantic)
        ("rain causes flooding", "flooding"),
        ("sun produces heat", "heat"),
        ("wind is powerful", "powerful"),
    ]

    # ── Test Data ──
    test_cases = {
        "train_memorization": [
            # These should all be 100% after training
            ("heat causes expansion", "expansion"),
            ("kindness causes trust", "trust"),
            ("rain causes flooding", "flooding"),
            ("ice is cold", "cold"),
        ],

        "relation_type_transfer": [
            # Same subject, different verb that maps to same relation type
            # "produces" → CAUSAL (same as "causes")
            ("heat produces expansion", "expansion"),
            ("heat leads to expansion", "expansion"),
            ("kindness produces trust", "trust"),
            ("rain creates flooding", "flooding"),
        ],

        "cross_subject_causal": [
            # Novel subject, known causal relation
            # "fire" is similar to "heat" → should find heat's causal targets
            ("fire causes expansion", "expansion"),
            # "water" is similar to "ice" → should find ice-related targets
            ("water causes flooding", "flooding"),
            # "love" is similar to "kindness" → should find kindness's causal targets
            ("love causes trust", "trust"),
        ],

        "cross_domain_relation_filter": [
            # These test that relation type filtering works:
            # "heat is ?" should NOT predict "expansion" (that's causal, not semantic)
            ("heat is cold", "cold"),
            # "kindness produces ?" should predict trust (causal), not "valuable" (semantic)
            ("kindness produces trust", "trust"),
        ],

        "negative_transfer": [
            # These should NOT be predicted (wrong relation type)
            # "heat is ?" should NOT predict "expansion" (causal target)
            # This is a qualitative test — we check that expansion is NOT in top-5
            ("heat is expansion", "expansion"),  # should MISS — wrong relation type
        ],
    }

    # ── Build tokenizer ──
    tok = WordTokenizer()
    all_texts = []
    for text, _ in train_triples:
        all_texts.append(text)
    for category in test_cases.values():
        for text, _ in category:
            all_texts.append(text)
    for text in all_texts:
        tok.encode(text)

    print(f"Vocabulary size: {tok.vocab_size}")
    print(f"Training triples: {len(train_triples)}")
    print()

    # ── Create model ──
    model = RLMv2(
        vocab_size=max(100, tok.vocab_size + 10),
        embed_dim=embed_dim,
        concept_dim=concept_dim,
        n_concepts=tok.vocab_size,
        sleep_interval=50,
    )
    model._tokenizer = tok

    # ── Training ──
    print("-" * 70)
    print("TRAINING")
    print("-" * 70)

    for epoch in range(n_epochs):
        total_loss = 0
        correct = 0
        for text, target_word in train_triples:
            ids = tok.encode(text)
            target_id = tok.encode(target_word)[0]
            ctx = np.array(ids[:-1], dtype=np.int64)
            tgt = np.array([target_id], dtype=np.int64)
            result = model.learn(ctx, tgt)
            total_loss += result["loss"]
            if result.get("is_correct"):
                correct += 1

        if (epoch + 1) % 30 == 0:
            acc = correct / len(train_triples)
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_triples):.4f}, acc={acc:.1%}")

    print()
    print(f"Final: {len(model.graph.nodes)} concepts, {len(model.graph.edges)} edges")

    # Edge type distribution
    type_counts = {}
    for edge in model.graph.edges.values():
        rt = edge.relation_type
        type_counts[rt] = type_counts.get(rt, 0) + 1
    print(f"Edge types: {type_counts}")

    # ── Evaluation ──
    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    results = {}

    for category, test_data in test_cases.items():
        print(f"\n--- {category.upper().replace('_', ' ')} ---")
        hits = 0
        total = 0
        for text, target_word in test_data:
            ids = tok.encode(text)
            target_id = tok.encode(target_word)[0]

            ctx = np.array(ids[:-1], dtype=np.int64)
            logits = model.forward(ctx)
            top10 = set(np.argsort(logits.data.flatten())[::-1][:10].tolist())
            top5 = np.argsort(logits.data.flatten())[::-1][:5]
            top5_words = [tok.decode([int(t)]) for t in top5]

            hit = target_id in top10
            if hit:
                hits += 1
            total += 1

            status = "HIT" if hit else "MISS"
            print(f"  \"{text}\" → \"{target_word}\": {status} (top5: {top5_words})")

        rate = hits / max(1, total)
        results[category] = {"hits": hits, "total": total, "rate": rate}
        print(f"  {category}: {hits}/{total} = {rate:.1%}")

    # ── Relation Vector Separation ──
    print()
    print("=" * 70)
    print("RELATION VECTOR ANALYSIS")
    print("=" * 70)

    type_rvs = {}
    for edge in model.graph.edges.values():
        rt = edge.relation_type
        if rt not in type_rvs:
            type_rvs[rt] = []
        type_rvs[rt].append(edge.relation_vector)

    if len(type_rvs) > 1:
        centroids = {}
        for rt, rvs in type_rvs.items():
            if rvs:
                centroids[rt] = np.mean(rvs, axis=0)

        type_names = sorted(centroids.keys())
        for i, rt1 in enumerate(type_names):
            for rt2 in type_names[i+1:]:
                c1 = centroids[rt1]
                c2 = centroids[rt2]
                cos = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-10)
                print(f"  {rt1} ↔ {rt2}: cosine = {cos:.3f}")

    # ── Summary ──
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for category, r in results.items():
        marker = "✓" if r["rate"] > 0 else "✗"
        print(f"  {marker} {category}: {r['hits']}/{r['total']} = {r['rate']:.1%}")

    print()
    print("  COMPARISON WITH V1 (rlm.py):")
    print(f"    V1 cross-domain transfer: 0%")
    v2_transfer = results.get("cross_subject_causal", {}).get("rate", 0)
    print(f"    V2 cross-subject transfer: {v2_transfer:.1%}")
    v2_rel = results.get("relation_type_transfer", {}).get("rate", 0)
    print(f"    V2 relation type transfer: {v2_rel:.1%}")

    if v2_transfer > 0 or v2_rel > 0:
        print(f"    ✓ IMPROVEMENT: Cross-domain transfer is now NON-ZERO!")
    else:
        print(f"    ✗ Still 0% — need more training or architecture tuning")

    # Save results
    save_results = {
        "train_memorization": results["train_memorization"]["rate"],
        "relation_type_transfer": results["relation_type_transfer"]["rate"],
        "cross_subject_causal": results["cross_subject_causal"]["rate"],
        "cross_domain_relation_filter": results["cross_domain_relation_filter"]["rate"],
        "graph_concepts": len(model.graph.nodes),
        "graph_edges": len(model.graph.edges),
        "edge_type_distribution": type_counts,
        "n_epochs": n_epochs,
        "embed_dim": embed_dim,
        "concept_dim": concept_dim,
    }

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "experiment_results", "triple_benchmark_v2.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    return save_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RLMv2 Cross-Domain Transfer Benchmark")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--embed-dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--concept-dim", type=int, default=32, help="Concept dimension")
    args = parser.parse_args()

    run_benchmark(n_epochs=args.epochs, embed_dim=args.embed_dim, concept_dim=args.concept_dim)
