"""
RLMv2 — Aggressive Cross-Domain Transfer Benchmark v3.1

Key fix: Use token embeddings directly as concept vectors (bypass projection).
The projection layers were the bottleneck — with 159 random embeddings,
they couldn't separate concepts. Using token embeddings directly means
the Hebbian updates operate on the same space as the logits.

Also:
- Explicit projection layer training via prediction error
- More focused training loop (repeat high-loss triples)
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ravana_ml.nn.rlm_v2 import RLMv2, RELATION_TYPES
from ravana_ml.tokenizer import WordTokenizer


def build_large_dataset():
    """Same as v3 but with 5 domains."""

    physics_train = [
        ("heat causes expansion", "expansion"),
        ("heat melts ice", "ice"),
        ("cold freezes water", "water"),
        ("fire produces smoke", "smoke"),
        ("fire creates heat", "heat"),
        ("friction produces heat", "heat"),
        ("gravity causes falling", "falling"),
        ("pressure causes rupture", "rupture"),
        ("light causes reflection", "reflection"),
        ("electricity powers motors", "motors"),
        ("ice is cold", "cold"),
        ("water is liquid", "liquid"),
        ("steel is strong", "strong"),
        ("glass is transparent", "transparent"),
        ("gold is valuable", "valuable"),
        ("water has density", "density"),
        ("light has wavelength", "wavelength"),
        ("atoms have electrons", "electrons"),
    ]

    social_train = [
        ("kindness causes trust", "trust"),
        ("kindness creates friendship", "friendship"),
        ("anger produces conflict", "conflict"),
        ("anger causes isolation", "isolation"),
        ("fear produces avoidance", "avoidance"),
        ("fear causes panic", "panic"),
        ("love creates bonds", "bonds"),
        ("jealousy destroys relationships", "relationships"),
        ("empathy builds connection", "connection"),
        ("greed causes corruption", "corruption"),
        ("trust is valuable", "valuable"),
        ("fear is debilitating", "debilitating"),
        ("love is powerful", "powerful"),
        ("anger is destructive", "destructive"),
        ("people have emotions", "emotions"),
        ("friendship has value", "value"),
    ]

    nature_train = [
        ("rain causes flooding", "flooding"),
        ("rain creates mud", "mud"),
        ("sun produces heat", "heat"),
        ("sun causes growth", "growth"),
        ("wind produces waves", "waves"),
        ("wind causes erosion", "erosion"),
        ("drought causes famine", "famine"),
        ("frost kills plants", "plants"),
        ("lightning starts fires", "fires"),
        ("snow causes avalanches", "avalanches"),
        ("wind is powerful", "powerful"),
        ("rain is refreshing", "refreshing"),
        ("desert is arid", "arid"),
        ("spring follows winter", "winter"),
        ("day follows night", "night"),
    ]

    biology_train = [
        ("exercise strengthens muscles", "muscles"),
        ("exercise causes sweating", "sweating"),
        ("sleep restores energy", "energy"),
        ("food provides nutrition", "nutrition"),
        ("dehydration causes cramps", "cramps"),
        ("viruses cause illness", "illness"),
        ("medicine treats disease", "disease"),
        ("stress weakens immunity", "immunity"),
        ("oxygen sustains life", "life"),
        ("poison destroys cells", "cells"),
        ("blood is essential", "essential"),
        ("bones are rigid", "rigid"),
        ("muscles are flexible", "flexible"),
        ("the brain has neurons", "neurons"),
        ("the heart has chambers", "chambers"),
    ]

    tech_train = [
        ("code creates software", "software"),
        ("bugs cause crashes", "crashes"),
        ("updates fix vulnerabilities", "vulnerabilities"),
        ("encryption protects data", "data"),
        ("viruses corrupt files", "files"),
        ("bandwidth affects speed", "speed"),
        ("algorithms process data", "data"),
        ("errors cause failures", "failures"),
        ("python is popular", "popular"),
        ("data is digital", "digital"),
        ("code is abstract", "abstract"),
        ("computers have memory", "memory"),
        ("networks have bandwidth", "bandwidth"),
    ]

    all_train = physics_train + social_train + nature_train + biology_train + tech_train

    tests = {
        "train_memorization": [
            ("heat causes expansion", "expansion"),
            ("kindness causes trust", "trust"),
            ("rain causes flooding", "flooding"),
            ("exercise strengthens muscles", "muscles"),
            ("code creates software", "software"),
            ("ice is cold", "cold"),
            ("trust is valuable", "valuable"),
            ("wind is powerful", "powerful"),
            ("blood is essential", "essential"),
            ("python is popular", "popular"),
        ],
        "relation_type_transfer": [
            ("heat produces expansion", "expansion"),
            ("heat leads to expansion", "expansion"),
            ("heat generates expansion", "expansion"),
            ("kindness produces trust", "trust"),
            ("kindness generates friendship", "friendship"),
            ("rain creates flooding", "flooding"),
            ("rain leads to flooding", "flooding"),
            ("exercise produces strength", "strength"),
            ("code generates software", "software"),
            ("anger leads to conflict", "conflict"),
            ("ice appears cold", "cold"),
            ("steel seems strong", "strong"),
        ],
        "cross_subject_same_domain": [
            ("fire causes expansion", "expansion"),
            ("friction melts ice", "ice"),
            ("cold produces contraction", "contraction"),
            ("love causes trust", "trust"),
            ("empathy creates friendship", "friendship"),
            ("storm causes flooding", "flooding"),
            ("storm creates mud", "mud"),
            ("running causes sweating", "sweating"),
            ("running strengthens muscles", "muscles"),
        ],
        "cross_domain_causal": [
            ("anger causes expansion", "expansion"),
            ("love produces heat", "heat"),
            ("heat causes trust", "trust"),
            ("fire produces friendship", "friendship"),
            ("kindness causes flooding", "flooding"),
            ("rain produces conflict", "conflict"),
            ("code causes illness", "illness"),
            ("exercise produces crashes", "crashes"),
        ],
        "relation_filter": [
            ("heat is cold", "cold"),
            ("kindness produces trust", "trust"),
            ("ice causes flooding", "flooding"),
        ],
    }

    return all_train, tests


def run_benchmark(n_epochs=1000, embed_dim=64, concept_dim=64):
    print("=" * 70)
    print("RLMv2 — Aggressive Cross-Domain Benchmark v3.1")
    print("=" * 70)
    print()

    train_triples, test_cases = build_large_dataset()

    # Build tokenizer
    tok = WordTokenizer()
    all_texts = []
    for text, _ in train_triples:
        all_texts.append(text)
    for category in test_cases.values():
        for text, _ in category:
            all_texts.append(text)
    for text in all_texts:
        tok.encode(text)

    print(f"Vocabulary: {tok.vocab_size} words")
    print(f"Training: {len(train_triples)} triples across 5 domains")
    print(f"Embed dim: {embed_dim}, Concept dim: {concept_dim}")
    print(f"Epochs: {n_epochs}")
    print()

    # Create model
    model = RLMv2(
        vocab_size=max(200, tok.vocab_size + 20),
        embed_dim=embed_dim,
        concept_dim=concept_dim,
        n_concepts=tok.vocab_size,
        sleep_interval=100,
    )
    model._tokenizer = tok

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Intensive co-occurrence + projection training
    # ═══════════════════════════════════════════════════════════════
    print("-" * 70)
    print("PHASE 1: Intensive Training (co-occurrence + projection)")
    print("-" * 70)

    # Train with repeated exposure to build strong concept embeddings
    for epoch in range(200):
        total_loss = 0
        correct = 0
        indices = np.random.permutation(len(train_triples))
        for idx in indices:
            text, target_word = train_triples[idx]
            ids = tok.encode(text)
            target_id = tok.encode(target_word)[0]
            ctx = np.array(ids[:-1], dtype=np.int64)
            tgt = np.array([target_id], dtype=np.int64)
            result = model.learn(ctx, tgt)
            total_loss += result["loss"]
            if result.get("is_correct"):
                correct += 1

        if (epoch + 1) % 50 == 0:
            acc = correct / len(train_triples)
            print(f"  Phase 1 Epoch {epoch+1}: loss={total_loss/len(train_triples):.4f}, acc={acc:.1%}")

    print(f"  After Phase 1: {len(model.graph.nodes)} concepts, {len(model.graph.edges)} edges")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Focused training on hard triples
    # ═══════════════════════════════════════════════════════════════
    print()
    print("-" * 70)
    print("PHASE 2: Focused Training (hard triples)")
    print("-" * 70)

    # Identify hard triples (not memorized yet) and train them more
    for epoch in range(n_epochs):
        total_loss = 0
        correct = 0

        # Train all triples
        indices = np.random.permutation(len(train_triples))
        for idx in indices:
            text, target_word = train_triples[idx]
            ids = tok.encode(text)
            target_id = tok.encode(target_word)[0]
            ctx = np.array(ids[:-1], dtype=np.int64)
            tgt = np.array([target_id], dtype=np.int64)
            result = model.learn(ctx, tgt)
            total_loss += result["loss"]
            if result.get("is_correct"):
                correct += 1

        # Every 200 epochs, do extra training on hard triples
        if (epoch + 1) % 200 == 0:
            acc = correct / len(train_triples)
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_triples):.4f}, acc={acc:.1%}")

            # Find hard triples
            hard_triples = []
            for text, target_word in train_triples:
                ids = tok.encode(text)
                target_id = tok.encode(target_word)[0]
                ctx = np.array(ids[:-1], dtype=np.int64)
                logits = model.forward(ctx)
                flat_logits = logits.data.flatten()
                top10 = set(np.argsort(flat_logits)[::-1][:10].tolist())
                if target_id not in top10:
                    hard_triples.append((text, target_word))

            if hard_triples:
                print(f"  Hard triples: {len(hard_triples)}/{len(train_triples)}")
                # Extra training on hard triples
                for _ in range(100):
                    for text, target_word in hard_triples:
                        ids = tok.encode(text)
                        target_id = tok.encode(target_word)[0]
                        ctx = np.array(ids[:-1], dtype=np.int64)
                        tgt = np.array([target_id], dtype=np.int64)
                        model.learn(ctx, tgt)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: EVALUATION
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    # Edge type distribution
    type_counts = {}
    for edge in model.graph.edges.values():
        rt = edge.relation_type
        type_counts[rt] = type_counts.get(rt, 0) + 1
    print(f"\n  Graph: {len(model.graph.nodes)} concepts, {len(model.graph.edges)} edges")
    print(f"  Edge types: {type_counts}")

    results = {}

    for category, test_data in test_cases.items():
        print(f"\n--- {category.upper().replace('_', ' ')} ---")
        hits_1 = 0
        hits_5 = 0
        hits_10 = 0
        total = 0

        for text, target_word in test_data:
            ids = tok.encode(text)
            target_id = tok.encode(target_word)[0]

            ctx = np.array(ids[:-1], dtype=np.int64)
            logits = model.forward(ctx)
            flat_logits = logits.data.flatten()

            top1 = int(np.argmax(flat_logits))
            top5 = set(np.argsort(flat_logits)[::-1][:5].tolist())
            top10 = set(np.argsort(flat_logits)[::-1][:10].tolist())
            top5_words = [tok.decode([int(t)]) for t in np.argsort(flat_logits)[::-1][:5]]

            if target_id == top1:
                hits_1 += 1
            if target_id in top5:
                hits_5 += 1
            if target_id in top10:
                hits_10 += 1
            total += 1

            hit = target_id in top10
            status = "✓" if hit else "✗"
            print(f"  {status} \"{text}\" → \"{target_word}\": top5={top5_words}")

        r1 = hits_1 / max(1, total)
        r5 = hits_5 / max(1, total)
        r10 = hits_10 / max(1, total)
        results[category] = {"top1": r1, "top5": r5, "top10": r10, "total": total}
        print(f"  {category}: top-1={r1:.1%}, top-5={r5:.1%}, top-10={r10:.1%}")

    # ═══════════════════════════════════════════════════════════════
    # RELATION VECTOR ANALYSIS
    # ═══════════════════════════════════════════════════════════════
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

    centroids = {}
    for rt, rvs in type_rvs.items():
        if rvs:
            centroids[rt] = np.mean(rvs, axis=0)

    type_names = sorted(centroids.keys())
    print(f"  Types with edges: {type_names}")
    for i, rt1 in enumerate(type_names):
        for rt2 in type_names[i+1:]:
            c1 = centroids[rt1]
            c2 = centroids[rt2]
            cos = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-10)
            print(f"  {rt1} ↔ {rt2}: cosine = {cos:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for category, r in results.items():
        marker = "✓" if r["top10"] > 0.5 else ("△" if r["top10"] > 0 else "✗")
        print(f"  {marker} {category}: top-1={r['top1']:.1%}, top-5={r['top5']:.1%}, top-10={r['top10']:.1%} (n={r['total']})")

    total_correct = sum(r["top10"] * r["total"] for r in results.values())
    total_probes = sum(r["total"] for r in results.values())
    print(f"\n  OVERALL top-10: {total_correct:.0f}/{total_probes} = {total_correct/total_probes:.1%}")

    train_mem = results.get("train_memorization", {}).get("top10", 0)
    rel_transfer = results.get("relation_type_transfer", {}).get("top10", 0)
    cross_subj = results.get("cross_subject_same_domain", {}).get("top10", 0)
    cross_domain = results.get("cross_domain_causal", {}).get("top10", 0)

    print(f"\n  FOR PAPER:")
    print(f"    Train memorization (top-10):    {train_mem:.1%}")
    print(f"    Relation type transfer (top-10): {rel_transfer:.1%}")
    print(f"    Cross-subject same-domain:       {cross_subj:.1%}")
    print(f"    Cross-domain causal:             {cross_domain:.1%}")

    # Save
    save_data = {
        "results": {k: {m: round(v, 4) for m, v in r.items()} for k, r in results.items()},
        "overall_top10": round(total_correct / total_probes, 4),
        "graph_concepts": len(model.graph.nodes),
        "graph_edges": len(model.graph.edges),
        "edge_types": type_counts,
        "config": {
            "n_epochs": n_epochs,
            "embed_dim": embed_dim,
            "concept_dim": concept_dim,
            "n_train": len(train_triples),
        },
    }
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "experiment_results", "triple_benchmark_v3.1.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results: {results_path}")

    return save_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--concept-dim", type=int, default=64)
    args = parser.parse_args()

    run_benchmark(n_epochs=args.epochs, embed_dim=args.embed_dim, concept_dim=args.concept_dim)
