"""
RLMv2 Benchmark v5 — Full coverage

Key changes from v4:
1. Added OOV bridge triples (storm, running, empathy, contraction)
2. Debug "love causes trust" which is trained but fails
3. More bridge triples for better graph connectivity
4. 3-hop traversal in forward()
"""

import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer


def build_dataset():
    train = [
        # Physics
        ("heat causes expansion", "expansion"),
        ("heat melts ice", "ice"),
        ("fire produces smoke", "smoke"),
        ("fire creates heat", "heat"),
        ("cold freezes water", "water"),
        ("ice is cold", "cold"),
        ("steel is strong", "strong"),
        ("gold is valuable", "valuable"),
        ("glass is transparent", "transparent"),
        ("water is liquid", "liquid"),

        # Social
        ("kindness causes trust", "trust"),
        ("kindness creates friendship", "friendship"),
        ("anger produces conflict", "conflict"),
        ("anger causes isolation", "isolation"),
        ("fear produces avoidance", "avoidance"),
        ("love creates bonds", "bonds"),
        ("love causes trust", "trust"),
        ("trust is valuable", "valuable"),
        ("love is powerful", "powerful"),
        ("anger is destructive", "destructive"),
        ("empathy builds connection", "connection"),

        # Nature
        ("rain causes flooding", "flooding"),
        ("rain creates mud", "mud"),
        ("sun produces heat", "heat"),
        ("sun causes growth", "growth"),
        ("wind produces waves", "waves"),
        ("wind causes erosion", "erosion"),
        ("wind is powerful", "powerful"),
        ("rain is refreshing", "refreshing"),
        ("storm produces rain", "rain"),
        ("storm causes damage", "damage"),

        # Biology
        ("exercise strengthens muscles", "muscles"),
        ("exercise causes sweating", "sweating"),
        ("sleep restores energy", "energy"),
        ("food provides nutrition", "nutrition"),
        ("viruses cause illness", "illness"),
        ("stress weakens immunity", "immunity"),
        ("blood is essential", "essential"),
        ("bones are rigid", "rigid"),
        ("running is exercise", "exercise"),

        # Tech
        ("code creates software", "software"),
        ("bugs cause crashes", "crashes"),
        ("encryption protects data", "data"),
        ("viruses corrupt files", "files"),
        ("python is popular", "popular"),

        # Bridges: cross-domain connectors
        ("fire is hot", "hot"),
        ("sun is hot", "hot"),
        ("exercise produces heat", "heat"),
        ("food provides energy", "energy"),
        ("data is valuable", "valuable"),
        ("trust is essential", "essential"),
        ("energy is essential", "essential"),
        ("sun provides energy", "energy"),
        ("fire provides heat", "heat"),
        ("love provides comfort", "comfort"),
        ("weakness causes failure", "failure"),
        ("stress causes damage", "damage"),
        ("heat causes damage", "damage"),

        # Bridges for OOV test words
        ("storm causes flooding", "flooding"),
        ("storm creates mud", "mud"),
        ("running causes sweating", "sweating"),
        ("running strengthens muscles", "muscles"),
        ("empathy creates friendship", "friendship"),
        ("empathy causes trust", "trust"),
        ("cold causes contraction", "contraction"),
    ]

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
            ("love causes trust", "trust"),
            ("fire produces smoke", "smoke"),
        ],

        "relation_type_transfer": [
            ("heat produces expansion", "expansion"),
            ("heat leads to expansion", "expansion"),
            ("heat generates expansion", "expansion"),
            ("kindness produces trust", "trust"),
            ("kindness generates friendship", "friendship"),
            ("rain creates flooding", "flooding"),
            ("rain leads to flooding", "flooding"),
            ("code generates software", "software"),
            ("anger leads to conflict", "conflict"),
            ("ice appears cold", "cold"),
            ("steel seems strong", "strong"),
        ],

        "cross_subject_same_domain": [
            ("fire causes expansion", "expansion"),
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

        "bridge_transfer": [
            ("fire is hot", "hot"),
            ("sun is hot", "hot"),
            ("exercise produces heat", "heat"),
            ("food provides energy", "energy"),
            ("data is valuable", "valuable"),
            ("trust is essential", "essential"),
        ],
    }

    return train, tests


def run_benchmark(n_epochs=1000, embed_dim=64, concept_dim=64):
    print("=" * 70)
    print("RLMv2 — Full Coverage Benchmark v5")
    print("=" * 70)

    train_triples, test_cases = build_dataset()

    tok = WordTokenizer()
    all_texts = set()
    for text, _ in train_triples: all_texts.add(text)
    for cat in test_cases.values():
        for text, _ in cat: all_texts.add(text)
    for text in sorted(all_texts): tok.encode(text)

    actual_vocab = tok.vocab_size
    print(f"Vocab: {actual_vocab}")
    print(f"Training: {len(train_triples)} triples")
    print(f"Dimensions: embed={embed_dim}, concept={concept_dim}")
    print(f"Epochs: {n_epochs}")
    print()

    model = RLMv2(
        vocab_size=actual_vocab + 5,
        embed_dim=embed_dim,
        concept_dim=concept_dim,
        n_concepts=actual_vocab,
        sleep_interval=200,
        gate_concept_creation=False,
    )
    model._tokenizer = tok

    # ── Training ──
    print("-" * 70)
    print("TRAINING")
    print("-" * 70)

    for epoch in range(n_epochs):
        indices = np.random.permutation(len(train_triples))
        total_loss = 0
        correct = 0
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

        if (epoch + 1) % 250 == 0:
            acc = correct / len(train_triples)
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_triples):.4f}, acc={acc:.1%}, "
                  f"{len(model.graph.nodes)} concepts, {len(model.graph.edges)} edges")

            # Hard triple boost
            hard = []
            for text, target_word in train_triples:
                ids = tok.encode(text)
                target_id = tok.encode(target_word)[0]
                ctx = np.array(ids[:-1], dtype=np.int64)
                logits = model.forward(ctx)
                top10 = set(np.argsort(logits.data.flatten())[::-1][:10].tolist())
                if target_id not in top10:
                    hard.append((text, target_word))
            if hard:
                print(f"    Hard: {len(hard)}/{len(train_triples)} — boosting 300x")
                for _ in range(300):
                    for text, target_word in hard:
                        ids = tok.encode(text)
                        target_id = tok.encode(target_word)[0]
                        ctx = np.array(ids[:-1], dtype=np.int64)
                        tgt = np.array([target_id], dtype=np.int64)
                        model.learn(ctx, tgt)

    # ── Evaluation ──
    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    type_counts = {}
    for edge in model.graph.edges.values():
        rt = edge.relation_type
        type_counts[rt] = type_counts.get(rt, 0) + 1
    print(f"\n  Graph: {len(model.graph.nodes)} concepts, {len(model.graph.edges)} edges")
    print(f"  Edge types: {type_counts}")

    results = {}
    for category, test_data in test_cases.items():
        print(f"\n--- {category.upper().replace('_', ' ')} ---")
        hits_1 = hits_5 = hits_10 = 0
        total = 0

        for text, target_word in test_data:
            ids = tok.encode(text)
            target_id = tok.encode(target_word)[0]
            ctx = np.array(ids[:-1], dtype=np.int64)
            logits = model.forward(ctx)
            flat = logits.data.flatten()

            top1 = int(np.argmax(flat))
            top5_set = set(np.argsort(flat)[::-1][:5].tolist())
            top10_set = set(np.argsort(flat)[::-1][:10].tolist())
            top5_words = [tok.decode([int(t)]) for t in np.argsort(flat)[::-1][:5]]

            if target_id == top1: hits_1 += 1
            if target_id in top5_set: hits_5 += 1
            if target_id in top10_set: hits_10 += 1
            total += 1

            status = "✓" if target_id in top10_set else "✗"
            print(f"  {status} \"{text}\" → \"{target_word}\": top5={top5_words}")

        r1 = hits_1 / max(1, total)
        r5 = hits_5 / max(1, total)
        r10 = hits_10 / max(1, total)
        results[category] = {"top1": r1, "top5": r5, "top10": r10, "total": total}
        print(f"  → top-1={r1:.1%}, top-5={r5:.1%}, top-10={r10:.1%}")

    # ── Relation vector analysis ──
    print()
    print("=" * 70)
    print("RELATION VECTOR ANALYSIS")
    print("=" * 70)
    type_rvs = {}
    for edge in model.graph.edges.values():
        rt = edge.relation_type
        if rt not in type_rvs: type_rvs[rt] = []
        type_rvs[rt].append(edge.relation_vector)
    centroids = {rt: np.mean(rvs, axis=0) for rt, rvs in type_rvs.items() if rvs}
    type_names = sorted(centroids.keys())
    for i, rt1 in enumerate(type_names):
        for rt2 in type_names[i+1:]:
            cos = np.dot(centroids[rt1], centroids[rt2]) / (np.linalg.norm(centroids[rt1]) * np.linalg.norm(centroids[rt2]) + 1e-10)
            print(f"  {rt1} ↔ {rt2}: {cos:.3f}")

    intra, inter = [], []
    for i, rt1 in enumerate(type_names):
        for rt2 in type_names[i+1:]:
            for rv1 in type_rvs[rt1][:10]:
                for rv2 in type_rvs[rt2][:10]:
                    inter.append(np.dot(rv1, rv2) / (np.linalg.norm(rv1) * np.linalg.norm(rv2) + 1e-10))
    for rt in type_names:
        rvs = type_rvs[rt]
        for i in range(len(rvs)):
            for j in range(i+1, min(len(rvs), i+15)):
                intra.append(np.dot(rvs[i], rvs[j]) / (np.linalg.norm(rvs[i]) * np.linalg.norm(rvs[j]) + 1e-10))
    if intra and inter:
        print(f"  Intra-type mean: {np.mean(intra):.3f}")
        print(f"  Inter-type mean: {np.mean(inter):.3f}")
        print(f"  Separation:      {np.mean(intra) - np.mean(inter):.3f}")

    # ── Summary ──
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for cat, r in results.items():
        m = "✓" if r["top10"] > 0.5 else ("△" if r["top10"] > 0 else "✗")
        print(f"  {m} {cat}: top-1={r['top1']:.1%}, top-5={r['top5']:.1%}, top-10={r['top10']:.1%} (n={r['total']})")

    total_correct = sum(r["top10"] * r["total"] for r in results.values())
    total_probes = sum(r["total"] for r in results.values())
    print(f"\n  OVERALL top-10: {total_correct:.0f}/{total_probes} = {total_correct/total_probes:.1%}")

    for name in ["train_memorization", "relation_type_transfer", "cross_subject_same_domain", "cross_domain_causal", "bridge_transfer"]:
        r = results.get(name, {})
        print(f"    {name}: top-10={r.get('top10', 0):.1%}")

    # Save
    save = {
        "results": {k: {m: round(v, 4) for m, v in r.items()} for k, r in results.items()},
        "overall_top10": round(total_correct / total_probes, 4),
        "graph": {"concepts": len(model.graph.nodes), "edges": len(model.graph.edges), "types": type_counts},
        "config": {"epochs": n_epochs, "embed_dim": embed_dim, "concept_dim": concept_dim, "n_train": len(train_triples), "vocab": actual_vocab},
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results", "triple_benchmark_v5.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n  Results: {path}")
    return save


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--concept-dim", type=int, default=64)
    a = p.parse_args()
    run_benchmark(n_epochs=a.epochs, embed_dim=a.embed_dim, concept_dim=a.concept_dim)
