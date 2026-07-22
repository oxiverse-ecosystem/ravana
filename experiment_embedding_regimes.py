"""
Embedding Regime Experiment
============================
Tests whether RLMv2's graph topology can USE semantic structure when it exists.

Three regimes + nearest-neighbor baseline:
  A. Random embeddings (current state — expected: 0% transfer)
  B. Hand-crafted semantic embeddings (2D — proves topology works)
  C. Co-occurrence learned embeddings (data-driven — realistic)
  D. Nearest-neighbor baseline (no graph — isolates graph contribution)

For every transfer query, we compare:
  - Graph prediction (activation spreading + edge traversal)
  - Nearest-neighbor retrieval (cosine similarity in embedding space)

If graph ≈ NN, the graph isn't adding reasoning.
If graph > NN, the graph is genuinely composing knowledge.
"""

import sys
import io
import numpy as np
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer


def build_facts():
    """Structured facts with clear semantic categories."""
    facts = [
        # Animals → attributes
        ("cat", "has", "tail"),
        ("cat", "has", "whiskers"),
        ("dog", "has", "tail"),
        ("dog", "has", "nose"),
        ("bird", "has", "wing"),
        ("bird", "has", "beak"),
        ("fish", "has", "fin"),
        ("fish", "has", "scale"),
        ("horse", "has", "tail"),
        ("horse", "has", "hoof"),
        # Animals → actions
        ("cat", "can", "purr"),
        ("dog", "can", "bark"),
        ("bird", "can", "fly"),
        ("fish", "can", "swim"),
        ("horse", "can", "gallop"),
        # Science causal
        ("heat", "causes", "expansion"),
        ("cold", "causes", "contraction"),
        ("friction", "causes", "wear"),
        ("pressure", "causes", "deformation"),
        ("voltage", "causes", "current"),
        # Social causal
        ("kindness", "causes", "trust"),
        ("anger", "causes", "conflict"),
        ("patience", "causes", "understanding"),
        ("honesty", "causes", "respect"),
        ("generosity", "causes", "gratitude"),
    ]
    return facts


def build_transfer_tests():
    """Transfer queries with expected answers."""
    tests = [
        # Same-relation novel subject (the critical test)
        {"query": "tiger has", "expected": "tail", "category": "animal_attr",
         "reasoning": "tiger≈cat,dog,horse → has tail"},
        {"query": "eagle has", "expected": "wing", "category": "animal_attr",
         "reasoning": "eagle≈bird → has wing"},
        {"query": "shark has", "expected": "fin", "category": "animal_attr",
         "reasoning": "shark≈fish → has fin"},
        {"query": "wolf has", "expected": "tail", "category": "animal_attr",
         "reasoning": "wolf≈dog → has tail"},
        {"query": "parrot can", "expected": "fly", "category": "animal_action",
         "reasoning": "parrot≈bird → can fly"},
        {"query": "whale can", "expected": "swim", "category": "animal_action",
         "reasoning": "whale≈fish → can swim"},
        # Cross-domain causal
        {"query": "warmth causes", "expected": "trust", "category": "cross_causal",
         "reasoning": "warmth≈kindness → causes trust"},
        {"query": "rudeness causes", "expected": "conflict", "category": "cross_causal",
         "reasoning": "rudeness≈anger → causes conflict"},
        {"query": "loyalty causes", "expected": "respect", "category": "cross_causal",
         "reasoning": "loyalty≈honesty → causes respect"},
        {"query": "voltage causes", "expected": "current", "category": "in_domain_causal",
         "reasoning": "voltage is in training, just checking memorization"},
    ]
    return tests


# ─── Regime A: Random Embeddings ───────────────────────────────────────────

def build_word_tokenizer(facts, transfer_tests):
    """Build vocab from all texts."""
    tok = WordTokenizer()
    for s, r, o in facts:
        tok.encode(f"{s} {r} {o}")
    for t in transfer_tests:
        tok.encode(t["query"])
    return tok


def run_regime_a(facts, transfer_tests):
    """Baseline: random embeddings. Expected: ~0% transfer."""
    print("\n" + "="*70)
    print("REGIME A: RANDOM EMBEDDINGS")
    print("="*70)

    tok = build_word_tokenizer(facts, transfer_tests)

    model = RLMv2(
        vocab_size=tok.vocab_size,
        embed_dim=32,
        concept_dim=32,
        n_concepts=500,
        sleep_interval=200,
        gate_concept_creation=False,
    )
    model._tokenizer = tok

    # Train
    for epoch in range(3):
        for s, r, o in facts:
            text = f"{s} {r} {o}"
            ids = tok.encode(text)
            if len(ids) < 2:
                continue
            ctx = np.array([ids[:-1]], dtype=np.int64)
            tgt = np.array([[ids[-1]]], dtype=np.int64)
            model.learn(ctx, tgt)

    # Evaluate
    results = evaluate_transfer(model, tok, transfer_tests, "A")
    nn_results = evaluate_nn_baseline(model, tok, facts, transfer_tests)
    return results, nn_results


# ─── Regime B: Hand-Crafted Semantic Embeddings ────────────────────────────

def run_regime_b(facts, transfer_tests):
    """Hand-crafted 2D embeddings encoding semantic categories."""
    print("\n" + "="*70)
    print("REGIME B: HAND-CRAFTED SEMANTIC EMBEDDINGS")
    print("="*70)

    tok = build_word_tokenizer(facts, transfer_tests)

    model = RLMv2(
        vocab_size=tok.vocab_size,
        embed_dim=32,
        concept_dim=32,
        n_concepts=500,
        sleep_interval=200,
        gate_concept_creation=False,
    )
    model._tokenizer = tok

    # Inject hand-crafted embeddings
    # Dimension 0: animal vs object vs abstract
    # Dimension 1: size / intensity
    # Dimensions 2-31: noise (to fill 32D)
    semantic_vectors = {
        # Animals (dim0=1.0, dim1=varies by size)
        "cat":     [1.0, 0.4, 0.8, 0.2],
        "dog":     [1.0, 0.5, 0.7, 0.3],
        "bird":    [1.0, 0.2, 0.6, 0.1],
        "fish":    [1.0, 0.3, 0.5, 0.4],
        "horse":   [1.0, 0.8, 0.9, 0.1],
        "tiger":   [1.0, 0.6, 0.85, 0.25],  # close to cat
        "eagle":   [1.0, 0.3, 0.65, 0.15],  # close to bird
        "shark":   [1.0, 0.7, 0.55, 0.45],  # close to fish
        "wolf":    [1.0, 0.55, 0.72, 0.32], # close to dog
        "parrot":  [1.0, 0.15, 0.62, 0.12], # close to bird
        "whale":   [1.0, 0.9, 0.52, 0.42],  # close to fish (aquatic)

        # Attributes (dim0=0.0, dim1=body part type)
        "tail":    [0.0, 0.8, 0.3, 0.1],
        "whiskers":[0.0, 0.3, 0.35, 0.15],
        "nose":    [0.0, 0.4, 0.4, 0.2],
        "wing":    [0.0, 0.6, 0.2, 0.7],
        "beak":    [0.0, 0.5, 0.25, 0.65],
        "fin":     [0.0, 0.7, 0.15, 0.8],
        "scale":   [0.0, 0.2, 0.1, 0.9],
        "hoof":    [0.0, 0.9, 0.45, 0.05],

        # Actions (dim0=-0.5, dim1=type)
        "purr":    [-0.5, 0.3, 0.1, 0.2],
        "bark":    [-0.5, 0.5, 0.15, 0.25],
        "fly":     [-0.5, 0.8, 0.2, 0.7],
        "swim":    [-0.5, 0.6, 0.1, 0.8],
        "gallop":  [-0.5, 0.9, 0.3, 0.1],

        # Relations
        "has":     [0.5, 0.0, 0.0, 0.0],
        "can":     [0.5, 0.0, 0.0, 0.0],
        "causes":  [0.5, 0.0, 0.0, 0.0],

        # Science (dim0=-1.0, dim1=intensity)
        "heat":        [-1.0, 0.8, 0.9, 0.1],
        "cold":        [-1.0, 0.2, 0.85, 0.15],
        "friction":    [-1.0, 0.6, 0.7, 0.3],
        "pressure":    [-1.0, 0.7, 0.75, 0.25],
        "voltage":     [-1.0, 0.5, 0.8, 0.2],
        "expansion":   [-1.0, 0.3, 0.6, 0.4],
        "contraction": [-1.0, 0.2, 0.55, 0.45],
        "wear":        [-1.0, 0.4, 0.5, 0.5],
        "deformation": [-1.0, 0.5, 0.45, 0.55],
        "current":     [-1.0, 0.6, 0.4, 0.6],

        # Social (dim0=-1.0, dim1=intensity — close to science for cross-domain)
        "kindness":     [-1.0, 0.75, 0.88, 0.12],  # close to heat
        "anger":        [-1.0, 0.65, 0.82, 0.28],  # close to friction
        "patience":     [-1.0, 0.55, 0.73, 0.33],  # close to pressure
        "honesty":      [-1.0, 0.45, 0.78, 0.22],  # close to voltage
        "generosity":   [-1.0, 0.7, 0.65, 0.35],   # close to heat
        "trust":        [-1.0, 0.35, 0.58, 0.42],   # close to expansion
        "conflict":     [-1.0, 0.25, 0.53, 0.47],   # close to contraction
        "understanding":[-1.0, 0.45, 0.48, 0.52],   # close to wear
        "respect":      [-1.0, 0.55, 0.43, 0.57],   # close to deformation
        "gratitude":    [-1.0, 0.65, 0.38, 0.62],   # close to current

        # Novel transfer words
        "warmth":   [-1.0, 0.73, 0.87, 0.13],  # very close to kindness
        "rudeness": [-1.0, 0.63, 0.83, 0.27],  # very close to anger
        "loyalty":  [-1.0, 0.47, 0.77, 0.23],  # very close to honesty
    }

    # Inject into embedding table
    for word, vec4 in semantic_vectors.items():
        tid = tok.word_to_id.get(word)
        if tid is None:
            continue
        # Expand 4D to 32D by repeating + small noise
        full_vec = np.zeros(32, dtype=np.float32)
        for i in range(32):
            full_vec[i] = vec4[i % 4] + np.random.randn() * 0.01
        full_vec /= np.linalg.norm(full_vec)
        model.token_embed.weight.data[tid] = full_vec

    # Train
    for epoch in range(3):
        for s, r, o in facts:
            text = f"{s} {r} {o}"
            ids = tok.encode(text)
            if len(ids) < 2:
                continue
            ctx = np.array([ids[:-1]], dtype=np.int64)
            tgt = np.array([[ids[-1]]], dtype=np.int64)
            model.learn(ctx, tgt)

    # Evaluate
    results = evaluate_transfer(model, tok, transfer_tests, "B")
    nn_results = evaluate_nn_baseline(model, tok, facts, transfer_tests)
    return results, nn_results


# ─── Regime C: Co-occurrence Learned Embeddings ────────────────────────────

def run_regime_c(facts, transfer_tests):
    """Co-occurrence skip-gram style pre-training."""
    print("\n" + "="*70)
    print("REGIME C: CO-OCCURRENCE LEARNED EMBEDDINGS")
    print("="*70)

    tok = build_word_tokenizer(facts, transfer_tests)

    model = RLMv2(
        vocab_size=tok.vocab_size,
        embed_dim=32,
        concept_dim=32,
        n_concepts=500,
        sleep_interval=200,
        gate_concept_creation=False,
    )
    model._tokenizer = tok

    # Pre-train embeddings via co-occurrence
    # Build co-occurrence matrix from facts
    word_ids = {}
    for s, r, o in facts:
        for w in [s, r, o]:
            tid = tok.word_to_id.get(w)
            if tid is not None:
                word_ids[w] = tid

    # Co-occurrence counting (window=2)
    cooc = defaultdict(lambda: defaultdict(int))
    for s, r, o in facts:
        words = [s, r, o]
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                if i != j:
                    t1 = tok.word_to_id.get(w1)
                    t2 = tok.word_to_id.get(w2)
                    if t1 is not None and t2 is not None:
                        cooc[t1][t2] += 1

    # Skip-gram training: update embeddings based on co-occurrence
    lr = 0.1
    for epoch in range(50):
        total_loss = 0
        for t1, neighbors in cooc.items():
            for t2, count in neighbors.items():
                # Pull co-occurring words closer
                v1 = model.token_embed.weight.data[t1]
                v2 = model.token_embed.weight.data[t2]
                diff = v1 - v2
                loss = np.dot(diff, diff) * count
                total_loss += loss
                grad = 2 * diff * count * lr / (epoch + 1)
                model.token_embed.weight.data[t1] -= grad
                model.token_embed.weight.data[t2] += grad

        # Normalize
        norms = np.linalg.norm(model.token_embed.weight.data, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        model.token_embed.weight.data /= norms

    # Show learned similarities
    print("\nLearned embedding similarities (top pairs):")
    embeds = model.token_embed.weight.data
    sims = []
    word_list = list(word_ids.items())
    for i, (w1, t1) in enumerate(word_list):
        for j, (w2, t2) in enumerate(word_list):
            if i < j:
                cos = float(np.dot(embeds[t1], embeds[t2]))
                sims.append((cos, w1, w2))
    sims.sort(reverse=True)
    for cos, w1, w2 in sims[:15]:
        print(f"  {w1:12s} <-> {w2:12s}: {cos:.3f}")

    # Train
    for epoch in range(3):
        for s, r, o in facts:
            text = f"{s} {r} {o}"
            ids = tok.encode(text)
            if len(ids) < 2:
                continue
            ctx = np.array([ids[:-1]], dtype=np.int64)
            tgt = np.array([[ids[-1]]], dtype=np.int64)
            model.learn(ctx, tgt)

    # Evaluate
    results = evaluate_transfer(model, tok, transfer_tests, "C")
    nn_results = evaluate_nn_baseline(model, tok, facts, transfer_tests)
    return results, nn_results


# ─── Evaluation Helpers ────────────────────────────────────────────────────

def evaluate_transfer(model, tok, tests, regime_label):
    """Evaluate transfer using graph-based prediction."""
    print(f"\n--- Regime {regime_label}: Graph-Based Transfer ---")
    results = []
    for test in tests:
        query = test["query"]
        expected = test["expected"]
        category = test["category"]

        ids = tok.encode(query)
        if len(ids) == 0:
            results.append({"test": test, "hit": False, "top5": []})
            continue

        ctx = np.array([ids], dtype=np.int64)
        logits = np.asarray(model.forward(ctx).data).flatten()
        top5_ids = list(np.argsort(logits)[::-1][:5])
        top5_words = [tok.decode([tid]) for tid in top5_ids]

        hit = expected in top5_words
        exp_tid = tok.word_to_id.get(expected, -1)
        top10 = set(np.argsort(logits)[::-1][:10])
        hit10 = exp_tid in top10

        marker = "✓" if hit10 else "✗"
        print(f"  {marker} {query:20s} → expected: {expected:12s} | top5: {top5_words}")

        results.append({
            "test": test,
            "hit10": hit10,
            "hit5": hit,
            "top5": top5_words,
            "top1": top5_words[0] if top5_words else "",
        })

    hit10_count = sum(1 for r in results if r["hit10"])
    hit5_count = sum(1 for r in results if r["hit5"])
    print(f"\n  Regime {regime_label} Graph: {hit10_count}/{len(results)} top-10, "
          f"{hit5_count}/{len(results)} top-5")
    return results


def evaluate_nn_baseline(model, tok, facts, tests):
    """Nearest-neighbor baseline: no graph, just embedding similarity."""
    print(f"\n--- Nearest-Neighbor Baseline (no graph) ---")

    # Build a simple mapping: for each relation, map subject → object
    # Then for novel subjects, find nearest trained subject and return its object
    rel_to_pairs = defaultdict(list)
    for s, r, o in facts:
        rel_to_pairs[r].append((s, o))

    embeds = model.token_embed.weight.data
    results = []

    for test in tests:
        query = test["query"]
        expected = test["expected"]

        # Parse query: "tiger has" → subject=tiger, relation=has
        parts = query.split()
        if len(parts) < 2:
            results.append({"test": test, "hit10": False, "hit5": False})
            continue
        subj_word = parts[0]
        rel_word = parts[1]

        subj_tid = tok.word_to_id.get(subj_word)
        if subj_tid is None:
            results.append({"test": test, "hit10": False, "hit5": False})
            continue

        subj_vec = embeds[subj_tid]

        # Find nearest trained subjects for this relation
        pairs = rel_to_pairs.get(rel_word, [])
        if not pairs:
            results.append({"test": test, "hit10": False, "hit5": False})
            continue

        scored = []
        for s, o in pairs:
            s_tid = tok.word_to_id.get(s)
            o_tid = tok.word_to_id.get(o)
            if s_tid is None or o_tid is None:
                continue
            sim = float(np.dot(subj_vec, embeds[s_tid]))
            scored.append((sim, o, o_tid))

        scored.sort(reverse=True)

        # Top-1 NN prediction
        if scored:
            nn_top1 = scored[0][1]
            # Top-5 NN predictions (unique objects)
            seen = set()
            nn_top5 = []
            for sim, o, o_tid in scored:
                if o not in seen:
                    seen.add(o)
                    nn_top5.append(o)
                if len(nn_top5) >= 5:
                    break
        else:
            nn_top1 = "?"
            nn_top5 = []

        exp_tid = tok.word_to_id.get(expected, -1)
        nn_top5_tids = [tok.word_to_id.get(w, -1) for w in nn_top5]
        hit5 = expected in nn_top5
        hit10 = expected in nn_top5  # NN only returns up to 5 unique objects

        marker = "✓" if hit5 else "✗"
        neighbors_str = ", ".join(f"{s}({sim:.2f})" for sim, s, _ in scored[:3])
        print(f"  {marker} {test['query']:20s} → expected: {expected:12s} | "
              f"NN top1: {nn_top1:12s} | neighbors: {neighbors_str}")

        results.append({
            "test": test,
            "hit10": hit10,
            "hit5": hit5,
            "nn_top1": nn_top1,
            "nn_top5": nn_top5,
        })

    hit5_count = sum(1 for r in results if r["hit5"])
    print(f"\n  NN Baseline: {hit5_count}/{len(results)} top-5")
    return results


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("EMBEDDING REGIME EXPERIMENT")
    print("Testing whether graph topology can USE semantic structure")
    print("="*70)

    facts = build_facts()
    transfer_tests = build_transfer_tests()

    print(f"\nTraining facts: {len(facts)}")
    print(f"Transfer tests: {len(transfer_tests)}")

    # Run all three regimes
    a_graph, a_nn = run_regime_a(facts, transfer_tests)
    b_graph, b_nn = run_regime_b(facts, transfer_tests)
    c_graph, c_nn = run_regime_c(facts, transfer_tests)

    # ─── Summary ────────────────────────────────────────────────────────────
    print()
    print('='*70)
    print('SUMMARY')
    print('='*70)

    def count_hits(results):
        h5 = sum(1 for r in results if r.get('hit5'))
        h10 = sum(1 for r in results if r.get('hit10'))
        n = len(results) if results else 1
        return h5, h10, n

    for label, g, nn in [('A (random)', a_graph, a_nn),
                          ('B (hand-crafted)', b_graph, b_nn),
                          ('C (co-occurrence)', c_graph, c_nn)]:
        g5, g10, n = count_hits(g)
        n5, _, _ = count_hits(nn)
        print(f'  Regime {label}:')
        print(f'    Graph:  {g5}/{n} top-5, {g10}/{n} top-10')
        print(f'    NN:     {n5}/{n} top-5')
        if g5 > n5:
            print(f'    >> Graph BEATS NN by {g5 - n5} — topology adds reasoning!')
        elif g5 == n5:
            print(f'    >> Graph == NN — graph may still add explanation depth')
        else:
            print(f'    >> Graph < NN — embeddings alone are stronger')

    print()
    print('If graph > NN, the graph is genuinely composing knowledge.')
    print('If graph ≈ NN, the graph is not adding reasoning beyond retrieval.')


if __name__ == '__main__':
    main()
