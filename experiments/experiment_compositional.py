"""
Compositional Hybrid Benchmark
================================
Tests whether traversal and embeddings can COOPERATE on tasks
neither can solve alone.

The key test:

    warmth ≈ kindness (embedding similarity)
    kindness causes trust
    trust causes cooperation

    Query: warmth causes ?
    Expected: cooperation

    Reasoning chain:
      warmth → kindness (embedding proximity)
      kindness → trust (traversal, depth 1)
      trust → cooperation (traversal, depth 2)

    Neither subsystem alone can solve this:
      - NN: warmth has no "causes" edge, returns nearest neighbor's object
      - Traversal: warmth has no concept node, returns nothing
      - Composition: NN seeds the traversal start, traversal follows chains
"""
import sys
import io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.word_tokenizer import WordTokenizer


def inject_embeddings(model, tok):
    """Semantic embeddings with clear clusters."""
    sem = {
        # Emotion cluster (warmth, kindness, honesty are close)
        "kindness": [0.9, 0.8, 0.2],
        "warmth":   [0.85, 0.75, 0.22],  # close to kindness
        "honesty":  [0.88, 0.78, 0.18],
        "cruelty":  [-0.9, -0.8, 0.2],   # far from kindness
        "coldness": [-0.85, -0.75, 0.22],
        # Middle concepts
        "trust":        [0.5, 0.6, 0.5],
        "cooperation":  [0.4, 0.7, 0.6],
        "respect":      [0.55, 0.55, 0.45],
        "distrust":     [-0.5, -0.6, 0.5],
        "conflict":     [-0.4, -0.7, 0.6],
        "isolation":    [-0.55, -0.55, 0.45],
        # Concrete cluster
        "cat":   [0.95, 0.3, 0.8],
        "dog":   [0.9, 0.4, 0.7],
        "tiger": [0.92, 0.32, 0.82],
        "tail":  [0.1, 0.8, 0.3],
        "fur":   [0.15, 0.75, 0.35],
        # Causal chain starters
        "heat":    [0.8, 0.9, 0.1],
        "warmth2": [0.78, 0.88, 0.12],  # alias for testing
        "anger":   [-0.8, 0.9, 0.1],
        "virus":   [-0.3, 0.8, 0.2],
        "illness": [-0.3, 0.5, 0.3],
        "absence": [-0.3, 0.3, 0.4],
        "bug":     [-0.35, 0.75, 0.15],
        "crash":   [-0.35, 0.5, 0.25],
        "outage":  [-0.35, 0.3, 0.35],
    }
    dim = model.embed_dim
    for word, v3 in sem.items():
        tid = tok.word_to_id.get(word)
        if tid is None:
            continue
        full = np.zeros(dim, dtype=np.float32)
        for i in range(dim):
            full[i] = v3[i % 3] + np.random.randn() * 0.005
        full /= np.linalg.norm(full)
        model.token_embed.weight.data[tid] = full


def nn_predict(model, tok, query, facts, top_n=5):
    """Nearest-neighbor baseline: find nearest subject with same relation, return object."""
    parts = query.split()
    if len(parts) < 2:
        return []
    subj_word, rel_word = parts[0], parts[1]
    subj_tid = tok.word_to_id.get(subj_word)
    if subj_tid is None:
        return []
    embeds = model.token_embed.weight.data
    subj_vec = embeds[subj_tid]

    # Find all (subject, object) pairs with matching relation
    pairs = [(s, o) for s, r, o in facts if r == rel_word]
    scored = []
    for s, o in pairs:
        s_tid = tok.word_to_id.get(s)
        if s_tid is None:
            continue
        sim = float(np.dot(subj_vec, embeds[s_tid]))
        scored.append((sim, s, o))
    scored.sort(reverse=True)

    # Return objects from top-N nearest subjects
    seen = set()
    results = []
    for sim, s, o in scored[:top_n * 3]:  # over-fetch to deduplicate
        if o not in seen:
            seen.add(o)
            results.append((o, sim, s))
        if len(results) >= top_n:
            break
    return results


def traversal_predict(model, tok, query, max_depth=2):
    """Pure traversal from a concept node."""
    parts = query.split()
    if len(parts) < 2:
        return []
    subj_word, rel_word = parts[0], parts[1]

    subj_tid = tok.word_to_id.get(subj_word)
    if subj_tid is None:
        return []

    bindings = model.binding_map.get_concepts(subj_tid, min_confidence=0.1)
    if not bindings:
        return []
    subject_cid = bindings[0].concept_id

    # Determine relation type
    causal = {"causes", "cause", "leads", "produces", "creates"}
    possessive = {"has", "have", "contains", "includes"}
    rel_type = None
    if rel_word in causal:
        rel_type = "causal"
    elif rel_word in possessive:
        rel_type = "possessive"

    frontier = {subject_cid}
    visited = {subject_cid}
    candidates = []

    for depth in range(1, max_depth + 1):
        next_frontier = set()
        for nid in frontier:
            for tgt_id, edge in model.graph.get_outgoing(nid):
                if tgt_id in visited:
                    continue
                if rel_type and edge.relation_type != rel_type:
                    continue
                next_frontier.add(tgt_id)
                tokens = model.binding_map.get_tokens(tgt_id, 0.0)
                for b in tokens:
                    word = tok.decode([b.token_id])
                    if word not in (subj_word, rel_word) and not word.startswith("?"):
                        candidates.append((word, depth))
        visited |= next_frontier
        frontier = next_frontier

    seen = {}
    for word, depth in candidates:
        if word not in seen or depth > seen[word]:
            seen[word] = depth
    results = sorted(seen.items(), key=lambda x: (-x[1], x[0]))
    return results[:10]


def compositional_predict(model, tok, query, facts, max_depth=2, k_neighbors=3):
    """The hybrid: try direct traversal first, then NN-seeded traversal, then NN fallback.

    Priority order:
    1. Direct traversal from query subject (if concept node exists)
    2. NN-seeded traversal (embed neighbors, then traverse from them)
    3. Pure NN fallback (no concept nodes, just embedding proximity)
    """
    parts = query.split()
    if len(parts) < 2:
        return []
    subj_word, rel_word = parts[0], parts[1]

    causal = {"causes", "cause", "leads", "produces", "creates"}
    possessive = {"has", "have", "contains", "includes"}
    rel_type = None
    if rel_word in causal:
        rel_type = "causal"
    elif rel_word in possessive:
        rel_type = "possessive"

    def traverse_from(cid, exclude_words):
        """Traverse from a concept node, return (word, depth) pairs."""
        frontier = {cid}
        visited = {cid}
        results = []
        for depth in range(1, max_depth + 1):
            next_frontier = set()
            for nid in frontier:
                for tgt_id, edge in model.graph.get_outgoing(nid):
                    if tgt_id in visited:
                        continue
                    if rel_type and edge.relation_type != rel_type:
                        continue
                    next_frontier.add(tgt_id)
                    tokens = model.binding_map.get_tokens(tgt_id, 0.0)
                    for b in tokens:
                        word = tok.decode([b.token_id])
                        if word not in exclude_words and not word.startswith("?"):
                            results.append((word, depth))
            visited |= next_frontier
            frontier = next_frontier
        # Deduplicate, prefer shallower
        seen = {}
        for word, depth in results:
            if word not in seen or depth < seen[word]:
                seen[word] = depth
        return sorted(seen.items(), key=lambda x: (x[1], x[0]))[:10]

    # Step 1: Try direct traversal from query subject
    subj_tid = tok.word_to_id.get(subj_word)
    if subj_tid is not None:
        bindings = model.binding_map.get_concepts(subj_tid, min_confidence=0.1)
        if bindings:
            direct = traverse_from(bindings[0].concept_id, {subj_word, rel_word})
            if direct:
                return [(w, d, 1.0, subj_word) for w, d in direct]

    # Step 2: NN-seeded traversal
    nn_results = nn_predict(model, tok, query, facts, top_n=k_neighbors)
    all_candidates = []
    for neighbor_word, sim, orig_subject in nn_results:
        neighbor_tid = tok.word_to_id.get(neighbor_word)
        if neighbor_tid is None:
            continue
        bindings = model.binding_map.get_concepts(neighbor_tid, min_confidence=0.1)
        if not bindings:
            continue
        traversed = traverse_from(bindings[0].concept_id, {subj_word, rel_word, neighbor_word})
        for word, depth in traversed:
            all_candidates.append((word, depth, sim, neighbor_word))

    if all_candidates:
        all_candidates.sort(key=lambda x: (-x[2], x[1]))
        seen = set()
        results = []
        for word, depth, sim, via in all_candidates:
            if word not in seen:
                seen.add(word)
                results.append((word, depth, sim, via))
        return results[:10]

    # Step 3: Pure NN fallback (no concept nodes at all)
    nn_words = [w for w, _, _ in nn_results]
    return [(w, 0, 0.0, "nn_fallback") for w in nn_words[:5]]


def build_facts():
    """Facts for compositional reasoning tests."""
    return [
        # Chain 1: kindness → trust → cooperation (the target)
        ("kindness", "causes", "trust"),
        ("trust", "causes", "cooperation"),
        # Chain 2: cruelty → distrust → isolation (mirror)
        ("cruelty", "causes", "distrust"),
        ("distrust", "causes", "isolation"),
        # Chain 3: anger → conflict
        ("anger", "causes", "conflict"),
        # Chain 4: honesty → respect
        ("honesty", "causes", "respect"),
        # Concrete facts (for NN baseline)
        ("cat", "has", "tail"),
        ("dog", "has", "tail"),
        ("cat", "has", "fur"),
        ("dog", "has", "fur"),
        # Simple causal
        ("virus", "causes", "illness"),
        ("illness", "causes", "absence"),
        ("bug", "causes", "crash"),
        ("crash", "causes", "outage"),
    ]


def main():
    print("=" * 70)
    print("COMPOSITIONAL HYBRID BENCHMARK")
    print("=" * 70)
    print()
    print("Tests whether NN + Traversal can COOPERATE on tasks")
    print("neither can solve alone.")
    print()

    facts = build_facts()

    # Build tokenizer
    tok = WordTokenizer()
    for s, r, o in facts:
        tok.encode(f"{s} {r} {o}")

    # Extra vocab for queries
    for w in ["warmth", "coldness", "tiger", "eagle"]:
        tok.encode(w)

    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=500, sleep_interval=200, gate_concept_creation=False)
    model._tokenizer = tok
    inject_embeddings(model, tok)

    print("Training...")
    for epoch in range(5):
        for s, r, o in facts:
            ids = tok.encode(f"{s} {r} {o}")
            if len(ids) < 2:
                continue
            ctx = np.array([ids[:-1]], dtype=np.int64)
            tgt = np.array([[ids[-1]]], dtype=np.int64)
            model.learn(ctx, tgt)

    print(f"Graph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")

    # ================================================================
    # TEST 1: Pure compositional (the key test)
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 1: COMPOSITIONAL REASONING")
    print("warmth ≈ kindness → trust → cooperation")
    print("Neither NN nor Traversal alone should solve this.")
    print("=" * 70)

    test_cases = [
        ("warmth causes", "cooperation",
         "warmth→kindness(embed)→trust(trav)→cooperation(trav)"),
        ("coldness causes", "isolation",
         "coldness→cruelty(embed)→distrust(trav)→isolation(trav)"),
    ]

    for query, expected, chain_desc in test_cases:
        print(f"\n  Query: {query}")
        print(f"  Expected: {expected}")
        print(f"  Chain: {chain_desc}")

        nn = nn_predict(model, tok, query, facts)
        tr = traversal_predict(model, tok, query)
        comp = compositional_predict(model, tok, query, facts, k_neighbors=3)

        nn_words = [w for w, _, _ in nn]
        tr_words = [w for w, _ in tr]
        comp_words = [w for w, _, _, _ in comp]

        nn_hit = expected in nn_words
        tr_hit = expected in tr_words
        comp_hit = expected in comp_words

        print(f"\n  NN:          [{'Y' if nn_hit else 'N'}] {nn_words[:5]}")
        if nn:
            for w, sim, via in nn[:3]:
                print(f"                 {w} (via {via}, sim={sim:.3f})")
        print(f"  Traversal:   [{'Y' if tr_hit else 'N'}] {tr_words[:5]}")
        print(f"  Composed:    [{'Y' if comp_hit else 'N'}] {comp_words[:5]}")
        if comp:
            for w, depth, sim, via in comp[:5]:
                print(f"                 {w} (depth={depth}, via {via}, sim={sim:.3f})")

    # ================================================================
    # TEST 2: Where NN already works (shouldn't regress)
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 2: NN-ONLY TASKS (shouldn't regress)")
    print("=" * 70)

    nn_only = [
        ("tiger has", "tail"),
        ("kindness causes", "trust"),  # direct edge, NN should find it
        ("cruelty causes", "distrust"),
    ]

    for query, expected in nn_only:
        nn = nn_predict(model, tok, query, facts)
        tr = traversal_predict(model, tok, query)
        comp = compositional_predict(model, tok, query, facts)

        nn_words = [w for w, _, _ in nn]
        tr_words = [w for w, _ in tr]
        comp_words = [w for w, _, _, _ in comp]

        print(f"\n  {query:25s} -> {expected}")
        print(f"    NN:        [{'Y' if expected in nn_words else 'N'}] {nn_words[:3]}")
        print(f"    Traverse:  [{'Y' if expected in tr_words else 'N'}] {tr_words[:3]}")
        print(f"    Composed:  [{'Y' if expected in comp_words else 'N'}] {comp_words[:3]}")

    # ================================================================
    # TEST 3: Where traversal already works (shouldn't regress)
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 3: TRAVERSAL-ONLY TASKS (shouldn't regress)")
    print("=" * 70)

    tr_only = [
        ("virus causes", "absence"),
        ("bug causes", "outage"),
    ]

    for query, expected in tr_only:
        nn = nn_predict(model, tok, query, facts)
        tr = traversal_predict(model, tok, query)
        comp = compositional_predict(model, tok, query, facts)

        nn_words = [w for w, _, _ in nn]
        tr_words = [w for w, _ in tr]
        comp_words = [w for w, _, _, _ in comp]

        print(f"\n  {query:25s} -> {expected}")
        print(f"    NN:        [{'Y' if expected in nn_words else 'N'}] {nn_words[:3]}")
        print(f"    Traverse:  [{'Y' if expected in tr_words else 'N'}] {tr_words[:3]}")
        print(f"    Composed:  [{'Y' if expected in comp_words else 'N'}] {comp_words[:3]}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("CAPABILITY MATRIX")
    print("=" * 70)
    print()
    print(f"{'Task':<30s} {'NN':<8s} {'Trav':<8s} {'Composed':<10s}")
    print("-" * 56)

    all_tests = [
        # (query, expected, category)
        ("warmth causes", "cooperation", "Compositional"),
        ("coldness causes", "isolation", "Compositional"),
        ("tiger has", "tail", "NN-only"),
        ("kindness causes", "trust", "NN-only"),
        ("cruelty causes", "distrust", "NN-only"),
        ("virus causes", "absence", "Traversal-only"),
        ("bug causes", "outage", "Traversal-only"),
    ]

    cat_results = {}
    for query, expected, cat in all_tests:
        nn = nn_predict(model, tok, query, facts)
        tr = traversal_predict(model, tok, query)
        comp = compositional_predict(model, tok, query, facts)

        nn_hit = expected in [w for w, _, _ in nn]
        tr_hit = expected in [w for w, _ in tr]
        comp_hit = expected in [w for w, _, _, _ in comp]

        if cat not in cat_results:
            cat_results[cat] = {"nn": 0, "tr": 0, "comp": 0, "total": 0}
        cat_results[cat]["nn"] += int(nn_hit)
        cat_results[cat]["tr"] += int(tr_hit)
        cat_results[cat]["comp"] += int(comp_hit)
        cat_results[cat]["total"] += 1

    for cat, r in cat_results.items():
        print(f"{cat:<30s} {r['nn']}/{r['total']:<6d} {r['tr']}/{r['total']:<6d} {r['comp']}/{r['total']:<6d}")

    print()
    print("If Composed > max(NN, Trav) on Compositional tasks,")
    print("the hybrid is genuinely composing capabilities.")
    print("If Composed == max(NN, Trav), it's just switching.")


if __name__ == "__main__":
    main()
