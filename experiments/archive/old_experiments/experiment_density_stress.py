"""
Graph Density Stress Test
==========================
Tests whether the capability matrix survives as graph density increases.

Scale: 10 → 100 → 1000 facts
Metrics:
  - Traversal accuracy (does it still find the right chain?)
  - Candidate explosion (how many candidates per query?)
  - Average branching factor (edges per node)
  - Path ambiguity (multiple valid chains?)
  - Composed precision (single answer vs noisy list)
"""
import sys
import io
import numpy as np
import time
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.word_tokenizer import WordTokenizer


# ============================================================
# FACT GENERATORS
# ============================================================

def generate_causal_chains(n_chains, chain_length=3, seed=42):
    """Generate causal chains of varying length.

    Each chain: A causes B causes C causes D ...
    Returns facts and test queries.
    """
    rng = np.random.RandomState(seed)
    facts = []
    test_queries = []  # (query, expected_answer)

    # Word pools for generating diverse chains
    emotions = ["joy", "sorrow", "anger", "peace", "fear", "hope", "pride",
                "shame", "love", "hate", "envy", "gratitude", "grief",
                "delight", "rage", "calm", "dread", "trust", "doubt", "awe"]
    states = ["harmony", "chaos", "growth", "decay", "freedom", "bondage",
              "clarity", "confusion", "strength", "weakness", "warmth",
              "coldness", "light", "darkness", "silence", "noise",
              "order", "turmoil", "balance", "imbalance"]
    actions = ["creates", "destroys", "transforms", "reveals", "hides",
               "amplifies", "diminishes", "unites", "divides", "heals"]
    objects = ["connection", "distance", "understanding", "ignorance",
               "beauty", "ugliness", "truth", "falsehood", "courage", "cowardice",
               "wisdom", "folly", "patience", "impatience", "kindness", "cruelty",
               "generosity", "selfishness", "honor", "disgrace"]

    all_words = emotions + states + actions + objects
    rng.shuffle(all_words)

    used = set()
    for i in range(n_chains):
        # Pick a chain starter
        starter = emotions[i % len(emotions)]
        chain = [starter]

        for step in range(chain_length - 1):
            # Pick next word (not already used in this chain)
            candidates = [w for w in all_words if w not in used and w not in chain]
            if not candidates:
                candidates = [f"concept_{i}_{step}"]
            next_word = candidates[rng.randint(0, len(candidates))]
            chain.append(next_word)
            used.add(next_word)

        # Create causal facts for the chain
        for j in range(len(chain) - 1):
            facts.append((chain[j], "causes", chain[j + 1]))

        # Test: first element causes last element
        test_queries.append((f"{chain[0]} causes", chain[-1]))

    return facts, test_queries


def generate_distractor_facts(n_facts, seed=43):
    """Generate distractor facts that share subjects/objects with causal chains.

    These create competing paths: virus → smith, virus → blood, etc.
    """
    rng = np.random.RandomState(seed)
    facts = []

    people = ["smith", "john", "mary", "alice", "bob", "charlie", "diana",
              "eve", "frank", "grace", "henry", "iris", "jack", "karen",
              "leo", "mona", "nick", "olivia", "paul", "quinn"]
    places = ["blood", "code", "wire", "soil", "water", "air", "fire",
              "earth", "light", "shadow", "memory", "dream", "stone",
              "wood", "glass", "paper", "cloth", "metal", "crystal", "fog"]
    relations = ["discovered_by", "located_in", "related_to", "part_of",
                 "associated_with", "found_in", "created_by", "named_after"]

    for i in range(n_facts):
        subj = people[i % len(people)]
        obj = places[i % len(places)]
        rel = relations[i % len(relations)]
        facts.append((subj, rel, obj))

    return facts


def generate_possessive_facts(n_facts, seed=44):
    """Generate possessive facts: X has Y."""
    rng = np.random.RandomState(seed)
    facts = []

    animals = ["cat", "dog", "bird", "fish", "horse", "tiger", "eagle",
               "shark", "wolf", "bear", "fox", "deer", "owl", "snake",
               "frog", "bat", "whale", "lion", "monkey", "rabbit"]
    parts = ["tail", "wing", "fin", "hoof", "claw", "beak", "scale",
             "feather", "fur", "horn", "tusk", "fang", "paw", "flipper",
             "antler", "shell", "stinger", "tentacle", "trunk", "mane"]

    for i in range(n_facts):
        facts.append((animals[i % len(animals)], "has", parts[i % len(parts)]))

    return facts


# ============================================================
# METRICS
# ============================================================

def compute_branching_factor(model):
    """Average out-degree of concept nodes."""
    out_degrees = []
    for nid in model.graph.nodes:
        outgoing = model.graph.get_outgoing(nid)
        out_degrees.append(len(outgoing))
    if not out_degrees:
        return 0.0
    return np.mean(out_degrees)


def compute_path_ambiguity(model, tok, query, max_depth=2):
    """Count distinct paths from subject to any target."""
    parts = query.split()
    if len(parts) < 2:
        return 0
    subj_word, rel_word = parts[0], parts[1]

    subj_tid = tok.word_to_id.get(subj_word)
    if subj_tid is None:
        return 0
    bindings = model.binding_map.get_concepts(subj_tid, min_confidence=0.1)
    if not bindings:
        return 0

    causal = {"causes", "cause", "leads", "produces", "creates"}
    rel_type = "causal" if rel_word in causal else None

    paths = []
    frontier = [(bindings[0].concept_id, [subj_word])]
    visited = {bindings[0].concept_id}

    for depth in range(max_depth):
        next_frontier = []
        for nid, path in frontier:
            for tgt_id, edge in model.graph.get_outgoing(nid):
                if tgt_id in visited:
                    continue
                if rel_type and edge.relation_type != rel_type:
                    continue
                tokens = model.binding_map.get_tokens(tgt_id, 0.0)
                for b in tokens:
                    word = tok.decode([b.token_id])
                    if not word.startswith("?"):
                        paths.append(path + [word])
                        next_frontier.append((tgt_id, path + [word]))
            visited.add(nid)
        frontier = next_frontier

    return len(paths)


def traverse_query(model, tok, query, max_depth=2):
    """Relation-aware traversal. Returns (candidates, n_visited)."""
    parts = query.split()
    if len(parts) < 2:
        return [], 0
    subj_word, rel_word = parts[0], parts[1]

    subj_tid = tok.word_to_id.get(subj_word)
    if subj_tid is None:
        return [], 0
    bindings = model.binding_map.get_concepts(subj_tid, min_confidence=0.1)
    if not bindings:
        return [], 0

    causal = {"causes", "cause", "leads", "produces", "creates"}
    rel_type = "causal" if rel_word in causal else None

    frontier = {bindings[0].concept_id}
    visited = {bindings[0].concept_id}
    candidates = []
    n_visited = 0

    for depth in range(1, max_depth + 1):
        next_frontier = set()
        for nid in frontier:
            for tgt_id, edge in model.graph.get_outgoing(nid):
                if tgt_id in visited:
                    continue
                if rel_type and edge.relation_type != rel_type:
                    continue
                next_frontier.add(tgt_id)
                n_visited += 1
                tokens = model.binding_map.get_tokens(tgt_id, 0.0)
                for b in tokens:
                    word = tok.decode([b.token_id])
                    if word not in (subj_word, rel_word) and not word.startswith("?"):
                        candidates.append((word, depth))
        visited |= next_frontier
        frontier = next_frontier

    seen = {}
    for word, depth in candidates:
        if word not in seen or depth < seen[word]:
            seen[word] = depth
    return sorted(seen.items(), key=lambda x: (x[1], x[0])), n_visited


def nn_query(model, tok, query, facts, top_k=5):
    """Nearest-neighbor baseline."""
    parts = query.split()
    if len(parts) < 2:
        return []
    subj_word, rel_word = parts[0], parts[1]
    subj_tid = tok.word_to_id.get(subj_word)
    if subj_tid is None:
        return []
    embeds = model.token_embed.weight.data
    subj_vec = embeds[subj_tid]

    pairs = [(s, o) for s, r, o in facts if r == rel_word]
    scored = []
    for s, o in pairs:
        s_tid = tok.word_to_id.get(s)
        if s_tid is None:
            continue
        sim = float(np.dot(subj_vec, embeds[s_tid]))
        scored.append((sim, o))
    scored.sort(reverse=True)

    seen = set()
    results = []
    for sim, o in scored:
        if o not in seen:
            seen.add(o)
            results.append(o)
        if len(results) >= top_k:
            break
    return results


def composed_query(model, tok, query, facts, max_depth=2, k_neighbors=3):
    """Composed: direct traversal → NN-seeded traversal → NN fallback."""
    parts = query.split()
    if len(parts) < 2:
        return []
    subj_word, rel_word = parts[0], parts[1]

    causal = {"causes", "cause", "leads", "produces", "creates"}
    rel_type = "causal" if rel_word in causal else None

    def traverse_from(cid, exclude_words):
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
        seen = {}
        for word, depth in results:
            if word not in seen or depth < seen[word]:
                seen[word] = depth
        return sorted(seen.items(), key=lambda x: (x[1], x[0]))[:10]

    # Direct traversal
    subj_tid = tok.word_to_id.get(subj_word)
    if subj_tid is not None:
        bindings = model.binding_map.get_concepts(subj_tid, min_confidence=0.1)
        if bindings:
            direct = traverse_from(bindings[0].concept_id, {subj_word, rel_word})
            if direct:
                return [w for w, _ in direct]

    # NN-seeded
    nn_results = nn_query(model, tok, query, facts, top_k=k_neighbors)
    all_candidates = []
    for neighbor_word in nn_results:
        n_tid = tok.word_to_id.get(neighbor_word)
        if n_tid is None:
            continue
        bindings = model.binding_map.get_concepts(n_tid, min_confidence=0.1)
        if not bindings:
            continue
        traversed = traverse_from(bindings[0].concept_id, {subj_word, rel_word, neighbor_word})
        for word, depth in traversed:
            all_candidates.append((word, depth, neighbor_word))

    if all_candidates:
        all_candidates.sort(key=lambda x: x[1])
        seen = set()
        return [w for w, _, _ in all_candidates if w not in seen and not seen.add(w)][:5]

    # NN fallback
    return nn_results[:5]


# ============================================================
# MAIN
# ============================================================

def run_density_test(n_causal_chains, n_distractors, n_possessive, chain_length=3):
    """Run a single density test."""
    label = f"{n_causal_chains}chains+{n_distractors}distr+{n_possessive}poss"

    # Generate facts
    causal_facts, test_queries = generate_causal_chains(n_causal_chains, chain_length=chain_length)
    distractor_facts = generate_distractor_facts(n_distractors)
    possessive_facts = generate_possessive_facts(n_possessive)
    all_facts = causal_facts + distractor_facts + possessive_facts

    # Build tokenizer
    tok = WordTokenizer()
    for s, r, o in all_facts:
        tok.encode(f"{s} {r} {o}")
    for q, _ in test_queries:
        tok.encode(q)

    # Create model
    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=max(500, len(all_facts) * 5),
                  sleep_interval=200, gate_concept_creation=False)
    model._tokenizer = tok

    # Inject semantic embeddings (hand-crafted clusters)
    rng = np.random.RandomState(42)
    for tid in range(tok.vocab_size):
        word = tok.decode([tid])
        if word in tok.word_to_id:
            # Create clustered embeddings
            h = hash(word) % 1000
            vec = np.zeros(model.embed_dim, dtype=np.float32)
            vec[0] = (h / 1000.0) * 2 - 1
            vec[1] = rng.randn() * 0.3
            vec[2] = rng.randn() * 0.3
            for i in range(3, model.embed_dim):
                vec[i] = rng.randn() * 0.1
            vec /= np.linalg.norm(vec)
            model.token_embed.weight.data[tid] = vec

    # Make similar words have similar embeddings
    # (crude but sufficient for testing)
    for s, r, o in all_facts:
        s_tid = tok.word_to_id.get(s)
        o_tid = tok.word_to_id.get(o)
        if s_tid is not None and o_tid is not None:
            # Blend subject and object embeddings slightly
            blend = 0.1
            model.token_embed.weight.data[s_tid] = (
                (1 - blend) * model.token_embed.weight.data[s_tid] +
                blend * model.token_embed.weight.data[o_tid]
            )
            model.token_embed.weight.data[s_tid] /= np.linalg.norm(
                model.token_embed.weight.data[s_tid])

    # Train
    t0 = time.time()
    for epoch in range(3):
        for s, r, o in all_facts:
            ids = tok.encode(f"{s} {r} {o}")
            if len(ids) < 2:
                continue
            ctx = np.array([ids[:-1]], dtype=np.int64)
            tgt = np.array([[ids[-1]]], dtype=np.int64)
            model.learn(ctx, tgt)
    train_time = time.time() - t0

    # Metrics
    n_nodes = len(model.graph.nodes)
    n_edges = len(model.graph.edges)
    branching = compute_branching_factor(model)

    # Test each query
    t0 = time.time()
    trav_hits = 0
    nn_hits = 0
    comp_hits = 0
    total_candidates = 0
    total_visited = 0
    total_ambiguity = 0

    for query, expected in test_queries:
        # Traversal
        trav_result, n_vis = traverse_query(model, tok, query)
        trav_words = [w for w, _ in trav_result]
        trav_hit = expected in trav_words
        trav_hits += int(trav_hit)
        total_candidates += len(trav_words)
        total_visited += n_vis

        # NN
        nn_result = nn_query(model, tok, query, all_facts)
        nn_hit = expected in nn_result
        nn_hits += int(nn_hit)

        # Composed
        comp_result = composed_query(model, tok, query, all_facts)
        comp_hit = expected in comp_result
        comp_hits += int(comp_hit)

        # Path ambiguity
        ambiguity = compute_path_ambiguity(model, tok, query)
        total_ambiguity += ambiguity

    test_time = time.time() - t0
    n_queries = len(test_queries)

    return {
        "label": label,
        "n_facts": len(all_facts),
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "branching": branching,
        "train_time": train_time,
        "test_time": test_time,
        "trav_acc": trav_hits / n_queries if n_queries else 0,
        "nn_acc": nn_hits / n_queries if n_queries else 0,
        "comp_acc": comp_hits / n_queries if n_queries else 0,
        "avg_candidates": total_candidates / n_queries if n_queries else 0,
        "avg_visited": total_visited / n_queries if n_queries else 0,
        "avg_ambiguity": total_ambiguity / n_queries if n_queries else 0,
        "n_queries": n_queries,
    }


def main():
    print("=" * 80)
    print("GRAPH DENSITY STRESS TEST")
    print("Does the capability matrix survive as graph density increases?")
    print("=" * 80)

    # Test configurations: (n_causal_chains, n_distractors, n_possessive)
    configs = [
        (10, 0, 0),      # Minimal: just causal chains
        (10, 20, 10),    # Small: with distractors
        (50, 50, 50),    # Medium
        (100, 100, 100), # Large
        (200, 200, 200), # Stress
    ]

    results = []
    for n_chains, n_distr, n_poss in configs:
        print(f"\n--- {n_chains} chains + {n_distr} distractors + {n_poss} possessive ---")
        r = run_density_test(n_chains, n_distr, n_poss)
        results.append(r)
        print(f"  Facts: {r['n_facts']}, Nodes: {r['n_nodes']}, Edges: {r['n_edges']}")
        print(f"  Branching: {r['branching']:.2f}")
        print(f"  Trav: {r['trav_acc']:.0%}  NN: {r['nn_acc']:.0%}  Composed: {r['comp_acc']:.0%}")
        print(f"  Avg candidates: {r['avg_candidates']:.1f}  Avg visited: {r['avg_visited']:.1f}")
        print(f"  Avg path ambiguity: {r['avg_ambiguity']:.1f}")
        print(f"  Train: {r['train_time']:.1f}s  Test: {r['test_time']:.1f}s")

    # Summary table
    print("\n" + "=" * 80)
    print("DENSITY SCALING SUMMARY")
    print("=" * 80)
    print(f"\n{'Config':<25s} {'Facts':<7s} {'Nodes':<7s} {'Edges':<7s} {'Br.F':<6s} "
          f"{'Trav':<6s} {'NN':<6s} {'Comp':<6s} {'Cands':<7s} {'Ambig':<7s}")
    print("-" * 88)
    for r in results:
        print(f"{r['label']:<25s} {r['n_facts']:<7d} {r['n_nodes']:<7d} {r['n_edges']:<7d} "
              f"{r['branching']:<6.2f} {r['trav_acc']:<6.0%} {r['nn_acc']:<6.0%} "
              f"{r['comp_acc']:<6.0%} {r['avg_candidates']:<7.1f} {r['avg_ambiguity']:<7.1f}")

    # Scaling analysis
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)

    if len(results) >= 2:
        first = results[0]
        last = results[-1]
        density_ratio = last['n_facts'] / first['n_facts'] if first['n_facts'] else 0
        trav_delta = last['trav_acc'] - first['trav_acc']
        nn_delta = last['nn_acc'] - first['nn_acc']
        comp_delta = last['comp_acc'] - first['comp_acc']
        cand_ratio = last['avg_candidates'] / first['avg_candidates'] if first['avg_candidates'] else 0

        print(f"\n  Density increase: {density_ratio:.0f}x")
        print(f"  Traversal accuracy delta: {trav_delta:+.0%}")
        print(f"  NN accuracy delta: {nn_delta:+.0%}")
        print(f"  Composed accuracy delta: {comp_delta:+.0%}")
        print(f"  Candidate explosion ratio: {cand_ratio:.1f}x")

        if abs(comp_delta) < 0.2:
            print("\n  ✓ Composed approach is STABLE across density range.")
        else:
            print("\n  ✗ Composed approach DEGRADES with density.")

        if cand_ratio < density_ratio * 0.5:
            print("  ✓ Candidate growth is SUB-LINEAR (traversal is discriminating).")
        else:
            print("  ✗ Candidate growth is LINEAR+ (traversal losing discrimination).")


if __name__ == "__main__":
    main()
