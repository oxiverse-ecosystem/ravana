"""
Graph Impurity & Cycle Benchmarks
===================================
Tests two failure modes not yet explored:

1. IMPURITY: Multiple valid chains from same subject.
   virus causes illness, virus causes concern, virus causes fever
   illness causes absence, illness causes treatment
   Query: virus causes ? → should return all valid endpoints with ranking

2. CYCLES: A causes B causes C causes A.
   Traversal should not loop infinitely. Visited-set should handle it.
   But does it produce meaningful results?

Also tests: conflicting facts, many-to-many relations.
"""
import sys
import io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.word_tokenizer import WordTokenizer


def inject_embeddings(model, tok, embed_map):
    """Inject hand-crafted embeddings."""
    dim = model.embed_dim
    for word, v3 in embed_map.items():
        tid = tok.word_to_id.get(word)
        if tid is None:
            continue
        full = np.zeros(dim, dtype=np.float32)
        rng = np.random.RandomState(hash(word) % 2**31)
        for i in range(dim):
            full[i] = v3[i % 3] + rng.randn() * 0.005
        full /= np.linalg.norm(full)
        model.token_embed.weight.data[tid] = full


def train_facts(model, tok, facts, epochs=5):
    """Train model on facts."""
    for epoch in range(epochs):
        for s, r, o in facts:
            ids = tok.encode(f"{s} {r} {o}")
            if len(ids) < 2:
                continue
            ctx = np.array([ids[:-1]], dtype=np.int64)
            tgt = np.array([[ids[-1]]], dtype=np.int64)
            model.learn(ctx, tgt)


def traverse(model, tok, query, max_depth=3):
    """Relation-aware traversal with depth tracking."""
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

    causal = {"causes", "cause", "leads", "produces", "creates"}
    rel_type = "causal" if rel_word in causal else None

    frontier = {bindings[0].concept_id}
    visited = {bindings[0].concept_id}
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
        if word not in seen or depth < seen[word]:
            seen[word] = depth
    return sorted(seen.items(), key=lambda x: (x[1], x[0]))


# ============================================================
# BENCHMARK 1: GRAPH IMPURITY
# ============================================================

def benchmark_impurity():
    """Multiple valid chains from same subject."""
    print("=" * 70)
    print("BENCHMARK 1: GRAPH IMPURITY")
    print("Multiple valid chains from same subject node")
    print("=" * 70)

    facts = [
        # virus has 3 direct causal targets
        ("virus", "causes", "illness"),
        ("virus", "causes", "concern"),
        ("virus", "causes", "fever"),
        # illness has 2 causal targets (depth 2 from virus)
        ("illness", "causes", "absence"),
        ("illness", "causes", "treatment"),
        # concern has 1 target (depth 2)
        ("concern", "causes", "panic"),
        # fever has 1 target (depth 2)
        ("fever", "causes", "delirium"),
        # Distractors: same subjects, different relations
        ("virus", "discovered_by", "smith"),
        ("virus", "located_in", "blood"),
        ("illness", "discovered_by", "jones"),
        ("illness", "located_in", "hospital"),
        # Other causal chains (noise)
        ("bug", "causes", "crash"),
        ("crash", "causes", "outage"),
        ("spark", "causes", "fire"),
        ("fire", "causes", "damage"),
        ("lie", "causes", "distrust"),
        ("distrust", "causes", "isolation"),
        # Possessive
        ("cat", "has", "tail"),
        ("dog", "has", "tail"),
    ]

    embed_map = {
        "virus": [-1, 0.8, 0.2], "illness": [-0.8, 0.5, 0.3],
        "concern": [-0.7, 0.6, 0.4], "fever": [-0.9, 0.4, 0.5],
        "absence": [-0.5, 0.3, 0.4], "treatment": [-0.4, 0.7, 0.3],
        "panic": [-0.6, 0.2, 0.6], "delirium": [-0.85, 0.35, 0.55],
        "bug": [-0.3, 0.75, 0.15], "crash": [-0.3, 0.5, 0.25],
        "outage": [-0.3, 0.3, 0.35], "spark": [-0.2, 0.7, 0.18],
        "fire": [-0.2, 0.5, 0.28], "damage": [-0.2, 0.3, 0.38],
        "lie": [-0.1, 0.72, 0.12], "distrust": [-0.1, 0.48, 0.22],
        "isolation": [0.3, 0.28, 0.32],
        "cat": [0.95, 0.3, 0.8], "dog": [0.9, 0.4, 0.7],
        "tail": [0.1, 0.8, 0.3],
        "smith": [0.5, 0.5, 0.5], "blood": [0.4, 0.4, 0.6],
        "jones": [0.55, 0.45, 0.55], "hospital": [0.45, 0.55, 0.45],
    }

    all_words = set()
    for s, r, o in facts:
        all_words.update([s, r, o])

    tok = WordTokenizer()
    for s, r, o in facts:
        tok.encode(f"{s} {r} {o}")

    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=500, gate_concept_creation=False)
    model._tokenizer = tok
    inject_embeddings(model, tok, embed_map)
    train_facts(model, tok, facts)

    print(f"\nGraph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")

    # Test: virus causes ?
    print("\n--- virus causes ? ---")
    print("Expected endpoints: absence, treatment, panic, delirium, concern, fever")
    print("(depth 1: concern, fever, illness; depth 2: absence, treatment, panic, delirium)")

    results = traverse(model, tok, "virus causes", max_depth=3)
    print(f"\nTraversal results ({len(results)} candidates):")
    for word, depth in results:
        print(f"  depth {depth}: {word}")

    # Check: are all expected endpoints found?
    expected_depth2 = {"absence", "treatment", "panic", "delirium"}
    expected_depth1 = {"concern", "fever", "illness"}
    found_d2 = {w for w, d in results if d == 2}
    found_d1 = {w for w, d in results if d == 1}

    print(f"\nDepth-1 endpoints found: {found_d1 & expected_depth1}/{expected_depth1}")
    print(f"Depth-2 endpoints found: {found_d2 & expected_depth2}/{expected_depth2}")

    # Are distractors filtered out?
    distractors = {"smith", "blood", "jones", "hospital"}
    found_distractors = {w for w, _ in results if w in distractors}
    print(f"Distractors found: {found_distractors if found_distractors else 'NONE (correct)'}")

    # Impurity score: how many distinct valid endpoints?
    valid = {w for w, _ in results if w not in distractors and not w.startswith("?")}
    print(f"Valid endpoints: {len(valid)}")
    print(f"Impurity handled: {'YES' if len(valid) >= 4 else 'NO'}")


# ============================================================
# BENCHMARK 2: CYCLES
# ============================================================

def benchmark_cycles():
    """Cyclic causal chains."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: CYCLES")
    print("A causes B causes C causes A — traversal should not loop")
    print("=" * 70)

    facts = [
        # Cycle: stress → insomnia → fatigue → stress
        ("stress", "causes", "insomnia"),
        ("insomnia", "causes", "fatigue"),
        ("fatigue", "causes", "stress"),
        # Cycle: rain → flood → erosion → rain
        ("rain", "causes", "flood"),
        ("flood", "causes", "erosion"),
        ("erosion", "causes", "rain"),
        # Linear chain (control)
        ("spark", "causes", "fire"),
        ("fire", "causes", "damage"),
        # Distractors
        ("stress", "discovered_by", "hans"),
        ("rain", "located_in", "clouds"),
    ]

    embed_map = {
        "stress": [-1, 0.8, 0.2], "insomnia": [-0.8, 0.5, 0.3],
        "fatigue": [-0.6, 0.3, 0.4], "rain": [-0.5, 0.7, 0.1],
        "flood": [-0.5, 0.5, 0.2], "erosion": [-0.5, 0.3, 0.3],
        "spark": [0.5, 0.7, 0.1], "fire": [0.5, 0.5, 0.2],
        "damage": [0.5, 0.3, 0.3],
        "hans": [0, 0, 0], "clouds": [0, 0, 0],
    }

    tok = WordTokenizer()
    for s, r, o in facts:
        tok.encode(f"{s} {r} {o}")

    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=500, gate_concept_creation=False)
    model._tokenizer = tok
    inject_embeddings(model, tok, embed_map)
    train_facts(model, tok, facts)

    print(f"\nGraph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")

    # Test cycle traversal
    queries = [
        ("stress causes", "insomnia (depth 1), fatigue (depth 2)"),
        ("rain causes", "flood (depth 1), erosion (depth 2)"),
        ("spark causes", "fire (depth 1), damage (depth 2) — linear control"),
    ]

    for query, expected_desc in queries:
        print(f"\n--- {query} ---")
        print(f"Expected: {expected_desc}")

        results = traverse(model, tok, query, max_depth=5)
        print(f"Traversal ({len(results)} candidates):")
        for word, depth in results:
            print(f"  depth {depth}: {word}")

        # Check: no infinite loop (results are finite)
        # Check: cycle nodes appear at correct depths
        # stress → insomnia (d1) → fatigue (d2) → stress (d3, but excluded as subject)
        # So from stress, we should find insomnia at d1, fatigue at d2, and nothing at d3+
        # because stress is already visited

        max_depth_found = max(d for _, d in results) if results else 0
        print(f"Max depth reached: {max_depth_found}")
        print(f"Finite result set: {'YES' if len(results) < 100 else 'NO — EXPLOSION!'}")


# ============================================================
# BENCHMARK 3: CONFLICTING FACTS
# ============================================================

def benchmark_conflicts():
    """Conflicting facts about the same subject."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: CONFLICTING FACTS")
    print("Same subject, same relation, different objects")
    print("=" * 70)

    facts = [
        # Conflict: coffee both causes and prevents
        ("coffee", "causes", "alertness"),
        ("coffee", "causes", "anxiety"),
        ("coffee", "causes", "insomnia"),
        # Conflict: exercise
        ("exercise", "causes", "strength"),
        ("exercise", "causes", "injury"),
        # Non-conflicting
        ("sunlight", "causes", "warmth"),
        ("warmth", "causes", "comfort"),
        # Distractors
        ("coffee", "discovered_by", "ethiopian"),
        ("exercise", "located_in", "gym"),
    ]

    embed_map = {
        "coffee": [-1, 0.8, 0.2], "alertness": [-0.8, 0.6, 0.3],
        "anxiety": [-0.9, 0.4, 0.5], "insomnia": [-0.7, 0.3, 0.6],
        "exercise": [0.5, 0.8, 0.2], "strength": [0.7, 0.6, 0.3],
        "injury": [0.3, 0.4, 0.5],
        "sunlight": [1, 0.9, 0.1], "warmth": [0.9, 0.8, 0.2],
        "comfort": [0.8, 0.7, 0.3],
        "ethiopian": [0, 0, 0], "gym": [0, 0, 0],
    }

    tok = WordTokenizer()
    for s, r, o in facts:
        tok.encode(f"{s} {r} {o}")

    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=500, gate_concept_creation=False)
    model._tokenizer = tok
    inject_embeddings(model, tok, embed_map)
    train_facts(model, tok, facts)

    print(f"\nGraph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")

    queries = [
        ("coffee causes", "alertness, anxiety, insomnia (all valid)"),
        ("exercise causes", "strength, injury (both valid)"),
        ("sunlight causes", "warmth (single valid)"),
    ]

    for query, expected_desc in queries:
        print(f"\n--- {query} ---")
        print(f"Expected: {expected_desc}")

        results = traverse(model, tok, query, max_depth=3)
        print(f"Traversal ({len(results)} candidates):")
        for word, depth in results:
            print(f"  depth {depth}: {word}")

        # All valid targets should be found
        valid_targets = {"alertness", "anxiety", "insomnia", "strength", "injury", "warmth", "comfort"}
        found_valid = {w for w, _ in results if w in valid_targets}
        distractors = {"ethiopian", "gym"}
        found_distractors = {w for w, _ in results if w in distractors}
        print(f"Valid targets: {found_valid}")
        print(f"Distractors: {found_distractors if found_distractors else 'NONE'}")


# ============================================================
# BENCHMARK 4: MANY-TO-MANY
# ============================================================

def benchmark_many_to_many():
    """Many-to-many relations: multiple subjects share multiple objects."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: MANY-TO-MANY RELATIONS")
    print("Multiple subjects cause the same effects")
    print("=" * 70)

    facts = [
        # All these cause inflammation
        ("virus", "causes", "inflammation"),
        ("bacteria", "causes", "inflammation"),
        ("allergen", "causes", "inflammation"),
        ("injury", "causes", "inflammation"),
        # Inflammation causes multiple things
        ("inflammation", "causes", "pain"),
        ("inflammation", "causes", "swelling"),
        ("inflammation", "causes", "fever"),
        # Some subjects also cause other things
        ("virus", "causes", "fever"),
        ("bacteria", "causes", "infection"),
        ("infection", "causes", "sepsis"),
        # Distractors
        ("virus", "discovered_by", "ijinsky"),
        ("bacteria", "located_in", "soil"),
    ]

    embed_map = {
        "virus": [-1, 0.8, 0.2], "bacteria": [-0.9, 0.7, 0.3],
        "allergen": [-0.8, 0.6, 0.4], "injury": [-0.7, 0.5, 0.5],
        "inflammation": [-0.5, 0.5, 0.5], "pain": [-0.3, 0.4, 0.6],
        "swelling": [-0.3, 0.3, 0.7], "fever": [-0.3, 0.2, 0.8],
        "infection": [-0.6, 0.3, 0.3], "sepsis": [-0.4, 0.2, 0.4],
        "ijinsky": [0, 0, 0], "soil": [0, 0, 0],
    }

    tok = WordTokenizer()
    for s, r, o in facts:
        tok.encode(f"{s} {r} {o}")

    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=500, gate_concept_creation=False)
    model._tokenizer = tok
    inject_embeddings(model, tok, embed_map)
    train_facts(model, tok, facts)

    print(f"\nGraph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")

    # Test: from each subject, what do we reach?
    subjects = ["virus", "bacteria", "allergen", "injury"]
    for subj in subjects:
        print(f"\n--- {subj} causes ? ---")
        results = traverse(model, tok, f"{subj} causes", max_depth=3)
        words = [w for w, d in results]
        print(f"  Found: {words}")

    # Test: can we reach sepsis from virus? (virus→bacteria? no, virus→infection? no)
    # virus → inflammation → pain/swelling/fever (depth 2)
    # virus → fever (depth 1)
    # virus doesn't reach sepsis (that's bacteria→infection→sepsis)
    print(f"\n--- Can virus reach sepsis? ---")
    results = traverse(model, tok, "virus causes", max_depth=5)
    sepsis_found = any(w == "sepsis" for w, _ in results)
    print(f"  Sepsis reachable from virus: {sepsis_found}")
    print(f"  (Expected: NO — sepsis requires bacteria→infection→sepsis path)")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("GRAPH IMPURITY & CYCLE BENCHMARKS")
    print("=" * 70)

    benchmark_impurity()
    benchmark_cycles()
    benchmark_conflicts()
    benchmark_many_to_many()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key questions answered:
1. IMPURITY: Can traversal handle multiple valid chains from same subject?
2. CYCLES: Does traversal avoid infinite loops?
3. CONFLICTS: Are conflicting facts both discoverable?
4. MANY-TO-MANY: Do shared objects create correct topology?
""")


if __name__ == "__main__":
    main()
