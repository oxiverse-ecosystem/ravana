"""
Directed Traversal vs Spreading Activation
============================================
Compares two inference modes on the same graph:
  1. Spreading activation (broadcast/diffusion) -- current
  2. Directed chain traversal (path-following) -- new

Also inspects the raw graph structure to verify chains are actually stored.
"""
import sys
import io
import numpy as np
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.word_tokenizer import WordTokenizer


def inspect_graph(model, tok):
    """Print the actual graph structure to verify chains are stored."""
    print("=" * 70)
    print("GRAPH STRUCTURE INSPECTION")
    print("=" * 70)
    graph = model.graph
    print(f"\nNodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")

    print("\n--- All Edges ---")
    for (src, tgt), edge in graph.edges.items():
        src_t = [tok.decode([b.token_id]) for b in model.binding_map.get_tokens(src, 0.0)]
        tgt_t = [tok.decode([b.token_id]) for b in model.binding_map.get_tokens(tgt, 0.0)]
        sl = src_t[0] if src_t else f"c{src}"
        tl = tgt_t[0] if tgt_t else f"c{tgt}"
        print(f"  {sl:15s} --[{edge.relation_type:10s}]--> {tl:15s}  w={edge.weight:.3f}")

    # Find 2-hop causal chains
    print("\n--- 2-Hop Causal Chains ---")
    chains = 0
    for (src, tgt), edge in graph.edges.items():
        if edge.relation_type not in ("causal", "possessive"):
            continue
        src_t = [tok.decode([b.token_id]) for b in model.binding_map.get_tokens(src, 0.0)]
        tgt_t = [tok.decode([b.token_id]) for b in model.binding_map.get_tokens(tgt, 0.0)]
        if not src_t or not tgt_t:
            continue
        for (src2, tgt2), edge2 in graph.edges.items():
            if src2 != tgt or edge2.relation_type != edge.relation_type:
                continue
            tgt2_t = [tok.decode([b.token_id]) for b in model.binding_map.get_tokens(tgt2, 0.0)]
            if not tgt2_t:
                continue
            print(f"  {src_t[0]} --[{edge.relation_type}]--> {tgt_t[0]} --[{edge2.relation_type}]--> {tgt2_t[0]}")
            chains += 1
    if chains == 0:
        print("  NONE FOUND")

    # Also find chains via ANY edge type (semantic included)
    print("\n--- 2-Hop Chains (any edge type) ---")
    chains2 = 0
    for (src, tgt), edge in graph.edges.items():
        src_t = [tok.decode([b.token_id]) for b in model.binding_map.get_tokens(src, 0.0)]
        tgt_t = [tok.decode([b.token_id]) for b in model.binding_map.get_tokens(tgt, 0.0)]
        if not src_t or not tgt_t:
            continue
        for (src2, tgt2), edge2 in graph.edges.items():
            if src2 != tgt:
                continue
            tgt2_t = [tok.decode([b.token_id]) for b in model.binding_map.get_tokens(tgt2, 0.0)]
            if not tgt2_t:
                continue
            print(f"  {src_t[0]} --[{edge.relation_type}]--> {tgt_t[0]} --[{edge2.relation_type}]--> {tgt2_t[0]}")
            chains2 += 1
    if chains2 == 0:
        print("  NONE FOUND")
    return chains


def directed_traverse(model, tok, query, max_depth=2, relation_filter=True):
    """Follow edges from subject, optionally filtered by relation type."""
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

    # Determine relation type from query word
    rel_type = None
    if relation_filter:
        causal = {"causes", "cause", "leads", "produces", "creates", "triggers"}
        possessive = {"has", "have", "contains", "includes"}
        if rel_word in causal:
            rel_type = "causal"
        elif rel_word in possessive:
            rel_type = "possessive"

    # Traverse
    frontier = {subject_cid}
    visited = {subject_cid}
    candidates = []  # (word, depth, path)

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
                    if word not in (subj_word, rel_word):
                        candidates.append((word, depth))
        visited |= next_frontier
        frontier = next_frontier

    # Deduplicate, prefer deeper answers (the chain endpoint)
    seen = {}
    for word, depth in candidates:
        if word not in seen or depth > seen[word]:
            seen[word] = depth
    # Sort: deeper first, then alphabetical
    results = sorted(seen.items(), key=lambda x: (-x[1], x[0]))
    return results[:10]


def nn_predict(model, tok, query, facts):
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
    return results[:5]


def spread_predict(model, tok, query):
    """Current forward() — spreading activation."""
    ids = tok.encode(query)
    if not ids:
        return []
    ctx = np.array([ids], dtype=np.int64)
    logits = np.asarray(model.forward(ctx).data).flatten()
    top_ids = list(np.argsort(logits)[::-1][:5])
    return [tok.decode([tid]) for tid in top_ids]


def inject_embeddings(model, tok):
    """Hand-crafted semantic embeddings."""
    sem = {
        "cat": [1,0.4,0.8], "dog": [1,0.5,0.7], "bird": [1,0.2,0.6],
        "fish": [1,0.3,0.5], "horse": [1,0.8,0.9],
        "tiger": [1,0.45,0.82], "eagle": [1,0.22,0.62], "shark": [1,0.35,0.52],
        "wolf": [1,0.52,0.72], "parrot": [1,0.18,0.58], "whale": [1,0.75,0.48],
        "tail": [0,0.8,0.3], "whiskers": [0,0.3,0.35], "nose": [0,0.4,0.4],
        "wing": [0,0.6,0.2], "beak": [0,0.5,0.25], "fin": [0,0.7,0.15],
        "scale": [0,0.2,0.1], "hoof": [0,0.9,0.45],
        "purr": [-0.5,0.3,0.1], "bark": [-0.5,0.5,0.15], "fly": [-0.5,0.8,0.2], "swim": [-0.5,0.6,0.1],
        "has": [0.5,0,0], "can": [0.5,0,0], "causes": [0.5,0,0],
        "virus": [-1,0.8,0.2], "illness": [-1,0.5,0.3], "absence": [-1,0.3,0.4],
        "bug": [-1,0.75,0.15], "crash": [-1,0.5,0.25], "outage": [-1,0.3,0.35],
        "spark": [-1,0.7,0.18], "fire": [-1,0.5,0.28], "damage": [-1,0.3,0.38],
        "lie": [-1,0.72,0.12], "distrust": [-1,0.48,0.22], "isolation": [0.3,0.28,0.32],
        "kindness": [-1,0.85,0.15], "anger": [-1,0.65,0.35], "honesty": [-1,0.75,0.25],
        "heat": [-1,0.9,0.1], "cold": [-1,0.1,0.9],
        "trust": [-1,0.4,0.2], "conflict": [-1,0.3,0.4], "respect": [-1,0.35,0.3],
        "expansion": [-1,0.5,0.15], "contraction": [0.1,0.15,0.85],
        "warmth": [-1,0.83,0.17], "rudeness": [-1,0.63,0.37],
        "loyalty": [-1,0.73,0.27], "frigidity": [-1,0.12,0.88],
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


def build_adversarial_facts():
    """Facts designed to test relation-aware traversal.

    Same source node 'virus' has multiple relation types:
      virus causes illness
      virus discovered_by smith
      virus located_in blood
    Traversal with relation_filter should find 'illness', not 'smith' or 'blood'.
    """
    return [
        # Causal chains (the targets)
        ("virus", "causes", "illness"),
        ("illness", "causes", "absence"),
        ("bug", "causes", "crash"),
        ("crash", "causes", "outage"),
        ("spark", "causes", "fire"),
        ("fire", "causes", "damage"),
        ("lie", "causes", "distrust"),
        ("distrust", "causes", "isolation"),
        # Distractors: same source, different relation type
        ("virus", "discovered_by", "smith"),
        ("virus", "located_in", "blood"),
        ("bug", "discovered_by", "john"),
        ("bug", "located_in", "code"),
        ("spark", "discovered_by", "franklin"),
        ("spark", "located_in", "wire"),
        ("lie", "discovered_by", "detective"),
        ("lie", "located_in", "testimony"),
        # Distractor chains (same relation type, different source)
        ("smith", "causes", "university"),
        ("blood", "causes", "transfusion"),
        ("code", "causes", "software"),
        ("wire", "causes", "electricity"),
        # Simple causal
        ("kindness", "causes", "trust"),
        ("anger", "causes", "conflict"),
        ("honesty", "causes", "respect"),
        ("heat", "causes", "expansion"),
        ("cold", "causes", "contraction"),
        # Retrieval facts
        ("cat", "has", "tail"), ("dog", "has", "tail"),
        ("bird", "has", "wing"), ("fish", "has", "fin"),
        ("cat", "can", "purr"), ("dog", "can", "bark"),
        ("bird", "can", "fly"), ("fish", "can", "swim"),
    ]


def build_tests(facts):
    """Tests for the adversarial benchmark."""
    retrieval = [
        {"query": "tiger has", "expected": "tail", "type": "R"},
        {"query": "eagle has", "expected": "wing", "type": "R"},
    ]
    multihop = [
        {"query": "virus causes", "expected": "absence", "type": "M"},
        {"query": "bug causes", "expected": "outage", "type": "M"},
        {"query": "spark causes", "expected": "damage", "type": "M"},
        {"query": "lie causes", "expected": "isolation", "type": "M"},
    ]
    hybrid = [
        {"query": "warmth causes", "expected": "trust", "type": "H"},
        {"query": "rudeness causes", "expected": "conflict", "type": "H"},
    ]
    return retrieval, multihop, hybrid


def run_tests(model, tok, tests, facts, label):
    """Run all three modes on a test set."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    results = []
    for test in tests:
        q, exp = test["query"], test["expected"]

        nn = nn_predict(model, tok, q, facts)
        sa = spread_predict(model, tok, q)
        tr = directed_traverse(model, tok, q, max_depth=2, relation_filter=True)
        tr_words = [w for w, _ in tr]
        tr_unfiltered = directed_traverse(model, tok, q, max_depth=2, relation_filter=False)
        tr_unf_words = [w for w, _ in tr_unfiltered]

        nn_hit = exp in nn
        sa_hit = exp in sa
        tr_hit = exp in tr_words
        tr_unf_hit = exp in tr_unf_words

        print(f"\n  {q:20s} -> {exp:12s}")
        print(f"    NN:        [{'Y' if nn_hit else 'N'}] {nn[:5]}")
        print(f"    Spread:    [{'Y' if sa_hit else 'N'}] {sa[:5]}")
        print(f"    Traverse:  [{'Y' if tr_hit else 'N'}] {tr_words[:5]}  (relation-filtered)")
        print(f"    Traverse*: [{'Y' if tr_unf_hit else 'N'}] {tr_unf_words[:5]}  (any edge)")

        results.append({"q": q, "exp": exp, "type": test["type"],
                        "nn": nn_hit, "sa": sa_hit, "tr": tr_hit, "tr_unf": tr_unf_hit})

    n = len(results)
    nn_n = sum(1 for r in results if r["nn"])
    sa_n = sum(1 for r in results if r["sa"])
    tr_n = sum(1 for r in results if r["tr"])
    tr_unf_n = sum(1 for r in results if r["tr_unf"])
    print(f"\n  Summary: NN={nn_n}/{n}  Spread={sa_n}/{n}  Traverse={tr_n}/{n}  Traverse*={tr_unf_n}/{n}")
    return results


def main():
    print("=" * 70)
    print("DIRECTED TRAVERSAL vs SPREADING ACTIVATION")
    print("(with adversarial graph: same source, multiple relation types)")
    print("=" * 70)

    facts = build_adversarial_facts()
    retrieval, multihop, hybrid = build_tests(facts)

    all_queries = [t["query"] for t in retrieval + multihop + hybrid]
    tok = WordTokenizer()
    for s, r, o in facts:
        tok.encode(f"{s} {r} {o}")
    for q in all_queries:
        tok.encode(q)

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

    # Inspect structure
    chains = inspect_graph(model, tok)

    # Run benchmarks
    r_res = run_tests(model, tok, retrieval, facts, "RETRIEVAL")
    m_res = run_tests(model, tok, multihop, facts, "MULTI-HOP (the critical test)")
    h_res = run_tests(model, tok, hybrid, facts, "HYBRID")

    # Grand summary
    print("\n" + "=" * 70)
    print("GRAND SUMMARY")
    print("=" * 70)
    print(f"\n{'Task':<15s} {'NN':<8s} {'Spread':<8s} {'Traverse':<10s} {'Trav*':<8s}")
    print("-" * 49)
    for name, res in [("Retrieval", r_res), ("Multi-hop", m_res), ("Hybrid", h_res)]:
        n = len(res)
        nn = sum(1 for r in res if r["nn"])
        sa = sum(1 for r in res if r["sa"])
        tr = sum(1 for r in res if r["tr"])
        tr_unf = sum(1 for r in res if r["tr_unf"])
        print(f"{name:<15s} {nn}/{n:<6d} {sa}/{n:<6d} {tr}/{n:<8d} {tr_unf}/{n:<6d}")

    print("\nTraverse  = relation-filtered (only follow matching edge types)")
    print("Traverse* = any edge (no relation filter)")
    print("\nIf Traverse > Traverse*: relation filtering is working correctly.")
    print("If Traverse < Traverse*: relation types are being assigned wrong.")
    print("If Traverse == Traverse*: all edges are the same type (classifier broken).")


if __name__ == "__main__":
    main()
