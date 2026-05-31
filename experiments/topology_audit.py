"""Topology audit: measure graph connectivity before/after factored edges."""
import numpy as np
from collections import deque, defaultdict
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.word_tokenizer import WordTokenizer


def shortest_path(graph, src, tgt, max_depth=10):
    """BFS shortest path between two concept nodes."""
    if src == tgt:
        return 0
    visited = {src}
    queue = deque([(src, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor, _ in graph.get_outgoing(node):
            if neighbor == tgt:
                return depth + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return -1  # unreachable


def audit_topology(model, tok, facts):
    """Run full topology audit on a trained model."""
    graph = model.graph
    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)
    
    # Build adjacency (bidirectional for path analysis)
    adj = defaultdict(set)
    for (src, tgt), _ in graph.edges.items():
        adj[src].add(tgt)
        adj[tgt].add(src)  # treat as undirected for connectivity
    
    # 1. Connected components
    visited = set()
    components = []
    for nid in graph.nodes:
        if nid in visited:
            continue
        comp = set()
        queue = deque([nid])
        while queue:
            n = queue.popleft()
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            for nb in adj[n]:
                if nb not in visited:
                    queue.append(nb)
        components.append(comp)
    
    # 2. Degree distribution
    degrees = {nid: len(adj[nid]) for nid in graph.nodes}
    avg_degree = sum(degrees.values()) / max(1, n_nodes)
    max_degree_node = max(degrees, key=degrees.get) if degrees else -1
    
    # 3. Hub analysis — which nodes connect multiple facts?
    hub_nodes = {nid: deg for nid, deg in degrees.items() if deg >= 2}
    
    # 4. Path lengths between concept pairs from different facts
    concept_ids = []
    for text, target in facts:
        ids = tok.encode(text)
        subject_tid = ids[0]
        target_tid = ids[-1]
        bindings_s = model.binding_map.get_tokens(subject_tid, 0.1)
        bindings_t = model.binding_map.get_tokens(target_tid, 0.1)
        if bindings_s and bindings_t:
            concept_ids.append((bindings_s[0].concept_id, bindings_t[0].concept_id))
    
    path_lengths = []
    reachable_pairs = 0
    total_pairs = 0
    for i in range(len(concept_ids)):
        for j in range(i + 1, len(concept_ids)):
            for a in concept_ids[i]:
                for b in concept_ids[j]:
                    total_pairs += 1
                    pl = shortest_path(graph, a, b)
                    if pl >= 0:
                        path_lengths.append(pl)
                        reachable_pairs += 1
    
    avg_path = sum(path_lengths) / max(1, len(path_lengths))
    
    # 5. Print report
    print(f"=== TOPOLOGY AUDIT ===")
    print(f"Nodes: {n_nodes}, Edges: {n_edges}")
    print(f"Connected components: {len(components)}")
    print(f"Component sizes: {[len(c) for c in components]}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Max degree: {degrees.get(max_degree_node, 0)} (node {max_degree_node})")
    
    # Print hub nodes with bindings
    if hub_nodes:
        print(f"\nHub nodes (degree >= 2):")
        for nid, deg in sorted(hub_nodes.items(), key=lambda x: -x[1]):
            bindings = model.binding_map.get_tokens(nid, 0.1)
            word = tok.decode([bindings[0].token_id]) if bindings else "?"
            print(f"  Node {nid} ({word}): degree={deg}")
    
    print(f"\nPath analysis:")
    print(f"  Reachable cross-fact pairs: {reachable_pairs}/{total_pairs}")
    print(f"  Average path length: {avg_path:.2f}")
    if path_lengths:
        print(f"  Path length distribution: {sorted(set(path_lengths))}")
    
    # 6. Edge list
    print(f"\nAll edges:")
    for (src, tgt), e in sorted(graph.edges.items()):
        sb = model.binding_map.get_tokens(src, 0.1)
        tb = model.binding_map.get_tokens(tgt, 0.1)
        sw = tok.decode([sb[0].token_id]) if sb else "?"
        tw = tok.decode([tb[0].token_id]) if tb else "?"
        print(f"  {sw} → {tw}  w={e.weight:.3f} conf={e.confidence:.3f} type={e.relation_type}")
    
    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "components": len(components),
        "avg_degree": avg_degree,
        "avg_path": avg_path,
        "reachable_pct": reachable_pairs / max(1, total_pairs) * 100,
    }


def run():
    facts = [
        ("cat has tail", "tail"),
        ("dog has tail", "tail"),
        ("bird has wing", "wing"),
        ("hawk has wing", "wing"),
    ]
    
    tok = WordTokenizer()
    for text, _ in facts:
        tok.encode(text)
    
    model = RLMv2(vocab_size=100, embed_dim=32, concept_dim=32, n_concepts=50)
    model._tokenizer = tok
    
    # Train
    for epoch in range(200):
        for text, target in facts:
            ids = tok.encode(text)
            model.learn(np.array(ids[:-1], dtype=np.int64), np.array([ids[-1]], dtype=np.int64))
    
    audit_topology(model, tok, facts)
    
    # Test transfer
    print(f"\n=== TRANSFER TESTS ===")
    tests = [
        ("cat has", "tail"),
        ("dog has", "tail"),
        ("bird has", "wing"),
        ("hawk has", "wing"),
        # Cross-domain: hawk is bird-like, should get wing
        ("hawk has", "wing"),
    ]
    for query, expected in tests:
        ids = tok.encode(query)
        logits = model.forward(np.array(ids, dtype=np.int64))
        ranked = np.argsort(logits.data.flatten())[::-1]
        target_id = tok.encode(expected)[0]
        top3 = [tok.decode([int(r)]) for r in ranked[:3]]
        hit = target_id in set(ranked[:10])
        print(f"  '{query}' → expected '{expected}': hit={hit}, top3={top3}")


if __name__ == "__main__":
    run()
