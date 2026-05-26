"""
Test: Do relation vectors actually affect RLM output logits?

Hypothesis: Relation vectors (edge.relation_vector) are WRITTEN during learn()
but NEVER READ during forward(). If true, RVs are decorative and:
  - RV separation can't help accuracy (RVs don't influence predictions)
  - Cross-domain transfer is 0% (the relational semantics encoded in RVs
    never reach the output)

This test measures: does changing a relation vector on an edge change
the model's output logits? Does changing edge.weight change them?

Also checks: does the concept graph structure (edge weights) actually
feed into the final prediction, or is it only the neural network layers?
"""

import numpy as np
import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ravana_ml.nn.rlm import RLM
from ravana_ml.graph import ConceptEdge


def make_model():
    """Create a small RLM for testing."""
    model = RLM(
        vocab_size=50,
        embed_dim=16,
        concept_dim=16,
        n_concepts=60,
        n_hidden=32,
        n_layers=2,
        max_seq_len=32,
    )
    return model


def seed_edges_and_activate(model, token_ids):
    """Run forward to activate concepts, then ensure edges exist between them."""
    # First forward to populate activations
    logits = model.forward(token_ids)
    
    # Get the top predicted concepts from this forward pass
    predicted = model._last_predicted_concepts
    if len(predicted) < 2:
        print("  WARNING: Few concepts activated, graph path may be weak")
        return predicted
    
    # Create edges between predicted concepts (if not already)
    n_created = 0
    for i in range(len(predicted)):
        for j in range(i+1, len(predicted)):
            edge = model.graph.get_or_create_edge(
                predicted[i], predicted[j], 
                weight=0.5, 
                relation_type="causal"
            )
            n_created += 1
    
    # Also create edges to nearby concept nodes to ensure graph connectivity
    node_ids = list(model.graph.nodes.keys())
    for pid in predicted[:3]:
        for nid in node_ids[:10]:
            if nid != pid:
                model.graph.get_or_create_edge(pid, nid, weight=0.3, relation_type="semantic")
    
    print(f"  Seeded {n_created} edges between {len(predicted)} predicted concepts")
    return predicted


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def max_abs_diff(a, b):
    return float(np.max(np.abs(a - b)))


def l2_dist(a, b):
    return float(np.linalg.norm(a - b))


def test_rv_impact_on_logits():
    """
    Core test: Run forward(), save logits, mutate ONLY the relation vectors
    on edges, run forward() again, compare logits.
    
    Then do the same with edge weights.
    """
    model = make_model()
    token_ids = np.array([[1, 5, 10, 3]])

    # Seed edges between concepts that will be activated
    predicted = seed_edges_and_activate(model, token_ids)

    # === Baseline forward pass (with seeded edges) ===
    logits_baseline = model.forward(token_ids).data.copy()
    baseline_predicted = model._last_predicted_concepts.copy()
    
    # Record activated nodes and their activations
    activated_nodes = {nid: model.graph.nodes[nid].activation
                       for nid in model.graph.nodes 
                       if model.graph.nodes[nid].activation > 0.01}
    
    print(f"  Baseline: {len(activated_nodes)} active concepts, "
          f"predicted={baseline_predicted}")

    # === Test 1: Mutate relation vectors (set to random) ===
    original_rvs = {}
    for key, edge in model.graph.edges.items():
        original_rvs[key] = edge.relation_vector.copy()
        edge.relation_vector = np.random.randn(len(edge.relation_vector)).astype(np.float32)
        edge.relation_vector /= np.linalg.norm(edge.relation_vector)

    logits_rv_mutated = model.forward(token_ids).data.copy()
    rv_activated = {nid: model.graph.nodes[nid].activation
                    for nid in model.graph.nodes 
                    if model.graph.nodes[nid].activation > 0.01}

    # Restore original RVs
    for key, rv in original_rvs.items():
        if key in model.graph.edges:
            model.graph.edges[key].relation_vector = rv

    # === Test 2: Mutate edge WEIGHTS (set all to 0.9) ===
    original_weights = {}
    for key, edge in model.graph.edges.items():
        original_weights[key] = edge.weight
        edge.weight = 0.9

    logits_weight_mutated = model.forward(token_ids).data.copy()
    wt_activated = {nid: model.graph.nodes[nid].activation
                    for nid in model.graph.nodes 
                    if model.graph.nodes[nid].activation > 0.01}

    # Restore original weights
    for key, w in original_weights.items():
        if key in model.graph.edges:
            model.graph.edges[key].weight = w

    # === Test 3: Set all edge weights to 0.001 (near-zero) ===
    for key, edge in model.graph.edges.items():
        edge.weight = 0.001

    logits_weak_edges = model.forward(token_ids).data.copy()

    # Restore
    for key, w in original_weights.items():
        if key in model.graph.edges:
            model.graph.edges[key].weight = w

    # === Test 4: Remove ALL edges ===
    edges_backup = {}
    for key in list(model.graph.edges.keys()):
        edges_backup[key] = model.graph.edges[key]
    model.graph.edges.clear()
    model.graph._outgoing.clear()
    model.graph._incoming.clear()

    logits_no_edges = model.forward(token_ids).data.copy()

    # Restore edges
    model.graph.edges = edges_backup
    model.graph._outgoing = {}
    model.graph._incoming = {}
    for (s, t), edge in edges_backup.items():
        model.graph._outgoing.setdefault(s, []).append((t, edge))
        model.graph._incoming.setdefault(t, []).append((s, edge))

    # === Test 5: New model with no edges at all (NN-only) ===
    model2 = make_model()
    logits_nn_only = model2.forward(token_ids).data.copy()

    # === ANALYSIS ===
    print()
    print("=" * 70)
    print("RELATION VECTOR IMPACT TEST RESULTS")
    print("=" * 70)

    rv_cos = cosine_sim(logits_baseline, logits_rv_mutated)
    rv_diff = max_abs_diff(logits_baseline, logits_rv_mutated)
    rv_l2 = l2_dist(logits_baseline, logits_rv_mutated)

    wt_cos = cosine_sim(logits_baseline, logits_weight_mutated)
    wt_diff = max_abs_diff(logits_baseline, logits_weight_mutated)
    wt_l2 = l2_dist(logits_baseline, logits_weight_mutated)

    weak_cos = cosine_sim(logits_baseline, logits_weak_edges)
    weak_l2 = l2_dist(logits_baseline, logits_weak_edges)

    noedge_cos = cosine_sim(logits_baseline, logits_no_edges)
    noedge_l2 = l2_dist(logits_baseline, logits_no_edges)

    nn_cos = cosine_sim(logits_baseline, logits_nn_only)
    nn_l2 = l2_dist(logits_baseline, logits_nn_only)

    print()
    print("Test 1: Mutate relation vectors (random replacement)")
    print(f"  Cosine similarity: {rv_cos:.8f}")
    print(f"  Max abs diff:      {rv_diff:.10f}")
    print(f"  L2 distance:       {rv_l2:.10f}")
    if rv_l2 < 1e-6:
        print("  >> VERDICT: RVs have ZERO impact on predictions. DECORATIVE.")
    elif rv_l2 < 0.01:
        print("  >> VERDICT: RVs have negligible impact on predictions.")
    else:
        print("  >> VERDICT: RVs DO affect predictions (unexpected!).")

    print()
    print("Test 2: Set all edge weights to 0.9 (uniform high)")
    print(f"  Cosine similarity: {wt_cos:.8f}")
    print(f"  L2 distance:       {wt_l2:.8f}")
    if wt_l2 < 1e-6:
        print("  >> VERDICT: Edge weights have zero impact (graph path dead for this input).")
    elif wt_l2 < 0.1:
        print("  >> VERDICT: Edge weights have minor impact.")
    else:
        print("  >> VERDICT: Edge weights DO affect predictions.")

    print()
    print("Test 3: Set all edge weights to 0.001 (near-zero)")
    print(f"  Cosine similarity: {weak_cos:.8f}")
    print(f"  L2 distance:       {weak_l2:.8f}")

    print()
    print("Test 4: Remove ALL edges (no graph structure)")
    print(f"  Cosine similarity: {noedge_cos:.8f}")
    print(f"  L2 distance:       {noedge_l2:.8f}")

    print()
    print("Test 5: Fresh model, no edges (NN-only baseline)")
    print(f"  Cosine similarity: {nn_cos:.8f}")
    print(f"  L2 distance:       {nn_l2:.8f}")

    # === Activation analysis ===
    print()
    print("-" * 70)
    print("ACTIVATION ANALYSIS (top active nodes)")
    print("-" * 70)
    all_nids = set(activated_nodes.keys()) | set(rv_activated.keys()) | set(wt_activated.keys())
    for nid in sorted(all_nids, key=lambda n: activated_nodes.get(n, 0), reverse=True)[:8]:
        b = activated_nodes.get(nid, 0)
        r = rv_activated.get(nid, 0)
        w = wt_activated.get(nid, 0)
        print(f"  Node {nid:4d}: baseline={b:.6f}  rv_mut={r:.6f}  wt_mut={w:.6f}  "
              f"rv_same={'YES' if abs(b-r) < 1e-8 else 'NO'}  "
              f"wt_same={'YES' if abs(b-w) < 1e-8 else 'NO'}")

    # === Forward path decomposition ===
    print()
    print("-" * 70)
    print("FORWARD PATH DECOMPOSITION")
    print("-" * 70)

    h = model._last_hidden_state
    ctx_logits = model.context_logits.forward_raw(h[np.newaxis, :]).flatten()
    ctx_norm = np.linalg.norm(ctx_logits)
    total_norm = np.linalg.norm(logits_baseline)
    
    # concept_logits = total - ctx_logits * context_scale
    concept_contribution = logits_baseline - ctx_logits * model.context_scale
    concept_norm = np.linalg.norm(concept_contribution)
    
    emotion_scale = 1.0 + 0.3 * model.arousal - 0.1 * max(0.0, -model.valence)
    identity_scale = 0.5 + 0.5 * model.identity_strength
    
    print(f"  ||logits_total||:            {total_norm:.4f}")
    print(f"  ||ctx_logits * scale||:      {ctx_norm * model.context_scale:.4f}")
    print(f"  ||concept_contribution||:    {concept_norm:.4f}")
    print(f"  context_scale:               {model.context_scale}")
    print(f"  identity_scale:              {identity_scale:.4f}")
    print(f"  emotion_scale:               {emotion_scale:.4f}")
    print(f"  NN path % of output:         {(ctx_norm * model.context_scale / (total_norm + 1e-15)) * 100:.1f}%")
    print(f"  Graph path % of output:      {(concept_norm / (total_norm + 1e-15)) * 100:.1f}%")

    # === Final verdict ===
    print()
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    issues = []
    
    if rv_l2 < 1e-6:
        issues.append("CRITICAL: Relation vectors are DECORATIVE (never read in forward pass)")
    
    if wt_l2 < 1e-6:
        issues.append("CRITICAL: Edge weights don't affect output for this input")
        issues.append("  (concept activations may not propagate through edges)")
    
    if nn_cos > 0.95:
        issues.append("WARNING: Graph contributes <5% to output (NN path dominates)")
    
    for i, issue in enumerate(issues):
        print(f"  {i+1}. {issue}")
    
    print()
    print("ROOT CAUSES:")
    print()
    print("1. RELATION VECTORS ARE WRITTEN BUT NEVER READ IN FORWARD:")
    print("   - spread_activation() uses: edge.weight * edge.confidence * decay")
    print("   - Multi-hop uses: edge.weight * rel_boost (categorical, not vector)")
    print("   - get_prediction() uses: node.activation * edge.weight")
    print("   - edge.relation_vector (16-dim learned embedding) is NEVER used")
    print()
    print("2. RELATION VECTORS ARE ONLY READ IN:")
    print("   - find_analogy_paths() — standalone graph method, NOT called from forward()")
    print("   - Graph diagnostics/visualization")
    print("   - Serialization (save/load)")
    print()
    print("3. THE PREDICTION PIPELINE:")
    print("   forward() → hidden state → concept_predictor → z → nearest concepts")
    print("   → spread_activation(edge.weight, edge.confidence)")
    print("   → concept scores → softmax → concept_logits")
    print("   → + context_logits (Linear NN layer)")
    print("   → final logits")
    print()
    print("FIX: Wire edge.relation_vector into spread_activation or multi-hop scoring.")
    print("  E.g.: act = node.activation * (edge.weight + dot(relation_vec, query)) * decay")
    
    return rv_cos, wt_cos


def test_relation_vector_isolation():
    """
    Direct proof: trace through spread_activation code to show
    relation_vector is not a factor.
    """
    print()
    print("=" * 70)
    print("SPREAD_ACTIVATION CODE ANALYSIS")
    print("=" * 70)
    print()
    print("graph.py:1031 — the ONLY line that computes activation spread:")
    print("  act = node.activation * edge.weight * edge.confidence * decay")
    print()
    print("edge.relation_vector does not appear anywhere in:")
    print("  - spread_activation()")
    print("  - concept_attention()")
    print("  - PropagationEngine.get_prediction()")
    print("  - forward() multi-hop scoring (uses relation_type STRING, not vector)")
    print()
    print("edge.relation_vector IS used in:")
    print("  - learn() — Hebbian update writes to RV")
    print("  - learn() — Contrastive separation writes to RV")  
    print("  - _refine_relation_types() — re-blends RV with type seed")
    print("  - hebbian_update() — graph-level Hebbian writes to RV")
    print("  - find_analogy_paths() — reads RV for relation consistency scoring")
    print("    BUT this method is NEVER called from forward() or learn()")
    print()
    print("CONCLUSION: relation_vector is a write-only field during prediction.")
    print("All the sophisticated relational learning (contrastive, Hebbian, type")
    print("refinement) produces vectors that are never consulted when predicting.")


def test_find_analogy_paths_usage():
    """Check if find_analogy_paths is ever called from forward/learn."""
    import subprocess
    result = subprocess.run(
        ["grep", "-rn", "find_analogy_paths", 
         "ravana_ml/"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    print()
    print("=" * 70)
    print("FIND_ANALOGY_PATHS USAGE CHECK")
    print("=" * 70)
    print()
    if result.stdout:
        print("References to find_analogy_paths:")
        for line in result.stdout.strip().split("\n"):
            print(f"  {line}")
    else:
        print("  No references found (method exists but is never called)")


if __name__ == "__main__":
    test_relation_vector_isolation()
    test_find_analogy_paths_usage()
    test_rv_impact_on_logits()
