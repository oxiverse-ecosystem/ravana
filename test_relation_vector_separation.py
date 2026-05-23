"""
Validate relation vector separation with extended training (2000 steps).

Tests whether relation vectors in the RLM concept graph actually separate
into distinct clusters by relation type (semantic, causal, temporal) after
extended Hebbian learning with diverse, relation-typed patterns.

KEY FINDING: Relation vectors do NOT separate. They converge toward
indistinguishability because:
1. The EMA update (0.85*old + 0.15*target_vector) makes all relation vectors
   drift toward their target concept vectors, erasing type-specific structure
2. The contrastive push separates by TARGET (different endpoints), not by
   RELATION TYPE — so it doesn't help type-based clustering
3. The initial seed vectors (which ARE type-separated) get overwhelmed

Metrics tracked:
1. Intra-type cosine similarity (should be HIGH for separation)
2. Inter-type cosine similarity (should be LOW for separation)  
3. Separation = intra - inter (positive = good)
4. Per-step cosine similarity showing convergence trajectory
5. Initial seed separation (before training) for comparison
"""
import numpy as np
import sys
import os
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ravana_ml.nn import RLM
from ravana_ml.graph import ConceptEdge


def make_diverse_patterns(vocab_size: int, n_patterns: int = 200):
    """Generate diverse training patterns covering semantic, causal, and temporal relations."""
    patterns = []

    # Use distinct token ranges for each relation type so the classifier
    # can pick up keyword cues from the tokenizer
    # Causal tokens: 0-19
    # Temporal tokens: 20-39
    # Semantic tokens: 40-59
    # (requires vocab_size >= 60)

    for i in range(n_patterns):
        if i % 3 == 0:
            # Causal: even→odd in 0-19
            src = (i * 2) % 20
            tgt = (i * 2 + 1) % 20
        elif i % 3 == 1:
            # Temporal: even→odd in 20-39
            src = 20 + (i * 2) % 20
            tgt = 21 + (i * 2) % 20
            tgt = tgt % vocab_size
        else:
            # Semantic: even→odd in 40-59
            src = 40 + (i * 2) % 20
            tgt = 41 + (i * 2) % 20
            tgt = tgt % vocab_size

        src = src % vocab_size
        tgt = tgt % vocab_size
        if src == tgt:
            tgt = (tgt + 1) % vocab_size

        context = np.array([src, src, src])
        patterns.append((context, np.array([tgt])))

    return patterns


def measure_seed_separation(relation_dim=16):
    """Measure separation of INITIAL relation vector seeds (before any training)."""
    type_seeds = {}
    for rtype in ["semantic", "causal", "temporal", "analogical", "contextual", "inferred"]:
        vec = ConceptEdge._init_relation_vector(rtype, relation_dim)
        type_seeds[rtype] = vec / (np.linalg.norm(vec) + 1e-15)

    types = list(type_seeds.keys())
    intra_sims = []
    inter_sims = []

    for t in types:
        # Intra-type: each type has only 1 seed vector, so intra = 1.0 by definition
        # But we can measure how similar the seed is to others of same type
        # (there's only 1 per type, so this is degenerate)
        intra_sims.append(1.0)  # self-similarity

    for i, t1 in enumerate(types):
        for j, t2 in enumerate(types):
            if j <= i:
                continue
            sim = float(np.dot(type_seeds[t1], type_seeds[t2]))
            inter_sims.append(sim)

    return {
        "intra": np.mean(intra_sims),
        "inter": np.mean(inter_sims) if inter_sims else 0.0,
        "separation": np.mean(intra_sims) - np.mean(inter_sims) if inter_sims else 0.0,
        "pairwise": {(types[i], types[j]): float(np.dot(type_seeds[types[i]], type_seeds[types[j]]))
                     for i in range(len(types)) for j in range(i+1, len(types))}
    }


def compute_relation_vector_stats(model: RLM):
    """Compute per-type and cross-type cosine similarity statistics."""
    type_vectors = defaultdict(list)

    for key, edge in model.graph.edges.items():
        if edge.shortcut or edge.edge_type == "inhibitory":
            continue
        rv = edge.relation_vector
        rv_norm = np.linalg.norm(rv)
        if rv_norm > 0:
            type_vectors[edge.relation_type].append(rv / rv_norm)

    stats = {}
    types = sorted(type_vectors.keys())

    # Intra-type: average pairwise cosine sim within each type
    for t in types:
        vecs = type_vectors[t]
        if len(vecs) < 2:
            stats[f"intra_{t}"] = {"mean": 1.0, "n": len(vecs)}
            continue
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(float(np.dot(vecs[i], vecs[j])))
        stats[f"intra_{t}"] = {"mean": np.mean(sims), "std": np.std(sims), "n": len(vecs), "sims": sims}

    # Inter-type: average cosine sim between types
    for i, t1 in enumerate(types):
        for j, t2 in enumerate(types):
            if j <= i:
                continue
            sims = []
            for v1 in type_vectors[t1]:
                for v2 in type_vectors[t2]:
                    sims.append(float(np.dot(v1, v2)))
            if sims:
                stats[f"inter_{t1}_vs_{t2}"] = {"mean": np.mean(sims), "std": np.std(sims), "n": len(sims)}

    # Overall scores
    intra_means = [v["mean"] for k, v in stats.items() if k.startswith("intra_")]
    inter_means = [v["mean"] for k, v in stats.items() if k.startswith("inter_")]
    if intra_means and inter_means:
        avg_intra = np.mean(intra_means)
        avg_inter = np.mean(inter_means)
        separation = avg_intra - avg_inter
    else:
        separation = 0.0
        avg_intra = 0.0
        avg_inter = 0.0

    stats["_summary"] = {
        "avg_intra_similarity": avg_intra,
        "avg_inter_similarity": avg_inter,
        "separation_score": separation,
        "n_types": len(types),
        "types": types,
        "n_edges_by_type": {t: len(type_vectors[t]) for t in types},
    }
    return stats


def analyze_relation_vector_drift(model: RLM, n_samples=10):
    """Check if relation vectors are drifting toward target concept vectors (the EMA trap)."""
    drift_scores = []
    for key, edge in model.graph.edges.items():
        if edge.shortcut or edge.edge_type == "inhibitory":
            continue
        src_id, tgt_id = key
        tgt_node = model.graph.get_node(tgt_id)
        if tgt_node is None:
            continue
        tgt_vec = tgt_node.vector
        tgt_norm = np.linalg.norm(tgt_vec)
        rv_norm = np.linalg.norm(edge.relation_vector)
        if tgt_norm > 0 and rv_norm > 0:
            # How similar is the relation vector to the target concept vector?
            sim = float(np.dot(edge.relation_vector / rv_norm, tgt_vec / tgt_norm))
            drift_scores.append((edge.relation_type, sim, edge.weight))
        if len(drift_scores) >= 100:
            break

    return drift_scores


def main():
    print("=" * 70)
    print("RELATION VECTOR SEPARATION VALIDATION — 2000 LEARN STEPS")
    print("=" * 70)

    # ── Phase 0: Measure initial seed separation ──
    print("\n--- Phase 0: Initial seed vector separation ---")
    seed_stats = measure_seed_separation(relation_dim=16)
    print(f"Seed separation (intra - inter): {seed_stats['separation']:.4f}")
    print(f"  Intra-type (self): {seed_stats['intra']:.4f}")
    print(f"  Inter-type (avg):  {seed_stats['inter']:.4f}")
    print("\n  Pairwise seed similarities:")
    for (t1, t2), sim in seed_stats["pairwise"].items():
        print(f"    {t1:>12} vs {t2:<12}: {sim:+.4f}")

    # ── Phase 1: Build model and train ──
    vocab_size = 64
    model = RLM(
        vocab_size=vocab_size,
        embed_dim=32,
        concept_dim=16,
        n_concepts=64,
        n_hidden=48,
        n_layers=2,
        sleep_interval=200,
    )

    patterns = make_diverse_patterns(vocab_size, n_patterns=300)
    print(f"\nGenerated {len(patterns)} training patterns (causal/temporal/semantic)")

    # ── Phase 2: Extended training ──
    n_steps = 2000
    snapshot_interval = 200
    snapshots = []

    print(f"\nTraining for {n_steps} steps...")
    print(f"{'Step':>5}  {'Edges':>6}  {'Intra':>7}  {'Inter':>7}  {'Separ':>7}  {'Types'}")
    print("-" * 70)

    t0 = time.time()
    for step in range(n_steps):
        pattern = patterns[step % len(patterns)]
        context, next_tok = pattern
        model.learn(context, next_tok)

        if step % snapshot_interval == 0 or step == n_steps - 1:
            stats = compute_relation_vector_stats(model)
            s = stats["_summary"]
            snapshots.append((step, stats))
            types_str = ",".join(f"{t}:{s['n_edges_by_type'].get(t,0)}" for t in s["types"])
            print(f"{step:>5}  {len(model.graph.edges):>6}  "
                  f"{s['avg_intra_similarity']:>7.4f}  "
                  f"{s['avg_inter_similarity']:>7.4f}  "
                  f"{s['separation_score']:>7.4f}  "
                  f"{types_str}")

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s ({n_steps / elapsed:.0f} steps/sec)")

    # ── Phase 3: Detailed analysis ──
    final_stats = snapshots[-1][1]
    summary = final_stats["_summary"]

    print("\n" + "=" * 70)
    print("FINAL RELATION VECTOR SEPARATION ANALYSIS")
    print("=" * 70)
    print(f"\nRelation types found: {summary['types']}")
    print(f"Edges by type: {summary['n_edges_by_type']}")
    print(f"\nAvg intra-type cosine similarity:  {summary['avg_intra_similarity']:.6f}")
    print(f"Avg inter-type cosine similarity:  {summary['avg_inter_similarity']:.6f}")
    print(f"Separation score (intra - inter):  {summary['separation_score']:.6f}")

    # Per-type breakdown
    print("\n--- Per-type intra similarity ---")
    for k, v in final_stats.items():
        if k.startswith("intra_"):
            t = k.replace("intra_", "")
            print(f"  {t:>15}: {v['mean']:.6f} (std={v.get('std', 0):.6f}, n={v['n']})")

    print("\n--- Cross-type similarities ---")
    for k, v in final_stats.items():
        if k.startswith("inter_"):
            print(f"  {k:>30}: {v['mean']:.6f} (std={v.get('std', 0):.6f}, n={v['n']})")

    # ── Phase 4: Relation vector drift analysis ──
    print("\n--- Relation vector drift toward target concepts ---")
    drift = analyze_relation_vector_drift(model)
    if drift:
        by_type = defaultdict(list)
        for rtype, sim, weight in drift:
            by_type[rtype].append(sim)
        print("  (How similar is each relation vector to its target concept vector?)")
        print("  (High similarity = relation vector became a copy of target, losing type info)")
        for rtype, sims in sorted(by_type.items()):
            print(f"  {rtype:>15}: mean_sim_to_target = {np.mean(sims):.4f} "
                  f"(std={np.std(sims):.4f}, n={len(sims)})")

    # ── Convergence trajectory ──
    print("\n--- Convergence Trajectory ---")
    print(f"{'Step':>5}  {'Separation':>10}  {'Intra':>7}  {'Inter':>7}")
    for step, stats in snapshots:
        s = stats["_summary"]
        print(f"{step:>5}  {s['separation_score']:>10.6f}  "
              f"{s['avg_intra_similarity']:>7.4f}  "
              f"{s['avg_inter_similarity']:>7.4f}")

    # ── Verdict ──
    print("\n" + "=" * 70)

    first_sep = snapshots[0][1]["_summary"]["separation_score"] if len(snapshots) > 0 else 0
    early_sep = snapshots[1][1]["_summary"]["separation_score"] if len(snapshots) > 1 else 0
    last_sep = snapshots[-1][1]["_summary"]["separation_score"]

    print(f"Separation trajectory: seed_ideal={seed_stats['separation']:.4f}, "
          f"step_0={first_sep:.4f}, step_200={early_sep:.4f}, step_1999={last_sep:.4f}")

    # Compute additional diagnostic: is semantic intra inflated by cluster size?
    semantic_intra = final_stats.get("intra_semantic", {}).get("mean", 0)
    causal_intra = final_stats.get("intra_causal", {}).get("mean", 0)
    n_semantic = summary["n_edges_by_type"].get("semantic", 0)
    n_causal = summary["n_edges_by_type"].get("causal", 0)
    n_temporal = summary["n_edges_by_type"].get("temporal", 0)
    temporal_missing = n_temporal == 0

    # Check causal-vs-semantic inter similarity (the dominant cross-type)
    causal_sem_inter = final_stats.get("inter_causal_vs_semantic", {}).get("mean", 0)

    # The real story: initial seeds ARE separated, but training collapses them
    peak_sep = max(s[1]["_summary"]["separation_score"] for s in snapshots[1:])  # skip step 0
    is_declining = last_sep < peak_sep * 0.5  # dropped more than 50% from peak
    seed_degraded = seed_stats['separation'] > 0.1 and last_sep < seed_stats['separation'] * 0.3

    # Cluster imbalance check
    cluster_ratio = max(n_semantic, 1) / max(n_causal, 1)
    severe_imbalance = cluster_ratio > 30  # >30:1 ratio

    if seed_stats['separation'] > 0.1 and last_sep < 0.02:
        verdict = "FAIL - COLLAPSE"
        detail = ("Initial seed vectors were well-separated ({:.4f}), but 2000 steps of "
                  "training COLLAPSED the separation to {:.4f}. "
                  "Root cause: the EMA relation vector update (0.85*old + 0.15*target_vector) "
                  "drifts all relation vectors toward their target concept vectors."
                  ).format(seed_stats['separation'], last_sep)
    elif is_declining and last_sep < 0.05:
        verdict = "FAIL - DECLINING"
        detail = ("Separation peaked at {:.4f} but fell to {:.4f}. "
                  "Extended training HURTS separation."
                  ).format(peak_sep, last_sep)
    elif last_sep > 0.1 and not seed_degraded:
        # Separation exists, but check if it's meaningful
        warnings = []
        if severe_imbalance:
            warnings.append(f"severe cluster imbalance ({n_semantic} semantic vs {n_causal} causal)")
        if temporal_missing:
            warnings.append("temporal type vanished (reclassified)")
        if causal_sem_inter > 0.6:
            warnings.append(f"causal-semantic inter-sim={causal_sem_inter:.3f} (high)")
        if semantic_intra > 0.85:
            warnings.append(f"semantic intra={semantic_intra:.3f} (EMA convergence artifact)")

        if len(warnings) >= 3:
            verdict = "PARTIAL"
            detail = ("Separation score {:.4f} is positive but likely driven by "
                      "cluster size asymmetry, not robust type encoding. "
                      "Warnings: {}"
                      ).format(last_sep, "; ".join(warnings))
        elif len(warnings) >= 1:
            verdict = "PASS (with caveats)"
            detail = ("Separation score {:.4f}. "
                      "Caveats: {}"
                      ).format(last_sep, "; ".join(warnings))
        else:
            verdict = "PASS"
            detail = f"Separation stable at {last_sep:.4f} after 2000 steps."
    elif last_sep > 0.05:
        verdict = "MARGINAL"
        detail = f"Weak separation ({last_sep:.4f}). May need stronger contrastive signal."
    else:
        verdict = "FAIL"
        detail = f"Separation at {last_sep:.4f}. Vectors not separating by type."

    print(f"\nVERDICT: {verdict}")
    print(f"  {detail}")
    print("=" * 70)

    return verdict, summary


if __name__ == "__main__":
    verdict, summary = main()
    if "PASS" in verdict:
        sys.exit(0)
    elif "MARGINAL" in verdict:
        sys.exit(1)
    else:
        sys.exit(2)
