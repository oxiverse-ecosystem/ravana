"""Closed-loop resilience experiment.

Tests whether the cognitive regulation system can recover from induced instability.

Protocol:
1. Train RLM on a simple task until stable
2. Record baseline geometry metrics
3. Induce semantic diffusion by adding many random edges
4. Record degraded metrics
5. Run regulated sleep cycles
6. Measure recovery trajectory

Success criteria:
- System detects phase transition (diffuse/crisis)
- Regulation responds (inhibition boost)
- Metrics recover toward baseline within N sleep cycles
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from ravana_ml.nn.rlm import RLM
from ravana_ml.graph import ConceptGraph


def run_resilience_experiment(seed=42, verbose=True):
    np.random.seed(seed)

    if verbose:
        print("=" * 60)
        print("RESILIENCE EXPERIMENT: Cognitive Self-Regulation")
        print("=" * 60)

    # ── Phase 1: Train to stability ──
    if verbose:
        print("\n── Phase 1: Training to stability ──")

    vocab_size = 64
    model = RLM(vocab_size=vocab_size, embed_dim=32, concept_dim=32,
                n_concepts=128, n_hidden=64, sleep_interval=15)

    # Generate training data: simple repeating patterns
    sequences = []
    for _ in range(100):
        start = np.random.randint(0, vocab_size - 10)
        seq = list(range(start, start + 10))
        sequences.append(seq)

    # Train
    for seq in sequences:
        for i in range(len(seq) - 1):
            x = np.array([seq[i]])
            y = np.array([seq[i + 1]])
            model.learn(x, y)

    # Baseline metrics
    baseline = model.graph.graph_diagnostics()
    baseline_phase = model.graph.classify_phase(baseline)

    if verbose:
        print(f"  Nodes: {len(model.graph.nodes)}")
        print(f"  Edges: {len(model.graph.edges)}")
        print(f"  Entropy: {baseline['graph_entropy']:.3f}")
        print(f"  Specificity: {baseline.get('inference_specificity_mean', 0):.3f}")
        print(f"  Separation: {baseline['relation_separation']:.3f}")
        print(f"  Phase: {baseline_phase['phase']} (confidence={baseline_phase['confidence']:.2f})")

    # ── Phase 2: Induce instability ──
    if verbose:
        print("\n── Phase 2: Inducing semantic diffusion ──")

    # Add many random edges to create overconnectivity (semantic fog)
    node_ids = list(model.graph.nodes.keys())
    n_random_edges = len(model.graph.edges) * 3  # triple the edge count

    for _ in range(n_random_edges):
        src = np.random.choice(node_ids)
        tgt = np.random.choice(node_ids)
        if src != tgt:
            model.graph.add_edge(src, tgt, weight=0.3 + np.random.random() * 0.4,
                                 relation_type=np.random.choice(["semantic", "causal", "temporal"]))

    # Activate nodes to create diffuse activation
    for nid in np.random.choice(node_ids, min(20, len(node_ids)), replace=False):
        model.graph.activate(nid, 0.3 + np.random.random() * 0.7)

    degraded = model.graph.graph_diagnostics()
    degraded_phase = model.graph.classify_phase(degraded)

    if verbose:
        print(f"  Edges after injection: {len(model.graph.edges)}  ({n_random_edges} added)")
        print(f"  Entropy: {degraded['graph_entropy']:.3f}  (was {baseline['graph_entropy']:.3f})")
        print(f"  Specificity: {degraded.get('inference_specificity_mean', 0):.3f}")
        print(f"  Separation: {degraded['relation_separation']:.3f}")
        print(f"  Phase: {degraded_phase['phase']} (confidence={degraded_phase['confidence']:.2f})")
        print(f"  Contradiction density: {degraded['contradiction_density']:.3f}")

    # ── Phase 3: Regulated recovery ──
    if verbose:
        print("\n── Phase 3: Regulated recovery (sleep cycles) ──")

    n_recovery_cycles = 20
    recovery_trajectory = []

    for cycle in range(n_recovery_cycles):
        # Run a sleep cycle (which includes regulation)
        model.sleep_cycle()

        # Record metrics
        metrics = model.graph.graph_diagnostics()
        phase_info = model.graph.classify_phase(metrics)
        regulation = model.graph._regulator.status()

        recovery_trajectory.append({
            "cycle": cycle + 1,
            "entropy": metrics["graph_entropy"],
            "specificity": metrics.get("inference_specificity_mean", 0),
            "separation": metrics["relation_separation"],
            "contradiction_density": metrics["contradiction_density"],
            "neighbor_preservation": metrics["neighbor_preservation"],
            "phase": phase_info["phase"],
            "inhibition_boost": regulation["inhibition_boost"],
            "oscillation_count": regulation["oscillation_count"],
        })

        if verbose and (cycle + 1) % 5 == 0:
            print(f"  Cycle {cycle + 1:2d}: entropy={metrics['graph_entropy']:.3f} "
                  f"specificity={metrics.get('inference_specificity_mean', 0):.3f} "
                  f"phase={phase_info['phase']:12s} "
                  f"inhibition={regulation['inhibition_boost']:.3f}")

    # ── Phase 4: Analysis ──
    if verbose:
        print("\n── Phase 4: Recovery Analysis ──")

    final = model.graph.graph_diagnostics()
    final_phase = model.graph.classify_phase(final)

    # Compute recovery metrics
    entropy_recovery = 1.0 - abs(final["graph_entropy"] - baseline["graph_entropy"]) / max(baseline["graph_entropy"], 0.01)
    specificity_recovery = 1.0 - abs(final.get("inference_specificity_mean", 0) - baseline.get("inference_specificity_mean", 0))

    # Did regulation detect the instability?
    phases_observed = [r["phase"] for r in recovery_trajectory]
    detected_diffuse = "diffuse" in phases_observed or "crisis" in phases_observed

    # Did regulation respond?
    inhibition_responses = [r["inhibition_boost"] for r in recovery_trajectory]
    regulation_responded = max(inhibition_responses) > 0.01

    # Did metrics recover?
    entropy_trend = np.polyfit(range(len(recovery_trajectory)),
                               [r["entropy"] for r in recovery_trajectory], 1)[0]
    specificity_trend = np.polyfit(range(len(recovery_trajectory)),
                                   [r["specificity"] for r in recovery_trajectory], 1)[0]

    # Count oscillations in phase
    phase_changes = sum(1 for i in range(1, len(phases_observed))
                        if phases_observed[i] != phases_observed[i - 1])

    # Recovery elasticity
    entropy_history = model.graph._geometry_history
    elasticity = entropy_history.compute_recovery_elasticity(
        "graph_entropy",
        baseline_value=baseline["graph_entropy"],
        perturbation_value=degraded["graph_entropy"],
        recovery_start=0
    )

    if verbose:
        print(f"  Baseline:  entropy={baseline['graph_entropy']:.3f}  specificity={baseline.get('inference_specificity_mean', 0):.3f}")
        print(f"  Degraded:  entropy={degraded['graph_entropy']:.3f}  specificity={degraded.get('inference_specificity_mean', 0):.3f}")
        print(f"  Final:     entropy={final['graph_entropy']:.3f}  specificity={final.get('inference_specificity_mean', 0):.3f}")
        print(f"  Entropy recovery:     {entropy_recovery:.2f}")
        print(f"  Specificity recovery: {specificity_recovery:.2f}")
        print(f"  Phase detected instability: {detected_diffuse}")
        print(f"  Regulation responded:       {regulation_responded}")
        print(f"  Entropy trend:   {entropy_trend:+.4f}  ({'recovering' if entropy_trend < 0 else 'worsening'})")
        print(f"  Specificity trend: {specificity_trend:+.4f}  ({'recovering' if specificity_trend > 0 else 'worsening'})")
        print(f"  Phase oscillations: {phase_changes}  (regulator oscillation count={recovery_trajectory[-1]['oscillation_count']})")
        print(f"  Recovery elasticity: {elasticity['elasticity']:.4f}")
        print(f"    Speed: {elasticity['speed']:.4f}  (tau={elasticity['tau']})")
        print(f"    Completeness: {elasticity['completeness']:.4f}")
        print(f"    Overshoot: {elasticity['overshoot']:.4f}")

    # ── Verdict ──
    success_criteria = {
        "detected_instability": detected_diffuse,
        "regulation_responded": regulation_responded,
        "entropy_recovering": entropy_trend < 0 or entropy_recovery > 0.5,
        "no_runaway_oscillation": phase_changes < n_recovery_cycles * 0.5,
        "elasticity_positive": elasticity["elasticity"] > 0,
    }

    all_passed = all(success_criteria.values())

    if verbose:
        print(f"\n  {'=' * 40}")
        print(f"  RESULTS: {'ALL CRITERIA MET' if all_passed else 'PARTIAL'}")
        for criterion, passed in success_criteria.items():
            print(f"    {'PASS' if passed else 'FAIL'}: {criterion}")
        print(f"  {'=' * 40}")

    return {
        "success": all_passed,
        "criteria": success_criteria,
        "baseline": baseline,
        "degraded": degraded,
        "final": final,
        "trajectory": recovery_trajectory,
        "entropy_recovery": entropy_recovery,
        "specificity_recovery": specificity_recovery,
    }


if __name__ == "__main__":
    import sys
    verbose = "--quiet" not in sys.argv
    result = run_resilience_experiment(verbose=verbose)
    sys.exit(0 if result["success"] else 1)
