"""
Within-Domain Ablation Experiment for RAVANA RLM

Measures individual contribution of three architectural fixes that took
within-domain association learning from 0% to 100% Top-1 accuracy:
  Fix 1: Relation-vector type seed anchoring
  Fix 2: Concept-creation gating
  Fix 3: Adaptive homeostatic downscaling

Runs 6 conditions:
  A. All fixes OFF (baseline — expect ~0%)
  B. Fix 1 only (anchored relation vectors)
  C. Fix 2 only (concept gating)
  D. Fix 3 only (adaptive downscale)
  E. Fixes 1+2 (no adaptive downscale)
  F. All three (current default — expect 100%)

Also runs sensitivity sweeps on key thresholds.

Usage:
    python experiments/experiment_ablation.py
    python experiments/experiment_ablation.py --quick   # fewer epochs
"""

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import time
import json
from collections import defaultdict
from ravana_ml.nn.rlm import RLM


# ── Vocabulary (same as test_rlm_full.py) ──

VOCAB = [
    '<start>', 'fire', 'hot', 'cold', 'bird', 'fly', 'swim',
    'apple', 'fruit', 'tech', 'sun', 'bright', 'dark',
    'water', 'liquid', 'pie', 'company', 'sky', 'ocean', 'night',
]
VOCAB_SIZE = len(VOCAB)
W = {w: i for i, w in enumerate(VOCAB)}


# ── Association pairs (the within-domain benchmark) ──

TRAIN_PAIRS = [
    ('<start>', 'fire',  'hot'),
    ('<start>', 'bird',  'fly'),
    ('<start>', 'sun',   'bright'),
    ('<start>', 'water', 'liquid'),
    ('<start>', 'apple', 'fruit'),
    ('<start>', 'cold',  'dark'),
    ('<start>', 'tech',  'company'),
    ('<start>', 'sky',   'bright'),
    ('<start>', 'ocean', 'liquid'),
    ('<start>', 'night', 'dark'),
]

TEST_PAIRS = TRAIN_PAIRS  # test on same associations


def make_model(anchor_rv=True, gate_concepts=True, adaptive_ds=True,
               sleep_interval=50):
    """Create an RLM with specified ablation flags."""
    return RLM(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        concept_dim=32,
        n_concepts=VOCAB_SIZE * 2,
        n_hidden=32,
        n_layers=3,
        sleep_interval=sleep_interval,
        anchor_relation_vectors=anchor_rv,
        gate_concept_creation=gate_concepts,
        adaptive_downscale=adaptive_ds,
    )


def evaluate_topk(model, pairs, k=1):
    """Evaluate top-k accuracy on association pairs."""
    correct = 0
    total = 0
    for ctx, trigger, target in pairs:
        input_ids = np.array([[W[ctx], W[trigger]]], dtype=np.int64)
        logits = model.forward(input_ids)
        data = logits.data if hasattr(logits, 'data') else np.array(logits)
        if data.ndim > 1:
            data = data[0]
        ranked = np.argsort(data)[::-1]
        if W[target] in set(ranked[:k]):
            correct += 1
        total += 1
    return correct / max(1, total)


def train(model, pairs, epochs=50, force_sleep_every=5):
    """Train model on association pairs with periodic sleep cycles."""
    errors = []
    for epoch in range(epochs):
        np.random.shuffle(pairs)
        for ctx, trigger, target in pairs:
            err = model.learn(
                np.array([[W[ctx], W[trigger]]], dtype=np.int64),
                np.array([[W[target]]], dtype=np.int64)
            )
            errors.append(err)
        # Force a sleep cycle every N epochs to stress retention
        if force_sleep_every > 0 and (epoch + 1) % force_sleep_every == 0:
            model.sleep_cycle()
    return errors


def run_condition(name, anchor_rv, gate_concepts, adaptive_ds,
                  epochs=50, n_seeds=3):
    """Run one ablation condition across multiple seeds."""
    top1_scores = []
    top10_scores = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        model = make_model(anchor_rv, gate_concepts, adaptive_ds, sleep_interval=20)
        train(model, TRAIN_PAIRS, epochs=epochs, force_sleep_every=5)
        top1 = evaluate_topk(model, TEST_PAIRS, k=1)
        top10 = evaluate_topk(model, TEST_PAIRS, k=10)
        top1_scores.append(top1)
        top10_scores.append(top10)

    return {
        "name": name,
        "top1_mean": float(np.mean(top1_scores)),
        "top1_std": float(np.std(top1_scores)),
        "top10_mean": float(np.mean(top10_scores)),
        "top10_std": float(np.std(top10_scores)),
    }


def run_sensitivity_sweep(param_name, values, epochs=50):
    """Run sensitivity sweep on a single parameter."""
    results = []
    for val in values:
        np.random.seed(42)
        model = make_model()  # all fixes ON

        if param_name == "concept_similarity_threshold":
            model._concept_similarity_threshold = val
        elif param_name == "downscale_floor":
            # Patch the homeostatic_downscale method to use a custom floor
            original_method = model.graph.homeostatic_downscale

            def patched_downscale(*args, floor=val, **kwargs):
                # Temporarily replace the adaptive factor floor
                for edge in model.graph.edges.values():
                    if edge.stability >= 0.8:
                        continue
                    usage = min(1.0, edge.confidence * edge.prediction_count / 10.0)
                    edge.weight *= (floor + (1.0 - floor) * usage)
                # Renormalize orphans
                node_max_edges = {}
                for (src, tgt), edge in model.graph.edges.items():
                    if src not in node_max_edges:
                        node_max_edges[src] = []
                    node_max_edges[src].append((tgt, edge.weight))
                for src, targets in node_max_edges.items():
                    if not targets:
                        continue
                    mean_w = np.mean([w for _, w in targets])
                    if mean_w < 0.1:
                        targets.sort(key=lambda x: x[1], reverse=True)
                        for tgt, _ in targets[:3]:
                            edge = model.graph.get_edge(src, tgt)
                            if edge and edge.weight < 0.2:
                                edge.weight = 0.2

            model.graph.homeostatic_downscale = patched_downscale

        train(model, TRAIN_PAIRS, epochs=epochs)
        top1 = evaluate_topk(model, TEST_PAIRS, k=1)
        results.append({"value": val, "top1": float(top1)})

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fewer epochs for fast testing")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    args = parser.parse_args()

    epochs = 30 if args.quick else 100
    n_seeds = args.seeds

    print("=" * 70)
    print("  RAVANA Within-Domain Ablation Experiment")
    print("=" * 70)
    print(f"  Epochs per condition: {epochs}")
    print(f"  Seeds per condition: {n_seeds}")
    print()

    # ── 6-condition ablation ──
    conditions = [
        # (name, anchor_rv, gate_concepts, adaptive_ds)
        ("Baseline (all OFF)",   False, False, False),
        ("Fix 1: RV anchoring",  True,  False, False),
        ("Fix 2: Concept gating", False, True,  False),
        ("Fix 3: Adaptive ds",   False, False, True),
        ("Fix 1+2 (no adapt)",   True,  True,  False),
        ("Full (all 3)",         True,  True,  True),
    ]

    results = []
    print("  Condition                   | Top-1 (mean±std) | Top-10 (mean±std)")
    print("  " + "-" * 65)

    for name, arv, gc, ads in conditions:
        t0 = time.time()
        res = run_condition(name, arv, gc, ads, epochs=epochs, n_seeds=n_seeds)
        dt = time.time() - t0
        results.append(res)
        print(f"  {name:<27} | {res['top1_mean']:.1%} ±{res['top1_std']:.1%}    "
              f"| {res['top10_mean']:.1%} ±{res['top10_std']:.1%}  ({dt:.0f}s)")

    # ── Sensitivity sweeps ──
    print("\n  Sensitivity sweeps (all fixes ON):")

    print("\n  concept_similarity_threshold:")
    thresh_results = run_sensitivity_sweep(
        "concept_similarity_threshold",
        [0.5, 0.6, 0.7, 0.8, 0.9],
        epochs=epochs
    )
    for r in thresh_results:
        print(f"    threshold={r['value']:.1f} -> top1={r['top1']:.1%}")

    print("\n  downscale_floor:")
    floor_results = run_sensitivity_sweep(
        "downscale_floor",
        [0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        epochs=epochs
    )
    for r in floor_results:
        print(f"    floor={r['value']:.2f} -> top1={r['top1']:.1%}")

    # ── Save results ──
    output = {
        "ablation": results,
        "sensitivity_threshold": thresh_results,
        "sensitivity_floor": floor_results,
        "config": {"epochs": epochs, "seeds": n_seeds},
    }
    out_path = os.path.join(_PROJECT_ROOT, "revisions", "ablation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
