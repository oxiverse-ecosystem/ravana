"""
Full-Vocabulary Within-Domain Ablation for RAVANA RLM

Uses the SimpleTokenizer (full character vocabulary) with realistic
multi-word facts instead of the simplified 20-token hardcoded vocab.

Measures individual contribution of three architectural fixes:
  Fix 1: Relation-vector type seed anchoring
  Fix 2: Concept-creation gating
  Fix 3: Adaptive homeostatic downscaling

Runs 6 ablation conditions + sensitivity sweeps.

Usage:
    python experiments/experiment_ablation_full.py
    python experiments/experiment_ablation_full.py --quick   # fewer epochs
    python experiments/experiment_ablation_full.py --seeds 5
"""

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import time
import json
from ravana_ml.nn.rlm import RLM
from ravana_ml.tokenizer import SimpleTokenizer
from experiments.experiment_cross_domain import (
    build_domain_a_science,
    build_domain_b_social,
    encode_fact,
)


def evaluate_topk(model, facts, tokenizer, k=1):
    """Evaluate top-k accuracy on a set of facts."""
    correct = 0
    total = 0
    for input_text, target_text, rel_type in facts:
        input_ids, target_ids = encode_fact(tokenizer, input_text, target_text)
        if len(input_ids) == 0:
            continue
        logits = model.forward(input_ids[np.newaxis, :])
        data = logits.data if hasattr(logits, 'data') else np.array(logits)
        if data.ndim > 1:
            data = data[0]

        target_id = ord(target_text[0]) if target_text else 0
        ranked = np.argsort(data)[::-1]
        if target_id in set(ranked[:k]):
            correct += 1
        total += 1
    return correct / max(1, total)


def train(model, facts, tokenizer, epochs=50, force_sleep_every=5):
    """Train model on facts with periodic sleep cycles."""
    errors = []
    for epoch in range(epochs):
        np.random.shuffle(facts)
        for input_text, target_text, rel_type in facts:
            input_ids, target_ids = encode_fact(tokenizer, input_text, target_text)
            err = model.learn(input_ids, target_ids)
            errors.append(err)
        if force_sleep_every > 0 and (epoch + 1) % force_sleep_every == 0:
            model.sleep_cycle()
    return errors


def make_model(tokenizer, anchor_rv=True, gate_concepts=True, adaptive_ds=True,
                deep_sleep_every=5):
    """Create an RLM with specified ablation flags."""
    return RLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        concept_dim=32,
        n_concepts=tokenizer.vocab_size,
        n_hidden=32,
        n_layers=3,
        sleep_interval=100,
        tokenizer=tokenizer,
        anchor_relation_vectors=anchor_rv,
        gate_concept_creation=gate_concepts,
        adaptive_downscale=adaptive_ds,
        deep_sleep_every=deep_sleep_every,
    )


def run_condition(name, facts, tokenizer, anchor_rv, gate_concepts, adaptive_ds,
                  epochs=100, n_seeds=3, deep_sleep_every=5):
    """Run one ablation condition across multiple seeds."""
    top1_scores = []
    top10_scores = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        model = make_model(tokenizer, anchor_rv, gate_concepts, adaptive_ds,
                           deep_sleep_every=deep_sleep_every)
        train(model, facts, tokenizer, epochs=epochs, force_sleep_every=10)
        top1 = evaluate_topk(model, facts, tokenizer, k=1)
        top10 = evaluate_topk(model, facts, tokenizer, k=10)
        top1_scores.append(top1)
        top10_scores.append(top10)

    return {
        "name": name,
        "top1_mean": float(np.mean(top1_scores)),
        "top1_std": float(np.std(top1_scores)),
        "top10_mean": float(np.mean(top10_scores)),
        "top10_std": float(np.std(top10_scores)),
    }


def run_sensitivity_sweep(param_name, values, facts, tokenizer, epochs=100,
                           deep_sleep_every=5):
    """Run sensitivity sweep on a single parameter."""
    results = []
    for val in values:
        np.random.seed(42)
        model = make_model(tokenizer, deep_sleep_every=deep_sleep_every)  # all fixes ON

        if param_name == "concept_similarity_threshold":
            model._concept_similarity_threshold = val
        elif param_name == "downscale_floor":
            original_method = model.graph.homeostatic_downscale

            def patched_downscale(*args, floor=val, **kwargs):
                for edge in model.graph.edges.values():
                    if edge.stability >= 0.8:
                        continue
                    usage = min(1.0, edge.confidence * edge.prediction_count / 10.0)
                    edge.weight *= (floor + (1.0 - floor) * usage)

            model.graph.homeostatic_downscale = patched_downscale

        train(model, facts, tokenizer, epochs=epochs, force_sleep_every=10)
        top1 = evaluate_topk(model, facts, tokenizer, k=1)
        results.append({"value": val, "top1": float(top1)})

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fewer epochs for fast testing")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--fast-sleep", action="store_true",
                        help="Use light/deep sleep alternation for faster sweeps")
    args = parser.parse_args()

    epochs = 50 if args.quick else 100
    n_seeds = args.seeds
    deep_sleep_every = 10 if args.fast_sleep else 5

    # Build vocabulary from both domains
    tokenizer = SimpleTokenizer()
    domain_a = build_domain_a_science()
    domain_b = build_domain_b_social()
    all_facts = domain_a["train"] + domain_b["train"]

    print("=" * 70)
    print("  RAVANA Full-Vocabulary Within-Domain Ablation")
    print("=" * 70)
    print(f"  Vocabulary: {tokenizer.vocab_size} tokens (SimpleTokenizer)")
    print(f"  Training facts: {len(all_facts)} ({len(domain_a['train'])} science + {len(domain_b['train'])} social)")
    print(f"  Epochs per condition: {epochs}")
    print(f"  Seeds per condition: {n_seeds}")
    print()

    # ── 6-condition ablation ──
    conditions = [
        ("Baseline (all OFF)",   False, False, False),
        ("Fix 1: RV anchoring",  True,  False, False),
        ("Fix 2: Concept gating", False, True,  False),
        ("Fix 3: Adaptive ds",   False, False, True),
        ("Fix 1+2 (no adapt)",   True,  True,  False),
        ("Full (all 3)",         True,  True,  True),
    ]

    results = []
    print(f"  {'Condition':<27} | {'Top-1 (mean +/- std)':>22} | {'Top-10 (mean +/- std)':>23}")
    print("  " + "-" * 75)

    for name, arv, gc, ads in conditions:
        t0 = time.time()
        res = run_condition(name, all_facts, tokenizer, arv, gc, ads,
                           epochs=epochs, n_seeds=n_seeds,
                           deep_sleep_every=deep_sleep_every)
        dt = time.time() - t0
        results.append(res)
        print(f"  {name:<27} | {res['top1_mean']:.1%} +/-{res['top1_std']:.1%}       "
              f"| {res['top10_mean']:.1%} +/-{res['top10_std']:.1%}  ({dt:.0f}s)")

    # ── Sensitivity sweeps ──
    print("\n  Sensitivity sweeps (all fixes ON):")

    print("\n  concept_similarity_threshold:")
    thresh_results = run_sensitivity_sweep(
        "concept_similarity_threshold",
        [0.5, 0.6, 0.7, 0.8, 0.9],
        all_facts, tokenizer, epochs=epochs,
        deep_sleep_every=deep_sleep_every
    )
    for r in thresh_results:
        print(f"    threshold={r['value']:.1f} -> top1={r['top1']:.1%}")

    print("\n  downscale_floor:")
    floor_results = run_sensitivity_sweep(
        "downscale_floor",
        [0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        all_facts, tokenizer, epochs=epochs,
        deep_sleep_every=deep_sleep_every
    )
    for r in floor_results:
        print(f"    floor={r['value']:.2f} -> top1={r['top1']:.1%}")

    # ── Summary: gap between baseline and full ──
    if results:
        baseline_t1 = results[0]["top1_mean"]
        full_t1 = results[-1]["top1_mean"]
        gap = full_t1 - baseline_t1
        print(f"\n  Gap (Baseline -> Full): {gap*100:.1f}pp Top-1")

    # ── Save results ──
    output = {
        "ablation": results,
        "sensitivity_threshold": thresh_results,
        "sensitivity_floor": floor_results,
        "config": {
            "epochs": epochs,
            "seeds": n_seeds,
            "vocab_size": tokenizer.vocab_size,
            "n_facts": len(all_facts),
            "deep_sleep_every": deep_sleep_every,
            "fast_sleep": args.fast_sleep,
        },
    }
    out_path = os.path.join(_PROJECT_ROOT, "revisions", "ablation_full_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
